from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from anticipation import ops as ant_ops
from anticipation.config import MAX_TIME
from anticipation.convert import events_to_midi, midi_to_wav
from anticipation.vocab import AUTOREGRESS, SEPARATOR, TIME_OFFSET

_clap_model_cache: dict[tuple[str, str], Any] = {}


def _decode_ids_simple(ids_tensor: torch.Tensor) -> list[str]:
    out = []
    for row in ids_tensor.tolist():
        out.append(" ".join(str(int(tok)) for tok in row))
    return out


def _to_event_tokens_for_midi(token_line: str) -> list[int]:
    tokens = [int(tok) for tok in token_line.strip().split() if tok.strip()]
    if not tokens:
        return []
    if tokens[0] == int(AUTOREGRESS):
        tokens = tokens[1:]
    usable = (len(tokens) // 3) * 3
    if usable <= 0:
        return []
    return tokens[:usable]


def _filter_event_tokens_time_within_max(event_tokens: list[int]) -> list[int]:
    if not event_tokens:
        return []
    t_lo = int(TIME_OFFSET)
    t_hi = t_lo + int(MAX_TIME)
    out: list[int] = []
    for i in range(0, (len(event_tokens) // 3) * 3, 3):
        t, d, n = event_tokens[i], event_tokens[i + 1], event_tokens[i + 2]
        if n == SEPARATOR:
            out.extend([t, d, n])
            continue
        if t < t_lo or t >= t_hi:
            continue
        out.extend([t, d, n])
    return out


def _save_rendered_outputs(
    run_dir: Path,
    stem: str,
    prompt: str,
    guided_token_lines: list[str],
    *,
    save_midi: bool,
    save_wav: bool,
    sound_font: str = "",
    wav_sample_rate: int = 44100,
    log_label: str = "",
    log_tag: str = "[ttm]",
    melody_controls: list[int] | None = None,
) -> list[dict[str, str | None]]:
    out_rows: list[dict[str, str | None]] = []
    if not save_midi and not save_wav:
        for _ in guided_token_lines:
            out_rows.append({"midi": None, "wav": None})
        return out_rows
    midi_dir = run_dir / "midi"
    wav_dir = run_dir / "wav"
    if save_midi or save_wav:
        midi_dir.mkdir(parents=True, exist_ok=True)
    if save_wav:
        wav_dir.mkdir(parents=True, exist_ok=True)

    melody_extra = list(melody_controls) if melody_controls else []
    for b_idx, token_line in enumerate(guided_token_lines):
        raw_tokens = _to_event_tokens_for_midi(token_line)
        usable = (len(raw_tokens) // 3) * 3
        if usable <= 0:
            print(f"[warn] skip render {log_label} batch_idx={b_idx}: no usable event tokens")
            out_rows.append({"midi": None, "wav": None})
            continue
        acc_events, gen_ctrl = ant_ops.split(raw_tokens[:usable])
        event_tokens = ant_ops.combine(acc_events, melody_extra + gen_ctrl)
        event_tokens = _filter_event_tokens_time_within_max(event_tokens)
        midi_rel: str | None = None
        wav_rel: str | None = None
        if not event_tokens:
            print(f"[warn] skip render {log_label} batch_idx={b_idx}: no tokens after time filter")
            out_rows.append({"midi": None, "wav": None})
            continue
        midi_path = midi_dir / f"{stem}_b{b_idx:03d}.mid"
        try:
            mid = events_to_midi(event_tokens)
            mid.save(str(midi_path))
            midi_rel = str(Path("midi") / midi_path.name)
        except Exception as e:
            print(f"[warn] midi save failed {log_label} batch_idx={b_idx}: {e}")
            out_rows.append({"midi": None, "wav": None})
            continue

        if save_wav:
            wav_path = wav_dir / f"{stem}_b{b_idx:03d}.wav"
            try:
                midi_to_wav(
                    str(midi_path),
                    wavfile=str(wav_path),
                    sound_font=(sound_font or None),
                    sample_rate=int(wav_sample_rate),
                )
                wav_rel = str(Path("wav") / wav_path.name)
            except Exception as e:
                print(
                    f"[warn] wav render failed {log_label} batch_idx={b_idx}: {e} "
                    "(midi output is still saved)"
                )
                out_rows.append({"midi": midi_rel, "wav": None})
                return out_rows
        out_rows.append({"midi": midi_rel, "wav": wav_rel})
    print(f"{log_tag} rendered {log_label} prompt={prompt!r} midi_dir={midi_dir} wav_dir={wav_dir}")
    return out_rows


def get_cached_clap_model(device: str, clap_ckpt: Path | None = None) -> Any:
    if clap_ckpt is None:
        raise ValueError("clap_ckpt must be configured explicitly; default checkpoint fallback is disabled.")
    ckpt = Path(clap_ckpt)
    if not ckpt.is_file():
        raise FileNotFoundError(f"CLAP checkpoint not found: {ckpt}")
    key = (device, str(ckpt.resolve()))
    if key not in _clap_model_cache:
        import laion_clap

        m = laion_clap.CLAP_Module(enable_fusion=False, amodel="HTSAT-base", device=device)
        m.load_ckpt(str(ckpt), verbose=False)
        m.eval()
        _clap_model_cache[key] = m.float()
    return _clap_model_cache[key]


def resolve_soundfont_for_wav(requested: str, *, log_tag: str = "[ttm]") -> str | None:
    if requested:
        p = Path(requested).expanduser()
        if p.exists():
            return str(p)
        print(f"[warn] sound_font path does not exist: {p}. WAV export disabled.")
        return None

    candidates = [
        Path.home() / ".fluidsynth" / "default_sound_font.sf2",
        Path("/usr/share/sounds/sf2/FluidR3_GM.sf2"),
        Path("/usr/share/soundfonts/FluidR3_GM.sf2"),
    ]
    for c in candidates:
        if c.exists():
            print(f"{log_tag} using detected soundfont: {c}")
            return str(c)
    print(
        "[warn] no SoundFont found for WAV rendering. "
        "Set `sound_font` in config to a valid .sf2 file. MIDI export will continue."
    )
    return None
