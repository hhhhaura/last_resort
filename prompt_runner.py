from __future__ import annotations

import csv
import json
import math
from contextlib import ExitStack
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from utils import (
    _decode_ids_simple,
    _save_rendered_outputs,
    get_cached_clap_model,
    ids_hash,
    resolve_soundfont_for_wav,
)

from constants import MAX_RENDER_STEPS, RENDER_EVERY_STEP, SAVE_MIDI, SAVE_WAV, TRACE_NUM_STEPS
from direct_grad_core import (
    DlpRuntime,
    initialize_dlp_batch,
    one_step_direct_grad,
    one_step_sampled_l2,
)


@dataclass
class PromptRunResult:
    prompt_idx: int
    prompt_id: int
    prompt_text: str
    prompt_dir: Path
    best_step_by_attr_loss: int
    best_attr_loss: float
    final_wav_abs: Path | None
    selected_step: int
    selected_policy: str


class IncrementalClapMetricsLogger:
    def __init__(self, run_dir: Path, *, device: str, clap_ckpt: str, resume: bool = True) -> None:
        self.run_dir = run_dir
        self.device = device
        self.clap_ckpt = Path(clap_ckpt)
        self.csv_path = run_dir / "incremental_metrics.csv"
        self.summary_path = run_dir / "incremental_metrics_summary.json"
        self._scores: list[float] = []
        self.completed_prompt_indices: set[int] = set()
        self._model = None
        header = ["prompt_idx", "id", "clap", "mean_clap", "n_clap"]
        if resume and self.csv_path.is_file():
            with self.csv_path.open("r", encoding="utf-8", newline="") as f:
                for row in csv.DictReader(f):
                    idx_raw = str(row.get("prompt_idx", "")).strip()
                    if idx_raw:
                        try:
                            self.completed_prompt_indices.add(int(idx_raw))
                        except ValueError:
                            pass
                    clap_raw = str(row.get("clap", "")).strip()
                    if clap_raw:
                        try:
                            self._scores.append(float(clap_raw))
                        except ValueError:
                            pass
        else:
            with self.csv_path.open("w", encoding="utf-8", newline="") as f:
                csv.writer(f).writerow(header)

    def warmup(self) -> None:
        self._ensure_model()

    def _ensure_model(self) -> None:
        if self._model is None:
            self._model = get_cached_clap_model(self.device, self.clap_ckpt)

    def step(self, *, prompt_idx: int, row_id: int, prompt: str, gen_wav_abs: Path | None) -> None:
        if prompt_idx in self.completed_prompt_indices:
            return
        clap_v = float("nan")
        if gen_wav_abs is not None and gen_wav_abs.is_file() and prompt.strip():
            try:
                self._ensure_model()
                assert self._model is not None
                audio_embed = self._model.get_audio_embedding_from_filelist([str(gen_wav_abs)], use_tensor=True)
                text_embed = self._model.get_text_embedding([prompt], use_tensor=True)
                sim = F.cosine_similarity(audio_embed, text_embed, dim=-1)
                clap_v = float(sim.item())
                self._scores.append(clap_v)
            except Exception as exc:
                print(f"[clap] failed idx={prompt_idx} id={row_id}: {exc}")

        mean_c = float(np.mean(self._scores)) if self._scores else float("nan")
        n_c = len(self._scores)
        with self.csv_path.open("a", encoding="utf-8", newline="") as f:
            csv.writer(f).writerow(
                [
                    prompt_idx,
                    row_id,
                    f"{clap_v:.6f}" if math.isfinite(clap_v) else "",
                    f"{mean_c:.6f}" if math.isfinite(mean_c) else "",
                    n_c,
                ]
            )
        self.completed_prompt_indices.add(prompt_idx)
        print(f"[clap] idx={prompt_idx} id={row_id} clap={clap_v:.4f} mean_clap={mean_c:.4f} (n={n_c})")

    def finalize(self) -> dict[str, Any]:
        mean_c = float(np.mean(self._scores)) if self._scores else float("nan")
        summary = {
            "mean_clap": mean_c,
            "n_clap": len(self._scores),
            "incremental_metrics_csv": str(self.csv_path),
        }
        with self.summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        return summary


def read_prompt_items(prompt_csv: Path, *, start_idx: int, end_idx: int) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with prompt_csv.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rt = row.get("req_time")
            if rt is None or (isinstance(rt, str) and not str(rt).strip()):
                rt = row.get("required_time")
            items.append(
                {
                    "id": int(row["id"]) if str(row.get("id", "")).strip() else len(items),
                    "prompt": str(row["prompt"]),
                    "required_time": rt,
                }
            )
    s = max(0, int(start_idx))
    return items[s:] if int(end_idx) < 0 else items[s : int(end_idx) + 1]


def _pick_selected_step(
    step_rows: list[dict[str, Any]],
    *,
    lm_threshold: float,
) -> tuple[int, str]:
    candidates_lm: list[tuple[float, float, int]] = []
    candidates_total: list[tuple[float, int]] = []
    for row in step_rows:
        if row.get("phase") == "final_ext":
            continue
        step = int(row["step"])
        attr_loss = row.get("attr_loss")
        loss = row.get("loss")
        debug = row.get("debug") or {}
        lm_mean = debug.get("lm_mean")

        if isinstance(loss, (int, float)) and math.isfinite(float(loss)):
            candidates_total.append((float(loss), step))
        if isinstance(attr_loss, (int, float)) and isinstance(lm_mean, (int, float)):
            if math.isfinite(float(attr_loss)) and math.isfinite(float(lm_mean)) and float(lm_mean) < lm_threshold:
                total_for_tie = float(loss) if isinstance(loss, (int, float)) and math.isfinite(float(loss)) else float("inf")
                candidates_lm.append((float(attr_loss), total_for_tie, step))

    if candidates_lm:
        candidates_lm.sort(key=lambda x: (x[0], x[1], x[2]))
        return int(candidates_lm[0][2]), "min_attr_given_lm"
    if candidates_total:
        candidates_total.sort(key=lambda x: (x[0], x[1]))
        return int(candidates_total[0][1]), "min_total_loss"
    return -1, "none"


def run_prompts(
    *,
    model: Any,
    discriminator: Any,
    runtime: DlpRuntime,
    conf: dict[str, Any],
    device: str,
    prompt_indices: list[int],
    prompt_ids: list[int],
    prompt_texts: list[str],
    prompt_dirs: list[Path],
    batch_size: int,
) -> list[PromptRunResult]:
    actual_batch = int(len(prompt_indices))
    assert actual_batch == int(batch_size)
    assert actual_batch == int(len(prompt_ids)) == int(len(prompt_texts)) == int(len(prompt_dirs))
    prompt_len = 0
    input_ids = torch.zeros((actual_batch, prompt_len), dtype=torch.long, device=device)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        "debug_trace_sequences": False,
    }
    discriminator.set_text_prompt(
        prompt_texts,
        device=torch.device(device),
    )

    inputs, cur_batch = initialize_dlp_batch(
        runtime,
        model=model,
        discriminator=discriminator,
        batch_size=actual_batch,
        seq_length=int(conf["seq_len"]),
        prompt_length=prompt_len,
        inputs=inputs,
        num_steps=int(TRACE_NUM_STEPS),
        temperature=float(conf.get("temperature", 1.0)),
        top_p=float(conf.get("top_p", 0.98)),
        do_sample=bool(conf.get("do_sample", True)),
    )

    txt_dirs: list[Path] = []
    steps_jsonls: list[Path] = []
    for prompt_dir in prompt_dirs:
        txt_dir = prompt_dir / "text"
        txt_dir.mkdir(parents=True, exist_ok=True)
        txt_dirs.append(txt_dir)
        steps_jsonls.append(prompt_dir / "steps.jsonl")

    sound_font = str(conf.get("sound_font", ""))
    resolved_sound_font = resolve_soundfont_for_wav(sound_font, log_tag="[last_resort]") if SAVE_WAV else None
    save_wav = bool(SAVE_WAV) and (resolved_sound_font is not None)
    wav_sr = int(conf.get("wav_sample_rate", 44100))

    for i in range(actual_batch):
        print(
            f"[last_resort] prompt_idx={prompt_indices[i]} prompt_id={prompt_ids[i]} prompt={prompt_texts[i]!r}"
        )
    print(
        f"[last_resort] device={device} batch_size={actual_batch} "
        f"num_steps={TRACE_NUM_STEPS} max_ctx={int(conf['seq_len'])}"
    )

    best_step = [-1] * actual_batch
    best_attr_loss = [float("inf")] * actual_batch
    step_rows: list[list[dict[str, Any]]] = [[] for _ in range(actual_batch)]
    step_to_normal_wav: list[dict[int, Path | None]] = [{} for _ in range(actual_batch)]
    bias_update_mode = str(conf.get("bias_update_mode", "direct_grad")).strip().lower()
    if bias_update_mode == "direct_grad":
        step_fn = one_step_direct_grad
    elif bias_update_mode == "sampled_l2":
        step_fn = one_step_sampled_l2
    else:
        raise ValueError(f"Unsupported bias_update_mode={bias_update_mode!r}. Use: direct_grad | sampled_l2")

    with ExitStack() as stack:
        step_writers = [
            stack.enter_context(p.open("w", encoding="utf-8"))
            for p in steps_jsonls
        ]
        for step in range(int(TRACE_NUM_STEPS)):
            cur_batch, loss_values, output_ids, attr_losses, step_debugs = step_fn(
                runtime,
                cur_batch,
                prompt_length=prompt_len,
            )
            decoded_lines = _decode_ids_simple(output_ids)
            loss_list = loss_values.detach().float().cpu().tolist()
            attr_list = attr_losses.detach().float().cpu().tolist()
            for i in range(actual_batch):
                loss_value = float(loss_list[i])
                attr_loss = float(attr_list[i])
                step_debug = step_debugs[i]
                normal_line = decoded_lines[i]
                normal_ids = [int(x) for x in normal_line.split()] if normal_line.strip() else []
                row = {
                    "step": step,
                    "loss": loss_value,
                    "attr_loss": attr_loss,
                    "normal_len": len(normal_ids),
                    "normal_hash": ids_hash(normal_ids) if normal_ids else "",
                    "debug": step_debug,
                }
                step_writers[i].write(json.dumps(row, ensure_ascii=True) + "\n")
                step_rows[i].append(row)

                (txt_dirs[i] / f"step_{step:03d}_normal.txt").write_text(normal_line + "\n", encoding="utf-8")

                render_allowed = bool(RENDER_EVERY_STEP)
                if int(MAX_RENDER_STEPS) > 0 and step >= int(MAX_RENDER_STEPS):
                    render_allowed = False
                if render_allowed:
                    rendered_normal = _save_rendered_outputs(
                        run_dir=prompt_dirs[i],
                        stem=f"step_{step:03d}_normal",
                        prompt=prompt_texts[i],
                        guided_token_lines=[normal_line],
                        save_midi=bool(SAVE_MIDI),
                        save_wav=save_wav,
                        sound_font=(resolved_sound_font or ""),
                        wav_sample_rate=wav_sr,
                        log_label=f"step={step} type=normal",
                        log_tag="[last_resort]",
                    )
                    wav_rel = rendered_normal[0].get("wav") if rendered_normal else None
                    wav_abs = (
                        prompt_dirs[i] / str(wav_rel)
                        if isinstance(wav_rel, str) and wav_rel
                        else None
                    )
                    step_to_normal_wav[i][step] = wav_abs
                else:
                    step_to_normal_wav[i][step] = None

                if attr_loss < best_attr_loss[i]:
                    best_attr_loss[i] = attr_loss
                    best_step[i] = int(step)

                print(
                    f"[last_resort] prompt_idx={prompt_indices[i]} step={step:03d} "
                    f"loss={loss_value:.6f} attr_loss={attr_loss:.6f} "
                    f"normal_hash={row['normal_hash']} "
                    f"grad_norm={step_debug.get('grad_norm', 0.0):.4f} "
                    f"bias_norm={step_debug.get('bias_norm', 0.0):.4f}"
                )

    out: list[PromptRunResult] = []
    for i in range(actual_batch):
        selected_step, selected_policy = _pick_selected_step(
            step_rows[i],
            lm_threshold=float(conf.get("selection_lm_loss_threshold", 1.0)),
        )
        final_wav_abs = step_to_normal_wav[i].get(selected_step)
        selected_row = None
        for r in step_rows[i]:
            if int(r.get("step", -1)) == selected_step and r.get("phase") != "final_ext":
                selected_row = r
                break
        selected_loss = (
            float(selected_row["loss"])
            if selected_row and isinstance(selected_row.get("loss"), (int, float))
            else float("nan")
        )
        selected_attr = (
            float(selected_row["attr_loss"])
            if selected_row and isinstance(selected_row.get("attr_loss"), (int, float))
            else float("nan")
        )
        selected_lm = (
            float((selected_row.get("debug") or {}).get("lm_mean"))
            if selected_row and isinstance((selected_row.get("debug") or {}).get("lm_mean"), (int, float))
            else float("nan")
        )

        summary = {
            "prompt_idx": int(prompt_indices[i]),
            "prompt_id": int(prompt_ids[i]),
            "prompt": prompt_texts[i],
            "num_steps": int(TRACE_NUM_STEPS),
            "best_step_by_attr_loss": int(best_step[i]),
            "best_attr_loss": float(best_attr_loss[i]),
            "prompt_dir": str(prompt_dirs[i]),
            "steps_file": str(steps_jsonls[i]),
            "save_midi": bool(SAVE_MIDI),
            "save_wav": bool(save_wav),
            "final_ext_pass": False,
            "rows_written": len(step_rows[i]),
            "selection_lm_threshold": float(conf.get("selection_lm_loss_threshold", 1.0)),
            "selected_policy": selected_policy,
            "selected_step": int(selected_step),
            "selected_wav": str(final_wav_abs) if final_wav_abs is not None else "",
            "selected_loss": selected_loss if math.isfinite(selected_loss) else "",
            "selected_attr_loss": selected_attr if math.isfinite(selected_attr) else "",
            "selected_lm_mean": selected_lm if math.isfinite(selected_lm) else "",
        }
        (prompt_dirs[i] / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(f"[last_resort] wrote summary: {prompt_dirs[i] / 'summary.json'}")

        out.append(
            PromptRunResult(
                prompt_idx=prompt_indices[i],
                prompt_id=prompt_ids[i],
                prompt_text=prompt_texts[i],
                prompt_dir=prompt_dirs[i],
                best_step_by_attr_loss=best_step[i],
                best_attr_loss=best_attr_loss[i],
                final_wav_abs=final_wav_abs,
                selected_step=selected_step,
                selected_policy=selected_policy,
            )
        )
    return out
