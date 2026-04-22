from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from sampler import LangevinSampler
from utils import _decode_ids_simple, _save_rendered_outputs, get_cached_clap_model, resolve_soundfont_for_wav

from constants import MAX_RENDER_STEPS, RENDER_EVERY_STEP, SAVE_MIDI, SAVE_WAV, TRACE_NUM_STEPS
from direct_grad_core import ids_hash, one_step_with_direct_norm_matched_bias


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


def run_single_prompt(
    *,
    model: Any,
    discriminator: Any,
    sampler: LangevinSampler,
    conf: dict[str, Any],
    device: str,
    prompt_idx: int,
    prompt_id: int,
    prompt_text: str,
    prompt_dir: Path,
) -> PromptRunResult:
    prompt_tokens: list[int] = []
    prompt_len = len(prompt_tokens)
    input_ids = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    inputs = {
        "input_ids": input_ids,
        "attention_mask": torch.ones_like(input_ids, dtype=torch.long),
        "debug_trace_sequences": False,
        "prompt_group_ids": torch.tensor([0], dtype=torch.long, device=device),
    }
    discriminator.set_text_prompt([prompt_text], [1.0], batch_size=1, device=torch.device(device))

    inputs, cur_batch = sampler.initialize_batch(
        model=model,
        discriminator=discriminator,
        batch_size=1,
        seq_length=int(conf["seq_len"]),
        prompt_length=prompt_len,
        inputs=inputs,
        num_steps=int(TRACE_NUM_STEPS),
        temperature=float(conf.get("temperature", 1.0)),
        top_p=float(conf.get("top_p", 0.98)),
        do_sample=bool(conf.get("do_sample", True)),
    )

    txt_dir = prompt_dir / "text"
    txt_dir.mkdir(parents=True, exist_ok=True)
    steps_jsonl = prompt_dir / "steps.jsonl"

    sound_font = str(conf.get("sound_font", ""))
    resolved_sound_font = resolve_soundfont_for_wav(sound_font, log_tag="[last_resort]") if SAVE_WAV else None
    save_wav = bool(SAVE_WAV) and (resolved_sound_font is not None)
    wav_sr = int(conf.get("wav_sample_rate", 44100))

    print(f"[last_resort] prompt_idx={prompt_idx} prompt_id={prompt_id} prompt={prompt_text!r}")
    print(f"[last_resort] device={device} num_steps={TRACE_NUM_STEPS} max_ctx={int(conf['seq_len'])}")

    best_step = -1
    best_attr_loss = float("inf")
    final_wav_abs: Path | None = None
    step_rows: list[dict[str, Any]] = []
    step_to_normal_wav: dict[int, Path | None] = {}

    with steps_jsonl.open("w", encoding="utf-8") as jf:
        for step in range(int(TRACE_NUM_STEPS)):
            cur_batch, loss_value, output_ids, sampled_full, attr_loss, step_debug = one_step_with_direct_norm_matched_bias(
                sampler,
                cur_batch,
                prompt_length=prompt_len,
                ar_event_only=bool(conf["direct_grad_ar_event_only"]),
                norm_match_eps=float(conf["direct_grad_norm_match_eps"]),
                use_masked_full_vocab_norm=bool(conf["direct_grad_norm_match_masked_full"]),
                use_lm_topk_support_norm=bool(conf["direct_grad_norm_match_topk"]),
                topk_k=int(conf["direct_grad_norm_match_topk_k"]) if int(conf["direct_grad_norm_match_topk_k"]) > 0 else int(conf["k_val"]),
                ratio_min=float(conf["direct_grad_norm_match_ratio_min"]),
                ratio_max=float(conf["direct_grad_norm_match_ratio_max"]),
            )
            normal_line = _decode_ids_simple(output_ids)[0]
            sampled_line = _decode_ids_simple(sampled_full)[0]
            normal_ids = [int(x) for x in normal_line.split()] if normal_line.strip() else []
            sampled_ids = [int(x) for x in sampled_line.split()] if sampled_line.strip() else []
            row = {
                "step": step,
                "loss": float(loss_value),
                "attr_loss": float(attr_loss),
                "normal_len": len(normal_ids),
                "sampled_len": len(sampled_ids),
                "normal_hash": ids_hash(normal_ids) if normal_ids else "",
                "sampled_hash": ids_hash(sampled_ids) if sampled_ids else "",
                "debug": step_debug,
            }
            jf.write(json.dumps(row, ensure_ascii=True) + "\n")
            jf.flush()
            step_rows.append(row)

            (txt_dir / f"step_{step:03d}_normal.txt").write_text(normal_line + "\n", encoding="utf-8")
            (txt_dir / f"step_{step:03d}_sampled.txt").write_text(sampled_line + "\n", encoding="utf-8")

            render_allowed = bool(RENDER_EVERY_STEP)
            if int(MAX_RENDER_STEPS) > 0 and step >= int(MAX_RENDER_STEPS):
                render_allowed = False
            if render_allowed:
                rendered_normal = _save_rendered_outputs(
                    run_dir=prompt_dir,
                    stem=f"step_{step:03d}_normal",
                    prompt=prompt_text,
                    guided_token_lines=[normal_line],
                    save_midi=bool(SAVE_MIDI),
                    save_wav=save_wav,
                    sound_font=(resolved_sound_font or ""),
                    wav_sample_rate=wav_sr,
                    log_label=f"step={step} type=normal",
                    log_tag="[last_resort]",
                )
                _save_rendered_outputs(
                    run_dir=prompt_dir,
                    stem=f"step_{step:03d}_sampled",
                    prompt=prompt_text,
                    guided_token_lines=[sampled_line],
                    save_midi=bool(SAVE_MIDI),
                    save_wav=save_wav,
                    sound_font=(resolved_sound_font or ""),
                    wav_sample_rate=wav_sr,
                    log_label=f"step={step} type=sampled",
                    log_tag="[last_resort]",
                )
                wav_rel = rendered_normal[0].get("wav") if rendered_normal else None
                wav_abs = (prompt_dir / str(wav_rel)) if isinstance(wav_rel, str) and wav_rel else None
                step_to_normal_wav[step] = wav_abs
            else:
                step_to_normal_wav[step] = None

            if attr_loss < best_attr_loss:
                best_attr_loss = float(attr_loss)
                best_step = int(step)

            print(
                f"[last_resort] step={step:03d} "
                f"loss={loss_value:.6f} attr_loss={attr_loss:.6f} "
                f"normal_hash={row['normal_hash']} sampled_hash={row['sampled_hash']} "
                f"ratio_mean={step_debug.get('norm_match_ratio_mean', 0.0):.4f} "
                f"tiny_frac={step_debug.get('norm_match_bn_tiny_frac', 0.0):.4f}"
            )

    selected_step, selected_policy = _pick_selected_step(
        step_rows,
        lm_threshold=float(conf.get("selection_lm_loss_threshold", 1.0)),
    )
    final_wav_abs = step_to_normal_wav.get(selected_step)
    selected_row = None
    for r in step_rows:
        if int(r.get("step", -1)) == selected_step and r.get("phase") != "final_ext":
            selected_row = r
            break
    selected_loss = float(selected_row["loss"]) if selected_row and isinstance(selected_row.get("loss"), (int, float)) else float("nan")
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
        "prompt_idx": int(prompt_idx),
        "prompt_id": int(prompt_id),
        "prompt": prompt_text,
        "num_steps": int(TRACE_NUM_STEPS),
        "best_step_by_attr_loss": int(best_step),
        "best_attr_loss": float(best_attr_loss),
        "prompt_dir": str(prompt_dir),
        "steps_file": str(steps_jsonl),
        "save_midi": bool(SAVE_MIDI),
        "save_wav": bool(save_wav),
        "ar_event_only": bool(conf["direct_grad_ar_event_only"]),
        "norm_match_eps": float(conf["direct_grad_norm_match_eps"]),
        "final_ext_pass": False,
        "rows_written": len(step_rows),
        "selection_lm_threshold": float(conf.get("selection_lm_loss_threshold", 1.0)),
        "selected_policy": selected_policy,
        "selected_step": int(selected_step),
        "selected_wav": str(final_wav_abs) if final_wav_abs is not None else "",
        "selected_loss": selected_loss if math.isfinite(selected_loss) else "",
        "selected_attr_loss": selected_attr if math.isfinite(selected_attr) else "",
        "selected_lm_mean": selected_lm if math.isfinite(selected_lm) else "",
    }
    (prompt_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"[last_resort] wrote summary: {prompt_dir / 'summary.json'}")

    return PromptRunResult(
        prompt_idx=prompt_idx,
        prompt_id=prompt_id,
        prompt_text=prompt_text,
        prompt_dir=prompt_dir,
        best_step_by_attr_loss=best_step,
        best_attr_loss=best_attr_loss,
        final_wav_abs=final_wav_abs,
        selected_step=selected_step,
        selected_policy=selected_policy,
    )
