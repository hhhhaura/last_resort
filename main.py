from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any


from constants import CONF, DAB_ROOT, ROOT, RUN_DIR, SEED, assert_frozen_constraints

if str(DAB_ROOT) not in sys.path:
    sys.path.append(str(DAB_ROOT))

from prompt_runner import IncrementalClapMetricsLogger, read_prompt_items, run_single_prompt

from io_layout import ensure_run_dir, prompt_dir_for, write_conf
from direct_grad_core import load_run_stack, set_seed


def _set_all_seeds(seed: int) -> None:
    random.seed(seed)
    set_seed(seed)


def _validate_dependencies(conf: dict[str, Any]) -> None:
    disc = conf["discriminator"]
    required = [
        Path(str(disc["distilled_ckpt"])),
        Path(str(disc["distilled_cfg"])),
        Path(str(disc["distilled_root"])),
        Path(str(disc["clamp3_root"])) / "code" / "config.py",
        Path(str(disc["clamp3_root"])) / "code" / "utils.py",
        Path(str(disc["clamp3_weights_path"])),
        Path(str(conf["prompt_csv"])),
    ]
    metrics = conf.get("metrics") or {}
    clap_ckpt = metrics.get("clap_ckpt")
    if clap_ckpt:
        required.append(Path(str(clap_ckpt)))
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing required dependency paths:\n" + "\n".join(missing))


def main() -> None:
    assert_frozen_constraints()
    _set_all_seeds(int(SEED))
    _validate_dependencies(CONF)

    run_dir = ensure_run_dir(Path(RUN_DIR))
    write_conf(run_dir, CONF)
    print(f"[last_resort] root={ROOT}")
    print(f"[last_resort] run_dir={run_dir.resolve()}")
    print(
        "[last_resort] frozen config: "
        f"use_scale_weights={CONF['use_scale_weights']} "
        f"bias_update_mode={CONF['bias_update_mode']} "
        f"per_sample_batch_size={CONF['per_sample_batch_size']} "
        f"prompt_batch_size={CONF['prompt_batch_size']}"
    )

    device = str(CONF["device"])
    model, discriminator, sampler = load_run_stack(CONF, dab_root=DAB_ROOT)

    prompt_items = read_prompt_items(
        Path(str(CONF["prompt_csv"])),
        start_idx=int(CONF["start_idx"]),
        end_idx=int(CONF["end_idx"]),
    )
    print(
        f"[last_resort] loaded_prompts={len(prompt_items)} "
        f"start_idx={int(CONF['start_idx'])} end_idx={int(CONF['end_idx'])}"
    )

    metrics = CONF.get("metrics") or {}
    clap_logger: IncrementalClapMetricsLogger | None = None
    if bool(metrics.get("clap", False)):
        clap_logger = IncrementalClapMetricsLogger(
            run_dir,
            device=device,
            clap_ckpt=str(metrics["clap_ckpt"]),
            resume=True,
        )
        clap_logger.warmup()

    results: list[dict[str, Any]] = []
    for p_idx, item in enumerate(prompt_items):
        prompt_id = int(item["id"])
        prompt_text = str(item["prompt"])
        prompt_dir = prompt_dir_for(run_dir, p_idx, prompt_id)
        summary_path = prompt_dir / "summary.json"
        if summary_path.is_file() and clap_logger is not None and p_idx in clap_logger.completed_prompt_indices:
            print(f"[last_resort] prompt_idx={p_idx} prompt_id={prompt_id} skip (already complete)")
            continue

        result = run_single_prompt(
            model=model,
            discriminator=discriminator,
            sampler=sampler,
            conf=CONF,
            device=device,
            prompt_idx=p_idx,
            prompt_id=prompt_id,
            prompt_text=prompt_text,
            prompt_dir=prompt_dir,
        )
        if clap_logger is not None:
            clap_logger.step(
                prompt_idx=result.prompt_idx,
                row_id=result.prompt_id,
                prompt=result.prompt_text,
                gen_wav_abs=result.final_wav_abs,
            )
        results.append(
            {
                "prompt_idx": result.prompt_idx,
                "prompt_id": result.prompt_id,
                "prompt_dir": str(result.prompt_dir),
                "best_step_by_attr_loss": result.best_step_by_attr_loss,
                "best_attr_loss": result.best_attr_loss,
                "selected_step": result.selected_step,
                "selected_policy": result.selected_policy,
            }
        )

    metrics_summary = clap_logger.finalize() if clap_logger is not None else {}
    run_summary = {
        "run_dir": str(run_dir),
        "num_prompts": len(prompt_items),
        "processed_prompts": len(results),
        "seed": int(SEED),
        "metrics": metrics_summary,
        "results": results,
    }
    (run_dir / "run_summary.json").write_text(json.dumps(run_summary, indent=2), encoding="utf-8")
    print(f"[last_resort] wrote run summary: {run_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
