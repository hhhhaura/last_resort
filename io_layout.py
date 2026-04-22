from __future__ import annotations

from pathlib import Path

import yaml


def ensure_run_dir(run_dir: Path) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "prompts").mkdir(parents=True, exist_ok=True)
    return run_dir


def prompt_dir_for(run_dir: Path, prompt_idx: int, prompt_id: int) -> Path:
    pdir = run_dir / "prompts" / f"prompt_{prompt_idx:04d}_{prompt_id}"
    (pdir / "text").mkdir(parents=True, exist_ok=True)
    return pdir


def write_conf(run_dir: Path, conf: dict) -> None:
    out = run_dir / "conf.yaml"
    with out.open("w", encoding="utf-8") as f:
        yaml.safe_dump(conf, f, sort_keys=False)
