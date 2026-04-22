"""Router: pick distilled CLAMP bridge vs chroma EBM vs chord trellis from merged experiment config."""

from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dab_ttm.discriminator.distilled_clamp_discriminator import DistilledClampTextDiscriminator


def load_ttm_discriminator(**kwargs):
    d = kwargs.get("discriminator", {})
    type = str(d.get("type", "")).strip().lower()
    assert type == "distilled_clamp", "Only distilled_clamp is supported for now"
    distilled_root = Path(os.path.expandvars(d.get("distilled_root", "${CTRLM_REPO}/ctrlm/distilled_clamp")))
    distilled_ckpt = str(d.get("distilled_ckpt", "")).strip()
    if not distilled_ckpt:
        raise ValueError(
            "discriminator.distilled_ckpt must be set to a .pt file; "
            "auto-resolving latest/best under distilled_root is disabled."
        )
    ckpt_path = Path(distilled_ckpt)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"discriminator.distilled_ckpt is not a file: {ckpt_path}")
    distilled_cfg = d.get("distilled_cfg", str(distilled_root / "configs" / "base.yaml"))
    clamp3_root = os.path.expandvars(d.get("clamp3_root", "${CTRLM_REPO}/clamp3"))
    clamp3_text_model = d.get("clamp3_text_model", "FacebookAI/xlm-roberta-base")
    clamp3_weights_path = d.get("clamp3_weights_path", "")
    return DistilledClampTextDiscriminator(
        distilled_ckpt=str(ckpt_path),
        distilled_cfg=str(distilled_cfg),
        distilled_root=str(distilled_root),
        clamp3_root=str(clamp3_root),
        clamp3_text_model=str(clamp3_text_model),
        clamp3_weights_path=str(clamp3_weights_path),
        attr_weight=float(d.get("attr_weight", 1.0)),
        attr_loss_type=str(d.get("attr_loss_type", "cosine")),
        lm_reg_weight=float(d.get("lm_reg_weight", 0.2)),
        bias_reg_weight=float(d.get("bias_reg_weight", 0.01)),
    )
