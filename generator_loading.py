"""Base model/tokenizer loaders for anticipation adapter."""

from __future__ import annotations

import sys
from pathlib import Path

from transformers import AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dab_ttm.generator.anticipation_base import load_anticipation_base_model


def load_tokenizer(model_name_or_path: str):
    return AutoTokenizer.from_pretrained(model_name_or_path)


def load_base_model(**kwargs):
    # Accept both key styles for compatibility.
    model_id = kwargs.get("model_name_or_path") or kwargs.get("pretrained_model_name_or_path")
    if model_id is None:
        raise ValueError(
            "Provide base_model_args.model_name_or_path (or pretrained_model_name_or_path) for anticipation model."
        )
    trust_remote_code = bool(kwargs.get("trust_remote_code", True))
    return load_anticipation_base_model(model_name_or_path=model_id, trust_remote_code=trust_remote_code)
