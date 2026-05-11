"""Unified discriminator contract, implementation, and loader."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from transformers import AutoTokenizer, BertConfig

from constants import AR_EVENT_VOCAB_SIZE


class SoftOnehotAnticipationDiscriminator(nn.Module):
    expects_soft_onehot: bool = True
    active_vocab_size: int = AR_EVENT_VOCAB_SIZE

    def __init__(self, attr_weight: float = 1.0, lm_reg_weight: float = 0.2, bias_reg_weight: float = 0.01):
        super().__init__()
        self.attr_weight = attr_weight
        self.lm_reg_weight = lm_reg_weight
        self.bias_reg_weight = bias_reg_weight


SoftOnehotSteeringDiscriminator = SoftOnehotAnticipationDiscriminator


def _prepend_sys_path_if_missing(path: str | Path) -> None:
    p = str(Path(path).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


_distilled_clamp_repo = Path(__file__).resolve().parents[1] / "distilled_clamp"
_prepend_sys_path_if_missing(_distilled_clamp_repo)

from distilled_clamp.config_phase import get_effective_cfg  # type: ignore
from distilled_clamp.models.loader import build_distilled_student_from_cfg  # type: ignore


def _load_clamp3_modules(clamp3_root: str | Path):
    clamp3_code = Path(clamp3_root) / "code"
    config_py = clamp3_code / "config.py"
    utils_py = clamp3_code / "utils.py"
    if not config_py.exists():
        raise FileNotFoundError(f"Missing clamp3 config module: {config_py}")
    if not utils_py.exists():
        raise FileNotFoundError(f"Missing clamp3 utils module: {utils_py}")

    spec_cfg = importlib.util.spec_from_file_location("clamp3_config_module", str(config_py))
    if spec_cfg is None or spec_cfg.loader is None:
        raise ImportError(f"Unable to load spec for {config_py}")
    clamp_cfg = importlib.util.module_from_spec(spec_cfg)
    sys.modules[spec_cfg.name] = clamp_cfg
    spec_cfg.loader.exec_module(clamp_cfg)

    prev_config = sys.modules.get("config")
    sys.modules["config"] = clamp_cfg
    try:
        spec_utils = importlib.util.spec_from_file_location("clamp3_utils_module", str(utils_py))
        if spec_utils is None or spec_utils.loader is None:
            raise ImportError(f"Unable to load spec for {utils_py}")
        clamp_utils = importlib.util.module_from_spec(spec_utils)
        sys.modules[spec_utils.name] = clamp_utils
        spec_utils.loader.exec_module(clamp_utils)
    finally:
        if prev_config is None:
            sys.modules.pop("config", None)
        else:
            sys.modules["config"] = prev_config
    return clamp_cfg, clamp_utils


class DistilledClampTextDiscriminator(SoftOnehotSteeringDiscriminator):
    def __init__(
        self,
        distilled_ckpt: str,
        distilled_cfg: str,
        distilled_root: str,
        clamp3_root: str,
        clamp3_text_model: str = "FacebookAI/xlm-roberta-base",
        clamp3_weights_path: str = "",
        attr_loss_type: str = "cosine",
        attr_weight: float = 1.0,
        lm_reg_weight: float = 0.2,
        bias_reg_weight: float = 0.01,
    ):
        super().__init__()
        self.attr_loss_type = attr_loss_type
        self.attr_weight = float(attr_weight)
        self.lm_reg_weight = float(lm_reg_weight)
        self.bias_reg_weight = float(bias_reg_weight)
        self.cached_text_emb_list: list[torch.Tensor] = []
        self.last_prompt_attr_losses: torch.Tensor | None = None

        with open(distilled_cfg, "r", encoding="utf-8") as f:
            distilled_full_cfg = yaml.safe_load(f)
        if "phase1" in distilled_full_cfg and "phase2" in distilled_full_cfg:
            self.distilled_cfg = get_effective_cfg(distilled_full_cfg, "align")
        else:
            # Backward compatibility for already-merged/phase-specific config files.
            self.distilled_cfg = distilled_full_cfg

        ckpt = torch.load(distilled_ckpt, map_location="cpu")
        raw_state = ckpt["model"]
        raw_keys = list(raw_state.keys())
        if not raw_keys:
            raise ValueError("distilled_ckpt contains an empty model state_dict.")
        has_new_keys = any(k.startswith("embedding.") for k in raw_keys)
        if not has_new_keys:
            raise ValueError(
                "Unsupported distilled_ckpt format. Expected updated "
                "DistilledAntiClamp2Model keys ('embedding.*', 'encoder.*', ...)."
            )

        _prepend_sys_path_if_missing(distilled_root)

        self.bridge_vocab_size = int(self.distilled_cfg["source"]["vocab_size"])
        if self.bridge_vocab_size != AR_EVENT_VOCAB_SIZE:
            raise ValueError(
                f"distilled source.vocab_size={self.bridge_vocab_size} must equal AR_EVENT_VOCAB_SIZE "
                f"(CONTROL_OFFSET)={AR_EVENT_VOCAB_SIZE}; steering passes a fixed AR-event slice."
            )
        source_cfg = self.distilled_cfg.get("source", {})
        self.bridge_pad_token_id = int(source_cfg.get("pad_token_id", self.bridge_vocab_size))
        self.bridge_mask_token_id = int(source_cfg.get("mask_token_id", self.bridge_pad_token_id + 1))
        self.bridge_embedding_vocab_size = int(self.bridge_mask_token_id + 1)
        if self.bridge_embedding_vocab_size < (self.bridge_vocab_size + 2):
            raise ValueError(
                "distilled source ids are inconsistent: expected pad=vocab_size and mask=pad+1 "
                f"(got vocab_size={self.bridge_vocab_size}, pad={self.bridge_pad_token_id}, "
                f"mask={self.bridge_mask_token_id})."
            )
        self.bridge = build_distilled_student_from_cfg(
            self.distilled_cfg,
            vocab_size=self.bridge_embedding_vocab_size,
        )
        print("[discriminator] bridge=distilled_clamp (DistilledAntiClamp2Model)")

        self.bridge.load_state_dict(raw_state, strict=True)
        self.bridge.eval()
        for p in self.bridge.parameters():
            p.requires_grad_(False)

        clamp_cfg, clamp_utils = _load_clamp3_modules(clamp3_root)

        audio_config = BertConfig(
            vocab_size=1,
            hidden_size=clamp_cfg.AUDIO_HIDDEN_SIZE,
            num_hidden_layers=clamp_cfg.AUDIO_NUM_LAYERS,
            num_attention_heads=clamp_cfg.AUDIO_HIDDEN_SIZE // 64,
            intermediate_size=clamp_cfg.AUDIO_HIDDEN_SIZE * 4,
            max_position_embeddings=clamp_cfg.MAX_AUDIO_LENGTH,
        )
        symbolic_config = BertConfig(
            vocab_size=1,
            hidden_size=clamp_cfg.M3_HIDDEN_SIZE,
            num_hidden_layers=clamp_cfg.PATCH_NUM_LAYERS,
            num_attention_heads=clamp_cfg.M3_HIDDEN_SIZE // 64,
            intermediate_size=clamp_cfg.M3_HIDDEN_SIZE * 4,
            max_position_embeddings=clamp_cfg.PATCH_LENGTH,
        )
        self.clamp = clamp_utils.CLaMP3Model(
            audio_config=audio_config,
            symbolic_config=symbolic_config,
            text_model_name=clamp3_text_model,
            hidden_size=clamp_cfg.CLAMP3_HIDDEN_SIZE,
            load_m3=clamp_cfg.CLAMP3_LOAD_M3,
        )
        if clamp3_weights_path:
            p = Path(clamp3_weights_path)
            if p.exists():
                checkpoint = torch.load(str(p), map_location="cpu", weights_only=True)
                self.clamp.load_state_dict(checkpoint["model"], strict=False)
        self.clamp.eval()
        for p in self.clamp.parameters():
            p.requires_grad_(False)
        self.text_tokenizer = AutoTokenizer.from_pretrained(clamp3_text_model)

        print(f"[discriminator] AR_EVENT_VOCAB_SIZE={AR_EVENT_VOCAB_SIZE} (CONTROL_OFFSET)")
        print(f"[discriminator] distilled_ckpt={distilled_ckpt}")
        print(f"[discriminator] distilled_cfg={distilled_cfg}")
        print(f"[discriminator] distilled_root={distilled_root}")
        print(
            "[discriminator] bridge vocab ids: "
            f"ar={self.bridge_vocab_size} pad={self.bridge_pad_token_id} "
            f"mask={self.bridge_mask_token_id} embed_vocab={self.bridge_embedding_vocab_size}"
        )
        print(f"[discriminator] clamp3_root={clamp3_root}")
        print(f"[discriminator] clamp3_text_model={clamp3_text_model}")
        if clamp3_weights_path:
            print(f"[discriminator] clamp3_weights_path={clamp3_weights_path}")

    @torch.no_grad()
    def encode_text_for_row(self, prompt_text: str, device: torch.device) -> torch.Tensor:
        tokens = self.text_tokenizer(prompt_text, return_tensors="pt", truncation=True, max_length=128)
        text_inputs = tokens["input_ids"].to(device)
        text_masks = tokens["attention_mask"].to(device)
        return self.clamp.get_text_features(
            text_inputs=text_inputs,
            text_masks=text_masks,
            get_global=True,
        )

    @torch.no_grad()
    def set_text_prompt(
        self,
        prompt_texts: list[str] | str,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        if isinstance(prompt_texts, str):
            prompt_texts = [prompt_texts]
        emb_list: list[torch.Tensor] = []
        for p in prompt_texts:
            emb = self.encode_text_for_row(str(p), device)
            emb_list.append(emb)
        self.cached_text_emb_list = emb_list

    def forward(self, onehot_generates: torch.Tensor, seq_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self.last_prompt_attr_losses = None
        pad_col = torch.zeros(
            onehot_generates.size(0),
            onehot_generates.size(1),
            1,
            dtype=onehot_generates.dtype,
            device=onehot_generates.device,
        )
        mask_col = torch.zeros_like(pad_col)
        onehot_generates = torch.cat([onehot_generates, pad_col, mask_col], dim=-1)
        emb_w = self.bridge.embedding.weight
        token_inputs = onehot_generates.to(dtype=emb_w.dtype)
        hidden_in = token_inputs @ emb_w
        pad_mask = ~seq_mask.bool()
        hidden = self.bridge.encoder(
            hidden_in,
            src_key_padding_mask=pad_mask,
        )
        bsz = hidden.size(0)
        q = self.bridge.pool_query.expand(bsz, -1, -1)
        pooled, _ = self.bridge.pool_attn(
            q,
            hidden,
            hidden,
            key_padding_mask=pad_mask,
            need_weights=False,
        )
        pooled = pooled.squeeze(1)
        pred = self.bridge.proj(pooled)
        pred = self.bridge.out_norm(pred)
        pred = F.normalize(pred, dim=-1)

        if not self.cached_text_emb_list:
            raise ValueError(
                "Text targets not set. Call set_text_prompt(...) first."
            )
        k = len(self.cached_text_emb_list)
        b = int(pred.size(0))
        if b != k:
            raise ValueError(
                f"Batch size {b} does not match number of text targets {k}. "
                "Expected one prompt target per batch row."
            )

        targets_bd = torch.stack(
            [
                t.to(pred.device, dtype=pred.dtype).squeeze(0)
                if t.ndim == 2 and int(t.size(0)) == 1
                else t.to(pred.device, dtype=pred.dtype)
                for t in self.cached_text_emb_list
            ],
            dim=0,
        )

        if self.attr_loss_type == "cosine":
            pred_n = F.normalize(pred, dim=-1)
            tgt_n = F.normalize(targets_bd, dim=-1)
            cos_b = F.cosine_similarity(pred_n, tgt_n, dim=-1)
            loss_b = 1.0 - cos_b
        else:
            raise ValueError(f"Invalid attr_loss_type: {self.attr_loss_type}")

        attr_loss = loss_b
        self.last_prompt_attr_losses = loss_b.detach().unsqueeze(1)
        return pred, attr_loss


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
