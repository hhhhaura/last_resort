"""Steering glue between base model, sampler, and discriminator.

Autograd note (TTM / ttm_fast): ``compute_steered_loss`` builds one graph that runs the base LM
(``forward_with_biases``) and the discriminator on the same batch. True overlap of a "generator"
forward vs a "discriminator" forward for the *same* step would require splitting this into two
phases with well-defined tensors between them, or overlapping *different* batches on different CUDA
streams (different graphs). See ``dab/ttm_fast.py`` for the dual-queue sketch and mode flags.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import AR_EVENT_VOCAB_SIZE


def _trace_seq(name: str, x: torch.Tensor, *, enabled: bool) -> None:
    if not enabled:
        return
    flat = x.detach().cpu().reshape(-1).tolist()
    print(
        f"[steer] trace {name}: shape={tuple(x.shape)} "
        f"prefix10={flat[:10]} suffix10={flat[-10:] if flat else []}"
    )


def _hard_onehot_ar_vocab(tok: torch.Tensor, *, ref: torch.Tensor) -> torch.Tensor:
    tok = tok.long()
    valid = (tok >= 0) & (tok < AR_EVENT_VOCAB_SIZE)
    clamped = tok.clamp(0, AR_EVENT_VOCAB_SIZE - 1)
    oh = F.one_hot(clamped, num_classes=AR_EVENT_VOCAB_SIZE).to(dtype=ref.dtype, device=ref.device)
    return oh * valid.unsqueeze(-1).to(dtype=oh.dtype)


def _bridge_soft_onehot_window(
    onehot_generates: torch.Tensor,
    output_ids: torch.Tensor,
    prompt_length: int,
    *,
    max_generated_steps: int | None = None,
):
    """Content-only bridge: hard prompt prefix, soft generated suffix."""
    pl = int(prompt_length)
    bsz, t, _vdim = onehot_generates.shape
    if pl <= 0:
        return None
    tok = output_ids[:, :pl].long()
    hard_head = _hard_onehot_ar_vocab(tok, ref=onehot_generates)
    if max_generated_steps is None:
        tail_end = t
    else:
        tail_end = min(t, pl + max(int(max_generated_steps), 0))
    soft_tail = onehot_generates[:, pl:tail_end, :AR_EVENT_VOCAB_SIZE]
    return torch.cat([hard_head, soft_tail], dim=1)


def _bridge_valid_mask(
    output_ids: torch.Tensor,
    prompt_length: int,
    out_len: int,
    *,
    max_generated_steps: int | None = None,
) -> torch.Tensor:
    """Mask valid AR-event ids for bridge positions; control/special ids are removed."""
    del prompt_length, max_generated_steps
    end = int(out_len)
    tok = output_ids[:, :end].long()
    if tok.shape[1] != int(out_len):
        raise ValueError(
            f"bridge mask length mismatch: got={tok.shape[1]} expected={int(out_len)}. "
            "dab_ttm requires strict equal-length, non-padded sequences."
        )
    valid = (tok >= 0) & (tok < AR_EVENT_VOCAB_SIZE)
    return valid.to(dtype=torch.float32, device=output_ids.device)


def _sanitize_disc_bridge_inputs(
    disc_onehot: torch.Tensor, mask_gen: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Make discriminator bridge inputs numerically/statically safe for GEMM."""
    if disc_onehot.ndim != 3:
        raise ValueError(f"disc_onehot must be 3D [B,S,V], got shape={tuple(disc_onehot.shape)}")
    if mask_gen.ndim != 2:
        raise ValueError(f"mask_gen must be 2D [B,S], got shape={tuple(mask_gen.shape)}")
    if disc_onehot.shape[0] != mask_gen.shape[0] or disc_onehot.shape[1] != mask_gen.shape[1]:
        raise ValueError(
            "disc_onehot/mask_gen shape mismatch: "
            f"disc_onehot={tuple(disc_onehot.shape)} mask_gen={tuple(mask_gen.shape)}"
        )
    # Student bridge does x @ w; keep x in fp32 to avoid bf16 GEMM instability.
    x = torch.nan_to_num(disc_onehot.float(), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    m = torch.nan_to_num(mask_gen.float(), nan=0.0, posinf=0.0, neginf=0.0).contiguous()
    # Zero-out fully invalid rows explicitly (already masked) for extra stability.
    valid_any = m.any(dim=1, keepdim=True)
    # Avoid CUDA sync (.item) in error states; mask invalid rows unconditionally.
    x = x * valid_any.unsqueeze(-1).to(dtype=x.dtype)
    return x, m


def _align_suffix_tensors(
    output_ids: torch.Tensor,
    gpt_logit: torch.Tensor,
    onehot_generates: torch.Tensor,
    biases: torch.Tensor,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Align LM/onehot/bias suffix lengths to a shared time axis."""
    t_logits = int(gpt_logit[:, prompt_length:, :].shape[1])
    t_onehot = int(onehot_generates[:, prompt_length:, :].shape[1])
    t_bias = int(biases[:, prompt_length:, :].shape[1])
    t = min(t_logits, t_onehot, t_bias)
    if t <= 0:
        raise ValueError(
            f"invalid aligned suffix length: logits={t_logits}, onehot={t_onehot}, bias={t_bias}"
        )
    lm_logp = torch.log_softmax(gpt_logit[:, prompt_length : prompt_length + t, :].float(), dim=-1)
    oh_suffix = onehot_generates[:, prompt_length : prompt_length + t, :AR_EVENT_VOCAB_SIZE].float()
    bias_var = biases[:, prompt_length : prompt_length + t, :AR_EVENT_VOCAB_SIZE].float()
    suffix_ids = output_ids[:, prompt_length : prompt_length + t].long()
    return lm_logp, oh_suffix, bias_var, suffix_ids


def compute_steered_loss(
    model,
    discriminator,
    inputs,
    biases,
    *,
    weight,
    prompt_length=None,
    target_density=None,
    loss_aggregation: str = "none",
):
    del target_density
    debug_trace = bool(inputs.get("debug_trace_sequences", False))
    output_ids, gpt_logit = model.forward_with_biases(
        **inputs,
        labels=inputs,
        use_full_prompt=False,
        biases=biases,
        bias_rep_space="logit",
        weight=weight,
    )

    if prompt_length is None:
        raise ValueError("prompt_length is required for bridge-guided steering.")

    emb = model.get_input_embeddings()
    tok = output_ids.long()
    _trace_seq("output_ids", tok, enabled=debug_trace)
    vdim = int(model.vocab_size)
    tok_valid = (tok >= 0) & (tok < vdim)
    if not bool(tok_valid.all().item()):
        bad = (~tok_valid).nonzero(as_tuple=False)
        ex = bad[:8].detach().cpu().tolist()
        raise ValueError(
            "dab_ttm steering received out-of-range token ids; padded/sentinel rows are unsupported. "
            f"examples={ex} vocab_size={vdim}"
        )
    tok_safe = tok.clamp(0, vdim - 1)
    onehot_generates = F.one_hot(tok_safe, num_classes=vdim).to(
        dtype=emb.weight.dtype, device=output_ids.device
    )
    onehot_generates = onehot_generates * tok_valid.unsqueeze(-1).to(dtype=onehot_generates.dtype)
    onehot_generates = onehot_generates.detach().requires_grad_(True)

    gen_steps = max(int(gpt_logit.shape[1]) - int(prompt_length), 0)
    bridged = _bridge_soft_onehot_window(
        onehot_generates,
        output_ids,
        int(prompt_length),
        max_generated_steps=gen_steps,
    )
    if bridged is not None:
        disc_onehot = bridged.float()
    else:
        gen_oh = onehot_generates[:, prompt_length:, :].float()
        if gen_oh.shape[-1] < AR_EVENT_VOCAB_SIZE:
            raise ValueError(
                f"Generator one-hot width {gen_oh.shape[-1]} < AR_EVENT_VOCAB_SIZE "
                f"(CONTROL_OFFSET)={AR_EVENT_VOCAB_SIZE}"
            )
        disc_onehot = gen_oh[:, :, :AR_EVENT_VOCAB_SIZE]
    mask_gen = _bridge_valid_mask(
        output_ids,
        int(prompt_length),
        int(disc_onehot.shape[1]),
        max_generated_steps=gen_steps,
    )
    mask_gen = mask_gen.to(device=disc_onehot.device, dtype=disc_onehot.dtype)
    disc_onehot, mask_gen = _sanitize_disc_bridge_inputs(disc_onehot, mask_gen)
    _trace_seq("disc_onehot", disc_onehot, enabled=debug_trace)
    _trace_seq("mask_gen", mask_gen, enabled=debug_trace)
    pred_vecs, attr_losses = discriminator(disc_onehot, mask_gen)

    lm_logp, oh_suffix, bias_var, suffix_ids = _align_suffix_tensors(
        output_ids, gpt_logit, onehot_generates, biases, int(prompt_length)
    )
    _trace_seq("suffix_ids", suffix_ids.float(), enabled=debug_trace)
    lm_logp = torch.nan_to_num(lm_logp, nan=0.0, posinf=0.0, neginf=-60.0)
    lm_logp = lm_logp[:, :, :AR_EVENT_VOCAB_SIZE]
    pad_mask = ((suffix_ids >= 0) & (suffix_ids < AR_EVENT_VOCAB_SIZE)).to(dtype=lm_logp.dtype)
    denom = pad_mask.sum(dim=1).clamp_min(1.0)
    lm_tok = -(oh_suffix * lm_logp).sum(dim=-1)
    lm_reg = (lm_tok * pad_mask).sum(dim=1) / denom
    # Regularize the actual bias optimization variable x (excluding prompt rows),
    # not the generated onehot proxy.
    bias_tok = bias_var.pow(2).mean(dim=-1)
    bias_reg = (bias_tok * pad_mask).sum(dim=1) / denom

    w_attr = float(getattr(discriminator, "cosine_weight", 1.0))
    w_lm = float(getattr(discriminator, "lm_reg_weight", 0.2))
    w_bias = float(getattr(discriminator, "bias_reg_weight", 0.01))
    loss_per_sample = (w_attr * attr_losses) + (w_lm * lm_reg) + (w_bias * bias_reg)
    agg = str(loss_aggregation).strip().lower()
    if agg != "none":
        raise ValueError(
            f"Unsupported loss_aggregation={loss_aggregation!r}; only 'none' is supported."
        )
    loss_for_grad = loss_per_sample
    loss = loss_per_sample.mean()

    attr_mean = attr_losses.mean()
    lm_mean = lm_reg.mean()
    bias_mean = bias_reg.mean()
    weighted_attr = w_attr * attr_mean
    weighted_lm = w_lm * lm_mean
    weighted_bias = w_bias * bias_mean
    term_stats = {
        "attr_mean": float(attr_mean.detach().item()),
        "lm_mean": float(lm_mean.detach().item()),
        "bias_mean": float(bias_mean.detach().item()),
        "pred_norm_mean": float(pred_vecs.detach().float().norm(dim=-1).mean().item()),
        "w_attr": w_attr,
        "w_lm": w_lm,
        "w_bias": w_bias,
        "weighted_attr": float(weighted_attr.detach().item()),
        "weighted_lm": float(weighted_lm.detach().item()),
        "weighted_bias": float(weighted_bias.detach().item()),
        "loss_aggregation": agg,
    }
    return (
        loss,
        loss_for_grad,
        output_ids,
        onehot_generates,
        gpt_logit,
        attr_losses,
        term_stats,
        pred_vecs.detach(),
        lm_reg.detach(),
        loss_per_sample.detach(),
    )

