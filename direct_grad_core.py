from __future__ import annotations

import array
import hashlib
import sys
from pathlib import Path
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sampler import LangevinSampler
from steering import compute_steered_loss

VERBOSE = True


def _vt(name: str, t: torch.Tensor, n: int = 5) -> None:
    if not VERBOSE:
        return
    flat = t.detach().float().reshape(-1)
    numel = flat.numel()
    head = [f"{v:.4f}" for v in flat[:n].tolist()]
    tail = [f"{v:.4f}" for v in flat[-n:].tolist()] if numel > n else []
    finite = flat[torch.isfinite(flat)]
    stats = (
        f"min={finite.min():.4f} max={finite.max():.4f} "
        f"mean={finite.mean():.4f} norm={flat.norm():.4f} "
        f"nan={int((~torch.isfinite(flat)).sum())} numel={numel}"
        if finite.numel()
        else f"all-non-finite numel={numel}"
    )
    tail_str = f" ... tail=[{', '.join(tail)}]" if tail else ""
    print(f"  [VERBOSE] {name}: shape={tuple(t.shape)} dtype={t.dtype} dev={t.device}")
    print(f"  [VERBOSE] {name}: head=[{', '.join(head)}]{tail_str}")
    print(f"  [VERBOSE] {name}: {stats}")


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_run_stack(conf: dict[str, Any]) -> tuple[Any, Any, LangevinSampler]:
    from discriminators import load_ttm_discriminator
    from generators import load_base_model

    device = str(conf["device"])
    model = load_base_model(**conf["base_model_args"]).to(device)
    discriminator = load_ttm_discriminator(**conf).to(device).eval()
    sampler = LangevinSampler(**conf)
    return model, discriminator, sampler


def ids_hash(ids: list[int]) -> str:
    arr = array.array("q", [int(x) for x in ids])
    return hashlib.sha1(arr.tobytes()).hexdigest()[:12]


def _calc_grad_suffix(
    sampler: LangevinSampler,
    loss_for_grad: torch.Tensor,
    onehot: torch.Tensor,
    *,
    retain_graph: bool,
) -> torch.Tensor:
    if not torch.is_tensor(loss_for_grad):
        raise TypeError(f"loss_for_grad must be a tensor, got {type(loss_for_grad)}")
    pl = int(sampler.prompt_length)

    if loss_for_grad.ndim == 0:
        gx_all = torch.autograd.grad(loss_for_grad, onehot, retain_graph=retain_graph, allow_unused=True)[0]
        if gx_all is None:
            raise RuntimeError("Gradient is None for scalar loss_for_grad.")
        gx = gx_all[:, pl:, :]
    elif loss_for_grad.ndim == 1:
        bsz = int(loss_for_grad.shape[0])
        if bsz != int(onehot.shape[0]):
            raise ValueError(
                "loss_for_grad batch mismatch: "
                f"loss_for_grad={tuple(loss_for_grad.shape)} onehot={tuple(onehot.shape)}"
            )
        rows = []
        for b in range(bsz):
            retain_this = bool(retain_graph) or (b < (bsz - 1))
            g_all = torch.autograd.grad(
                loss_for_grad[b],
                onehot,
                retain_graph=retain_this,
                allow_unused=True,
            )[0]
            if g_all is None:
                raise RuntimeError(f"Gradient is None for loss_for_grad[{b}].")
            rows.append(g_all[b : b + 1, pl:, :])
        gx = torch.cat(rows, dim=0)
    else:
        raise ValueError(f"loss_for_grad must be scalar or [B], got shape={tuple(loss_for_grad.shape)}")

    gx = torch.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
    gx = gx.clamp(min=-1e3, max=1e3)
    return gx.detach()


def one_step_with_direct_norm_matched_bias(
    sampler: LangevinSampler,
    x: torch.Tensor,
    *,
    prompt_length: int,
    ar_event_only: bool,
    norm_match_eps: float,
    use_masked_full_vocab_norm: bool,
    use_lm_topk_support_norm: bool,
    topk_k: int,
    ratio_min: float,
    ratio_max: float,
) -> tuple[torch.Tensor, float, torch.Tensor, torch.Tensor, float, dict[str, Any]]:
    steer_w = float(sampler.weight_val)
    cur_step_idx = int(sampler._langevin_step)
    if VERBOSE:
        print(f"\n[VERBOSE] === one_step_with_direct_norm_matched_bias step={cur_step_idx} steer_w={steer_w:.6f} ===")
        _vt("x (cur_batch input) [B,gen_steps,vocab]", x)

    scale_mode = sampler._resolve_scale_mode_for_step(step_idx=cur_step_idx)
    sampler.model.set_biases(
        batch_size=x.size(0),
        seq_len=x.size(1) + int(prompt_length),
        prompt_length=prompt_length,
        attribute=None,
        device=sampler.device,
        disc_weight=sampler.weight_val,
        use_scale_weights=scale_mode,
        **sampler._set_bias_kwargs,
    )
    sampler.last_step_debug["scale_mode"] = str(scale_mode)
    sampler._langevin_step += 1
    sampler.last_step_debug["steer_weight"] = steer_w

    batch_size = x.size(0)
    bias_dim = x.size(-1)
    prompt_bias = torch.zeros(
        batch_size,
        prompt_length,
        bias_dim,
        device=x.device,
        dtype=x.dtype,
    )
    x_full = torch.cat([prompt_bias, x], dim=1)
    (
        loss,
        loss_for_grad,
        output_ids,
        onehot,
        logits,
        attr_losses,
        term_stats,
        _pred_vecs,
        _lm_reg,
        _loss_per_sample,
    ) = compute_steered_loss(
        sampler.model,
        sampler.discriminator,
        sampler._inputs,
        x_full,
        weight=steer_w,
        prompt_length=prompt_length,
        loss_aggregation=sampler.loss_aggregation,
    )
    sampler.last_step_debug.update(term_stats)
    gx = _calc_grad_suffix(sampler, loss_for_grad, onehot, retain_graph=True)
    sampler.last_step_debug["grad_norm"] = float(gx.float().norm().item())
    active_vocab_size = int(
        getattr(sampler.discriminator, "active_vocab_size", int(sampler.model.get_input_embeddings().weight.size(0)))
    )
    # NOTE: bypass norm-matching and use direct negative gradient as bias.
    # Original norm-matched construction is intentionally disabled:
    # bias, nm_stats = _build_direct_norm_matched_bias(...)
    bias = torch.zeros_like(x)
    t_use = min(int(bias.shape[1]), int(gx.shape[1]))
    v_use = min(int(bias.shape[2]), int(gx.shape[2]))
    if t_use > 0 and v_use > 0:
        bias[:, :t_use, :v_use] = -gx[:, :t_use, :v_use]
    nm_stats = {
        "norm_match_ratio_mean": -1.0,
        "norm_match_ratio_median": -1.0,
        "norm_match_bn_tiny_frac": -1.0,
        "norm_match_steps_used": float(t_use),
        "norm_match_use_masked_full_vocab_norm": -1.0,
        "norm_match_use_lm_topk_support_norm": -1.0,
        "norm_match_topk_k": -1.0,
        "norm_match_ratio_min": -1.0,
        "norm_match_ratio_max": -1.0,
        "norm_match_active_vocab_cap": float(active_vocab_size),
    }

    _loss_for_grad, output_ids, sampled_ids, attr_losses_np = sampler.compute_p_lm_soft(
        loss_for_grad, output_ids, onehot, logits, attr_losses
    )
    sampled_full = torch.cat([output_ids[:, :prompt_length].long(), sampled_ids.long()], dim=1)
    sampler.last_step_debug["bias_norm"] = float(bias.float().norm().item())
    sampler.last_step_debug.update(nm_stats)
    attr_losses_t = torch.as_tensor(attr_losses_np, dtype=torch.float32).reshape(-1)
    attr_loss = float(attr_losses_t[0].item())
    return bias, float(loss.item()), output_ids.detach(), sampled_full.detach(), attr_loss, dict(sampler.last_step_debug)
