from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from constants import AR_EVENT_VOCAB_SIZE
from utils import _trace_seq, ids_hash

VERBOSE = True

EPS = 1e-10


@dataclass
class DlpRuntime:
    weight_val: Any
    k_val: int
    temp: float
    device: str
    weight_strat: Any
    disc_weight: float
    use_scale_weights: bool
    initialization: str
    initialization_noise_rate: float
    loss_aggregation: str
    bias_update_mode: str
    debug_trace_sequences: bool = False

    model: Any = None
    discriminator: Any = None
    inputs: dict[str, Any] | None = None
    prompt_length: int = 0
    active_vocab_size: int = 0
    embed_map: Any = None
    _set_bias_kwargs: dict[str, Any] = field(default_factory=dict)
    _langevin_step: int = 0
    last_step_debug: dict[str, Any] = field(default_factory=dict)


def create_dlp_runtime(conf: dict[str, Any]) -> DlpRuntime:
    loss_aggregation = str(conf.get("loss_aggregation", "none")).strip().lower()
    if loss_aggregation != "none":
        raise ValueError(
            f"Unsupported loss_aggregation={conf.get('loss_aggregation')!r}. Only 'none' is supported."
        )
    bias_update_mode = str(conf.get("bias_update_mode", "sampled_l2")).strip().lower()
    if bias_update_mode not in {"sampled_l2", "direct_grad"}:
        raise ValueError(
            "Unsupported bias_update_mode="
            f"{conf.get('bias_update_mode')!r}. Use: sampled_l2 | direct_grad"
        )
    return DlpRuntime(
        weight_val=conf["weight_val"],
        k_val=int(conf.get("k_val", 250)),
        temp=float(conf["proposal_temp"]),
        device=str(conf["device"]),
        weight_strat=conf.get("weight_strat", "uniform"),
        disc_weight=float(conf.get("disc_weight", 0.9)),
        use_scale_weights=bool(conf.get("use_scale_weights", True)),
        initialization=conf.get("initialization", "random_disc"),
        initialization_noise_rate=float(conf.get("initialization_noise_rate", 0.5)),
        loss_aggregation=loss_aggregation,
        bias_update_mode=bias_update_mode,
        debug_trace_sequences=bool(conf.get("debug_trace_sequences", False)),
    )


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


def load_run_stack(conf: dict[str, Any]) -> tuple[Any, Any, DlpRuntime]:
    from discriminators import load_ttm_discriminator
    from generators import load_base_model

    device = str(conf["device"])
    model = load_base_model(**conf["base_model_args"]).to(device)
    discriminator = load_ttm_discriminator(**conf).to(device).eval()
    runtime = create_dlp_runtime(conf)
    return model, discriminator, runtime


def _resolve_scale_mode_for_step(runtime: DlpRuntime, *, step_idx: int) -> bool:
    del step_idx
    return runtime.use_scale_weights


def _bridge_valid_mask(
    output_ids: torch.Tensor,
    out_len: int,
) -> torch.Tensor:
    """Mask valid AR-event ids for bridge positions; control/special ids are removed."""
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
    loss_aggregation: str = "none",
):
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
    if debug_trace:
        flat = tok.detach().cpu().reshape(-1).tolist()
        print(
            f"[dlp] trace output_ids: shape={tuple(tok.shape)} "
            f"prefix10={flat[:10]} suffix10={flat[-10:] if flat else []}"
        )
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
    del gen_steps
    if onehot_generates.shape[-1] < AR_EVENT_VOCAB_SIZE:
        raise ValueError(
            f"Generator one-hot width {onehot_generates.shape[-1]} < AR_EVENT_VOCAB_SIZE "
            f"(CONTROL_OFFSET)={AR_EVENT_VOCAB_SIZE}"
        )
    disc_onehot = onehot_generates[:, :, :AR_EVENT_VOCAB_SIZE].float()
    mask_gen = _bridge_valid_mask(
        output_ids,
        int(disc_onehot.shape[1]),
    )
    mask_gen = mask_gen.to(device=disc_onehot.device, dtype=disc_onehot.dtype)
    disc_onehot, mask_gen = _sanitize_disc_bridge_inputs(disc_onehot, mask_gen)
    if debug_trace:
        flat = disc_onehot.detach().cpu().reshape(-1).tolist()
        print(
            f"[dlp] trace disc_onehot: shape={tuple(disc_onehot.shape)} "
            f"prefix10={flat[:10]} suffix10={flat[-10:] if flat else []}"
        )
        flat = mask_gen.detach().cpu().reshape(-1).tolist()
        print(
            f"[dlp] trace mask_gen: shape={tuple(mask_gen.shape)} "
            f"prefix10={flat[:10]} suffix10={flat[-10:] if flat else []}"
        )
    pred_vecs, attr_losses = discriminator(disc_onehot, mask_gen)

    lm_logp, oh_suffix, bias_var, suffix_ids = _align_suffix_tensors(
        output_ids, gpt_logit, onehot_generates, biases, int(prompt_length)
    )
    if debug_trace:
        flat = suffix_ids.float().detach().cpu().reshape(-1).tolist()
        print(
            f"[dlp] trace suffix_ids: shape={tuple(suffix_ids.shape)} "
            f"prefix10={flat[:10]} suffix10={flat[-10:] if flat else []}"
        )
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


def _calc_grad_suffix(
    prompt_length: int,
    loss_for_grad: torch.Tensor,
    onehot: torch.Tensor,
    *,
    retain_graph: bool,
) -> torch.Tensor:
    pl = int(prompt_length)

    if loss_for_grad.ndim != 1:
        raise ValueError(
            f"loss_for_grad must be 1D [B] (per-sample); got shape={tuple(loss_for_grad.shape)}"
        )
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

    gx = torch.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
    gx = gx.clamp(min=-1e3, max=1e3)
    return gx.detach()


def get_unfiltered_dist(
    runtime: DlpRuntime,
    gx: torch.Tensor,
    cur_token_ids: torch.Tensor,
) -> torch.Tensor:
    token_dist = torch.ones_like(gx).to(runtime.device)
    cur = cur_token_ids[:, runtime.prompt_length :].long()
    cur = cur.clamp(0, gx.shape[-1] - 1)
    if cur.shape[1] != token_dist.shape[1]:
        t = min(int(cur.shape[1]), int(token_dist.shape[1]))
        cur = cur[:, :t]
        token_dist = token_dist[:, :t, :]
        gx = gx[:, :t, :]
    token_dist[
        torch.arange(token_dist.size(0))[:, None, None],
        torch.arange(token_dist.size(1))[None, :, None],
    ] = EPS
    unfiltered_dist = gx * token_dist
    return -1 * unfiltered_dist


def _apply_filter(unfiltered_dist: torch.Tensor, topk_ids: torch.Tensor) -> torch.Tensor:
    return unfiltered_dist[
        torch.arange(unfiltered_dist.size(0))[:, None, None],
        torch.arange(unfiltered_dist.size(1))[None, :, None],
        topk_ids,
    ]


def _topk_to_tokens(topk_ids: torch.Tensor, sampled_indices: torch.Tensor) -> torch.Tensor:
    return topk_ids[
        torch.arange(topk_ids.size(0))[:, None],
        torch.arange(topk_ids.size(1))[None, :],
        sampled_indices,
    ]


def _sanitize_dist_logits(dist_logits: torch.Tensor) -> torch.Tensor:
    finite = torch.isfinite(dist_logits)
    valid_rows = finite.any(dim=-1)
    if valid_rows.all():
        return dist_logits
    fixed = dist_logits.clone()
    bad = ~valid_rows
    fixed[bad, 0] = 0.0
    fixed[bad, 1:] = -1e9
    return fixed


def get_dlp_dist(
    runtime: DlpRuntime,
    loss_for_grad: torch.Tensor,
    onehot: torch.Tensor,
    cur_token_ids: torch.Tensor,
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    gx = _calc_grad_suffix(
        runtime.prompt_length, loss_for_grad, onehot, retain_graph=False
    )
    runtime.last_step_debug["grad_norm"] = float(gx.float().norm().item())
    logits = torch.nan_to_num(
        logits[:, runtime.prompt_length :, :].float(),
        nan=-1e9,
        posinf=1e9,
        neginf=-1e9,
    )
    if gx.shape[1] != logits.shape[1]:
        t = min(int(gx.shape[1]), int(logits.shape[1]))
        gx = gx[:, :t, :]
        logits = logits[:, :t, :]
    if runtime.active_vocab_size < logits.shape[-1]:
        logits = logits.clone()
        logits[:, :, runtime.active_vocab_size :] = -float("inf")
    unfiltered_dist = get_unfiltered_dist(runtime, gx, cur_token_ids)
    k = max(1, min(int(runtime.k_val), int(logits.shape[-1])))
    topk_ids = torch.topk(logits, k, dim=-1).indices
    return unfiltered_dist, topk_ids


def compute_p_lm_soft(
    runtime: DlpRuntime,
    loss_for_grad: torch.Tensor,
    output_ids: torch.Tensor,
    onehot: torch.Tensor,
    logits: torch.Tensor,
    attr_losses: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any]:
    unfiltered_dist, topk_ids = get_dlp_dist(
        runtime, loss_for_grad, onehot, output_ids, logits
    )
    dist_logits = _apply_filter(unfiltered_dist, topk_ids)
    dist_logits = _sanitize_dist_logits(dist_logits)
    proposal_dist = torch.distributions.Categorical(logits=dist_logits.float() / runtime.temp)
    probs = torch.softmax(dist_logits.float() / runtime.temp, dim=-1)
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
    runtime.last_step_debug["proposal_entropy"] = float(entropy.item())
    sampled_indices = proposal_dist.sample()
    sampled_tokens = _topk_to_tokens(topk_ids, sampled_indices)
    prev_tokens = output_ids[:, runtime.prompt_length :].detach()
    changed = (sampled_tokens != prev_tokens).float().mean()
    runtime.last_step_debug["proposal_vs_decode_mismatch"] = float(changed.item())
    return loss_for_grad, output_ids, sampled_tokens, attr_losses.detach().cpu().numpy()


def compute_bias_l2_pen(
    runtime: DlpRuntime,
    sampled_ids: torch.Tensor,
    steer_weight: float | None = None,
) -> torch.Tensor:
    sw = float(runtime.weight_val if steer_weight is None else steer_weight)
    with torch.no_grad():
        cur_embeds = runtime.embed_map(sampled_ids)
        t1 = torch.einsum("ve -> v", [runtime.embed_map.weight**2])[None, None, :]
        t2 = torch.einsum("bse, ve -> bsv", [cur_embeds, runtime.embed_map.weight])
        t3 = torch.einsum("bse -> bs", [cur_embeds**2]).unsqueeze(-1)
        bias = -1 * sw * (t1 - 2 * t2 + t3)
    return bias


def initialize_dlp_batch(
    runtime: DlpRuntime,
    model: Any,
    discriminator: Any,
    batch_size: int,
    seq_length: int,
    prompt_length: int,
    inputs: dict[str, Any],
    **kwargs: Any,
) -> tuple[dict[str, Any], torch.Tensor]:
    runtime._langevin_step = 0
    kwargs.pop("num_steps", None)
    runtime.model = model
    runtime.discriminator = discriminator
    runtime.active_vocab_size = int(
        getattr(discriminator, "active_vocab_size", model.get_input_embeddings().weight.size(0))
    )
    runtime.inputs = inputs
    runtime.debug_trace_sequences = bool(
        inputs.get("debug_trace_sequences", runtime.debug_trace_sequences)
    )
    runtime.prompt_length = prompt_length
    runtime._set_bias_kwargs = dict(kwargs)
    init_scale_mode = _resolve_scale_mode_for_step(runtime, step_idx=0)
    model.set_biases(
        batch_size=batch_size,
        seq_len=seq_length,
        prompt_length=prompt_length,
        attribute=None,
        device=runtime.device,
        disc_weight=runtime.weight_val,
        use_scale_weights=init_scale_mode,
        **kwargs,
    )
    runtime.embed_map = model.get_input_embeddings()
    logit_dim = model.get_input_embeddings().weight.size(0)
    embed_dim = model.get_input_embeddings().weight.size(1)
    last_dim = logit_dim
    if runtime.initialization == "random_disc":
        sampled_ints = torch.randint(
            0, logit_dim, (batch_size, seq_length - prompt_length)
        ).to(runtime.device)
        if last_dim == embed_dim:
            initial_bias = runtime.embed_map(sampled_ints)
        else:
            initial_bias = compute_bias_l2_pen(runtime, sampled_ints)
    elif runtime.initialization == "random_cont":
        initial_bias = runtime.initialization_noise_rate * torch.randn(
            batch_size, seq_length - prompt_length, last_dim
        ).to(runtime.device)
    elif runtime.initialization == "zero":
        initial_bias = torch.zeros(
            batch_size, seq_length - prompt_length, last_dim
        ).to(runtime.device)
    else:
        raise ValueError(
            f"Unsupported initialization mode: {runtime.initialization}. "
            "Use: zero | random_disc | random_cont"
        )
    initial_bias = initial_bias.detach()
    initial_bias.requires_grad = True
    _trace_seq(runtime, "initialize.input_ids", inputs["input_ids"])
    return inputs, initial_bias


def one_step_direct_grad(
    runtime: DlpRuntime,
    x: torch.Tensor,
    *,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    steer_w = float(runtime.weight_val)
    cur_step_idx = int(runtime._langevin_step)
    if VERBOSE:
        print(f"\n[VERBOSE] === one_step_direct_grad step={cur_step_idx} steer_w={steer_w:.6f} ===")
        _vt("x (cur_batch input) [B,gen_steps,vocab]", x)

    scale_mode = _resolve_scale_mode_for_step(runtime, step_idx=cur_step_idx)
    assert runtime.model is not None
    runtime.model.set_biases(
        batch_size=x.size(0),
        seq_len=x.size(1) + int(prompt_length),
        prompt_length=prompt_length,
        attribute=None,
        device=runtime.device,
        disc_weight=runtime.weight_val,
        use_scale_weights=scale_mode,
        **runtime._set_bias_kwargs,
    )
    runtime.last_step_debug["scale_mode"] = str(scale_mode)
    runtime._langevin_step += 1
    runtime.last_step_debug["steer_weight"] = steer_w

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
    assert runtime.inputs is not None and runtime.discriminator is not None
    (
        loss,
        loss_for_grad,
        output_ids,
        onehot,
        logits,
        attr_losses,
        term_stats,
        _pred_vecs,
        lm_reg_per_sample,
        _loss_per_sample,
    ) = compute_steered_loss(
        runtime.model,
        runtime.discriminator,
        runtime.inputs,
        x_full,
        weight=steer_w,
        prompt_length=prompt_length,
        loss_aggregation=runtime.loss_aggregation,
    )
    gx = _calc_grad_suffix(prompt_length, loss_for_grad, onehot, retain_graph=True)
    bias = torch.zeros_like(x)
    t_use = min(int(bias.shape[1]), int(gx.shape[1]))
    v_use = min(int(bias.shape[2]), int(gx.shape[2]))
    if t_use > 0 and v_use > 0:
        bias[:, :t_use, :v_use] = -gx[:, :t_use, :v_use]

    _loss_for_grad, output_ids, _sampled_ids, attr_losses_np = compute_p_lm_soft(
        runtime, loss_for_grad, output_ids, onehot, logits, attr_losses
    )
    attr_losses_t = torch.as_tensor(attr_losses_np, dtype=torch.float32).reshape(-1)
    loss_per_sample = loss_for_grad.detach().float().reshape(-1)
    grad_norm_per_sample = gx.float().flatten(1).norm(dim=1)
    bias_norm_per_sample = bias.float().flatten(1).norm(dim=1)
    step_debugs: list[dict[str, Any]] = []
    for i in range(int(loss_per_sample.shape[0])):
        d = dict(term_stats)
        lm_i = float(lm_reg_per_sample[i].detach().item())
        d["steer_weight"] = steer_w
        d["scale_mode"] = str(scale_mode)
        # Critical for batched selection: use row-local LM metric, not batch mean.
        d["lm_mean"] = lm_i
        d["weighted_lm"] = float(d["w_lm"]) * lm_i
        d["grad_norm"] = float(grad_norm_per_sample[i].item())
        d["bias_norm"] = float(bias_norm_per_sample[i].item())
        d["proposal_entropy"] = float("nan")
        d["proposal_vs_decode_mismatch"] = float("nan")
        step_debugs.append(d)
    del loss
    return bias, loss_per_sample, output_ids.detach(), attr_losses_t.detach(), step_debugs


def one_step_sampled_l2(
    runtime: DlpRuntime,
    x: torch.Tensor,
    *,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    steer_w = float(runtime.weight_val)
    cur_step_idx = int(runtime._langevin_step)
    if VERBOSE:
        print(f"\n[VERBOSE] === one_step_sampled_l2 step={cur_step_idx} steer_w={steer_w:.6f} ===")
        _vt("x (cur_batch input) [B,gen_steps,vocab]", x)

    scale_mode = _resolve_scale_mode_for_step(runtime, step_idx=cur_step_idx)
    assert runtime.model is not None
    runtime.model.set_biases(
        batch_size=x.size(0),
        seq_len=x.size(1) + int(prompt_length),
        prompt_length=prompt_length,
        attribute=None,
        device=runtime.device,
        disc_weight=runtime.weight_val,
        use_scale_weights=scale_mode,
        **runtime._set_bias_kwargs,
    )
    runtime.last_step_debug["scale_mode"] = str(scale_mode)
    runtime._langevin_step += 1
    runtime.last_step_debug["steer_weight"] = steer_w

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
    assert runtime.inputs is not None and runtime.discriminator is not None
    (
        loss,
        loss_for_grad,
        output_ids,
        onehot,
        logits,
        attr_losses,
        term_stats,
        _pred_vecs,
        lm_reg_per_sample,
        _loss_per_sample,
    ) = compute_steered_loss(
        runtime.model,
        runtime.discriminator,
        runtime.inputs,
        x_full,
        weight=steer_w,
        prompt_length=prompt_length,
        loss_aggregation=runtime.loss_aggregation,
    )
    _loss_for_grad, output_ids, sampled_ids, attr_losses_np = compute_p_lm_soft(
        runtime, loss_for_grad, output_ids, onehot, logits, attr_losses
    )
    bias = compute_bias_l2_pen(runtime, sampled_ids, steer_weight=steer_w)
    attr_losses_t = torch.as_tensor(attr_losses_np, dtype=torch.float32).reshape(-1)
    loss_per_sample = loss_for_grad.detach().float().reshape(-1)
    bias_norm_per_sample = bias.float().flatten(1).norm(dim=1)
    step_debugs: list[dict[str, Any]] = []
    for i in range(int(loss_per_sample.shape[0])):
        d = dict(term_stats)
        lm_i = float(lm_reg_per_sample[i].detach().item())
        d["steer_weight"] = steer_w
        d["scale_mode"] = str(scale_mode)
        # Critical for batched selection: use row-local LM metric, not batch mean.
        d["lm_mean"] = lm_i
        d["weighted_lm"] = float(d["w_lm"]) * lm_i
        d["bias_norm"] = float(bias_norm_per_sample[i].item())
        d["grad_norm"] = float("nan")
        d["proposal_entropy"] = float(runtime.last_step_debug.get("proposal_entropy", float("nan")))
        d["proposal_vs_decode_mismatch"] = float(
            runtime.last_step_debug.get("proposal_vs_decode_mismatch", float("nan"))
        )
        step_debugs.append(d)
    del loss
    return bias, loss_per_sample, output_ids.detach(), attr_losses_t.detach(), step_debugs
