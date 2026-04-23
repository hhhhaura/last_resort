from __future__ import annotations
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EPS = 1e-10


class BaseSampler(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def initialize_batch(self, model, seq_length):
        raise NotImplementedError

    def step(self, **kwargs):
        raise NotImplementedError


def _calc_grad_suffix(
    sampler: "LangevinSampler",
    loss_for_grad: torch.Tensor,
    onehot: torch.Tensor,
    *,
    retain_graph: bool,
) -> torch.Tensor:
    """Row-wise ``autograd.grad`` w.r.t. ``onehot`` on positions ``[prompt_length:, :]``.

    - ``retain_graph=False`` (via ``LangevinSampler.calc_grad``): proposal path inside
      ``compute_p_lm_soft`` / ``get_dlp_dist`` — frees the graph after this backward.
    - ``retain_graph=True`` (from ``LangevinSampler.step`` direct-grad branch): snapshot
      gradient **before** ``compute_p_lm_soft`` so a second backward on ``onehot`` still works.
    """
    if not torch.is_tensor(loss_for_grad):
        raise TypeError(f"loss_for_grad must be a tensor, got {type(loss_for_grad)}")

    pl = int(sampler.prompt_length)
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


class LangevinSampler(BaseSampler):
    def __init__(
        self,
        weight_val,
        proposal_temp,
        device,
        k_val=250,
        weight_strat="uniform",
        disc_weight=0.9,
        use_scale_weights=True,
        initialization="random_disc",
        initialization_noise_rate=0.5,
        loss_aggregation="none",
        bias_update_mode="sampled_l2",
        **kwargs,
    ):
        super().__init__()
        self.weight_val = weight_val
        self.a_s = []
        self.hops = []
        self.k_val = int(k_val)
        self.temp = float(proposal_temp)
        self.device = str(device)
        self.weight_strat = weight_strat
        self.disc_weight = disc_weight
        self.use_scale_weights = use_scale_weights
        self.initialization = initialization
        self.initialization_noise_rate = initialization_noise_rate
        self.loss_aggregation = str(loss_aggregation).strip().lower()
        if self.loss_aggregation != "none":
            raise ValueError(
                f"Unsupported loss_aggregation={loss_aggregation!r}. Only 'none' is supported."
            )
        self.bias_update_mode = str(bias_update_mode).strip().lower()
        if self.bias_update_mode not in {"sampled_l2", "direct_grad"}:
            raise ValueError(
                "Unsupported bias_update_mode="
                f"{bias_update_mode!r}. Use: sampled_l2 | direct_grad"
            )
        self._langevin_step = 0
        self.last_step_debug = {}
        self._set_bias_kwargs = {}
        self._debug_trace_sequences = bool(kwargs.get("debug_trace_sequences", False))

    def _trace_seq(self, name: str, x: torch.Tensor) -> None:
        if not self._debug_trace_sequences:
            return
        flat = x.detach().cpu().reshape(-1).tolist()
        print(
            f"[sampler] trace {name}: shape={tuple(x.shape)} "
            f"prefix10={flat[:10]} suffix10={flat[-10:] if flat else []}"
        )

    def initialize_batch(
        self,
        model,
        discriminator,
        batch_size,
        seq_length,
        prompt_length,
        inputs,
        **kwargs,
    ):
        self._langevin_step = 0
        kwargs.pop("num_steps", None)
        self.model = model
        self.discriminator = discriminator
        self.active_vocab_size = int(
            getattr(discriminator, "active_vocab_size", model.get_input_embeddings().weight.size(0))
        )
        self._inputs = inputs
        self._debug_trace_sequences = bool(inputs.get("debug_trace_sequences", self._debug_trace_sequences))
        self.prompt_length = prompt_length
        self._set_bias_kwargs = dict(kwargs)
        init_scale_mode = self._resolve_scale_mode_for_step(step_idx=0)
        model.set_biases(
            batch_size=batch_size,
            seq_len=seq_length,
            prompt_length=prompt_length,
            attribute=None,
            device=self.device,
            disc_weight=self.weight_val,
            use_scale_weights=init_scale_mode,
            **kwargs,
        )
        self.embed_map = model.get_input_embeddings()
        logit_dim = model.get_input_embeddings().weight.size(0)
        embed_dim = model.get_input_embeddings().weight.size(1)
        last_dim = logit_dim
        if self.initialization == "random_disc":
            sampled_ints = torch.randint(
                0, logit_dim, (batch_size, seq_length - prompt_length)
            ).to(self.device)
            if last_dim == embed_dim:
                initial_bias = self.embed_map(sampled_ints)
            else:
                initial_bias = self.compute_bias_l2_pen(sampled_ints)
        elif self.initialization == "random_cont":
            initial_bias = self.initialization_noise_rate * torch.randn(
                batch_size, seq_length - prompt_length, last_dim
            ).to(self.device)
        elif self.initialization == "zero":
            initial_bias = torch.zeros(
                batch_size, seq_length - prompt_length, last_dim
            ).to(self.device)
        else:
            raise ValueError(
                f"Unsupported initialization mode: {self.initialization}. "
                "Use: zero | random_disc | random_cont"
            )
        self.weights = self.weight_val
        initial_bias = initial_bias.detach()
        initial_bias.requires_grad = True
        self._trace_seq("initialize.input_ids", inputs["input_ids"])
        return inputs, initial_bias

    def get_unfiltered_dist(self, gx, cur_token_ids):
        """Turn ∂loss/∂onehot into unnormalized scores over vocab, before top-k.

        - ``gx[b,t,v]`` is (clamped) gradient w.r.t. soft mass on token ``v`` at generated
          position ``t``. Increasing one-hot mass on tokens that *reduce* loss gives *negative*
          ``gx`` on those indices (typically).
        - We **down-weight the current hard token** by multiplying its column by ``EPS`` so
          the proposal can move off the decode state (otherwise that column often dominates).
        - We return **``-gx * mask``**: the negative makes high-gradient "good" directions into
          **larger** logits for the categorical proposal (we want to sample alternatives that
          the steered loss favors).
        After this, ``_apply_filter`` keeps only the LM top-``k`` columns; those values are the
        **logits** (unnormalized) for a ``Categorical`` over ``k`` *slots*, not over full ``V``.
        """
        token_dist = torch.ones_like(gx).to(self.device)
        cur = cur_token_ids[:, self.prompt_length :].long()
        # Guard against any out-of-range ids reaching advanced indexing on CUDA.
        cur = cur.clamp(0, gx.shape[-1] - 1)
        if cur.shape[1] != token_dist.shape[1]:
            # Keep proposal geometry consistent if prompt slicing shifted this step.
            t = min(int(cur.shape[1]), int(token_dist.shape[1]))
            cur = cur[:, :t]
            token_dist = token_dist[:, :t, :]
            gx = gx[:, :t, :]
        token_dist[
            torch.arange(token_dist.size(0))[:, None, None],
            torch.arange(token_dist.size(1))[None, :, None],
            # cur.unsqueeze(-1),
        ] = EPS
        unfiltered_dist = gx * token_dist
        return -1 * unfiltered_dist

    def _apply_filter(self, unfiltered_dist, topk_ids):
        filtered_dist_logits = unfiltered_dist[
            torch.arange(unfiltered_dist.size(0))[:, None, None],
            torch.arange(unfiltered_dist.size(1))[None, :, None],
            topk_ids,
        ]
        return filtered_dist_logits

    def _topk_to_tokens(self, topk_ids, sampled_indices):
        actual_ids = topk_ids[
            torch.arange(topk_ids.size(0))[:, None],
            torch.arange(topk_ids.size(1))[None, :],
            sampled_indices,
        ]
        return actual_ids

    def get_dlp_dist(self, loss_for_grad, onehot, cur_token_ids, logits):
        gx = _calc_grad_suffix(self, loss_for_grad, onehot, retain_graph=False)
        self.last_step_debug["grad_norm"] = float(gx.float().norm().item())
        logits = torch.nan_to_num(logits[:, self.prompt_length :, :].float(), nan=-1e9, posinf=1e9, neginf=-1e9)
        if gx.shape[1] != logits.shape[1]:
            t = min(int(gx.shape[1]), int(logits.shape[1]))
            gx = gx[:, :t, :]
            logits = logits[:, :t, :]
        if self.active_vocab_size < logits.shape[-1]:
            logits = logits.clone()
            logits[:, :, self.active_vocab_size :] = -float("inf")
        unfiltered_dist = self.get_unfiltered_dist(gx, cur_token_ids)
        k = max(1, min(int(self.k_val), int(logits.shape[-1])))
        topk_ids = torch.topk(logits, k, dim=-1).indices
        return unfiltered_dist, topk_ids

    def compute_p_lm_soft(self, loss_for_grad, output_ids, onehot, logits, attr_losses):
        unfiltered_dist, topk_ids = self.get_dlp_dist(loss_for_grad, onehot, output_ids, logits)
        dist_logits = self._apply_filter(unfiltered_dist, topk_ids)
        dist_logits = self._sanitize_dist_logits(dist_logits)
        proposal_dist = torch.distributions.Categorical(logits=dist_logits.float() / self.temp)
        probs = torch.softmax(dist_logits.float() / self.temp, dim=-1)
        entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=-1).mean()
        self.last_step_debug["proposal_entropy"] = float(entropy.item())
        sampled_indices = proposal_dist.sample()
        sampled_tokens = self._topk_to_tokens(topk_ids, sampled_indices)
        prev_tokens = output_ids[:, self.prompt_length :].detach()
        changed = (sampled_tokens != prev_tokens).float().mean()
        self.last_step_debug["proposal_vs_decode_mismatch"] = float(changed.item())
        return loss_for_grad, output_ids, sampled_tokens, attr_losses.detach().cpu().numpy()

    def _sanitize_dist_logits(self, dist_logits):
        finite = torch.isfinite(dist_logits)
        valid_rows = finite.any(dim=-1)
        if valid_rows.all():
            return dist_logits
        fixed = dist_logits.clone()
        bad = ~valid_rows
        fixed[bad, 0] = 0.0
        fixed[bad, 1:] = -1e9
        return fixed

    def compute_bias_l2_pen(self, sampled_ids, steer_weight: float | None = None):
        sw = float(self.weight_val if steer_weight is None else steer_weight)
        with torch.no_grad():
            cur_embeds = self.embed_map(sampled_ids)
            t1 = torch.einsum("ve -> v", [self.embed_map.weight**2])[None, None, :]
            t2 = torch.einsum("bse, ve -> bsv", [cur_embeds, self.embed_map.weight])
            t3 = torch.einsum("bse -> bs", [cur_embeds**2]).unsqueeze(-1)
            bias = -1 * sw * (t1 - 2 * t2 + t3)
        return bias

    def _resolve_scale_mode_for_step(self, *, step_idx: int):
        del step_idx
        return self.use_scale_weights

