from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import torch
import torch.nn as nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from steering import compute_steered_loss

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
        loss_aggregation="mean",
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
        if self.loss_aggregation not in {"mean", "none", "prompt_mean"}:
            raise ValueError(
                f"Unsupported loss_aggregation={loss_aggregation!r}. Use: mean | none | prompt_mean"
            )
        self.bias_update_mode = str(bias_update_mode).strip().lower()
        if self.bias_update_mode not in {"sampled_l2", "direct_grad"}:
            raise ValueError(
                "Unsupported bias_update_mode="
                f"{bias_update_mode!r}. Use: sampled_l2 | direct_grad"
            )
        self._langevin_step = 0
        self.last_step_debug = {}
        self.prev_output_ids = None
        self.prev_pred_vecs = None
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
        self.prev_output_ids = None
        self.prev_pred_vecs = None
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

    @staticmethod
    def _ids_hash(ids_row: torch.Tensor) -> str:
        b = ids_row.detach().to("cpu").to(torch.int64).numpy().tobytes()
        return hashlib.sha1(b).hexdigest()[:12]

    def calc_grad(self, loss_for_grad, onehot):
        if not torch.is_tensor(loss_for_grad):
            raise TypeError(f"loss_for_grad must be a tensor, got {type(loss_for_grad)}")

        if loss_for_grad.ndim == 0:
            gx = torch.autograd.grad(loss_for_grad, onehot, allow_unused=True)[0]
            if gx is None:
                raise RuntimeError("Gradient is None for scalar loss_for_grad.")
            gx = gx.detach()[:, self.prompt_length :, :]
        elif loss_for_grad.ndim == 1:
            bsz = int(loss_for_grad.shape[0])
            if bsz != int(onehot.shape[0]):
                raise ValueError(
                    "loss_for_grad batch mismatch: "
                    f"loss_for_grad={tuple(loss_for_grad.shape)} onehot={tuple(onehot.shape)}"
                )
            rows = []
            for b in range(bsz):
                g_all = torch.autograd.grad(
                    loss_for_grad[b],
                    onehot,
                    retain_graph=(b < (bsz - 1)),
                    allow_unused=True,
                )[0]
                if g_all is None:
                    raise RuntimeError(f"Gradient is None for loss_for_grad[{b}].")
                rows.append(g_all[b : b + 1, self.prompt_length :, :])
            gx = torch.cat(rows, dim=0).detach()
        else:
            raise ValueError(
                f"loss_for_grad must be scalar or [B], got shape={tuple(loss_for_grad.shape)}"
            )
        gx = torch.nan_to_num(gx, nan=0.0, posinf=0.0, neginf=0.0)
        gx = gx.clamp(min=-1e3, max=1e3)
        self.last_step_debug["grad_norm"] = float(gx.float().norm().item())
        return gx

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
        gx = self.calc_grad(loss_for_grad, onehot)
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

    def step(self, x, **kwargs):
        """One Langevin step (DLP) via ``compute_steered_loss``."""
        del kwargs
        steer_w = float(self.weight_val)
        cur_step_idx = int(self._langevin_step)
        scale_mode = self._resolve_scale_mode_for_step(step_idx=cur_step_idx)
        self.model.set_biases(
            batch_size=x.size(0),
            seq_len=x.size(1) + int(self.prompt_length),
            prompt_length=self.prompt_length,
            attribute=None,
            device=self.device,
            disc_weight=self.weight_val,
            use_scale_weights=scale_mode,
            **self._set_bias_kwargs,
        )
        self.last_step_debug["scale_mode"] = str(scale_mode)
        self._langevin_step += 1
        self.last_step_debug["steer_weight"] = steer_w

        batch_size = x.size(0)
        bias_dim = self.model.get_input_embeddings().weight.shape[0]
        prompt_bias = torch.zeros(
            batch_size, self.prompt_length, bias_dim, device=x.device, dtype=x.dtype
        )
        x_full = torch.cat([prompt_bias, x], dim=1)
        self._trace_seq("step.x_full", x_full)
        disc = self.discriminator
        (
            loss,
            loss_for_grad,
            output_ids,
            onehot,
            logits,
            attr_losses,
            term_stats,
            pred_vecs,
            lm_reg,
            loss_per_sample,
        ) = compute_steered_loss(
            self.model,
            disc,
            self._inputs,
            x_full,
            weight=steer_w,
            prompt_length=self.prompt_length,
            loss_aggregation=self.loss_aggregation,
        )
        self._trace_seq("step.output_ids", output_ids)
        self.last_step_debug.update(term_stats)
        cur_seq = output_ids[:, self.prompt_length :].detach()
        if self.prev_output_ids is None:
            self.last_step_debug["step_seq_hamming"] = 0.0
            self.last_step_debug["step_seq_hamming_count"] = 0.0
            self.last_step_debug["step_seq_hash_prev_b0"] = "init"
            self.last_step_debug["step_seq_hash_cur_b0"] = self._ids_hash(cur_seq[0])
        else:
            # Effective prompt length can change after control insertions, so generated suffix
            # length may differ across consecutive steps.
            if cur_seq.shape == self.prev_output_ids.shape:
                step_diff = (cur_seq != self.prev_output_ids).float()
                self.last_step_debug["step_seq_hamming"] = float(step_diff.mean().item())
                self.last_step_debug["step_seq_hamming_count"] = float(step_diff.sum().item())
            else:
                t = min(int(cur_seq.shape[1]), int(self.prev_output_ids.shape[1]))
                if t > 0:
                    step_diff = (cur_seq[:, :t] != self.prev_output_ids[:, :t]).float()
                    self.last_step_debug["step_seq_hamming"] = float(step_diff.mean().item())
                    # Include suffix length mismatch as changed tokens.
                    extra = abs(int(cur_seq.shape[1]) - int(self.prev_output_ids.shape[1])) * int(cur_seq.shape[0])
                    self.last_step_debug["step_seq_hamming_count"] = float(step_diff.sum().item() + extra)
                else:
                    self.last_step_debug["step_seq_hamming"] = 0.0
                    self.last_step_debug["step_seq_hamming_count"] = float(
                        abs(int(cur_seq.shape[1]) - int(self.prev_output_ids.shape[1])) * int(cur_seq.shape[0])
                    )
            self.last_step_debug["step_seq_hash_prev_b0"] = self._ids_hash(self.prev_output_ids[0])
            self.last_step_debug["step_seq_hash_cur_b0"] = self._ids_hash(cur_seq[0])
        if self.prev_pred_vecs is None:
            self.last_step_debug["pred_delta_l2_mean"] = 0.0
        else:
            pred_delta = (pred_vecs - self.prev_pred_vecs).float().norm(dim=-1)
            self.last_step_debug["pred_delta_l2_mean"] = float(pred_delta.mean().item())
        self.prev_output_ids = cur_seq.clone()
        self.prev_pred_vecs = pred_vecs.clone()

        direct_gx = None
        if self.bias_update_mode == "direct_grad":
            # compute_p_lm_soft() also reads gradients from the same graph; take the
            # direct-gradient snapshot first and retain graph for the proposal read.
            direct_gx = _calc_grad_suffix(self, loss_for_grad, onehot, retain_graph=True)

        _loss_for_grad, output_ids, sampled_ids, attr_losses = self.compute_p_lm_soft(
            loss_for_grad, output_ids, onehot, logits, attr_losses
        )
        if self.bias_update_mode == "direct_grad":
            if direct_gx is None:
                raise RuntimeError("direct_grad selected but direct gradient snapshot is missing.")
            bias = torch.zeros_like(x)
            t_use = min(int(bias.shape[1]), int(direct_gx.shape[1]))
            v_use = min(int(bias.shape[2]), int(direct_gx.shape[2]))
            if t_use > 0 and v_use > 0:
                bias[:, :t_use, :v_use] = -direct_gx[:, :t_use, :v_use]
            self.last_step_debug["bias_update_mode"] = self.bias_update_mode
        else:
            bias = self.compute_bias_l2_pen(sampled_ids, steer_weight=steer_w)
            self.last_step_debug["bias_update_mode"] = self.bias_update_mode
        self.last_step_debug["bias_norm"] = float(bias.float().norm().item())
        prompt_attr_losses = getattr(self.discriminator, "last_prompt_attr_losses", None)
        if prompt_attr_losses is not None:
            prompt_attr_losses = prompt_attr_losses.detach().cpu().numpy()
        lm_reg_np = lm_reg.detach().cpu().numpy()
        loss_per_sample_np = loss_per_sample.detach().cpu().numpy()
        return bias, loss, output_ids, [prompt_attr_losses, {"lm_reg": lm_reg_np, "loss_per_sample": loss_per_sample_np}, attr_losses]

    def _resolve_scale_mode_for_step(self, *, step_idx: int):
        del step_idx
        return self.use_scale_weights

