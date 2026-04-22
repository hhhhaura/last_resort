"""Unified generator loaders and anticipation base adapter."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM

PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANTICIPATION_ROOT = PROJECT_ROOT.parent / "anticipation"
if str(ANTICIPATION_ROOT) not in sys.path:
    sys.path.insert(0, str(ANTICIPATION_ROOT))

from anticipation.config import MAX_DUR, MAX_NOTE, MAX_TIME
from anticipation.vocab import (
    AUTOREGRESS,
    CONTROL_OFFSET,
    DUR_OFFSET,
    NOTE_OFFSET,
    TIME_OFFSET,
    VOCAB_SIZE,
)


def _normalize_scale_weights_mode(x) -> Literal["off", "full", "partial"]:
    if x is True:
        return "full"
    if x is False:
        return "off"
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "1", "yes", "on", "full"):
            return "full"
        if s in ("false", "0", "no", "off"):
            return "off"
        if s == "partial":
            return "partial"
    raise ValueError(f"use_scale_weights must be one of false/true/full/partial (got {x!r})")


def _apply_dlp_bias_to_logits(
    next_logits: torch.Tensor,
    bias_row: torch.Tensor,
    delta: torch.Tensor,
    mode: Literal["off", "full", "partial"],
    slot: int,
) -> torch.Tensor:
    if mode == "off":
        return next_logits + delta
    if mode == "full":
        with torch.no_grad():
            ln = next_logits.detach().float().norm(dim=-1, p=2)
            bn = bias_row.detach().float().norm(dim=-1, p=2)
            ratio = torch.ones_like(ln)
            mask = bn > 1e-12
            ratio[mask] = ln[mask] / bn[mask].clamp_min(1e-12)
        scale = ratio.unsqueeze(-1).to(dtype=delta.dtype)
        return next_logits + scale * delta
    if slot in (0, 1):
        with torch.no_grad():
            ln = next_logits.detach().float().norm(dim=-1, p=2)
            bn = bias_row.detach().float().norm(dim=-1, p=2)
            ratio = torch.ones_like(ln)
            mask = bn > 1e-12
            ratio[mask] = ln[mask] / bn[mask].clamp_min(1e-12)
        ratio = torch.clamp(ratio, max=1.0)
        scale = ratio.unsqueeze(-1).to(dtype=delta.dtype)
        return next_logits + scale * delta
    return next_logits + delta


def _validate_ids_batch2d(
    ids: torch.Tensor,
    where: str,
    *,
    model_vocab: int,
    max_ctx: int,
) -> None:
    if ids.dtype not in (torch.int32, torch.int64):
        raise ValueError(f"{where}: input_ids must be int tensor, got dtype={ids.dtype}")
    if ids.numel() == 0:
        raise ValueError(f"{where}: empty input_ids")
    mn = int(ids.min().item())
    mx = int(ids.max().item())
    if mn < 0 or mx >= model_vocab:
        raise ValueError(f"{where}: token id out of range. min={mn} max={mx} model_vocab={model_vocab}")
    if ids.shape[1] > max_ctx:
        raise ValueError(
            f"{where}: sequence length exceeds model context. len={ids.shape[1]} max_ctx={max_ctx}"
        )


class AnticipationForDLP(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model
        self.seq_len = None
        self.vocab_size = int(VOCAB_SIZE)
        self._gen_temperature = 1.0
        self._gen_top_p = 0.98
        self._gen_do_sample = True
        self._scale_weights_mode: Literal["off", "full", "partial"] = "full"
        self.last_effective_prompt_len: int | None = None
        self._debug_trace_sequences: bool = False

    def to(self, *args, **kwargs):
        self.base_model = self.base_model.to(*args, **kwargs)
        return super().to(*args, **kwargs)

    def eval(self):
        self.base_model.eval()
        return super().eval()

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_biases(
        self,
        batch_size,
        seq_len,
        attribute,
        prompt_length,
        device,
        use_scale_weights,
        init_noise_rate=0.5,
        disc_weight=1.0,
        **kwargs,
    ):
        del batch_size, attribute, prompt_length, device, init_noise_rate, disc_weight
        self._scale_weights_mode = _normalize_scale_weights_mode(use_scale_weights)
        self.seq_len = int(seq_len)
        self._gen_temperature = float(kwargs.get("temperature", 1.0))
        self._gen_top_p = float(kwargs.get("top_p", 0.98))
        self._gen_do_sample = bool(kwargs.get("do_sample", True))
        self._debug_trace_sequences = bool(kwargs.get("debug_trace_sequences", False))

    def _trace_seq(self, name: str, ids: torch.Tensor) -> None:
        if not self._debug_trace_sequences:
            return
        flat = ids.detach().cpu().reshape(-1).tolist()
        print(
            f"[gen] trace {name}: shape={tuple(ids.shape)} "
            f"prefix10={flat[:10]} suffix10={flat[-10:] if flat else []}"
        )

    def _add_ar_prefix(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        ok = input_ids[0].shape[0] == 0 or input_ids[0][0] != int(AUTOREGRESS)
        if not ok:
            raise ValueError("Mixed AR prefix in batch: some rows start with AUTOREGRESS and others do not.")
        ar_col = torch.full(
            (input_ids.shape[0], 1),
            int(AUTOREGRESS),
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        input_ids = torch.cat([ar_col, input_ids], dim=1)
        if attention_mask is not None:
            one = torch.ones((attention_mask.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)
            attention_mask = torch.cat([one, attention_mask], dim=1)
        return input_ids, attention_mask

    def _resolve_seq_len(self) -> int:
        if self.seq_len is not None:
            return int(self.seq_len)
        return int(
            getattr(
                self.base_model.config,
                "n_positions",
                getattr(self.base_model.config, "max_position_embeddings", 1024),
            )
        )

    @staticmethod
    def _top_p_filtering(logits: torch.Tensor, top_p: float) -> torch.Tensor:
        if top_p >= 1.0:
            return logits
        sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_indices_to_remove = cumulative_probs > top_p
        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
        sorted_indices_to_remove[..., 0] = False
        indices_to_remove = torch.zeros_like(sorted_indices_to_remove, dtype=torch.bool)
        indices_to_remove.scatter_(dim=-1, index=sorted_indices, src=sorted_indices_to_remove)
        filtered = logits.masked_fill(indices_to_remove, float("-inf"))
        all_inf = ~torch.isfinite(filtered).any(dim=-1, keepdim=True)
        return torch.where(all_inf, logits, filtered)

    @staticmethod
    def _apply_event_slot_mask(
        next_logits: torch.Tensor,
        slot_idx: int,
        *,
        prev_time_token: torch.Tensor | None = None,
    ) -> torch.Tensor:
        masked = next_logits.clone()
        masked[:, int(CONTROL_OFFSET) :] = float("-inf")
        if slot_idx == 0:
            masked[:, int(DUR_OFFSET) : int(DUR_OFFSET + MAX_DUR)] = float("-inf")
            masked[:, int(NOTE_OFFSET) : int(NOTE_OFFSET + MAX_NOTE)] = float("-inf")
            if prev_time_token is not None:
                t_lo = int(TIME_OFFSET)
                t_hi = t_lo + int(MAX_TIME)
                bsz = masked.shape[0]
                for b in range(bsz):
                    pt = int(prev_time_token[b].item())
                    if t_lo < pt < t_hi:
                        masked[b, t_lo:pt] = float("-inf")
        elif slot_idx == 1:
            masked[:, int(TIME_OFFSET) : int(TIME_OFFSET + MAX_TIME)] = float("-inf")
            masked[:, int(NOTE_OFFSET) : int(NOTE_OFFSET + MAX_NOTE)] = float("-inf")
        else:
            masked[:, int(TIME_OFFSET) : int(TIME_OFFSET + MAX_TIME)] = float("-inf")
            masked[:, int(DUR_OFFSET) : int(DUR_OFFSET + MAX_DUR)] = float("-inf")
        bad = ~torch.isfinite(masked).any(dim=-1, keepdim=True)
        safe_next = next_logits.clone()
        safe_next[:, int(CONTROL_OFFSET) :] = float("-inf")
        return torch.where(bad, safe_next, masked)

    def forward_with_biases(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        use_full_prompt=False,
        biases=None,
        bias_rep_space="logit",
        weight=1.0,
        **kwargs,
    ):
        del labels, use_full_prompt, bias_rep_space, kwargs
        if input_ids is None:
            raise ValueError("AnticipationForDLP requires input_ids.")
        seq_len = self._resolve_seq_len()
        weight_f = float(weight)

        model_vocab = int(self.base_model.get_input_embeddings().weight.size(0))
        max_ctx = int(
            getattr(
                self.base_model.config,
                "n_positions",
                getattr(self.base_model.config, "max_position_embeddings", 1024),
            )
        )

        input_ids, attention_mask = self._add_ar_prefix(input_ids, attention_mask)
        _validate_ids_batch2d(input_ids, "initial", model_vocab=model_vocab, max_ctx=max_ctx)
        bsz = input_ids.shape[0]
        cur_ids = input_ids.clone()
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        cur_mask = attention_mask
        self._trace_seq("forward.input_ids", cur_ids)

        effective_prompt_len_with_ar = int(cur_ids.shape[1])
        effective_prompt_len = effective_prompt_len_with_ar - 1
        target_content_len = min(int(seq_len), max_ctx - 1)
        target_total_with_ar = target_content_len + 1
        if effective_prompt_len_with_ar > target_total_with_ar:
            raise ValueError(
                "prompt exceeds configured decode target: "
                f"prompt_with_ar={effective_prompt_len_with_ar} target_with_ar={target_total_with_ar}"
            )
        gen_len = max(int(target_total_with_ar - effective_prompt_len_with_ar), 0)
        _validate_ids_batch2d(cur_ids, "prompt_ready", model_vocab=model_vocab, max_ctx=max_ctx)

        step_logits_raw: list[torch.Tensor] = []
        with torch.no_grad():
            past_key_values = None
            dec_input = cur_ids

            for step_idx in range(gen_len):
                if past_key_values is None:
                    dec_input = cur_ids
                slot = step_idx % 3
                if past_key_values is None:
                    _validate_ids_batch2d(
                        dec_input, f"decode_step_{step_idx}:dec_input", model_vocab=model_vocab, max_ctx=max_ctx
                    )
                _validate_ids_batch2d(cur_ids, f"decode_step_{step_idx}:cur_ids", model_vocab=model_vocab, max_ctx=max_ctx)
                out = self.base_model(
                    input_ids=dec_input,
                    attention_mask=cur_mask,
                    return_dict=True,
                    use_cache=True,
                    past_key_values=past_key_values,
                )
                past_key_values = out.past_key_values
                next_logits = out.logits[:, -1, : self.vocab_size]
                if biases is not None:
                    pos = int(cur_ids.shape[1]) - 1
                    if 0 <= pos < biases.shape[1]:
                        bias_row = biases[:, pos, : self.vocab_size].to(next_logits.dtype)
                        delta = weight_f * bias_row
                        next_logits = _apply_dlp_bias_to_logits(
                            next_logits,
                            bias_row,
                            delta,
                            self._scale_weights_mode,
                            slot,
                        )
                step_logits_raw.append(next_logits)
                prev_t: torch.Tensor | None = None
                if slot == 0 and cur_ids.shape[1] >= 3:
                    prev_t = cur_ids[:, -3].long()
                z = self._apply_event_slot_mask(next_logits, slot, prev_time_token=prev_t)
                z = z / max(self._gen_temperature, 1e-6)
                z = self._top_p_filtering(z, self._gen_top_p)
                probs = torch.softmax(z.float(), dim=-1).to(dtype=z.dtype)
                if self._gen_do_sample:
                    next_token = torch.multinomial(probs, 1)
                else:
                    next_token = torch.argmax(probs, dim=-1, keepdim=True)
                cur_ids = torch.cat([cur_ids, next_token.to(cur_ids.dtype)], dim=1)
                _validate_ids_batch2d(
                    cur_ids,
                    f"decode_step_{step_idx}:cur_ids_after_cat",
                    model_vocab=model_vocab,
                    max_ctx=max_ctx,
                )
                cur_mask = torch.cat(
                    [cur_mask, torch.ones((cur_mask.size(0), 1), dtype=cur_mask.dtype, device=cur_mask.device)],
                    dim=1,
                )
                dec_input = next_token.to(cur_ids.dtype)

        ret_output_ids = cur_ids[:, 1:]
        self._trace_seq("forward.output_ids", ret_output_ids)
        if step_logits_raw:
            gen_logits = torch.stack(step_logits_raw, dim=1)
        else:
            gen_logits = torch.zeros(
                bsz,
                0,
                self.vocab_size,
                device=cur_ids.device,
                dtype=self.base_model.get_input_embeddings().weight.dtype,
            )
        prompt_logits = torch.zeros(
            bsz,
            effective_prompt_len,
            self.vocab_size,
            device=gen_logits.device,
            dtype=gen_logits.dtype,
        )
        gpt_logit = torch.cat([prompt_logits, gen_logits], dim=1)
        self.last_effective_prompt_len = int(effective_prompt_len)
        return ret_output_ids, gpt_logit


def load_anticipation_base_model(model_name_or_path: str, trust_remote_code: bool = True):
    base = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=torch.bfloat16,
        trust_remote_code=trust_remote_code,
    )
    return AnticipationForDLP(base)


def load_base_model(**kwargs):
    model_id = kwargs.get("model_name_or_path") or kwargs.get("pretrained_model_name_or_path")
    if model_id is None:
        raise ValueError(
            "Provide base_model_args.model_name_or_path (or pretrained_model_name_or_path) for anticipation model."
        )
    trust_remote_code = bool(kwargs.get("trust_remote_code", True))
    return load_anticipation_base_model(model_name_or_path=model_id, trust_remote_code=trust_remote_code)
