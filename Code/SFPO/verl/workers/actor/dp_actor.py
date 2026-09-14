# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import itertools
import os
import time
import weakref
from typing import Iterable, Tuple

# Power-throttle chunk size for full-model diagnostic norm waves (see
# fsdp_workers._SFPO_FOREACH_CHUNK); values unchanged, scheduling only.
_GXPO_NORM_CHUNK = max(1, int(os.environ.get('GXPO_NORM_CHUNK', '32')))

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.workers.actor.gxpo_state import (GXPOState, RetentionKind, adamw_direction,
                                          adamw_direction_from_step,
                                          compute_gxpo_adamw_direction_retention_scale,
                                          compute_gxpo_retention_scale,
                                          compute_gxpo_update_retention_scale)
from verl.workers.actor.optimizer_transaction import snapshot_optimizer_state
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
from verl.workers.actor.kd_loss import KD_TOPK_CHUNK_TOKENS, compute_forward_kl_topk_chunked, compute_reverse_kl_topk_chunked
from verl.workers.actor.opd2_signal import OPD2_CHUNK_TOKENS, gather_from_logits, topk_from_logits
from verl.workers.actor.sled_delta import (SLED_CHUNK_TOKENS, sled_contrast_logits,
                                           sled_truncated_frozen)
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl.utils.seqlen_balancing import rearrange_micro_batches, get_reverse_idx
import verl.utils.torch_functional as verl_F

try:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
except ImportError:
    # The base vLLM image may omit FlashAttention while Transformers SDPA is
    # available. Keep the actor importable for that supported path; the
    # fallback is only used when remove-padding is explicitly enabled.
    from einops import rearrange

    def index_first_axis(hidden_states, indices):
        return hidden_states[indices]

    def unpad_input(hidden_states, attention_mask):
        batch_size, seqlen = attention_mask.shape
        indices = torch.nonzero(attention_mask.reshape(-1), as_tuple=False).flatten()
        hidden_states = hidden_states.reshape(batch_size * seqlen, *hidden_states.shape[2:])[indices]
        lengths = attention_mask.sum(dim=-1, dtype=torch.int32)
        cu_seqlens = torch.zeros(batch_size + 1, device=attention_mask.device, dtype=torch.int32)
        cu_seqlens[1:] = torch.cumsum(lengths, dim=0)
        return hidden_states, indices, cu_seqlens, int(lengths.max().item()) if lengths.numel() else 0

    def pad_input(hidden_states, indices, batch, seqlen):
        output = torch.zeros(
            (batch * seqlen, *hidden_states.shape[1:]),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        output.index_copy_(0, indices, hidden_states)
        return output.view(batch, seqlen, *hidden_states.shape[1:])

__all__ = ['DataParallelPPOActor']


def _merge_metrics(dst: dict, src: dict):
    """Merge a mini-batch metrics dict into the accumulated dict (lists extend, scalars overwrite)."""
    for key, val in src.items():
        if isinstance(val, list):
            dst.setdefault(key, []).extend(val)
        else:
            dst[key] = val


class DataParallelPPOActor(BasePPOActor):

    def __init__(
        self,
        config,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.use_remove_padding = self.config.get('use_remove_padding', False)
        print(f'Actor use_remove_padding={self.use_remove_padding}')
        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.compute_entropy_from_logits = (
            torch.compile(verl_F.entropy_from_logits, dynamic=True)
            if self.config.get('use_torch_compile', True)  #  use torch compile by default
            else verl_F.entropy_from_logits)

        # cumulative backward-pass counter (1 BP = one full gradient over a mini-batch)
        self.cumulative_bp = 0

        # Keep algorithmic/full gradient evaluations separate from actual
        # autograd calls. The latter includes gradient-accumulation microbatches.
        self.raw_backward_calls = 0
        # GXPO: shutoff-gate state + lazily allocated per-parameter buffers
        self.gxpo_state = None
        self._gxpo_bufs = None
        self._gxpo_retention_cache = None
        self._gxpo_unsupported_optimizer_warned = False
        self._gxpo_precision_validated = False
        self._gxpo_strict_precision = bool(self.config.get('gxpo_strict_precision', True))
        self._gxpo_fsdp_invariant_threshold = bool(
            self.config.get('gxpo_fsdp_invariant_threshold', True))
        # Optional duty-cycle guard for the high-power GXPO actor path. This
        # is scheduling only: it never changes the loss, gradients, clipping,
        # optimizer, or GXPO reposition math.
        raw_duty_cycle = self.config.get(
            'gxpo_actor_duty_cycle', os.environ.get('GXPO_ACTOR_DUTY_CYCLE', '0'))
        self._gxpo_actor_duty_cycle = float(raw_duty_cycle or 0.0)
        if not 0.0 <= self._gxpo_actor_duty_cycle <= 1.0:
            raise ValueError('gxpo_actor_duty_cycle must be in [0, 1]')
        self._gxpo_power_guard_active_s = 0.0
        self._gxpo_power_guard_sleep_s = 0.0
        # Cache of the constant-per-outer-step reposition direction sum-of-squares
        # consumed by _optimizer_state_metrics; keyed by a weakref to the pairs list.
        self._reposition_dir_cache = None
        if self.config.get('use_gxpo', False) and actor_optimizer is not None:
            self.gxpo_optimizer_state_mode = str(
                self.config.get('gxpo_optimizer_state_mode', 'transactional')).lower()
            if self.gxpo_optimizer_state_mode not in (
                    'transactional', 'transactional_fast_state'):
                raise ValueError(
                    'gxpo_optimizer_state_mode must be transactional or '
                    'transactional_fast_state (the moment-polluting legacy '
                    'mode was removed), '
                    f'got {self.gxpo_optimizer_state_mode!r}')
            # Mirrors the SFT arm's contraction guard. theta_tilde = theta0 +
            # alpha*scale*(theta2-theta0): alpha*scale below 1 lands SHORT of theta2, so the
            # 3-pass update contracts instead of extrapolating. 0.0 = warn only; 1.0 floors
            # the per-coordinate multiplier so the reposition can never land short.
            min_eff = float(self.config.get('gxpo_min_effective_multiplier', 0.0))
            if min_eff < 0.0:
                raise ValueError('gxpo_min_effective_multiplier must be non-negative, '
                                 f'got {min_eff}')
            self.gxpo_min_effective_multiplier = min_eff
            self._gxpo_contraction_warned = False
            self.gxpo_state = GXPOState(
                K=self.config.get('gxpo_k', 5),
                alpha=self.config.get('gxpo_alpha', 0.5),
                delta=self.config.get('gxpo_delta', 1e-8),
                tau=self.config.get('gxpo_tau', 3.0),
                omega=self.config.get('gxpo_omega', 0.1),
                zscore_w=self.config.get('gxpo_zscore_w', 30),
                shutoff_mode=self.config.get('gxpo_shutoff_mode', 'trajectory_aware'),
                fallback_mode=self.config.get('gxpo_fallback_mode', 'permanent'),
                fallback_window=self.config.get('gxpo_fallback_window', 10),
                trigger_patience=self.config.get('gxpo_trigger_patience', 1),
                trigger_robust=self.config.get('gxpo_trigger_robust', False),
                min_post_warmup_obs=int(self.config.get('gxpo_trigger_min_obs', 0)),
                max_active_steps=int(self.config.get('gxpo_max_active_steps', 0)),
                abs_threshold=float(self.config.get('gxpo_trigger_abs_threshold', 0.0)),
                sustain_window=int(self.config.get("gxpo_trigger_sustain_w", 10)),
                relative_threshold=float(self.config.get('gxpo_relative_threshold', 0.0)),
            )
            self._gxpo_diag_freq = int(self.config.get('gxpo_diag_freq', 10))

    def _forward_micro_batch(self, micro_batch, temperature, need_entropy=True) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len); ``None`` when ``need_entropy=False``
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 3, seqlen) -> (3, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                           attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."),
                                                          indices).transpose(0, 1).unsqueeze(
                                                              1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, \
                                                                                                position_ids_rmpad, \
                                                                                                sp_size=self.ulysses_sequence_parallel_size)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(input_ids_rmpad_rolled, None,
                                                                                self.ulysses_sequence_parallel_size)

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.actor_module(input_ids=input_ids_rmpad,
                                           attention_mask=None,
                                           position_ids=position_ids_rmpad,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)

                logits_rmpad.div_(temperature)

                # compute entropy (skipped entirely when the caller discards it)
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad) if need_entropy else None  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    if need_entropy:
                        entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                                gather_dim=0,
                                                                unpad_dim=0,
                                                                padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                if need_entropy:
                    full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                             indices=indices,
                                             batch=batch_size,
                                             seqlen=seqlen)
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
                else:
                    entropy = None
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                output = self.actor_module(input_ids=input_ids,
                                           attention_mask=attention_mask,
                                           position_ids=position_ids,
                                           **multi_modal_inputs,
                                           use_cache=False)  # prevent model thinks we are generating
                logits = output.logits
                logits.div_(temperature)
                logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab_size)
                log_probs = logprobs_from_logits(logits, micro_batch['responses'])
                entropy = verl_F.entropy_from_logits(logits) if need_entropy else None  # (bsz, response_length)

            return entropy, log_probs

    def _forward_kd_micro_batch(self, micro_batch, temperature) -> torch.Tensor:
        """Dense response-logits forward for offline KD (no rmpad).

        Returns bf16/fp32 logits of shape (bsz, response_length, vocab). The
        dense path is used deliberately: the offline teacher top-K cache is
        stored dense per response position, while rmpad row order is
        data-dependent and cannot be pre-aligned. Logits stay in autocast
        dtype here; ``kd_loss`` upcasts in token chunks (see kd_loss.py), so
        no full [tokens, vocab] FP32 copy ever exists.
        """
        response_length = micro_batch['responses'].size(-1)
        multi_modal_inputs = {}
        if 'multi_modal_inputs' in micro_batch:
            for key in micro_batch['multi_modal_inputs'][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch['multi_modal_inputs']],
                                                    dim=0)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            input_ids = micro_batch['input_ids']
            attention_mask = micro_batch['attention_mask']
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)
            output = self.actor_module(input_ids=input_ids,
                                       attention_mask=attention_mask,
                                       position_ids=position_ids,
                                       **multi_modal_inputs,
                                       use_cache=False)
            logits = output.logits
            logits.div_(temperature)
            logits = logits[:, -response_length - 1:-1, :]  # (bsz, response_length, vocab)
            # Clone out of the autocast graph inputs; the caller holds this
            # only until the chunked KL below consumes it, then deletes it.
            return logits

    def _pool_kd_loss(self, per_tok: torch.Tensor, b_sel: torch.Tensor, j_sel: torch.Tensor,
                      response_mask: torch.Tensor) -> torch.Tensor:
        """Scatter a flat [N] per-token KD loss back to [B, R] and pool with
        ``loss_agg_mode``.

        Root-caused 2026-09-14 (scratchpad/opd2_audit*, same mechanism as the
        OPD^2 collapse): ``distillation_losses.mean()`` is a flat mean over
        every response token in the micro-batch, so a long response gets
        proportionally more gradient than a short one -- the same
        length-reward-hacking path that blew up entropy in
        ``qwen25-1p5b_onpolicy_kd_gxpo_k3_a0.1`` (entropy 0.84->4.2 by step 4,
        val collapsed to ~0.03). ``loss_agg_mode='seq-mean-token-mean'`` (this
        launcher's default going forward) means every response, long or
        short, contributes one equal vote.
        Default 'token-mean' reproduces the old flat mean exactly.
        """
        grid = torch.zeros_like(response_mask, dtype=per_tok.dtype)
        grid[b_sel, j_sel] = per_tok
        mode = self.config.get('loss_agg_mode', 'token-mean')
        return core_algos.agg_loss(grid, response_mask, mode)

    def _forward_kd_flat(self, micro_batch, temperature, response_mask, has_multi_modal_inputs):
        """Response-filtered flat student logits with aligned teacher rows.

        The default dense path materializes full-sequence [B, S, V] logits
        including every pad token, then masks. With remove-padding enabled
        this runs the forward on unpadded tokens and returns only real
        response rows ([N, V], N excludes all padding), gathering the teacher
        top-K rows in the same rmpad order -- so the [N, V] peak and the fp32
        chunk work skip prompt tails and response tails entirely.

        Falls back to the dense path whenever rmpad is inapplicable (no
        remove-padding, multimodal, sequence-parallel, 3D position ids, or
        non-[B, R, K] teacher tensors). The first call cross-checks rmpad
        against dense and pins the safe path loudly instead of risking silent
        corruption. Disable via +actor_rollout_ref.actor.kd_rmpad=False.
        Returns (flat_logits, t_logps, t_ids, b_sel, j_sel) or None when there
        are no real response tokens. ``b_sel``/``j_sel`` are each flat token's
        row/col in the [B, R] response grid, for scattering the per-token KD
        loss back for ``loss_agg_mode`` pooling.
        """
        use_kd_tensors = ('teacher_topk_log_probs' in micro_batch and 'teacher_topk_ids' in micro_batch)
        t_logps_raw = micro_batch['teacher_topk_log_probs'] if use_kd_tensors else None
        t_ids_raw = micro_batch['teacher_topk_ids'] if use_kd_tensors else None
        teacher_ok = (torch.is_tensor(t_logps_raw) and torch.is_tensor(t_ids_raw)
                      and t_logps_raw.dim() == 3 and t_ids_raw.dim() == 3
                      and t_logps_raw.shape == t_ids_raw.shape)
        want_rmpad = (self.use_remove_padding
                      and not has_multi_modal_inputs
                      and self.ulysses_sequence_parallel_size == 1
                      and bool(self.config.get('kd_rmpad', True))
                      and teacher_ok)
        if want_rmpad and not getattr(self, '_kd_rmpad_checked', False):
            return self._forward_kd_flat_checked(micro_batch, temperature, response_mask,
                                                 t_logps_raw, t_ids_raw)
        if want_rmpad and getattr(self, '_kd_rmpad_ok', False):
            return self._forward_kd_flat_rmpad(micro_batch, temperature, response_mask,
                                               t_logps_raw, t_ids_raw)
        return self._forward_kd_flat_dense(micro_batch, temperature, response_mask)

    def _forward_kd_flat_dense(self, micro_batch, temperature, response_mask):
        """Legacy dense path: full-sequence logits, then response mask.

        Also returns (b_sel, j_sel) -- the [B, R] row/col each flat token came
        from -- so the caller can scatter distillation_losses back into a
        [B, R] grid and pool it with ``loss_agg_mode`` instead of a flat
        token-mean (see the ``kd_loss`` comment at its call sites).
        """
        kd_logits = self._forward_kd_micro_batch(micro_batch=micro_batch, temperature=temperature)
        R = response_mask.size(-1)
        flat_mask = response_mask.bool().reshape(-1)
        if not bool(flat_mask.any().item()):
            del kd_logits
            return None
        flat_logits = kd_logits.reshape(-1, kd_logits.size(-1))[flat_mask]
        t_logps = micro_batch['teacher_topk_log_probs']
        t_ids = micro_batch['teacher_topk_ids']
        if torch.is_tensor(t_logps) and t_logps.dim() == 3:
            t_logps = t_logps.reshape(-1, t_logps.size(-1))[flat_mask]
        if torch.is_tensor(t_ids) and t_ids.dim() == 3:
            t_ids = t_ids.reshape(-1, t_ids.size(-1))[flat_mask]
        flat_idx = flat_mask.nonzero(as_tuple=False).squeeze(-1)
        b_sel = flat_idx // R
        j_sel = flat_idx % R
        del kd_logits
        return flat_logits, t_logps, t_ids, b_sel, j_sel

    def _forward_kd_flat_rmpad(self, micro_batch, temperature, response_mask, t_logps_raw, t_ids_raw):
        """Rmpad KD forward. Returns (flat_logits, t_logps, t_ids, b_sel, j_sel) or None."""
        input_ids = micro_batch['input_ids']
        attention_mask = micro_batch['attention_mask']
        position_ids = micro_batch['position_ids']
        responses = micro_batch['responses']
        B, S = input_ids.shape
        R = responses.size(1)
        if position_ids.dim() == 3:
            return None
        input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)
        input_ids_rmpad = input_ids_rmpad.transpose(0, 1)
        position_ids_rmpad = index_first_axis(
            rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = self.actor_module(input_ids=input_ids_rmpad,
                                       attention_mask=None,
                                       position_ids=position_ids_rmpad,
                                       use_cache=False)
            logits_rmpad = output.logits.squeeze(0)
            logits_rmpad.div_(temperature)
        # Span convention must match the dense path: labels are rolled by -1, so
        # the logit at position S-R-1+j predicts response token j, i.e. the
        # response span is [S-R-1, S-1) -- NOT the last R positions. (An early
        # version used [S-R, S); the one-time check caught it: row count
        # matched but maxdiff was ~41.)
        resp_start = S - R - 1
        flat = indices  # [nnz] flat positions into [B, S]
        bb = flat // S
        ss = flat % S
        in_span = (ss >= resp_start) & (ss < (S - 1))
        jj = (ss - resp_start).clamp(0, R - 1)
        real = in_span & response_mask.bool()[bb, jj]
        if not bool(real.any().item()):
            del logits_rmpad
            return None
        sel = torch.nonzero(real, as_tuple=False).squeeze(-1)
        flat_logits = logits_rmpad[sel]
        b_sel = bb[sel]
        j_sel = (ss[sel] - resp_start).clamp(0, R - 1).long()
        t_logps = t_logps_raw[b_sel, j_sel]
        t_ids = t_ids_raw[b_sel, j_sel]
        del logits_rmpad
        return flat_logits, t_logps, t_ids, b_sel, j_sel

    def _forward_kd_flat_checked(self, micro_batch, temperature, response_mask, t_logps_raw, t_ids_raw):
        """One-time rmpad-vs-dense cross-check; pins the safe path loudly."""
        self._kd_rmpad_checked = True
        rmpad_out = self._forward_kd_flat_rmpad(micro_batch, temperature, response_mask,
                                                t_logps_raw, t_ids_raw)
        if rmpad_out is None:
            dense_out = self._forward_kd_flat_dense(micro_batch, temperature, response_mask)
            agree = dense_out is None
            self._kd_rmpad_ok = agree
            print(f'[KD-RMPAD] cross-check vs dense: {"PASS" if agree else "FAIL"} '
                  f'(both-empty={agree}); using dense KD forward henceforth.', flush=True)
            return dense_out
        flat_logits, _, _, b_sel, j_sel = rmpad_out
        N = flat_logits.size(0)
        R = micro_batch['responses'].size(1)
        expect = int(response_mask.bool().sum().item())
        detail, ok = f'count-mismatch rmpad={N} mask={expect}', False
        if N == expect:
            kd_logits = self._forward_kd_micro_batch(micro_batch=micro_batch, temperature=temperature)
            grid = kd_logits.reshape(-1, kd_logits.size(-1))
            ref_idx = b_sel * R + j_sel
            maxdiff = 0.0
            for s in range(0, N, 4096):
                e = min(s + 4096, N)
                d = (flat_logits[s:e].float() - grid[ref_idx[s:e]].float()).abs().max().item()
                maxdiff = max(maxdiff, d)
            del kd_logits, grid, ref_idx
            ok = maxdiff < 2e-2
            detail = f'N={N} maxdiff={maxdiff:.2e}'
        self._kd_rmpad_ok = bool(ok)
        print(f'[KD-RMPAD] cross-check vs dense: {"PASS" if ok else "FAIL"} ({detail}); '
              f'using {"rmpad" if ok else "dense"} KD forward henceforth.', flush=True)
        if ok:
            return rmpad_out
        return self._forward_kd_flat_dense(micro_batch, temperature, response_mask)

    def _clip_grads(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        return grad_norm

    def _ensure_fsdp_gradient_sync(self):
        """Ensure FSDP leaves a local reduced gradient for the optimizer.

        FSDP's ``no_sync()`` intentionally leaves full unsharded gradients.
        The hybrid-engine rollout path does not use gradient accumulation via
        ``no_sync()``, but vLLM state/weight transitions can leave this private
        flag disabled on Torch 2.9. Re-enable it at the update boundary so
        sharded AdamW never sees a full gradient for a local flat parameter.
        """
        if not isinstance(self.actor_module, FSDP):
            return
        for module in self.actor_module.modules():
            if isinstance(module, FSDP) and hasattr(module, '_sync_gradients'):
                module._sync_gradients = True

    def _reshard_full_fsdp_grads(self):
        """Reduce-scatter any full flat gradients left by the FSDP runtime.

        Torch 2.9 can leave an unsharded ``FlatParameter.grad`` after the
        hybrid vLLM state transition even with the sync flag enabled. The
        private reducer is the same implementation used by FSDP's backward
        hook and handles flat-parameter padding and rank groups correctly.
        """
        if not isinstance(self.actor_module, FSDP):
            return
        # FSDP queues reduce-scatter on its post-backward CUDA stream. The
        # custom remove-padding/FlashAttention path can return from backward
        # before that stream has finalized ``.grad``; wait before inspecting
        # shapes or clipping.
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        modules = FSDP.fsdp_modules(self.actor_module)
        candidates = []
        local_flags = []
        for module in modules:
            handle = getattr(module, '_handle', None)
            flat_param = getattr(handle, 'flat_param', None)
            is_full = bool(
                handle is not None and flat_param is not None and
                handle.uses_sharded_strategy and flat_param.grad is not None and
                flat_param.grad.numel() != flat_param.numel()
            )
            candidates.append((module, handle, flat_param))
            local_flags.append(int(is_full))
        if torch.distributed.is_initialized() and candidates:
            flags = torch.tensor(local_flags, dtype=torch.int32,
                                 device=torch.cuda.current_device())
            torch.distributed.all_reduce(flags, op=torch.distributed.ReduceOp.SUM)
            world = torch.distributed.get_world_size()
            if bool(((flags != 0) & (flags != world)).any().item()):
                raise RuntimeError(
                    'FSDP full-gradient mismatch is present on only a subset of ranks'
                )
            needs = [int(v) == world for v in flags.tolist()]
        else:
            needs = [bool(v) for v in local_flags]
        if not any(needs):
            return
        from torch.distributed.fsdp import _runtime_utils as fsdp_runtime
        from torch.distributed.fsdp._common_utils import TrainingState
        for need, (module, handle, flat_param) in zip(needs, candidates):
            if not need:
                continue
            if flat_param.grad is None:
                raise RuntimeError('FSDP full-gradient repair requires a gradient on every rank')
            # The normal hook already finalized the FSDP state by the time this
            # repair runs. Temporarily enter its backward state so the official
            # reducer/cast helpers accept the manually recovered full gradient.
            old_training_state = module.training_state
            had_post_backward_called = hasattr(flat_param, '_post_backward_called')
            old_post_backward_called = getattr(flat_param, '_post_backward_called', False)
            module.training_state = TrainingState.FORWARD_BACKWARD
            flat_param._post_backward_called = True
            try:
                fsdp_runtime._reduce_grad(module, handle)
                handle.prepare_gradient_for_optim()
            finally:
                module.training_state = old_training_state
                if had_post_backward_called:
                    flat_param._post_backward_called = old_post_backward_called
                else:
                    delattr(flat_param, '_post_backward_called')

    def _cast_optimizer_grads_to_param_dtype(self):
        """Normalize mixed-precision FSDP grads after clipping/reduction.

        FSDP's ``clip_grad_norm_`` must run first: before it completes its
        reduce-scatter, a flat parameter can temporarily expose the full
        unsharded gradient while the parameter itself is a local shard.
        """
        for param in self.actor_module.parameters():
            grad = param.grad
            if grad is None or grad.dtype == param.dtype:
                continue
            if grad.shape != param.shape:
                raise RuntimeError(
                    f"FSDP gradient shape {tuple(grad.shape)} does not match "
                    f"parameter shape {tuple(param.shape)} after clipping; "
                    "gradient synchronization may be disabled"
                )
            # ``.data`` is intentional here. PyTorch's gradient assignment
            # checks dtypes even though AdamW expects an FP32 master parameter
            # with a reduced BF16 gradient.
            param.grad.data = grad.to(dtype=param.dtype)

    def _optimizer_step(self):
        grad_norm = self._clip_grads()
        self._cast_optimizer_grads_to_param_dtype()
        self.actor_optimizer.step()
        return grad_norm

    def _gxpo_power_guard(self, phase_start: float):
        """Wait between high-power CUDA phases to enforce a duty cycle.

        ``loss.backward`` and optimizer kernels are asynchronous from Python's
        point of view. Synchronizing before measuring is intentional: without
        it, the guard would sleep while CUDA was still executing and would not
        flatten the board-power transient. A value of 0 disables the guard;
        1.0 keeps the same synchronization/measurement path but inserts no
        sleep. The guard is enabled only for an actor constructed with GXPO.
        """
        duty = self._gxpo_actor_duty_cycle
        if self.gxpo_state is None or duty <= 0.0 or not torch.cuda.is_available():
            return

        torch.cuda.synchronize()
        active_s = max(time.perf_counter() - phase_start, 0.0)
        self._gxpo_power_guard_active_s += active_s
        if duty >= 1.0:
            return

        sleep_s = active_s * (1.0 / duty - 1.0)
        if sleep_s > 0.0:
            time.sleep(sleep_s)
            self._gxpo_power_guard_sleep_s += sleep_s

    def _optimizer_state_metrics(self, reposition_pairs=None):
        """Return scalar AdamW-state diagnostics without changing optimizer state.

        ``reposition_pairs`` contains ``(post_reposition, pre_reposition)``
        tensors when the caller is measuring a SFPO/GXPO jump.  The tensors are
        already available in those update paths; this helper only reduces scalar
        sums and never copies a full model to the driver.
        """
        if self.actor_optimizer is None:
            return {}
        params = [p for p in self.actor_module.parameters() if p.requires_grad]
        if reposition_pairs is not None and len(reposition_pairs) != len(params):
            reposition_pairs = None

        # The (post_reposition - pre_reposition) direction is constant across every
        # mini-batch of one SFPO slow phase (the caller builds one pairs list per
        # sfpo_update_actor and reuses it), so its sum-of-squares (stats[4]) is
        # computed once per pairs list and cached. The cache holds a weakref only,
        # so the caller's weight lists can still be freed. NOTE: stats[5] multiplies
        # the direction by the CURRENT exp_avg, which the optimizer updates on every
        # step, so it must be recomputed per call to stay bit-identical.
        cached_dir_sq = None
        if reposition_pairs is not None and self._reposition_dir_cache is not None:
            pairs_ref, dir_sq = self._reposition_dir_cache
            if pairs_ref() is reposition_pairs:
                cached_dir_sq = dir_sq

        stats = None
        for index, param in enumerate(params):
            grad = param.grad
            state = self.actor_optimizer.state.get(param, {})
            exp_avg = state.get('exp_avg')
            exp_avg_sq = state.get('exp_avg_sq')
            if grad is None or exp_avg is None or exp_avg_sq is None:
                continue
            if grad.device != exp_avg.device or grad.device != exp_avg_sq.device:
                # Optimizer offload can leave state on CPU.  Skipping this
                # optional diagnostic avoids an extra full-model transfer.
                continue
            grad_f = grad.detach().float()
            avg_f = exp_avg.detach().float()
            avg_sq_f = exp_avg_sq.detach().float()
            if stats is None:
                stats = torch.zeros(6, dtype=torch.float32, device=grad_f.device)
            stats[0] += grad_f.square().sum()
            stats[1] += avg_f.square().sum()
            stats[2] += avg_sq_f.square().sum()
            stats[3] += (grad_f * avg_f).sum()
            if reposition_pairs is not None:
                post, pre = reposition_pairs[index]
                if post.device != grad_f.device or pre.device != grad_f.device:
                    continue
                direction_f = post.detach().float() - pre.detach().float()
                if cached_dir_sq is None:
                    stats[4] += direction_f.square().sum()
                stats[5] += (direction_f * avg_f).sum()

        if stats is None:
            return {}
        if reposition_pairs is not None:
            if cached_dir_sq is not None:
                stats[4] = cached_dir_sq
            else:
                self._reposition_dir_cache = (weakref.ref(reposition_pairs), stats[4].clone())
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        grad_norm = stats[0].sqrt().item()
        avg_norm = stats[1].sqrt().item()
        direction_norm = stats[4].sqrt().item()
        eps = 1e-12
        return {
            'optimizer/exp_avg_norm': avg_norm,
            'optimizer/exp_avg_sq_norm': stats[2].sqrt().item(),
            'optimizer/fresh_grad_vs_momentum_cosine': stats[3].item() / (grad_norm * avg_norm + eps),
            'optimizer/reposition_direction_vs_momentum_cosine': (
                stats[5].item() / (direction_norm * avg_norm + eps)
                if reposition_pairs is not None else float('nan')
            ),
        }


    def compute_entorpy(self, data: DataProto) -> torch.Tensor:
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages', 'reward']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        batch = data.select(batch_keys=select_keys).batch

        print("*************************")
        print("*************************")
        print(batch.shape)
        print("*************************")
        print("*************************")
        dataloader = batch.split(self.config.ppo_mini_batch_size)
        entropies_cpu = []
        reward = []

        with torch.inference_mode():
            for batch_idx, data in enumerate(dataloader):
                mini_batch = data
                micro_batches = mini_batch.split(8)
                for data in micro_batches:
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload

                    responses = data['responses']
                    response_length = responses.size(1)
                    attention_mask = data['attention_mask']
                    response_mask = attention_mask[:, -response_length:]
                    old_log_prob = data['old_log_probs']
                    advantages = data['advantages']

                    log_reward_tensor = data['reward'].view(-1, 8)

                    print(log_reward_tensor.shape)

                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

                    entropies_cpu.append(entropy.detach().to("cpu"))

                    del entropy, log_prob, data

                torch.cuda.empty_cache()

            final_entropy = torch.cat(entropies_cpu, dim=0)

        print(final_entropy.shape)


        return final_entropy


    def compute_log_prob(self, data: DataProto) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info['micro_batch_size']
        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        use_dynamic_bsz = data.meta_info['use_dynamic_bsz']

        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids']
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info['max_token_len'] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(micro_batch, temperature=temperature)
            log_probs_lst.append(log_probs)
            entropy_lst.append(entropy)
        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == log_probs.size(0), f"{len(indices)} vs. {log_probs.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            log_probs = log_probs[revert_indices]
            entropys = entropys[revert_indices]

        return log_probs, entropys

    def _forward_opd2_micro_batch(self, micro_batch) -> torch.Tensor:
        """Dense RESPONSE-ONLY logits for OPD^2 (no rmpad, no temperature scaling).

        Differs from ``_forward_kd_micro_batch`` in one way that matters at OPD^2's
        length budget: ``logits_to_keep`` makes the model apply ``lm_head`` only to
        the positions we actually consume, instead of over the whole prompt+response
        and slicing afterwards. At 1024 prompt + 8192 response that skips a
        [b, 9216, V] intermediate -- ~2.8GB bf16 per row -- which is what caps
        ``opd2_micro_batch_size``. Same trick verl already uses in
        ``_get_per_token_logps``.

        Returns RAW logits (bsz, response_length, vocab); the caller divides by the
        temperature in fp32 (see topk_from_logits).
        """
        response_length = micro_batch['responses'].size(-1)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            position_ids = micro_batch['position_ids']
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)
            output = self.actor_module(input_ids=micro_batch['input_ids'],
                                       attention_mask=micro_batch['attention_mask'],
                                       position_ids=position_ids,
                                       logits_to_keep=response_length + 1,
                                       use_cache=False)
            # logits_to_keep=R+1 returns the last R+1 positions; position j predicts
            # token j+1, so the R rows that predict the response are all but the last.
            return output.logits[:, :-1, :]

    def compute_opd2_student(self, micro_batch, temperature: float, top_k: int,
                             chunk_tokens: int = OPD2_CHUNK_TOKENS):
        """Student side of the OPD^2 signal for ONE micro-batch of rows.

        OPD^2 weights its expectation corrections by the STUDENT's probability
        and truncates them to the STUDENT's top-K, so the columns every model is
        later gathered at are chosen here (see
        ``verl.workers.actor.opd2_signal`` for the formula).

        Deliberately per-micro-batch, not per-shard: a whole-shard
        ``[B, R, K]`` index tensor is 8.6TB at the paper's 256x8192 with K=1024.
        The caller consumes each group's columns against the teacher and frees
        them before moving on.

        Returns ``(base_gt [b, R], base_topk [b, R, K], topk_idx [b, R, K])`` --
        full-vocab-normalized log-probs at the rollout temperature, i.e. exactly
        what ``compute_log_prob`` produces for ``old_log_probs``.
        """
        self.actor_module.eval()
        responses = micro_batch['responses']
        b, r = responses.shape
        with torch.no_grad():
            # ponytail: dense [b, R, V] logits (~2.5GB bf16 at R=8192, V=152k,
            # b=1) -- the binding constraint on opd2_micro_batch_size. Upgrade
            # path if the length budget grows again: FSDP-safe hidden-state
            # forward + chunked lm_head, as the teacher side already does.
            #
            # The forward returns RAW logits: the temperature division happens in
            # FP32 inside topk_from_logits, because bf16's spacing at a logit of
            # ~30 is ~0.25 -- the same order as the OPD^2 signal itself.
            logits = self._forward_opd2_micro_batch(micro_batch=micro_batch)
            gt, lp, idx = topk_from_logits(logits.reshape(b * r, -1),
                                           responses.reshape(b * r),
                                           top_k, temperature=temperature,
                                           chunk_tokens=chunk_tokens)
            del logits
        k = lp.size(-1)
        return gt.view(b, r), lp.view(b, r, k), idx.view(b, r, k)

    def _sled_resolve_early_layer(self, early_layer: int) -> int:
        """0-based transformer-layer index for the SLED early exit.

        ``-1`` (default) selects the middle layer. Anything else is a literal
        0-based layer index. Raises a clear error when the depth cannot be
        read and no explicit layer was given.
        """
        if int(early_layer) >= 0:
            return int(early_layer)
        cfg = getattr(self.actor_module, 'config', None)
        n = None
        if cfg is not None:
            for attr in ('num_hidden_layers', 'n_layer', 'num_layers'):
                if hasattr(cfg, attr):
                    n = int(getattr(cfg, attr))
                    break
        if n is None:
            raise ValueError('SLED early_layer=-1 needs the model depth (num_hidden_layers); '
                             'pass an explicit +actor_rollout_ref.actor.sled_early_layer instead.')
        return max(0, n // 2)

    def _sled_norm_and_head(self):
        """(norm, lm_head) for the early-exit projection (Qwen2/Llama layout)."""
        mod = self.actor_module
        base = getattr(mod, 'model', None) or getattr(mod, 'transformer', None) or mod
        norm = getattr(base, 'norm', None) or getattr(base, 'final_layer_norm', None) \
            or getattr(base, 'ln_f', None)
        head = getattr(mod, 'lm_head', None) or getattr(mod, 'output_layer', None)
        if norm is None or head is None:
            raise ValueError('SLED early exit needs (model.norm, lm_head); '
                             f'got norm={norm is not None}, head={head is not None}.')
        return norm, head

    def compute_sled_delta_frozen(self, micro_batch, temperature: float, alpha: float,
                                  early_layer: int, top_k: int,
                                  chunk_tokens: int = SLED_CHUNK_TOKENS):
        """Frozen SLED-Delta statistics for ONE micro-batch of rows.

        Runs the actor (which still holds the rollout snapshot: this is called
        once per step BEFORE any update_policy pass) with hidden states, builds
        the early-exit virtual teacher ``q`` from the SAME snapshot, and returns
        frozen tensors -- ``A_delta``, ``log q(y)``, ``log p(y)``,
        ``E_p[log q]``, entropies, KLs -- plus the student's top-K columns
        (ids + log-probs) for the live OPD term at train time.

        All expectations are truncated to the student's top-K columns (the
        OPD^2 convention); no full-vocabulary fp32 matrix is ever held, only
        the ``[C, V]`` transient inside the top-K/gather helpers.

        Returns a dict of ``[b, R]`` (``[b, R, K]`` for top-K) fp32 tensors.
        Everything is target-side (no grad). Prompt/pad positions are zeroed;
        the caller masks them anyway.
        """
        was_training = self.actor_module.training
        self.actor_module.eval()
        responses = micro_batch['responses']
        b, r = responses.shape
        device = responses.device
        early_idx = self._sled_resolve_early_layer(early_layer)
        k = max(1, int(top_k))
        chunk_tokens = max(1, int(chunk_tokens))
        with torch.no_grad():
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                position_ids = micro_batch['position_ids']
                if position_ids.dim() == 3:  # qwen2vl mrope
                    position_ids = position_ids.transpose(0, 1)
                output = self.actor_module(input_ids=micro_batch['input_ids'],
                                           attention_mask=micro_batch['attention_mask'],
                                           position_ids=position_ids,
                                           output_hidden_states=True,
                                           use_cache=False)
                hiddens = output.hidden_states
                # Response token t is predicted by the row at -R-1+t (same
                # alignment as _forward_kd_micro_batch's logits slice).
                early_h = hiddens[1 + early_idx][:, -r - 1:-1, :].reshape(b * r, -1)
                final_h = hiddens[-1][:, -r - 1:-1, :].reshape(b * r, -1)
                del output, hiddens
                norm, head = self._sled_norm_and_head()
                n_tokens = b * r
                gt_ids = responses.reshape(n_tokens)
                a_parts, logp_parts, logq_parts, ep_parts = [], [], [], []
                ent_p_parts, ent_q_parts, kl_qp_parts, kl_pq_parts = [], [], [], []
                tk_lp_parts, tk_idx_parts = [], []
                for start in range(0, n_tokens, chunk_tokens):
                    end = min(start + chunk_tokens, n_tokens)
                    # RAW logits (temperature is divided in fp32 inside the
                    # top-K/gather helpers, mirroring the OPD^2 precision
                    # discipline). Full-vocabulary softmax happens only inside
                    # those helpers; the reductions below see just K columns.
                    with FSDP.summon_full_params(self.actor_module, writeback=False, recurse=False):
                        e_raw = head(norm(early_h[start:end])).float()
                        f_raw = head(norm(final_h[start:end])).float()
                    s_raw = sled_contrast_logits(f_raw, e_raw, alpha)
                    gt_c = gt_ids[start:end]
                    logp_gt_c, tk_lp_c, tk_idx_c = topk_from_logits(
                        f_raw, gt_c, k, temperature=temperature, chunk_tokens=chunk_tokens)
                    logq_gt_c, tk_lq_c = gather_from_logits(
                        s_raw, gt_c, tk_idx_c, temperature=temperature,
                        chunk_tokens=chunk_tokens)
                    stats_c = sled_truncated_frozen(tk_lp_c, tk_lq_c, logp_gt_c, logq_gt_c)
                    a_parts.append(stats_c['a_delta'])
                    logp_parts.append(logp_gt_c)
                    logq_parts.append(logq_gt_c)
                    ep_parts.append(stats_c['ep_logq'])
                    ent_p_parts.append(stats_c['ent_p'])
                    ent_q_parts.append(stats_c['ent_q'])
                    kl_qp_parts.append(stats_c['kl_q_p'])
                    kl_pq_parts.append(stats_c['kl_p_q'])
                    tk_lp_parts.append(tk_lp_c)
                    tk_idx_parts.append(tk_idx_c)
                    del e_raw, f_raw, s_raw
                del early_h, final_h
            out = {
                'sled_a_delta': torch.cat(a_parts, 0).view(b, r).to(device),
                'sled_logp_gt': torch.cat(logp_parts, 0).view(b, r).to(device),
                'sled_logq_gt': torch.cat(logq_parts, 0).view(b, r).to(device),
                'sled_ep_logq': torch.cat(ep_parts, 0).view(b, r).to(device),
                'sled_ent_p': torch.cat(ent_p_parts, 0).view(b, r).to(device),
                'sled_ent_q': torch.cat(ent_q_parts, 0).view(b, r).to(device),
                'sled_kl_q_p': torch.cat(kl_qp_parts, 0).view(b, r).to(device),
                'sled_kl_p_q': torch.cat(kl_pq_parts, 0).view(b, r).to(device),
                'sled_topk_logp': torch.cat(tk_lp_parts, 0).view(b, r, -1).to(device),
                'sled_topk_ids': torch.cat(tk_idx_parts, 0).view(b, r, -1).to(torch.int32).to(device),
            }
            self.actor_module.train(was_training)
            return out

    @torch.no_grad()
    def _forward_sled_live_topk(self, micro_batch, temperature: float, gather_ids,
                                chunk_tokens: int = SLED_CHUNK_TOKENS):
        """Live ``log pi_theta`` at the stored SLED top-K columns.

        Dense response-span forward (same peak class as the GRPO forward that
        just ran), fp32 temperature division, chunked gather. No grad: the
        gate is target-side; only the sampled ``log_prob`` carries gradient.
        Returns fp32 ``[b, R, K]``.
        """
        was_training = self.actor_module.training
        self.actor_module.eval()
        response_length = micro_batch['responses'].size(-1)
        chunk_tokens = max(1, int(chunk_tokens))
        b = micro_batch['responses'].size(0)
        row_chunk = max(1, int(self.config.get('sled_micro_batch_size', 2)))
        ids = gather_ids.long().reshape(b, response_length, -1)
        row_parts = []
        position_ids = micro_batch['position_ids']
        if position_ids.dim() == 3:  # qwen2vl mrope
            position_ids = position_ids.transpose(0, 1)
        for row_start in range(0, b, row_chunk):
            row_end = min(row_start + row_chunk, b)
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                output = self.actor_module(
                    input_ids=micro_batch['input_ids'][row_start:row_end],
                    attention_mask=micro_batch['attention_mask'][row_start:row_end],
                    position_ids=position_ids[row_start:row_end],
                    use_cache=False)
                logits = output.logits[:, -response_length - 1:-1, :]  # (b, R, V) RAW
                del output
            rb, rr, v = logits.shape
            row_ids = ids[row_start:row_end].reshape(rb * rr, -1)
            flat = logits.reshape(rb * rr, v)
            parts = []
            for start in range(0, rb * rr, chunk_tokens):
                end = min(start + chunk_tokens, rb * rr)
                lp = torch.nn.functional.log_softmax(
                    flat[start:end].float() / temperature, dim=-1)
                parts.append(lp.gather(-1, row_ids[start:end]))
                del lp
            row_parts.append(torch.cat(parts, 0).reshape(rb, rr, -1).float())
            del logits, flat, row_ids, parts
        out = torch.cat(row_parts, 0)
        self.actor_module.train(was_training)
        return out

    def _make_minibatch_iterator(self, data: DataProto):
        """Select PPO keys and split the batch into mini-batches (shared by all update paths)."""
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
        if self.config.get('use_kd', False):
            # Offline KD teacher cache (dense per response position). Keys are
            # appended only when present so non-KD batches keep working.
            for kd_key in ('teacher_topk_log_probs', 'teacher_topk_ids'):
                if kd_key in data.batch.keys() and kd_key not in select_keys:
                    select_keys.append(kd_key)
        if self.config.get('use_sled_delta', False):
            # GRPO+SLED-Delta: frozen target-side tensors attached once per step
            # by compute_sled_delta_signal. Presence-checked so batches without
            # them (sled_loss_coef=0, non-SLED runs) keep working untouched.
            for sled_key in ('sled_a_delta', 'sled_logq_gt', 'sled_logp_gt', 'sled_ep_logq',
                             'sled_topk_ids', 'sled_topk_logp'):
                if sled_key in data.batch.keys() and sled_key not in select_keys:
                    select_keys.append(sled_key)
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = 'multi_modal_inputs' in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ['multi_modal_inputs']
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            non_tensor_select_keys = None
            dataloader = batch.split(self.config.ppo_mini_batch_size)
        return dataloader, has_multi_modal_inputs, select_keys, non_tensor_select_keys

    def _backward_minibatch(self,
                            mini_batch,
                            temperature,
                            has_multi_modal_inputs,
                            select_keys,
                            non_tensor_select_keys=None,
                            recompute_old_log_probs=False,
                            collect_metrics=True):
        """Zero grads and run one full forward/backward over a PPO mini-batch.

        Counts as exactly one backward pass (gradient accumulation over
        micro-batches). Does NOT clip or step the optimizer. When
        `recompute_old_log_probs`, old_log_probs are refreshed under no_grad at
        the current parameters (GXPO probe/correction passes).
        """
        # split batch into micro_batches
        if has_multi_modal_inputs:
            self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
            micro_batches = mini_batch.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif self.config.use_dynamic_bsz:
            max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
            micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
        else:
            self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
            # split batch into micro_batches
            micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

        self.actor_optimizer.zero_grad()
        metrics = {}

        current_device = torch.cuda.current_device()
        target_device = torch.device('cuda', current_device)
        for data in micro_batches:
            # Support all hardwares. Skip the transfer when tensors are already
            # resident on the current device -- SFPO hoists the H2D copy above its
            # K+1 update loop, so the per-micro-batch .to() would be a repeated
            # no-op. Genuinely CPU-resident batches (offload paths) still transfer.
            if isinstance(data, DataProto):
                first_tensor = next((v for v in data.batch.values() if isinstance(v, torch.Tensor)), None)
                if first_tensor is not None and first_tensor.device == target_device:
                    data = {**data.batch, **data.non_tensor_batch}
                else:
                    data = {**data.batch.to(current_device), **data.non_tensor_batch}
            else:
                first_tensor = next((v for v in data.values() if isinstance(v, torch.Tensor)), None)
                if not (first_tensor is not None and first_tensor.device == target_device):
                    data = data.to(current_device)  # actor device is cpu when using offload
            responses = data['responses']
            response_length = responses.size(1)
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            # Offline KD switch (loss-only delta; GXPO 3-pass math is untouched).
            # Computed before the old_log_prob refresh so pure-KD probe passes
            # can skip it (one forward saved per pass per micro-batch).
            use_kd = (self.config.get('use_kd', False)
                      and 'teacher_topk_log_probs' in data
                      and 'teacher_topk_ids' in data)
            pure_kd = use_kd and not self.config.get('kd_use_pg', False)
            # GRPO+SLED-Delta is PG-only (the else branch below); pure_kd never
            # touches it. Pre-declared here so `if sled_diag is not None` after
            # the branch doesn't UnboundLocalError on the pure-KD path.
            sled_loss = None
            sled_gate = None
            sled_a_opd = None
            sled_diag = None
            if recompute_old_log_probs and pure_kd:
                # Pure KD never consumes old_log_probs.
                old_log_prob = None
            elif recompute_old_log_probs:
                with torch.no_grad():
                    # Entropy is discarded on this probe pass; skip softmax+logsumexp.
                    _, old_log_prob = self._forward_micro_batch(micro_batch=data,
                                                                temperature=temperature,
                                                                need_entropy=False)
            else:
                old_log_prob = data['old_log_probs']
            advantages = data['advantages']

            clip_ratio = self.config.clip_ratio
            entropy_coeff = self.config.entropy_coeff

            kd_coef = float(self.config.get('kd_coef', 1.0))

            # Entropy is consumed only through the logged entropy_loss metric and the
            # `- entropy_loss * entropy_coeff` term. Pure KD needs no entropy
            # forward at all (saves one full forward per micro-batch per probe
            # pass); the zero placeholder keeps metric keys stable.
            if pure_kd:
                # Pure-KD loss never subtracts the entropy term (policy_loss is
                # kd_coef * kd_loss), so entropy is metrics/trigger only. Compute
                # it on metric passes and skip the duplicate full forward on
                # silent probe passes (GXPO pass 2): one forward saved per
                # micro-batch per pass. With entropy_coeff=0 all passes skip.
                need_entropy = collect_metrics or entropy_coeff != 0
            else:
                need_entropy = collect_metrics or entropy_coeff != 0

            kd_loss = None
            kd_student_mass = None
            kd_teacher_mass = None
            if pure_kd:
                if need_entropy:
                    entropy, _ = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)
                    del entropy
                else:
                    entropy_loss = None
                # Response-filtered flat logits (+ aligned teacher rows); the
                # rmpad path skips every pad token, the dense fallback masks
                # them as before. Consumed in token chunks below.
                kd_flat = self._forward_kd_flat(micro_batch=data, temperature=temperature,
                                                response_mask=response_mask,
                                                has_multi_modal_inputs=has_multi_modal_inputs)
                if kd_flat is not None:
                    flat_logits, t_logps, t_ids, b_sel, j_sel = kd_flat
                    # On-policy runs set kd_reverse_kl=True: reverse KL is
                    # mode-seeking and will not pump entropy on the student's
                    # own uncertain prefixes the way forward KL does.
                    kd_fn = (compute_reverse_kl_topk_chunked
                             if self.config.get('kd_reverse_kl', False)
                             else compute_forward_kl_topk_chunked)
                    kd_out = kd_fn(
                        flat_logits,
                        t_logps,
                        t_ids.long(),
                        log_prob_min_clamp=self.config.get('kd_log_prob_min_clamp', -10.0),
                        loss_max_clamp=self.config.get('kd_loss_max_clamp', 10.0),
                        chunk_tokens=int(self.config.get('kd_chunk_tokens',
                                                          KD_TOPK_CHUNK_TOKENS)),
                    )
                    kd_loss = self._pool_kd_loss(kd_out['distillation_losses'], b_sel, j_sel,
                                                 response_mask)
                    kd_student_mass = kd_out['student_mass'].mean().detach()
                    kd_teacher_mass = kd_out['teacher_mass'].mean().detach()
                    del kd_out, flat_logits, t_logps, t_ids, b_sel, j_sel
                else:
                    kd_loss = torch.zeros((), device=response_mask.device)
                # The helper frees the [tokens, vocab] logits before returning
                # so the peak never stacks a logits copy on top of the chunked
                # FP32 work.
                del kd_flat
                if entropy_loss is None:
                    entropy_loss = kd_loss.new_zeros(())
                # Metric-compat placeholders (ray_trainer merges these keys).
                pg_loss = kd_loss.new_zeros(())
                pg_clipfrac = kd_loss.new_zeros(())
                ppo_kl = kd_loss.new_zeros(())
                policy_loss = kd_coef * kd_loss
            else:
                # all return: (bsz, response_length)
                if need_entropy:
                    entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
                else:
                    _, log_prob = self._forward_micro_batch(micro_batch=data,
                                                            temperature=temperature,
                                                            need_entropy=False)

                pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(
                    old_log_prob=old_log_prob,
                    log_prob=log_prob,
                    advantages=advantages,
                    eos_mask=response_mask,
                    cliprange=clip_ratio,
                    # 'token-mean' is verl's historical reduction and stays the
                    # default, so GRPO/SFPO/GXPO arms are untouched.
                    # 'seq-mean-token-mean' is trl's loss_type="grpo", which the
                    # OPD^2 recipe pins. Shared by all three GXPO passes, so the
                    # extrapolated update sees the same objective as the probes.
                    loss_agg_mode=self.config.get('loss_agg_mode', 'token-mean'))
                # compute entropy loss from entropy. A skipped entropy implies
                # entropy_coeff == 0, so the zero placeholder keeps policy_loss (and its
                # gradients) bit-identical.
                if need_entropy:
                    entropy_loss = verl_F.masked_mean(entropy, response_mask)
                else:
                    entropy_loss = pg_loss.new_zeros(())

                # compute policy loss
                policy_loss = pg_loss - entropy_loss * entropy_coeff
                if use_kd:
                    # Auxiliary KD on top of the PG loss (kd_use_pg=True).
                    kd_flat = self._forward_kd_flat(micro_batch=data, temperature=temperature,
                                                    response_mask=response_mask,
                                                    has_multi_modal_inputs=has_multi_modal_inputs)
                    if kd_flat is not None:
                        flat_logits, t_logps, t_ids, b_sel, j_sel = kd_flat
                        kd_fn = (compute_reverse_kl_topk_chunked
                                 if self.config.get('kd_reverse_kl', False)
                                 else compute_forward_kl_topk_chunked)
                        kd_out = kd_fn(
                            flat_logits,
                            t_logps,
                            t_ids.long(),
                            log_prob_min_clamp=self.config.get('kd_log_prob_min_clamp', -10.0),
                            loss_max_clamp=self.config.get('kd_loss_max_clamp', 10.0),
                            chunk_tokens=int(self.config.get('kd_chunk_tokens',
                                                              KD_TOPK_CHUNK_TOKENS)),
                        )
                        kd_loss = self._pool_kd_loss(kd_out['distillation_losses'], b_sel, j_sel,
                                                     response_mask)
                        kd_student_mass = kd_out['student_mass'].mean().detach()
                        kd_teacher_mass = kd_out['teacher_mass'].mean().detach()
                        del kd_out, flat_logits, t_logps, t_ids, b_sel, j_sel
                    else:
                        kd_loss = policy_loss.new_zeros(())
                    del kd_flat
                    policy_loss = policy_loss + kd_coef * kd_loss
                # GRPO+SLED-Delta auxiliary loss (loss-level composition: the
                # GRPO clip above never sees the SLED advantage, and the SLED
                # sign gate below never touches the GRPO term). sled_* already
                # default to None above; only overwritten below if enabled.
                if (self.config.get('use_sled_delta', False) and 'sled_a_delta' in data
                        and float(self.config.get('sled_loss_coef', 1.0)) != 0.0):
                    from verl.workers.actor.sled_delta import (live_opd_advantage,
                                                               sled_gated_per_token_loss)
                    sled_coef = float(self.config.get('sled_loss_coef', 1.0))
                    sled_grpo_coef = float(self.config.get('sled_grpo_coef', 1.0))
                    live_topk_lp = self._forward_sled_live_topk(
                        micro_batch=data, temperature=temperature,
                        gather_ids=data['sled_topk_ids'],
                        chunk_tokens=int(self.config.get('sled_chunk_tokens',
                                                         SLED_CHUNK_TOKENS)))
                    a_delta_mb = data['sled_a_delta']
                    sled_a_opd, sled_gate = live_opd_advantage(
                        a_delta_mb, data['sled_logq_gt'], data['sled_ep_logq'],
                        log_prob.detach(), live_topk_lp,
                        data['sled_topk_logp'].float().exp())
                    sled_tok = sled_gated_per_token_loss(log_prob, a_delta_mb, sled_gate)
                    sled_loss = core_algos.agg_loss(
                        sled_tok, response_mask,
                        self.config.get('loss_agg_mode', 'token-mean'))
                    policy_loss = sled_grpo_coef * policy_loss + sled_coef * sled_loss
                    # Per-token dL/dlogpi weights for the cheap interaction
                    # proxy (no extra backward passes): GRPO weight is -A*r on
                    # unclipped tokens and 0 where the clip binds; SLED weight
                    # is g*A_delta (times sled_coef downstream).
                    with torch.no_grad():
                        ratio_mb = torch.exp(log_prob.detach() - old_log_prob)
                        pg1 = -advantages * ratio_mb
                        pg2 = -advantages * torch.clamp(ratio_mb, 1.0 - clip_ratio,
                                                        1.0 + clip_ratio)
                        w_grpo = torch.where(pg2 > pg1, pg1.new_zeros(()), pg1)
                    sled_diag = {
                        'a_delta': a_delta_mb.detach(),
                        'a_opd': sled_a_opd.detach(),
                        'gate': sled_gate.detach(),
                        'r_delta': (data['sled_logq_gt'] - data['sled_logp_gt']).detach(),
                        'w_grpo': w_grpo.detach(),
                        'w_sled': (sled_gate.detach() * a_delta_mb.detach()),
                        'grpo_adv': advantages.detach(),
                        'pg_loss': pg_loss.detach(),
                        'policy_loss': policy_loss.detach(),
                        'sled_loss': sled_loss.detach(),
                        'sled_coef': sled_coef,
                    }
                    del live_topk_lp, sled_tok, ratio_mb, pg1, pg2, w_grpo
                del log_prob

            if self.config.use_kl_loss:
                ref_log_prob = data['ref_log_prob']
                # compute kl loss
                kld = core_algos.kl_penalty(logprob=log_prob,
                                            ref_logprob=ref_log_prob,
                                            kl_penalty=self.config.kl_loss_type)
                kl_loss = masked_mean(kld, response_mask)

                policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                if collect_metrics:
                    # deferred D2H sync: converted to python floats once per mini-batch below
                    append_to_dict(metrics, {'actor/kl_loss': kl_loss.detach()})
                    metrics['actor/kl_coef'] = self.config.kl_loss_coef

            if self.config.use_dynamic_bsz:
                # relative to the dynamic bsz
                loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
            else:
                loss = policy_loss / self.gradient_accumulation
            # Authoritative raw backward call site; one policy gradient can
            # contain many calls when gradients are accumulated.
            backward_start = time.perf_counter()
            loss.backward()
            self.raw_backward_calls += 1
            self._gxpo_power_guard(backward_start)

            if collect_metrics:
                # GPU-scalar accumulation with a single deferred D2H sync per
                # mini-batch below, instead of one .item() sync per micro-batch.
                micro_metrics = {
                    'actor/entropy_loss': entropy_loss.detach(),
                    'actor/pg_loss': pg_loss.detach(),
                    'actor/pg_clipfrac': pg_clipfrac.detach(),
                    'actor/ppo_kl': ppo_kl.detach(),
                }
                if kd_loss is not None:
                    micro_metrics['actor/kd_loss'] = kd_loss.detach()
                    if kd_student_mass is not None:
                        micro_metrics['actor/kd_student_mass'] = kd_student_mass
                    if kd_teacher_mass is not None:
                        micro_metrics['actor/kd_teacher_mass'] = kd_teacher_mass
                if sled_diag is not None:
                    # Loss-level composition, logged per component. The GRPO
                    # term is untouched by the SLED gate (see _backward_minibatch).
                    from verl.workers.actor.sled_delta import (agreement_quadrants,
                                                               per_token_grad_weight_cosine)
                    sd = sled_diag
                    m = response_mask
                    live = m.bool()
                    micro_metrics['loss/grpo'] = sd['pg_loss']
                    micro_metrics['loss/sled_delta'] = sd['sled_loss']
                    micro_metrics['loss/sled_weighted'] = sd['sled_loss'] * sd['sled_coef']
                    micro_metrics['loss/total'] = sd['policy_loss']
                    micro_metrics['sled/loss_coef'] = torch.tensor(
                        sd['sled_coef'], device=sd['sled_loss'].device)
                    micro_metrics['sled/adv_mean'] = verl_F.masked_mean(sd['a_delta'], m).detach()
                    micro_metrics['sled/adv_abs_mean'] = verl_F.masked_mean(
                        sd['a_delta'].abs(), m).detach()
                    micro_metrics['sled/adv_std'] = verl_F.masked_mean(
                        (sd['a_delta'] - micro_metrics['sled/adv_mean'])**2, m).sqrt().detach()
                    micro_metrics['sled/delta_mean'] = verl_F.masked_mean(sd['r_delta'], m).detach()
                    micro_metrics['sled/delta_abs_mean'] = verl_F.masked_mean(
                        sd['r_delta'].abs(), m).detach()
                    micro_metrics['sled/delta_std'] = verl_F.masked_mean(
                        (sd['r_delta'] - micro_metrics['sled/delta_mean'])**2, m).sqrt().detach()
                    micro_metrics['sled/opd_adv_mean'] = verl_F.masked_mean(sd['a_opd'], m).detach()
                    micro_metrics['sled/opd_adv_abs_mean'] = verl_F.masked_mean(
                        sd['a_opd'].abs(), m).detach()
                    micro_metrics['sled/opd_adv_std'] = verl_F.masked_mean(
                        (sd['a_opd'] - micro_metrics['sled/opd_adv_mean'])**2, m).sqrt().detach()
                    micro_metrics['sled/gate_keep_fraction'] = verl_F.masked_mean(
                        sd['gate'], m).detach()
                    quad = agreement_quadrants(sd['grpo_adv'], sd['a_delta'], m)
                    for qk, qv in quad.items():
                        micro_metrics[f'grpo_sled/{qk}'] = torch.tensor(
                            qv, device=sd['sled_loss'].device)
                    micro_metrics['grpo_sled/grad_weight_cosine'] = torch.tensor(
                        per_token_grad_weight_cosine(sd['w_grpo'], sd['w_sled'], m),
                        device=sd['sled_loss'].device)
                    micro_metrics['grpo/adv_mean'] = verl_F.masked_mean(sd['grpo_adv'], m).detach()
                    del sd, m, live
                append_to_dict(metrics, micro_metrics)

        # FSDP 2.9 may leave a full flat gradient after the backward hooks in
        # the vLLM hybrid transition. Repair it before GXPO captures or clips.
        self._reshard_full_fsdp_grads()

        # Materialize deferred GPU scalars in one sync. Values are bit-identical to
        # the previous per-micro-batch .item() conversions; list lengths unchanged.
        for key in ('actor/entropy_loss', 'actor/pg_loss', 'actor/pg_clipfrac', 'actor/ppo_kl',
                    'actor/kl_loss', 'actor/kd_loss', 'actor/kd_student_mass',
                    'actor/kd_teacher_mass', 'loss/grpo', 'loss/sled_delta', 'loss/sled_weighted',
                    'loss/total', 'sled/loss_coef', 'sled/adv_mean', 'sled/adv_abs_mean',
                    'sled/adv_std', 'sled/delta_mean', 'sled/delta_abs_mean', 'sled/delta_std',
                    'sled/opd_adv_mean', 'sled/opd_adv_abs_mean', 'sled/opd_adv_std',
                    'sled/gate_keep_fraction', 'grpo/adv_mean', 'grpo_sled/agree',
                    'grpo_sled/disagree', 'grpo_sled/pp', 'grpo_sled/pn', 'grpo_sled/np',
                    'grpo_sled/nn', 'grpo_sled/grad_weight_cosine'):
            vals = metrics.get(key)
            if vals and isinstance(vals[0], torch.Tensor):
                metrics[key] = torch.stack(vals).tolist()

        self.cumulative_bp += 1
        return metrics

    def update_policy(self, data: DataProto, reposition_pairs=None):
        # make sure we are in training mode
        self.actor_module.train()
        self._ensure_fsdp_gradient_sync()
        bp_start = self.cumulative_bp
        raw_backward_start = self.raw_backward_calls

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        dataloader, has_multi_modal_inputs, select_keys, non_tensor_select_keys = self._make_minibatch_iterator(data)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(dataloader):
                mb_metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                                      non_tensor_select_keys)
                if reposition_pairs is not None:
                    append_to_dict(mb_metrics, self._optimizer_state_metrics(reposition_pairs))
                _merge_metrics(metrics, mb_metrics)

                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})
        metrics['actor/cumulative_bp'] = self.cumulative_bp
        metrics['actor/policy_grad_evals_step'] = self.cumulative_bp - bp_start
        metrics['actor/cumulative_policy_grad_evals'] = self.cumulative_bp
        metrics['actor/raw_backward_calls_step'] = self.raw_backward_calls - raw_backward_start
        metrics['actor/cumulative_raw_backward_calls'] = self.raw_backward_calls
        return metrics

    # ------------------------------------------------------------------
    # GXPO: Gradient Extrapolation-Based Policy Optimization
    # ------------------------------------------------------------------

    @staticmethod
    def _gxpo_default_metrics(enabled: float = 0.0, z_score: float = 0.0) -> dict:
        """Metrics defined on a step where the GXPO 3-pass update did NOT run.

        Only the gate state qualifies. Everything else -- norms, retention
        ratios, scales, displacements -- is a measurement of an update that did
        not happen, and zero-filling it made a fallback step indistinguishable
        in wandb from a real step whose retention happened to be zero. The
        retention keys are therefore absent here, exactly as the per-family
        keys are absent from a GXPO step that did not use that family.
        ``reduce_metrics`` is a per-key mean with no key union, so an absent key
        simply does not contribute to that step.
        """
        return {
            'actor/gxpo_enabled': enabled,
            'actor/gxpo_trigger_z': z_score,
            'actor/gxpo_trigger_stat': 0.0,
            'actor/gxpo_trigger_streak': 0.0,
        }

    def _gxpo_init_buffers(self):
        if self._gxpo_bufs is not None:
            return
        self._gxpo_params = [p for p in self.actor_module.parameters() if p.requires_grad]
        # Three buffers, never four. Both optimizer-aware estimators need
        # u0 = theta1 - theta0, and neither needs a buffer of its own: for a
        # parameter they own, the g1 slot is dead weight. g1 exists only to form
        # the legacy coordinatewise ratio r = g1/g0 and its moment diagnostics --
        # a ratio that describes neither Muon's displacement (its step size is
        # gradient-magnitude invariant) nor AdamW's (its step is the
        # moment-preconditioned direction, not the gradient). So those parameters
        # store u0 in their g1 slot instead and never capture g1 at all; the
        # AdamW path recovers theta1 = theta0 + u0 from it and forms both
        # directions. The shutoff gate is unaffected -- it reads g0 and the
        # corrective gradient, never g1.
        self._gxpo_bufs = {
            name: [torch.empty_like(p) for p in self._gxpo_params]
            for name in ('theta0', 'g0', 'g1')
        }

    def _gxpo_adamw_direction_supported(self) -> bool:
        """Whether this optimizer's non-Muon parameters take a decoupled-AdamW step.

        The AdamW-direction estimator reconstructs ``d_t`` from the parameter
        displacement, which is exact for -- and only for -- an update of the form
        ``theta_{t+1} = (1 - lr * wd) * theta_t - lr * d_t``. Two implementations
        in this tree have that form: ``torch.optim.AdamW`` and the AdamW branch
        of ``verl.workers.muon.Muon``. Anything else (SGD, ``torch.optim.Adam``,
        whose weight decay is coupled into the gradient, an unknown wrapper) is
        NOT silently treated as AdamW.
        """
        optimizer = self.actor_optimizer
        if isinstance(optimizer, torch.optim.AdamW):
            return True
        try:
            from verl.workers.muon import Muon
        except ImportError:  # pragma: no cover - muon is always importable in-tree
            return False
        return isinstance(optimizer, Muon)

    def _gxpo_retention_kinds(self):
        """Per-parameter retention classification, cached for the actor's lifetime.

        Returns a list of :class:`RetentionKind` values aligned with
        ``self._gxpo_params``, or None when every parameter takes the legacy
        gradient-space path (the historical fast path, kept so a pure-legacy run
        allocates and branches exactly as it always did).

        ``gxpo_retention_space``:
          ``auto``   -- optimizer-aware. A Muon-owned matrix gets per-matrix
                        update-space retention; a parameter owned by a recognized
                        decoupled-AdamW implementation gets coordinatewise
                        AdamW-direction retention. An unrecognized optimizer falls
                        back to legacy gradient space with one warning rather than
                        being silently modelled as AdamW.
          ``grad``   -- force the legacy raw-gradient estimator everywhere
                        (clean A/B control; reproduces the pre-patch behavior).
          ``update`` -- force update-space everywhere (unchanged forced control).
        """
        if self._gxpo_retention_cache is not None:
            return self._gxpo_retention_cache[0]

        space = str(self.config.get('gxpo_retention_space', 'auto')).lower()
        if space not in ('auto', 'grad', 'update'):
            raise ValueError(
                f"gxpo_retention_space must be one of auto|grad|update, got '{space}'")

        if space == 'grad':
            kinds = None
        elif space == 'update':
            kinds = [RetentionKind.MUON_UPDATE] * len(self._gxpo_params)
        else:
            # Muon tags every parameter it owns at construction time
            # (verl/workers/muon.py, `self.state[p]['use_muon'] = ...`). Absent
            # for any other optimizer, hence the None default below.
            state = getattr(self.actor_optimizer, 'state', {})
            adamw_ok = self._gxpo_adamw_direction_supported()
            non_muon = (RetentionKind.ADAMW_DIRECTION if adamw_ok
                        else RetentionKind.LEGACY_GRAD)
            kinds = []
            for parameter in self._gxpo_params:
                use_muon = state.get(parameter, {}).get('use_muon', None)
                kinds.append(RetentionKind.MUON_UPDATE if use_muon else non_muon)
            if not adamw_ok and not self._gxpo_unsupported_optimizer_warned:
                self._gxpo_unsupported_optimizer_warned = True
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    print(f'[GXPO] WARNING: gxpo_retention_space=auto does not recognize '
                          f'{type(self.actor_optimizer).__name__} as a decoupled-AdamW '
                          f'optimizer; its parameters fall back to the legacy '
                          f'gradient-space estimator r = g1/g0 rather than being modelled '
                          f'with the AdamW optimizer-direction rule.', flush=True)
            if all(kind == RetentionKind.LEGACY_GRAD for kind in kinds):
                kinds = None

        self._gxpo_retention_cache = (kinds,)
        return kinds

    @staticmethod
    def _gxpo_u0_slot_mask(kinds):
        """Which g1 slots hold ``u0 = theta1 - theta0`` instead of a raw gradient.

        Both non-legacy estimators need the first probe displacement and neither
        needs raw ``g1``: Muon reads ``<u0, u1>/<u0, u0>`` off it, and the AdamW
        path reconstructs ``theta1 = theta0 + u0`` to form ``d0`` and ``d1``. So
        both borrow the g1 slot and GXPO still allocates three model-sized
        buffers, never four. The shutoff gate is unaffected -- it reads g0 and
        the corrective gradient, never g1.
        """
        if kinds is None:
            return None
        return [kind != RetentionKind.LEGACY_GRAD for kind in kinds]

    def _gxpo_param_group_hparams(self):
        """Return ``(lrs, weight_decays)`` aligned with ``self._gxpo_params``.

        Read from the live param groups rather than assumed constant: LR is
        schedule-driven and both may differ per group. ``torch.optim.AdamW``
        names the decay ``weight_decay``; Muon names it ``wd``.
        """
        lookup = {}
        for group in self.actor_optimizer.param_groups:
            lr = float(group.get('lr', 0.0))
            decay = float(group.get('weight_decay', group.get('wd', 0.0)))
            for parameter in group['params']:
                lookup[id(parameter)] = (lr, decay)
        pairs = [lookup.get(id(p), (0.0, 0.0)) for p in self._gxpo_params]
        return [lr for lr, _ in pairs], [wd for _, wd in pairs]

    @staticmethod
    def _all_ranks_flag(value: bool, device, reduce_op=None) -> bool:
        """Return a rank-consistent boolean without changing the caller's branch order."""
        flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
        if torch.distributed.is_initialized():
            if reduce_op is None:
                reduce_op = torch.distributed.ReduceOp.MIN
            torch.distributed.all_reduce(flag, op=reduce_op)
        return bool(flag.item())

    def _gxpo_global_tensor_rms(self, tensors):
        """Return per-flat-tensor RMS values invariant to FSDP shard size.

        Used for both retention activity thresholds: RMS(g0) on the legacy
        gradient path and RMS(d0) on the AdamW optimizer-direction path. One
        collective covers every tensor.

        ``tensors`` may be a generator, and each element is reduced to two
        scalars before the next is produced. That lets the AdamW caller stream
        freshly derived directions through here without ever holding a second
        full-model copy of them.
        """
        norms, counts = [], []
        for tensor in tensors:
            widened = tensor.float()
            norms.append(torch.linalg.vector_norm(widened))
            counts.append(widened.numel())
        if not norms:
            return []
        stats = torch.stack((torch.stack(norms).square(),
                             torch.tensor(counts, dtype=torch.float32,
                                          device=norms[0].device)))
        if torch.distributed.is_initialized() and isinstance(self.actor_module, FSDP):
            # FSDP.process_group is the sharding group for both FULL_SHARD and
            # HYBRID_SHARD. Do not reduce over the replica dimension a second
            # time: those ranks contain duplicate synchronized gradients.
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM,
                                         group=self.actor_module.process_group)
        return (stats[0] / stats[1].clamp_min(1.0)).sqrt().unbind()

    def _gxpo_update_space_dots(self, params, theta0, u0_bufs, muon_mask):
        """Return per-parameter ``[<u0, u1>, <u0, u0>]``, summed across FSDP shards.

        Retention in update space is a property of the *whole* parameter matrix,
        but under FSDP each rank holds only a slice of it. Both dot products are
        therefore reduced over the sharding process group before the ratio is
        formed -- reducing only one of them, or neither, would make rho a
        function of FSDP_SIZE. One collective covers every parameter.

        Returns None when nothing uses the update-space path.
        """
        selected = [i for i, use in enumerate(muon_mask) if use]
        if not selected:
            return None
        rows = []
        for i in selected:
            u0 = u0_bufs[i].float()
            # disp2 = theta2 - theta0, and u1 = disp2 - u0, so
            # <u0,u1> = <u0,disp2> - <u0,u0>. disp2 is materialized rather than
            # differencing <u0,theta2> - <u0,theta0>: those two dots are large
            # and nearly equal (weights ~1e-2, steps ~1e-7), so subtracting them
            # would cancel away every significant digit.
            disp2 = params[i].data.float() - theta0[i].float()
            self_dot = u0.square().sum()
            rows.append(torch.stack(((u0 * disp2).sum() - self_dot, self_dot)))
        dots = torch.stack(rows)
        if torch.distributed.is_initialized() and isinstance(self.actor_module, FSDP):
            # Same group and same reasoning as _gxpo_global_tensor_rms: shard
            # dimension only, never the replica dimension.
            torch.distributed.all_reduce(dots, op=torch.distributed.ReduceOp.SUM,
                                         group=self.actor_module.process_group)
        return dict(zip(selected, dots.unbind()))

    def _gxpo_adamw_direction_rms(self, indices, theta0, u0_bufs, lrs, weight_decays):
        """Return ``{index: RMS(d0)}`` for the AdamW-direction parameters.

        The activity gate divides by ``d0``, so it must be thresholded on ``d0``
        -- not on ``g0``, which is a different quantity once the moment
        preconditioner is in the loop. Each ``d0`` is rebuilt transiently and
        released as soon as its norm is taken, so this costs one extra
        elementwise pass and no persistent memory; the main loop rebuilds it.

        One collective, reduced over the FSDP sharding process group only, keeps
        the threshold invariant to shard count exactly as the g0 path is.
        """
        if not indices:
            return {}

        # Some Muon parameter groups intentionally have lr=0 during a schedule
        # boundary. No AdamW direction exists for those parameters; the main
        # reposition loop already assigns neutral retention there. Do not send
        # them through adamw_direction(), whose division-by-zero guard is meant
        # to protect direct callers.
        valid_indices = [i for i in indices if lrs[i] > 0.0]
        if not valid_indices:

            return {}
        def directions():
            for i in valid_indices:
                # u0 lives in the g1 slot, so d0 comes straight off the step --
                # forming theta0 + u0 only to subtract theta0 back off would round
                # away digits the ratio's denominator needs.
                yield adamw_direction_from_step(theta0[i].float(), u0_bufs[i].float(),
                                                lrs[i], weight_decays[i])

        return dict(zip(valid_indices, self._gxpo_global_tensor_rms(directions())))

    def _gxpo_validate_precision_contract(self):
        """Fail on every rank if AdamW/GXPO state has fallen out of FP32."""
        if self._gxpo_precision_validated or not self._gxpo_strict_precision:
            return
        errors = []
        for index, parameter in enumerate(self._gxpo_params):
            if parameter.dtype != torch.float32:
                errors.append(f'param[{index}]={parameter.dtype}')
            if parameter.grad is not None and parameter.grad.dtype != torch.float32:
                errors.append(f'grad[{index}]={parameter.grad.dtype}')
            for name, buffers in self._gxpo_bufs.items():
                if buffers[index].dtype != torch.float32:
                    errors.append(f'{name}[{index}]={buffers[index].dtype}')
            for state_name, value in self.actor_optimizer.state.get(parameter, {}).items():
                if torch.is_tensor(value) and (value.is_floating_point() or value.is_complex()):
                    if value.dtype != torch.float32:
                        errors.append(f'optimizer.{state_name}[{index}]={value.dtype}')
        device = self._gxpo_params[0].device
        valid_everywhere = self._all_ranks_flag(not errors, device)
        if not valid_everywhere:
            detail = ', '.join(errors[:8]) if errors else 'another rank reported a mismatch'
            raise RuntimeError(
                'GXPO precision contract violated: expected FP32 master parameters, retained '
                f'gradients, AdamW state, and GXPO buffers; {detail}')
        self._gxpo_precision_validated = True
        if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
            print('[precision] GXPO master/grads/AdamW-state/theta0/g0/g1 verified fp32')

    def _gxpo_release_buffers(self):
        """Free the theta0/g0/g1 caches (3 model-shard param-dtype buffers of dead VRAM).

        Only called once the shutoff gate has tripped permanently: `is_enabled`
        then returns False forever, `_gxpo_init_buffers` is never reached again,
        and the buffers would otherwise stay resident for the rest of training.
        """
        self._gxpo_bufs = None
        self._gxpo_params = []
        # Keyed on the released parameter objects; must not outlive them.
        self._gxpo_retention_cache = None

    def _gxpo_capture_grads(self, bufs, skip=None):
        """Copy p.grad into bufs. ``skip`` marks slots holding something else."""
        params = self._gxpo_params
        if skip is not None:
            for i, (p, buf) in enumerate(zip(params, bufs)):
                if skip[i]:
                    continue           # slot holds u0 for the update-space path
                if p.grad is None:
                    buf.zero_()
                else:
                    buf.copy_(p.grad)
            return
        grads = [p.grad for p in params]
        if all(grad is not None for grad in grads):
            torch._foreach_copy_(bufs, grads)
            return
        for grad, buf in zip(grads, bufs):
            if grad is None:
                buf.zero_()
            else:
                buf.copy_(grad)

    @staticmethod
    def _gxpo_copy_parameters(destinations, sources):
        """Copy cached parameter tensors with one foreach dispatch."""
        torch._foreach_copy_(destinations, sources)

    def _gxpo_restore_theta0(self):
        with torch.no_grad():
            self._gxpo_copy_parameters([p.data for p in self._gxpo_params],
                                        self._gxpo_bufs['theta0'])

    def _gxpo_minibatch_step(self, mini_batch, temperature, has_multi_modal_inputs, select_keys,
                             non_tensor_select_keys, force_standard=False, trigger_enabled=True,
                             defer_trigger=False):
        """One GXPO 3-pass update on a single PPO mini-batch (Algorithm 1 of the paper).

        Faithful port of gxpo_single_minibatch_update from the reference
        implementation: probe passes capture raw g0/g1 before optimizer-gradient
        clipping, retention ratio r = g1/g0_safe is clamped to [-2, 3], and the geometric scale
        S_K/S_2 is clamped to [1, K/2+1], and the slow correction is taken at
        theta_tilde = theta0 + alpha * scale * (theta2 - theta0).

        force_standard: skip extrapolation for this one step (degenerate batch, e.g. mass
        format-parse failures) without touching the shutoff gate's rolling baseline/trigger state -- the
        step is simply not fed into the gate at all, since it was never asked to.

        trigger_enabled: whether the outer training-step warmup has completed. During
        warmup, GXPO still updates its rolling baseline but cannot trip the shutoff gate.
        """
        state = self.gxpo_state
        force_all_steps = bool(self.config.get('gxpo_force_all_steps', False))
        if force_all_steps:
            # Force mode is an explicit ablation: never honor a statistical
            # shutoff or the hard active-step budget during this run.
            state.trigger_index = float('inf')
            state.budget_stop = None
        step_idx = state.step_count
        K, alpha, delta = state.K, state.alpha, state.delta
        min_eff = self.gxpo_min_effective_multiplier
        recompute_old = self.config.get('gxpo_recompute_old_log_probs', False)
        skip_corrective = self.config.get('gxpo_skip_corrective', False)

        def standard_step(fallback_triggered=False):
            metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                               non_tensor_select_keys)
            step_start = time.perf_counter()
            grad_norm = self._optimizer_step()
            self._gxpo_power_guard(step_start)
            append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})
            append_to_dict(metrics, self._gxpo_default_metrics(enabled=0.0))
            append_to_dict(metrics, {'actor/gxpo_fallback_triggered': float(fallback_triggered)})
            if not defer_trigger:
                state.step_count = step_idx + 1
            return metrics

        if force_standard or not state.is_enabled(step_idx):
            # A degenerate-batch skip is an actual GXPO fallback; a normal
            # post-trigger GRPO step is a planned shutoff and is represented
            # by prediction_active=0 plus fallback_step.
            return standard_step(fallback_triggered=force_standard)

        self._gxpo_init_buffers()
        params = self._gxpo_params
        theta0, g0_bufs, g1_bufs = (self._gxpo_bufs[k] for k in ('theta0', 'g0', 'g1'))
        kinds = self._gxpo_retention_kinds()
        # Slots whose g1 buffer holds u0 rather than a raw gradient.
        u0_slots = self._gxpo_u0_slot_mask(kinds)
        adamw_indices = ([i for i, kind in enumerate(kinds)
                          if kind == RetentionKind.ADAMW_DIRECTION] if kinds else [])
        muon_mask = ([kind == RetentionKind.MUON_UPDATE for kind in kinds]
                     if kinds else None)
        # Per-parameter (lr, weight_decay) as of each probe step. AdamW's
        # decoupled update is theta_{t+1} = (1 - lr*wd) * theta_t - lr * d_t, so
        # both are needed to invert it. They are read from the live param groups
        # after each step rather than assumed constant: LR is schedule-driven and
        # both may differ per group.
        lrs0 = wds0 = lrs1 = wds1 = None

        with torch.no_grad():
            self._gxpo_copy_parameters(theta0, [p.data for p in params])

        flag_device = torch.device('cuda', torch.cuda.current_device())
        optimizer_transaction = snapshot_optimizer_state(self.actor_optimizer)

        def restore_probe_state():
            self._gxpo_restore_theta0()
            optimizer_transaction.restore()

        def probe_optimizer_step():
            try:
                self.actor_optimizer.step()
            except BaseException:
                restore_probe_state()
                raise

        def probe_clip_grads():
            try:
                return self._clip_grads()
            except BaseException:
                restore_probe_state()
                raise

        def fallback():
            # A failed probe must not leak either its parameters or optimizer state.
            restore_probe_state()
            self.actor_optimizer.zero_grad(set_to_none=True)
            return standard_step(fallback_triggered=True)

        # Pass 1: g0 at theta0. Keep its loss metrics (actor/entropy_loss etc.) — the skip-corrective
        # ablation has no Pass 3 to source them from, and ray_trainer reads actor/entropy_loss every step.
        try:
            probe_metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                     non_tensor_select_keys, collect_metrics=skip_corrective)
        except BaseException:
            restore_probe_state()
            raise
        # Capture raw gradients for the retention ratio; clip only the optimizer step.
        self._gxpo_capture_grads(g0_bufs)
        step_start = time.perf_counter()
        gn0 = probe_clip_grads().detach().item()
        clip_scale_g0 = min(1.0, float(self.config.get('grad_clip', 1.0)) / (abs(gn0) + 1e-12))
        valid_gn0 = (gn0 == gn0 and abs(gn0) != float('inf') and gn0 > 1e-8)
        valid_gn0_global = self._all_ranks_flag(valid_gn0, flag_device)
        if valid_gn0_global:
            self._cast_optimizer_grads_to_param_dtype()
            probe_optimizer_step()
            self._gxpo_validate_precision_contract()
        self._gxpo_power_guard(step_start)
        if not valid_gn0_global:
            return fallback()

        # u0 = theta1 - theta0, the first real optimizer step, written into the
        # g1 slot of every parameter on an optimizer-aware estimator (see
        # _gxpo_init_buffers). Done in place via copy-then-subtract so no
        # full-size temporary is materialized.
        if u0_slots is not None:
            with torch.no_grad():
                for i, use in enumerate(u0_slots):
                    if use:
                        g1_bufs[i].copy_(params[i].data)
                        g1_bufs[i].sub_(theta0[i])
        if adamw_indices:
            lrs0, wds0 = self._gxpo_param_group_hparams()

        # Pass 2: g1 at theta_{t,1}
        try:
            self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                     non_tensor_select_keys, recompute_old_log_probs=recompute_old,
                                     collect_metrics=False)
        except BaseException:
            restore_probe_state()
            raise
        # Capture raw gradients before clipping, matching g0.
        self._gxpo_capture_grads(g1_bufs, skip=u0_slots)
        step_start = time.perf_counter()
        gn1 = probe_clip_grads().detach().item()
        clip_scale_g1 = min(1.0, float(self.config.get('grad_clip', 1.0)) / (abs(gn1) + 1e-12))
        valid_gn1 = (gn1 == gn1 and abs(gn1) != float('inf'))
        valid_gn1_global = self._all_ranks_flag(valid_gn1, flag_device)
        if valid_gn1_global:
            self._cast_optimizer_grads_to_param_dtype()
            probe_optimizer_step()
        self._gxpo_power_guard(step_start)
        if not valid_gn1_global:
            return fallback()
        if adamw_indices:
            lrs1, wds1 = self._gxpo_param_group_hparams()

        # Retention ratio, geometric scale, reposition (theta2 is the live p.data)
        device = theta0[0].device
        # stats layout: [g0_sq, g1_sq, dot_g0_g1, disp2_sq, dispK_sq, sum_r, sum_r_sq,
        #                n_active, n_total, scale_sum, errK_sq, dot_ce, closed_sq, explicit_sq,
        #                ratio_clipped, eff_sum, n_grad_coords, g0_sq_grad_only]
        # n_total (index 8) counts every coordinate and is the denominator for the
        # scale/effective-multiplier means. The r-ratio diagnostics (sum_r, n_active,
        # ratio_clipped, dot_g0_g1) exist only for parameters that actually took the
        # gradient-space path, so they are normalized by n_grad_coords (index 16)
        # and paired with g0_sq_grad_only (index 17) instead. Muon-owned parameters
        # have no g1 -- their slot holds u0 -- and no meaningful g1/g0 ratio.
        # These values are diagnostics only. FP64 widening of every model-sized gradient
        # buffer made each GXPO step perform several extra full-model reads and reductions.
        # FP32 accumulation is sufficient for the reported metrics and does not feed the
        # retention scale, reposition, optimizer, or shutoff decision.
        stats = torch.zeros(18, dtype=torch.float32, device=device)
        # Update-space aggregates are reported separately from the coordinatewise
        # ones: folding a per-matrix scalar into retention_mean/scale_mean would
        # silently make runs on the two paths incomparable.
        upd_stats = torch.zeros(3, dtype=torch.float32, device=device)  # [rho_sum, scale_sum, n_neg]
        n_update_space = 0
        # AdamW optimizer-direction aggregates, again kept apart: r = d1/d0 is a
        # different quantity from both the legacy gradient ratio and Muon's rho,
        # and averaging them together would report a number describing nothing.
        # [sum_r, sum_r_sq, n_active, ratio_clipped, n_coords, scale_sum,
        #  d0_sq, d1_sq, dot_d0_d1]
        # The last three are the AdamW analogue of g0_sq/g1_sq/dot_g0_g1: without
        # them an 'auto' run reports no magnitude or turn information about the
        # directions it is extrapolating, because the g1 slot holds u0.
        adamw_stats = torch.zeros(9, dtype=torch.float32, device=device)
        n_grad_params = 0
        adamw_scale_max = torch.zeros(1, dtype=torch.float32, device=device)
        n_adamw_params = 0
        scale_max = torch.zeros(1, dtype=torch.float32, device=device)
        param_sq = torch.zeros(1, dtype=torch.float32, device=device)
        do_diag = self._gxpo_diag_freq > 0 and (step_idx % self._gxpo_diag_freq == 0)

        diagnostic_start = time.perf_counter()
        with torch.no_grad():
            # Batched diagnostic reductions: one foreach kernel group replaces the old
            # per-parameter square/cast/sum chains and their full-model temporaries.
            # Numerically equivalent only (summation order differs), which is fine --
            # these feed nothing algorithmic (see comment above).
            if g0_bufs:
                for i in range(0, len(g0_bufs), _GXPO_NORM_CHUNK):
                    j = i + _GXPO_NORM_CHUNK
                    stats[0] += torch.stack(torch._foreach_norm(g0_bufs[i:j])).square().sum()
                    # stats[1] (g1_sq) is NOT batched here any more: a Muon-owned
                    # parameter's g1 slot holds u0, a displacement, and folding
                    # that into a gradient norm would be meaningless. It is
                    # accumulated in the gradient-space branch of the loop below.
                    param_sq += torch.stack(torch._foreach_norm(theta0[i:j])).square().sum()
                    torch.cuda.synchronize()

            global_g0_rms = (
                self._gxpo_global_tensor_rms(g0_bufs)
                if self._gxpo_fsdp_invariant_threshold else [None] * len(g0_bufs)
            )
            # Collectives for the two optimizer-aware paths: each must be reached
            # by every rank unconditionally, so they sit outside the
            # per-parameter loop. Every rank classifies the same parameters, so
            # every rank takes the same branch here.
            update_dots = (
                self._gxpo_update_space_dots(params, theta0, g1_bufs, muon_mask)
                if muon_mask is not None and any(muon_mask) else None
            )
            adamw_d0_rms = (
                self._gxpo_adamw_direction_rms(adamw_indices, theta0, g1_bufs, lrs0, wds0)
                if adamw_indices and self._gxpo_fsdp_invariant_threshold else {}
            )
            _p2_sync_every = max(1, _GXPO_NORM_CHUNK)
            for _p2_i, (p, t0, g0b, g1b, g0_rms) in enumerate(
                    zip(params, theta0, g0_bufs, g1_bufs, global_g0_rms)):
                if _p2_i and _p2_i % _p2_sync_every == 0:
                    torch.cuda.synchronize()
                # ---- ALGORITHMIC PATH: retention scale + reposition write.
                # ---- Grad-level precision: g0/g1 buffers inherit param dtype
                # ---- via empty_like. The r = g1/g0 ratio, K-step Horner sums,
                # ---- and displacement math run transiently in FP32 (required
                # ---- resolution at grad magnitudes ~1e-4..1e-2 with a 1e-8
                # ---- activity gate); the write-back casts to param dtype.
                # ---- Under FP32 params these float() calls are no-ops.
                kind = kinds[_p2_i] if kinds is not None else RetentionKind.LEGACY_GRAD
                is_update_space = kind == RetentionKind.MUON_UPDATE
                is_adamw_direction = kind == RetentionKind.ADAMW_DIRECTION
                if is_update_space:
                    # Muon-owned matrix: gradient ratios say nothing about its
                    # displacement (its step size is gradient-magnitude
                    # invariant), so read retention off the two real optimizer
                    # steps instead, as a scalar that leaves disp2's direction --
                    # the orthogonalized direction Muon chose -- intact.
                    #
                    # compute_gxpo_retention_scale is deliberately NOT called
                    # here. Its result was previously computed and thrown away,
                    # which cost a dozen full-size FP32 temporaries per matrix
                    # for a ratio we proved carries no signal for Muon. g1b holds
                    # u0 for these parameters, so calling it would be wrong now
                    # as well as wasteful.
                    rho_u, scale = compute_gxpo_update_retention_scale(
                        None, None, K, delta, dots=update_dots[_p2_i])
                    upd_stats[0] += rho_u
                    upd_stats[1] += scale
                    upd_stats[2] += (rho_u < 0).float()
                    n_update_space += 1
                elif is_adamw_direction:
                    # AdamW-owned parameter: retention is the ratio of the two
                    # *adaptive* directions the optimizer actually applied, not
                    # of the raw gradients. Both are reconstructed from the real
                    # probe displacements, which is what makes this exact rather
                    # than a re-derivation of AdamW's internals:
                    #
                    #   theta1 = theta0 + u0                (u0 lives in the g1 slot)
                    #   d0 = (c0 * theta0 - theta1) / lr0
                    #   d1 = (c1 * theta1 - theta2) / lr1,  c = 1 - lr * weight_decay
                    #
                    # so d_t carries AdamW's moments, bias correction, epsilon
                    # convention, and the clipped gradient it was really handed.
                    # g1b holds u0 for these parameters, so the gradient-space
                    # estimator is not merely wrong here, it is inapplicable.
                    lr0, wd0 = lrs0[_p2_i], wds0[_p2_i]
                    lr1, wd1 = lrs1[_p2_i], wds1[_p2_i]
                    n_adamw_params += 1
                    adamw_stats[4] += g0b.numel()
                    if lr0 > 0.0 and lr1 > 0.0:
                        t0f = t0.float()
                        u0 = g1b.float()
                        theta1 = t0f + u0
                        # d0 from the step, not the endpoint: theta1 rounds u0
                        # (~1e-7) against t0 (~1e-2), and d0 is what d1 divides by.
                        d0 = adamw_direction_from_step(t0f, u0, lr0, wd0)
                        del u0
                        d1 = adamw_direction(theta1, p.data.float(), lr1, wd1)
                        del theta1
                        r, scale, active, ratio_clipped = (
                            compute_gxpo_adamw_direction_retention_scale(
                                d0, d1, K, delta, d0_rms=adamw_d0_rms.get(_p2_i)))
                        adamw_stats[6] += d0.square().sum()
                        adamw_stats[7] += d1.square().sum()
                        adamw_stats[8] += (d0 * d1).sum()
                        del d0, d1
                        adamw_stats[0] += r.sum()
                        adamw_stats[1] += r.square().sum()
                        adamw_stats[2] += active.sum()
                        adamw_stats[3] += ratio_clipped.sum()
                        adamw_stats[5] += scale.sum()
                        # A zero-sized parameter can produce an empty scale
                        # tensor under FSDP. It contributes no coordinates and
                        # must not be reduced with amax (PyTorch rejects an
                        # empty reduction); the neutral accumulator remains
                        # valid for the minibatch.
                        if scale.numel():
                            adamw_scale_max = torch.maximum(
                                adamw_scale_max, scale.amax().reshape(1))
                    else:
                        # LR warmup step: the probe steps moved nothing along a
                        # direction, so there is no retention to read. Neutral
                        # scale, and no coordinate counted as active.
                        scale = torch.ones((), dtype=torch.float32, device=device)
                        adamw_stats[5] += float(g0b.numel())
                        adamw_scale_max = torch.maximum(
                            adamw_scale_max, torch.ones_like(adamw_scale_max))
                else:
                    # ---- Grad-level precision: g0/g1 buffers inherit param dtype
                    # ---- via empty_like. The r = g1/g0 ratio, K-step Horner sums,
                    # ---- and displacement math run transiently in FP32 (required
                    # ---- resolution at grad magnitudes ~1e-4..1e-2 with a 1e-8
                    # ---- activity gate); the write-back casts to param dtype.
                    # ---- Under FP32 params these float() calls are no-ops.
                    n_grad_params += 1
                    g0f = g0b.float()
                    g1f = g1b.float()
                    r, scale, active, ratio_clipped = compute_gxpo_retention_scale(
                        g0f, g1f, K, delta, clip_scale_g0=clip_scale_g0,
                        clip_scale_g1=clip_scale_g1, g0_rms=g0_rms)
                    stats[1] += g1f.square().sum()
                    stats[2] += (g0f * g1f).sum()
                    stats[5] += r.sum()
                    stats[6] += r.square().sum()
                    stats[7] += active.sum()
                    stats[14] += ratio_clipped.sum()
                    stats[16] += g0b.numel()
                    stats[17] += g0f.square().sum()
                stats[8] += g0b.numel()
                # scale_mean/effective_multiplier are coordinate-weighted means
                # (they divide by stats[8], a coordinate count). The update-space
                # path produces ONE scalar for the whole matrix, so it must be
                # weighted by numel here or it contributes 1/numel of its due and
                # drags the reported mean far below the true multiplier.
                scale_weight = g0b.numel() if scale.numel() == 1 else 1
                stats[9] += scale.float().sum() * scale_weight
                if scale.numel():
                    scale_max = torch.maximum(scale_max, scale.float().amax().reshape(1))

                disp2 = p.data.float() - t0.float()
                stats[3] += disp2.square().sum()

                if do_diag and kind == RetentionKind.LEGACY_GRAD:
                    # Table 6: closed-form S_K/S_2 vs explicit Horner sums.
                    # Gradient-space only: this compares two ways of evaluating
                    # S_K(r)/S_2(r), and the update-space path has no r.
                    s_expl = torch.ones_like(r)
                    for _ in range(K - 1):
                        s_expl.mul_(r).add_(1.0)
                    s2_expl = 1.0 + r
                    scale_expl = torch.where(s2_expl.abs() > delta, s_expl / s2_expl, scale)
                    d_closed = disp2 * scale
                    d_expl = disp2 * scale_expl
                    stats[10] += (d_closed - d_expl).square().sum()
                    stats[11] += (d_closed * d_expl).sum()
                    stats[12] += d_closed.square().sum()
                    stats[13] += d_expl.square().sum()

                dispK = disp2 * scale  # out-of-place: keeps the FP32 chain exact
                stats[4] += dispK.square().sum()
                # alpha*scale is the multiplier actually applied to (theta2-theta0).
                # Out-of-place so the diagnostic scale above stays untouched.
                eff = scale * alpha
                if min_eff > 0.0:
                    eff.clamp_(min=min_eff)
                stats[15] += eff.sum() * scale_weight
                # Write-back casts FP32 -> param dtype. t0 is widened first:
                # in-place add_ rejects a mixed-dtype tensor argument.
                p.data.copy_(disp2.mul_(eff).add_(t0.float()))
        self._gxpo_power_guard(diagnostic_start)

        # Pass 3: slow correction at theta_tilde. Skipped for the no-corrective ablation, where the
        # reposition above IS the update (params already sit at theta_tilde, nothing more to do).
        # Optimizer state across the two probe steps. Either way theta_tilde stands:
        #   'transactional'            -- the fast trajectory was only a probe, so every
        #                                 probe-driven optimizer mutation is rolled back and
        #                                 the slow correction is taken from the moments (and
        #                                 step counter) the minibatch started with.
        #   'transactional_fast_state' -- no refresh: the two probe steps' moments and step
        #                                 counter are kept, and the slow correction is taken
        #                                 from them. Adam's step counter therefore advances
        #                                 3x per minibatch here, not 1x.
        # The snapshot taken before probe 1 stays live in both modes: it is the failure-path
        # rollback used by restore_probe_state() when a pass goes non-finite.
        if self.gxpo_optimizer_state_mode == 'transactional':
            optimizer_transaction.restore()

        if skip_corrective:
            pass3_metrics = probe_metrics  # reuse g0-probe loss metrics; no Pass 3 exists here
            gn_slow = gn1  # report the last real probe norm; no corrective grad exists
            gslow_stats = torch.zeros(2, dtype=torch.float32, device=device)
        else:
            try:
                pass3_metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                                         non_tensor_select_keys, recompute_old_log_probs=recompute_old,
                                                         collect_metrics=True)
            except BaseException:
                restore_probe_state()
                raise
            append_to_dict(pass3_metrics, self._optimizer_state_metrics())
            # Capture corrective-gradient diagnostics before clipping, matching g0/g1.
            # _clip_grads() returns the pre-clip norm but mutates p.grad in place.
            gslow_stats = torch.zeros(2, dtype=torch.float32, device=device)
            with torch.no_grad():
                for p, g0b in zip(params, g0_bufs):
                    if p.grad is None:
                        continue
                    gradf = p.grad.float()
                    g0f = g0b.float()
                    gslow_stats[0] += gradf.square().sum()
                    gslow_stats[1] += (gradf * g0f).sum()
            step_start = time.perf_counter()
            gn_slow = probe_clip_grads().detach().item()
            valid_gn_slow = (gn_slow == gn_slow and abs(gn_slow) != float('inf'))
            valid_gn_slow_global = self._all_ranks_flag(valid_gn_slow, flag_device)
            if valid_gn_slow_global:
                self._cast_optimizer_grads_to_param_dtype()
                probe_optimizer_step()
            self._gxpo_power_guard(step_start)
            if not valid_gn_slow_global:
                return fallback()

        # single global reduction so every rank takes the identical gate decision
        if torch.distributed.is_initialized():
            # adamw_stats holds coordinatewise sums over this rank's shard and is
            # reduced with the rest. upd_stats deliberately is not: rho is already
            # globally reduced inside _gxpo_update_space_dots, and n_update_space
            # is a parameter count identical on every rank, so summing would
            # multiply both by the world size.
            full = torch.cat([stats, gslow_stats, adamw_stats])
            torch.distributed.all_reduce(full, op=torch.distributed.ReduceOp.SUM)
            maxima = torch.cat([scale_max, adamw_scale_max])
            torch.distributed.all_reduce(maxima, op=torch.distributed.ReduceOp.MAX)
            scale_max, adamw_scale_max = maxima[:1], maxima[1:]
            torch.distributed.all_reduce(param_sq, op=torch.distributed.ReduceOp.SUM)
            stats, gslow_stats, adamw_stats = (full[:stats.numel()],
                                               full[stats.numel():stats.numel() + gslow_stats.numel()],
                                               full[stats.numel() + gslow_stats.numel():])
        param_norm = float(param_sq.sqrt().item())

        (adamw_sum_r, adamw_sum_r_sq, adamw_n_active, adamw_ratio_clipped,
         adamw_n_coords, adamw_scale_sum, adamw_d0_sq, adamw_d1_sq,
         adamw_dot_d0d1) = adamw_stats.tolist()
        adamw_coords = max(adamw_n_coords, 1.0)
        adamw_r_mean = adamw_sum_r / adamw_coords
        adamw_r_var = max(adamw_sum_r_sq / adamw_coords - adamw_r_mean**2, 0.0)
        adamw_d0_norm, adamw_d1_norm = adamw_d0_sq**0.5, adamw_d1_sq**0.5

        upd_rho_sum, upd_scale_sum, upd_n_neg = upd_stats.tolist()
        _upd_denom = max(n_update_space, 1)
        upd_rho_mean = upd_rho_sum / _upd_denom
        upd_scale_mean = upd_scale_sum / _upd_denom
        upd_neg_frac = upd_n_neg / _upd_denom

        vals = torch.cat([stats, gslow_stats, scale_max]).tolist()
        (g0_sq, g1_sq, dot01, disp2_sq, dispK_sq, sum_r, sum_r_sq, n_active, n_total, scale_sum,
         errK_sq, dot_ce, closed_sq, explicit_sq, ratio_clipped, eff_sum, n_grad_coords,
         g0_sq_grad_only, gslow_sq, dot0slow, scale_mx) = vals

        eps = 1e-12
        g0_norm, g1_norm, gslow_norm = g0_sq**0.5, g1_sq**0.5, gslow_sq**0.5
        # The r-ratio family is normalized over the coordinates that actually
        # took the gradient-space path. Under Muon with retention_space=auto that
        # is the AdamW-owned parameters only; n_grad_coords == n_total whenever no
        # parameter uses the update-space path, so pure-AdamW runs are unchanged.
        grad_coords = max(n_grad_coords, 1.0)
        r_mean = sum_r / grad_coords
        r_var = max(sum_r_sq / grad_coords - r_mean**2, 0.0)
        disp2_norm, dispK_norm = disp2_sq**0.5, dispK_sq**0.5

        # Contraction guard: alpha*scale < 1 means theta_tilde landed between theta0 and
        # theta2, so the 3-pass update made less progress than the two probe steps it paid
        # for. Warn once rather than let it pass silently.
        eff_mean = eff_sum / max(n_total, 1.0)
        if eff_mean < 1.0 and not self._gxpo_contraction_warned:
            self._gxpo_contraction_warned = True
            if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                print(f'[GXPO] WARNING: effective displacement multiplier '
                      f'alpha*scale={eff_mean:.4f} < 1 (alpha={alpha}, '
                      f'scale_mean={scale_sum / max(n_total, 1.0):.4f}, K={K}, '
                      f'r_mean={r_mean:.4f}). theta_tilde lands SHORT of theta2: the '
                      f'3-pass update is contracting, not extrapolating. Raise gxpo_alpha '
                      f'or lower gxpo_k; set gxpo_min_effective_multiplier=1.0 to clamp.',
                      flush=True)

        # Cosine gate mode (F1): the observation is |cos(g0, gslow)| computed from the
        # PRE-clip probe/corrective gradients -- a direct measurement of whether the
        # extrapolated direction still agrees with real optimization pressure. Healthy
        # production runs sit at 0.92-0.98; failing ones collapse (see
        # .audit/gxpo_algorithm_findings.md). Disagreement score = 1 - |cos| so the
        # existing "z >= tau trips" convention applies unchanged.
        cos_override = None
        if state.shutoff_mode == 'cosine':
            cos_override = 1.0 - abs(dot0slow / (g0_norm * gslow_norm + eps))
        # The gate reads PRE-clip norms (gn0/gn_slow, straight off _clip_grads), not the post-clip
        # g0_norm/gslow_norm logged below. Post-clip norms saturate at grad_clip, so a genuine
        # gradient blow-up pins the trigger stat to a constant and the z-score *shrinks* exactly
        # when it should fire -- observed in gxpo_kodcode_seed42_v2_fmtskip_k10_tau1.0 (548cfptf),
        # where raw grad_norm hit 2.2 while trigger_stat sat at 1.0 and z fell 0.77 -> 0.51.
        # gn_slow is already gn1 in the no-corrective ablation (see above), so this covers both.
        gate_slow_norm = gn_slow
        if defer_trigger:
            # Outer granularity defers the gate update until all minibatches in
            # this full batch have been reduced to one trigger statistic.
            z_score = 0.0
            trigger_stat = state.resolve_trigger_observation(
                g0_norm=gn0, g_slow_norm=gate_slow_norm, stat_override=cos_override)
            triggered = False
        else:
            z_score, trigger_stat, triggered = state.update_trigger_state(
                step=step_idx,
                g0_norm=gn0,
                g_slow_norm=gate_slow_norm,
                allow_trigger=trigger_enabled,
                defer_trigger=False,
                stat_override=cos_override)
            state.step_count = step_idx + 1

        if triggered and state.fallback_mode == 'permanent':
            # Permanent shutoff: extrapolation is disabled forever, so the cached
            # theta0/g0/g1 buffers are dead VRAM -- release them.
            self._gxpo_release_buffers()

        metrics = pass3_metrics
        append_to_dict(metrics, {'actor/grad_norm': float(gn_slow)})
        # Retention-kind census. A metric is emitted ONLY when the estimator that
        # defines it actually ran: under 'auto' the g1 slot holds u0 rather than a
        # gradient, so g1_norm / r_mean / r_std / cos_g0_g1 have no value, and
        # reporting 0.0 for them is worse than reporting nothing -- wandb cannot
        # distinguish a placeholder from a measurement and draws a flat line
        # through every retention plot. Absent is unambiguous: reduce_metrics is a
        # per-key np.mean with no key union, so a key simply missing from some
        # steps averages over the steps where it meant something.
        # 0 legacy_grad, 1 adamw_direction, 2 muon_update, 3 mixed. Codes match
        # the SFT arm so the two can be plotted together.
        has_grad = n_grad_coords > 0
        has_adamw = adamw_n_coords > 0
        has_muon = n_update_space > 0
        _kinds_present = sum((has_grad, has_adamw, has_muon))
        retention_kind_code = (3.0 if _kinds_present > 1
                               else 2.0 if has_muon
                               else 1.0 if has_adamw
                               else 0.0)
        append_to_dict(metrics, {
            'actor/gxpo_enabled': 1.0,
            'actor/gxpo_trigger_z': float(z_score),
            'actor/gxpo_trigger_stat': float(trigger_stat),
            'actor/gxpo_trigger_streak': float(state.trigger_streak),
            'actor/gxpo_retention_kind': retention_kind_code,
            'actor/gxpo_legacy_grad_params': float(n_grad_params),
            'actor/gxpo_update_space_params': float(n_update_space),
            'actor/gxpo_adamw_direction_params': float(n_adamw_params),
            # Path-agnostic: these describe the reposition itself, not a ratio,
            # so they are defined whichever estimator ran.
            'actor/gxpo_g0_norm': g0_norm,
            'actor/gxpo_gslow_norm': gslow_norm,
            'actor/gxpo_scale_mean': scale_sum / max(n_total, 1.0),
            'actor/gxpo_effective_multiplier': eff_mean,
            'actor/gxpo_contracting': 1.0 if eff_mean < 1.0 else 0.0,
            'actor/gxpo_scale_max': scale_mx,
            'actor/gxpo_disp2_norm': disp2_norm,
            'actor/gxpo_dispK_norm': dispK_norm,
            'actor/gxpo_dispK_over_disp2': dispK_norm / (disp2_norm + eps),
            'actor/gxpo_cos_g0_gslow': dot0slow / (g0_norm * gslow_norm + eps),
            'actor/gxpo_clip_scale_g0': float(clip_scale_g0),
            'actor/gxpo_fallback_triggered': 0.0,
            'reposition/jump_norm': abs(alpha) * dispK_norm,
            'reposition/jump_relative_to_param_norm': abs(alpha) * dispK_norm / (param_norm + eps),
        })
        if has_grad:
            # Legacy gradient-space family (r = g1/g0), on the coordinates that
            # actually took that path. cos_g0_g1 uses g0 restricted to those same
            # coordinates so it is a cosine between two comparable vectors.
            append_to_dict(metrics, {
                'actor/gxpo_g1_norm': g1_norm,
                'actor/gxpo_r_mean': r_mean,
                'actor/gxpo_r_std': r_var**0.5,
                'actor/gxpo_cos_g0_g1': dot01 / (g0_sq_grad_only**0.5 * g1_norm + eps),
                'actor/gxpo_inactive_frac': 1.0 - n_active / grad_coords,
                'actor/gxpo_ratio_clip_frac': ratio_clipped / max(n_active, 1.0),
                'actor/gxpo_clip_scale_g1': float(clip_scale_g1),
                'actor/gxpo_relative_threshold_reject_frac': 1.0 - n_active / grad_coords,
            })
        if has_muon:
            # Muon update-space family: a per-matrix scalar rho, never averaged
            # with either coordinatewise r.
            append_to_dict(metrics, {
                'actor/gxpo_retention_rho_mean': upd_rho_mean,
                'actor/gxpo_update_scale_mean': upd_scale_mean,
                'actor/gxpo_retention_rho_negative_frac': upd_neg_frac,
            })
        if has_adamw:
            # AdamW optimizer-direction family (r = d1/d0) on its own
            # denominator. d0_norm/d1_norm/cos_d0_d1 are the direct analogues of
            # g0_norm/g1_norm/cos_g0_g1 for the directions AdamW actually took.
            append_to_dict(metrics, {
                'actor/gxpo_adamw_r_mean': adamw_r_mean,
                'actor/gxpo_adamw_r_std': adamw_r_var**0.5,
                'actor/gxpo_adamw_scale_mean': adamw_scale_sum / adamw_coords,
                'actor/gxpo_adamw_scale_max': float(adamw_scale_max.item()),
                'actor/gxpo_adamw_inactive_frac': 1.0 - adamw_n_active / adamw_coords,
                'actor/gxpo_adamw_ratio_clip_frac':
                    adamw_ratio_clipped / max(adamw_n_active, 1.0),
                'actor/gxpo_adamw_d0_norm': adamw_d0_norm,
                'actor/gxpo_adamw_d1_norm': adamw_d1_norm,
                'actor/gxpo_adamw_cos_d0_d1':
                    adamw_dot_d0d1 / (adamw_d0_norm * adamw_d1_norm + eps),
            })
        if do_diag:
            errK = errK_sq**0.5
            append_to_dict(metrics, {
                'actor/gxpo_diag_thetaK_abs_err': errK,
                'actor/gxpo_diag_thetatilde_abs_err': alpha * errK,
                'actor/gxpo_diag_disp_cosine_err':
                    1.0 - dot_ce / (closed_sq**0.5 * explicit_sq**0.5 + eps),
            })
        if triggered:
            print(f'[GXPO] shutoff triggered at minibatch step {step_idx}: '
                  f'z={z_score:.3f} >= tau={state.tau} after '
                  f'{state.trigger_patience} consecutive observations -> single-pass GRPO from now on')
        return metrics

    def update_policy_gxpo(self, data: DataProto, trigger_enabled: bool = True,
                           trigger_stop: bool = False):
        """GXPO actor update: 3-pass extrapolated step per mini-batch while the
        shutoff gate is open, single-pass GRPO afterwards."""
        assert self.gxpo_state is not None, 'update_policy_gxpo called without use_gxpo=True'
        self.actor_module.train()
        self._ensure_fsdp_gradient_sync()
        bp_start = self.cumulative_bp
        raw_backward_start = self.raw_backward_calls
        self._gxpo_power_guard_active_s = 0.0
        self._gxpo_power_guard_sleep_s = 0.0

        # Guard against degenerate batches (e.g. a format-parse collapse: most rollouts get
        # reward 0 because they failed to parse, not because the problem was hard). Such a batch
        # has near-uniform reward -> near-zero-magnitude probe gradients g0/g1 -> a spurious,
        # huge z-score that permanently trips the shutoff gate on a fluke rather than a real
        # divergence. Skip extrapolation for this step only; the gate's rolling baseline is left untouched
        # since a degenerate batch was never a valid observation of the gate's trigger statistic.
        # The on-policy math reward is binary: both malformed and incorrect
        # responses are intentionally scored 0.  Therefore zero-reward mass
        # cannot be used as a format-error detector; doing so would disable
        # GXPO on nearly every batch.  Keep the legacy guard opt-in for tasks
        # that still have a separate format reward.
        reward_binary = bool(self.config.get('gxpo_binary_reward', False))
        format_error_ratio = (data.batch['token_level_scores'].sum(-1) == 0).float().mean().item()
        format_skip_enabled = bool(self.config.get('gxpo_format_error_skip_enabled', False)) and not reward_binary
        # This value is also read by _gxpo_minibatch_step, but that method has
        # its own local scope. Keep the outer update decision explicit here so
        # a trigger can transition to the standard step without a NameError.
        force_all_steps = bool(self.config.get('gxpo_force_all_steps', False))
        force_standard = (
            (format_skip_enabled and format_error_ratio > self.config.get('gxpo_format_error_skip_threshold', 0.5))
            or (trigger_stop and not force_all_steps)
        )

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        dataloader, has_multi_modal_inputs, select_keys, non_tensor_select_keys = self._make_minibatch_iterator(data)

        metrics = {}
        # GXPO's fallback gate follows SFPO exactly: entropy is computed by the
        # trainer from completed outer batches. The actor must never replace it
        # with a minibatch gradient-norm gate.
        entropy_trigger = self.config.get('gxpo_trigger_signal', 'entropy') == 'entropy'
        defer_trigger = entropy_trigger or self.config.get('gxpo_trigger_granularity', 'outer') == 'outer'
        for epoch in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                mb_metrics = self._gxpo_minibatch_step(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                                       non_tensor_select_keys, force_standard=force_standard,
                                                       trigger_enabled=trigger_enabled,
                                                       defer_trigger=defer_trigger)
                _merge_metrics(metrics, mb_metrics)
        if force_standard:
            metrics['actor/gxpo_format_skip'] = 1.0
        if defer_trigger:
            # Match SFPO: reduce minibatch trigger statistics to exactly one
            # scalar for this outer batch, score it against the preceding
            # rolling window, then append it to that window.
            stat_values = metrics.get('actor/gxpo_trigger_stat', [])
            if not isinstance(stat_values, list):
                stat_values = [stat_values]
            stat_values = [float(value) for value in stat_values
                           if value == value and abs(value) != float('inf')]
            outer_stat = sum(stat_values) / len(stat_values) if stat_values else 0.0

            if entropy_trigger:
                # The trainer already computed the SFPO-style entropy gate
                # before this actor update. Preserve those values verbatim;
                # no gradient statistic is fed into GXPOState.
                outer_z = float(data.meta_info.get('gxpo_trigger_z', 0.0))
                outer_stat = float(data.meta_info.get('gxpo_trigger_stat', 0.0))
                triggered = False
                self.gxpo_state.step_count += 1
            elif force_standard:
                outer_z = 0.0
                triggered = False
                self.gxpo_state.step_count += 1
            elif self.gxpo_state.trigger_index != float('inf'):
                # The gate is permanently closed; this outer batch is still
                # one training step, but it is not a new gate observation.
                outer_z = 0.0
                triggered = False
                self.gxpo_state.step_count += 1
            else:
                outer_step = self.gxpo_state.step_count
                outer_z, outer_stat, triggered = self.gxpo_state.update_trigger_state(
                    step=outer_step,
                    g0_norm=outer_stat,
                    g_slow_norm=outer_stat,
                    allow_trigger=trigger_enabled,
                    defer_trigger=False,
                    stat_override=outer_stat)
                self.gxpo_state.step_count = outer_step + 1

            metrics['actor/gxpo_trigger_z'] = float(outer_z)
            metrics['actor/gxpo_trigger_stat'] = float(outer_stat)
            metrics['actor/gxpo_trigger_candidate'] = float(outer_z >= self.gxpo_state.tau)
            metrics['actor/gxpo_trigger_streak'] = float(self.gxpo_state.trigger_streak)
            if triggered and self.gxpo_state.fallback_mode == 'permanent':
                # Permanent shutoff: free the now-dead extrapolation buffers.
                self._gxpo_release_buffers()
            if triggered:
                print(f'[GXPO] outer-batch shutoff triggered after '
                      f'{self.gxpo_state.trigger_patience} consecutive violating batches: '
                      f'z={outer_z:.3f} >= tau={self.gxpo_state.tau}')
        metrics['actor/cumulative_bp'] = self.cumulative_bp
        metrics['actor/policy_grad_evals_step'] = self.cumulative_bp - bp_start
        metrics['actor/cumulative_policy_grad_evals'] = self.cumulative_bp
        metrics['actor/raw_backward_calls_step'] = self.raw_backward_calls - raw_backward_start
        metrics['actor/cumulative_raw_backward_calls'] = self.raw_backward_calls
        metrics['actor/gxpo_power_guard_duty_cycle'] = self._gxpo_actor_duty_cycle
        metrics['actor/gxpo_power_guard_active_s'] = self._gxpo_power_guard_active_s
        metrics['actor/gxpo_power_guard_sleep_s'] = self._gxpo_power_guard_sleep_s
        enabled_values = metrics.get('actor/gxpo_enabled', 0.0)
        if isinstance(enabled_values, list):
            enabled_values = sum(enabled_values) / len(enabled_values) if enabled_values else 0.0
        metrics['actor/gxpo_prediction_active'] = float(enabled_values)
        # Mirrors train/gxpo_optim_state_kept on the SFT arm: 0 = probe optimizer
        # mutations rolled back (transactional), 1 = kept (transactional_fast_state).
        metrics['actor/gxpo_optim_state_kept'] = (
            0.0 if self.gxpo_optimizer_state_mode == 'transactional' else 1.0)
        metrics['actor/gxpo_trigger_warmup_active'] = float(not trigger_enabled)
        metrics['actor/gxpo_trigger_patience'] = float(self.gxpo_state.trigger_patience)
        metrics['actor/gxpo_fallback_step'] = (
            float(self.gxpo_state.trigger_index)
            if self.gxpo_state.trigger_index != float('inf') else float('nan')
        )
        if self.gxpo_state.trigger_index != float('inf'):
            metrics['actor/gxpo_shutoff_step'] = float(self.gxpo_state.trigger_index)
        # WARN-1 fix: a hard-budget stop (max_active_steps) closes the gate without
        # producing triggered=True anywhere, so the theta0/g0/g1 caches would stay
        # resident forever. Release them once the stop is terminal.
        if (self.gxpo_state.budget_stop and self.gxpo_state.fallback_mode != 'temporary'
                and self._gxpo_bufs is not None):
            self._gxpo_release_buffers()
            metrics['actor/gxpo_budget_buffers_released'] = 1.0
        return metrics
