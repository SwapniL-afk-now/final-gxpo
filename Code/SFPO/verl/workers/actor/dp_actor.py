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
import weakref
from typing import Iterable, Tuple

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.workers.actor import BasePPOActor
from verl.workers.actor.gxpo_state import GXPOState, compute_gxpo_retention_scale
from verl.utils.py_functional import append_to_dict
from verl.utils.torch_functional import logprobs_from_logits, masked_mean
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
        # Cache of the constant-per-outer-step reposition direction sum-of-squares
        # consumed by _optimizer_state_metrics; keyed by a weakref to the pairs list.
        self._reposition_dir_cache = None
        if self.config.get('use_gxpo', False) and actor_optimizer is not None:
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

    def _clip_grads(self):
        assert self.config.grad_clip is not None

        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        return grad_norm

    def _optimizer_step(self):
        grad_norm = self._clip_grads()
        self.actor_optimizer.step()
        return grad_norm

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

    def _make_minibatch_iterator(self, data: DataProto):
        """Select PPO keys and split the batch into mini-batches (shared by all update paths)."""
        select_keys = ['responses', 'input_ids', 'attention_mask', 'position_ids', 'old_log_probs', 'advantages']
        if self.config.use_kl_loss:
            select_keys.append('ref_log_prob')
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
            if recompute_old_log_probs:
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

            # Entropy is consumed only through the logged entropy_loss metric and the
            # `- entropy_loss * entropy_coeff` term. When metrics are not collected
            # and the coefficient is zero, skip the softmax+logsumexp entirely.
            need_entropy = collect_metrics or entropy_coeff != 0

            # all return: (bsz, response_length)
            if need_entropy:
                entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
            else:
                _, log_prob = self._forward_micro_batch(micro_batch=data,
                                                        temperature=temperature,
                                                        need_entropy=False)

            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                          log_prob=log_prob,
                                                                          advantages=advantages,
                                                                          eos_mask=response_mask,
                                                                          cliprange=clip_ratio)
            # compute entropy loss from entropy. A skipped entropy implies
            # entropy_coeff == 0, so the zero placeholder keeps policy_loss (and its
            # gradients) bit-identical.
            if need_entropy:
                entropy_loss = verl_F.masked_mean(entropy, response_mask)
            else:
                entropy_loss = pg_loss.new_zeros(())

            # compute policy loss
            policy_loss = pg_loss - entropy_loss * entropy_coeff

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
            loss.backward()
            self.raw_backward_calls += 1

            if collect_metrics:
                # GPU-scalar accumulation with a single deferred D2H sync per
                # mini-batch below, instead of one .item() sync per micro-batch.
                micro_metrics = {
                    'actor/entropy_loss': entropy_loss.detach(),
                    'actor/pg_loss': pg_loss.detach(),
                    'actor/pg_clipfrac': pg_clipfrac.detach(),
                    'actor/ppo_kl': ppo_kl.detach(),
                }
                append_to_dict(metrics, micro_metrics)

        # Materialize deferred GPU scalars in one sync. Values are bit-identical to
        # the previous per-micro-batch .item() conversions; list lengths unchanged.
        for key in ('actor/entropy_loss', 'actor/pg_loss', 'actor/pg_clipfrac', 'actor/ppo_kl',
                    'actor/kl_loss'):
            vals = metrics.get(key)
            if vals and isinstance(vals[0], torch.Tensor):
                metrics[key] = torch.stack(vals).tolist()

        self.cumulative_bp += 1
        return metrics

    def update_policy(self, data: DataProto, reposition_pairs=None):
        # make sure we are in training mode
        self.actor_module.train()
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
        return {
            'actor/gxpo_enabled': enabled,
            'actor/gxpo_trigger_z': z_score,
            'actor/gxpo_trigger_stat': 0.0,
            'actor/gxpo_trigger_streak': 0.0,
            'actor/gxpo_g0_norm': 0.0,
            'actor/gxpo_g1_norm': 0.0,
            'actor/gxpo_gslow_norm': 0.0,
            'actor/gxpo_r_mean': 0.0,
            'actor/gxpo_r_std': 0.0,
            'actor/gxpo_scale_mean': 0.0,
            'actor/gxpo_scale_max': 0.0,
            'actor/gxpo_disp2_norm': 0.0,
            'actor/gxpo_dispK_norm': 0.0,
            'actor/gxpo_dispK_over_disp2': 0.0,
            'actor/gxpo_cos_g0_g1': 0.0,
            'actor/gxpo_cos_g0_gslow': 0.0,
            'actor/gxpo_inactive_frac': 0.0,
            'actor/gxpo_ratio_clip_frac': 0.0,
        }

    def _gxpo_init_buffers(self):
        if self._gxpo_bufs is not None:
            return
        self._gxpo_params = [p for p in self.actor_module.parameters() if p.requires_grad]
        self._gxpo_bufs = {
            name: [torch.empty_like(p) for p in self._gxpo_params] for name in ('theta0', 'g0', 'g1')
        }

    def _gxpo_release_buffers(self):
        """Free the theta0/g0/g1 caches (~3 model-shard fp32 buffers of dead VRAM).

        Only called once the shutoff gate has tripped permanently: `is_enabled`
        then returns False forever, `_gxpo_init_buffers` is never reached again,
        and the buffers would otherwise stay resident for the rest of training.
        """
        self._gxpo_bufs = None
        self._gxpo_params = []

    def _gxpo_capture_grads(self, bufs):
        grads = [p.grad for p in self._gxpo_params]
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
        step_idx = state.step_count
        K, alpha, delta = state.K, state.alpha, state.delta
        recompute_old = self.config.get('gxpo_recompute_old_log_probs', False)
        skip_corrective = self.config.get('gxpo_skip_corrective', False)

        def standard_step(fallback_triggered=False):
            metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                               non_tensor_select_keys)
            grad_norm = self._optimizer_step()
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

        with torch.no_grad():
            self._gxpo_copy_parameters(theta0, [p.data for p in params])

        def fallback():
            # ponytail: light fallback — restore theta0 only; probe-step optimizer-moment
            # pollution is accepted (rare non-finite event), no optimizer-state deepcopy
            self._gxpo_restore_theta0()
            self.actor_optimizer.zero_grad(set_to_none=True)
            return standard_step(fallback_triggered=True)

        # Pass 1: g0 at theta0. Keep its loss metrics (actor/entropy_loss etc.) — the skip-corrective
        # ablation has no Pass 3 to source them from, and ray_trainer reads actor/entropy_loss every step.
        probe_metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                 non_tensor_select_keys, collect_metrics=skip_corrective)
        # Capture raw gradients for the retention ratio; clip only the optimizer step.
        self._gxpo_capture_grads(g0_bufs)
        gn0 = self._clip_grads().detach().item()
        if not (gn0 == gn0 and abs(gn0) != float('inf')) or gn0 <= 1e-8:
            return fallback()
        self.actor_optimizer.step()

        # Pass 2: g1 at theta_{t,1}
        self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                 non_tensor_select_keys, recompute_old_log_probs=recompute_old,
                                 collect_metrics=False)
        # Capture raw gradients before clipping, matching g0.
        self._gxpo_capture_grads(g1_bufs)
        gn1 = self._clip_grads().detach().item()
        if not (gn1 == gn1 and abs(gn1) != float('inf')):
            return fallback()
        self.actor_optimizer.step()

        # Retention ratio, geometric scale, reposition (theta2 is the live p.data)
        device = theta0[0].device
        # stats layout: [g0_sq, g1_sq, dot_g0_g1, disp2_sq, dispK_sq, sum_r, sum_r_sq,
        #                n_active, n_total, scale_sum, errK_sq, dot_ce, closed_sq, explicit_sq, ratio_clipped]
        # These values are diagnostics only. FP64 widening of every model-sized gradient
        # buffer made each GXPO step perform several extra full-model reads and reductions.
        # FP32 accumulation is sufficient for the reported metrics and does not feed the
        # retention scale, reposition, optimizer, or shutoff decision.
        stats = torch.zeros(15, dtype=torch.float32, device=device)
        scale_max = torch.zeros(1, dtype=torch.float32, device=device)
        param_sq = torch.zeros(1, dtype=torch.float32, device=device)
        do_diag = self._gxpo_diag_freq > 0 and (step_idx % self._gxpo_diag_freq == 0)

        with torch.no_grad():
            # Batched diagnostic reductions: one foreach kernel group replaces the old
            # per-parameter square/cast/sum chains and their full-model temporaries.
            # Numerically equivalent only (summation order differs), which is fine --
            # these feed nothing algorithmic (see comment above).
            if g0_bufs:
                stats[0] += torch.stack(torch._foreach_norm(g0_bufs)).square().sum()
                stats[1] += torch.stack(torch._foreach_norm(g1_bufs)).square().sum()
                param_sq += torch.stack(torch._foreach_norm(theta0)).square().sum()

            for p, t0, g0b, g1b in zip(params, theta0, g0_bufs, g1_bufs):
                # ---- ALGORITHMIC PATH: retention scale + reposition write stay
                # ---- op-for-op identical to the previous per-parameter loop.
                r, scale, active, ratio_clipped = compute_gxpo_retention_scale(g0b, g1b, K, delta)
                stats[2] += (g0b * g1b).sum()
                stats[7] += active.sum()
                stats[8] += g0b.numel()
                stats[14] += ratio_clipped.sum()
                stats[9] += scale.float().sum()
                scale_max = torch.maximum(scale_max, scale.float().amax().reshape(1))

                stats[5] += r.sum()
                stats[6] += r.square().sum()

                disp2 = p.data - t0
                stats[3] += disp2.square().sum()

                if do_diag:
                    # Table 6: closed-form S_K/S_2 vs explicit Horner sums
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

                dispK = disp2.mul_(scale)  # disp2 buffer becomes dispK
                stats[4] += dispK.square().sum()
                p.data.copy_(dispK.mul_(alpha).add_(t0))

        # Pass 3: slow correction at theta_tilde. Skipped for the no-corrective ablation, where the
        # reposition above IS the update (params already sit at theta_tilde, nothing more to do).
        if skip_corrective:
            pass3_metrics = probe_metrics  # reuse g0-probe loss metrics; no Pass 3 exists here
            gn_slow = gn1  # report the last real probe norm; no corrective grad exists
            gslow_stats = torch.zeros(2, dtype=torch.float32, device=device)
        else:
            pass3_metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                                     non_tensor_select_keys, recompute_old_log_probs=recompute_old,
                                                     collect_metrics=True)
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
            gn_slow = self._clip_grads().detach().item()
            if not (gn_slow == gn_slow and abs(gn_slow) != float('inf')):
                return fallback()
            self.actor_optimizer.step()

        # single global reduction so every rank takes the identical gate decision
        if torch.distributed.is_initialized():
            full = torch.cat([stats, gslow_stats])
            torch.distributed.all_reduce(full, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(scale_max, op=torch.distributed.ReduceOp.MAX)
            torch.distributed.all_reduce(param_sq, op=torch.distributed.ReduceOp.SUM)
            stats, gslow_stats = full[:15], full[15:]
        param_norm = float(param_sq.sqrt().item())

        vals = torch.cat([stats, gslow_stats, scale_max]).tolist()
        (g0_sq, g1_sq, dot01, disp2_sq, dispK_sq, sum_r, sum_r_sq, n_active, n_total, scale_sum,
         errK_sq, dot_ce, closed_sq, explicit_sq, ratio_clipped, gslow_sq, dot0slow, scale_mx) = vals

        eps = 1e-12
        g0_norm, g1_norm, gslow_norm = g0_sq**0.5, g1_sq**0.5, gslow_sq**0.5
        r_mean = sum_r / max(n_total, 1.0)
        r_var = max(sum_r_sq / max(n_total, 1.0) - r_mean**2, 0.0)
        disp2_norm, dispK_norm = disp2_sq**0.5, dispK_sq**0.5

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
        append_to_dict(metrics, {
            'actor/gxpo_enabled': 1.0,
            'actor/gxpo_trigger_z': float(z_score),
            'actor/gxpo_trigger_stat': float(trigger_stat),
            'actor/gxpo_trigger_streak': float(state.trigger_streak),
            'actor/gxpo_g0_norm': g0_norm,
            'actor/gxpo_g1_norm': g1_norm,
            'actor/gxpo_gslow_norm': gslow_norm,
            'actor/gxpo_r_mean': r_mean,
            'actor/gxpo_r_std': r_var**0.5,
            'actor/gxpo_scale_mean': scale_sum / max(n_total, 1.0),
            'actor/gxpo_scale_max': scale_mx,
            'actor/gxpo_disp2_norm': disp2_norm,
            'actor/gxpo_dispK_norm': dispK_norm,
            'actor/gxpo_dispK_over_disp2': dispK_norm / (disp2_norm + eps),
            'actor/gxpo_cos_g0_g1': dot01 / (g0_norm * g1_norm + eps),
            'actor/gxpo_cos_g0_gslow': dot0slow / (g0_norm * gslow_norm + eps),
            'actor/gxpo_inactive_frac': 1.0 - n_active / max(n_total, 1.0),
            'actor/gxpo_ratio_clip_frac': ratio_clipped / max(n_active, 1.0),
            'actor/gxpo_fallback_triggered': 0.0,
            'reposition/jump_norm': abs(alpha) * dispK_norm,
            'reposition/jump_relative_to_param_norm': abs(alpha) * dispK_norm / (param_norm + eps),
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
        bp_start = self.cumulative_bp
        raw_backward_start = self.raw_backward_calls

        # Guard against degenerate batches (e.g. a format-parse collapse: most rollouts get
        # reward 0 because they failed to parse, not because the problem was hard). Such a batch
        # has near-uniform reward -> near-zero-magnitude probe gradients g0/g1 -> a spurious,
        # huge z-score that permanently trips the shutoff gate on a fluke rather than a real
        # divergence. Skip extrapolation for this step only; the gate's rolling baseline is left untouched
        # since a degenerate batch was never a valid observation of the gate's trigger statistic.
        format_error_ratio = (data.batch['token_level_scores'].sum(-1) == 0).float().mean().item()
        force_standard = (
            format_error_ratio > self.config.get('gxpo_format_error_skip_threshold', 0.5)
            or trigger_stop
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
        enabled_values = metrics.get('actor/gxpo_enabled', 0.0)
        if isinstance(enabled_values, list):
            enabled_values = sum(enabled_values) / len(enabled_values) if enabled_values else 0.0
        metrics['actor/gxpo_prediction_active'] = float(enabled_values)
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
