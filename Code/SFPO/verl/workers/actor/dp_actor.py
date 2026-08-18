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

from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

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

        # GXPO: shutoff-gate state + lazily allocated per-parameter buffers
        self.gxpo_state = None
        self._gxpo_bufs = None
        if self.config.get('use_gxpo', False) and actor_optimizer is not None:
            self.gxpo_state = GXPOState(
                K=self.config.get('gxpo_k', 5),
                alpha=self.config.get('gxpo_alpha', 0.5),
                delta=self.config.get('gxpo_delta', 1e-8),
                tau=self.config.get('gxpo_tau', 0.5),
                omega=self.config.get('gxpo_omega', 0.1),
                shutoff_mode=self.config.get('gxpo_shutoff_mode', 'trajectory_aware'),
                fallback_mode=self.config.get('gxpo_fallback_mode', 'permanent'),
                fallback_window=self.config.get('gxpo_fallback_window', 10),
            )
            self._gxpo_diag_freq = int(self.config.get('gxpo_diag_freq', 10))

    def _forward_micro_batch(self, micro_batch, temperature) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
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

                # compute entropy
                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)

                # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                log_probs = logprobs_from_logits(logits=logits_rmpad, labels=input_ids_rmpad_rolled)

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outpus_and_unpad(log_probs, gather_dim=0, unpad_dim=0, padding_size=pad_size)
                    entropy_rmpad = gather_outpus_and_unpad(entropy_rmpad,
                                                            gather_dim=0,
                                                            unpad_dim=0,
                                                            padding_size=pad_size)
                # pad back to (bsz, seqlen)
                full_entropy = pad_input(hidden_states=entropy_rmpad.unsqueeze(-1),
                                         indices=indices,
                                         batch=batch_size,
                                         seqlen=seqlen)
                full_log_probs = pad_input(hidden_states=log_probs.unsqueeze(-1),
                                           indices=indices,
                                           batch=batch_size,
                                           seqlen=seqlen)

                # only return response part:
                entropy = full_entropy.squeeze(-1)[:, -response_length - 1:-1]  # (bsz, response_length)
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
                entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)

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
                            recompute_old_log_probs=False):
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

        for data in micro_batches:
            # Support all hardwares
            if isinstance(data, DataProto):
                data = {**data.batch.to(torch.cuda.current_device()), **data.non_tensor_batch}
            else:
                data = data.to(torch.cuda.current_device())  # actor device is cpu when using offload
            responses = data['responses']
            response_length = responses.size(1)
            attention_mask = data['attention_mask']
            response_mask = attention_mask[:, -response_length:]
            if recompute_old_log_probs:
                with torch.no_grad():
                    _, old_log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)
            else:
                old_log_prob = data['old_log_probs']
            advantages = data['advantages']

            clip_ratio = self.config.clip_ratio
            entropy_coeff = self.config.entropy_coeff

            # all return: (bsz, response_length)
            entropy, log_prob = self._forward_micro_batch(micro_batch=data, temperature=temperature)

            pg_loss, pg_clipfrac, ppo_kl = core_algos.compute_policy_loss(old_log_prob=old_log_prob,
                                                                          log_prob=log_prob,
                                                                          advantages=advantages,
                                                                          eos_mask=response_mask,
                                                                          cliprange=clip_ratio)
            # compute entropy loss from entropy
            entropy_loss = verl_F.masked_mean(entropy, response_mask)

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
                metrics['actor/kl_loss'] = kl_loss.detach().item()
                metrics['actor/kl_coef'] = self.config.kl_loss_coef

            if self.config.use_dynamic_bsz:
                # relative to the dynamic bsz
                loss = policy_loss * (len(data) / self.config.ppo_mini_batch_size)
            else:
                loss = policy_loss / self.gradient_accumulation
            loss.backward()

            micro_metrics = {
                'actor/entropy_loss': entropy_loss.detach().item(),
                'actor/pg_loss': pg_loss.detach().item(),
                'actor/pg_clipfrac': pg_clipfrac.detach().item(),
                'actor/ppo_kl': ppo_kl.detach().item(),
            }
            append_to_dict(metrics, micro_metrics)

        self.cumulative_bp += 1
        return metrics

    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        dataloader, has_multi_modal_inputs, select_keys, non_tensor_select_keys = self._make_minibatch_iterator(data)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(dataloader):
                mb_metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                                      non_tensor_select_keys)
                _merge_metrics(metrics, mb_metrics)

                print('>>> Weight update!!!')
                grad_norm = self._optimizer_step()
                append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})
        metrics['actor/cumulative_bp'] = self.cumulative_bp
        self.actor_optimizer.zero_grad()
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
                             non_tensor_select_keys, force_standard=False):
        """One GXPO 3-pass update on a single PPO mini-batch (Algorithm 1 of the paper).

        Faithful port of gxpo_single_minibatch_update from the reference
        implementation: probe passes capture raw g0/g1 before optimizer-gradient
        clipping, retention ratio r = g1/g0_safe is clamped to [-2, 3], and the geometric scale
        S_K/S_2 is clamped to [1, K/2+1], and the slow correction is taken at
        theta_tilde = theta0 + alpha * scale * (theta2 - theta0).

        force_standard: skip extrapolation for this one step (degenerate batch, e.g. mass
        format-parse failures) without touching the shutoff gate's EMA/trigger state -- the
        step is simply not fed into the gate at all, since it was never asked to.
        """
        state = self.gxpo_state
        step_idx = state.step_count
        K, alpha, delta = state.K, state.alpha, state.delta
        recompute_old = self.config.get('gxpo_recompute_old_log_probs', False)
        skip_corrective = self.config.get('gxpo_skip_corrective', False)

        def standard_step():
            metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                               non_tensor_select_keys)
            grad_norm = self._optimizer_step()
            append_to_dict(metrics, {'actor/grad_norm': grad_norm.detach().item()})
            append_to_dict(metrics, self._gxpo_default_metrics(enabled=0.0))
            state.step_count = step_idx + 1
            return metrics

        if force_standard or not state.is_enabled(step_idx):
            return standard_step()

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
            return standard_step()

        # Pass 1: g0 at theta0. Keep its loss metrics (actor/entropy_loss etc.) — the skip-corrective
        # ablation has no Pass 3 to source them from, and ray_trainer reads actor/entropy_loss every step.
        probe_metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                 non_tensor_select_keys)
        # Capture raw gradients for the retention ratio; clip only the optimizer step.
        self._gxpo_capture_grads(g0_bufs)
        gn0 = self._clip_grads().detach().item()
        if not (gn0 == gn0 and abs(gn0) != float('inf')) or gn0 <= 1e-8:
            return fallback()
        self.actor_optimizer.step()

        # Pass 2: g1 at theta_{t,1}
        self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                 non_tensor_select_keys, recompute_old_log_probs=recompute_old)
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
        stats = torch.zeros(15, dtype=torch.float64, device=device)
        scale_max = torch.zeros(1, dtype=torch.float64, device=device)
        do_diag = self._gxpo_diag_freq > 0 and (step_idx % self._gxpo_diag_freq == 0)

        with torch.no_grad():
            for p, t0, g0b, g1b in zip(params, theta0, g0_bufs, g1_bufs):
                stats[0] += g0b.double().pow(2).sum()
                stats[1] += g1b.double().pow(2).sum()
                stats[2] += (g0b.double() * g1b.double()).sum()
                r, scale, active, ratio_clipped = compute_gxpo_retention_scale(g0b, g1b, K, delta)
                stats[7] += active.sum()
                stats[8] += g0b.numel()
                stats[14] += ratio_clipped.sum()
                stats[9] += scale.double().sum()
                scale_max = torch.maximum(scale_max, scale.max().double().reshape(1))

                stats[5] += r.double().sum()
                stats[6] += r.double().pow(2).sum()

                disp2 = p.data - t0
                stats[3] += disp2.double().pow(2).sum()

                if do_diag:
                    # Table 6: closed-form S_K/S_2 vs explicit Horner sums
                    s_expl = torch.ones_like(r)
                    for _ in range(K - 1):
                        s_expl.mul_(r).add_(1.0)
                    s2_expl = 1.0 + r
                    scale_expl = torch.where(s2_expl.abs() > delta, s_expl / s2_expl, scale)
                    d_closed = disp2 * scale
                    d_expl = disp2 * scale_expl
                    stats[10] += (d_closed - d_expl).double().pow(2).sum()
                    stats[11] += (d_closed.double() * d_expl.double()).sum()
                    stats[12] += d_closed.double().pow(2).sum()
                    stats[13] += d_expl.double().pow(2).sum()

                dispK = disp2.mul_(scale)  # disp2 buffer becomes dispK
                stats[4] += dispK.double().pow(2).sum()
                p.data.copy_(dispK.mul_(alpha).add_(t0))

        # Pass 3: slow correction at theta_tilde. Skipped for the no-corrective ablation, where the
        # reposition above IS the update (params already sit at theta_tilde, nothing more to do).
        if skip_corrective:
            pass3_metrics = probe_metrics  # reuse g0-probe loss metrics; no Pass 3 exists here
            gn_slow = gn1  # report the last real probe norm; no corrective grad exists
            gslow_stats = torch.zeros(2, dtype=torch.float64, device=device)
        else:
            pass3_metrics = self._backward_minibatch(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                                     non_tensor_select_keys, recompute_old_log_probs=recompute_old)
            gn_slow = self._clip_grads().detach().item()
            if not (gn_slow == gn_slow and abs(gn_slow) != float('inf')):
                return fallback()

            gslow_stats = torch.zeros(2, dtype=torch.float64, device=device)
            with torch.no_grad():
                for p, g0b in zip(params, g0_bufs):
                    if p.grad is None:
                        continue
                    gslow_stats[0] += p.grad.double().pow(2).sum()
                    gslow_stats[1] += (p.grad.double() * g0b.double()).sum()
            self.actor_optimizer.step()

        # single global reduction so every rank takes the identical gate decision
        if torch.distributed.is_initialized():
            full = torch.cat([stats, gslow_stats])
            torch.distributed.all_reduce(full, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(scale_max, op=torch.distributed.ReduceOp.MAX)
            stats, gslow_stats = full[:14], full[14:]

        vals = torch.cat([stats, gslow_stats, scale_max]).tolist()
        (g0_sq, g1_sq, dot01, disp2_sq, dispK_sq, sum_r, sum_r_sq, n_active, n_total, scale_sum,
         errK_sq, dot_ce, closed_sq, explicit_sq, ratio_clipped, gslow_sq, dot0slow, scale_mx) = vals

        eps = 1e-12
        g0_norm, g1_norm, gslow_norm = g0_sq**0.5, g1_sq**0.5, gslow_sq**0.5
        r_mean = sum_r / max(n_total, 1.0)
        r_var = max(sum_r_sq / max(n_total, 1.0) - r_mean**2, 0.0)
        disp2_norm, dispK_norm = disp2_sq**0.5, dispK_sq**0.5

        # The gate reads PRE-clip norms (gn0/gn_slow, straight off _clip_grads), not the post-clip
        # g0_norm/gslow_norm logged below. Post-clip norms saturate at grad_clip, so a genuine
        # gradient blow-up pins the trigger stat to a constant and the z-score *shrinks* exactly
        # when it should fire -- observed in gxpo_kodcode_seed42_v2_fmtskip_k10_tau1.0 (548cfptf),
        # where raw grad_norm hit 2.2 while trigger_stat sat at 1.0 and z fell 0.77 -> 0.51.
        # gn_slow is already gn1 in the no-corrective ablation (see above), so this covers both.
        gate_slow_norm = gn_slow
        z_score, trigger_stat, triggered = state.update_trigger_state(step=step_idx,
                                                                      g0_norm=gn0,
                                                                      g_slow_norm=gate_slow_norm)
        state.step_count = step_idx + 1

        metrics = pass3_metrics
        append_to_dict(metrics, {'actor/grad_norm': float(gn_slow)})
        append_to_dict(metrics, {
            'actor/gxpo_enabled': 1.0,
            'actor/gxpo_trigger_z': float(z_score),
            'actor/gxpo_trigger_stat': float(trigger_stat),
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
                  f'|z|={abs(z_score):.3f} >= tau={state.tau} -> single-pass GRPO from now on')
        return metrics

    def update_policy_gxpo(self, data: DataProto):
        """GXPO actor update: 3-pass extrapolated step per mini-batch while the
        shutoff gate is open, single-pass GRPO afterwards."""
        assert self.gxpo_state is not None, 'update_policy_gxpo called without use_gxpo=True'
        self.actor_module.train()

        # Guard against degenerate batches (e.g. a format-parse collapse: most rollouts get
        # reward 0 because they failed to parse, not because the problem was hard). Such a batch
        # has near-uniform reward -> near-zero-magnitude probe gradients g0/g1 -> a spurious,
        # huge z-score that permanently trips the shutoff gate on a fluke rather than a real
        # divergence. Skip extrapolation for this step only; the gate's EMA is left untouched
        # since a degenerate batch was never a valid observation of the gate's trigger statistic.
        format_error_ratio = (data.batch['token_level_scores'].sum(-1) == 0).float().mean().item()
        force_standard = format_error_ratio > self.config.get('gxpo_format_error_skip_threshold', 0.5)

        temperature = data.meta_info['temperature']  # temperature must be in the data.meta_info to avoid slient error
        dataloader, has_multi_modal_inputs, select_keys, non_tensor_select_keys = self._make_minibatch_iterator(data)

        metrics = {}
        for epoch in range(self.config.ppo_epochs):
            for mini_batch in dataloader:
                mb_metrics = self._gxpo_minibatch_step(mini_batch, temperature, has_multi_modal_inputs, select_keys,
                                                       non_tensor_select_keys, force_standard=force_standard)
                _merge_metrics(metrics, mb_metrics)
        if force_standard:
            metrics['actor/gxpo_format_skip'] = 1.0
        metrics['actor/cumulative_bp'] = self.cumulative_bp
        if self.gxpo_state.trigger_index != float('inf'):
            metrics['actor/gxpo_shutoff_step'] = float(self.gxpo_state.trigger_index)
        self.actor_optimizer.zero_grad()
        return metrics
