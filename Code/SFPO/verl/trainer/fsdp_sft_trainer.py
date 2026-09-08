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
A lightweight one-file FSDP SFT Trainer
TODO(zhangchi.usc1992)
- Add calculation of mfu
- Add validation
"""

import os

os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

import gc
import json
import logging
import re
import shutil
import subprocess
from contextlib import nullcontext
import torch
import torch.distributed
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision, ShardingStrategy, CPUOffload
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM, PreTrainedModel, AutoConfig
from omegaconf import OmegaConf
from verl.utils.torch_functional import get_cosine_schedule_with_warmup
from tensordict import TensorDict
from torch.utils.data import DataLoader, DistributedSampler
from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis

from verl.utils.fsdp_utils import get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager
from verl.utils.dataset import SFTDataset
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from verl.utils.attention import resolve_attention_implementation
from peft import LoraConfig, TaskType, get_peft_model

from verl.workers.sharding_manager import FSDPUlyssesShardingManager
from verl.utils.ulysses import ulysses_pad_and_slice_inputs, gather_outpus_and_unpad
from verl import DataProto

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_SFT_LOGGING_LEVEL', 'WARN'))


def extract_step(path):
    match = re.search(r'global_step_(\d+)', path)
    if match:
        return int(match.group(1))
    return None


def convert_to_regular_types(obj):
    """Convert Hydra configs and other special types to regular Python types."""
    from omegaconf import ListConfig, DictConfig
    if isinstance(obj, (ListConfig, DictConfig)):
        return {k: convert_to_regular_types(v) for k, v in obj.items()} if isinstance(obj, DictConfig) else list(obj)
    elif isinstance(obj, (list, tuple)):
        return [convert_to_regular_types(x) for x in obj]
    elif isinstance(obj, dict):
        return {k: convert_to_regular_types(v) for k, v in obj.items()}
    return obj


# Sequence chunk for the reference log-normalizer. A [B, T, V] bf16 slice of
# 1024 positions (~0.6GB at V=152064) keeps the peak far below the full-row
# materialization while the logsumexp itself always runs in FP32.
REF_LSE_CHUNK_TOKENS = 1024


def k3_kl_per_token(student_logp_taken, ref_logp_taken):
    """Schulman K3 estimator of KL(pi_theta || pi_ref), per taken token.

    Same formula as the RL stack's ``low_var_kl`` (verl.trainer.ppo.core_algos):
    with r = log pi_ref(taken) - log pi_theta(taken), k3 = exp(r) - r - 1 is
    unbiased for the KL and non-negative, using only the taken token's two
    logprobs -- O(1) per token, no vocabulary sum. Inputs are flat [N].
    """
    assert student_logp_taken.shape == ref_logp_taken.shape, (
        f'taken logprobs must share shape, got {tuple(student_logp_taken.shape)} and '
        f'{tuple(ref_logp_taken.shape)}')
    r = ref_logp_taken.float() - student_logp_taken.float()
    return torch.clamp(torch.exp(r) - r - 1.0, min=-10.0, max=10.0)


def mean_over_batch_rows(sums_2d, counts_1d=None):
    """Equal-weight mean over batch rows (one response per row).

    `counts_1d` holds each row's supervised-token count. With it, a row is first
    reduced to its own per-token mean, so every response contributes equally to
    the batch no matter how long it is -- which is what "one response per row"
    is supposed to mean. Without it the row is a raw SUM over its tokens, and a
    2000-token response carries 2.5x the gradient of an 800-token one: exactly
    the length bias this normalization exists to remove. Passing the counts also
    keeps CE and the KL anchor on a common per-token scale, so kl_beta means the
    same thing regardless of response length.
    """
    row_sums = sums_2d.sum(dim=1)
    if counts_1d is not None:
        row_sums = row_sums / counts_1d.clamp_min(1.0)
    return row_sums.mean()


def mean_response_entropy(shift_logits_3d, resp_mask):
    """Mean per-response token entropy over masked positions.

    shift_logits_3d is [B, T, V] (any float dtype), resp_mask [B, T] bool.
    Full-vocabulary log-softmax runs in FP32, chunked over positions so the
    transient stays near 0.6GB at V=152064. Returns a scalar tensor.
    """
    batch_rows, seq_len = resp_mask.shape
    ent_sums = torch.zeros(batch_rows, device=shift_logits_3d.device)
    for start in range(0, seq_len, REF_LSE_CHUNK_TOKENS):
        end = min(start + REF_LSE_CHUNK_TOKENS, seq_len)
        chunk_mask = resp_mask[:, start:end]
        logp = torch.log_softmax(shift_logits_3d[:, start:end, :].float(), dim=-1)
        tok_ent = -(logp.exp() * logp).sum(dim=-1)
        chunk_rows = torch.arange(batch_rows, device=logp.device).unsqueeze(1).expand_as(
            chunk_mask)[chunk_mask]
        ent_sums.index_add_(0, chunk_rows, tok_ent[chunk_mask])
        del logp, tok_ent
    # Per-token mean, not a per-response sum: this feeds the GXPO shutoff gate,
    # which z-scores it against a rolling window. A raw sum scales with response
    # length, so batch-to-batch length variation would show up as entropy
    # "movement" the gate could trip on.
    return mean_over_batch_rows(
        ent_sums.unsqueeze(1), resp_mask.sum(dim=1).float())


class FSDPSFTTrainer(object):

    def __init__(self, config, device_mesh: DeviceMesh, ulysses_device_mesh: DeviceMesh):
        self.config = config
        self.device_mesh = device_mesh
        self.ulysses_device_mesh = ulysses_device_mesh
        self.sharding_manager = FSDPUlyssesShardingManager(self.ulysses_device_mesh)
        # build tokenizer first
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)
        from verl.utils import hf_tokenizer
        self.tokenizer = hf_tokenizer(local_model_path, trust_remote_code=self.config.model.trust_remote_code)
        if self.config.data.chat_template is not None:
            raise ValueError('Apply Chat template from config is not supported yet.')

        # normalize dp size
        self._normalize_config_bsz()

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, 'ulysses_sequence_parallel_size', 1)
        self.use_remove_padding = getattr(self.config, 'use_remove_padding', False)
        if self.device_mesh.get_rank() == 0:
            print(f'Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}')
            print(f'Using remove padding: {self.use_remove_padding}')

        self._build_dataloader()
        # Frozen reference policy for the optional KL(pi_theta || pi_ref) anchor.
        # Must be initialized BEFORE _build_model_optimizer, which builds it
        # when data.kl_beta > 0.
        self._ref_model = None
        self._last_kl_mean = 0.0
        self._last_entropy_mean = 0.0
        # build model
        self._build_model_optimizer()

        # GXPO-style update on the supervised objective: shutoff gate + lazily allocated
        # per-parameter buffers. Absent `optim.use_gxpo` keeps the plain 1-pass SFT path.
        self.gxpo_state = None
        self._gxpo_bufs = None
        # Checkpoint retention: keep only the single most-recent global_step_N
        # (for continuity/inspection) plus a separate best_checkpoint chosen by
        # the periodic vLLM benchmark score (eval_greedy/avg_pass1).
        self._best_eval_score = float('-inf')
        self._best_eval_step = None
        if self.config.optim.get('use_gxpo', False):
            from verl.workers.actor.gxpo_state import GXPOState
            from verl.workers.actor.optimizer_transaction import snapshot_optimizer_state
            self._snapshot_optimizer_state = snapshot_optimizer_state
            mode = str(self.config.optim.get('gxpo_optimizer_state_mode', 'transactional')).lower()
            if mode not in ('transactional', 'transactional_fast_state'):
                raise ValueError(
                    'optim.gxpo_optimizer_state_mode must be transactional or '
                    'transactional_fast_state (the moment-polluting legacy '
                    f'mode was removed), got {mode!r}')
            self.gxpo_optimizer_state_mode = mode
            # Effective displacement multiplier guard. theta_tilde = theta0 +
            # alpha*scale*(theta2-theta0), so alpha*scale is what decides whether the
            # reposition lands PAST theta2 (>1, the extrapolation the method is for) or
            # SHORT of it (<1, a contraction that spends three passes to undo progress two
            # of them made). 0.0 keeps the guard warn-only; set it to 1.0 to floor the
            # per-coordinate multiplier so theta_tilde can never land short of theta2.
            min_eff = float(self.config.optim.get('gxpo_min_effective_multiplier', 0.0))
            if min_eff < 0.0:
                raise ValueError('GXPO gxpo_min_effective_multiplier must be non-negative, '
                                 f'got {min_eff}')
            self.gxpo_min_effective_multiplier = min_eff
            self._gxpo_contraction_warned = False
            self._gxpo_precision_validated = False
            self._gxpo_strict_precision = bool(self.config.optim.get('gxpo_strict_precision', True))
            self._gxpo_fsdp_invariant_threshold = bool(
                self.config.optim.get('gxpo_fsdp_invariant_threshold', True))
            # Every knob is read from config so launchers own the full gate
            # profile; the fallbacks below mirror GXPOState's own defaults, so
            # runs that set nothing behave exactly as before.
            self.gxpo_state = GXPOState(
                K=self.config.optim.get('gxpo_k', 5),
                alpha=self.config.optim.get('gxpo_alpha', 0.5),
                delta=self.config.optim.get('gxpo_delta', 1e-8),
                tau=self.config.optim.get('gxpo_tau', 0.5),
                omega=self.config.optim.get('gxpo_omega', 0.1),
                zscore_w=self.config.optim.get('gxpo_zscore_w', 30),
                shutoff_mode=self.config.optim.get('gxpo_shutoff_mode', 'trajectory_aware'),
                fallback_mode=self.config.optim.get('gxpo_fallback_mode', 'permanent'),
                fallback_window=self.config.optim.get('gxpo_fallback_window', 10),
                trigger_patience=self.config.optim.get('gxpo_trigger_patience', 1),
                trigger_robust=self.config.optim.get('gxpo_trigger_robust', False),
                min_post_warmup_obs=self.config.optim.get('gxpo_trigger_min_obs', 0),
                warmup_steps=self.config.optim.get('gxpo_warmup', 0),
                abs_threshold=self.config.optim.get('gxpo_trigger_abs_threshold', 0.0),
                sustain_window=self.config.optim.get('gxpo_trigger_sustain_w', 10),
                max_active_steps=self.config.optim.get('gxpo_max_active_steps', 0),
                relative_threshold=self.config.optim.get('gxpo_relative_threshold', 0.0),
            )

        # TODO: add checkpoint manager
        if self.device_mesh.get_rank() == 0:
            print(self.config)

    def _normalize_config_bsz(self):
        dp_size = self.device_mesh.size(0) if not self.ulysses_device_mesh else self.ulysses_device_mesh.size(0)
        if self.device_mesh.get_rank() == 0:
            print(f'Normalize batch size by dp {dp_size}')

        assert self.config.data.train_batch_size % dp_size == 0, f"Global batch size {self.config.data.train_batch_size} is not divisible by dp size {dp_size}"

        self.config.data.train_batch_size //= dp_size

        assert self.config.data.train_batch_size % self.config.data.micro_batch_size_per_gpu == 0

    def _build_dataloader(self):
        config = self.config
        # Normalize hydra ListConfig -> plain list/str. Overrides like
        # data.train_files="['a.parquet']" arrive as ListConfig, which the
        # dataset's isinstance(..., List) check cannot see through.
        def _as_plain(value):
            from omegaconf import ListConfig, DictConfig
            if isinstance(value, (ListConfig, DictConfig)):
                return OmegaConf.to_container(value, resolve=True)
            return value
        # build dataset
        self.train_dataset = SFTDataset(parquet_files=_as_plain(config.data.train_files),
                                        tokenizer=self.tokenizer,
                                        prompt_key=config.data.prompt_key,
                                        prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                        response_key=config.data.response_key,
                                        response_dict_keys=config.data.get('response_dict_keys', None),
                                        max_length=config.data.max_length,
                                        truncation=config.data.truncation)
        # Validation is optional for training-only runs.  Keep the dataset and
        # dataloader absent when data.val_files is null/empty so no validation
        # data is loaded or evaluated.
        val_files = _as_plain(config.data.get('val_files', None))
        self.val_dataset = None
        self.val_sampler = None
        self.val_dataloader = None
        if val_files:
            self.val_dataset = SFTDataset(parquet_files=val_files,
                                          tokenizer=self.tokenizer,
                                          prompt_key=config.data.prompt_key,
                                          prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                          response_key=config.data.response_key,
                                          response_dict_keys=config.data.get('response_dict_keys', None),
                                          max_length=config.data.max_length,
                                          truncation=config.data.truncation)

        # build dataloader
        # Use data parallel rank and size instead of global rank and world size

        # If doing SP, we need to use the local rank and size
        if self.config.ulysses_sequence_parallel_size > 1:
            rank = self.ulysses_device_mesh.get_local_rank('dp')
            world_size = self.ulysses_device_mesh.size(0)
            if self.ulysses_device_mesh.get_rank() == 0:
                print(f'Using SP rank {rank} and size {world_size} for data distribution')
                print(f'Each SP rank gets different data, but the same data WITHIN the same rank')
        else:
            rank = self.device_mesh.get_rank()
            world_size = self.device_mesh.size()
        if self.device_mesh.get_rank() == 0:
            print(f'Using FSDP rank {rank} and size {world_size} for data distribution')

        # trainer.seed was declared in the config but never reached the sampler, so the
        # shuffle order was always DistributedSampler's default seed=0 regardless of it.
        self.train_sampler = DistributedSampler(self.train_dataset,
                                                shuffle=True,
                                                num_replicas=world_size,
                                                rank=rank,
                                                seed=config.trainer.get('seed', 0),
                                                drop_last=True)
        self.train_dataloader = DataLoader(dataset=self.train_dataset,
                                           batch_size=config.data.train_batch_size,
                                           sampler=self.train_sampler,
                                           num_workers=8,
                                           pin_memory=True,
                                           drop_last=True)

        if self.val_dataset is not None:
            self.val_sampler = DistributedSampler(self.val_dataset,
                                                  shuffle=False,
                                                  num_replicas=world_size,
                                                  rank=rank,
                                                  drop_last=True)
            self.val_dataloader = DataLoader(dataset=self.val_dataset,
                                             batch_size=config.data.micro_batch_size_per_gpu,
                                             sampler=self.val_sampler,
                                             num_workers=8,
                                             pin_memory=True,
                                             drop_last=True)

    def _build_model_optimizer(self):
        # TODO (zhangchi.usc1992):
        # 1. support pretrain from random weights
        # 2. support init directly from sharded weights
        local_model_path = copy_to_local(src=self.config.model.partial_pretrain, verbose=True)

        if self.config.model.get('external_lib', None) is not None:
            # This is used to import external_lib into the huggingface systems
            import importlib
            importlib.import_module(self.config.model.external_lib)

        log_gpu_memory_usage('Before model allocation', logger=logger)

        trust_remote_code = self.config.model.trust_remote_code
        # load config first
        config = AutoConfig.from_pretrained(local_model_path, trust_remote_code=trust_remote_code)
        if self.config.ulysses_sequence_parallel_size > 1:
            assert self.use_remove_padding, "Sequence parallel is only supported when remove_padding is enabled"
            from verl.models.registry import check_model_support_rmpad
            check_model_support_rmpad(config.model_type)

        if self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1:
            from verl.models.transformers.monkey_patch import apply_monkey_patch
            apply_monkey_patch(config, verbose=True)

        # This may be very large
        attn_implementation = resolve_attention_implementation(self.config.model, self.config.model.get('override_config', {}))
        print(f'Attention backend: {attn_implementation}')
        init_context = get_init_weight_context_manager(use_meta_tensor=not config.tie_word_embeddings,
                                                       mesh=self.device_mesh)

        with init_context():
            self.model: PreTrainedModel = AutoModelForCausalLM.from_pretrained(local_model_path,
                                                                               config=config,
                                                                               torch_dtype=torch.float32,
                                                                               attn_implementation=attn_implementation,
                                                                               trust_remote_code=trust_remote_code)

            # Apply Liger kernel if use_liger is enabled
            if self.config.model.get('use_liger', False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=self.model)

            if self.config.model.get('lora_rank', 0) > 0:
                self.model.enable_input_require_grads()
                # Convert config to regular Python types before creating PEFT model
                lora_config = {
                    'task_type': TaskType.CAUSAL_LM,
                    'r': self.config.model.lora_rank,
                    'lora_alpha': self.config.model.lora_alpha,
                    'target_modules': convert_to_regular_types(self.config.model.target_modules),
                    'bias': "none"
                }
                self.model = get_peft_model(self.model, LoraConfig(**lora_config))

        if self.config.model.enable_gradient_checkpointing:
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})

        log_gpu_memory_usage('After model allocation', logger=logger)

        mixed_precision = MixedPrecision(param_dtype=torch.bfloat16,
                                         reduce_dtype=torch.float32,
                                         buffer_dtype=torch.float32)

        auto_wrap_policy = get_fsdp_wrap_policy(self.model,
                                                config=self.config.model.fsdp_config.wrap_policy,
                                                is_lora=self.config.model.get('lora_rank', 0) > 0)
        if self.device_mesh.get_rank() == 0:
            print(auto_wrap_policy)

        if not self.config.model.fsdp_config.cpu_offload:
            cpu_offload = None
        else:
            cpu_offload = CPUOffload(offload_params=self.config.model.fsdp_config.offload_params)

        self.fsdp_model = FSDP(module=self.model,
                               auto_wrap_policy=auto_wrap_policy,
                               param_init_fn=init_fn,
                               sharding_strategy=ShardingStrategy.FULL_SHARD,
                               mixed_precision=mixed_precision,
                               device_mesh=self.device_mesh,
                               sync_module_states=True,
                               device_id=torch.cuda.current_device(),
                               cpu_offload=cpu_offload,
                               use_orig_params=False)

        log_gpu_memory_usage('After FSDP wrapping', logger=logger)

        self.optimizer = optim.AdamW(self.fsdp_model.parameters(),
                                     lr=self.config.optim.lr,
                                     betas=self.config.optim.betas,
                                     weight_decay=self.config.optim.weight_decay)

        # Frozen reference policy for the KL(pi_theta || pi_ref) anchor. A plain
        # bf16 copy (~3GB for 1.5B) with no FSDP wrapping: it only ever runs
        # forward under no_grad, so sharding it would buy nothing. Defaults to
        # the student's own starting weights (the standard RL KL reference);
        # override with data.kl_ref_model. Mirrors the student's liger patch so
        # the KL cannot see a fused-vs-eager numerics gap.
        if float(self.config.data.get('kl_beta', 0.0)) > 0.0:
            ref_path = copy_to_local(
                src=self.config.data.get('kl_ref_model', None) or self.config.model.partial_pretrain,
                verbose=True)
            ref_model = AutoModelForCausalLM.from_pretrained(
                ref_path, config=config, torch_dtype=torch.bfloat16,
                attn_implementation=attn_implementation,
                trust_remote_code=self.config.model.trust_remote_code)
            if self.config.model.get('use_liger', False):
                from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance
                _apply_liger_kernel_to_instance(model=ref_model)
            ref_model.requires_grad_(False)
            ref_model.eval()
            self._ref_model = ref_model.cuda()

        log_gpu_memory_usage('After initialize optimizer', logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        epoch_steps = self.steps_per_epoch * self.config.trainer.total_epochs
        configured_steps = self.config.trainer.get('total_training_steps', None)
        self.total_steps = epoch_steps if configured_steps is None else min(
            epoch_steps, int(configured_steps))

        if self.device_mesh.get_rank() == 0:
            print(
                f'Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}'
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=self.total_steps)

    def _compute_loss_and_backward(self, batch, do_backward=True, loss_scale=1.0):
        """Compute loss with optional sequence parallelism and remove padding features.

        `loss_scale` multiplies the loss *before* backward only. Gradient accumulation
        must divide by the micro-batch count here, not on the returned scalar: backward
        runs inside this function, so scaling the return value afterwards fixes the
        logged number while leaving the accumulated gradient n_micro_batches too large.
        The returned loss stays unscaled so callers keep reporting the same quantity.
        """
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        if use_sp and float(self.config.data.get('kl_beta', 0.0)) > 0.0:
            raise ValueError('data.kl_beta>0 (reference KL anchor) is not supported with sequence parallelism')
        # Move inputs to GPU and prepare loss mask
        input_ids = batch['input_ids'].cuda()
        attention_mask = batch['attention_mask'].cuda()
        position_ids = batch['position_ids'].cuda()
        loss_mask = batch.pop('loss_mask')[:, :-1].reshape(-1).cuda()
        loss_fct = nn.CrossEntropyLoss(reduction='none')

        # Context manager for sequence parallel if needed
        context = self.sharding_manager if use_sp else nullcontext()
        with context:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                if not use_sp:
                    # Standard forward pass without sequence parallel
                    labels = input_ids[:, 1:].contiguous()
                    output = self.fsdp_model(input_ids=input_ids,
                                             attention_mask=attention_mask,
                                             position_ids=position_ids,
                                             use_cache=False)
                    logits = output.logits
                    shift_logits = logits[..., :-1, :].contiguous()
                    # Flatten the tokens (labels were [B, S-1]; without the
                    # view CE sees batch 4 vs 16380 and dies in validation).
                    shift_labels = labels.contiguous().view(-1)
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss * loss_mask.to(loss.device)
                    # Per-response (not per-token) objective: each response counts
                    # once no matter how long it is, so long answers cannot
                    # dominate the batch. Legacy runs keep the token mean below.
                    normalize = bool(self.config.data.get('normalize_by_sequence', False))
                    kl_beta = float(self.config.data.get('kl_beta', 0.0))
                    # Mean response-token entropy for the entropy-signal gate.
                    # Computed only when a GXPO run actually gates on entropy;
                    # the slow correction pass runs last, so the gate below
                    # always sees this step's post-update entropy.
                    if (self.config.optim.get('use_gxpo', False)
                            and str(self.config.optim.get(
                                'gxpo_trigger_signal', 'grad')) == 'entropy'):
                        resp_mask_ent = loss_mask.to(loss.device).view(
                            input_ids.shape[0], -1).bool()
                        seq_len = shift_logits.size(0) // input_ids.shape[0]
                        self._last_entropy_mean = float(mean_response_entropy(
                            shift_logits.view(input_ids.shape[0], seq_len, -1),
                            resp_mask_ent).detach())
                    if normalize:
                        batch_rows = input_ids.shape[0]
                        # Supervised tokens per response (EOS included). Used to
                        # length-normalize every row below.
                        resp_counts = loss_mask.to(loss.device).view(
                            batch_rows, -1).sum(dim=1).float()
                        ce_mean = mean_over_batch_rows(
                            loss.view(batch_rows, -1), resp_counts)
                        kl_mean = None
                        if kl_beta > 0.0:
                            if self._ref_model is None:
                                raise RuntimeError(
                                    'data.kl_beta>0 but no reference model was built')
                            resp_mask = loss_mask.to(loss.device).view(
                                batch_rows, -1).bool()
                            # K3 needs only the taken token: its student logprob
                            # is already here (masked NLL), and the reference
                            # logprob needs just a logsumexp, never the [V]
                            # probabilities. Chunked over positions for peak.
                            student_logp_flat = (-loss).view(batch_rows, -1)
                            with torch.no_grad():
                                ref_logits = self._ref_model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    position_ids=position_ids,
                                    use_cache=False).logits[..., :-1, :].contiguous()
                            ref_logp_rows = []
                            seq_len = ref_logits.size(1)
                            for start in range(0, seq_len, REF_LSE_CHUNK_TOKENS):
                                end = min(start + REF_LSE_CHUNK_TOKENS, seq_len)
                                chunk = ref_logits[:, start:end, :].float()
                                taken = shift_labels.view(
                                    batch_rows, -1)[:, start:end].unsqueeze(-1)
                                ref_logp_rows.append(
                                    chunk.gather(-1, taken).squeeze(-1)
                                    - chunk.logsumexp(dim=-1))
                                del chunk
                            ref_logp_flat = torch.cat(ref_logp_rows, dim=1)
                            del ref_logits, ref_logp_rows
                            kl_tok = k3_kl_per_token(
                                student_logp_flat[resp_mask],
                                ref_logp_flat[resp_mask])
                            row_ids = torch.arange(
                                batch_rows, device=loss.device).unsqueeze(1).expand_as(
                                    resp_mask)[resp_mask]
                            kl_sums = torch.zeros(
                                batch_rows, device=loss.device).index_add_(0, row_ids, kl_tok)
                            kl_mean = mean_over_batch_rows(
                                kl_sums.unsqueeze(1), resp_counts)
                            self._last_kl_mean = float(kl_mean.detach())
                            del kl_tok
                        loss = ce_mean + (kl_beta * kl_mean if kl_mean is not None else 0.0)
                        del shift_logits, shift_labels, logits, labels
                        if do_backward:
                            (loss * loss_scale).backward()
                        return loss
                    del shift_logits, shift_labels, logits, labels
                else:
                    # IMPORTANT: We have a big assumption here, so we can shard the SAME sequence across SP ranks
                    # i.e., each GPU has <1 sequence, and each SP group has 1 sequence
                    # 1. All SP ranks will receive the *SAME* batch
                    # 2. Different SP groups will receive *DIFFERENT* batches
                    # This is implemented by the DistributedSampler

                    batch_size, seqlen = input_ids.shape
                    # Remove padding
                    input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1),
                                                               attention_mask)  # input_ids_rmpad (total_nnz, ...)
                    input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                    # Unpad position_ids to align rotary
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."),
                                                          indices).transpose(0, 1)

                    # Pad and slice inputs for sequence parallelism
                    input_ids_rmpad_sliced, position_ids_rmpad_padded, pad_size = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad, position_ids_rmpad, sp_size=get_ulysses_sequence_parallel_world_size())
                    # For computing loss
                    input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled, None, get_ulysses_sequence_parallel_world_size())
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                    # Forward pass
                    output = self.fsdp_model(
                        input_ids=input_ids_rmpad_sliced,
                        attention_mask=None,  # Not needed with flash attention varlen
                        position_ids=position_ids_rmpad_padded,
                        use_cache=False)

                    # Compute loss locally then aggregate
                    logits_rmpad = output.logits.squeeze(0)
                    input_ids_rmpad_rolled = input_ids_rmpad_rolled.to(logits_rmpad.device)
                    loss = loss_fct(logits_rmpad, input_ids_rmpad_rolled)
                    # Gather and unpad for sequence parallelism
                    loss = gather_outpus_and_unpad(loss, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                    # This is the loss collected from all ulysses ranks
                    full_loss = pad_input(hidden_states=loss.unsqueeze(-1),
                                          indices=indices,
                                          batch=batch_size,
                                          seqlen=seqlen)
                    full_loss = full_loss.squeeze(-1)[:, :-1]  # Remove last token's loss
                    full_loss = full_loss.reshape(-1)
                    loss_mask = loss_mask.to(full_loss.device)
                    loss = full_loss * loss_mask

                valid_token_this_rank = torch.sum(loss_mask)

                if self.config.data.balance_dp_token:
                    torch.distributed.all_reduce(valid_token_this_rank)
                    dp_size = self.ulysses_device_mesh.size('dp') if use_sp else torch.distributed.get_world_size()
                else:
                    dp_size = 1

                loss = torch.sum(loss) / valid_token_this_rank * dp_size

                if do_backward:
                    (loss * loss_scale).backward()
                return loss

    def _accumulate_and_clip(self, batch: TensorDict, capture_bufs=None):
        """One full gradient pass over `batch`: zero_grad, accumulate micro-batches, clip.

        Returns (step_loss, grad_norm). Leaves grads populated so the caller decides whether
        to capture them, step, or discard. `_compute_loss_and_backward` pops 'loss_mask' from
        the micro-batch it is handed, so the split is redone here on every pass -- GXPO runs
        this three times over the same batch.

        If `capture_bufs` is given, the raw grads are copied into it *before* clipping. GXPO
        needs the pre-clip gradients: clip_grad_norm_(max_norm=1.0) renormalizes every pass to
        norm 1.0 in place, which would flatten the per-coordinate retention ratio r=g1/g0 and
        the shutoff gate's g-norm statistic (both would see identical unit-norm grads).
        """
        self.optimizer.zero_grad()

        micro_batches = batch.split(self.config.data.micro_batch_size_per_gpu)
        n_micro_batches = len(micro_batches)
        step_loss = 0
        for micro_batch in micro_batches:
            loss = self._compute_loss_and_backward(
                batch=micro_batch.clone(recurse=False),
                loss_scale=1.0 / n_micro_batches) / n_micro_batches
            step_loss += loss.item()

        if capture_bufs is not None:
            self._gxpo_capture_grads(capture_bufs)

        grad_norm = self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)
        return step_loss, grad_norm

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()

        log_gpu_memory_usage('Before optimizer zero_grad', logger=logger)

        if self.gxpo_state is not None:
            step_loss, metrics = self._gxpo_training_step(batch)
        else:
            step_loss, grad_norm = self._accumulate_and_clip(batch)
            log_gpu_memory_usage('Before optimizer step', logger=logger)
            self.optimizer.step()
            log_gpu_memory_usage('After optimizer step', logger=logger)
            metrics = {'train/grad_norm': grad_norm.detach().item(), 'train/gxpo_enabled': 0.0}
        self.lr_scheduler.step()

        # reduce loss across dp ranks
        lr = self.lr_scheduler.get_last_lr()[0]

        log_gpu_memory_usage('After offload weights', logger=logger)

        step_loss = torch.tensor(step_loss).cuda()
        torch.distributed.all_reduce(step_loss, op=torch.distributed.ReduceOp.AVG)
        return {'train/loss': step_loss.detach().item(), 'train/lr(1e-3)': lr * 1e3, **metrics}

    def _gxpo_capture_grads(self, bufs):
        for p, buf in zip(self._gxpo_params, bufs):
            if p.grad is None:
                buf.zero_()
            else:
                buf.copy_(p.grad)

    @staticmethod
    def _all_ranks_flag(value: bool, device) -> bool:
        """Return a rank-consistent boolean without changing the caller's branch order."""
        flag = torch.tensor(1 if value else 0, dtype=torch.int32, device=device)
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(flag, op=torch.distributed.ReduceOp.MIN)
        return bool(flag.item())

    def _gxpo_global_g0_rms(self, gradients):
        """Return per-flat-parameter RMS values invariant to FSDP shard size."""
        if not gradients:
            return []
        norms = torch.stack([torch.linalg.vector_norm(gradient.float()) for gradient in gradients])
        counts = torch.tensor([gradient.numel() for gradient in gradients],
                              dtype=torch.float32, device=norms.device)
        stats = torch.stack((norms.square(), counts))
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        return (stats[0] / stats[1].clamp_min(1.0)).sqrt().unbind()

    def _gxpo_validate_precision_contract(self):
        """Fail on every rank if AdamW/GXPO state has fallen out of FP32."""
        if self._gxpo_precision_validated or not self._gxpo_strict_precision:
            return
        errors = []
        for index, p in enumerate(self._gxpo_params):
            if p.dtype != torch.float32:
                errors.append(f'param[{index}]={p.dtype}')
            if p.grad is not None and p.grad.dtype != torch.float32:
                errors.append(f'grad[{index}]={p.grad.dtype}')
            for name, bufs in self._gxpo_bufs.items():
                if bufs[index].dtype != torch.float32:
                    errors.append(f'{name}[{index}]={bufs[index].dtype}')
            for state_name, value in self.optimizer.state.get(p, {}).items():
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
            print('[precision] GXPO-SFT master/grads/AdamW-state/theta0/g0/g1 verified fp32')

    def _gxpo_training_step(self, batch: TensorDict):
        """GXPO 3-pass update on a supervised (cross-entropy) objective.

        Same geometry as the RL version (`dp_actor._gxpo_minibatch_step`) but with no
        importance ratio and no advantages: the identical CE loss is simply re-evaluated at
        theta0, theta1 and theta_tilde. Probe g0 at theta0 -> step -> probe g1 at theta1 ->
        step (theta2 is live) -> reposition to theta0 + alpha*scale*(theta2-theta0) -> slow
        correction. Falls back to a single standard step once the shutoff gate trips.
        """
        state = self.gxpo_state
        step_idx = state.step_count
        K, alpha, delta = state.K, state.alpha, state.delta
        min_eff = self.gxpo_min_effective_multiplier

        def standard_step():
            step_loss, grad_norm = self._accumulate_and_clip(batch)
            self.optimizer.step()
            state.step_count = step_idx + 1
            return step_loss, {'train/grad_norm': grad_norm.detach().item(), 'train/gxpo_enabled': 0.0}

        if not state.is_enabled(step_idx):
            return standard_step()

        if self._gxpo_bufs is None:
            self._gxpo_params = [p for p in self.fsdp_model.parameters() if p.requires_grad]
            self._gxpo_bufs = {n: [torch.empty_like(p) for p in self._gxpo_params] for n in ('theta0', 'g0', 'g1')}
        params = self._gxpo_params
        theta0, g0_bufs, g1_bufs = (self._gxpo_bufs[k] for k in ('theta0', 'g0', 'g1'))

        with torch.no_grad():
            for p, t0 in zip(params, theta0):
                t0.copy_(p.data)

        flag_device = torch.device('cuda', torch.cuda.current_device())
        optimizer_transaction = self._snapshot_optimizer_state(self.optimizer)
        grad_clip = float(self.config.optim.clip_grad)

        def finite(x):
            return x == x and abs(x) != float('inf')

        def restore_probe_state():
            with torch.no_grad():
                for p, t0 in zip(params, theta0):
                    p.data.copy_(t0)
            optimizer_transaction.restore()

        def probe_optimizer_step():
            try:
                self.optimizer.step()
            except BaseException:
                restore_probe_state()
                raise

        def fallback():
            # A failed probe must not leak either its parameters or optimizer state.
            restore_probe_state()
            self.optimizer.zero_grad(set_to_none=True)
            return standard_step()

        # Pass 1: g0 at theta0 (capture raw grads pre-clip -- see _accumulate_and_clip)
        try:
            _, gn0 = self._accumulate_and_clip(batch, capture_bufs=g0_bufs)
        except BaseException:
            restore_probe_state()
            raise
        gn0 = gn0.detach().item()
        clip_scale_g0 = min(1.0, grad_clip / (abs(gn0) + 1e-12))
        valid_gn0 = finite(gn0) and gn0 > 1e-8
        valid_gn0_global = self._all_ranks_flag(valid_gn0, flag_device)
        if valid_gn0_global:
            probe_optimizer_step()
            self._gxpo_validate_precision_contract()
        if not valid_gn0_global:
            return fallback()

        # Pass 2: g1 at theta_1
        try:
            _, gn1 = self._accumulate_and_clip(batch, capture_bufs=g1_bufs)
        except BaseException:
            restore_probe_state()
            raise
        gn1 = gn1.detach().item()
        clip_scale_g1 = min(1.0, grad_clip / (abs(gn1) + 1e-12))
        valid_gn1 = finite(gn1)
        valid_gn1_global = self._all_ranks_flag(valid_gn1, flag_device)
        if valid_gn1_global:
            probe_optimizer_step()
        if not valid_gn1_global:
            return fallback()

        # Retention ratio, geometric scale, reposition (theta2 is the live p.data).
        # Ratio uses the gradients that actually drove each clipped probe
        # update, with an FSDP-invariant RMS activity gate (shared
        # transactional core in verl.workers.actor.gxpo_state).
        from verl.workers.actor.gxpo_state import compute_gxpo_retention_scale
        device = theta0[0].device
        # stats: [g0_sq, g1_sq, dot01, disp2_sq, dispK_sq, sum_r, sum_r_sq, n_total,
        #         scale_sum, eff_sum]
        stats = torch.zeros(10, dtype=torch.float64, device=device)
        global_g0_rms = (
            self._gxpo_global_g0_rms(g0_bufs)
            if self._gxpo_fsdp_invariant_threshold else [None] * len(g0_bufs)
        )
        with torch.no_grad():
            for p, t0, g0b, g1b, g0_rms in zip(params, theta0, g0_bufs, g1_bufs, global_g0_rms):
                stats[0] += g0b.double().pow(2).sum()
                stats[1] += g1b.double().pow(2).sum()
                stats[2] += (g0b.double() * g1b.double()).sum()
                stats[7] += g0b.numel()

                r, scale, _active, _ratio_clipped = compute_gxpo_retention_scale(
                    g0b, g1b, K, delta, clip_scale_g0=clip_scale_g0,
                    clip_scale_g1=clip_scale_g1, g0_rms=g0_rms)
                stats[5] += r.double().sum()
                stats[6] += r.double().pow(2).sum()
                if scale.numel():
                    stats[8] += scale.double().sum()

                disp2 = (p.data - t0).float()
                stats[3] += disp2.double().pow(2).sum()
                dispK = disp2 * scale.float()
                stats[4] += dispK.double().pow(2).sum()
                # alpha*scale is the multiplier actually applied to (theta2-theta0).
                # Out-of-place: Tensor.float() returns self when scale is already fp32,
                # so an in-place mul here would corrupt the diagnostic scale above.
                eff = scale.float() * alpha
                if min_eff > 0.0:
                    eff.clamp_(min=min_eff)
                stats[9] += eff.double().sum()
                p.data.copy_(disp2.mul_(eff).add_(t0.float()))

        # Optimizer state across the two probe steps. Either way theta_tilde stands:
        #   'transactional'            -- the fast trajectory was only a probe, so every
        #                                 probe-driven AdamW mutation is rolled back to x and
        #                                 the slow correction is taken from the moments (and
        #                                 step counter) the batch started with.
        #   'transactional_fast_state' -- no refresh: the two probe steps' moments and step
        #                                 counter are kept, and the slow correction is taken
        #                                 from them. AdamW's step counter therefore advances
        #                                 3x per batch here, not 1x.
        # The snapshot taken before pass 1 stays live in both modes: it is the failure-path
        # rollback used by restore_probe_state()/fallback() when a pass goes non-finite.
        if self.gxpo_optimizer_state_mode == 'transactional':
            optimizer_transaction.restore()

        # Pass 3: slow correction at theta_tilde
        try:
            step_loss, gn_slow = self._accumulate_and_clip(batch)
        except BaseException:
            restore_probe_state()
            raise
        gn_slow = gn_slow.detach().item()
        valid_gn_slow = finite(gn_slow)
        valid_gn_slow_global = self._all_ranks_flag(valid_gn_slow, flag_device)
        if valid_gn_slow_global:
            probe_optimizer_step()
        if not valid_gn_slow_global:
            return fallback()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

        (g0_sq, g1_sq, dot01, disp2_sq, dispK_sq, sum_r, sum_r_sq, n_total,
         scale_sum, eff_sum) = stats.tolist()

        eps = 1e-12
        # gn_slow is already the global pre-clip grad norm (clip_grad_norm_ reduces across
        # FSDP shards); feed it to the shutoff gate rather than the clipped post-step grads,
        # whose norm is pinned at ~1.0 and makes the z-score degenerate.
        g0_norm, g1_norm, gslow_norm = g0_sq**0.5, g1_sq**0.5, gn_slow
        r_mean = sum_r / max(n_total, 1.0)
        r_var = max(sum_r_sq / max(n_total, 1.0) - r_mean**2, 0.0)
        disp2_norm, dispK_norm = disp2_sq**0.5, dispK_sq**0.5

        # The entropy signal gates on this step's mean response-token entropy
        # (stashed by the slow correction pass above); anything else keeps the
        # established grad-norm observation. 'grad' preserves prior SFT runs.
        trigger_signal = str(self.config.optim.get('gxpo_trigger_signal', 'grad'))
        stat_override = None
        if trigger_signal == 'entropy':
            stat_override = float(self._last_entropy_mean)
        z_score, trigger_stat, triggered = state.update_trigger_state(step=step_idx,
                                                                      g0_norm=g0_norm,
                                                                      g_slow_norm=gslow_norm,
                                                                      stat_override=stat_override)
        state.step_count = step_idx + 1

        # Contraction guard: alpha*scale < 1 means theta_tilde landed between theta0 and
        # theta2, i.e. GXPO spent three passes to make LESS progress than the two probe
        # steps already had. That silently inverts the method, so say so loudly once.
        eff_mean = eff_sum / max(n_total, 1.0)
        if eff_mean < 1.0 and not self._gxpo_contraction_warned:
            self._gxpo_contraction_warned = True
            if self.device_mesh.get_rank() == 0:
                print(f'[GXPO-SFT] WARNING at step {step_idx}: effective displacement '
                      f'multiplier alpha*scale={eff_mean:.4f} < 1 (alpha={alpha}, '
                      f'scale_mean={scale_sum / max(n_total, 1.0):.4f}, K={K}, '
                      f'r_mean={r_mean:.4f}). theta_tilde lands SHORT of theta2, so the '
                      f'3-pass update is contracting rather than extrapolating. Raise '
                      f'gxpo_alpha (>= {1.0 / max(scale_sum / max(n_total, 1.0), 1e-9):.3f} '
                      f'at this scale) or lower gxpo_k; set '
                      f'gxpo_min_effective_multiplier=1.0 to clamp instead.',
                      flush=True)

        if triggered:
            print(f'[GXPO-SFT] shutoff triggered at step {step_idx}: '
                  f'|z|={abs(z_score):.3f} >= tau={state.tau} -> single-pass SFT from now on')

        # metric names mirror the RL arm so the two runs can be plotted together
        gxpo_metrics = {
            'train/grad_norm': float(gn_slow),
            'train/gxpo_enabled': 1.0,
            'train/gxpo_optim_state_kept':
                0.0 if self.gxpo_optimizer_state_mode == 'transactional' else 1.0,
            'train/gxpo_trigger_z': float(z_score),
            'train/gxpo_trigger_stat': float(trigger_stat),
            'train/gxpo_g0_norm': g0_norm,
            'train/gxpo_g1_norm': g1_norm,
            'train/gxpo_gslow_norm': gslow_norm,
            'train/gxpo_r_mean': r_mean,
            'train/gxpo_r_std': r_var**0.5,
            'train/gxpo_scale_mean': scale_sum / max(n_total, 1.0),
            # alpha*scale as actually applied (post-clamp). >1 extrapolates past theta2,
            # <1 contracts toward theta0. This is the number to watch, not scale_mean.
            'train/gxpo_effective_multiplier': eff_mean,
            'train/gxpo_contracting': 1.0 if eff_mean < 1.0 else 0.0,
            'train/gxpo_disp2_norm': disp2_norm,
            'train/gxpo_dispK_norm': dispK_norm,
            'train/gxpo_dispK_over_disp2': dispK_norm / (disp2_norm + eps),
            'train/gxpo_cos_g0_g1': dot01 / (g0_norm * g1_norm + eps),
        }
        return step_loss, gxpo_metrics

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def save_checkpoint(self, step):
        # save checkpoint
        from torch.distributed.fsdp import FullStateDictConfig, StateDictType
        cfg = FullStateDictConfig(offload_to_cpu=True, rank0_only=True)
        with FSDP.state_dict_type(self.fsdp_model, StateDictType.FULL_STATE_DICT, cfg):
            state_dict = self.fsdp_model.state_dict()

        path = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
        # save huggingface model
        if self.device_mesh.get_rank() == 0:
            os.makedirs(path, exist_ok=True)
            self.model.save_pretrained(path, state_dict=state_dict)
            self.tokenizer.save_pretrained(path)
            if self.config.trainer.default_hdfs_dir:
                hdfs_io.makedirs(self.config.trainer.default_hdfs_dir, exist_ok=True)
                hdfs_io.copy(src=path, dst=self.config.trainer.default_hdfs_dir, dirs_exist_ok=True)
        torch.distributed.barrier()

    @staticmethod
    def _move_optimizer_state(optimizer, device):
        for state in optimizer.state.values():
            for name, value in list(state.items()):
                if torch.is_tensor(value):
                    state[name] = value.to(device)

    @staticmethod
    def _move_tensor_tree(value, device):
        if torch.is_tensor(value):
            return value.to(device)
        if isinstance(value, list):
            return [FSDPSFTTrainer._move_tensor_tree(item, device) for item in value]
        if isinstance(value, tuple):
            return tuple(FSDPSFTTrainer._move_tensor_tree(item, device) for item in value)
        if isinstance(value, dict):
            return {key: FSDPSFTTrainer._move_tensor_tree(item, device)
                    for key, item in value.items()}
        return value

    def _suspend_training_for_vllm(self):
        """Release the live FSDP model and optimizer from the GPU for vLLM."""
        if (torch.distributed.is_initialized() and
                torch.distributed.get_world_size() != 1):
            raise RuntimeError(
                'SFT benchmark vLLM evaluation currently requires world_size=1; '
                'multi-rank model suspension is not supported safely.')
        self.optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        self._vllm_training_device = torch.device('cuda', torch.cuda.current_device())
        self._move_optimizer_state(self.optimizer, torch.device('cpu'))
        self.fsdp_model.cpu()
        if self._gxpo_bufs is not None:
            self._gxpo_bufs = self._move_tensor_tree(self._gxpo_bufs, torch.device('cpu'))
        gc.collect()
        torch.cuda.empty_cache()

    def _resume_training_after_vllm(self):
        self.fsdp_model.to(self._vllm_training_device)
        if self._gxpo_bufs is not None:
            self._gxpo_bufs = self._move_tensor_tree(
                self._gxpo_bufs, self._vllm_training_device)
        self._move_optimizer_state(self.optimizer, self._vllm_training_device)
        torch.cuda.empty_cache()

    def _run_vllm_evaluation(self, checkpoint, tracking, global_step, rank):
        """Run the unified six-benchmark vLLM evaluator at a saved step.

        The evaluator runs synchronously after checkpointing. The live FSDP
        model, optimizer state, and GXPO buffers are moved to CPU first so the
        subprocess can own the assigned GPU without an unsafe VRAM overlap.
        """
        every = int(self.config.trainer.get('benchmark_eval_freq', 0))
        if every <= 0 or global_step % every != 0:
            return

        self._suspend_training_for_vllm()
        try:
            if rank == 0:
                code_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
                script = os.environ.get(
                    'SFT_VLLM_EVAL_SCRIPT',
                    os.path.join(code_root, 'train-scripts', 'eval_sft_6bench.sh'))
                output_root = os.environ.get(
                    'SFT_VLLM_EVAL_OUTPUT_ROOT',
                    os.path.join(self.config.trainer.default_local_dir, 'vllm_eval'))
                output_dir = os.path.join(output_root, f'global_step_{global_step}')
                os.makedirs(output_dir, exist_ok=True)

                env = os.environ.copy()
                # The trainer process itself was launched by torchrun/torchelastic, whose
                # rendezvous variables (MASTER_ADDR/PORT, RANK, WORLD_SIZE, TORCHELASTIC_*)
                # are inherited via os.environ.copy(). vLLM also initializes its own
                # torch.distributed process group internally, and if it sees these vars it
                # tries to attach to the *live* trainer's elastic-agent store instead of
                # creating its own -- that handshake never completes and vLLM hangs forever
                # at process-group init (looks like a GPU-memory stall but never allocates
                # any KV cache). Strip them so vLLM always gets an isolated, fresh group.
                for key in list(env):
                    if key in ('MASTER_ADDR', 'MASTER_PORT', 'RANK', 'LOCAL_RANK',
                               'WORLD_SIZE', 'LOCAL_WORLD_SIZE', 'GROUP_RANK',
                               'GROUP_WORLD_SIZE', 'ROLE_RANK', 'ROLE_WORLD_SIZE',
                               'ROLE_NAME') or key.startswith('TORCHELASTIC_'):
                        del env[key]
                env['EVAL_OUTPUT_DIR'] = output_dir
                env['EVAL_RESPONSE_LENGTH'] = os.environ.get(
                    'SFT_VLLM_EVAL_MAX_TOKENS', '3072')
                env['EVAL_MAX_MODEL_LEN'] = os.environ.get(
                    'SFT_VLLM_EVAL_MAX_MODEL_LEN', '4096')
                env['EVAL_MAX_NUM_SEQS'] = os.environ.get(
                    'SFT_VLLM_EVAL_MAX_NUM_SEQS', '256')
                env['EVAL_GPU_UTIL'] = os.environ.get(
                    'SFT_VLLM_EVAL_GPU_UTIL', '0.85')
                env['EVAL_TEMP'] = os.environ.get(
                    'SFT_VLLM_EVAL_TEMP', '0.7')
                env['EVAL_STEP'] = str(global_step)
                env['PYTHONPATH'] = code_root + os.pathsep + env.get('PYTHONPATH', '')
                print(f'[SFT-VLLM] step {global_step}: evaluating {checkpoint}', flush=True)
                subprocess.run(['bash', script, checkpoint], cwd=code_root, env=env, check=True)

                # Sampled-eval shape (n samples x seeds) is configurable so cheap
                # single-sample cadence runs (n=1, one seed) read back the files
                # the harness actually writes; defaults preserve the SFT-pair
                # convention (n=8, three seeds). The greedy pass can be skipped
                # entirely (eval_skip_greedy) for sampled-only cadence evals.
                sample_n = int(self.config.trainer.get('eval_sample_n', 8))
                seed_count = int(self.config.trainer.get('eval_seed_count', 3))
                skip_greedy = bool(int(self.config.trainer.get('eval_skip_greedy', 0)))
                greedy_path = os.path.join(output_dir, f'sft_greedy_{seed_count}seed.json')
                sampled_path = os.path.join(
                    output_dir, f'sft_sampled_{sample_n}_{seed_count}seed.json')
                missing = []
                if not skip_greedy and not os.path.isfile(greedy_path):
                    missing.append(greedy_path)
                if not os.path.isfile(sampled_path):
                    missing.append(sampled_path)
                if missing:
                    raise RuntimeError(
                        f'[SFT-VLLM] evaluator did not write expected JSON files: {missing}')
                greedy = {}
                if not skip_greedy:
                    with open(greedy_path) as handle:
                        greedy = json.load(handle)
                with open(sampled_path) as handle:
                    sampled = json.load(handle)
                metrics = {}
                benchmark_order = ('math500', 'aime24', 'aime25', 'amc23', 'minerva', 'olympiadbench')
                for key in benchmark_order:
                    if key in greedy.get('benchmarks', {}):
                        metrics[f'eval_greedy/{key}_pass1'] = float(
                            greedy['benchmarks'][key]['mean'])
                    if key in sampled.get('benchmarks', {}):
                        bench = sampled['benchmarks'][key]
                        metrics[f'eval_sampled/{key}_pass{sample_n}'] = float(
                            bench['pass_at_n']['mean'])
                        metrics[f'eval_sampled/{key}_average{sample_n}'] = float(
                            bench['average_at_n']['mean'])
                if 'avg_pass1' in greedy.get('benchmarks', {}):
                    metrics['eval_greedy/avg_pass1'] = float(
                        greedy['benchmarks']['avg_pass1']['mean'])
                if 'macro' in sampled.get('benchmarks', {}):
                    metrics[f'eval_sampled/avg_pass{sample_n}'] = float(
                        sampled['benchmarks']['macro']['pass_at_n']['mean'])
                    metrics[f'eval_sampled/avg_average{sample_n}'] = float(
                        sampled['benchmarks']['macro']['average_at_n']['mean'])
                if not skip_greedy:
                    metrics['eval_greedy/benchmark_count'] = float(
                        sum(key in greedy.get('benchmarks', {}) for key in benchmark_order))
                if tracking is not None:
                    tracking.log(data=metrics, step=global_step)
                print(f'[SFT-VLLM] step {global_step}: logged {len(metrics)} metrics', flush=True)

                # Best-checkpoint tracks greedy avg when available, else the
                # sampled average (sampled-only cadence runs have no greedy).
                best_score = metrics.get(
                    'eval_greedy/avg_pass1',
                    metrics.get(f'eval_sampled/avg_average{sample_n}'))
                self._update_best_checkpoint(checkpoint, global_step, best_score)
        finally:
            self._resume_training_after_vllm()

        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _update_best_checkpoint(self, checkpoint, global_step, score):
        """Copy `checkpoint` to a separate best_checkpoint/ dir if `score` beats
        every checkpoint seen so far. The caller passes eval_greedy/avg_pass1
        when the greedy pass ran, else the sampled average. Mirrors the
        best_checkpoint/best_ckpt.json convention used by the RL trainer
        (ray_trainer.py) so both trainers are inspected the same way. Rank-0 only;
        called from inside the existing rank==0 evaluation block.
        """
        if score is None:
            return
        root = self.config.trainer.default_local_dir
        if score > self._best_eval_score:
            print(f'[best-ckpt] step {global_step}: new best avg_pass1 {score:.4f} '
                  f'(prev {self._best_eval_score:.4f}); saving', flush=True)
            self._best_eval_score = score
            self._best_eval_step = global_step
            best_dir = os.path.join(root, 'best_checkpoint')
            shutil.rmtree(best_dir, ignore_errors=True)
            shutil.copytree(checkpoint, best_dir)
            with open(os.path.join(root, 'best_ckpt.json'), 'w') as f:
                json.dump({
                    'best_step': global_step,
                    'best_score': score,
                    'metric': 'eval_greedy/avg_pass1',
                    'path': 'best_checkpoint',
                }, f)
        else:
            print(f'[best-ckpt] step {global_step}: avg_pass1 {score:.4f} <= '
                  f'best {self._best_eval_score:.4f} (from step {self._best_eval_step})',
                  flush=True)

    def _prune_old_checkpoints(self, current_checkpoint_name, rank):
        """Keep exactly one latest global_step_N directory plus the separate
        best_checkpoint (never touched here, since it doesn't match the
        global_step_ prefix). Mirrors the RL trainer's retention contract.
        """
        if rank == 0:
            root = self.config.trainer.default_local_dir
            for name in os.listdir(root):
                if name.startswith('global_step_') and name != current_checkpoint_name:
                    path = os.path.join(root, name)
                    if os.path.isdir(path):
                        print(f'Removing old checkpoint directory: {path}', flush=True)
                        shutil.rmtree(path, ignore_errors=True)
        if torch.distributed.is_initialized():
            torch.distributed.barrier()

    def _run_validation(self, tracking, global_step, rank):
        """Run optional validation loss evaluation when a validation dataset exists."""
        if self.val_dataloader is None:
            return
        max_batches = self.config.trainer.get('val_max_batches', 0)
        val_losses = []
        for i, val_data in enumerate(self.val_dataloader):
            if max_batches and i >= max_batches:
                break
            val_data = TensorDict(val_data, batch_size=self.config.data.micro_batch_size_per_gpu).cuda()
            val_losses.append(self.validation_step(val_data))
        if rank == 0:
            avg_val_loss = torch.mean(torch.stack(val_losses))
            tracking.log(data={'val/loss': avg_val_loss.detach().item()}, step=global_step)
        torch.distributed.barrier()

    def fit(self):
        rank = self.device_mesh.get_rank()

        # TODO: add a unified tracking
        # tracking exists on all ranks (None off rank 0): epoch-end
        # _run_validation runs collectively on every rank and only touches
        # tracking on rank 0. Without this, rank>0 dies with UnboundLocalError
        # at the first epoch boundary.
        tracking = None
        if rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger,
                                config=OmegaConf.to_container(self.config, resolve=True))

        global_step = 0
        # Use the same cap that sized the optimizer scheduler. A cap must not
        # leave the cosine schedule decaying over the unreachable full epoch run.
        total_training_steps = self.total_steps
        self.total_training_steps = total_training_steps
        benchmark_eval_freq = int(self.config.trainer.get('benchmark_eval_freq', 0))
        print(f'Total training steps: {self.total_training_steps}')
        if benchmark_eval_freq > 0 and self.device_mesh.size(0) != 1:
            raise ValueError(
                'trainer.benchmark_eval_freq requires a single FSDP process/GPU; '
                f'got world size {self.device_mesh.size(0)}')

        # TODO (zhangchi.usc1992) add back checkpoint manager. Currently, it blocks when uploading to hdfs. So very slow.

        for epoch in range(self.config.trainer.total_epochs):
            self.train_sampler.set_epoch(epoch=epoch)
            for data in tqdm(self.train_dataloader,
                             total=self.steps_per_epoch,
                             desc=f"Epoch {epoch+1}/{self.config.trainer.total_epochs}"):
                global_step += 1
                data = TensorDict(data, batch_size=self.config.data.train_batch_size).cuda()
                metric = self.training_step(data)
                if rank == 0:
                    tracking.log(data=metric, step=global_step)

                # periodic validation / checkpointing (the defaults keep the old
                # validate-and-save-at-epoch-end-only behaviour)
                test_freq = self.config.trainer.get('test_freq', 0)
                save_freq = self.config.trainer.get('save_freq', 0)
                if test_freq > 0 and global_step % test_freq == 0 and global_step < self.total_training_steps:
                    self._run_validation(tracking, global_step, rank)
                eval_due = (benchmark_eval_freq > 0 and
                            global_step % benchmark_eval_freq == 0 and
                            global_step < self.total_training_steps)
                save_due = (save_freq > 0 and
                            global_step % save_freq == 0 and
                            global_step < self.total_training_steps)
                if eval_due:
                    del data
                    torch.cuda.empty_cache()
                    checkpoint_name = f'global_step_{global_step}'
                    checkpoint = os.path.join(
                        self.config.trainer.default_local_dir, checkpoint_name)
                    self.save_checkpoint(step=global_step)
                    self._run_vllm_evaluation(checkpoint, tracking, global_step, rank)
                    # Retention: only the current global_step_N and the separate
                    # best_checkpoint (updated inside _run_vllm_evaluation) survive.
                    self._prune_old_checkpoints(checkpoint_name, rank)
                elif save_due:
                    checkpoint_name = f'global_step_{global_step}'
                    self.save_checkpoint(step=global_step)
                    self._prune_old_checkpoints(checkpoint_name, rank)

                # for early exit validation
                if global_step >= self.total_training_steps:
                    # Perform final validation
                    self._run_validation(tracking, global_step, rank)

                    # Save final checkpoint and evaluate it when the cadence lands here.
                    checkpoint_name = f'global_step_{global_step}'
                    checkpoint = os.path.join(
                        self.config.trainer.default_local_dir, checkpoint_name)
                    self.save_checkpoint(step=global_step)
                    if (benchmark_eval_freq > 0 and
                            global_step % benchmark_eval_freq == 0):
                        self._run_vllm_evaluation(checkpoint, tracking, global_step, rank)
                    self._prune_old_checkpoints(checkpoint_name, rank)
                    return

            # validation
            self._run_validation(tracking, global_step, rank)

            # save checkpoint
            self.save_checkpoint(step=global_step)


from verl.trainer.fsdp_sft_trainer import FSDPSFTTrainer
import hydra

from torch.distributed.device_mesh import init_device_mesh

from verl.utils.distributed import initialize_global_process_group


@hydra.main(config_path='config', config_name='sft_trainer', version_base=None)
def main(config):
    local_rank, rank, world_size = initialize_global_process_group()

    device_mesh = init_device_mesh(device_type='cuda', mesh_shape=(world_size,), mesh_dim_names=('fsdp',))
    dp_size = world_size // config.ulysses_sequence_parallel_size
    ulysses_device_mesh = init_device_mesh(device_type='cuda',
                                           mesh_shape=(dp_size, config.ulysses_sequence_parallel_size),
                                           mesh_dim_names=('dp', 'sp'))
    trainer = FSDPSFTTrainer(config=config, device_mesh=device_mesh, ulysses_device_mesh=ulysses_device_mesh)
    trainer.fit()


if __name__ == '__main__':
    main()
