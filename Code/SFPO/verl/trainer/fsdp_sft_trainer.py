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

import json
import logging
import math
import re
import shutil
import subprocess
import sys
import time
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
try:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
    _FLASH_ATTN_AVAILABLE = True
except (ImportError, OSError):
    # Standard SFT does not need flash-attn; keep the import optional so the
    # eager-attention fallback can run on environments without its CUDA ABI.
    _FLASH_ATTN_AVAILABLE = False
    pad_input = unpad_input = rearrange = index_first_axis = None

from verl.utils.fsdp_utils import (
    get_fsdp_wrap_policy, init_fn, get_init_weight_context_manager,
    offload_fsdp_model_to_cpu, load_fsdp_model_to_gpu,
    offload_fsdp_optimizer, load_fsdp_optimizer,
)
from verl.utils.dataset import SFTDataset
from verl.utils.fs import copy_to_local
from verl.utils.tracking import Tracking
from verl.utils.ulysses import get_ulysses_sequence_parallel_world_size, set_ulysses_sequence_parallel_group
from torch.distributed.device_mesh import DeviceMesh

import verl.utils.hdfs_io as hdfs_io
from verl.utils.debug import log_gpu_memory_usage
from verl.trainer.sft_utils import resolve_total_training_steps
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
        self.fsdp_strategy = str(self.config.model.fsdp_config.get('strategy', 'fsdp1')).lower()
        if self.fsdp_strategy not in {'fsdp1', 'fsdp2'}:
            raise ValueError(f'Unsupported SFT FSDP strategy: {self.fsdp_strategy!r}')
        self._is_fsdp2 = self.fsdp_strategy == 'fsdp2'

        # Set sequence parallel size
        self.config.ulysses_sequence_parallel_size = getattr(self.config, 'ulysses_sequence_parallel_size', 1)
        self.use_remove_padding = getattr(self.config, 'use_remove_padding', False)
        if self.device_mesh.get_rank() == 0:
            print(f'Using sequence parallel size: {self.config.ulysses_sequence_parallel_size}')
            print(f'Using remove padding: {self.use_remove_padding}')

        self._build_dataloader()
        # build model
        self._build_model_optimizer()

        # GXPO-style update on the supervised objective: shutoff gate + lazily allocated
        # per-parameter buffers. Absent `optim.use_gxpo` keeps the plain 1-pass SFT path.
        self._backward_calls = 0
        self._cumulative_train_time = 0.0
        self._cumulative_tokens = 0
        self.gxpo_state = None
        self._gxpo_bufs = None
        self._best_eval_score = float('-inf')
        self._best_eval_step = None
        if self.config.optim.get('use_gxpo', False):
            from verl.workers.actor.gxpo_state import GXPOState
            self.gxpo_state = GXPOState(
                K=self.config.optim.get('gxpo_k', 5),
                alpha=self.config.optim.get('gxpo_alpha', 0.5),
                delta=self.config.optim.get('gxpo_delta', 1e-8),
                tau=self.config.optim.get('gxpo_tau', 0.5),
                omega=self.config.optim.get('gxpo_omega', 0.1),
                zscore_w=self.config.optim.get('gxpo_zscore_w', 30),
                shutoff_mode=self.config.optim.get('gxpo_shutoff_mode', 'trajectory_aware'),
                warmup_steps=self.config.optim.get('gxpo_warmup', 0),
                trigger_patience=self.config.optim.get("gxpo_trigger_patience", 1),
                trigger_robust=self.config.optim.get("gxpo_trigger_robust", False),
                min_post_warmup_obs=self.config.optim.get("gxpo_min_post_warmup_obs", 0),
                max_active_steps=self.config.optim.get("gxpo_max_active_steps", 0),
                abs_threshold=self.config.optim.get("gxpo_abs_threshold", 0.0),
                sustain_window=self.config.optim.get("gxpo_sustain_window", 10),
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
        # build dataset
        self.train_dataset = SFTDataset(parquet_files=config.data.train_files,
                                        tokenizer=self.tokenizer,
                                        prompt_key=config.data.prompt_key,
                                        prompt_dict_keys=config.data.get('prompt_dict_keys', None),
                                        response_key=config.data.response_key,
                                        response_dict_keys=config.data.get('response_dict_keys', None),
                                        max_length=config.data.max_length,
                                        truncation=config.data.truncation)
        self.val_dataset = SFTDataset(parquet_files=config.data.val_files,
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


    def _wrap_fsdp2_model(self):
        """Apply PyTorch composable FSDP2 to transformer blocks and the root model."""
        from torch.distributed._composable.fsdp import (
            CPUOffloadPolicy,
            MixedPrecisionPolicy,
            OffloadPolicy,
            fully_shard,
        )

        fsdp_config = self.config.model.fsdp_config
        mp_policy = MixedPrecisionPolicy(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.float32,
            output_dtype=torch.bfloat16,
            cast_forward_inputs=True,
        )
        offload_policy = (
            CPUOffloadPolicy()
            if fsdp_config.get('cpu_offload', False)
            else OffloadPolicy()
        )
        layer_names = set(getattr(self.model, '_no_split_modules', None) or ())
        wrapped = 0
        # FSDP2 requires child modules to be sharded before their parent.
        for module in self.model.modules():
            if module is self.model:
                continue
            if module.__class__.__name__ in layer_names:
                fully_shard(
                    module,
                    mesh=self.device_mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                    reshard_after_forward=True,
                )
                wrapped += 1
        fully_shard(
            self.model,
            mesh=self.device_mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            reshard_after_forward=True,
        )
        self.fsdp_model = self.model
        if self.device_mesh.get_rank() == 0:
            print(f'FSDP2 fully_shard wrapped {wrapped} transformer blocks; '
                  f'cpu_offload={fsdp_config.get("cpu_offload", False)}')

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
                                                                               torch_dtype=torch.bfloat16,
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

        if self._is_fsdp2:
            self._wrap_fsdp2_model()
        else:
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

        log_gpu_memory_usage('After initialize optimizer', logger=logger)

        self.steps_per_epoch = len(self.train_dataloader)
        self.total_steps = resolve_total_training_steps(
            self.steps_per_epoch,
            self.config.trainer.total_epochs,
            self.config.trainer.get('total_training_steps', None),
        )

        if self.device_mesh.get_rank() == 0:
            print(
                f'Number of steps/epoch {self.steps_per_epoch}, number of epochs {self.config.trainer.total_epochs}, total number of steps {self.total_steps}'
            )

        num_warmup_steps = int(self.total_steps * self.config.optim.warmup_steps_ratio)

        self.lr_scheduler = get_cosine_schedule_with_warmup(optimizer=self.optimizer,
                                                            num_warmup_steps=num_warmup_steps,
                                                            num_training_steps=self.total_steps)

    def _compute_loss_and_backward(self, batch, do_backward=True):
        """Compute loss with optional sequence parallelism and remove padding features"""
        use_sp = self.use_remove_padding and self.config.ulysses_sequence_parallel_size > 1
        if use_sp and not _FLASH_ATTN_AVAILABLE:
            raise RuntimeError("Sequence-parallel SFT requires a compatible flash-attn installation")

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
                    shift_labels = labels.contiguous()
                    # Flatten the tokens
                    shift_logits = shift_logits.view(-1, self.model.config.vocab_size)
                    shift_labels = shift_labels.view(-1)
                    # Enable model parallelism
                    shift_labels = shift_labels.to(shift_logits.device)
                    loss = loss_fct(shift_logits, shift_labels)
                    loss = loss * loss_mask.to(loss.device)
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
                    loss.backward()
                    self._backward_calls += 1
                return loss


    def _clip_grad_norm(self):
        if not self._is_fsdp2:
            return self.fsdp_model.clip_grad_norm_(max_norm=self.config.optim.clip_grad)

        # FSDP2 CPUOffloadPolicy exposes gradients as CPU-backed DTensors.
        # Compute on local shards, then reduce only scalar statistics on CUDA;
        # applying arithmetic directly to a CPU DTensor would dispatch an
        # unsupported CPU collective with NCCL.
        total_sq = 0.0
        local_grads = []
        for parameter in self.fsdp_model.parameters():
            grad = parameter.grad
            if grad is None:
                continue
            local = grad.to_local() if hasattr(grad, "to_local") else grad
            local_grads.append(local)
            total_sq += float(local.detach().float().pow(2).sum().item())
        norm = torch.tensor(total_sq, dtype=torch.float64, device=torch.cuda.current_device())
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(norm, op=torch.distributed.ReduceOp.SUM)
        total_norm = norm.sqrt()
        max_norm = float(self.config.optim.clip_grad)
        clip_coef = min(1.0, max_norm / (float(total_norm.item()) + 1e-6))
        if clip_coef < 1.0:
            for local in local_grads:
                local.mul_(clip_coef)
        return total_norm.to(dtype=torch.float32)

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
            loss = self._compute_loss_and_backward(batch=micro_batch.clone(recurse=False)) / n_micro_batches
            step_loss += loss.item()

        if capture_bufs is not None:
            self._gxpo_capture_grads(capture_bufs)

        grad_norm = self._clip_grad_norm()
        return step_loss, grad_norm

    def training_step(self, batch: TensorDict):
        self.fsdp_model.train()
        step_start = time.perf_counter()
        backward_start = self._backward_calls
        token_count = int(batch['loss_mask'].sum().item())

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
        token_count_tensor = torch.tensor(float(token_count), device=step_loss.device)
        torch.distributed.all_reduce(token_count_tensor, op=torch.distributed.ReduceOp.SUM)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - step_start
        self._cumulative_train_time += elapsed
        self._cumulative_tokens += int(token_count_tensor.item())
        backward_calls = self._backward_calls - backward_start
        metrics.update({
            'eff/step_time_s': elapsed,
            'eff/cumulative_train_time_s': self._cumulative_train_time,
            'eff/tokens_step': int(token_count_tensor.item()),
            'eff/cumulative_tokens': self._cumulative_tokens,
            'eff/backward_calls_step': backward_calls,
            'eff/cumulative_backward_calls': self._backward_calls,
        })
        return {'train/loss': step_loss.detach().item(), 'train/lr(1e-3)': lr * 1e3, **metrics}

    def _gxpo_capture_grads(self, bufs):
        for p, buf in zip(self._gxpo_params, bufs):
            if p.grad is None:
                buf.zero_()
            else:
                grad = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                buf.copy_(grad)

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

        def standard_step():
            step_loss, grad_norm = self._accumulate_and_clip(batch)
            self.optimizer.step()
            state.step_count = step_idx + 1
            return step_loss, {
                'train/grad_norm': grad_norm.detach().item(),
                'train/gxpo_enabled': 0.0,
                'train/gxpo_budget_stop': float(state.budget_stop is True),
            }

        if not state.is_enabled(step_idx):
            return standard_step()

        if self._gxpo_bufs is None:
            self._gxpo_params = [p for p in self.fsdp_model.parameters() if p.requires_grad]
            self._gxpo_bufs = {n: [torch.empty_like(p.to_local() if hasattr(p, "to_local") else p) for p in self._gxpo_params] for n in ('theta0', 'g0', 'g1')}
        params = self._gxpo_params
        theta0, g0_bufs, g1_bufs = (self._gxpo_bufs[k] for k in ('theta0', 'g0', 'g1'))

        with torch.no_grad():
            for p, t0 in zip(params, theta0):
                t0.copy_(p.data.to_local() if hasattr(p.data, "to_local") else p.data)

        def finite(x):
            return x == x and abs(x) != float('inf')

        def fallback():
            with torch.no_grad():
                for p, t0 in zip(params, theta0):
                    (p.data.to_local() if hasattr(p.data, "to_local") else p.data).copy_(t0)
            self.optimizer.zero_grad(set_to_none=True)
            return standard_step()

        # Pass 1: g0 at theta0 (capture raw grads pre-clip -- see _accumulate_and_clip)
        _, gn0 = self._accumulate_and_clip(batch, capture_bufs=g0_bufs)
        gn0 = gn0.detach().item()
        if not finite(gn0) or gn0 <= 1e-8:
            return fallback()
        self.optimizer.step()

        # Pass 2: g1 at theta_1
        _, gn1 = self._accumulate_and_clip(batch, capture_bufs=g1_bufs)
        gn1 = gn1.detach().item()
        if not finite(gn1):
            return fallback()
        self.optimizer.step()

        # Retention ratio, geometric scale, reposition on local FSDP2 shards.
        # Keep all vector math on CPU locals and reduce only scalar statistics on CUDA.
        local_stats = [0.0] * 9
        with torch.no_grad():
            for p, t0, g0b, g1b in zip(params, theta0, g0_bufs, g1_bufs):
                p_local = p.data.to_local() if hasattr(p.data, "to_local") else p.data
                g0d = g0b.float()
                g1d = g1b.float()
                local_stats[0] += float(g0d.pow(2).sum().item())
                local_stats[1] += float(g1d.pow(2).sum().item())
                local_stats[2] += float((g0d * g1d).sum().item())
                local_stats[7] += g0b.numel()

                sgn = torch.where(g0b >= 0, 1.0, -1.0)
                r = g1b / (g0b.abs().clamp(min=delta) * sgn)
                r.clamp_(-2.0, 3.0).nan_to_num_(nan=1.0)
                local_stats[5] += float(r.float().sum().item())
                local_stats[6] += float(r.float().pow(2).sum().item())

                one_minus_r = 1.0 - r
                s_k = (1.0 - r.pow(K)) / (one_minus_r + delta)
                s_2 = (1.0 - r * r) / (one_minus_r + delta)
                scale = (s_k / (s_2 + delta)).clamp_(1.0, K / 2.0 + 1.0)
                local_stats[8] += float(scale.float().sum().item())

                disp2 = p_local - t0
                local_stats[3] += float(disp2.float().pow(2).sum().item())
                disp_k = disp2 * scale
                local_stats[4] += float(disp_k.float().pow(2).sum().item())
                p_local.copy_(disp_k.mul(alpha).add_(t0))

        device = torch.device("cuda", torch.cuda.current_device())
        stats = torch.tensor(local_stats, dtype=torch.float64, device=device)

        # Pass 3: slow correction at theta_tilde. The raw gradient is not
        # copied into another model-sized buffer: after clipping, a common
        # positive scale does not change cosine direction.
        step_loss, gn_slow = self._accumulate_and_clip(batch)
        local_slow = [0.0, 0.0]
        with torch.no_grad():
            for p, g0b in zip(params, g0_bufs):
                if p.grad is not None:
                    grad = p.grad.to_local() if hasattr(p.grad, "to_local") else p.grad
                    gradd = grad.float()
                    local_slow[0] += float((g0b.float() * gradd).sum().item())
                    local_slow[1] += float(gradd.float().pow(2).sum().item())
        slow_dot_stats = torch.tensor(local_slow, dtype=torch.float64, device=device)
        gn_slow = gn_slow.detach().item()
        if not finite(gn_slow):
            return fallback()
        self.optimizer.step()

        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(slow_dot_stats, op=torch.distributed.ReduceOp.SUM)

        (g0_sq, g1_sq, dot01, disp2_sq, dispK_sq, sum_r, sum_r_sq, n_total,
         scale_sum) = stats.tolist()
        dot0slow, slow_sq = slow_dot_stats.tolist()

        eps = 1e-12
        # gn_slow is already the global pre-clip grad norm (clip_grad_norm_ reduces across
        # FSDP shards); feed it to the shutoff gate rather than the clipped post-step grads,
        # whose norm is pinned at ~1.0 and makes the z-score degenerate.
        g0_norm, g1_norm, gslow_norm = g0_sq**0.5, g1_sq**0.5, gn_slow
        r_mean = sum_r / max(n_total, 1.0)
        r_var = max(sum_r_sq / max(n_total, 1.0) - r_mean**2, 0.0)
        disp2_norm, dispK_norm = disp2_sq**0.5, dispK_sq**0.5

        cosine_g0_gslow = dot0slow / (g0_norm * (slow_sq ** 0.5) + eps)
        disagreement = 1.0 - abs(cosine_g0_gslow)
        z_score, trigger_stat, triggered = state.update_trigger_state(
            step=step_idx, g0_norm=g0_norm, g_slow_norm=gslow_norm,
            stat_override=disagreement)
        state.step_count = step_idx + 1

        if triggered:
            print(f'[GXPO-SFT] shutoff triggered at step {step_idx}: '
                  f'z={z_score:.3f} >= tau={state.tau} -> single-pass SFT from now on')

        # metric names mirror the RL arm so the two runs can be plotted together
        return step_loss, {
            'train/grad_norm': float(gn_slow),
            'train/gxpo_enabled': 1.0,
            'train/gxpo_trigger_z': float(z_score),
            'train/gxpo_trigger_stat': float(trigger_stat),
            'train/gxpo_disagreement': float(disagreement),
            'train/gxpo_g0_norm': g0_norm,
            'train/gxpo_g1_norm': g1_norm,
            'train/gxpo_gslow_norm': gslow_norm,
            'train/gxpo_r_mean': r_mean,
            'train/gxpo_r_std': r_var**0.5,
            'train/gxpo_scale_mean': scale_sum / max(n_total, 1.0),
            'train/gxpo_disp2_norm': disp2_norm,
            'train/gxpo_dispK_norm': dispK_norm,
            'train/gxpo_dispK_over_disp2': dispK_norm / (disp2_norm + eps),
            'train/gxpo_cos_g0_g1': dot01 / (g0_norm * g1_norm + eps),
            'train/gxpo_cos_g0_gslow': float(cosine_g0_gslow),
        }

    def validation_step(self, batch: TensorDict):
        self.fsdp_model.eval()
        with torch.no_grad():
            loss = self._compute_loss_and_backward(batch, do_backward=False)
            torch.distributed.all_reduce(loss, op=torch.distributed.ReduceOp.AVG)
        return loss

    def save_checkpoint(self, step):
        # Save a plain Hugging Face checkpoint for vLLM and downstream evaluation.
        if self._is_fsdp2:
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions, get_model_state_dict,
            )
            state_dict = get_model_state_dict(
                self.fsdp_model,
                options=StateDictOptions(full_state_dict=True, cpu_offload=True),
            )
        else:
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

    def _run_greedy_eval(self, global_step, rank):
        """Evaluate a saved SFT checkpoint through the repository vLLM evaluator.

        Saving first is intentional: vLLM needs a plain Hugging Face checkpoint,
        while the live trainer owns the FSDP model.  Rank zero launches the
        single-GPU evaluator; all ranks participate in the surrounding
        checkpoint/barrier calls so this remains safe for a future multi-rank
        launch.
        """
        root = self.config.trainer.get('eval_benchmark_root', None)
        if not root:
            raise RuntimeError('eval_benchmark_root is required for greedy SFT evaluation')

        run_dir = os.path.abspath(self.config.trainer.default_local_dir)
        self.save_checkpoint(step=global_step)

        # With legacy FSDP1, release model and optimizer CUDA storage while
        # the synchronous vLLM evaluator owns this GPU. FSDP2 uses its native
        # CPUOffloadPolicy when configured, so its runtime state is already
        # offloaded between operations.
        legacy_eval_offload = not self._is_fsdp2
        if legacy_eval_offload:
            offload_fsdp_model_to_cpu(self.fsdp_model)
            offload_fsdp_optimizer(self.optimizer)
            torch.cuda.empty_cache()
        torch.distributed.barrier()

        eval_kind = str(self.config.trainer.get('eval_kind', 'math'))
        evaluator_name = (
            'evaluate_knights_and_knaves_sft.py'
            if eval_kind == 'knights_and_knaves'
            else 'evaluate_sft_terminal.py'
        )
        if eval_kind not in {'math', 'knights_and_knaves'}:
            raise RuntimeError(f'unsupported SFT eval_kind={eval_kind!r}')
        evaluator = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '..', '..', 'tools', evaluator_name)
        )
        result_path = os.path.join(run_dir, 'final_sft_eval.json')
        eval_command = [
            sys.executable,
            evaluator,
            '--run-dir', run_dir,
            '--step', str(global_step),
            '--data-root', os.path.abspath(root),
            '--seeds', '0',
            '--n', '1',
            '--temperature', '0.0',
            '--top-p', '1.0',
            '--max-tokens', str(self.config.trainer.get('eval_greedy_max_new_tokens', 3072)),
            '--max-examples', str(self.config.trainer.get('eval_greedy_max_examples', 0)),
            '--prompt-length', str(self.config.trainer.get('eval_greedy_prompt_max_length', 2048)),
            '--gpu-memory-utilization',
            str(self.config.trainer.get('eval_greedy_vllm_gpu_memory_utilization', 0.18)),
            '--tensor-parallel-size', '1',
        ]
        try:
            if rank == 0:
                print('[SFT greedy] launching vLLM evaluator: ' + ' '.join(eval_command), flush=True)
                # The evaluator is launched from inside torchrun. Do not let it
                # inherit torchrun process-group coordinates: vLLM creates its
                # own single-process engine and otherwise may attach to the
                # trainer MASTER_ADDR/MASTER_PORT and wait indefinitely.
                eval_env = os.environ.copy()
                # vLLM is launched after the trainer has initialized CUDA. Force
                # its child engine to spawn instead of forking that CUDA context.
                eval_env.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
                for key in (
                    "MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE",
                    "LOCAL_RANK", "LOCAL_WORLD_SIZE", "GROUP_RANK",
                    "ROLE_RANK", "ROLE_WORLD_SIZE", "TORCHELASTIC_RUN_ID",
                    "TORCHELASTIC_RESTART_COUNT", "TORCHELASTIC_MAX_RESTARTS",
                    "TORCHELASTIC_ERROR_FILE",
                    "GROUP_WORLD_SIZE", "ROLE_NAME", "TORCHELASTIC_USE_AGENT_STORE",
                ):
                    eval_env.pop(key, None)
                timeout_s = int(self.config.trainer.get('eval_greedy_timeout_s', 900))
                try:
                    subprocess.run(
                        eval_command, check=True, cwd=os.path.dirname(evaluator),
                        env=eval_env, start_new_session=True, timeout=(timeout_s if timeout_s > 0 else None),
                    )
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError(
                        f'SFT vLLM evaluation timed out after {timeout_s}s at step {global_step}; '
                        f'checkpoint preserved at {run_dir}/global_step_{global_step}'
                    ) from exc
                except subprocess.CalledProcessError as exc:
                    raise RuntimeError(
                        f'SFT vLLM evaluation failed at step {global_step}; '
                        f'checkpoint preserved at {run_dir}/global_step_{global_step}'
                    ) from exc
        finally:
            if legacy_eval_offload:
                load_fsdp_model_to_gpu(self.fsdp_model)
                load_fsdp_optimizer(self.optimizer, torch.cuda.current_device())
                torch.cuda.synchronize()
        torch.distributed.barrier()

        if rank != 0:
            return {}

        with open(result_path) as handle:
            result = json.load(handle)
        if eval_kind == 'knights_and_knaves':
            benchmark_names = (
                'iid_3ppl', 'iid_4ppl', 'iid_5ppl', 'iid_6ppl',
                'ood_7ppl', 'ood_8ppl',
            )
        else:
            benchmark_names = ('math500', 'aime24', 'aime25', 'amc23', 'minerva', 'olympiadbench')
        metrics = {
            f'eval_greedy/{name}_pass1': float(
                result['benchmarks'][name]['mean']['pass_at_1']
            )
            for name in benchmark_names
        }
        metrics['eval_greedy/avg_pass1'] = float(
            result['benchmarks']['avg_pass_at_1']['mean']
        )
        metrics['eval_greedy/benchmark_count'] = len(benchmark_names)
        metrics['eval_greedy/global_step'] = int(global_step)
        print(
            f'[SFT greedy] step={global_step} '
            f'avg_pass1={metrics["eval_greedy/avg_pass1"]:.6f}',
            flush=True,
        )
        return metrics

    def _maybe_save_best_checkpoint(self, metrics, global_step):
        """Pin the checkpoint selected by W&B's canonical greedy pass@1 key."""
        rank = self.device_mesh.get_rank()
        score = float(metrics.get('eval_greedy/avg_pass1', float('-inf'))) if rank == 0 else float('-inf')
        score_tensor = torch.tensor(
            score, dtype=torch.float64, device=torch.cuda.current_device()
        )
        torch.distributed.broadcast(score_tensor, src=0)
        score = float(score_tensor.item())
        improved = math.isfinite(score) and score > self._best_eval_score
        evaluated_path = os.path.join(
            self.config.trainer.default_local_dir, f'global_step_{global_step}'
        )
        previous_best_step = self._best_eval_step
        if not improved:
            if rank == 0:
                # Every evaluation first writes a checkpoint for vLLM.  Keep
                # only the best evaluated checkpoint to avoid filling the
                # filesystem when greedy_eval_freq is small.
                if os.path.isdir(evaluated_path):
                    shutil.rmtree(evaluated_path)
                    print(
                        f'[SFT] removed non-best evaluated checkpoint: '
                        f'global_step_{global_step}',
                        flush=True,
                    )
            torch.distributed.barrier()
            return False

        self._best_eval_score = score
        self._best_eval_step = int(global_step)
        # _run_greedy_eval already saved this exact evaluated checkpoint.
        if rank == 0:
            if previous_best_step is not None and previous_best_step != self._best_eval_step:
                previous_path = os.path.join(
                    self.config.trainer.default_local_dir,
                    f'global_step_{previous_best_step}',
                )
                if os.path.isdir(previous_path):
                    shutil.rmtree(previous_path)
            best_path = os.path.join(self.config.trainer.default_local_dir, 'best_ckpt.json')
            with open(best_path, 'w') as handle:
                json.dump({
                    'best_step': self._best_eval_step,
                    'best_score': self._best_eval_score,
                    'metric': 'eval_greedy/avg_pass1',
                    'path': f'global_step_{self._best_eval_step}',
                }, handle, indent=2)
            print(
                f'[SFT] new best checkpoint: step={self._best_eval_step} '
                f'eval_greedy/avg_pass1={self._best_eval_score:.6f}',
                flush=True,
            )
        torch.distributed.barrier()
        return True

    def _run_validation(self, tracking, global_step, rank):
        """Mean val loss, logged as val/loss. `trainer.val_max_batches` caps the number of
        val batches so a periodic eval over a large test split stays affordable."""
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
        tracking = None
        if rank == 0:
            tracking = Tracking(project_name=self.config.trainer.project_name,
                                experiment_name=self.config.trainer.experiment_name,
                                default_backend=self.config.trainer.logger,
                                config=OmegaConf.to_container(self.config, resolve=True))

        global_step = 0
        # The scheduler was built with this exact horizon. Do not recompute it
        # from the uncapped epoch budget here.
        self.total_training_steps = self.total_steps
        print(f'Total training steps: {self.total_training_steps}')

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
                greedy_freq = self.config.trainer.get('greedy_eval_freq', 0)
                if greedy_freq > 0 and global_step % greedy_freq == 0 and global_step < self.total_training_steps:
                    greedy_metrics = self._run_greedy_eval(global_step, rank)
                    if rank == 0:
                        tracking.log(data=greedy_metrics, step=global_step)
                    self._maybe_save_best_checkpoint(greedy_metrics, global_step)
                if (save_freq > 0 and global_step % save_freq == 0
                        and global_step < self.total_training_steps
                        and not (greedy_freq > 0 and global_step % greedy_freq == 0)
                        and not self.config.trainer.get('keep_best_only', True)):
                    self.save_checkpoint(step=global_step)

                # for early exit validation
                if global_step >= self.total_training_steps:
                    # Perform final validation
                    self._run_validation(tracking, global_step, rank)

                    greedy_metrics = self._run_greedy_eval(global_step, rank)
                    if rank == 0:
                        tracking.log(data=greedy_metrics, step=global_step)
                    self._maybe_save_best_checkpoint(greedy_metrics, global_step)

                    # The evaluated checkpoint is retained only when it is the
                    # best pass@1 checkpoint; do not recreate a discarded final copy.
                    return

            # validation
            self._run_validation(tracking, global_step, rank)

            # Evaluation checkpoints are the only checkpoints retained by this
            # audit; epoch-end saves would create unscored duplicates.


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
