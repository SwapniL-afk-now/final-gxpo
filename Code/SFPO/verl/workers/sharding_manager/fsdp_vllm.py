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

import os
import logging
import time
import torch
import numpy as np
from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.api import ShardingStrategy, ShardedStateDictConfig, StateDictType, FullStateDictConfig
from torch.distributed.device_mesh import DeviceMesh

from verl.third_party.vllm import LLM
from verl.third_party.vllm import parallel_state as vllm_ps
from verl import DataProto
from verl.utils.torch_functional import (broadcast_dict_tensor, allgather_dict_tensors)
from verl.protocol import all_gather_data_proto
from verl.utils.debug import log_gpu_memory_usage
from verl.third_party.vllm import vllm_version

from verl.utils.lora_utils import merge_lora_state_dict

from .base import BaseShardingManager

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv('VERL_PPO_LOGGING_LEVEL', 'WARN'))


def _memory_snapshot():
    if not torch.cuda.is_available():
        return {}
    torch.cuda.synchronize()
    free_bytes, total_bytes = torch.cuda.mem_get_info()
    return {
        'allocated_gb': torch.cuda.memory_allocated() / (1024**3),
        'reserved_gb': torch.cuda.memory_reserved() / (1024**3),
        'free_device_gb': free_bytes / (1024**3),
        'used_device_gb': (total_bytes - free_bytes) / (1024**3),
        'total_device_gb': total_bytes / (1024**3),
    }


class FSDPVLLMShardingManager(BaseShardingManager):

    def __init__(self,
                 module: FSDP,
                 inference_engine: LLM,
                 model_config,
                 full_params: bool = False,
                 device_mesh: DeviceMesh = None):
        self.module = module
        self.inference_engine = inference_engine
        self.model_config = model_config
        self.device_mesh = device_mesh
        self.performance_events = {}

        # Full params
        self.full_params = full_params
        if full_params:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.FULL_STATE_DICT,
                                     state_dict_config=FullStateDictConfig())
        else:
            FSDP.set_state_dict_type(self.module,
                                     state_dict_type=StateDictType.SHARDED_STATE_DICT,
                                     state_dict_config=ShardedStateDictConfig())

        # LoRA: read the adapter scaling (alpha/r) from the live peft module once;
        # None means the actor is not LoRA-wrapped and sync passes weights through untouched
        self.lora_scaling = next((m.scaling['default']
                                  for m in module.modules()
                                  if isinstance(getattr(m, 'scaling', None), dict) and 'default' in m.scaling), None)

        self.tp_size = vllm_ps.get_tensor_model_parallel_world_size()
        self.tp_rank = vllm_ps.get_tensor_model_parallel_rank()

        # Note that torch_random_states may be different on each dp rank
        self.torch_random_states = torch.cuda.get_rng_state()
        # get a random rng states
        if self.device_mesh is not None:
            gen_dp_rank = self.device_mesh['dp'].get_local_rank()
            torch.cuda.manual_seed(gen_dp_rank + 1000)  # make sure all tp ranks have the same random states
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)
        else:
            self.gen_random_states = None

    def __enter__(self):
        # NOTE: Basically, we only need `torch.cuda.empty_cache()` before vllm wake_up and
        # after vllm sleep, since vllm has its own caching memory allocator CuMemAllocator.
        # Out of vllm scope, we should avoid empty cache to let pytorch using caching memory
        # to speed up memory allocations.
        #
        # pytorch: https://pytorch.org/docs/stable/notes/cuda.html#memory-management
        # vllm: https://github.com/vllm-project/vllm/blob/v0.7.3/vllm/device_allocator/cumem.py#L103
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        self.performance_events = {
            'before_vllm_wake_memory': _memory_snapshot(),
        }
        wake_start = time.monotonic()

        log_gpu_memory_usage('Before state_dict() in sharding manager memory', logger=logger)
        params = self.module.state_dict()
        if self.lora_scaling is not None:
            params = merge_lora_state_dict(params, self.lora_scaling)
        log_gpu_memory_usage('After state_dict() in sharding manager memory', logger=logger)
        # Copy, not share memory
        load_format = 'hf' if self.full_params else 'dtensor'

        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.sync_model_weights(params, load_format=load_format)
        else:
            self.inference_engine.wake_up()
            world_size = torch.distributed.get_world_size()
            model = self.inference_engine.llm_engine.model_executor.driver_worker.worker.model_runner.model
            loaded_params = model.load_weights(
                ((name, param.full_tensor() if world_size != 1 else param) for name, param in params.items()))
            logger.info(f"vLLM load wegiths, loaded_params: {len(loaded_params)}")

        log_gpu_memory_usage('After sync model weights in sharding manager', logger=logger)

        del params
        torch.cuda.synchronize()
        self.performance_events.update({
            'vllm_wake_and_weight_sync_s': time.monotonic() - wake_start,
            'after_vllm_wake_memory': _memory_snapshot(),
        })
        self.performance_events['before_rollout_memory'] = _memory_snapshot()
        log_gpu_memory_usage('After del state_dict and empty_cache in sharding manager', logger=logger)

        # TODO: offload FSDP model weights
        # self.module.cpu()
        # torch.cuda.empty_cache()
        # if torch.distributed.get_rank() == 0:
        # print(f'after model to cpu in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        # important: need to manually set the random states of each tp to be identical.
        if self.device_mesh is not None:
            self.torch_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.gen_random_states)

    def __exit__(self, exc_type, exc_value, traceback):
        self.performance_events['after_rollout_memory'] = _memory_snapshot()
        self.performance_events['before_vllm_sleep_memory'] = dict(
            self.performance_events['after_rollout_memory'])
        peak = {
            'allocated_gb': torch.cuda.max_memory_allocated() / (1024**3),
            'reserved_gb': torch.cuda.max_memory_reserved() / (1024**3),
        }
        self.performance_events['rollout_peak_memory'] = peak
        log_gpu_memory_usage('Before vllm offload in sharding manager', logger=logger)
        sleep_start = time.monotonic()
        sleep_level = int(os.environ.get('VLLM_SLEEP_LEVEL', '1'))
        # vLLM 0.27.1 supports level 2: discard weights and KV cache, which
        # leaves the actor phase the largest possible amount of device memory.
        if vllm_version in ('0.4.2', '0.5.4', '0.6.3'):
            self.inference_engine.offload_model_weights()
        else:
            self.inference_engine.sleep(level=sleep_level)
        torch.cuda.empty_cache()
        self.performance_events.update({
            'vllm_sleep_s': time.monotonic() - sleep_start,
            'vllm_sleep_level': sleep_level,
            'after_vllm_sleep_memory': _memory_snapshot(),
        })
        self._rollout_lifecycle = dict(self.performance_events)
        log_gpu_memory_usage('After vllm offload in sharding manager', logger=logger)

        # self.module.to('cuda')
        # if torch.distributed.get_rank() == 0:
        #     print(f'after actor module to cuda in sharding manager memory allocated: {torch.cuda.memory_allocated() / 1e9}GB, reserved: {torch.cuda.memory_reserved() / 1e9}GB')

        self.module.train()

        # add empty cache after each compute
        torch.cuda.empty_cache()

        # restore random states
        if self.device_mesh is not None:
            self.gen_random_states = torch.cuda.get_rng_state()
            torch.cuda.set_rng_state(self.torch_random_states)

    def preprocess_data(self, data: DataProto) -> DataProto:
        """All gather across tp group to make each rank has identical input."""
        if self.tp_size == 1:
            return data

        # TODO: Current impl doesn't consider FSDP with torch micro-dp
        if vllm_version in ('0.3.1', '0.4.2', '0.5.4', '0.6.3'):
            group = vllm_ps.get_tensor_model_parallel_group()
        else:
            group = vllm_ps.get_tensor_model_parallel_group().device_group

        all_gather_data_proto(data=data, process_group=group)
        return data

    def postprocess_data(self, data: DataProto) -> DataProto:
        """Get chunk data of this tp rank since we do all gather in preprocess."""
        if self.tp_size == 1:
            return data

        return data.chunk(chunks=self.tp_size)[self.tp_rank]
