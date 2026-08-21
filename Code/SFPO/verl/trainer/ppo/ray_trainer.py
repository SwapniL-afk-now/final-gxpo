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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import json
import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy
import time
import copy
import torch

import shutil
from os.path import join as osj

import ray
import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.ppo.metric_utils import compute_data_metrics, compute_throughout_metrics, compute_timing_metrics, reduce_metrics, compute_reward_metrics
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
from .presampling_selector import DataProfiler
from tools.gxpo_efficiency_runtime import (
    BENCHMARK_ORDER, append_jsonl, json_safe, make_run_manifest, sample_gpu_telemetry,
    scalar, source_to_benchmark, write_json,
)

WorkerType = Type[Worker]


class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def get_n_gpus(self) -> int:
        """Get the number of gpus in this cluster."""
        return sum([n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [n_gpus for process_on_nodes in self.resource_pool_spec.values() for n_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics, current_kl


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns, reward = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
        data.batch['reward'] = reward
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def ckpt_steps_to_remove(steps, keep, best_step=None):
    """Which global_step_* dirs to delete: keep the newest `keep` for resume, never the best one."""
    steps = sorted(steps)
    keep = max(int(keep), 1)
    return [s for s in steps[:-keep] if s != best_step] if len(steps) > keep else []


def write_retained_jsonl(path, value, keep):
    """Write a bounded JSONL file atomically; keep<=0 preserves append-only behavior."""
    keep = int(keep)
    if keep <= 0:
        append_jsonl(path, value)
        return

    lines = []
    if os.path.exists(path) and keep > 1:
        with open(path, 'r') as handle:
            lines = [line.rstrip('\n') for line in handle if line.strip()]
        lines = lines[-(keep - 1):]
    lines.append(json.dumps(json_safe(value), sort_keys=True))

    directory = os.path.dirname(path) or '.'
    os.makedirs(directory, exist_ok=True)
    temporary_path = os.path.join(directory, f'.{os.path.basename(path)}.{uuid.uuid4().hex}.tmp')
    try:
        with open(temporary_path, 'w') as handle:
            handle.write('\n'.join(lines) + '\n')
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()
        self.data_profiler = DataProfiler()
        self.start_epoch = 0
        self.current_epoch = 0
        self.entropy_container = []
        self.stop_SFPO = False
        self.sfpo_trigger_streak = 0
        self.sfpo_trigger_step = None
        self.sfpo_entropy_reset_done = False
        # GXPO uses the same entropy-trigger gate as SFPO.  Keep a separate
        # history so the two algorithms cannot affect one another when runs
        # share trainer code.
        self.gxpo_entropy_container = []
        self.stop_GXPO = False
        self.gxpo_trigger_streak = 0
        self.gxpo_trigger_step = None
        self.gxpo_entropy_reset_done = False

        self.targeted_easy = self.config.data.get('target_zero_variance', 0.25) / 3
        self.targeted_hard = self.config.data.get('target_zero_variance', 0.25) * 2 / 3 # more exploration on hard examples, because they are more likely to change (conidering that overall reward is increasing.)

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

        if self.config.data.get('dynamic_filtering_strategy', 'None') == 'all_linear_backoff':
            self.skip_easy = self.config.data.skip_easy
            self.skip_hard = self.config.data.skip_hard
        if self.config.data.get('dynamic_filtering_strategy', 'None') == 'all_probabilistic':
            self.p_easy = self.config.data.p_easy
            self.p_hard = self.config.data.p_hard

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # check if train_batch_size is larger than ppo_mini_batch_size
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # assert config.data.train_batch_size >= config.actor_rollout_ref.actor.ppo_mini_batch_size
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            assert config.data.train_batch_size >= config.critic.ppo_mini_batch_size
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        # check eval config
        if config.actor_rollout_ref.rollout.val_kwargs.do_sample:
            assert config.actor_rollout_ref.rollout.temperature > 0, \
                "validation gen temperature should be greater than 0 when enabling do_sample"

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         system_prompt=self.config.data.get('system_prompt', None),
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation='error',
                                         filter_overlong_prompts=self.config.data.filter_overlong_prompts)
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            # fixed seed if data.seed is set (reproducible / matched across runs), else time-based
            seed = self.config.data.get('seed', None)
            if seed is None:
                seed = int(time.time() * 1000) % (2**32)
            print(f'Train dataloader shuffle seed: {seed}')
            train_dataloader_generator.manual_seed(int(seed))

            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        val_files = list(self.config.data.get('val_files') or [])
        if val_files:
            self.val_dataset = RLHFDataset(parquet_files=val_files,
                                           tokenizer=self.tokenizer,
                                           processor=self.processor,
                                           prompt_key=self.config.data.prompt_key,
                                           image_key=self.config.data.get('image_key', 'images'),
                                           max_prompt_length=self.config.data.max_prompt_length,
                                           filter_prompts=True,
                                           system_prompt=self.config.data.get('system_prompt', None),
                                           return_raw_chat=self.config.data.get('return_raw_chat', False),
                                           truncation='error',
                                           filter_overlong_prompts=self.config.data.filter_overlong_prompts)
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                # Validation datasets are sent to inference engines as a whole batch,
                # which will schedule the memory themselves.
                batch_size=len(self.val_dataset),
                num_workers=8,
                shuffle=False,
                drop_last=False,
                collate_fn=collate_fn)
        else:
            # Performance smokes explicitly omit validation. Do not even open a
            # validation parquet in that mode.
            self.val_dataset = None
            self.val_dataloader = ()

        assert len(self.train_dataloader) >= 1
        if val_files:
            assert len(
                self.val_dataloader
            ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def _validate(self):
        """Run validation over one or more RNG seeds (trainer.validation_seeds).

        For each seed logs per-source metrics as val/.../seed{S}, and reports
        the mean and std across seeds as the canonical val/.../{source} keys.
        """
        seeds = self.config.trainer.get('validation_seeds', None)
        if not seeds:
            return self._decorate_greedy_validation(self._validate_once(gen_seed=None))

        per_seed = {s: self._validate_once(gen_seed=int(s)) for s in seeds}

        merged = {}
        # collect the set of metric names across seeds (they share the same keys)
        keys = set().union(*[d.keys() for d in per_seed.values()])
        for key in keys:
            vals = [per_seed[s][key] for s in seeds if key in per_seed[s]]
            for s in seeds:
                if key in per_seed[s]:
                    merged[f'{key}/seed{int(s)}'] = per_seed[s][key]
            merged[key] = float(np.mean(vals))
            merged[f'{key}/std'] = float(np.std(vals))
        return self._decorate_greedy_validation(merged)

    def _decorate_greedy_validation(self, metrics):
        """Expose stable paper keys from the existing per-source validation."""
        result = dict(metrics)
        values = {}
        for key, value in metrics.items():
            if not key.startswith('val/pass_at_1/') or key.endswith('/std') or '/seed' in key:
                continue
            source = key[len('val/pass_at_1/'):]
            benchmark = source_to_benchmark(source)
            if benchmark is not None:
                values[benchmark] = float(value)
        for benchmark in BENCHMARK_ORDER:
            if benchmark in values:
                result[f'eval_greedy/{benchmark}_pass1'] = values[benchmark]
        if os.environ.get('GXPO_EFFICIENCY_RUN') and set(values) != set(BENCHMARK_ORDER):
            missing = sorted(set(BENCHMARK_ORDER) - set(values))
            raise RuntimeError(f'GXPO efficiency validation requires all six benchmarks; missing {missing}')
        if values:
            # Macro-average across benchmark datasets, never across examples.
            result['eval_greedy/avg_pass1'] = float(np.mean([values[k] for k in BENCHMARK_ORDER if k in values]))
            result['eval_greedy/benchmark_count'] = len(values)
        result['eval_greedy/global_step'] = int(self.global_steps)
        return result

    def _validate_once(self, gen_seed=None):
        reward_tensor_lst = []
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []
        print('Repeated validation with n={}'.format(self.config.actor_rollout_ref.rollout.val_kwargs.n))
        print('Validation config:')
        pprint(self.config.actor_rollout_ref.rollout.val_kwargs)

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            if 'multi_modal_inputs' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
                'gen_seed': gen_seed,
            }
            print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch)

            # evaluate using reward_function
            reward_tensor = self.val_reward_fn(test_batch)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        # Also report the average
        sum_acc = 0
        sum_count = 0
        val_n = self.config.actor_rollout_ref.rollout.val_kwargs.n
        for data_source, rewards in data_source_reward.items():
            rewards = np.array(rewards)
            #ignore the format score
            rewards[rewards < 0.95] = 0
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            print(f'>>> Test: val/test_score/{data_source}: {np.mean(rewards)}')
            sum_acc += np.mean(rewards)
            sum_count += 1

            # pass@1 (mean correctness over n samples) and pass@n (any-correct per prompt);
            # interleaved repeat => contiguous groups of n samples per prompt
            correct = (rewards >= 0.95).astype(np.float64)
            metric_dict[f'val/pass_at_1/{data_source}'] = float(correct.mean())
            if val_n > 1 and len(correct) % val_n == 0:
                per_prompt = correct.reshape(-1, val_n)
                metric_dict[f'val/pass_at_{val_n}/{data_source}'] = float(per_prompt.max(axis=1).mean())
                # avg@n: mean accuracy across the n samples, averaged over prompts
                metric_dict[f'val/avg_at_{val_n}/{data_source}'] = float(per_prompt.mean(axis=1).mean())
        metric_dict['val/test_score/average'] = sum_acc / sum_count
        print(f'>>> Test: val/test_score/average: {sum_acc / sum_count}')
        print(metric_dict)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _best_ckpt_score(self, val_metrics):
        """Checkpoint-selection score: macro-mean val pass@1 across val sources (seed-mean).

        Unweighted across sources, so a 30-problem AIME set counts the same as 40-problem AMC23.
        """
        vals = [v for k, v in val_metrics.items()
                if k.startswith('val/pass_at_1/') and '/seed' not in k and not k.endswith('/std')]
        return float(np.mean(vals)) if vals else float('-inf')

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # save data profiler self.data_profiler
        dataprofiler_local_path = os.path.join(local_global_step_folder, 'data_profiler.pt')
        if self.data_profiler is not None:
            self.data_profiler.save(dataprofiler_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

        # save the current epoch to the checkpoint
        local_current_epoch = os.path.join(self.config.trainer.default_local_dir, 'current_epoch.txt')
        with open(local_current_epoch, 'w') as f:
            f.write(str(self.current_epoch))

        if self.config.data.get('dynamic_filtering_strategy', 'None') == 'all_linear_backoff':
            ## save self.skip_easy and self.skip_hard
            filtering_dynamics = {
                'strategy': 'all_linear_backoff',
                'skip_easy': self.skip_easy,
                'skip_hard': self.skip_hard
            }
            filtering_dynamics_local_path = os.path.join(local_global_step_folder, 'filtering_dynamics.pt')
            # save filtering_dynamics
            torch.save(filtering_dynamics, filtering_dynamics_local_path)

        if self.config.data.get('dynamic_filtering_strategy', 'None') == 'all_probabilistic':
            ## save self.p_easy and self.p_hard
            filtering_dynamics = {
                'strategy': 'all_probabilistic',
                'p_easy': self.p_easy,
                'p_hard': self.p_hard
            }
            filtering_dynamics_local_path = os.path.join(local_global_step_folder, 'filtering_dynamics.pt')
            # save filtering_dynamics
            torch.save(filtering_dynamics, filtering_dynamics_local_path)

        # save self.sampling_num
        local_sampling_num = os.path.join(self.config.trainer.default_local_dir, 'sampling_num.txt')
        with open(local_sampling_num, 'w') as f:
            f.write(str(self.sampling_num))

        # Record which step is the best-pass@1 one so it can be found (and pinned) after the run.
        best_step = getattr(self, '_best_val_step', None)
        if best_step is not None:
            with open(os.path.join(self.config.trainer.default_local_dir, 'best_ckpt.json'), 'w') as f:
                json.dump({'best_step': best_step,
                           'best_score': getattr(self, '_best_val_score', None),
                           'metric': 'macro-mean val/pass_at_1',
                           'path': f'global_step_{best_step}'}, f)

        # Keep only the latest n checkpoints (for resume), plus the pinned best-pass@1 one.
        # GXPO sets n=1, so this is at most one resumable checkpoint plus one best checkpoint.
        n = max(int(self.config.trainer.get('keep_last_ckpts', 1)), 1)
        if self.config.trainer.get('keep_all_ckpts', False):
            return

        checkpoint_dirs = [d for d in os.listdir(self.config.trainer.default_local_dir)
                        if d.startswith('global_step_')]
        steps = []
        for d in checkpoint_dirs:
            try:
                step = int(d.split('_')[-1])
                steps.append(step)
            except ValueError:
                continue  # Ignore malformed directories

        for step in ckpt_steps_to_remove(steps, n, best_step):
            dir_to_remove = os.path.join(self.config.trainer.default_local_dir, f'global_step_{step}')
            print(f"Removing old checkpoint directory: {dir_to_remove}")
            shutil.rmtree(dir_to_remove, ignore_errors=True)

    def _load_checkpoint(self):
        print('self.config.trainer.resume_mode', self.config.trainer.resume_mode)
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)

        # Restore best-validation state before continuing, otherwise a resumed run
        # could incorrectly replace the real best checkpoint with a lower score.
        best_meta_path = os.path.join(self.config.trainer.default_local_dir, 'best_ckpt.json')
        if os.path.exists(best_meta_path):
            try:
                with open(best_meta_path, 'r') as f:
                    best_meta = json.load(f)
                self._best_val_step = int(best_meta['best_step'])
                self._best_val_score = float(best_meta['best_score'])
                print(f"Restored best validation checkpoint: step={self._best_val_step}, "
                      f"score={self._best_val_score:.6f}")
            except (KeyError, TypeError, ValueError, OSError, json.JSONDecodeError) as exc:
                print(f'Warning: could not restore best checkpoint metadata from {best_meta_path}: {exc}')
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load data profiler
        self.data_profiler.load(os.path.join(global_step_folder, 'data_profiler.pt'))

        # load the current epoch to the checkpoint
        local_current_epoch = os.path.join(self.config.trainer.default_local_dir, 'current_epoch.txt')
        # import ipdb; ipdb.set_trace()
        if os.path.exists(local_current_epoch):
            with open(local_current_epoch, 'r') as f:
                self.start_epoch = int(f.read())
            print(f'Loaded current epoch: {self.start_epoch}')
        else:
            print(f'Warning: No current epoch found at {local_current_epoch}, set epoch = 0')
            self.start_epoch = 0

        if self.config.data.get('dynamic_filtering_strategy', 'None') == 'all_probabilistic':
            filtering_dynamics_local_path = os.path.join(global_step_folder, 'filtering_dynamics.pt')
            if os.path.exists(filtering_dynamics_local_path):
                filtering_dynamics = torch.load(filtering_dynamics_local_path)
                self.p_easy = filtering_dynamics['p_easy']
                self.p_hard = filtering_dynamics['p_hard']
                print(f'Loaded filtering dynamics: {filtering_dynamics}')
            else:
                print(f'Warning: No filtering dynamics found at {filtering_dynamics_local_path}, use default from config.')
                print(f'self.p_easy: {self.p_easy}, self.p_hard: {self.p_hard}')

        # load self.sampling_num
        local_sampling_num = os.path.join(self.config.trainer.default_local_dir, 'sampling_num.txt')
        if os.path.exists(local_sampling_num):
            with open(local_sampling_num, 'r') as f:
                self.sampling_num = int(f.read())
            print(f'Loaded sampling num: {self.sampling_num}')
        else:
            print(f'Warning: No sampling num found at {local_sampling_num}, set sampling num = 0')
            self.sampling_num = 0

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def _init_efficiency_logging(self):
        run_dir = self.config.trainer.get('default_local_dir', '.')
        self._efficiency_run_dir = os.path.abspath(os.path.expanduser(str(run_dir)))
        os.makedirs(self._efficiency_run_dir, exist_ok=True)
        self._efficiency_state = {
            'cum_prompts': 0.0,
            'cum_responses': 0.0,
            'cum_prompt_tokens': 0.0,
            'cum_completion_tokens': 0.0,
            'cum_total_tokens': 0.0,
            'cum_policy_grad_evals': 0.0,
            'cum_raw_backward_calls': 0.0,
            'cum_train_active_s': 0.0,
            'cum_rollout_s': 0.0,
            'cum_reward_s': 0.0,
            'cum_ref_logprob_s': 0.0,
            'cum_old_logprob_s': 0.0,
            'cum_actor_update_s': 0.0,
            'cum_data_sync_other_s': 0.0,
            'cum_validation_s': 0.0,
            'cum_checkpoint_s': 0.0,
            'energy_kwh': 0.0,
            'last_telemetry_time': None,
            'last_power_w': None,
            'last_telemetry': {},
        }
        manifest = make_run_manifest(self.config, self._efficiency_run_dir, os.getcwd())
        write_json(os.path.join(self._efficiency_run_dir, 'run_manifest.json'), manifest)
        print('[fair comparison config]')
        print(json.dumps({
            'model': manifest.get('hyperparameters', {}).get('actor_rollout_ref', {}).get('model', {}).get('path'),
            'method': manifest.get('method'),
            'train_batch_size': manifest.get('hyperparameters', {}).get('data', {}).get('train_batch_size'),
            'rollout_n': manifest.get('hyperparameters', {}).get('actor_rollout_ref', {}).get('rollout', {}).get('n'),
            'learning_rate': manifest.get('hyperparameters', {}).get('actor_rollout_ref', {}).get('actor', {}).get('optim', {}).get('lr'),
            'prompt_length': manifest.get('hyperparameters', {}).get('data', {}).get('max_prompt_length'),
            'response_length': manifest.get('hyperparameters', {}).get('data', {}).get('max_response_length'),
            'validation_interval': manifest.get('hyperparameters', {}).get('trainer', {}).get('test_freq'),
            'gpu_count': manifest.get('gpu_count_configured'),
        }, indent=2, sort_keys=True))

    def _efficiency_step_metrics(self, batch, metrics, timing_raw, active_s, fit_start_time, did_validate):
        state = self._efficiency_state
        step = int(self.global_steps)
        responses = batch.batch['responses']
        response_length = responses.shape[-1]
        attention_mask = batch.batch['attention_mask']
        prompt_mask = attention_mask[:, :-response_length].bool()
        response_mask = attention_mask[:, -response_length:].bool()
        n_rollouts = int(self.config.actor_rollout_ref.rollout.n)
        generated_responses = float(len(batch))
        generated_prompts = generated_responses / max(n_rollouts, 1)
        prompt_tokens = float(prompt_mask.sum().item())
        completion_tokens = float(response_mask.sum().item())
        total_tokens = prompt_tokens + completion_tokens

        policy_step = scalar(metrics.get('actor/policy_grad_evals_step'), 0.0) or 0.0
        raw_step = scalar(metrics.get('actor/raw_backward_calls_step'), 0.0) or 0.0
        lifecycle = dict(batch.meta_info.get('vllm_lifecycle', {}))
        generation_s = scalar(batch.meta_info.get('vllm_generation_time_s'))
        rollout_s = float(generation_s if generation_s is not None else timing_raw.get('gen', 0.0) or 0.0)
        reward_s = float(timing_raw.get('reward', 0.0) or 0.0)
        ref_s = float(timing_raw.get('ref', 0.0) or 0.0)
        old_s = float(timing_raw.get('old_log_prob', 0.0) or 0.0)
        actor_s = float(timing_raw.get('update_actor', 0.0) or 0.0)
        weight_sync_s = float(lifecycle.get('vllm_wake_and_weight_sync_s', 0.0) or 0.0)
        total_step_s = float(timing_raw.get('step', active_s) or active_s)
        known_s = rollout_s + reward_s + ref_s + old_s + actor_s
        other_s = max(float(active_s) - known_s, 0.0)
        validation_s = float(timing_raw.get('testing', 0.0) or 0.0)
        checkpoint_s = float(timing_raw.get('save_checkpoint', 0.0) or 0.0)

        state['cum_prompts'] += generated_prompts
        state['cum_responses'] += generated_responses
        state['cum_prompt_tokens'] += prompt_tokens
        state['cum_completion_tokens'] += completion_tokens
        state['cum_total_tokens'] += total_tokens
        state['cum_policy_grad_evals'] += policy_step
        state['cum_raw_backward_calls'] += raw_step
        state['cum_train_active_s'] += float(active_s)
        state['cum_rollout_s'] += rollout_s
        state['cum_reward_s'] += reward_s
        state['cum_ref_logprob_s'] += ref_s
        state['cum_old_logprob_s'] += old_s
        state['cum_actor_update_s'] += actor_s
        state['cum_data_sync_other_s'] += other_s
        state['cum_validation_s'] += validation_s
        state['cum_checkpoint_s'] += checkpoint_s

        telemetry_interval = max(int(os.environ.get('GXPO_GPU_TELEMETRY_INTERVAL', '10')), 1)
        if step % telemetry_interval == 0:
            telemetry = sample_gpu_telemetry()
            now = time.monotonic()
            if telemetry:
                if state['last_telemetry_time'] is not None and state['last_power_w'] is not None:
                    state['energy_kwh'] += state['last_power_w'] * (now - state['last_telemetry_time']) / 3.6e6
                state['last_telemetry_time'] = now
                state['last_power_w'] = telemetry.get('system/gpu_power_mean_w')
                state['last_telemetry'] = telemetry

        rewards = batch.batch.get('reward')
        reward_values = rewards.detach().float().reshape(-1) if rewards is not None else None
        if reward_values is not None and reward_values.numel():
            reward_mean = float(reward_values.mean().item())
            reward_std = float(reward_values.std(unbiased=False).item())
            reward_min = float(reward_values.min().item())
            reward_max = float(reward_values.max().item())
            reward_variance = float(reward_values.var(unbiased=False).item())
            positive_fraction = float((reward_values > 0).float().mean().item())
        else:
            reward_mean = reward_std = reward_min = reward_max = reward_variance = positive_fraction = float('nan')

        valid_adv = batch.batch['advantages'][response_mask].detach().float()
        advantage_mean = float(valid_adv.mean().item()) if valid_adv.numel() else float('nan')
        advantage_std = float(valid_adv.std(unbiased=False).item()) if valid_adv.numel() else float('nan')
        nonzero_advantage = float((valid_adv != 0).float().mean().item()) if valid_adv.numel() else float('nan')
        response_lengths = response_mask.sum(-1).detach().float().cpu().numpy()
        actor_entropy = scalar(metrics.get('actor/entropy_loss'))
        actor_grad = scalar(metrics.get('actor/grad_norm'))
        actor_clip = scalar(metrics.get('actor/pg_clipfrac'))
        actor_kl = scalar(metrics.get('actor/ppo_kl'))
        peak_alloc = scalar(metrics.get('perf/max_memory_allocated_gb'))
        peak_reserved = scalar(metrics.get('perf/max_memory_reserved_gb'))
        actor_tokens = scalar(metrics.get('perf/actor_tokens_processed'), total_tokens) or total_tokens
        rollout_tokens_per_second = total_tokens / rollout_s if rollout_s > 0 else None
        rollout_sequences_per_second = generated_responses / rollout_s if rollout_s > 0 else None
        actor_tokens_per_second = actor_tokens / actor_s if actor_s > 0 else None
        memory_before_rollout = lifecycle.get('before_rollout_memory', {})
        memory_rollout_peak = lifecycle.get('rollout_peak_memory', {})
        memory_before_sleep = lifecycle.get('before_vllm_sleep_memory', {})
        memory_after_sleep = lifecycle.get('after_vllm_sleep_memory', {})
        memory_before_actor = metrics.get('perf/before_actor_update_memory', {})
        memory_actor_peak = metrics.get('perf/actor_update_peak_memory', {})
        memory_after_actor = metrics.get('perf/after_actor_update_memory', {})

        out = {
            'train/global_step': step,
            'eff/outer_step': step,
            'eff/generated_prompts_step': generated_prompts,
            'eff/generated_responses_step': generated_responses,
            'eff/generated_prompt_tokens_step': prompt_tokens,
            'eff/generated_completion_tokens_step': completion_tokens,
            'eff/generated_total_tokens_step': total_tokens,
            'eff/cum_prompts': state['cum_prompts'],
            'eff/cum_responses': state['cum_responses'],
            'eff/cum_prompt_tokens': state['cum_prompt_tokens'],
            'eff/cum_completion_tokens': state['cum_completion_tokens'],
            'eff/cum_total_tokens': state['cum_total_tokens'],
            'eff/policy_grad_evals_step': policy_step,
            'eff/cum_policy_grad_evals': state['cum_policy_grad_evals'],
            'eff/raw_backward_calls_step': raw_step,
            'eff/cum_raw_backward_calls': state['cum_raw_backward_calls'],
            'time/step_train_active_s': float(active_s),
            'time/total_step_s': total_step_s,
            'time/rollout_s': rollout_s,
            'time/weight_sync_s': weight_sync_s,
            'time/reward_s': reward_s,
            'time/ref_logprob_s': ref_s,
            'time/old_logprob_s': old_s,
            'time/actor_update_s': actor_s,
            'time/data_sync_other_s': other_s,
            'time/validation_s': validation_s,
            'time/checkpoint_s': checkpoint_s,
            'time/cum_train_active_s': state['cum_train_active_s'],
            'time/cum_rollout_s': state['cum_rollout_s'],
            'time/cum_reward_s': state['cum_reward_s'],
            'time/cum_ref_logprob_s': state['cum_ref_logprob_s'],
            'time/cum_old_logprob_s': state['cum_old_logprob_s'],
            'time/cum_actor_update_s': state['cum_actor_update_s'],
            'time/cum_data_sync_other_s': state['cum_data_sync_other_s'],
            'time/cum_validation_s': state['cum_validation_s'],
            'time/cum_checkpoint_s': state['cum_checkpoint_s'],
            'time/cum_end_to_end_elapsed_s': time.monotonic() - fit_start_time,
            'perf/total_step_time_s': total_step_s,
            'perf/rollout_time_s': rollout_s,
            'perf/rollout_generated_tokens': completion_tokens,
            'perf/rollout_tokens_per_second': rollout_tokens_per_second,
            'perf/rollout_sequences_per_second': rollout_sequences_per_second,
            'perf/actor_update_time_s': actor_s,
            'perf/actor_tokens_processed': actor_tokens,
            'perf/actor_tokens_per_second': actor_tokens_per_second,
            'perf/logprob_time_s': old_s,
            'perf/weight_sync_time_s': weight_sync_s,
            'perf/vllm_sleep_level': lifecycle.get('vllm_sleep_level'),
            'perf/vllm_sleep_time_s': lifecycle.get('vllm_sleep_s'),
            'system/peak_vram_allocated_gb': peak_alloc,
            'system/peak_vram_reserved_gb': peak_reserved,
            'system/before_rollout_memory': memory_before_rollout,
            'system/rollout_peak_memory': memory_rollout_peak,
            'system/after_rollout_memory': lifecycle.get('after_rollout_memory', {}),
            'system/before_vllm_sleep_memory': memory_before_sleep,
            'system/after_vllm_sleep_memory': memory_after_sleep,
            'system/before_actor_update_memory': memory_before_actor,
            'system/actor_update_peak_memory': memory_actor_peak,
            'system/after_actor_update_memory': memory_after_actor,
            'system/after_vllm_wake_memory': lifecycle.get('after_vllm_wake_memory', {}),
            'system/energy_kwh': state['energy_kwh'],
            'train/reward_mean': reward_mean,
            'train/reward_std': reward_std,
            'train/reward_min': reward_min,
            'train/reward_max': reward_max,
            'train/reward_variance': reward_variance,
            'train/positive_reward_fraction': positive_fraction,
            'train/advantage_mean': advantage_mean,
            'train/advantage_std': advantage_std,
            'train/nonzero_advantage_fraction': nonzero_advantage,
            'train/entropy_mean': actor_entropy,
            'train/entropy_std': 0.0,
            'train/response_length_mean': float(np.mean(response_lengths)),
            'train/response_length_std': float(np.std(response_lengths)),
            'train/response_length_p50': float(np.percentile(response_lengths, 50)),
            'train/response_length_p90': float(np.percentile(response_lengths, 90)),
            'train/response_length_max': float(np.max(response_lengths)),
            'train/grad_norm': actor_grad,
            'train/grad_norm_pre_clip': actor_grad,
            'train/clip_fraction': actor_clip,
            'train/approx_kl': actor_kl,
        }
        # Keep the schema stable even on hosts without nvidia-smi. Missing
        # telemetry is represented as null in local JSON instead of silently
        # changing the exported columns between runs.
        out.update({
            'system/gpu_util_mean': None,
            'system/gpu_util_p50': None,
            'system/gpu_util_p90': None,
            'system/gpu_util_peak': None,
            'system/gpu_power_mean_w': None,
            'system/gpu_power_peak_w': None,
        })
        out.update(state['last_telemetry'])

        actor_cfg = self.config.actor_rollout_ref.actor
        if actor_cfg.get('use_gxpo', False):
            alpha = scalar(actor_cfg.get('gxpo_alpha'), 0.5)
            out.update({
                'gxpo/k': scalar(actor_cfg.get('gxpo_k'), 10),
                'gxpo/alpha': alpha,
                'gxpo/prediction_active': scalar(metrics.get('actor/gxpo_prediction_active'), scalar(metrics.get('actor/gxpo_enabled'), 0.0)),
                'gxpo/fallback_triggered': scalar(metrics.get('actor/gxpo_fallback_triggered'), 0.0),
                'gxpo/fallback_step': scalar(metrics.get('actor/gxpo_fallback_step')),
                'gxpo/g0_norm': scalar(metrics.get('actor/gxpo_g0_norm')),
                'gxpo/g1_norm': scalar(metrics.get('actor/gxpo_g1_norm')),
                'gxpo/fresh_grad_norm': scalar(metrics.get('actor/gxpo_gslow_norm')),
                'gxpo/g0_g1_cosine': scalar(metrics.get('actor/gxpo_cos_g0_g1')),
                'gxpo/g1_fresh_cosine': scalar(metrics.get('actor/gxpo_cos_g0_gslow')),
                'gxpo/two_step_displacement_norm': scalar(metrics.get('actor/gxpo_disp2_norm')),
                'gxpo/predicted_displacement_norm': scalar(metrics.get('actor/gxpo_dispK_norm')),
                'gxpo/reposition_displacement_norm': (alpha or 0.0) * (scalar(metrics.get('actor/gxpo_dispK_norm'), 0.0) or 0.0),
                'gxpo/prediction_to_observed_displacement_ratio': scalar(metrics.get('actor/gxpo_dispK_over_disp2')),
                'gxpo/prediction_scale_mean': scalar(metrics.get('actor/gxpo_scale_mean')),
                'gxpo/prediction_scale_std': scalar(metrics.get('actor/gxpo_r_std')),
                'gxpo/prediction_scale_max': scalar(metrics.get('actor/gxpo_scale_max')),
                'gxpo/retention_mean': scalar(metrics.get('actor/gxpo_r_mean')),
                'gxpo/retention_std': scalar(metrics.get('actor/gxpo_r_std')),
                'gxpo/retention_abs_mean': scalar(metrics.get('actor/gxpo_r_mean')),
                'gxpo/active_coordinate_fraction': 1.0 - (scalar(metrics.get('actor/gxpo_inactive_frac'), 0.0) or 0.0),
                'gxpo/unsafe_coordinate_fraction': scalar(metrics.get('actor/gxpo_ratio_clip_frac')),
                'gxpo/reliability_stat': scalar(metrics.get('actor/gxpo_trigger_stat')),
                'gxpo/reliability_threshold': scalar(actor_cfg.get('gxpo_tau'), 0.0),
                'gxpo/zscore_window': scalar(actor_cfg.get('gxpo_zscore_w'), 30.0),
                'gxpo/trigger_streak': scalar(metrics.get('actor/gxpo_trigger_streak'), 0.0),
                'gxpo/trigger_candidate': scalar(metrics.get('actor/gxpo_trigger_candidate'), 0.0),
                'gxpo/trigger_patience': scalar(actor_cfg.get('gxpo_trigger_patience'), 1.0),
                'gxpo/entropy_window_ready': scalar(metrics.get('actor/gxpo_entropy_window_ready'), 0.0),
                'gxpo/trigger_warmup_active': scalar(metrics.get('actor/gxpo_trigger_warmup_active'), 0.0),
            })
        elif actor_cfg.get('use_sfpo', False):
            alpha = scalar(actor_cfg.get('sfpo_step_size'), 0.5)
            out.update({
                'sfpo/k': scalar(actor_cfg.get('sfpo_inner_steps'), 10),
                'sfpo/alpha': alpha,
                'sfpo/fast_phase_active': scalar(metrics.get('actor/sfpo_fast_phase_active'), 0.0),
                'sfpo/fallback_triggered': scalar(metrics.get('actor/sfpo_fallback_triggered'), 1.0 if self.stop_SFPO else 0.0),
                'sfpo/fallback_step': float(self.sfpo_trigger_step) if self.stop_SFPO and self.sfpo_trigger_step is not None else None,
                'sfpo/trigger_z': scalar(metrics.get('actor/sfpo_trigger_z'), 0.0),
                'sfpo/trigger_streak': scalar(metrics.get('actor/sfpo_trigger_streak'), 0.0),
                'sfpo/trigger_candidate': scalar(metrics.get('actor/sfpo_trigger_candidate'), 0.0),
                'sfpo/trigger_patience': scalar(actor_cfg.get('sfpo_trigger_patience'), 1.0),
                'sfpo/entropy_window_ready': scalar(metrics.get('actor/sfpo_entropy_window_ready'), 0.0),
                'sfpo/fast_updates_executed_step': scalar(metrics.get('actor/sfpo_fast_updates_executed_step'), 0.0),
                'sfpo/reposition_displacement_norm': scalar(metrics.get('actor/sfpo_reposition_displacement_norm')),
                'sfpo/fresh_grad_norm': scalar(metrics.get('actor/grad_norm')),
            })

        return out

    def _write_efficiency_rows(self, metrics, did_validate):
        row = {'step': int(self.global_steps)}
        row.update(metrics)
        append_jsonl(os.path.join(self._efficiency_run_dir, 'train_metrics.jsonl'), row)
        # Keep the pre-existing filename for older analysis scripts.
        append_jsonl(os.path.join(self._efficiency_run_dir, 'metrics.jsonl'), row)
        if did_validate:
            val_row = {'step': int(self.global_steps), 'eval_greedy/global_step': int(self.global_steps)}
            for key, value in metrics.items():
                if (key.startswith('eval_greedy/') or key.startswith('eff/') or key.startswith('time/') or
                        key.startswith('val/best_')):
                    val_row[key] = value
            validation_path = os.path.join(self._efficiency_run_dir, 'greedy_validation.jsonl')
            keep_last_validations = self.config.trainer.get('keep_last_validations', 0)
            write_retained_jsonl(validation_path, val_row, keep_last_validations)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))
        self._init_efficiency_logging()

        self.global_steps = 0
        self.sampling_num = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', False) and self.global_steps == 0:
            print('Start Initial Eval...')
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        # rebuttal backbone: durable per-step metrics log (independent of wandb)
        fit_start_time = time.monotonic()
        metrics_jsonl_path = os.path.join(self._efficiency_run_dir, 'train_metrics.jsonl')
        os.makedirs(os.path.dirname(metrics_jsonl_path) or '.', exist_ok=True)

        sq_start = 0
        from collections import defaultdict
        data_avg_reward = defaultdict(list)
        data_full_reward = defaultdict(list)
        data_correct_num = defaultdict(list)
        reward_history = []
        skip_log_list = []

        generation_accumulation = None
        if self.config.data.get('generation_accumulation'):
            print('generation_accumulation is true.')
            generation_accumulation = self.config.data.get('generation_accumulation')
            print(f'generation_accumulation: {generation_accumulation}')

        if hasattr(self.config.data, 'real_train_batch_size'):
            real_train_batch_size = self.config.data.real_train_batch_size * self.config.actor_rollout_ref.rollout.n
        else:
            real_train_batch_size = self.config.actor_rollout_ref.actor.ppo_mini_batch_size * self.config.actor_rollout_ref.rollout.n
        dynamic_sampling_batch = None

        if generation_accumulation is not None:
             question_scale_batch_list = []

        adaptive_sampling_bs = self.config.data.get('sampling_batch_size', 0)
        default_sampling_batch_size = self.config.data.get('sampling_batch_size', 0)
        last_sampling = False

        easy_data_num = 0
        hard_data_num = 0
        total_num = 0

        if self.config.trainer.get('max_steps', -1) > 0:
            self.total_training_steps = self.config.trainer.max_steps
            print(f"Set total training steps to {self.total_training_steps}.")

        refresh_generation_time_flag = True
        generation_time = 0

        self.entropy_record = None

        if self.config.data.get('dynamic_filtering', False): acc_batch = None
        for epoch in range(self.start_epoch, self.config.trainer.total_epochs):
            self.current_epoch = epoch
            self.inner_epoch_step = 0

            # torch.cuda.synchronize()           # wait for all GPU work to finish

            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                self.inner_epoch_step += 1

                batch: DataProto = DataProto.from_single_dict(batch_dict)

                if refresh_generation_time_flag:
                    # Reset the generation time at the beginning of each epoch
                    generation_time = 0
                    refresh_generation_time_flag = False
                    generation_start = time.time()

                if self.config.data.get('dynamic_filtering', False):
                    if self.config.data.dynamic_filtering_strategy == 'linear_backoff':
                        batch = self.data_profiler.filter_examples_linear_backoff(epoch, batch, self.config.data.filter_easy_n)
                    elif self.config.data.dynamic_filtering_strategy == 'all_probabilistic':
                        batch, skip_log = self.data_profiler.filter_examples_all_probabilistic(epoch, batch, self.p_easy, self.p_hard, return_log=True)
                        skip_log_list.append([epoch, skip_log])
                    elif self.config.data.dynamic_filtering_strategy == 'keep_all':
                        pass
                    else:
                        raise NotImplementedError(f'Unknown dynamic filtering strategy {self.config.data.dynamic_filtering_strategy}')

                    if acc_batch is None:
                        acc_batch = batch
                    else:
                        acc_batch = DataProto.concat([acc_batch, batch])

                    print(f'Dynamic Filtering: len(batch): {len(batch)}')
                    print(f'Dynamic Filtering: len(acc_batch): {len(acc_batch)}')
                    if len(acc_batch) >= adaptive_sampling_bs:
                        batch = acc_batch
                        acc_batch = None
                        if self.config.data.get('full_sampling_bs', False):
                            batch = select_batch_slice(batch, adaptive_sampling_bs)
                        else:
                            remainder = len(batch) % self.config.trainer.n_gpus_per_node
                            batch = select_batch_slice(batch, len(batch) - remainder)
                        print(f'Dynamic Filtering for Sampling: len(batch): {len(batch)}')
                    else:
                        continue

                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                is_last_step = self.global_steps >= self.total_training_steps
                active_train_start = time.monotonic()

                with _timer('step', timing_raw):
                    # generate a batch
                    print('start generation...')
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    #generation accumulation
                    self.sampling_num += len(batch)

                    # import pdb; pdb.set_trace()
                    if generation_accumulation is not None:
                        if len(question_scale_batch_list) < generation_accumulation - 1:
                            question_scale_batch_list.append(batch)
                            print('Append batch.')
                            print(f'buffer size: {len(question_scale_batch_list)}')
                            continue
                        else:
                            print('Use buffered batch.')
                            print(f'buffer size: {len(question_scale_batch_list)}')
                            question_scale_batch_list.append(batch)
                            batch = DataProto.concat(question_scale_batch_list)
                            question_scale_batch_list = []

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        with _timer('reward', timing_raw):
                            reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics, _ = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    reward_metrics = compute_reward_metrics(batch, self.config)
                    metrics.update(reward_metrics)
                    batch_num = len(batch)

                    easy_data_num += reward_metrics["examples/all_correct_example_ratio"] * batch_num
                    hard_data_num += reward_metrics["examples/format_example_ratio"] * batch_num
                    total_num += batch_num

                    # collect metrics
                    metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))

                    log_index_tensor = torch.tensor(batch.non_tensor_batch['index'].astype(int)).view(-1, self.config.actor_rollout_ref.rollout.n).max(dim=1)[0]
                    log_data_source = batch.non_tensor_batch['data_source'].tolist()[::self.config.actor_rollout_ref.rollout.n]
                    log_reward_tensor = batch.batch['reward'].view(-1, self.config.actor_rollout_ref.rollout.n)
                    log_avg_reward = log_reward_tensor.mean(dim=1).tolist()
                    correct_num = (batch.batch['reward'].view(-1, self.config.actor_rollout_ref.rollout.n) > 0.95).sum(dim=1)

                    # training accuracy / pass metrics over the n rollouts per prompt
                    _train_correct = (log_reward_tensor > 0.95).float()  # (num_prompts, n)
                    train_avg = _train_correct.mean().item()                    # mean accuracy across all samples
                    train_pass_n = _train_correct.max(dim=1)[0].mean().item()     # any-correct per prompt
                    n_rollouts = int(self.config.actor_rollout_ref.rollout.n)
                    # Keep the original dynamic names and also emit the canonical
                    # names used by the GSPO trainer/W&B dashboards.
                    metrics['train/avg_at_n'] = train_avg
                    metrics['train/pass_at_n'] = train_pass_n
                    metrics['train/pass_at_1'] = train_avg
                    metrics['train/accuracy'] = train_avg
                    metrics['train/failure_rate'] = 1.0 - train_avg
                    metrics[f'train/avg_at_{n_rollouts}'] = train_avg
                    metrics[f'train/pass_at_{n_rollouts}'] = train_pass_n

                    reward_history.append((log_data_source, log_index_tensor, log_reward_tensor))
                    for i in range(len(log_index_tensor)):
                        data_full_reward[log_index_tensor[i].item()].append(log_reward_tensor[i])
                        data_avg_reward[log_index_tensor[i].item()].append(log_reward_tensor[i].mean().item())
                        data_correct_num[log_index_tensor[i].item()].append(correct_num[i].item())

                    data_id_list = self.data_profiler.get_data_id_list(log_data_source, log_index_tensor)
                    self.data_profiler.add_reward_list(epoch, data_id_list, log_avg_reward)
                    print(f'len(self.data_profiler): {len(self.data_profiler)}')

                    #### Important point: Generation completed!!!!!!!! #####
                    # When we reach this point, we have finish the example generation and adv computation.
                    # It can involve many dataloader steps.

                    # Compute generation time
                    generation_time += time.time() - generation_start
                    refresh_generation_time_flag = True
                    print(f'Total generation time: {generation_time}')
                    # log generation time in metrics
                    metrics.update({'timing_s/total_generation_time': generation_time})
                    # import ipdb; ipdb.set_trace()

                    # adjust exploration factor
                    hard_data_ratio = hard_data_num / total_num
                    easy_data_ratio = easy_data_num / total_num

                    min_p = self.config.data.get('min_p', 0.05)
                    max_p = self.config.data.get('max_p', 0.95)
                    print(f'min_p: {min_p}, max_p: {max_p}')

                    if epoch > 0 and self.config.data.get('dynamic_filtering_strategy', 'None') == 'all_probabilistic':
                        min_p = self.config.data.get('min_p', 0.05)
                        max_p = self.config.data.get('max_p', 0.95)

                        if hard_data_ratio >= self.targeted_hard: # filter more hard data in future
                            self.p_hard = max(self.p_hard - 0.01, min_p)
                        else: # filter less hard data in future
                            self.p_hard = min(self.p_hard + 0.01, max_p)

                        if easy_data_ratio >= self.targeted_easy: # filter more easy data in future
                            self.p_easy = max(self.p_easy - 0.01, min_p)
                        else: # filter less easy data in future
                            self.p_easy = min(self.p_easy + 0.01, max_p)

                    examples_log = {
                        'examples/too_hard_ratio': hard_data_ratio,
                        'examples/too_easy_ratio': easy_data_ratio,
                        'examples/total_sampled': total_num,
                    }

                    if self.config.data.get('dynamic_filtering_strategy', 'None') == 'all_probabilistic':
                        examples_log['examples/p_easy'] = self.p_easy
                        examples_log['examples/p_hard'] = self.p_hard
                        print(f'p_easy: {self.p_easy}, p_hard: {self.p_hard}')

                    metrics.update(examples_log)

                    # reset ratio statistics
                    easy_data_num = 0
                    hard_data_num = 0
                    total_num = 0

                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    
                    if self.config.trainer.balance_batch:
                        self._balance_batch(batch, metrics=metrics)


                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)
                        
                    actor_cfg = self.config.actor_rollout_ref.actor
                    sfpo_warmup_steps = int(actor_cfg.get('sfpo_warmup_steps', 0))
                    sfpo_trigger_enabled = self.global_steps > sfpo_warmup_steps
                    zscore_w = actor_cfg.get('zscore_w', 0)
                    sfpo_reset_after_warmup = bool(actor_cfg.get('sfpo_reset_entropy_after_warmup', True))
                    if (sfpo_trigger_enabled and sfpo_reset_after_warmup and
                            not self.sfpo_entropy_reset_done):
                        # Do not compare the first post-warmup observation with a
                        # distribution produced during the fast phase.  Start a
                        # fresh baseline and wait for one complete rolling window.
                        self.entropy_container.clear()
                        self.sfpo_entropy_reset_done = True
                        self.sfpo_trigger_streak = 0

                    sfpo_baseline_ready = zscore_w > 0 and len(self.entropy_container) >= zscore_w
                    sfpo_trigger_z = 0.0
                    sfpo_trigger_candidate = False
                    if sfpo_trigger_enabled and sfpo_baseline_ready:
                        u = float(np.mean(self.entropy_container[-zscore_w:]))
                        std = float(np.std(self.entropy_container[-zscore_w:])) + 1e-9
                        sfpo_trigger_z = (self.entropy_container[-1] - u) / std

                        sfpo_trigger_candidate = sfpo_trigger_z >= actor_cfg.zscore_threshold
                        if sfpo_trigger_candidate:
                            self.sfpo_trigger_streak += 1
                        else:
                            self.sfpo_trigger_streak = 0
                        sfpo_trigger_patience = max(1, int(actor_cfg.get('sfpo_trigger_patience', 1)))
                        if self.sfpo_trigger_streak >= sfpo_trigger_patience:
                            self.stop_SFPO = True
                            if self.sfpo_trigger_step is None:
                                self.sfpo_trigger_step = int(self.global_steps)

                    gxpo_trigger_enabled = False
                    gxpo_baseline_ready = False
                    gxpo_trigger_z = 0.0
                    gxpo_trigger_candidate = False
                    gxpo_trigger_stat = (
                        float(self.gxpo_entropy_container[-1])
                        if self.gxpo_entropy_container else 0.0
                    )
                    if actor_cfg.get('use_gxpo', False):
                        gxpo_warmup_steps = int(actor_cfg.get('gxpo_warmup_steps', 0))
                        gxpo_trigger_enabled = self.global_steps > gxpo_warmup_steps
                        gxpo_zscore_w = int(actor_cfg.get('gxpo_zscore_w', 30))
                        gxpo_reset_after_warmup = bool(
                            actor_cfg.get('gxpo_reset_entropy_after_warmup', True))
                        if (gxpo_trigger_enabled and gxpo_reset_after_warmup and
                                not self.gxpo_entropy_reset_done):
                            # Exactly SFPO's warmup boundary: do not compare a
                            # post-warmup value with the fast-phase entropy.
                            self.gxpo_entropy_container.clear()
                            self.gxpo_entropy_reset_done = True
                            self.gxpo_trigger_streak = 0
                            gxpo_trigger_stat = 0.0

                        gxpo_baseline_ready = (
                            gxpo_zscore_w > 0 and
                            len(self.gxpo_entropy_container) >= gxpo_zscore_w)
                        if gxpo_trigger_enabled and gxpo_baseline_ready:
                            # SFPO ordering: score the latest completed outer
                            # batch against the preceding rolling window.
                            u = float(np.mean(self.gxpo_entropy_container[-gxpo_zscore_w:]))
                            std = float(np.std(self.gxpo_entropy_container[-gxpo_zscore_w:])) + 1e-9
                            gxpo_trigger_z = (gxpo_trigger_stat - u) / std
                            gxpo_trigger_candidate = gxpo_trigger_z >= float(
                                actor_cfg.get('gxpo_tau', 3.0))
                            if gxpo_trigger_candidate:
                                self.gxpo_trigger_streak += 1
                            else:
                                self.gxpo_trigger_streak = 0
                            gxpo_trigger_patience = max(
                                1, int(actor_cfg.get('gxpo_trigger_patience', 1)))
                            if (self.gxpo_trigger_streak >= gxpo_trigger_patience and
                                    not self.stop_GXPO):
                                self.stop_GXPO = True
                                if self.gxpo_trigger_step is None:
                                    self.gxpo_trigger_step = int(self.global_steps)
                                print(
                                    f'[GXPO] entropy shutoff triggered at outer batch '
                                    f'{self.global_steps}: z={gxpo_trigger_z:.3f} '
                                    f'>= tau={float(actor_cfg.get("gxpo_tau", 3.0)):.3f} '
                                    f'after {gxpo_trigger_patience} consecutive observations '
                                    f'-> single-pass GRPO from now on')

                    # implement critic warmup
                    ####################################MODIFICATION####################################
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            if self.config.actor_rollout_ref.actor.get('use_gxpo', False):
                                batch.meta_info['gxpo_trigger_enabled'] = (
                                    gxpo_trigger_enabled
                                )
                                batch.meta_info['gxpo_trigger_stop'] = self.stop_GXPO
                                batch.meta_info['gxpo_trigger_z'] = gxpo_trigger_z
                                batch.meta_info['gxpo_trigger_stat'] = gxpo_trigger_stat
                                batch.meta_info['gxpo_trigger_streak'] = self.gxpo_trigger_streak
                                batch.meta_info['gxpo_trigger_candidate'] = gxpo_trigger_candidate
                                batch.meta_info['gxpo_entropy_window_ready'] = gxpo_baseline_ready
                                actor_output = self.actor_rollout_wg.gxpo_update_actor(batch)
                            elif self.config.actor_rollout_ref.actor.get('use_sfpo', False) and not self.stop_SFPO:
                                actor_output = self.actor_rollout_wg.sfpo_update_actor(batch)
                            else:
                                actor_output = self.actor_rollout_wg.update_actor(batch)

                        # reduce_metrics expects scalar/list values.  Performance
                        # memory snapshots are intentionally nested dictionaries;
                        # preserve them separately and merge them back after the
                        # scalar reduction so they remain available to the
                        # durable efficiency logger.
                        actor_raw_metrics = actor_output.meta_info['metrics']
                        actor_nested_metrics = {
                            key: value for key, value in actor_raw_metrics.items()
                            if isinstance(value, dict)
                        }
                        actor_scalar_metrics = {
                            key: value for key, value in actor_raw_metrics.items()
                            if not isinstance(value, dict)
                        }
                        actor_output_metrics = reduce_metrics(actor_scalar_metrics)
                        actor_output_metrics.update(actor_nested_metrics)
                        if self.config.actor_rollout_ref.actor.get('use_gxpo', False):
                            actor_output_metrics.update({
                                'actor/gxpo_trigger_z': gxpo_trigger_z,
                                'actor/gxpo_trigger_stat': gxpo_trigger_stat,
                                'actor/gxpo_trigger_streak': float(self.gxpo_trigger_streak),
                                'actor/gxpo_trigger_candidate': float(gxpo_trigger_candidate),
                                'actor/gxpo_trigger_warmup_active': float(
                                    not (gxpo_trigger_enabled and gxpo_baseline_ready)),
                                'actor/gxpo_entropy_window_ready': float(gxpo_baseline_ready),
                                'actor/gxpo_trigger_patience': float(
                                    max(1, int(actor_cfg.get('gxpo_trigger_patience', 1)))
                                ),
                            })
                            if self.stop_GXPO:
                                actor_output_metrics.update({
                                    'actor/gxpo_fallback_triggered': 1.0,
                                    'actor/gxpo_fallback_step': float(
                                        self.gxpo_trigger_step or self.global_steps),
                                    'actor/gxpo_shutoff_step': float(
                                        self.gxpo_trigger_step or self.global_steps),
                                })
                        if self.config.actor_rollout_ref.actor.get('use_sfpo', False):
                            actor_output_metrics['actor/sfpo_trigger_warmup_active'] = float(
                                not (sfpo_trigger_enabled and sfpo_baseline_ready)
                            )
                            actor_output_metrics.update({
                                'actor/sfpo_trigger_z': sfpo_trigger_z,
                                'actor/sfpo_trigger_streak': float(self.sfpo_trigger_streak),
                                'actor/sfpo_trigger_candidate': float(sfpo_trigger_candidate),
                                'actor/sfpo_trigger_patience': float(
                                    max(1, int(actor_cfg.get('sfpo_trigger_patience', 1)))
                                ),
                                'actor/sfpo_entropy_window_ready': float(sfpo_baseline_ready),
                            })
                        if self.config.actor_rollout_ref.actor.get('use_sfpo', False) and self.stop_SFPO:
                            actor_output_metrics.update({
                                'actor/sfpo_k': float(self.config.actor_rollout_ref.actor.get('sfpo_inner_steps', 10)),
                                'actor/sfpo_alpha': float(self.config.actor_rollout_ref.actor.get('sfpo_step_size', 0.5)),
                                'actor/sfpo_fast_phase_active': 0.0,
                                'actor/sfpo_fallback_triggered': 1.0,
                                'actor/sfpo_fallback_step': float(self.sfpo_trigger_step or self.global_steps),
                                'actor/sfpo_fast_updates_executed_step': 0.0,
                            })
                        metrics.update(actor_output_metrics)
                    ####################################MODIFICATION####################################

                    self.entropy_container.append(float(actor_output_metrics['actor/entropy_loss']))
                    if self.config.actor_rollout_ref.actor.get('use_gxpo', False):
                        self.gxpo_entropy_container.append(
                            float(actor_output_metrics['actor/entropy_loss']))
                    active_train_elapsed_s = time.monotonic() - active_train_start

                    # validate
                    did_validate = False
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        print('start validation...')
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)
                        did_validate = True

                    # Two independent reasons to write a checkpoint: the periodic one that makes the
                    # run resumable, and a new best-pass@1 one that the paper reports. Both write a
                    # normal global_step_N dir; retention keeps the last few and pins the best.
                    need_save = (self.config.trainer.save_freq > 0 and
                                 (is_last_step or self.global_steps % self.config.trainer.save_freq == 0))

                    if did_validate:
                        score = self._best_ckpt_score(val_metrics)
                        metrics['val/best_ckpt_score'] = score
                        if score > getattr(self, '_best_val_score', float('-inf')):
                            print(f'[best-ckpt] step {self.global_steps}: new best pass@1 {score:.4f} '
                                  f'(prev {getattr(self, "_best_val_score", float("-inf")):.4f}); saving')
                            self._best_val_score = score
                            self._best_val_step = self.global_steps
                            need_save = True
                        else:
                            print(f'[best-ckpt] step {self.global_steps}: pass@1 {score:.4f} <= '
                                  f'best {self._best_val_score:.4f} (from step {self._best_val_step})')
                        metrics['val/best_ckpt_step'] = getattr(self, '_best_val_step', 0)
                        metrics['val/best_pass_at_1'] = getattr(self, '_best_val_score', float('-inf'))

                    if need_save:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                    # reward_metrics = compute_reward_metrics(batch, self.config)

                # Efficiency timing intentionally excludes validation and checkpoint overhead.
                metrics.update(self._efficiency_step_metrics(
                    batch=batch, metrics=metrics, timing_raw=timing_raw,
                    active_s=active_train_elapsed_s, fit_start_time=fit_start_time,
                    did_validate=did_validate))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
                n_gpus = self.resource_pool_manager.get_n_gpus()
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                sampling_metircs = {}
                sampling_metircs['sampling/sampling_total_num'] = self.sampling_num
                metrics.update(sampling_metircs)

                metrics['train/elapsed_hours'] = (time.monotonic() - fit_start_time) / 3600.0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)
                self._write_efficiency_rows(metrics, did_validate)

                if is_last_step:
                    write_json(os.path.join(self._efficiency_run_dir, 'summary.json'), {
                        'run_name': os.environ.get('GXPO_RUN_NAME', self.config.trainer.experiment_name),
                        'terminal_step': int(self.global_steps),
                        'terminal_checkpoint': os.path.join(self._efficiency_run_dir, f'global_step_{self.global_steps}'),
                        'final_eval_pending': True,
                        'greedy_validation_only_for_efficiency': True,
                        'headline_wall_clock_metric': 'time/cum_train_active_s',
                        'headline_bp_metric': 'eff/cum_policy_grad_evals',
                    })
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                print(f'epoch: {epoch}')
                print(f'Global_steps done: {self.global_steps}')
                print(f'sampling_num: {self.sampling_num}')
                # reward_metrics = {}
                self.global_steps += 1
                print(f'Start global_steps: {self.global_steps}')

            print(f'Epoch {epoch} done!')
            if hasattr(self.config.trainer, 'save_dir') and self.config.trainer.save_dir:
                if not os.path.exists(self.config.trainer.save_dir):
                    os.makedirs(self.config.trainer.save_dir)
                print(f'Saving data_avg_reward and data_full_reward at {self.config.trainer.save_dir}')
                torch.save(data_avg_reward, f'{self.config.trainer.save_dir}/data_avg_reward.pt')
                torch.save(data_full_reward, f'{self.config.trainer.save_dir}/data_full_reward.pt')
                torch.save(data_correct_num, f'{self.config.trainer.save_dir}/data_correct_num.pt')
                torch.save(reward_history, f'{self.config.trainer.save_dir}/reward_history.pt')
                torch.save(skip_log_list, f'{self.config.trainer.save_dir}/skip_log_list.pt')
