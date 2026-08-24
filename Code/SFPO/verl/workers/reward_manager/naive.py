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
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

from verl import DataProto
from verl.utils.reward_score import _default_compute_score
import torch
import os


class NaiveRewardManager:
    """The reward manager.

    Scoring is dispatched across a persistent ProcessPoolExecutor: code rewards
    (taco/apps prime_code unit-test execution) are seconds each and were the
    dominant cost when run in a serial loop (~900s/step on 256 completions while
    hundreds of CPUs sat idle). The pool is created once and reused every step, so
    cheap rewards (math string-match) pay the fork cost once, not per step, and
    their results are byte-identical to the old serial path. Worker count via
    REWARD_NUM_WORKERS (default min(128, nproc)).
    """

    def __init__(self, tokenizer, num_examine, compute_score=None) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or _default_compute_score
        self._executor = None

    def _pool(self):
        if self._executor is None:
            n = int(os.environ.get("REWARD_NUM_WORKERS", min(64, os.cpu_count() or 8)))
            # 'spawn': fresh interpreters with no inherited CUDA state, so prime_code's
            # inner multiprocessing.Process (nested fork) doesn't trip torch's
            # "not valid in a forked process" guard that killed the default-fork pool.
            self._executor = ProcessPoolExecutor(
                max_workers=max(1, n),
                mp_context=multiprocessing.get_context("spawn"),
            )
        return self._executor

    def _score_all(self, args_list):
        """args_list: list of dicts of compute_score kwargs. Returns list of scores.
        Runs in parallel; falls back to serial if the pool can't be used (e.g. an
        unpicklable custom compute_score)."""
        try:
            pool = self._pool()
            futures = [pool.submit(self.compute_score, **kw) for kw in args_list]
            return [f.result() for f in futures]
        except Exception as e:
            print(f"[NaiveRewardManager] parallel scoring failed ({e}); falling back to serial.")
            self._executor = None
            return [self.compute_score(**kw) for kw in args_list]

    def _decode(self, data):
        """Decode prompts/responses and collect compute_score kwargs + placement info."""
        args_list, valid_response_lengths, data_sources, response_strs = [], [], [], []
        for i in range(len(data)):
            data_item = data[i]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            data_source = data_item.non_tensor_batch['data_source']
            args_list.append(dict(
                prompt_str=prompt_str,
                data_source=data_source,
                solution_str=response_str,
                ground_truth=data_item.non_tensor_batch['reward_model']['ground_truth'],
                extra_info=data_item.non_tensor_batch.get('extra_info', None),
            ))
            valid_response_lengths.append(valid_response_length)
            data_sources.append(data_source)
            response_strs.append(response_str)
        return args_list, valid_response_lengths, data_sources, response_strs

    def verify(self, data):
        args_list, _, _, _ = self._decode(data)
        scores = self._score_all(args_list)
        data.batch['acc'] = torch.tensor(scores, dtype=torch.float32,
                                         device=data.batch['prompts'].device)
        return scores

    def __call__(self, data: DataProto):
        """We will expand this function gradually based on the available datasets"""

        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if 'rm_scores' in data.batch.keys():
            return data.batch['rm_scores']

        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        already_print_data_sources = {}

        args_list, valid_response_lengths, data_sources, response_strs = self._decode(data)
        scores = self._score_all(args_list)

        for i in range(len(data)):
            data_source = data_sources[i]
            reward_tensor[i, valid_response_lengths[i] - 1] = scores[i]

            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                if os.environ.get('GXPO_CONCISE_LOGS') != '1':
                    print("[prompt]", args_list[i]['prompt_str'])
                    print("[response]", response_strs[i])
                    print("[ground_truth]", args_list[i]['ground_truth'])
                    print("[score]", scores[i])

        return reward_tensor
