"""Off-policy KD dataset: fixed prompt+response pairs with cached teacher top-K.

Produces the exact batch layout ``generate_sequences`` returns for on-policy
rollouts (prompts left-padded, responses right-padded, EOS-truncated attention
via ``get_eos_mask``, continued position ids), plus the cached teacher columns
that ``score_batch_and_attach`` would otherwise attach per step:

  tensors: input_ids, attention_mask, position_ids, responses,
           teacher_topk_ids, teacher_topk_log_probs
  non-tensors: data_source, ground_truth, reward, index, prompt_text,
               response_text

Every cached response must end with the EOS id (enforced by W0 prep); the
constructor fails fast otherwise, because an unterminated row cannot teach
stopping and would recreate the on-policy length runaway.
"""
from __future__ import annotations

import copy
from typing import List, Optional, Union

import numpy as np
import pandas as pd
import torch
from omegaconf import ListConfig
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask

OUT_OF_VOCAB_LOGPROB = -1e4


class OffPolicyKDDataset(Dataset):
    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer: PreTrainedTokenizer,
                 max_prompt_length: int = 1024,
                 response_length: int = 3072,
                 teacher_topk: int = 16,
                 cache_dir: str = '~/.cache/verl/rlhf'):
        if not isinstance(parquet_files, (List, ListConfig)):
            parquet_files = [parquet_files]
        self.tokenizer = tokenizer
        self.max_prompt_length = int(max_prompt_length)
        self.response_length = int(response_length)
        self.teacher_topk = int(teacher_topk)
        pad = tokenizer.pad_token_id
        self.pad_token_id = pad if pad is not None else tokenizer.eos_token_id
        self.eos_token_id = tokenizer.eos_token_id

        frames = [pd.read_parquet(f) for f in parquet_files]
        self.dataframe = pd.concat(frames, ignore_index=True)
        print(f'off-policy KD dataset len: {len(self.dataframe)}')

        # Fail fast: every fixed response must terminate; the KD mask covers
        # the EOS position, which is the only way the student learns to stop.
        for i, r in enumerate(self.dataframe['response_ids']):
            ids = list(map(int, r))
            assert len(ids) <= self.response_length, f'row {i}: response {len(ids)} > {self.response_length}'
            assert ids and ids[-1] == self.eos_token_id, f'row {i}: missing trailing EOS'
            assert len(self.dataframe['prompt_ids'].iloc[i]) <= self.max_prompt_length, \
                f'row {i}: prompt too long'
        tids = self.dataframe['teacher_topk_ids']
        assert all(len(t) == len(self.dataframe['response_ids'].iloc[i])
                   for i, t in enumerate(tids)), 'teacher cache / response length mismatch'

    def __len__(self):
        return len(self.dataframe)

    def _pad_left(self, ids: List[int], target: int) -> torch.Tensor:
        ids = list(map(int, ids))[-target:]
        return torch.tensor([self.pad_token_id] * (target - len(ids)) + ids, dtype=torch.long)

    def _pad_right(self, ids: List[int], target: int, pad: int) -> torch.Tensor:
        ids = list(map(int, ids))[:target]
        return torch.tensor(ids + [pad] * (target - len(ids)), dtype=torch.long)

    def __getitem__(self, item):
        row = self.dataframe.iloc[item]
        prompt_ids = list(map(int, row['prompt_ids']))
        response_ids = list(map(int, row['response_ids']))
        K = self.teacher_topk

        prompt_t = self._pad_left(prompt_ids, self.max_prompt_length)
        prompt_mask = (prompt_t != self.pad_token_id).long()
        response_t = self._pad_right(response_ids, self.response_length, self.pad_token_id)
        eos_mask = verl_F.get_eos_mask(response_t.unsqueeze(0), self.eos_token_id).squeeze(0).long()

        input_ids = torch.cat([prompt_t, response_t])
        attention_mask = torch.cat([prompt_mask, eos_mask])
        prompt_pos = compute_position_id_with_mask(prompt_mask.unsqueeze(0)).squeeze(0)
        delta = torch.arange(1, self.response_length + 1)
        response_pos = prompt_pos[-1:] + delta
        position_ids = torch.cat([prompt_pos, response_pos])

        # Cells may come back from parquet as object arrays of per-row arrays;
        # list() first so the stack is always dense.
        t_ids = np.asarray(list(row['teacher_topk_ids']), dtype=np.int64)
        t_lps = np.asarray(list(row['teacher_topk_logps']), dtype=np.float32)
        assert t_ids.shape == (len(response_ids), K), f'row {item}: bad cache shape {t_ids.shape}'
        n = t_ids.shape[0]
        tids_padded = np.full((self.response_length, K), self.pad_token_id, dtype=np.int64)
        tlps_padded = np.full((self.response_length, K), OUT_OF_VOCAB_LOGPROB, dtype=np.float32)
        tids_padded[:n] = t_ids
        tlps_padded[:n] = t_lps

        # The naive reward manager reads non_tensor_batch['reward_model']
        # ['ground_truth'] per row; build the dict at load so no parquet
        # struct round-trip is involved.
        reward_model = {'style': 'rule', 'ground_truth': str(row['ground_truth'])}
        return {
            # 'prompts' is consumed by the naive reward manager for decoding.
            'prompts': prompt_t,
            'reward_model': reward_model,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'responses': response_t,
            'teacher_topk_ids': torch.from_numpy(tids_padded),
            'teacher_topk_log_probs': torch.from_numpy(tlps_padded),
            'data_source': str(row['data_source']),
            'ground_truth': str(row['ground_truth']),
            'reward': float(row['reward']),
            # fit() does _group_index.astype(int): positional int, stable per
            # row. The uuid string rides along separately for provenance.
            'index': int(item),
            'source_index': str(row.get('source_index', '')),
            'prompt_text': str(row.get('prompt_text', '')),
            'response_text': str(row.get('response_text', '')),
        }
