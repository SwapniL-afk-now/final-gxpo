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
Offline KD-SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

from typing import List, Union
import hashlib
import json

import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer


def tokenizer_fingerprint(tokenizer) -> str:
    """Stable identity for token IDs, special tokens, and chat formatting."""
    payload = {
        "vocab": sorted((str(k), int(v)) for k, v in tokenizer.get_vocab().items()),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "length": int(len(tokenizer)),
        "special_tokens_map": tokenizer.special_tokens_map,
        "all_special_ids": [int(v) for v in tokenizer.all_special_ids],
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class KDSFTDataset(Dataset):
    """
    This is an in-memory offline KD-SFT dataset
    """

    def __init__(self,
                 parquet_files: Union[str, List[str]],
                 tokenizer,
                 prompt_key='prompt',
                 prompt_dict_keys=None,
                 response_key='response',
                 response_dict_keys=None,
                 max_length=1024,
                 truncation='error',
                 teacher_topk_log_probs_key='teacher_topk_log_probs',
                 teacher_topk_ids_key='teacher_topk_ids',
                 teacher_topk=32,
                 response_ids_key='response_ids'):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation
        self.teacher_topk_log_probs_key = teacher_topk_log_probs_key
        self.teacher_topk_ids_key = teacher_topk_ids_key
        self.teacher_topk = int(teacher_topk)
        self.response_ids_key = response_ids_key
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]
        else:
            # Accept list/tuple/hydra ListConfig alike.
            parquet_files = list(parquet_files)

        self.parquet_files = parquet_files
        if isinstance(tokenizer, str):
            tokenizer = hf_tokenizer(tokenizer)
        self.tokenizer: PreTrainedTokenizer = tokenizer

        self.prompt_key = prompt_key if isinstance(prompt_key, (tuple, list)) else [prompt_key]
        self.response_key = response_key if isinstance(response_key, (tuple, list)) else [response_key]
        self.prompt_dict_keys = [] if not prompt_dict_keys else prompt_dict_keys
        self.response_dict_keys = [] if not response_dict_keys else response_dict_keys

        self.max_length = max_length

        self._download()
        self._read_files_and_tokenize()

    def _download(self):
        for i, parquet_file in enumerate(self.parquet_files):
            self.parquet_files[i] = copy_to_local(parquet_file, verbose=True)

    def _read_files_and_tokenize(self):

        def series_to_item(ls):
            import pandas, numpy
            while isinstance(ls, (pandas.core.series.Series, numpy.ndarray)) and len(ls) == 1:
                ls = ls[0]
            return ls

        dataframes = []
        for parquet_file in self.parquet_files:
            # read parquet files and cache
            dataframe = pd.read_parquet(parquet_file)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes)
        self.prompts = self.dataframe[self.prompt_key]
        for key in self.prompt_dict_keys:
            # type(x): pandas.core.series.Series
            # type(x[0]): numpy.ndarray
            # type(x[0][0]): dict
            try:
                self.prompts = self.prompts.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.prompts={self.prompts}')
                raise
        if isinstance(self.prompts, pd.DataFrame):
            self.prompts = self.prompts.iloc[:, 0]
        self.prompts = self.prompts.tolist()
        self.responses = self.dataframe[self.response_key]
        for key in self.response_dict_keys:
            try:
                self.responses = self.responses.apply(lambda x: series_to_item(x)[key], axis=1)
            except Exception:
                print(f'self.responses={self.responses}')
                raise
        if isinstance(self.responses, pd.DataFrame):
            self.responses = self.responses.iloc[:, 0]
        self.responses = self.responses.tolist()

        required_keys = (
            self.teacher_topk_log_probs_key, self.teacher_topk_ids_key,
            self.response_ids_key,
            'kd_student_tokenizer_fingerprint',
            'kd_teacher_tokenizer_fingerprint',
        )
        for key in required_keys:
            if key not in self.dataframe.columns:
                raise ValueError(
                    f'KD cache key {key!r} not in {list(self.dataframe.columns)}; '
                    'rebuild the cache with tools/kd_sft/build_teacher_topk.py')
        expected_fp = tokenizer_fingerprint(self.tokenizer)
        student_fps = set(self.dataframe['kd_student_tokenizer_fingerprint'].astype(str))
        teacher_fps = set(self.dataframe['kd_teacher_tokenizer_fingerprint'].astype(str))
        if student_fps != {expected_fp} or teacher_fps != {expected_fp}:
            raise ValueError(
                'KD cache tokenizer identity does not match the student tokenizer; '
                'teacher and student vocab/merges/special tokens/chat template must be identical')
        self.teacher_log_probs = self.dataframe[self.teacher_topk_log_probs_key].tolist()
        self.teacher_ids = self.dataframe[self.teacher_topk_ids_key].tolist()
        self.response_ids = self.dataframe[self.response_ids_key].tolist()
    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        # apply chat template
        prompt_chat = [{'role': 'user', 'content': prompt}]

        # string
        prompt_chat_str = tokenizer.apply_chat_template(prompt_chat, add_generation_prompt=True, tokenize=False)
        response_chat_str = response + tokenizer.eos_token

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]

        # Use the exact student-tokenizer IDs captured during cache generation.
        # This prevents response text detokenize/retokenize drift.
        import numpy as np
        response_ids = torch.tensor(np.asarray(self.response_ids[item]).flatten(), dtype=torch.long)
        response_attention_mask = torch.ones_like(response_ids)

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if sequence_length < self.max_length:
            padded_input_ids = torch.ones(size=(self.max_length - sequence_length,),
                                          dtype=input_ids.dtype) * self.tokenizer.pad_token_id
            padded_attention_mask = torch.zeros(size=(self.max_length - sequence_length,), dtype=attention_mask.dtype)

            input_ids = torch.cat((input_ids, padded_input_ids))
            attention_mask = torch.cat((attention_mask, padded_attention_mask))
        elif sequence_length > self.max_length:
            if self.truncation == 'left':
                # actually, left truncation may not be reasonable
                input_ids = input_ids[-self.max_length:]
                attention_mask = attention_mask[-self.max_length:]
            elif self.truncation == 'right':
                input_ids = input_ids[:self.max_length]
                attention_mask = attention_mask[:self.max_length]
            elif self.truncation == 'error':
                raise NotImplementedError(f'{sequence_length=} is larger than {self.max_length=}')
            else:
                raise NotImplementedError(f'Unknown truncation method {self.truncation}')

        position_ids = compute_position_id_with_mask(attention_mask)

        loss_mask = attention_mask.clone()
        if prompt_length > 1:
            # mask out prompt for SFT.
            loss_mask[:min(prompt_length, loss_mask.size(0)) - 1] = 0
        # mask out the last token in response
        loss_mask[min(prompt_length + response_length, loss_mask.size(0)) - 1] = 0

        out = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }
        # Canvas is aligned to input positions. The trainer shifts [:, 1:] in
        # the same way as labels and consumes only response loss positions.
        def _to_dense_topk(value, dtype):
            arr = value if isinstance(value, np.ndarray) else np.asarray(value)
            if arr.dtype == object:
                arr = np.stack([np.asarray(row, dtype=dtype) for row in arr])
            return np.asarray(arr, dtype=dtype)

        t_lp = _to_dense_topk(self.teacher_log_probs[item], np.float32)
        t_id = _to_dense_topk(self.teacher_ids[item], np.int64)
        vocab_size = len(tokenizer)
        if response_ids.numel() and (int(response_ids.min()) < 0 or int(response_ids.max()) >= vocab_size):
            raise ValueError(f'KD cache response_ids row {item} contains IDs outside student vocab {vocab_size}')
        if t_id.size and (int(t_id.min()) < 0 or int(t_id.max()) >= vocab_size):
            raise ValueError(f'KD cache teacher_topk_ids row {item} contains IDs outside student vocab {vocab_size}')
        if t_lp.ndim != 2 or t_lp.shape[1] != self.teacher_topk or t_lp.shape != t_id.shape:
            raise ValueError(f'KD cache row {item} has shape {t_lp.shape}/{t_id.shape}; '
                             f'expected [R, {self.teacher_topk}]')
        if t_lp.shape[0] != response_length:
            raise ValueError(f'KD cache row {item} has {t_lp.shape[0]} response tokens, '
                             f'but exact response_ids has {response_length}')
        canvas_lp = torch.zeros(self.max_length, self.teacher_topk, dtype=torch.float32)
        canvas_id = torch.zeros(self.max_length, self.teacher_topk, dtype=torch.long)
        start = min(prompt_length, self.max_length)
        end = min(prompt_length + response_length, self.max_length)
        take = max(0, end - start)
        if take:
            canvas_lp[start:end] = torch.from_numpy(t_lp[:take])
            canvas_id[start:end] = torch.from_numpy(t_id[:take])
        out['teacher_topk_log_probs'] = canvas_lp
        out['teacher_topk_ids'] = canvas_id
        return out
