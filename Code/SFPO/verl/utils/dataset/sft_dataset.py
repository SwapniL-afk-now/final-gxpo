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
SFT dataset
- We assume user pass a single parquet file.
- We load all the data into the memory.
Each parquet file contains
"""

import os
from pathlib import Path
from typing import List, Union

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, PreTrainedTokenizer

from verl.utils.fs import copy_to_local
from verl.utils.model import compute_position_id_with_mask
from verl.utils import hf_tokenizer
from verl.utils.dataset.kd_utils import (
    SparseTeacherStore,
    format_prompt_response,
    tokenizer_fingerprint,
)


class SFTDataset(Dataset):
    """
    This is an in-memory SFTDataset
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
                 kd_enabled=False,
                 kd_topk=32):
        assert truncation in ['error', 'left', 'right']
        self.truncation = truncation

        # Hydra supplies list-valued overrides as ``ListConfig`` rather than a
        # native list. Normalize any iterable path collection before calling
        # copy_to_local, otherwise the whole ListConfig is treated as one path.
        if isinstance(parquet_files, (str, bytes, os.PathLike)):
            parquet_files = [os.fspath(parquet_files)]
        else:
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
        self.kd_enabled = bool(kd_enabled)
        self.kd_topk = int(kd_topk)
        if self.kd_enabled and self.kd_topk <= 0:
            raise ValueError('kd_topk must be positive when KD is enabled')

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
            dataframe['kd_parquet_dir_internal'] = str(Path(parquet_file).resolve().parent)
            dataframes.append(dataframe)
        self.dataframe = pd.concat(dataframes, ignore_index=True)
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

        self._kd_stores = {}
        self._kd_refs = None
        if self.kd_enabled:
            required = {
                'kd_row_index', 'kd_token_count', 'kd_sidecar',
                'teacher_tokenizer_fingerprint', 'teacher_topk', 'source',
            }
            missing = required.difference(self.dataframe.columns)
            if missing:
                raise ValueError(f'KD dataset is missing columns: {sorted(missing)}')
            student_fingerprint = tokenizer_fingerprint(self.tokenizer)
            self._kd_refs = []
            for row in self.dataframe.itertuples(index=False):
                if int(row.teacher_topk) != self.kd_topk:
                    raise ValueError(
                        f'KD row top-K {row.teacher_topk} does not match config {self.kd_topk}'
                    )
                if str(row.teacher_tokenizer_fingerprint) != student_fingerprint:
                    raise ValueError('KD row tokenizer fingerprint does not match the student')
                sidecar = Path(str(row.kd_sidecar))
                if not sidecar.is_absolute():
                    sidecar = Path(str(row.kd_parquet_dir_internal)) / sidecar
                sidecar = sidecar.resolve()
                key = str(sidecar)
                if key not in self._kd_stores:
                    self._kd_stores[key] = SparseTeacherStore(
                        sidecar,
                        expected_topk=self.kd_topk,
                        expected_tokenizer=student_fingerprint,
                    )
                self._kd_refs.append((
                    key, int(row.kd_row_index), int(row.kd_token_count), str(row.source)
                ))

    def __len__(self):
        return len(self.prompts)

    def __getitem__(self, item):
        tokenizer = self.tokenizer

        prompt = self.prompts[item]
        response = self.responses[item]

        prompt_chat_str, response_chat_str = format_prompt_response(
            tokenizer, prompt, response
        )

        # tokenize
        prompt_ids_output = tokenizer(prompt_chat_str, return_tensors='pt', add_special_tokens=False)
        prompt_ids = prompt_ids_output['input_ids'][0]
        prompt_attention_mask = prompt_ids_output['attention_mask'][0]

        response_ids_output = tokenizer(response_chat_str, return_tensors='pt', add_special_tokens=False)
        response_ids = response_ids_output['input_ids'][0]
        response_attention_mask = response_ids_output['attention_mask'][0]

        prompt_length = prompt_ids.shape[0]
        response_length = response_ids.shape[0]

        input_ids = torch.cat((prompt_ids, response_ids), dim=-1)
        attention_mask = torch.cat((prompt_attention_mask, response_attention_mask), dim=-1)

        # padding to max length
        sequence_length = input_ids.shape[0]
        if self.kd_enabled and sequence_length > self.max_length:
            raise ValueError(
                f'KD row {item} has {sequence_length} tokens and cannot be truncated to '
                f'{self.max_length}'
            )
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

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'position_ids': position_ids,
            'loss_mask': loss_mask
        }
        if self.kd_enabled:
            if prompt_length <= 0:
                raise ValueError(f'KD row {item} has an empty prompt')
            store_key, row_index, expected_count, source = self._kd_refs[item]
            teacher = self._kd_stores[store_key].row(row_index)
            actual_count = len(teacher['token_ids'])
            if actual_count != expected_count or actual_count != response_length:
                raise ValueError(
                    f'KD row {item} token count mismatch: sidecar={actual_count}, '
                    f'parquet={expected_count}, tokenizer={response_length}'
                )
            response_numpy = response_ids.detach().cpu().numpy().astype(np.int32, copy=False)
            if not np.array_equal(teacher['token_ids'], response_numpy):
                raise ValueError(
                    f'KD row {item} response token IDs do not match the student tokenizer'
                )

            shifted_length = self.max_length - 1
            teacher_ids = torch.zeros((shifted_length, self.kd_topk), dtype=torch.int32)
            teacher_logprobs = torch.full(
                (shifted_length, self.kd_topk), float('-inf'), dtype=torch.float16
            )
            teacher_counts = torch.zeros(shifted_length, dtype=torch.uint8)
            teacher_mass = torch.zeros(shifted_length, dtype=torch.float16)
            teacher_mask = torch.zeros(shifted_length, dtype=torch.bool)
            loss_start = prompt_length - 1
            loss_end = loss_start + response_length
            if loss_start < 0 or loss_end > shifted_length:
                raise ValueError(f'KD row {item} does not fit the shifted loss sequence')
            teacher_ids[loss_start:loss_end] = torch.from_numpy(
                np.array(teacher['topk_ids'], dtype=np.int32, copy=True)
            )
            teacher_logprobs[loss_start:loss_end] = torch.from_numpy(
                np.array(teacher['topk_logprobs'], dtype=np.float16, copy=True)
            )
            teacher_counts[loss_start:loss_end] = torch.from_numpy(
                np.array(teacher['topk_counts'], dtype=np.uint8, copy=True)
            )
            teacher_mass[loss_start:loss_end] = torch.from_numpy(
                np.array(teacher['topk_mass'], dtype=np.float16, copy=True)
            )
            teacher_mask[loss_start:loss_end] = True
            source_id = (
                0 if source == 'taco_verified_teacher'
                else 1 if source == 'taco_dataset_solution_fallback'
                else 2
            )
            result.update({
                'kd_teacher_ids': teacher_ids,
                'kd_teacher_logprobs': teacher_logprobs,
                'kd_teacher_counts': teacher_counts,
                'kd_teacher_mass': teacher_mass,
                'kd_token_mask': teacher_mask,
                'kd_source_id': torch.tensor(source_id, dtype=torch.int8),
            })
        return result
