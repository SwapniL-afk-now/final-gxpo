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
TeacherScoringWorker: frozen-teacher top-K scoring for per-step on-policy KD.

Why a plain HuggingFace forward and NOT vLLM
--------------------------------------------
The teacher never generates. It only needs, for each token position of a
sequence the *student* already produced, the teacher's top-K token ids and
their log-probs. That is exactly one teacher-forced forward pass followed by
a top-K over the vocabulary axis.

Routing that through vLLM (the JEGPO `{max_tokens: 1, prompt_logprobs: K}`
trick) is a generation API contorted into a scoring call, and it drags in:
  * a KV-cache pre-allocation (gpu_memory_utilization x total VRAM) that is
    never used, because nothing is decoded,
  * a second engine that must be slept/woken around the student engine,
  * FlashInfer JIT compilation and a CUDA-graph/warmup path,
  * rank-ordered `prompt_logprobs` parsing with per-rank None holes.
None of that buys anything for a single forward pass.

This implementation instead keeps a frozen bf16 HF model and computes
    hidden = model.model(input_ids, attention_mask)      # [B, L, H]
    logits_chunk = model.lm_head(hidden[:, chunk])       # [B, C, V]
    logprobs = logits_chunk - logsumexp(logits_chunk)    # exact log-softmax
    topk over V
The lm_head is applied in *token chunks* so the [B, C, V] logits tensor is
bounded regardless of sequence length -- the same chunking discipline the
repo already uses in `verl/workers/actor/kd_loss.py`.

GPU residency is handled by moving the teacher between CPU and GPU
(`to_gpu()` / `to_cpu()`), which replaces vLLM's sleep/wake and is what lets
the student engine and the teacher stay non-co-resident.
"""
import logging
from typing import List, Optional, Tuple

import numpy as np
import ray
import torch

logger = logging.getLogger(__file__)

# Teacher and student can pad their embedding matrices to different widths even
# when they share a tokenizer (Qwen2.5-Math-7B pads to 152064, Qwen2.5-1.5B to
# 151936). A large *finite* negative logprob keeps a clamped entry inert:
# exp() underflows to exactly 0, so it contributes no teacher mass. A -inf
# sentinel would instead produce 0 * -inf = NaN in the KD loss.
OUT_OF_VOCAB_LOGPROB = -1e4


def _clamp_to_student_vocab(ids: np.ndarray, lps: np.ndarray, student_vocab_size: int,
                            pad_id: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """Neutralize teacher top-K entries outside the student vocabulary. Shape preserving."""
    out_of_range = (ids < 0) | (ids >= student_vocab_size)
    num_clamped = int(out_of_range.sum())
    if num_clamped == 0:
        return ids, lps, 0
    ids = ids.copy()
    lps = lps.copy()
    ids[out_of_range] = pad_id
    lps[out_of_range] = OUT_OF_VOCAB_LOGPROB
    return ids, lps, num_clamped


@ray.remote
class TeacherScoringWorker:
    """Frozen HF teacher, one replica per GPU, parked on CPU when not scoring."""

    def __init__(
        self,
        model_path: str,
        k: int = 32,
        dtype: str = "bfloat16",
        pad_token_id: int = 0,
        student_vocab_size: int = 151936,
        micro_batch_size: int = 4,
        chunk_tokens: int = 1024,
        attn_implementation: str = "flash_attention_2",
        start_on_cpu: bool = True,
    ):
        from transformers import AutoModelForCausalLM

        self.k = k
        self.pad_token_id = pad_token_id
        self.student_vocab_size = student_vocab_size
        self.micro_batch_size = micro_batch_size
        self.chunk_tokens = chunk_tokens
        self._num_ids_clamped = 0
        self.torch_dtype = getattr(torch, dtype)

        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=self.torch_dtype,
            attn_implementation=attn_implementation,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        self.model.requires_grad_(False)
        self._on_gpu = False
        if not start_on_cpu:
            self.to_gpu()

    # ---------------------------------------------------------- residency ----
    def to_gpu(self):
        if not self._on_gpu:
            self.model.to("cuda")
            self._on_gpu = True

    def to_cpu(self):
        if self._on_gpu:
            self.model.to("cpu")
            self._on_gpu = False
            torch.cuda.empty_cache()

    def is_on_gpu(self) -> bool:
        return self._on_gpu

    def num_ids_clamped(self) -> int:
        return self._num_ids_clamped

    # ------------------------------------------------------------ scoring ----
    @torch.no_grad()
    def _score_micro_batch(self, seqs: List[List[int]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        device = next(self.model.parameters()).device
        lengths = [len(s) for s in seqs]
        max_len = max(lengths)
        bsz = len(seqs)

        input_ids = torch.full((bsz, max_len), self.pad_token_id, dtype=torch.long)
        attention_mask = torch.zeros((bsz, max_len), dtype=torch.long)
        for i, s in enumerate(seqs):
            input_ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
            attention_mask[i, :len(s)] = 1
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)

        # Hidden states only: [B, L, H] is small; the [B, L, V] logits are the
        # memory hazard and are materialized one chunk at a time below.
        base = getattr(self.model, "model", None)
        if base is None:  # defensive: non-standard wrapper
            base = self.model.base_model
        hidden = base(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).last_hidden_state

        # Position j of the logits predicts the token at position j+1, so the
        # scored rows cover sequence positions 1..L-1 -- the same convention the
        # driver's response-span slice assumes.
        hidden = hidden[:, :-1, :]
        scored_len = hidden.shape[1]

        topk_ids = torch.empty((bsz, scored_len, self.k), dtype=torch.long, device=device)
        topk_lps = torch.empty((bsz, scored_len, self.k), dtype=torch.float32, device=device)

        for start in range(0, scored_len, self.chunk_tokens):
            end = min(start + self.chunk_tokens, scored_len)
            logits = self.model.lm_head(hidden[:, start:end, :]).float()
            # exact log-softmax without materializing a second [B, C, V] tensor
            logits -= torch.logsumexp(logits, dim=-1, keepdim=True)
            vals, idx = torch.topk(logits, self.k, dim=-1)
            topk_ids[:, start:end] = idx
            topk_lps[:, start:end] = vals
            del logits, vals, idx

        topk_ids_np = topk_ids.cpu().numpy()
        topk_lps_np = topk_lps.cpu().numpy()
        del hidden, topk_ids, topk_lps

        results = []
        for i, L in enumerate(lengths):
            # drop padded tail: sequence i contributes L-1 scored rows
            ids_i = topk_ids_np[i, :L - 1]
            lps_i = np.nan_to_num(topk_lps_np[i, :L - 1], nan=OUT_OF_VOCAB_LOGPROB,
                                  posinf=0.0, neginf=OUT_OF_VOCAB_LOGPROB)
            ids_i, lps_i, n_clamped = _clamp_to_student_vocab(
                ids_i, lps_i, self.student_vocab_size, self.pad_token_id)
            self._num_ids_clamped += n_clamped
            results.append((ids_i, lps_i))
        return results

    def score_batch(self, sequence_id_lists: List[List[int]]) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Score full (prompt+response) token-id sequences.

        Returns one (ids [S_i, k] int64, logprobs [S_i, k] float32) pair per
        input sequence, where S_i = len(sequence_id_lists[i]) - 1.
        """
        if not self._on_gpu:
            raise RuntimeError("TeacherScoringWorker.score_batch called while parked on CPU; "
                               "call to_gpu() first.")
        # The two Qwen2.5 tokenizers are intentionally shared. Fail early if
        # a caller accidentally supplies a sequence containing an ID outside
        # the teacher vocabulary rather than producing a misleading KD target.
        teacher_vocab_size = int(getattr(self.model.config, "vocab_size", 0))
        max_input_id = max((max(s) for s in sequence_id_lists if s), default=-1)
        if max_input_id >= teacher_vocab_size:
            raise ValueError(
                f"Student input id {max_input_id} is outside teacher vocabulary {teacher_vocab_size}; "
                "use a teacher with the same tokenizer/id mapping."
            )
        # Length-sorted micro-batches keep padding waste low; original order is
        # restored before returning.
        order = sorted(range(len(sequence_id_lists)), key=lambda i: len(sequence_id_lists[i]))
        out: List[Optional[Tuple[np.ndarray, np.ndarray]]] = [None] * len(sequence_id_lists)
        for start in range(0, len(order), self.micro_batch_size):
            idxs = order[start:start + self.micro_batch_size]
            res = self._score_micro_batch([sequence_id_lists[i] for i in idxs])
            for local_i, global_i in enumerate(idxs):
                out[global_i] = res[local_i]
        return out
