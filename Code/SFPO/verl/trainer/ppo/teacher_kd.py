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
Driver-side glue for per-step on-policy KD (Option B: separate teacher Ray
group).

Owns:
  * spawning the frozen-teacher TeacherScoringWorker replicas onto the same
    GPU placement group used by the student actor/rollout pool,
  * the phased GPU residency that keeps the frozen teacher and the student
    vLLM engine from being co-resident (teacher is parked on CPU between steps);
    waking/sleeping transfers the full HF teacher and is an intentional per-step cost,
  * fanning the current on-policy batch out to the teacher replicas and
    re-assembling a dense [B, response_length, K] teacher_topk_ids /
    teacher_topk_log_probs pair that `verl.workers.actor.dp_actor`'s existing
    KD branch consumes unmodified.
"""
import logging
from typing import List, Optional

import numpy as np
import ray
import torch
from omegaconf import DictConfig

from verl import DataProto
from verl.workers.teacher_scoring_worker import (
    OUT_OF_VOCAB_LOGPROB,
    TeacherScoringWorker,
)

logger = logging.getLogger(__file__)


def build_teacher_group(
    config: DictConfig,
    resource_pool,
    num_replicas: int = 2,
) -> List["ray.actor.ActorHandle"]:
    """Spawn `num_replicas` TeacherScoringWorker actors, TP=1 each, packed onto
    the same placement group (and therefore the same locked GPUs) as the
    student actor/rollout resource pool.
    """
    from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

    kd_cfg = config.actor_rollout_ref.actor
    teacher_cfg = kd_cfg.get("kd_teacher", {}) or {}

    model_path = kd_cfg.get("kd_teacher_path", None) or teacher_cfg.get("path", None)
    if not model_path:
        raise ValueError("actor.kd_teacher_path (or actor.kd_teacher.path) must be set when actor.use_kd=True")

    k = int(kd_cfg.get("kd_topk", 32))
    dtype = teacher_cfg.get("dtype", "bfloat16")
    student_vocab_size = int(teacher_cfg.get("student_vocab_size", 151936))
    pad_token_id = int(teacher_cfg.get("pad_token_id", 0))
    micro_batch_size = int(teacher_cfg.get("micro_batch_size", 4))
    chunk_tokens = int(teacher_cfg.get("chunk_tokens", 1024))
    attn_implementation = teacher_cfg.get("attn_implementation", "flash_attention_2")

    pgs = resource_pool.get_placement_groups()
    bundle_slots = []
    for pg in pgs:
        for bundle_idx in range(pg.bundle_count):
            bundle_slots.append((pg, bundle_idx))
    if not bundle_slots:
        raise ValueError("Shared resource pool exposes no GPU bundles for the teacher group.")
    if num_replicas <= 0:
        raise ValueError(f"kd_teacher.num_replicas must be positive, got {num_replicas}")
    if num_replicas > len(bundle_slots):
        # One teacher replica per available GPU: asking for more would stack two
        # 7B teachers on one device for no throughput gain.
        logger.warning("Requested %d teacher replicas but only %d GPU bundles are available; "
                       "clamping to %d.", num_replicas, len(bundle_slots), len(bundle_slots))
        num_replicas = len(bundle_slots)

    # Each bundle declares 1 logical GPU and the student actor/rollout worker
    # already holds a fractional share of it (verl's own colocation convention,
    # single_controller/ray/base.py). The teacher is parked on CPU except during
    # its scoring phase, so it takes the same small fractional share rather than
    # a whole GPU, which would over-subscribe the bundle. Ray fractions do not
    # partition VRAM; max_colocate_count must be >1 for a teacher lease to fit.
    max_colocate_count = int(getattr(resource_pool, "max_collocate_count", 1))
    if max_colocate_count <= 1:
        raise ValueError(
            "KD teacher placement requires trainer.max_colocate_count > 1; "
            f"got {max_colocate_count}. Leave it unset for the KD default or set it to 10."
        )
    teacher_num_gpus = 1.0 / float(max_colocate_count)

    handles = []
    for i in range(num_replicas):
        pg, bundle_idx = bundle_slots[i]
        actor_cls = TeacherScoringWorker.options(
            num_gpus=teacher_num_gpus,
            scheduling_strategy=PlacementGroupSchedulingStrategy(
                placement_group=pg,
                placement_group_bundle_index=bundle_idx,
            ),
        )
        handle = actor_cls.remote(
            model_path=model_path,
            k=k,
            dtype=dtype,
            pad_token_id=pad_token_id,
            student_vocab_size=student_vocab_size,
            micro_batch_size=micro_batch_size,
            chunk_tokens=chunk_tokens,
            attn_implementation=attn_implementation,
            start_on_cpu=True,
        )
        handles.append(handle)
    # block until every replica has finished loading weights (on CPU)
    ray.get([h.is_on_gpu.remote() for h in handles])
    return handles


def sleep_teachers(handles: List["ray.actor.ActorHandle"]) -> None:
    """Park every teacher replica back on CPU and release its VRAM."""
    ray.get([h.to_cpu.remote() for h in handles])


def wake_teachers(handles: List["ray.actor.ActorHandle"]) -> None:
    """Move every teacher replica onto its GPU for a scoring phase."""
    ray.get([h.to_gpu.remote() for h in handles])


def _row_to_unpadded_ids(input_ids_row: torch.Tensor, attention_mask_row: torch.Tensor) -> List[int]:
    valid = attention_mask_row.bool()
    return input_ids_row[valid].tolist()


def score_batch_and_attach(
    batch: DataProto,
    handles: List["ray.actor.ActorHandle"],
    response_length: int,
    k: int,
    pad_token_id: int = 0,
) -> DataProto:
    """Fan `batch`'s sequences out to the teacher replicas, then union dense
    [B, response_length, k] `teacher_topk_ids` / `teacher_topk_log_probs`
    tensors (response-span sliced, right-padded) back onto `batch`.

    Must be called with the teacher replicas already moved to GPU, and with
    the student engine already asleep (phased GPU residency is the caller's
    responsibility -- see ray_trainer.fit()).
    """
    input_ids = batch.batch["input_ids"]
    attention_mask = batch.batch["attention_mask"]
    prompt_len = input_ids.shape[1] - response_length
    response_mask = attention_mask[:, -response_length:]

    batch_size = input_ids.shape[0]
    sequence_id_lists = []
    prompt_valid_lens = []
    response_valid_lens = []
    for i in range(batch_size):
        prompt_mask_row = attention_mask[i, :prompt_len]
        resp_mask_row = response_mask[i]
        prompt_valid_lens.append(int(prompt_mask_row.sum().item()))
        response_valid_lens.append(int(resp_mask_row.sum().item()))
        sequence_id_lists.append(_row_to_unpadded_ids(input_ids[i], attention_mask[i]))

    # round-robin shard across teacher replicas
    num_replicas = len(handles)
    if num_replicas <= 0:
        raise ValueError("Cannot score KD batch without teacher replicas")
    shard_indices = [[] for _ in range(num_replicas)]
    for i in range(batch_size):
        shard_indices[i % num_replicas].append(i)

    futures = []
    for r, idxs in enumerate(shard_indices):
        if not idxs:
            continue
        seqs = [sequence_id_lists[i] for i in idxs]
        futures.append((idxs, handles[r].score_batch.remote(seqs)))

    per_row_result = [None] * batch_size
    for idxs, fut in futures:
        results = ray.get(fut)
        for local_i, global_i in enumerate(idxs):
            per_row_result[global_i] = results[local_i]

    ids_out = np.full((batch_size, response_length, k), pad_token_id, dtype=np.int64)
    lps_out = np.full((batch_size, response_length, k), OUT_OF_VOCAB_LOGPROB, dtype=np.float32)

    for i in range(batch_size):
        ids_i, lps_i = per_row_result[i]
        p_len = prompt_valid_lens[i]
        r_len = response_valid_lens[i]
        # scored row j (0-indexed into ids_i/lps_i, which cover sequence
        # positions 1..S-1) predicts sequence token at position j+1 using
        # context [0..j]. Response token t (0-indexed) sits at sequence
        # position p_len + t, so it is predicted by scored row (p_len + t - 1).
        start = max(p_len - 1, 0)
        end = start + r_len
        span_ids = ids_i[start:end]
        span_lps = lps_i[start:end]
        n = min(span_ids.shape[0], response_length)
        ids_out[i, :n] = span_ids[:n]
        lps_out[i, :n] = span_lps[:n]

    teacher_topk_ids = torch.from_numpy(ids_out)
    teacher_topk_log_probs = torch.from_numpy(lps_out)

    kd_proto = DataProto.from_dict(
        tensors={
            "teacher_topk_ids": teacher_topk_ids,
            "teacher_topk_log_probs": teacher_topk_log_probs,
        }
    )
    return batch.union(kd_proto)
