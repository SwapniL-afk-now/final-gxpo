#!/usr/bin/env python3
"""W1: score fixed off-policy sequences with the frozen teacher, cache top-K.

Reuses verl's TeacherScoringWorker (plain-HF forward, flash_attention_2, NO
vLLM) plus the production response-span slicing (score_batch_and_attach), so
cached targets are bit-identical to what per-step on-policy scoring would
attach. Run once per dataset; the trainer then loads the cache and the
teacher never touches a GPU during training.

Cache parquet columns (per-row python lists, unpadded):
  prompt_ids, response_ids, data_source, ground_truth, reward, source_index
  teacher_topk_ids    : [R, K] int   (response span only)
  teacher_topk_logps  : [R, K] float (response span only)
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import ray
import torch
from verl import DataProto
from verl.trainer.ppo.teacher_kd import score_batch_and_attach


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--out-parquet", required=True)
    ap.add_argument("--teacher-path", required=True)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--micro-batch-size", type=int, default=8)
    ap.add_argument("--chunk-tokens", type=int, default=2048)
    ap.add_argument("--rows-per-call", type=int, default=256)
    a = ap.parse_args()

    ray.init(ignore_reinit_error=True, num_gpus=1)
    from verl.workers.teacher_scoring_worker import TeacherScoringWorker
    h = TeacherScoringWorker.options(num_gpus=1).remote(
        model_path=a.teacher_path, k=a.topk, dtype="bfloat16", pad_token_id=0,
        student_vocab_size=151936, micro_batch_size=a.micro_batch_size,
        chunk_tokens=a.chunk_tokens, attn_implementation="flash_attention_2",
        start_on_cpu=True)
    ray.get(h.to_gpu.remote())
    assert ray.get(h.is_on_gpu.remote()), "teacher failed to reach GPU"

    df = pd.read_parquet(a.in_parquet)
    print(f"scoring {len(df)} rows", flush=True)
    out_rows = []
    for lo in range(0, len(df), a.rows_per_call):
        chunk = df.iloc[lo:lo + a.rows_per_call]
        p = [list(map(int, x)) for x in chunk["prompt_ids"]]
        r = [list(map(int, x)) for x in chunk["response_ids"]]
        plen = max(len(x) for x in p)
        rlen = max(len(x) for x in r)
        B = len(chunk)
        input_ids = torch.zeros(B, plen + rlen, dtype=torch.long)
        attn = torch.zeros(B, plen + rlen, dtype=torch.long)
        for i in range(B):
            input_ids[i, plen - len(p[i]):plen] = torch.tensor(p[i])
            attn[i, plen - len(p[i]):plen] = 1
            input_ids[i, plen:plen + len(r[i])] = torch.tensor(r[i])
            attn[i, plen:plen + len(r[i])] = 1
        batch = DataProto.from_dict(
            tensors={"input_ids": input_ids, "attention_mask": attn})
        scored = score_batch_and_attach(batch, [h], response_length=rlen,
                                        k=a.topk, pad_token_id=0)
        tids = scored.batch["teacher_topk_ids"].numpy()
        tlps = scored.batch["teacher_topk_log_probs"].numpy()
        for i, (_, row) in enumerate(chunk.iterrows()):
            n = len(r[i])
            out_rows.append({
                "prompt_ids": p[i], "response_ids": r[i],
                "data_source": row["data_source"], "ground_truth": row["ground_truth"],
                "reward": float(row["reward"]), "source_index": row["source_index"],
                "teacher_topk_ids": tids[i, :n].astype(np.int32).tolist(),
                "teacher_topk_logps": tlps[i, :n].astype(np.float32).tolist(),
            })
        print(f"  scored {lo + B}/{len(df)}", flush=True)

    ray.get(h.to_cpu.remote())
    out = pd.DataFrame(out_rows)
    out.to_parquet(a.out_parquet, index=False)
    tmass = np.mean([np.exp(np.array(x, dtype=np.float64)).sum(axis=-1).mean()
                     for x in out["teacher_topk_logps"]])
    print(json.dumps({"rows": len(out), "mean_teacher_topk_mass": float(tmass)}))
    print(f"wrote {a.out_parquet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
