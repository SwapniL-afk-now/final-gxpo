#!/usr/bin/env python3
"""Generate student responses for batched on-policy KD (final-gxpo).

Reads prompt-only parquets (DAPO / lighteval / benchmark schema), generates
ONE response per prompt with vLLM offline (TP=1, CUDA graphs, FlashInfer),
and writes (prompt, response, response_ids) rows. The next loop phase scores
these pairs with the teacher (build_teacher_topk.py --mode score) and
trains the student on its OWN distribution -- the on-policy KD loop:
student generates -> teacher scores -> chunked-KL GXPO update.

Response ids are stored verbatim (incl. trailing eos when the stop is eos);
SFTDataset consumes them without re-tokenizing, so alignment is exact.

Usage:
  python tools/kd_sft/generate_student_responses.py --in-parquet <prompts> \
    --model-path <student-ckpt> --tokenizer-path <student-tok> \
    --out data/kd/gen_student_s0.parquet [--num-shards 2 --shard 0]
"""

from __future__ import annotations

import argparse
import glob
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-parquet", required=True)
    ap.add_argument("--model-path", required=True)
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--max-prompt-tokens", type=int, default=1023)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--flush-every", type=int, default=512)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    df = pd.read_parquet(args.in_parquet)
    if "prompt" not in df.columns:
        raise SystemExit(f"{args.in_parquet} has no 'prompt' column")
    rows = []
    for idx, row in df.iterrows():
        p = row["prompt"]
        content = p[0]["content"] if isinstance(p, list) else str(p)
        chat = tok.apply_chat_template([{"role": "user", "content": content}],
                                       add_generation_prompt=True, tokenize=False)
        if len(tok(chat, add_special_tokens=False)["input_ids"]) > args.max_prompt_tokens:
            continue
        rows.append({"dataset_index": int(idx), "prompt": content, "chat": chat})
    rows = rows[args.shard::args.num_shards]
    print(f"[student-gen] {len(rows)} prompts (shard {args.shard}/{args.num_shards})", flush=True)

    work = f"{args.out}.shard{args.shard}.work.parquet"
    done = set()
    for pf in glob.glob(f"{work}.part*.parquet"):
        try:
            done.update(pd.read_parquet(pf, columns=["prompt", "dataset_index"])
                        .apply(lambda r: (r["prompt"], int(r["dataset_index"])), axis=1).tolist())
        except Exception:
            pass
    rows = [r for r in rows if (r["prompt"], r["dataset_index"]) not in done]
    if done:
        print(f"[student-gen] resume: {len(done)} rows cached", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.model_path, dtype="bfloat16",
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=0.90, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, enable_chunked_prefill=True,
              trust_remote_code=True, enforce_eager=False)
    params = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                            max_tokens=args.max_tokens)
    outs = llm.generate([r["chat"] for r in rows], params)

    pending, part_idx, kept = [], 0, 0
    for r, o in zip(rows, outs):
        gen = o.outputs[0]
        gen_ids = list(gen.token_ids)
        if not gen_ids:
            continue
        pending.append({"prompt": r["prompt"], "dataset_index": r["dataset_index"],
                        "response": gen.text, "response_ids": gen_ids})
        kept += 1
        if len(pending) >= args.flush_every:
            pd.DataFrame(pending).to_parquet(f"{work}.part{part_idx}.parquet")
            pending.clear()
            part_idx += 1
    if pending:
        pd.DataFrame(pending).to_parquet(f"{work}.part{part_idx}.parquet")
    parts = sorted(glob.glob(f"{work}.part*.parquet"))
    if parts:
        pd.concat([pd.read_parquet(p) for p in parts],
                  ignore_index=True).to_parquet(args.out if args.num_shards == 1 else work)
    print(f"[student-gen] kept={kept} -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
