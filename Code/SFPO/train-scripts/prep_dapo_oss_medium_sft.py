#!/usr/bin/env python3
"""Build SFT train/val parquet from the DAPO gpt-oss-120b medium-effort file.

Keeps reward==1 rows only, pairs `reasoning_prompt` (boxed-answer style, which
matches the teacher traces) with `gpt-oss-120b-response`, holds out 500 rows
for val, and writes plain {prompt, response} string columns as expected by
verl's SFTDataset (prompt_key=prompt, response_key=response).

Usage:
  python3 prep_dapo_oss_medium_sft.py \
    --input data/DAPO-MATH-17k-oss-reasoning/math-oss-medium-17k.jsonl \
    --out-dir /workspace/jepa-grpo-cache/data/dapo_oss_medium_r1 \
    [--val-size 500] [--seed 42] [--tokenizer /workspace/models/Qwen2.5-1.5B-Instruct]
"""
import argparse
import json
import os

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--val-size', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tokenizer', default=None)
    args = ap.parse_args()

    rows = []
    n_total = n_correct = n_empty = 0
    with open(args.input) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            n_total += 1
            if (r.get('reward') or 0) < 1.0:
                continue
            resp = r.get('gpt-oss-120b-response') or ''
            if not resp.strip():
                n_empty += 1
                continue
            n_correct += 1
            rows.append({'prompt': r['reasoning_prompt'], 'response': resp})

    print(f'total={n_total} correct_nonempty={n_correct} skipped_incorrect={n_total - n_correct - n_empty} skipped_empty={n_empty}')

    # Length audit against the 2048-token SFT budget (prompt+response, right-truncated).
    if args.tokenizer:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)
        lens = sorted(len(tok(p['prompt'])['input_ids']) + len(tok(p['response'])['input_ids']) for p in rows)
    else:
        lens = sorted(len(p['prompt']) + len(p['response']) for p in rows)
    unit = 'tokens' if args.tokenizer else 'chars'
    import statistics
    qs = statistics.quantiles(lens, n=100)
    print(f'combined length ({unit}): p50={qs[49]:.0f} p90={qs[89]:.0f} p99={qs[98]:.0f} max={lens[-1]}')
    if args.tokenizer:
        over = sum(1 for L in lens if L > 2048)
        print(f'rows exceeding 2048 tokens (tail-truncated by trainer): {over}/{len(lens)}')

    import random
    rng = random.Random(args.seed)
    rng.shuffle(rows)
    val = rows[:args.val_size]
    train = rows[args.val_size:]
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(train).to_parquet(os.path.join(args.out_dir, 'train.parquet'), index=False)
    pd.DataFrame(val).to_parquet(os.path.join(args.out_dir, 'val.parquet'), index=False)
    print(f'wrote train={len(train)} val={len(val)} -> {args.out_dir}/{{train,val}}.parquet')


if __name__ == '__main__':
    main()
