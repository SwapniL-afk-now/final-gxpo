#!/usr/bin/env python3
"""Build SFT train/val parquet from bespokelabs/Bespoke-Stratos-17k.

All 16710 rows are single user->assistant turns sharing one system prompt.
verl's SFTDataset has no system column, so the system text is folded into
`prompt` as "<system>\\n\\n<question>", paired with the assistant trace as
`response` (prompt_key=prompt, response_key=response).

Rows whose combined (prompt+response) token length exceeds --max-tokens are
dropped (right-truncation would cut the boxed answer off the tail): at 16384
this keeps ~95% of rows. Holds out 500 rows for val.

Usage:
  python3 prep_bespoke_stratos_sft.py \
    --out-dir /office/dev_workspace/swapnil/data/Bespoke-Stratos-17k \
    [--max-tokens 16384] [--val-size 500] [--seed 42]
"""
import argparse
import os
import random
import statistics

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--max-tokens', type=int, default=16384)
    ap.add_argument('--val-size', type=int, default=500)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--tokenizer', default=(
        '/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct'
        '/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306'))
    args = ap.parse_args()

    from datasets import load_dataset
    from transformers import AutoTokenizer

    ds = load_dataset('bespokelabs/Bespoke-Stratos-17k', split='train')
    tok = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows, lens = [], []
    n_empty = 0
    for ex in ds:
        user = ex['conversations'][0]['value']
        asst = ex['conversations'][-1]['value']
        if not asst.strip() or not user.strip():
            n_empty += 1
            continue
        prompt = ex['system'].strip() + '\n\n' + user.strip()
        response = asst.strip()
        n_tok = len(tok(prompt)['input_ids']) + len(tok(response)['input_ids'])
        lens.append(n_tok)
        rows.append({'prompt': prompt, 'response': response, 'n_tok': n_tok})

    lens_sorted = sorted(lens)
    qs = statistics.quantiles(lens_sorted, n=100)
    print(f'rows={len(rows)} skipped_empty={n_empty}')
    print(f'combined tokens: p50={qs[49]:.0f} p90={qs[89]:.0f} p99={qs[98]:.0f} max={lens_sorted[-1]}')

    kept = [r for r in rows if r['n_tok'] <= args.max_tokens]
    print(f'kept={len(kept)} dropped_over_{args.max_tokens}={len(rows) - len(kept)}')
    for r in kept:
        del r['n_tok']

    rng = random.Random(args.seed)
    rng.shuffle(kept)
    val = kept[:args.val_size]
    train = kept[args.val_size:]
    os.makedirs(args.out_dir, exist_ok=True)
    pd.DataFrame(train).to_parquet(os.path.join(args.out_dir, 'train.parquet'), index=False)
    pd.DataFrame(val).to_parquet(os.path.join(args.out_dir, 'val.parquet'), index=False)
    print(f'wrote train={len(train)} val={len(val)} -> {args.out_dir}/{{train,val}}.parquet')


if __name__ == '__main__':
    main()
