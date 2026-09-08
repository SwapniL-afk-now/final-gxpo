#!/usr/bin/env python3
"""Hold out a val split from a tools/kd_sft/build_teacher_topk.py cache.

Mirrors train-scripts/prep_bespoke_stratos_sft.py's val-split convention
(fixed row count, fixed seed) so KDSFTTrainer's mandatory val_files has
something to read for its cheap per-epoch val-loss logging.

Usage:
  python tools/kd_sft/split_kd_cache_train_val.py \
    --cache data/kd/dapo_lighteval_topk16.parquet \
    --out-train data/kd/dapo_lighteval_topk16_train.parquet \
    --out-val data/kd/dapo_lighteval_topk16_val.parquet \
    [--val-size 500] [--seed 42]
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--out-train", required=True)
    ap.add_argument("--out-val", required=True)
    ap.add_argument("--val-size", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_parquet(args.cache)
    if args.val_size >= len(df):
        raise SystemExit(f"--val-size {args.val_size} >= cache size {len(df)}")

    val = df.sample(n=args.val_size, random_state=args.seed)
    train = df.drop(val.index)

    train.to_parquet(args.out_train)
    val.to_parquet(args.out_val)
    print(f"[split] cache={len(df)} -> train={len(train)} val={len(val)} "
          f"(seed={args.seed})")


if __name__ == "__main__":
    main()
