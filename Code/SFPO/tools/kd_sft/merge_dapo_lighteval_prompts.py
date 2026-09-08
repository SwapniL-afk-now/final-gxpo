#!/usr/bin/env python3
"""Merge dapo_math/train.parquet + lighteval-math/train.parquet into one
prompt-only parquet for tools/kd_sft/build_teacher_topk.py, which takes a
single --train-parquet. Only the `prompt` column is needed by that script
(read_prompts uses prompt[0]['content']); ground-truth/reward_model columns
are dropped since teacher generation here needs no correctness signal.

Usage:
  python tools/kd_sft/merge_dapo_lighteval_prompts.py \
    --dapo <dapo_math/train.parquet> --lighteval <lighteval-math/train.parquet> \
    --out data/kd/dapo_lighteval_prompts.parquet
"""
import argparse

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dapo", required=True)
    ap.add_argument("--lighteval", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    frames = []
    for path, source in ((args.dapo, "dapo_math"), (args.lighteval, "lighteval-math")):
        df = pd.read_parquet(path)
        if "prompt" not in df.columns:
            raise SystemExit(f"{path} has no 'prompt' column (got {list(df.columns)})")
        frames.append(pd.DataFrame({"prompt": df["prompt"], "source": source}))

    merged = pd.concat(frames, ignore_index=True)
    merged.to_parquet(args.out)
    print(f"[merge] dapo={len(frames[0])} lighteval={len(frames[1])} -> "
          f"{len(merged)} rows -> {args.out}")


if __name__ == "__main__":
    main()
