#!/usr/bin/env python3
"""Merge offline KD train/test caches into one prompt corpus."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd


def key(row):
    response_ids = np.asarray(row.response_ids).reshape(-1).astype(int).tolist()
    return str(row.prompt), str(row.response), json.dumps(response_ids)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--train', required=True, type=Path)
    ap.add_argument('--test', required=True, type=Path)
    ap.add_argument('--out', required=True, type=Path)
    args = ap.parse_args()
    for path in (args.train, args.test):
        if not path.is_file():
            raise SystemExit(f'missing KD split: {path}')
    frames = [pd.read_parquet(path) for path in (args.train, args.test)]
    required = {'prompt', 'response', 'response_ids', 'teacher_topk_log_probs', 'teacher_topk_ids'}
    for path, frame in zip((args.train, args.test), frames):
        missing = required.difference(frame.columns)
        if missing:
            raise SystemExit(f'{path} missing columns: {sorted(missing)}')
    merged = pd.concat(frames, ignore_index=True)
    merged['_merge_key'] = [key(row) for row in merged.itertuples(index=False)]
    merged = merged.drop_duplicates('_merge_key', keep='first').drop(columns='_merge_key')
    merged = merged.reset_index(drop=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_name(f'.{args.out.name}.tmp')
    merged.to_parquet(tmp, index=False)
    os.replace(tmp, args.out)
    print({'train_rows': len(frames[0]), 'test_rows': len(frames[1]),
           'merged_rows': len(merged), 'output': str(args.out)}, flush=True)


if __name__ == '__main__':
    main()
