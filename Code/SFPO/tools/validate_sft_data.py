#!/usr/bin/env python3
"""Validate the prompt/response parquet contract used by the SFT launchers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

DEFAULT_SOURCE = "DigitalLearningGmbH/MATH-lighteval"
REQUIRED_COLUMNS = {"prompt", "response", "data_source", "extra_info"}


def validate_file(path: Path, expected_split: str, source_dataset: str) -> int:
    if not path.is_file():
        raise ValueError(f"missing SFT parquet: {path}")
    frame = pd.read_parquet(path)
    missing = REQUIRED_COLUMNS.difference(frame.columns)
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    if frame.empty:
        raise ValueError(f"{path}: parquet is empty")

    for index, row in frame.iterrows():
        prompt = row["prompt"]
        response = row["response"]
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError(f"{path}: row {index} has an empty/non-string prompt")
        if not isinstance(response, str) or not response.strip():
            raise ValueError(f"{path}: row {index} has an empty/non-string worked response")
        if row["data_source"] != source_dataset:
            raise ValueError(
                f"{path}: row {index} data_source={row['data_source']!r}; "
                f"expected {source_dataset!r}"
            )
        extra = row["extra_info"]
        if not isinstance(extra, dict) or extra.get("split") != expected_split:
            raise ValueError(
                f"{path}: row {index} extra_info.split must be {expected_split!r}"
            )
    return len(frame)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--train-file", type=Path)
    parser.add_argument("--test-file", type=Path)
    parser.add_argument("--source-dataset", default=DEFAULT_SOURCE)
    args = parser.parse_args()

    train_file = args.train_file or args.data_root / "train.parquet"
    test_file = args.test_file or args.data_root / "test.parquet"

    counts = {
        "train": validate_file(train_file, "train", args.source_dataset),
        "test": validate_file(test_file, "test", args.source_dataset),
    }
    print(json.dumps({"data_root": str(args.data_root.resolve()), "rows": counts}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
