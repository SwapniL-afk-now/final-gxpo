#!/usr/bin/env python3
"""Validate the local SFPO-compatible GXPO math parquet assets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

try:
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - exercised by the CLI
    raise SystemExit("pyarrow is required to verify GXPO math assets") from exc


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("GXPO_DATA_ROOT", str(REPO_ROOT / "data"))).resolve()

ASSETS = {
    "dapo": {
        "source": "haizhongzheng/DAPO-Math-17K-cleaned",
        "split": "train",
        "path": DATA_ROOT / "dapo_math" / "train.parquet",
        "expected_data_source": "dapo_math",
    },
    "lighteval": {
        "source": "xDAN2099/lighteval-MATH",
        "split": "train",
        "path": DATA_ROOT / "lighteval-math" / "train.parquet",
        "expected_data_source": "xDAN2099/lighteval-MATH",
        "expected_rows": 7500,
    },
    "math500": {
        "source": "HuggingFaceH4/MATH-500",
        "split": "test",
        "path": DATA_ROOT / "math500" / "test.parquet",
        "expected_data_source": "HuggingFaceH4/MATH-500",
    },
    "aime24": {
        "source": "HuggingFaceH4/aime_2024",
        "split": "train",
        "path": DATA_ROOT / "aime2024" / "test.parquet",
        "expected_data_source": "HuggingFaceH4/aime_2024",
    },
    "aime25": {
        "source": "MathArena/aime_2025",
        "split": "train",
        "path": DATA_ROOT / "aime2025" / "test.parquet",
        "expected_data_source": "MathArena/aime_2025",
    },
    "amc": {
        "source": "AI-MO/aimo-validation-amc",
        "split": "train",
        "path": DATA_ROOT / "amc" / "test.parquet",
        "expected_data_source": "AI-MO/aimo-validation-amc",
    },
    "minerva": {
        "source": "math-ai/minervamath",
        "split": "test",
        "path": DATA_ROOT / "minervamath" / "test.parquet",
        "expected_data_source": "math-ai/minervamath",
    },
    "olympiadbench": {
        "source": "math-ai/olympiadbench",
        "split": "test",
        "path": DATA_ROOT / "olympiadbench" / "test.parquet",
        "expected_data_source": "math-ai/olympiadbench",
    },
}

REQUIRED_COLUMNS = {"data_source", "prompt", "ability", "reward_model", "extra_info"}


def _validate_row(row: dict[str, Any], spec: dict[str, Any], row_index: int) -> None:
    missing = REQUIRED_COLUMNS.difference(row)
    if missing:
        raise ValueError(f"row {row_index}: missing columns {sorted(missing)}")
    if row["data_source"] != spec["expected_data_source"]:
        raise ValueError(
            f"row {row_index}: data_source={row['data_source']!r}, "
            f"expected {spec['expected_data_source']!r}"
        )
    if row["ability"] != "math":
        raise ValueError(f"row {row_index}: ability must be 'math'")

    prompt = row["prompt"]
    if not isinstance(prompt, list) or len(prompt) != 1 or not isinstance(prompt[0], dict):
        raise ValueError(f"row {row_index}: prompt must be a one-message list")
    if prompt[0].get("role") != "user" or not isinstance(prompt[0].get("content"), str):
        raise ValueError(f"row {row_index}: prompt must contain a user message with text")
    if not prompt[0]["content"].strip():
        raise ValueError(f"row {row_index}: prompt content is empty")

    reward_model = row["reward_model"]
    if not isinstance(reward_model, dict) or reward_model.get("style") != "rule":
        raise ValueError(f"row {row_index}: reward_model must use style='rule'")
    if reward_model.get("ground_truth") is None:
        raise ValueError(f"row {row_index}: reward_model.ground_truth is null")

    extra_info = row["extra_info"]
    if not isinstance(extra_info, dict) or "split" not in extra_info or "index" not in extra_info:
        raise ValueError(f"row {row_index}: extra_info must contain split and index")
    if extra_info["split"] != spec["split"]:
        raise ValueError(
            f"row {row_index}: extra_info.split={extra_info['split']!r}, "
            f"expected {spec['split']!r}"
        )


def _first_schema(row: dict[str, Any]) -> dict[str, Any]:
    prompt = row["prompt"][0]
    reward = row["reward_model"]
    content = prompt["content"]
    return {
        "data_source": row["data_source"],
        "prompt": [{"role": prompt["role"], "content": content[:160] + ("..." if len(content) > 160 else "")}],
        "ability": row["ability"],
        "reward_model": {
            "style": reward["style"],
            "ground_truth": str(reward["ground_truth"])[:120],
        },
        "extra_info": row["extra_info"],
    }


def validate_asset(name: str, *, print_schema: bool = False) -> int:
    spec = ASSETS[name]
    path = Path(spec["path"])
    if not path.is_file():
        raise ValueError(f"{name}: missing parquet: {path}")
    table = pq.read_table(path)
    missing_columns = REQUIRED_COLUMNS.difference(table.column_names)
    if missing_columns:
        raise ValueError(f"{name}: missing columns {sorted(missing_columns)}")
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"{name}: parquet is empty: {path}")
    expected_rows = spec.get("expected_rows")
    if expected_rows is not None and len(rows) != expected_rows:
        raise ValueError(f"{name}: rows={len(rows)}, expected exactly {expected_rows}")
    for row_index, row in enumerate(rows):
        _validate_row(row, spec, row_index)
    if print_schema:
        print(f"First processed example ({name}):")
        print(json.dumps(_first_schema(rows[0]), ensure_ascii=False, indent=2))
    return len(rows)


def _print_table(names: list[str], counts: dict[str, Optional[int]], failures: dict[str, str]) -> None:
    print("Asset | HF source | Split | Rows | Output | Status")
    print("--- | --- | --- | ---: | --- | ---")
    for name in names:
        spec = ASSETS[name]
        status = "PASS" if name not in failures else f"FAIL: {failures[name]}"
        rows = "-" if counts.get(name) is None else str(counts[name])
        print(f"{name} | {spec['source']} | {spec['split']} | {rows} | {spec['path']} | {status}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", action="append", choices=sorted(ASSETS), help="Validate only this asset; repeatable")
    parser.add_argument("--show-first", action="store_true", help="Print a compact first-example schema")
    parser.add_argument("--training-summary", action="store_true", help="Print DAPO/MATH row counts for launcher preflight")
    args = parser.parse_args()

    names = args.asset or list(ASSETS)
    counts: dict[str, Optional[int]] = {}
    failures: dict[str, str] = {}
    for name in names:
        try:
            counts[name] = validate_asset(name, print_schema=args.show_first)
        except Exception as exc:  # noqa: BLE001 - CLI should report every failed asset
            counts[name] = None
            failures[name] = str(exc)

    _print_table(names, counts, failures)
    if args.training_summary:
        if counts.get("dapo") is None or counts.get("lighteval") is None:
            return 1
        total = counts["dapo"] + counts["lighteval"]  # type: ignore[operator]
        print("Training sources:")
        print(f"  DAPO: {ASSETS['dapo']['path']} ({counts['dapo']} rows)")
        print(f"  MATH: {ASSETS['lighteval']['path']} ({counts['lighteval']} rows)")
        print(f"  Total logical training rows: {total}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
