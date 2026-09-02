#!/usr/bin/env python3
"""Create deduplicated, versioned coding-evaluation parquet files.

The input benchmark files are never modified. Rows matching a training prompt
canonical hash are removed from every evaluation benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def canonical_prompt(value: object) -> str:
    if isinstance(value, str):
        text = value
    else:
        value = value.tolist() if hasattr(value, "tolist") else value
        text = "\n".join(str(item.get("content", "")) for item in value)
    return " ".join(text.replace(
        "Respond with a complete Python program in a single ```python code block.", ""
    ).split())


def prompt_hash(value: object) -> str:
    return hashlib.sha256(canonical_prompt(value).encode("utf-8")).hexdigest()


def load_hashes(path: Path) -> set[str]:
    manifest = json.loads(path.read_text())
    hashes = set(manifest.get("canonical_hashes", []))
    if hashes:
        return hashes
    return set(manifest.get("problem_hashes", []))


def process(path: Path, source: str, blocked: set[str], output: Path,
            after_date: str | None = None) -> dict:
    frame = pd.read_parquet(path).copy()
    if "prompt" not in frame:
        raise ValueError(f"{path} has no prompt column")
    hashes = frame["prompt"].map(prompt_hash)
    kept = frame.loc[~hashes.isin(blocked)].copy()
    temporal_removed = 0
    if after_date is not None and source == "livecodebench":
        date_column = next(
            (column for column in ("contest_date", "release_date", "question_date", "date")
             if column in kept.columns),
            None,
        )
        if date_column is None:
            raise ValueError(
                "LiveCodeBench input has no recognized date column; cannot create "
                "the requested temporal split"
            )
        dates = pd.to_datetime(kept[date_column], errors="coerce", utc=True)
        if dates.isna().any():
            raise ValueError(f"{path} contains LiveCodeBench rows with invalid dates")
        cutoff = pd.Timestamp(after_date, tz="UTC")
        keep_temporal = dates > cutoff
        temporal_removed = int((~keep_temporal).sum())
        kept = kept.loc[keep_temporal].copy()
    if "data_source" not in kept:
        kept["data_source"] = source
    else:
        kept["data_source"] = source
    output.parent.mkdir(parents=True, exist_ok=True)
    kept.to_parquet(output, index=False)
    return {"source": source, "input": str(path), "output": str(output),
            "input_rows": len(frame), "output_rows": len(kept),
            "removed_rows": int(len(frame) - len(kept)),
            "temporal_removed_rows": temporal_removed,
            "temporal_after": after_date}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--humanevalplus", type=Path, required=True)
    parser.add_argument("--mbppplus", type=Path, required=True)
    parser.add_argument("--livecodebench", type=Path, required=True)
    parser.add_argument("--livecodebench-after", default="2024-07-31")
    args = parser.parse_args()
    blocked = load_hashes(args.training_manifest)
    specs = (("humanevalplus", args.humanevalplus),
             ("mbppplus", args.mbppplus),
             ("livecodebench", args.livecodebench))
    manifest = {"schema_version": 1, "blocked_prompt_hashes": len(blocked), "benchmarks": []}
    for source, path in specs:
        if not path.is_file():
            raise SystemExit(f"Missing benchmark parquet: {path}")
        manifest["benchmarks"].append(
            process(path, source, blocked, args.output_dir / f"{source}.parquet",
                    after_date=args.livecodebench_after)
        )
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
