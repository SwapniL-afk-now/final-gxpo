#!/usr/bin/env python3
"""Materialize the stored DeepSeek-R1 math traces for FSDP distillation.

This tool never calls a model.  It reads the already-generated ``generation``
column from ``wangx0t/numina-deepseek-DeepSeek-R1-Distill-Qwen-7B`` and writes
the prompt/response parquet files consumed by the SFT and offline KD paths.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


DATASET_NAME = "wangx0t/numina-deepseek-DeepSeek-R1-Distill-Qwen-7B"
TEACHER_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B"


def _as_python(value):
    return value.tolist() if hasattr(value, "tolist") else value


def canonical_problem(value: object) -> str:
    value = _as_python(value)
    if isinstance(value, (list, tuple)):
        value = "\n".join(str(item.get("content", "")) for item in value if isinstance(item, dict))
    return " ".join(str(value).strip().split())


def stable_hash(value: object) -> str:
    return hashlib.sha256(canonical_problem(value).encode("utf-8")).hexdigest()


def user_prompt(example: dict) -> list[dict[str, str]]:
    """Keep the dataset's user problem; do not copy the stored assistant text."""
    messages = _as_python(example.get("messages"))
    if isinstance(messages, (list, tuple)):
        users = [
            {"role": "user", "content": str(message.get("content", ""))}
            for message in messages
            if isinstance(message, dict) and message.get("role") == "user"
        ]
        if users and users[0]["content"].strip():
            return users[:1]
    problem = str(example.get("problem", "")).strip()
    if not problem:
        raise ValueError("Numina row has neither a usable user message nor problem")
    return [{"role": "user", "content": problem}]


def materialize_rows(frame: pd.DataFrame, *, teacher_model: str = TEACHER_MODEL) -> pd.DataFrame:
    required = {"problem", "generation"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    if "model_name" in frame.columns:
        model_names = {str(value) for value in frame["model_name"].dropna().unique()}
        if model_names and model_names != {teacher_model}:
            raise ValueError(
                f"Dataset model_name values {sorted(model_names)} do not match {teacher_model}"
            )

    rows = []
    seen = set()
    for source_index, record in frame.iterrows():
        example = record.to_dict()
        prompt = user_prompt(example)
        response = str(example.get("generation", "")).strip()
        if not response:
            raise ValueError(f"Row {source_index} has an empty stored teacher generation")
        problem_hash = stable_hash(prompt)
        if problem_hash in seen:
            continue
        seen.add(problem_hash)
        rows.append({
            "problem_id": str(example.get("id", source_index)),
            "problem_hash": problem_hash,
            "source": "numina_r1_distill_teacher_trace",
            "data_source": DATASET_NAME,
            "teacher_model": teacher_model,
            "prompt": prompt,
            "response": response,
            "teacher_generated": True,
        })
    if not rows:
        raise ValueError("No usable stored teacher traces were found")
    return pd.DataFrame(rows)


def split_rows(frame: pd.DataFrame, train_size: int, val_size: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    if train_size <= 0 or val_size <= 0:
        raise ValueError("train_size and val_size must be positive")
    needed = train_size + val_size
    if len(frame) < needed:
        raise ValueError(f"Need {needed} unique rows, found {len(frame)}")
    order = np.random.default_rng(seed).permutation(len(frame))
    shuffled = frame.iloc[order].reset_index(drop=True)
    train = shuffled.iloc[:train_size].copy()
    val = shuffled.iloc[train_size:needed].copy()
    train["split"] = "train"
    val["split"] = "validation"
    if set(train["problem_hash"]) & set(val["problem_hash"]):
        raise AssertionError("train/validation problem overlap")
    return train.reset_index(drop=True), val.reset_index(drop=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default=TEACHER_MODEL)
    parser.add_argument("--train-size", type=int, default=70000)
    parser.add_argument("--val-size", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--revision", default=None)
    args = parser.parse_args()
    if args.teacher_model != TEACHER_MODEL:
        raise SystemExit("This materializer is pinned to the requested DeepSeek-R1 teacher")

    from datasets import load_dataset

    print(f"Loading stored traces from {args.dataset} (split=train)", flush=True)
    kwargs = {"split": "train"}
    if args.revision:
        kwargs["revision"] = args.revision
    dataset = load_dataset(args.dataset, **kwargs)
    frame = materialize_rows(dataset.to_pandas(), teacher_model=args.teacher_model)
    train, val = split_rows(frame, args.train_size, args.val_size, args.seed)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    train_path = args.output_dir / "math_r1_train.parquet"
    val_path = args.output_dir / "math_r1_val.parquet"
    train.to_parquet(train_path, index=False)
    val.to_parquet(val_path, index=False)
    manifest = {
        "schema_version": 1,
        "dataset": args.dataset,
        "teacher_model": args.teacher_model,
        "seed": args.seed,
        "train_size": len(train),
        "validation_size": len(val),
        "discarded_after_split": len(frame) - len(train) - len(val),
        "response_column": "generation",
        "teacher_generation_during_training": False,
        "train_problem_hashes": sorted(train["problem_hash"].tolist()),
        "validation_problem_hashes": sorted(val["problem_hash"].tolist()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(train)} train rows and {len(val)} validation rows to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
