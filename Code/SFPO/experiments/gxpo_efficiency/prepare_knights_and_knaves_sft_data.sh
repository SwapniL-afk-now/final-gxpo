#!/usr/bin/env bash
# Prepare the K-and-K logical-reasoning SFT experiment.
#
# Training: 900 examples each from 3, 4, 5, and 6-person puzzles.
# Validation: the remaining 100 training examples per difficulty.
# IID test: official 100-example test sets for 3--6 people.
# OOD test: official 100-example test sets for 7--8 people.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
SFPO_ROOT="$REPO_ROOT/Code/SFPO"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  source "$REPO_ROOT/.env"
  set +a
fi
if [[ -z "${HF_TOKEN:-}" && -n "${HF_API_KEY:-}" ]]; then
  export HF_TOKEN="$HF_API_KEY"
fi

DATA_ROOT="${SFT_DATA_ROOT:-$SFPO_ROOT/data/knights_and_knaves_sft}"
SOURCE_DATASET="${SFT_SOURCE_DATASET:-K-and-K/knights-and-knaves}"
SPLIT_SEED="${SFT_SPLIT_SEED:-42}"
TRAIN_PER_DIFFICULTY="${SFT_KK_TRAIN_PER_DIFFICULTY:-900}"
VALIDATION_PER_DIFFICULTY="${SFT_KK_VALIDATION_PER_DIFFICULTY:-100}"
FORCE_REBUILD="${SFT_FORCE_REBUILD:-0}"
mkdir -p "$DATA_ROOT"

if [[ -s "$DATA_ROOT/train.parquet" && -s "$DATA_ROOT/test.parquet" \
      && -s "$DATA_ROOT/iid_test.parquet" && -s "$DATA_ROOT/ood_test.parquet" \
      && "$FORCE_REBUILD" != "1" ]]; then
  echo "K&K SFT data already exists; validating without overwriting: $DATA_ROOT"
  "$PYTHON_BIN" "$SFPO_ROOT/tools/validate_sft_data.py" \
    --data-root "$DATA_ROOT" --source-dataset "$SOURCE_DATASET"
  exit 0
fi

"$PYTHON_BIN" - "$DATA_ROOT" "$SOURCE_DATASET" "$SFPO_ROOT" \
  "$SPLIT_SEED" "$TRAIN_PER_DIFFICULTY" "$VALIDATION_PER_DIFFICULTY" <<'PY'
from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from huggingface_hub import hf_hub_download

data_root = Path(sys.argv[1]).resolve()
source_dataset = sys.argv[2]
sfpo_root = Path(sys.argv[3]).resolve()
split_seed = int(sys.argv[4])
train_per_difficulty = int(sys.argv[5])
validation_per_difficulty = int(sys.argv[6])
if train_per_difficulty <= 0 or validation_per_difficulty <= 0:
    raise SystemExit("K&K train and validation counts must be positive")


def read_jsonl(path_in_repo: str) -> tuple[list[dict[str, Any]], Path]:
    path = Path(
        hf_hub_download(
            repo_id=source_dataset,
            filename=path_in_repo,
            repo_type="dataset",
            token=os.environ.get("HF_TOKEN"),
        )
    )
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream if line.strip()]
    return rows, path


def canonical_answer(row: dict[str, Any]) -> dict[str, str]:
    names = row.get("names")
    solution = row.get("solution")
    if not isinstance(names, list) or not isinstance(solution, list) or len(names) != len(solution):
        raise ValueError("names and solution must be same-length lists")
    return {str(name): ("knight" if bool(value) else "knave")
            for name, value in zip(names, solution)}


def format_response(row: dict[str, Any]) -> tuple[str, str]:
    answer = canonical_answer(row)
    cot_head = str(row.get("cot_head", "")).strip()
    repeat_steps = row.get("cot_repeat_steps", [])
    cot_foot = str(row.get("cot_foot", "")).strip()
    if not cot_head or not isinstance(repeat_steps, list) or not cot_foot:
        raise ValueError("K&K row is missing a complete chain-of-thought response")
    reasoning = "\n".join([cot_head, *(str(step).strip() for step in repeat_steps), cot_foot])
    if not reasoning.strip():
        raise ValueError("K&K row has an empty reasoning response")
    final = "\n".join(["Final Answer:", *(f"{name}: {label}" for name, label in answer.items())])
    return f"{reasoning}\n\n{final}", json.dumps(answer, sort_keys=True)


def convert_row(
    row: dict[str, Any], people: int, source_file: str, source_index: int,
    output_split: str, source_split: str,
) -> dict[str, Any]:
    prompt = str(row.get("quiz", "")).strip()
    if not prompt:
        raise ValueError("K&K row has an empty quiz")
    response, answer_json = format_response(row)
    return {
        "prompt": prompt,
        "response": response,
        "data_source": source_dataset,
        "ground_truth": answer_json,
        "people": int(people),
        "extra_info": {
            "split": output_split,
            "source_split": source_split,
            "source_file": source_file,
            "source_index": int(source_index),
            "source_dataset": source_dataset,
        },
    }


def write_parquet(name: str, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise SystemExit(f"refusing to write empty {name} split")
    pd.DataFrame(rows).to_parquet(data_root / f"{name}.parquet", index=False)


train_rows: list[dict[str, Any]] = []
validation_rows: list[dict[str, Any]] = []
iid_rows: list[dict[str, Any]] = []
ood_rows: list[dict[str, Any]] = []
source_files: dict[str, dict[str, Any]] = {}

for people in range(3, 7):
    relative = f"train/people{people}_num1000.jsonl"
    source, cached_path = read_jsonl(relative)
    source_files[relative] = {
        "sha256": hashlib.sha256(cached_path.read_bytes()).hexdigest(),
        "rows": len(source),
    }
    required = train_per_difficulty + validation_per_difficulty
    if len(source) < required:
        raise SystemExit(f"{relative} has {len(source)} rows; need {required}")
    order = list(range(len(source)))
    random.Random(split_seed + people).shuffle(order)
    for position, source_index in enumerate(order[:required]):
        output_split = "train" if position < train_per_difficulty else "test"
        converted = convert_row(
            source[source_index], people, relative, source_index,
            output_split, "train",
        )
        (train_rows if output_split == "train" else validation_rows).append(converted)

for people in range(3, 9):
    relative = f"test/people{people}_num100.jsonl"
    source, cached_path = read_jsonl(relative)
    source_files[relative] = {
        "sha256": hashlib.sha256(cached_path.read_bytes()).hexdigest(),
        "rows": len(source),
    }
    target = iid_rows if people <= 6 else ood_rows
    target.extend(
        convert_row(row, people, relative, index, "evaluation", "test")
        for index, row in enumerate(source)
    )

write_parquet("train", train_rows)
write_parquet("test", validation_rows)
write_parquet("iid_test", iid_rows)
write_parquet("ood_test", ood_rows)

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=sfpo_root.parents[1]
    ).decode().strip()
except Exception:
    git_commit = None

files = {}
for name in ("train", "test", "iid_test", "ood_test"):
    path = data_root / f"{name}.parquet"
    files[name] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(pd.read_parquet(path)),
    }

manifest = {
    "schema_version": 1,
    "kind": "sft_data",
    "source_dataset": source_dataset,
    "source_config": "train_and_test",
    "files": files,
    "source_files": source_files,
    "rows": {name: value["rows"] for name, value in files.items()},
    "training_people": [3, 4, 5, 6],
    "iid_test_people": [3, 4, 5, 6],
    "ood_test_people": [7, 8],
    "train_per_difficulty": train_per_difficulty,
    "validation_per_difficulty": validation_per_difficulty,
    "split_seed": split_seed,
    "conversion": "quiz_plus_cot_plus_canonical_boolean_answer",
    "git_commit": git_commit,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
(data_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

"$PYTHON_BIN" "$SFPO_ROOT/tools/validate_sft_data.py" \
  --data-root "$DATA_ROOT" --source-dataset "$SOURCE_DATASET"
