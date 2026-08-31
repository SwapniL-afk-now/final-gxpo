#!/usr/bin/env bash
# Download GAIR/lima and convert it to the local prompt/response SFT contract.
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

DATA_ROOT="${SFT_DATA_ROOT:-$SFPO_ROOT/data/lima_sft}"
SOURCE_DATASET="${SFT_SOURCE_DATASET:-GAIR/lima}"
VALIDATION_FRACTION="${SFT_VALIDATION_FRACTION:-0.1}"
SPLIT_SEED="${SFT_SPLIT_SEED:-42}"
FORCE_REBUILD="${SFT_FORCE_REBUILD:-0}"
mkdir -p "$DATA_ROOT"

if [[ -s "$DATA_ROOT/train.parquet" && -s "$DATA_ROOT/test.parquet" && "$FORCE_REBUILD" != "1" ]]; then
  echo "LIMA SFT data already exists; validating without overwriting: $DATA_ROOT"
  "$PYTHON_BIN" "$SFPO_ROOT/tools/validate_sft_data.py" \
    --data-root "$DATA_ROOT" --source-dataset "$SOURCE_DATASET"
  exit 0
fi

"$PYTHON_BIN" - "$DATA_ROOT" "$SOURCE_DATASET" "$SFPO_ROOT" \
  "$VALIDATION_FRACTION" "$SPLIT_SEED" <<'PY'
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
validation_fraction = float(sys.argv[4])
split_seed = int(sys.argv[5])
if not 0.0 < validation_fraction < 1.0:
    raise SystemExit("SFT_VALIDATION_FRACTION must be between 0 and 1")


def as_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, dict):
        for key in ("content", "text", "value"):
            if key in value:
                return as_text(value[key])
    return str(value).strip() if value is not None else ""


def conversation_to_pair(value: Any) -> tuple[str, str]:
    """Convert LIMA's conversation field into one prompt and final response."""
    if not isinstance(value, list):
        raise ValueError("unsupported conversation value")
    turns = []
    for item in value:
        if isinstance(item, dict):
            role = str(item.get("role", item.get("from", "user"))).lower()
            content = as_text(
                item.get("content", item.get("value", item.get("text", "")))
            )
        else:
            role = "user" if len(turns) % 2 == 0 else "assistant"
            content = as_text(item)
        if role in {"human", "user"}:
            role = "user"
        elif role in {"gpt", "assistant", "model", "bot"}:
            role = "assistant"
        if content:
            turns.append((role, content))
    if len(turns) < 2:
        raise ValueError("conversation must contain at least a user turn and response")
    response_role, response = turns[-1]
    if response_role != "assistant":
        raise ValueError("conversation final turn is not an assistant response")
    history = turns[:-1]
    prompt_parts = []
    for role, content in history:
        label = "User" if role in {"user", "human"} else "Assistant"
        prompt_parts.append(f"{label}: {content}")
    return "\n\n".join(prompt_parts), response


def convert_row(
    row: dict[str, Any], split: str, index: int, source_split: str = "train"
) -> dict[str, Any]:
    if "conversations" in row:
        prompt, response = conversation_to_pair(row["conversations"])
    else:
        prompt = as_text(row.get("prompt", row.get("instruction", row.get("question"))))
        response = as_text(row.get("response", row.get("output", row.get("answer"))))
    if not prompt or not response:
        raise ValueError(f"{split} row {index} has an empty prompt or response")
    return {
        "prompt": prompt,
        "response": response,
        "data_source": source_dataset,
        "extra_info": {
            "split": split,
            "index": index,
            "source_split": source_split,
            "source_dataset": source_dataset,
        },
    }


def read_jsonl(split: str) -> list[dict[str, Any]]:
    path = hf_hub_download(
        repo_id=source_dataset,
        filename=f"{split}.jsonl",
        repo_type="dataset",
        token=os.environ.get("HF_TOKEN"),
    )
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]

source_train = read_jsonl("train")
converted_train = []
skipped_rows = []
for index, row in enumerate(source_train):
    try:
        converted_train.append(convert_row(row, "train", index))
    except ValueError as exc:
        # The Hub train split contains one incomplete multi-turn record. Do not
        # invent an assistant target for it; retain the exclusion in provenance.
        skipped_rows.append({"split": "train", "index": index, "reason": str(exc)})

if len(converted_train) < 2:
    raise SystemExit("LIMA train split has fewer than two usable prompt/response rows")

validation_count = max(1, round(len(converted_train) * validation_fraction))
if validation_count >= len(converted_train):
    raise SystemExit("validation split would consume the entire usable LIMA train split")
order = list(range(len(converted_train)))
random.Random(split_seed).shuffle(order)
validation_indices = set(order[:validation_count])
rows = {"train": [], "test": []}
for converted_index, row in enumerate(converted_train):
    split = "test" if converted_index in validation_indices else "train"
    row["extra_info"]["split"] = split
    rows[split].append(row)

for split in rows:
    pd.DataFrame(rows[split]).to_parquet(data_root / f"{split}.parquet", index=False)

# The official LIMA test split is prompt-only. Preserve it for future
# generation-based evaluation, but never treat it as supervised targets.
official_test = read_jsonl("test")
official_prompts = []
for index, row in enumerate(official_test):
    conversation = row.get("conversations")
    if not isinstance(conversation, list) or not conversation:
        raise ValueError(f"official test row {index} has no prompt conversation")
    official_prompts.append({
        "prompt": as_text(conversation[0]),
        "data_source": source_dataset,
        "extra_info": {"index": index, "source_split": "test", "source_dataset": source_dataset},
    })
(data_root / "official_test_prompts.jsonl").write_text(
    "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in official_prompts)
)

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=sfpo_root.parents[1]
    ).decode().strip()
except Exception:
    git_commit = None

files = {}
for split in ("train", "test"):
    path = data_root / f"{split}.parquet"
    files[split] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "rows": len(rows[split]),
    }
official_test_path = data_root / "official_test_prompts.jsonl"
files["official_test_prompts"] = {
    "path": str(official_test_path),
    "sha256": hashlib.sha256(official_test_path.read_bytes()).hexdigest(),
    "rows": len(official_prompts),
}

manifest = {
    "schema_version": 1,
    "kind": "sft_data",
    "source_dataset": source_dataset,
    "source_config": "default",
    "files": files,
    "rows": {split: len(rows[split]) for split in ("train", "test")},
    "official_test_rows": len(official_prompts),
    "validation_fraction": validation_fraction,
    "split_seed": split_seed,
    "skipped_rows": skipped_rows,
    "conversion": "conversations_to_prompt_response",
    "git_commit": git_commit,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
(data_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(json.dumps(manifest, indent=2, sort_keys=True))
PY

"$PYTHON_BIN" "$SFPO_ROOT/tools/validate_sft_data.py" \
  --data-root "$DATA_ROOT" --source-dataset "$SOURCE_DATASET"
