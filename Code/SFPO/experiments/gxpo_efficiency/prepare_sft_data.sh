#!/usr/bin/env bash
# Prepare full worked-solution SFT data for the Qwen2.5-Math comparison.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../../.." && pwd)"
SFPO_ROOT="$REPO_ROOT/Code/SFPO"
PYTHON_BIN="${PYTHON_BIN:-$REPO_ROOT/.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python)"
fi

DATA_ROOT="${SFT_DATA_ROOT:-$SFPO_ROOT/data/math_l35_sft}"
LEVELS="${SFT_LEVELS:-3,4,5}"
FORCE_REBUILD="${SFT_FORCE_REBUILD:-0}"
mkdir -p "$DATA_ROOT"

if [[ -s "$DATA_ROOT/train.parquet" && -s "$DATA_ROOT/test.parquet" && "$FORCE_REBUILD" != "1" ]]; then
  echo "SFT data already exists; validating without overwriting: $DATA_ROOT"
  "$PYTHON_BIN" "$SFPO_ROOT/tools/validate_sft_data.py" --data-root "$DATA_ROOT"
  exit 0
fi

cd "$SFPO_ROOT"
"$PYTHON_BIN" examples/data_preprocess/math_dataset.py \
  --local_dir "$DATA_ROOT" \
  --levels "$LEVELS" \
  --sft

"$PYTHON_BIN" "$SFPO_ROOT/tools/validate_sft_data.py" --data-root "$DATA_ROOT"

"$PYTHON_BIN" - "$DATA_ROOT" "$LEVELS" "$SFPO_ROOT/data" <<'PY'
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

data_root = Path(sys.argv[1]).resolve()
levels = sys.argv[2]
benchmark_root = Path(sys.argv[3]).resolve()
files = {}
rows = {}
for split in ("train", "test"):
    path = data_root / f"{split}.parquet"
    files[split] = {
        "path": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }
    rows[split] = int(len(pd.read_parquet(path)))

benchmarks = [
    str(benchmark_root / "math500/test.parquet"),
    str(benchmark_root / "aime2024/test.parquet"),
    str(benchmark_root / "aime2025/test.parquet"),
    str(benchmark_root / "amc/test.parquet"),
    str(benchmark_root / "minervamath/test.parquet"),
    str(benchmark_root / "olympiadbench/test.parquet"),
]
sft_paths = {str((data_root / f"{split}.parquet").resolve()) for split in ("train", "test")}
if sft_paths.intersection(str(Path(path).resolve()) for path in benchmarks):
    raise SystemExit("SFT output overlaps a benchmark evaluation parquet")

try:
    git_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=benchmark_root.parents[1]
    ).decode().strip()
except Exception:
    git_commit = None

manifest = {
    "schema_version": 1,
    "kind": "sft_data",
    "source_dataset": "DigitalLearningGmbH/MATH-lighteval",
    "levels": [level.strip() for level in levels.split(",")],
    "files": files,
    "rows": rows,
    "benchmark_evaluation_files": benchmarks,
    "git_commit": git_commit,
    "created_at_utc": datetime.now(timezone.utc).isoformat(),
}
(data_root / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
print(f"Wrote {data_root / 'manifest.json'}")
PY
