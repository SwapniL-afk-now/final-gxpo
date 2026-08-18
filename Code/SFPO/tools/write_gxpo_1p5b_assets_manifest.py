#!/usr/bin/env python3
"""Write non-secret metadata for the local 1.5B GXPO asset set."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

from verify_gxpo_math_assets import ASSETS, validate_asset


REPO_ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Qwen/Qwen2.5-Math-1.5B-Instruct"
MODEL_PATH = REPO_ROOT / "models" / "Qwen2.5-Math-1.5B-Instruct"
MANIFEST_PATH = REPO_ROOT / "assets" / "gxpo_1p5b_assets.json"


def main() -> int:
    if not MODEL_PATH.is_dir():
        raise SystemExit(f"Missing local model directory: {MODEL_PATH}")
    revision_path = REPO_ROOT / "assets" / "gxpo_1p5b_model_revision.txt"
    revision = revision_path.read_text(encoding="utf-8").strip() if revision_path.is_file() else "unknown"

    training = []
    for name in ("dapo", "lighteval"):
        spec = ASSETS[name]
        rows = validate_asset(name)
        training.append({
            "hf_id": spec["source"],
            "split": spec["split"],
            "selected_range": "all train examples" if name == "dapo" else "first 7500 examples",
            "row_count": rows,
            "local_parquet": str(Path(spec["path"]).resolve()),
        })

    benchmarks = []
    for name in ("math500", "aime24", "aime25", "amc", "minerva", "olympiadbench"):
        spec = ASSETS[name]
        rows = validate_asset(name)
        benchmarks.append({
            "name": name,
            "hf_id": spec["source"],
            "split": spec["split"],
            "row_count": rows,
            "local_parquet": str(Path(spec["path"]).resolve()),
        })

    manifest = {
        "model": {
            "hf_id": MODEL_ID,
            "local_path": str(MODEL_PATH.resolve()),
            "resolved_revision": revision,
        },
        "training_data": training,
        "benchmarks": benchmarks,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "preprocessing_scripts": {
            "dapo": "examples/data_preprocess/dapo_math.py",
            "lighteval": "examples/data_preprocess/lighteval_math.py",
            "math500": "examples/data_preprocess/test_math500.py",
            "aime24": "examples/data_preprocess/test_aime2024.py",
            "aime25": "examples/data_preprocess/test_aime2025.py",
            "amc": "examples/data_preprocess/test_amc.py",
            "minerva": "examples/data_preprocess/test_minervamath.py",
            "olympiadbench": "examples/data_preprocess/test_olympiad.py",
        },
    }
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Asset manifest: {MANIFEST_PATH}")
    print("Manifest contains non-secret metadata only.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
