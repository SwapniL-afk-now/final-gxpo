#!/usr/bin/env python3
"""Upload the missing public SFPO/GXPO checkpoint trees."""

import os
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "swapnil7777/learn-to-predict"
CODE = Path(__file__).resolve().parents[1]
ROOT = CODE / "results" / "gxpo_efficiency"
JOBS = [
    (
        ROOT / "qwen25_math_1p5b_sfpo_k10_a03_b64_mb16_w50_v2_20260819" / "global_step_190",
        "sfpo_k10_a03_w50/global_step_190",
    ),
    (
        ROOT / "qwen25_math_1p5b_sfpo_k10_a03_b64_mb16_w50_v2_20260819" / "global_step_400",
        "sfpo_k10_a03_w50/global_step_400",
    ),
    (
        ROOT / "qwen25_math_1p5b_gxpo_k10_a03_b64_mb16_w50_tau3_v2_20260819" / "global_step_190",
        "gxpo_k10_a03_w50_tau3/global_step_190",
    ),
    (
        ROOT / "qwen25_math_1p5b_gxpo_k10_a03_b64_mb16_w50_tau3_v2_20260819" / "global_step_400",
        "gxpo_k10_a03_w50_tau3/global_step_400",
    ),
]


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if not token:
        raise SystemExit("HF_TOKEN/HF_API_KEY is not set")
    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
    for local_dir, path_in_repo in JOBS:
        if not local_dir.is_dir():
            raise SystemExit(f"Missing checkpoint directory: {local_dir}")
        print(f"START {path_in_repo}", flush=True)
        api.upload_folder(
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=f"Upload {path_in_repo}",
        )
        print(f"DONE {path_in_repo}", flush=True)
    print(f"COMPLETE https://huggingface.co/{REPO_ID}", flush=True)


if __name__ == "__main__":
    main()
