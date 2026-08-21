#!/usr/bin/env python3
"""Upload the completed Llama 3.2 3B GRPO checkpoint trees."""

import os
from pathlib import Path

from huggingface_hub import HfApi


REPO_ID = "swapnil7777/learn-to-predict"
ROOT = Path(__file__).resolve().parents[1] / "results" / "gxpo_efficiency"
RUN = ROOT / "llama32_3b_grpo_b64_mb16_seed3407_v1_20260820"
JOBS = [
    (RUN / "global_step_290", "llama32_3b_grpo_b64_mb16_seed3407_v1/global_step_290"),
    (RUN / "global_step_400", "llama32_3b_grpo_b64_mb16_seed3407_v1/global_step_400"),
]
REQUIRED = (
    "actor/model_world_size_1_rank_0.pt",
    "actor/optim_world_size_1_rank_0.pt",
    "actor/extra_state_world_size_1_rank_0.pt",
    "data.pt",
    "data_profiler.pt",
)


def main() -> None:
    token = os.environ.get("HF_TOKEN") or os.environ.get("HF_API_KEY")
    if not token:
        raise SystemExit("HF_TOKEN/HF_API_KEY is not set")
    for local_dir, _ in JOBS:
        missing = [name for name in REQUIRED if not (local_dir / name).is_file()]
        if missing:
            raise SystemExit(f"Incomplete checkpoint {local_dir}: missing {missing}")

    api = HfApi(token=token)
    api.create_repo(repo_id=REPO_ID, repo_type="model", private=False, exist_ok=True)
    for local_dir, path_in_repo in JOBS:
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
