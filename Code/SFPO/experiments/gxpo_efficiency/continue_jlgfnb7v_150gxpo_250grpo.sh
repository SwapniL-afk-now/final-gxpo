#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../../.." && pwd)"
RUN_NAME="qwen25-math-1p5b_gxpo_muon_k5_a05_b256_mb64_gpu12_tp1_seed3407"
RUN_DIR="$REPO_ROOT/Code/SFPO/results/gxpo_efficiency/$RUN_NAME"
CHECKPOINT="$RUN_DIR/global_step_400/actor"
TRAINER="$SCRIPT_DIR/qwen25_math_1p5b_gxpo_muon_transactional_b64_mb16.sh"
MERGER="/office/dev_workspace/swapnil/final-gxpo-h200/Code/SFPO/scripts/model_merger.py"
PYTHON="/office/dev_workspace/swapnil/final-gxpo-h200/.venv/bin/python"

if [[ -f "$REPO_ROOT/.env" ]]; then
  set -a
  # shellcheck disable=SC1091
  source "$REPO_ROOT/.env"
  set +a
fi

[[ -f "$RUN_DIR/wandb_id.txt" ]] || { echo "Missing W&B run id: $RUN_DIR/wandb_id.txt" >&2; exit 2; }
[[ -d "$CHECKPOINT" ]] || { echo "Missing resume checkpoint: $CHECKPOINT" >&2; exit 2; }

# common.sh writes train.log with tee; retain the completed run's local log.
if [[ -f "$RUN_DIR/train.log" && ! -e "$RUN_DIR/train_step_1_400.log" ]]; then
  cp "$RUN_DIR/train.log" "$RUN_DIR/train_step_1_400.log"
fi

export GPU_IDS="1,2"
export GPU_COUNT="2"
export FSDP_SIZE="2"
export MAX_STEPS="800"
export SAVE_FREQ="20"
export GXPO_MAX_ACTIVE_STEPS="150"
export GXPO_RUN_NAME="$RUN_NAME"
export TRAINER_RESUME_MODE="auto"
export TRAINER_RESUME_FROM_PATH="False"
export WANDB_PROJECT="gxpo-efficiency-final"
export FINAL_EVAL_ENABLED="False"

# The GXPO launcher creates a fresh local GXPO gate on resume: global steps
# 401-550 use GXPO, then the hard budget makes steps 551-800 ordinary GRPO.
bash "$TRAINER"

HF_REPO_ID="${HF_REPO_ID:-swapnil7777/learn-to-predict}"
HF_PATH="$RUN_DIR/global_step_800/actor/huggingface"

"$PYTHON" "$MERGER" --local_dir "$RUN_DIR/global_step_800/actor"

HF_REPO_PATH="$RUN_NAME/global_step_800/actor/huggingface"
"$PYTHON" - "$HF_PATH" "$HF_REPO_ID" "$HF_REPO_PATH" <<'PY'
import os
import sys
from huggingface_hub import HfApi

folder, repo_id, path_in_repo = sys.argv[1:]
token = os.environ["HF_TOKEN"]
api = HfApi(token=token)
api.create_repo(repo_id=repo_id, repo_type="model", private=False, exist_ok=True)
api.upload_folder(
    folder_path=folder,
    path_in_repo=path_in_repo,
    repo_id=repo_id,
    repo_type="model",
    commit_message="Upload continued GXPO/GRPO checkpoint at global step 800",
)
print(f"Uploaded {folder} to https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")
PY
