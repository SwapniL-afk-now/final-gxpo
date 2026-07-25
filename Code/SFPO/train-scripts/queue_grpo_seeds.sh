#!/usr/bin/env bash
# Queue the 3 rebuttal GRPO seeds sequentially on one GPU (matches the sfpo/gxpo 3-seed setup).
# Usage: GPU=0 ./train-scripts/queue_grpo_seeds.sh
set -uo pipefail
GPU="${GPU:?set GPU=0|1}"
cd "$(dirname "$0")/.."
for SEED in 42 123 777; do
  echo "=== [$(date)] starting GRPO seed $SEED on GPU $GPU ==="
  TRAIN_SEED=$SEED GPU=$GPU ./train-scripts/run_rebuttal_grpo.sh
  echo "=== [$(date)] finished GRPO seed $SEED (exit $?) ==="
done
echo "=== all GRPO seeds done ==="
