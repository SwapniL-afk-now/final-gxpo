#!/usr/bin/env bash
# Run the 4 Group-2 GXPO ablations sequentially on one GPU (mirrors queue_grpo_seeds.sh).
# Reviewers can instead run any single run_ablation_*.sh on its own; this is just our runner.
# Usage: GPU=1 ./train-scripts/queue_ablations.sh
set -uo pipefail
GPU="${GPU:?set GPU=0|1}"
cd "$(dirname "$0")/.."
for s in no_corrective direct_full_extrapolation no_fallback temporary_fallback; do
  echo "=== [$(date)] starting ablation: $s on GPU $GPU ==="
  GPU="$GPU" "./train-scripts/run_ablation_${s}.sh"
  echo "=== [$(date)] finished ablation: $s (exit $?) ==="
done
echo "=== all ablations done ==="
