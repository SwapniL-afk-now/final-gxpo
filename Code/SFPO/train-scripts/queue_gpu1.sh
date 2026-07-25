#!/usr/bin/env bash
# GPU1 queue: after gxpo_muon (pid $1) frees GPU1, run the code runs then 1 ablation.
# Balanced against GPU0's queue so both GPUs finish at ~the same time.
# Usage: ./train-scripts/queue_gpu1.sh <gxpo_muon_pid>
set -uo pipefail
WAIT_PID="$1"
cd /workspace/gradient-extrapolation-based-policy-optimization/Code/SFPO
source /workspace/Joint-Embedding-Guided-Policy-Optimization/.venv/bin/activate

echo "[gpu1-queue] waiting for gxpo_muon (pid $WAIT_PID) to free GPU1..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[gpu1-queue] GPU1 free -> starting chain $(date)"

for job in \
  "run_rebuttal_code_grpo.sh" \
  "run_rebuttal_code_gxpo.sh" \
  "run_ablation_temporary_fallback.sh" ; do
  echo "=== [gpu1-queue] $(date) START $job ==="
  ./train-scripts/wait_gpu_free.sh 1   # drain prior run's VRAM before launch (OOM-handoff fix)
  GPU=1 TRAIN_SEED=42 "./train-scripts/$job" || echo "[gpu1-queue] $job FAILED (continuing)"
  echo "=== [gpu1-queue] $(date) END $job ==="
done
echo "[gpu1-queue] all GPU1 jobs done $(date)"
