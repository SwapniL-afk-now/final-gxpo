#!/usr/bin/env bash
# GPU0 queue: after grpo seed777 (pid $1) frees GPU0, run 3 GXPO ablations.
# Balanced against GPU1's queue so both GPUs finish at ~the same time.
# Usage: ./train-scripts/queue_gpu0.sh <seed777_pid>
set -uo pipefail
WAIT_PID="$1"
cd /workspace/gradient-extrapolation-based-policy-optimization/Code/SFPO
source /workspace/Joint-Embedding-Guided-Policy-Optimization/.venv/bin/activate

echo "[gpu0-queue] waiting for grpo_seed777 (pid $WAIT_PID) to free GPU0..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[gpu0-queue] GPU0 free -> starting chain $(date)"

for job in \
  "run_ablation_direct_full_extrapolation.sh" \
  "run_ablation_no_corrective.sh" \
  "run_ablation_no_fallback.sh" ; do
  echo "=== [gpu0-queue] $(date) START $job ==="
  ./train-scripts/wait_gpu_free.sh 0   # drain prior run's VRAM before launch (OOM-handoff fix)
  GPU=0 TRAIN_SEED=42 "./train-scripts/$job" || echo "[gpu0-queue] $job FAILED (continuing)"
  echo "=== [gpu0-queue] $(date) END $job ==="
done
echo "[gpu0-queue] all GPU0 jobs done $(date)"
