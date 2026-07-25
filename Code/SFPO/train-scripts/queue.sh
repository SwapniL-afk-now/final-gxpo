#!/usr/bin/env bash
# Generic sequential job queue for one GPU, with the VRAM-drain handoff fix.
# Waits for WAIT_PID to exit (the run currently holding the GPU), then runs each
# job in order, draining the GPU's VRAM before every launch so a lingering
# Ray/vLLM worker can't OOM the next run on step 1 (the earlier crash cause).
# Usage: ./train-scripts/queue.sh <gpu> <wait_pid> <script1> [script2 ...]
set -uo pipefail
GPU="${1:?gpu index}"; shift
WAIT_PID="${1:?wait pid}"; shift
cd /workspace/gradient-extrapolation-based-policy-optimization/Code/SFPO
source /workspace/Joint-Embedding-Guided-Policy-Optimization/.venv/bin/activate

echo "[queue gpu$GPU] waiting for pid $WAIT_PID to free GPU$GPU..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[queue gpu$GPU] GPU$GPU free -> starting chain $(date)"

for job in "$@"; do
  echo "=== [queue gpu$GPU] $(date) START $job ==="
  ./train-scripts/wait_gpu_free.sh "$GPU"   # drain prior run's VRAM before launch
  GPU="$GPU" TRAIN_SEED=42 "./train-scripts/$job" || echo "[queue gpu$GPU] $job FAILED (continuing)"
  echo "=== [queue gpu$GPU] $(date) END $job ==="
done
echo "[queue gpu$GPU] all jobs done $(date)"
