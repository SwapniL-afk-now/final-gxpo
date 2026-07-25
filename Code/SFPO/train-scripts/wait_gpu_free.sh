#!/usr/bin/env bash
# Wait until a GPU's memory has actually drained before launching the next job.
# Root-cause fix for the back-to-back-handoff OOM: the previous run's Ray/vLLM
# workers keep holding VRAM for seconds-to-minutes after its Python PID exits, so
# the next run peaks over the 97GB cap on step 1 and gets OOM-killed. Polling real
# free memory (not a blind sleep) is what actually removes the race.
# Usage: ./wait_gpu_free.sh <gpu_index> [free_mib_threshold] [timeout_s]
set -uo pipefail
GPU="${1:?gpu index}"
NEED_FREE="${2:-90000}"   # require ~>=88GB free (a clean run needs ~90GB on this card)
TIMEOUT="${3:-600}"
start=$(date +%s)
while :; do
  free=$(nvidia-smi -i "$GPU" --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
  free=${free:-0}
  if [ "$free" -ge "$NEED_FREE" ]; then
    echo "[wait_gpu_free] GPU$GPU has ${free}MiB free (>=${NEED_FREE}) -> ok $(date)"
    sleep 10   # small settle so ctx/driver bookkeeping fully lands
    exit 0
  fi
  if [ $(( $(date +%s) - start )) -ge "$TIMEOUT" ]; then
    echo "[wait_gpu_free] TIMEOUT after ${TIMEOUT}s; GPU$GPU only ${free}MiB free (<${NEED_FREE}). Proceeding anyway." >&2
    exit 0
  fi
  echo "[wait_gpu_free] GPU$GPU ${free}MiB free (<${NEED_FREE}); waiting..."
  sleep 15
done
