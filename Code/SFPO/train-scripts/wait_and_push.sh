#!/usr/bin/env bash
# Wait for a running training PID to exit, then push its best checkpoint to HF --
# but only if the run actually FINISHED (reached MAX_STEPS=200); a crash is not pushed.
# Usage: ./train-scripts/wait_and_push.sh <pid> <run_dir_name> <hf_repo_id>
set -uo pipefail
PID="$1"; RUN="$2"; REPO="$3"
SFPO=/workspace/gradient-extrapolation-based-policy-optimization/Code/SFPO
cd "$SFPO"
set -a; . /workspace/gradient-extrapolation-based-policy-optimization/.env; set +a
export HUGGING_FACE_HUB_TOKEN="${HF_TOKEN}"
source /workspace/Joint-Embedding-Guided-Policy-Optimization/.venv/bin/activate

echo "[push-queue] waiting for pid $PID ($RUN) to finish..."
while kill -0 "$PID" 2>/dev/null; do sleep 60; done
echo "[push-queue] $RUN process ended $(date); waiting 30s for final ckpt flush"
sleep 30

LAST=$(grep -oE "step:[0-9]+ - critic" "runs/$RUN/train.log" 2>/dev/null | grep -oE "[0-9]+" | sort -n | tail -1)
if [ "${LAST:-0}" -lt 200 ]; then
  echo "[push-queue] $RUN ended at step ${LAST:-0} < 200 (not finished) -- SKIPPING push"; exit 0
fi
CKPT=$(ls -dt runs/$RUN/global_step_*/ 2>/dev/null | head -1)
if [ -z "$CKPT" ]; then echo "[push-queue] ERROR: no checkpoint dir in runs/$RUN"; exit 1; fi
echo "[push-queue] finished at step $LAST; pushing ${CKPT}actor -> $REPO"
python train-scripts/push_ckpt.py "${CKPT}actor" "$REPO"
echo "[push-queue] done: https://huggingface.co/$REPO"
