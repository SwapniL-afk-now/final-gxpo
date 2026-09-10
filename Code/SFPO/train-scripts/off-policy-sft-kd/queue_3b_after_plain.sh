#!/usr/bin/env bash
# Waits for the running 1.5B plain SFT+KL job to exit, then starts the same
# recipe on Qwen2.5-3B-Instruct on the SAME GPU (0).
#
# Waiting on the launcher PID (not a log marker) means this also releases if the
# 1.5B run dies early -- the GPU is free either way, which is the condition that
# actually matters. The GPU-free poll after it is the real gate.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
D="$(pwd)"

WAIT_PID="${WAIT_PID:?set WAIT_PID}"
GPU="${GPU:-0}"

echo "[queue] $(date '+%F %T') waiting for pid $WAIT_PID (1.5B plain) to exit..."
while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 60; done
echo "[queue] $(date '+%F %T') pid $WAIT_PID gone."

# Do not race the dying process's VRAM release, or the 3B load OOMs on startup.
echo "[queue] waiting for GPU $GPU to drop below 5GB..."
for _ in $(seq 1 120); do
  used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU" | tr -d ' ')
  echo "[queue] GPU $GPU used=${used}MiB"
  [[ "$used" -lt 5000 ]] && break
  sleep 30
done
sleep 30

MODEL_3B=/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1
# MAX_LENGTH tracks the 32B-trace default (p50 3,172 tokens -- 2688 would drop
# over half the cache); MAX_TOKEN_LEN_PER_GPU likewise must clear the longest
# real row (10,425), hence 12288.
export EXP="sftkl_3b_b0.1_r1_32b_flat16k_b96_lr1e-6_seed42"

echo "[queue] $(date '+%F %T') launching 3B: EXP=$EXP GPU=$GPU"
GPU="$GPU" \
MODEL="$MODEL_3B" \
MAX_STEPS="${MAX_STEPS:-400}" \
MAX_LENGTH=16384 \
MAX_TOKEN_LEN_PER_GPU=12288 \
MICRO_BATCH_SIZE=2 \
EXP="$EXP" \
RUN_DIR="$D/runs/$EXP" \
  ./run_kd_sft_hybrid_dapo_lighteval_1p5b.sh
echo "[queue] $(date '+%F %T') 3B finished EXIT=$?"
