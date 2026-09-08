#!/usr/bin/env bash
# SFT baseline: plain 1-pass supervised finetuning of Llama-3.2-3B-Instruct on DAPO
# problems with gpt-oss-120b medium-effort teacher traces (reward==1 only, ~14k rows;
# built by prep_dapo_oss_medium_sft.py). Mirrors run_sft_baseline.sh (Qwen2.5-1.5B);
# paired with run_sft_gxpo_llama32_3b.sh, which is identical except for the
# GXPO-style 3-pass update.
#
# lr 1e-5, batch 64, 200 steps, validate every 10 steps, seed 42.
# Trains, then scores the final checkpoint on all 6 benchmarks with the same
# vLLM harness as the KD/RL arms (eval_sft_6bench.sh).
#
# Memory (3.21B params, fp32 master): params 12.9 + AdamW moments 25.7 + grads
# 12.9 = ~51.5GB static + ~5-8GB activations (micro 2 x 8192, ckpt on) ≈ 58GB
# peak on 1 GPU -- fits an 80GB card with headroom. Single-GPU on purpose: the
# GXPO arm needs 2-way FSDP sharding (its theta0/g0/g1 buffers + optimizer
# snapshot add ~64GB), so run the two 3B arms SEQUENTIALLY, not concurrently.
# Usage: GPU_IDS=0 ./run_sft_baseline_llama32_3b.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port), an invalid bootstrap address.
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

GPU_IDS="${GPU_IDS:-0}"
NPROC="$(awk -F, '{print NF}' <<<"$GPU_IDS")"
TRAIN_SEED="${TRAIN_SEED:-42}"
# Fragmentation guard for the repeated large alloc/free cycles (optimizer
# snapshots, FSDP all-gather buffers). Safe here: no in-process vLLM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LR=1e-5
PROJECT=rebuttul

MODEL="${MODEL:-${MODEL_LLAMA32_3B:-/workspace/models/Llama-3.2-3B-Instruct}}"
DATA_DIR="${DATA_DIR:-/office/dev_workspace/swapnil/data/DAPO-MATH-17k-oss-reasoning}"
TRAIN="${TRAIN:-$DATA_DIR/math-oss-medium-17k.jsonl}"
VAL="${VAL:-$DATA_DIR/math-oss-medium-17k.jsonl}"

EXP="sft_baseline_llama32_3b_dapoossmed_seed${TRAIN_SEED}"
RUN_DIR="${RUN_DIR:-./runs/${EXP}}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online
export WANDB_DIR="$RUN_DIR"
# In-training vLLM eval is OFF (benchmark_eval_freq=0): the GXPO arm must run
# 2-GPU, where the trainer hard-refuses periodic eval (world size != 1 raises),
# so both 3B arms skip it for symmetry and rely on the post-training eval below.
export SFT_VLLM_EVAL_SCRIPT="${SFT_VLLM_EVAL_SCRIPT:-$(cd "$(dirname "$0")" && pwd)/eval_sft_6bench.sh}"
export SFT_VLLM_EVAL_OUTPUT_ROOT="${SFT_VLLM_EVAL_OUTPUT_ROOT:-$RUN_DIR/vllm_eval}"
export SFT_VLLM_EVAL_MAX_TOKENS="${SFT_VLLM_EVAL_MAX_TOKENS:-16384}"
export SFT_VLLM_EVAL_GPU_UTIL="${SFT_VLLM_EVAL_GPU_UTIL:-0.9}"
export SFT_VLLM_KEEP_CHECKPOINTS="${SFT_VLLM_KEEP_CHECKPOINTS:-final}"

"${PYTHON_BIN:-python}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE:-2}" \
    data.max_length=8192 \
    data.truncation=right \
    model.partial_pretrain="$MODEL" \
    model.enable_gradient_checkpointing=True \
    optim.lr=$LR \
    optim.betas="[0.9,0.999]" \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.0 \
    optim.clip_grad=1.0 \
    use_remove_padding=true \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs=3 \
    trainer.total_training_steps=200 \
    +trainer.test_freq=10 \
    +trainer.save_freq=100 \
    +trainer.benchmark_eval_freq=0 \
    +trainer.val_max_batches=50 \
    trainer.seed=$TRAIN_SEED \
    2>&1 | tee "$RUN_DIR/train.log"

# Post-training 6-benchmark eval (amc23, aime24, aime25, math500,
# olympiadbench, minervamath) with the same harness and metric flags as the
# GXPO arm. SKIP_EVAL=1 to skip.
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  CKPT_ABS="$(cd "$RUN_DIR/global_step_200" && pwd)"
  GPU="${GPU_IDS%%,*}" ./eval_sft_6bench.sh "$CKPT_ABS" 2>&1 | tee "$RUN_DIR/eval_6bench.log"
fi
