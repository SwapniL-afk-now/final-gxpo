#!/usr/bin/env bash
# SFT baseline: plain 1-pass supervised finetuning of Qwen2.5-3B-Instruct on
# Bespoke-Stratos-17k (bespokelabs/Bespoke-Stratos-17k, DeepSeek-R1 traces;
# prepared by prep_bespoke_stratos_sft.py into train/val parquet with the
# shared system prompt folded into `prompt`). Paired with
# run_sft_gxpo_bespoke_3b.sh, which is identical except for the GXPO-style
# 3-pass update.
#
# Bespoke traces are long (combined p50 ~3.7k, p99 ~22k tokens), so unlike the
# DAPO pair this uses max_length=16384 with rows pre-filtered to fit, and
# micro_batch_size=1. lr 1e-5, batch 64, 200 steps.
# Usage: GPU=0 ./run_sft_baseline_bespoke_3b.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port), an invalid bootstrap address.
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

GPU="${GPU:?set GPU=0|1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LR=1e-5
PROJECT="${PROJECT:-gxpo-efficiency-final}"

MODEL="${MODEL:-/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-3B-Instruct/snapshots/aa8e72537993ba99e69dfaafa59ed015b17504d1}"
DATA_DIR="${DATA_DIR:-/office/dev_workspace/swapnil/data/Bespoke-Stratos-17k}"
TRAIN="${TRAIN:-$DATA_DIR/train.parquet}"
VAL="${VAL:-$DATA_DIR/val.parquet}"

EXP="sft_baseline_stratos3b_seed${TRAIN_SEED}"
RUN_DIR="${RUN_DIR:-./runs/${EXP}}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online
export WANDB_DIR="$RUN_DIR"
export MATH500="${MATH500:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/math500.parquet}"
export AIME24="${AIME24:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/aime24.parquet}"
export AIME25="${AIME25:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/aime25.parquet}"
export AMC23="${AMC23:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/amc23.parquet}"
export MINERVA="${MINERVA:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/minervamath.parquet}"
export OLYMPIAD="${OLYMPIAD:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench/olympiadbench.parquet}"

"${PYTHON_BIN:-python}" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE:-1}" \
    data.max_length=16384 \
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
    +trainer.test_freq=0 \
    +trainer.save_freq=0 \
    +trainer.benchmark_eval_freq=0 \
    trainer.seed=$TRAIN_SEED \
    2>&1 | tee "$RUN_DIR/train.log"

# Post-training greedy six-benchmark eval at step 200. SKIP_EVAL=1 to skip.
# --max-tokens 16384: Bespoke trains long thinking traces; truncating generation
# at 3072 would cut the boxed answer off and zero both arms' scores.
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  CKPT_ABS="$(cd "$RUN_DIR/global_step_200" && pwd)"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "${PYTHON_BIN:-python}" "$CODE_ROOT/tools/kd_sft/evaluate_greedy.py" \
    --checkpoint-dir "$CKPT_ABS" --base-model "$MODEL" \
    --data-files "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD" \
    --seed "${EVAL_SEED:-0}" --n 1 --temperature 0 --top-p 1.0 \
    --max-tokens 16384 --tp 1 --gpu-memory-utilization 0.85 \
    --output "$RUN_DIR/eval_greedy_6bench.json" 2>&1 | tee "$RUN_DIR/eval_greedy_6bench.log"
fi
