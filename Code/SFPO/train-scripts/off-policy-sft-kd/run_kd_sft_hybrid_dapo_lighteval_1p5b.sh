#!/usr/bin/env bash
# Off-policy SFT knowledge distillation on DeepScaleR problem/solution pairs:
# supervised CE plus kl_beta * D_KL(p_teacher || p_student), one plain optimizer
# step per batch (verl/trainer/fsdp_sft_trainer.py).
#
# The KL is a DISTILLATION term, not the RL-style anchor this file used to run.
# Two things changed: the frozen model is the DeepScaleR-1.5B-Preview teacher
# rather than a copy of the student's own init, and the divergence is a FULL
# forward KL summed over all 151,936 logits at every supervised token
# (data.kl_full=True) rather than the K3 one-token estimator of the reverse KL.
# Both models are fed the same student tokenization, so position i is the same
# token for both and their logit axes are the same vocabulary -- that is all
# "matched tokens" requires. No teacher top-K cache and no teacher generation:
# the teacher only runs forward passes. `solution` remains the hard CE target.
#
# Training data is downloaded from
# sam-12labs/DeepScaleR-Preview-Dataset_DeepSeek-R1-Distill-Qwen-32B_reasoning_traces
# and cached locally by prep_32b_traces_kd.py. Each upstream row holds one
# `problem` plus a LIST of 32B reasoning traces with a per-trace correctness
# flag; the cache explodes it into one row per CORRECT trace with `problem` as
# the prompt and the trace as `solution`. A problem with several correct traces
# therefore yields several same-prompt rows, which the row-based SFTDataset
# consumes as independent samples. The validation holdout is at PROBLEM level
# (seed-42 shuffled), which is what makes val/loss meaningful here.
#
# Validation loss and the six-benchmark eval both run every 10 steps; the final
# pass below uses two independent TP=1 vLLM workers and logs sampled
# pass@4/average@4 metrics at temperature 0.7.
#
# Multi-GPU is supported WITH in-training eval in this tree: every rank offloads
# to CPU, rank 0 runs the vLLM subprocess pinned to its own GPU, the other ranks
# resume onto theirs, and the barrier after it has a 10h timeout. (An older tree
# raised on world_size != 1; that guard is not here.)
#
# Usage:
#   GPU=0 ./run_kd_sft_hybrid_dapo_lighteval_1p5b.sh
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_ADDRESS=local

# Bind every launcher to the standalone target environment. Do not inherit a
# deleted/stale venv, CUDA toolkit, or FlashInfer backend from the caller.
VENV_ROOT="${VENV_ROOT:-/office/dev_workspace/swapnil/final-gxpo-h200/.venv}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_ROOT/bin/python}"
export VIRTUAL_ENV="$VENV_ROOT"
export PATH="$VENV_ROOT/bin:$VENV_ROOT/lib/python3.12/site-packages/nvidia/cu13/bin:$PATH"
export CUDA_HOME="${CUDA_HOME:-$VENV_ROOT/lib/python3.12/site-packages/nvidia/cu13}"
export CUDA_PATH="$CUDA_HOME"
export CUDACXX="${CUDACXX:-$CUDA_HOME/bin/nvcc}"
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
GPU="${GPU:?set GPU=0|1, or a comma list such as GPU=0,1}"
export GPU
NPROC="$(awk -F, '{print NF}' <<< "$GPU")"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-$(( NPROC > 1 ? 0 : 1 ))}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-$(( NPROC > 1 ? 0 : 1 ))}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LR="${LR:-5e-6}"
PROJECT="${PROJECT:-gxpo-efficiency-final}"

# Uses final-gxpo-h200's venv specifically (this repo, final-gxpo, ships no
# .venv of its own). Override PYTHON_BIN to point elsewhere.
MODEL="${MODEL:-/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
MAX_LENGTH="${MAX_LENGTH:-16384}"
DATASET_ID="${DATASET_ID:-sam-12labs/DeepScaleR-Preview-Dataset_DeepSeek-R1-Distill-Qwen-32B_reasoning_traces}"
DATASET_DIR="${DATASET_DIR:-$CODE_ROOT/data/r1_32b_traces}"
VAL_SIZE="${VAL_SIZE:-1000}"
DATA_SEED="${DATA_SEED:-42}"
TRAIN="${TRAIN:-$DATASET_DIR/train_32b_correct_max${MAX_LENGTH}.parquet}"
VAL="${VAL:-$DATASET_DIR/val_32b_correct_max${MAX_LENGTH}_${VAL_SIZE}.parquet}"
if [[ ! -f "$TRAIN" || ! -f "$VAL" ]]; then
  mkdir -p "$DATASET_DIR"
  "$PYTHON_BIN" "$(dirname "${BASH_SOURCE[0]}")/prep_32b_traces_kd.py" \
    "$DATASET_ID" "$DATASET_DIR" "$MODEL" "$MAX_LENGTH" "$VAL_SIZE" "$DATA_SEED"
fi

# The distillation teacher. DeepScaleR-1.5B-Preview shares the student's Qwen2
# architecture and its 151,936 vocabulary, so one tokenization serves both and
# the trainer's vocab_size preflight passes. The trainer builds it from ITS OWN
# config, not the student's -- the two disagree on rope_theta (1e4 vs 1e6) and
# tie_word_embeddings (False vs True), and the student config would silently
# drop the teacher's untied lm_head.
KL_REF_MODEL="${KL_REF_MODEL:-/office/dev_workspace/swapnil/final-gxpo/models/DeepScaleR-1.5B-Preview}"
# Distillation weight: loss = CE + KL_BETA * KL(teacher || student). 1.0 is plain
# CE+KL. This is NOT comparable to the 0.1 the old K3 anchor used -- that anchor
# started at exactly 0 (reference == student init) while a full teacher KL starts
# at several nats/token. Check train/kl on a smoke run before a long one.
KL_BETA="${KL_BETA:-1.0}"
# Token budget per micro-batch. The binding constraint is the [tokens, 151936]
# logits tensor in the CE path (~3.7GB bf16 at 12288) plus ~1.5GB for the fp32
# KL chunk. Must stay above the longest real sequence -- the trainer RAISES if
# one row exceeds the budget -- and the 32B correct traces run long: longest
# 10,425, p99 7,575, p50 3,172 student tokens (prompt + trace + eos), so 8192
# is too small and 12288 is the setting.
MAX_TOKEN_LEN_PER_GPU="${MAX_TOKEN_LEN_PER_GPU:-12288}"
# FSDP sharding strategy. FULL_SHARD (ZeRO-3) all-gathers every flat parameter
# in forward and AGAIN in backward; SHARD_GRAD_OP (ZeRO-2) holds the forward
# all-gather through backward, halving parameter traffic for one extra
# unsharded bf16 copy of the model (~3GB for 1.5B). On a PCIe-only box with a
# 1.5B model that is the right trade. Single-rank runs keep full_shard so
# nothing about the existing GPU=<n> arm changes.
if (( NPROC > 1 )); then
  FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-shard_grad_op}"
else
  FSDP_SHARDING_STRATEGY="${FSDP_SHARDING_STRATEGY:-full_shard}"
fi
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-512}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"

NPROC_TAG=""
if (( NPROC > 1 )); then NPROC_TAG="_${NPROC}gpu"; fi
EXP="${EXP:-sftkd_plain_fullkl_b${KL_BETA}_r1_32b_b${TRAIN_BATCH_SIZE}_lr${LR}_len${MAX_LENGTH}_seed${TRAIN_SEED}${NPROC_TAG}}"
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
# In-training six-benchmark eval reads the eval_sft_6bench layout (present);
# the legacy dapo layout it falls back to does not exist in this checkout.
export EVAL_DATA_ROOT="${EVAL_DATA_ROOT:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/eval_sft_6bench}"
# The harness resolves its own location relative to the caller's cwd, which is
# wrong when the trainer launches it -- pin the SFPO root explicitly.
export EVAL_CODE_ROOT="${EVAL_CODE_ROOT:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO}"
# In-training eval shares the GPU with the suspended training state (~29GB
# held), so its vLLM budget must fit alongside -- 0.6 (~58GB) fits, the
# default 0.85 does not. The post-training eval below runs after the trainer
# exits and keeps 0.85 on the free GPU.
export SFT_VLLM_EVAL_GPU_UTIL="${SFT_VLLM_EVAL_GPU_UTIL:-0.6}"
# Cadence evals generate ONE sampled response per prompt (temp 0.7, top_p 1.0,
# no greedy pass); the trainer reads back the matching sft_sampled_1_1seed.json
# via eval_sample_n/eval_seed_count/eval_skip_greedy.
export EVAL_N="${EVAL_N:-1}"
export EVAL_SEEDS="${EVAL_SEEDS:-0}"
export SFT_VLLM_EVAL_SKIP_GREEDY="${SFT_VLLM_EVAL_SKIP_GREEDY:-1}"

for f in "$TRAIN" "$VAL" "$KL_REF_MODEL"; do
  [[ -f "$f" || -d "$f" ]] || { echo "PREFLIGHT FAIL: missing $f" >&2; exit 2; }
done

if [[ "${SKIP_EVAL:-0}" != "1" ]]; then
  for f in "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD"; do
    [[ -f "$f" ]] || { echo "PREFLIGHT FAIL: missing six-benchmark file $f (set SKIP_EVAL=1 to train without evaluation)" >&2; exit 2; }
  done
fi

"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.prompt_key=problem \
    data.response_key=solution \
    data.train_batch_size="$TRAIN_BATCH_SIZE" \
    data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    data.max_length="$MAX_LENGTH" \
    data.truncation=error \
    data.normalize_by_sequence=True \
    data.kl_beta="$KL_BETA" \
    data.kl_ref_model="$KL_REF_MODEL" \
    data.kl_full=True \
    ++data.kl_full_chunk_tokens="${KL_CHUNK_TOKENS:-2048}" \
    data.use_dynamic_bsz=True \
    data.max_token_len_per_gpu="$MAX_TOKEN_LEN_PER_GPU" \
    ++data.val_micro_batch_size_per_gpu="${VAL_MICRO_BATCH_SIZE:-2}" \
    model.partial_pretrain="$MODEL" \
    model.enable_gradient_checkpointing="${GRAD_CKPT:-True}" \
    ++model.fsdp_config.sharding_strategy="$FSDP_SHARDING_STRATEGY" \
    model.use_liger=True \
    optim.lr="$LR" \
    optim.betas="[0.9,0.999]" \
    optim.weight_decay=0.01 \
    optim.warmup_steps_ratio=0.0 \
    optim.clip_grad=1.0 \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.default_hdfs_dir=null \
    trainer.logger=['console','wandb'] \
    trainer.total_epochs="${TOTAL_EPOCHS:-25}" \
    trainer.total_training_steps="${MAX_STEPS:-300}" \
    ++trainer.test_freq="${TEST_FREQ:-25}" \
    ++trainer.save_freq="${SAVE_FREQ:-0}" \
    ++trainer.benchmark_eval_freq="${BENCHMARK_EVAL_FREQ:-50}" \
    ++trainer.eval_sample_n=1 \
    ++trainer.eval_seed_count=1 \
    ++trainer.eval_skip_greedy=1 \
    trainer.seed="$TRAIN_SEED" \
    2>&1 | tee "$RUN_DIR/train.log"

if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  CODE_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
  CKPT_ABS="$(ls -d "$RUN_DIR"/global_step_* | sort -t_ -k3 -n | tail -1)"
  CKPT_ABS="$(cd "$CKPT_ABS" && pwd)"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "$PYTHON_BIN" "$CODE_ROOT/tools/kd_sft/evaluate_greedy.py" \
    --checkpoint-dir "$CKPT_ABS" --base-model "$MODEL" \
    --data-files "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD" \
    --seed "${EVAL_SEED:-0}" --n "${EVAL_FINAL_N:-4}" --temperature 0.7 --top-p 1.0 \
    --max-tokens "${EVAL_MAX_TOKENS:-3072}" --tp 1 --gpu-devices "$GPU" --gpu-memory-utilization 0.85 \
    --max-model-len "${EVAL_MAX_MODEL_LEN:-4096}" --max-num-seqs 256 --max-num-batched-tokens 65536 \
    --attention-backend "$VLLM_ATTENTION_BACKEND" \
    --log-wandb --wandb-project "$PROJECT" --wandb-run "$EXP-post-eval" \
    --output "$RUN_DIR/eval_pass4_6bench.json" 2>&1 | tee "$RUN_DIR/eval_pass4_6bench.log"
  rm -rf "$CKPT_ABS"
fi
