#!/usr/bin/env bash
# SFT + reference-KL training on flat teacher responses (prompt + teacher_response):
# supervised CE plus kl_beta * D_KL(pi_theta || pi_ref) (verl/trainer/fsdp_sft_trainer.py),
# one plain optimizer step per batch. The KL term here is the plain RL-style
# anchor against a frozen reference policy -- not a distillation term: no
# teacher top-K cache is needed or used.
#
# Training data is the flat verified-correct teacher-response corpus; the
# deleted top-K cache (dapo_lighteval_topk16_*) is no longer referenced anywhere
# in this file.
#
# Evaluation: six-benchmark vLLM eval runs in-training after every 5 steps
# (trainer.benchmark_eval_freq=5, same eval_greedy/* wandb keys as the final
# post-eval) plus one post-training pass below.
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
export PYTHONPATH="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd):${PYTHONPATH:-}"
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASH_ATTN}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export VLLM_WORKER_MULTIPROC_METHOD=spawn
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-1}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-1}"

GPU="${GPU:?set GPU=0|1}"
export GPU
TRAIN_SEED="${TRAIN_SEED:-42}"
LR="${LR:-1e-6}"
PROJECT="${PROJECT:-gxpo-efficiency-final}"

# Uses final-gxpo-h200's venv specifically (this repo, final-gxpo, ships no
# .venv of its own). Override PYTHON_BIN to point elsewhere.
MODEL="${MODEL:-/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
# Flat verified-correct teacher responses (prompt + teacher_response[_ids]).
FLAT_TRAIN="${FLAT_TRAIN:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data/teacher_responses/qwen25_math7b_dapo_lighteval_train_n1_correct_only.parquet}"
TRAIN="${TRAIN:-$FLAT_TRAIN}"
VAL="${VAL:-$TRAIN}"

# Reference policy for the KL anchor. Defaults to the student's own starting
# weights (the standard RL KL reference); point at a frozen teacher to anchor
# to the teacher instead.
KL_REF_MODEL="${KL_REF_MODEL:-$MODEL}"
# KL anchor weight. Tune on a smoke run: too high stalls learning on the
# teacher traces, too low lets the policy drift off them.
KL_BETA="${KL_BETA:-0.1}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-4}"

EXP="${EXP:-sftkl_b${KL_BETA}_flat14k_b96_lr${LR}_seed${TRAIN_SEED}}"
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

"$PYTHON_BIN" -m torch.distributed.run --standalone --nnodes=1 --nproc_per_node=1 \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.prompt_key=prompt \
    data.response_key=teacher_response \
    data.train_batch_size=96 \
    data.micro_batch_size_per_gpu="$MICRO_BATCH_SIZE" \
    data.max_length="$MAX_LENGTH" \
    data.truncation=error \
    data.normalize_by_sequence=True \
    data.kl_beta="$KL_BETA" \
    data.kl_ref_model="$KL_REF_MODEL" \
    model.partial_pretrain="$MODEL" \
    model.enable_gradient_checkpointing=True \
    ++model.fsdp_config.strategy=fsdp2 \
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
    trainer.total_epochs=3 \
    trainer.total_training_steps="${MAX_STEPS:-200}" \
    ++trainer.test_freq=0 \
    ++trainer.save_freq=0 \
    ++trainer.benchmark_eval_freq=5 \
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
    --seed "${EVAL_SEED:-0}" --n 1 --temperature 0.7 --top-p 1.0 \
    --max-tokens "${EVAL_MAX_TOKENS:-3072}" --tp 1 --gpu-memory-utilization 0.85 \
    --max-model-len "${EVAL_MAX_MODEL_LEN:-4096}" --max-num-seqs 256 --max-num-batched-tokens 65536 \
    --attention-backend "$VLLM_ATTENTION_BACKEND" \
    --log-wandb --wandb-project "$PROJECT" --wandb-run "$EXP-post-eval" \
    --output "$RUN_DIR/eval_pass1_6bench.json" 2>&1 | tee "$RUN_DIR/eval_pass1_6bench.log"
fi
