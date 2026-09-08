#!/usr/bin/env bash
# SFT baseline: plain 1-pass supervised finetuning of Qwen2.5-1.5B-Instruct on
# Bespoke-Stratos-17k (bespokelabs/Bespoke-Stratos-17k, DeepSeek-R1 traces;
# prepared by prep_bespoke_stratos_sft.py into train/val parquet with the
# shared system prompt folded into `prompt`). Paired with
# run_sft_gxpo_stratos_1p5b.sh, which is identical except each batch takes the
# GXPO 3-pass update (3 optimizer steps per batch instead of 1).
#
# Config: batch 96, micro-batch 1, max_length 16384, lr 1e-5, liger kernels,
# 200 steps (1.25 epochs). No mid-run benchmark evals or checkpoints; only
# the final global_step_200 HF checkpoint is kept (fsdp2 => no resumable
# training-state dir). Two unavoidable trainer behaviors, both disclosed:
# (a) cheap val-loss logging at the 3 epoch ends (val split required at
# startup; no generations, no checkpoints); (b) the trainer unconditionally
# runs its own repo-harness greedy eval once AT THE END (temp 0.0, 16k budget)
# and would delete the final checkpoint if that eval crashed -- it is wired
# to succeed so the checkpoint is retained for the real reporting below.
# Post-training pass@1 on all 6 benchmarks: temp 0.7, 1 response/prompt,
# 16k generation budget.
# Usage: GPU=0 ./run_sft_baseline_stratos_1p5b.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port), an invalid bootstrap address.
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

GPU="${GPU:?set GPU=0|1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LR=1e-5
PROJECT="${PROJECT:-gxpo-efficiency-final}"

MODEL="${MODEL:-/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
DATA_DIR="${DATA_DIR:-/office/dev_workspace/swapnil/data/Bespoke-Stratos-17k}"
TRAIN="${TRAIN:-$DATA_DIR/train.parquet}"
VAL="${VAL:-$DATA_DIR/val.parquet}"
# Symlink layout for the trainer's mandatory end-of-run harness eval
# (math500/test.parquet, aime2024/test.parquet, ...).
EVAL_ROOT="${EVAL_ROOT:-/office/dev_workspace/swapnil/data/eval_sft_6bench_terminal}"

EXP="sft_baseline_stratos1p5b_b96_e3_lr1e-5_seed${TRAIN_SEED}"
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
    data.train_batch_size=96 \
    data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE:-1}" \
    data.max_length=16384 \
    data.truncation=right \
    model.partial_pretrain="$MODEL" \
    model.enable_gradient_checkpointing=True \
    ++model.fsdp_config.strategy=fsdp2 \
    model.use_liger=True \
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
    ++trainer.test_freq=0 \
    ++trainer.save_freq=0 \
    ++trainer.greedy_eval_freq=0 \
    trainer.eval_kind=math \
    trainer.eval_benchmark_root="$EVAL_ROOT" \
    trainer.eval_greedy_max_new_tokens=16384 \
    +trainer.eval_greedy_max_examples=0 \
    +trainer.eval_greedy_prompt_max_length=2048 \
    trainer.eval_greedy_vllm_gpu_memory_utilization=0.6 \
    trainer.eval_greedy_vllm_max_num_batched_tokens=32768 \
    trainer.eval_greedy_vllm_max_num_seqs=16 \
    +trainer.eval_greedy_timeout_s=0 \
    trainer.seed=$TRAIN_SEED \
    2>&1 | tee "$RUN_DIR/train.log"

# Post-training pass@1 six-benchmark eval on the final checkpoint.
# SKIP_EVAL=1 to skip. temp 0.7, n=1, 16k generation budget with vLLM sized to
# match (max_model_len covers prompt headroom + 16k; 16 seqs keep 16k-KV in 96GB).
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  CKPT_ABS="$(ls -d "$RUN_DIR"/global_step_* | sort -t_ -k3 -n | tail -1)"
  CKPT_ABS="$(cd "$CKPT_ABS" && pwd)"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "${PYTHON_BIN:-python}" "$CODE_ROOT/tools/kd_sft/evaluate_greedy.py" \
    --checkpoint-dir "$CKPT_ABS" --base-model "$MODEL" \
    --data-files "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD" \
    --seed "${EVAL_SEED:-0}" --n 1 --temperature 0.7 --top-p 1.0 \
    --max-tokens 16384 --tp 1 --gpu-memory-utilization 0.85 \
    --max-model-len 20480 --max-num-seqs 16 --max-num-batched-tokens 32768 \
    --output "$RUN_DIR/eval_pass1_6bench.json" 2>&1 | tee "$RUN_DIR/eval_pass1_6bench.log"
fi
