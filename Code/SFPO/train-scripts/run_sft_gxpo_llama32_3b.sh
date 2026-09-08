#!/usr/bin/env bash
# SFT + GXPO-style update on Llama-3.2-3B-Instruct: identical to
# run_sft_baseline_llama32_3b.sh except each step is the GXPO 3-pass
# extrapolated update (probe g0, probe g1, reposition, slow correction) applied
# to a plain cross-entropy objective -- no importance ratio, no advantages.
# K=3, alpha=0.1, tau=5.0, warmup=3.
#
# Costs ~3x the baseline's wall time (3 backward passes per step).
# lr 1e-5, batch 64, 200 steps, validate every 10 steps, seed 42.
# Trains, then scores the final checkpoint on all 6 benchmarks with the same
# vLLM harness as the KD/RL arms (eval_sft_6bench.sh).
#
# Memory (3.21B params, fp32 master): baseline's ~51.5GB static PLUS GXPO-only
# theta0/g0/g1 buffers (3x12.9=38.6GB) PLUS the transactional optimizer snapshot
# (moments clone, 25.7GB, alive through the whole step) ≈ 121GB peak on 1 GPU --
# guaranteed OOM on an 80GB card. Hence 2-way FSDP FULL_SHARD (GPU_IDS=0,1):
# ~61GB/rank peak, fits with headroom. Global batch stays 64 (32/rank), so the
# math is unchanged. Run the two 3B arms SEQUENTIALLY: this job takes both GPUs.
# Usage: GPU_IDS=0,1 ./run_sft_gxpo_llama32_3b.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port), an invalid bootstrap address.
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

GPU_IDS="${GPU_IDS:-0,1}"
NPROC="$(awk -F, '{print NF}' <<<"$GPU_IDS")"
TRAIN_SEED="${TRAIN_SEED:-42}"
# Fragmentation guard for the repeated large alloc/free cycles (optimizer
# snapshots, FSDP all-gather buffers). Safe here: no in-process vLLM.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
LR=1e-5
PROJECT=rebuttul
K=3
ALPHA=0.1
# Calibrated on Qwen2.5-1.5B SFT grad-norm logs (dense CE grad-norms are ~2.5x
# noisier vs their EMA than the RL arm's; tau=5.0 gave 0 false trips over 500
# steps there). Llama-3.2-3B grad-norm scale may differ -- check the new run's
# train/gxpo_trigger_z logs and adjust via GXPO_TAU if the gate never trips or
# trips constantly.
GXPO_TAU="${GXPO_TAU:-5.0}"
# The EMA gate starts cold (mu=1); step-0 z~30 for norms~30 would instantly, permanently
# shut off extrapolation. The EMA tracks within ~2 steps, so suppress the trigger for 3.
GXPO_WARMUP=3

MODEL="${MODEL:-${MODEL_LLAMA32_3B:-/workspace/models/Llama-3.2-3B-Instruct}}"
DATA_DIR="${DATA_DIR:-/office/dev_workspace/swapnil/data/DAPO-MATH-17k-oss-reasoning}"
TRAIN="${TRAIN:-$DATA_DIR/math-oss-medium-17k.jsonl}"
VAL="${VAL:-$DATA_DIR/math-oss-medium-17k.jsonl}"

# GXPO optimizer state across the two probe steps. Parameters are repositioned to
# theta_tilde in both modes; only the AdamW state the slow correction starts from differs:
#   transactional            -- snapshot the AdamW state before probe 1 and roll back to it
#                               after repositioning, so the slow correction is taken from
#                               the moments the batch started with (probes stay probes).
#   transactional_fast_state -- no refresh: the probe steps' moments and step counter are
#                               kept and the slow correction is taken from them, so AdamW's
#                               step counter advances 3x per batch instead of 1x.
# The mode is tagged into the run name so the two arms never share a run dir or wandb run;
# transactional is left untagged because it is the established baseline.
GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
case "$GXPO_OPTIMIZER_STATE_MODE" in
  transactional)            OPT_STATE_TAG="" ;;
  transactional_fast_state) OPT_STATE_TAG="_optkeep" ;;
  *) echo "PREFLIGHT FAIL: GXPO_OPTIMIZER_STATE_MODE must be transactional or transactional_fast_state, got '$GXPO_OPTIMIZER_STATE_MODE'" >&2; exit 2 ;;
esac

EXP="sft_gxpo_k${K}_a${ALPHA}_tau${GXPO_TAU}_llama32_3b_dapoossmed_seed${TRAIN_SEED}${OPT_STATE_TAG}"
RUN_DIR="${RUN_DIR:-./runs/${EXP}}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online
export WANDB_DIR="$RUN_DIR"
# In-training vLLM eval is OFF (benchmark_eval_freq=0): the trainer hard-raises
# for world size != 1 (fsdp_sft_trainer.py fit()), so periodic eval is
# impossible on this 2-GPU job. Post-training eval below covers the comparison.
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
    +optim.use_gxpo=True \
    +optim.gxpo_k=$K \
    +optim.gxpo_alpha=$ALPHA \
    +optim.gxpo_delta=1e-8 \
    +optim.gxpo_tau=$GXPO_TAU \
    +optim.gxpo_warmup=$GXPO_WARMUP \
    +optim.gxpo_optimizer_state_mode="$GXPO_OPTIMIZER_STATE_MODE" \
    +optim.gxpo_omega=0.1 \
    +optim.gxpo_shutoff_mode=trajectory_aware \
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
# baseline arm. Single-GPU eval on the first id. SKIP_EVAL=1 to skip.
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  CKPT_ABS="$(cd "$RUN_DIR/global_step_200" && pwd)"
  GPU="${GPU_IDS%%,*}" ./eval_sft_6bench.sh "$CKPT_ABS" 2>&1 | tee "$RUN_DIR/eval_6bench.log"
fi
