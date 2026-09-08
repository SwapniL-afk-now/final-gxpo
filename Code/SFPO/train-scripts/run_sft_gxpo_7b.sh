#!/usr/bin/env bash
# SFT + GXPO-style update (7B): identical to run_sft_baseline_7b.sh except each step is the GXPO
# 3-pass extrapolated update (probe g0, probe g1, reposition, slow correction) applied to a
# plain cross-entropy objective -- no importance ratio, no advantages. K/alpha/tau match the
# 1.5B SFT+GXPO arm (run_sft_gxpo.sh) so the two sizes are directly comparable.
#
# Costs ~3x the baseline's wall time (3 backward passes per step).
# 7B needs FSDP across 2 GPUs (AdamW states alone are ~3x params in fp32); 1-GPU like the
# 1.5B pair cannot fit it. lr 1e-5, batch 32, 500 steps (3 epochs of data, capped at 500),
# validate every 10 steps, seed 42.
# SFT dense cross-entropy grad-norms are ~2.5x noisier vs their EMA than the RL arm's
# advantage-grad norms, so tau stays at 5.0 (not the RL 3.0) with a 3-step warmup.
# Usage: ./run_sft_gxpo_7b.sh
#   GPU_IDS=0,1 NPROC=2 K=5 ALPHA=0.5 ./run_sft_gxpo_7b.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port), an invalid bootstrap address.
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent jobs of GPUs ("Total available GPUs 0")

GPU_IDS="${GPU_IDS:-0,1}"
NPROC="${NPROC:-2}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LR="${LR:-1e-5}"
PROJECT="${WANDB_PROJECT:-rebuttul}"
K="${K:-5}"
ALPHA="${ALPHA:-0.5}"
GXPO_TAU="${GXPO_TAU:-5.0}"
GXPO_WARMUP="${GXPO_WARMUP:-3}"

MODEL="${MODEL:-/workspace/models/Qwen2.5-7B-Instruct}"
TRAIN="${TRAIN:-/workspace/jepa-grpo-cache/data/math_l35_sft/train.parquet}"
VAL="${VAL:-/workspace/jepa-grpo-cache/data/math_l35_sft/test.parquet}"

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

EXP="sft_gxpo_7b_k${K}_a${ALPHA}_tau${GXPO_TAU}_mathl35_seed${TRAIN_SEED}${OPT_STATE_TAG}"
RUN_DIR="./runs/${EXP}"
mkdir -p "$RUN_DIR"

export CUDA_VISIBLE_DEVICES="$GPU_IDS"
export WANDB_PROJECT="$PROJECT"
export WANDB_MODE=online
export WANDB_DIR="$RUN_DIR"

torchrun --standalone --nnodes=1 --nproc_per_node="$NPROC" \
    -m verl.trainer.fsdp_sft_trainer \
    data.train_files="$TRAIN" \
    data.val_files="$VAL" \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=32 \
    data.micro_batch_size_per_gpu=2 \
    data.max_length=2048 \
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
    trainer.total_training_steps=500 \
    +trainer.test_freq=10 \
    +trainer.save_freq=100 \
    +trainer.val_max_batches=50 \
    trainer.seed=$TRAIN_SEED \
    2>&1 | tee "$RUN_DIR/train.log"
