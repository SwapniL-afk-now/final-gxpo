#!/usr/bin/env bash
# SFT + GXPO-style update: identical to run_sft_baseline.sh except each step is the GXPO
# 3-pass extrapolated update (probe g0, probe g1, reposition, slow correction) applied to a
# plain cross-entropy objective -- no importance ratio, no advantages.
# K=3, alpha=0.5, tau=5.0, warmup=3.
#
# Costs ~3x the baseline's wall time (3 backward passes per step).
# lr 1e-5, batch 64, 200 steps, no validation dataset or periodic evaluation.
# After training, greedily scores global_step_200 on all 6 benchmarks.
# Usage: GPU=1 ./run_sft_gxpo.sh
set -euo pipefail

# /etc/environment ships RAY_ADDRESS="127.0.0.1" (no port), an invalid bootstrap address.
export RAY_ADDRESS=local   # force an isolated Ray cluster per job -- unaddressed ray.init() auto-attaches to any existing local cluster (via /tmp/ray/session_latest), starving concurrent GPU0/GPU1 jobs of GPUs ("Total available GPUs 0")

GPU="${GPU:?set GPU=0|1}"
TRAIN_SEED="${TRAIN_SEED:-42}"
LR=1e-5
PROJECT="${PROJECT:-gxpo-efficiency-final}"
K=3
ALPHA="${ALPHA:-0.5}"
# SFT dense cross-entropy grad-norms are ~2.5x noisier vs their EMA than the RL arm's
# advantage-grad norms. On clean baseline SFT the natural post-warmup |z| tops out at ~4.9,
# so RL's tau=2.0 false-trips ~23x over 500 steps; tau=5.0 gives 0 false trips while still
# catching a genuine >5-sigma divergence. (data analysis over the seed42 SFT grad-norm logs.)
GXPO_TAU=5.0
# The EMA gate starts cold (mu=1); step-0 z~30 for norms~30 would instantly, permanently
# shut off extrapolation. The EMA tracks within ~2 steps, so suppress the trigger for 3.
GXPO_WARMUP=3

MODEL="${MODEL:-/office/shared_cache/.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct/snapshots/989aa7980e4cf806f80c7fef2b1adb7bc71aa306}"
DATA_DIR="${DATA_DIR:-/office/dev_workspace/swapnil/data/DAPO-MATH-17k-oss-reasoning}"
TRAIN="${TRAIN:-/office/dev_workspace/swapnil/data/DAPO-MATH-17k-oss-reasoning/math-oss-low-17k.filtered_resp3072.jsonl}"

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

EXP="sft_gxpo_k${K}_a${ALPHA}_tau${GXPO_TAU}_dapoossmed_seed${TRAIN_SEED}${OPT_STATE_TAG}"
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
    data.val_files=null \
    data.prompt_key=prompt \
    data.response_key=response \
    data.train_batch_size=64 \
    data.micro_batch_size_per_gpu="${SFT_MICRO_BATCH_SIZE:-2}" \
    data.max_length=3072 \
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
    +trainer.test_freq=0 \
    +trainer.save_freq=0 \
    +trainer.benchmark_eval_freq=0 \
    trainer.seed=$TRAIN_SEED \
    2>&1 | tee "$RUN_DIR/train.log"

# Post-training greedy six-benchmark eval at step 200. SKIP_EVAL=1 to skip.
if [[ "${SKIP_EVAL:-0}" != 1 ]]; then
  CODE_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
  CKPT_ABS="$(cd "$RUN_DIR/global_step_200" && pwd)"
  CUDA_VISIBLE_DEVICES="$GPU" PYTHONPATH="$CODE_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
    "${PYTHON_BIN:-python}" "$CODE_ROOT/tools/kd_sft/evaluate_greedy.py" \
    --checkpoint-dir "$CKPT_ABS" --base-model "$MODEL" \
    --data-files "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD" \
    --seed "${EVAL_SEED:-0}" --n 1 --temperature 0 --top-p 1.0 \
    --max-tokens 3072 --tp 1 --gpu-memory-utilization 0.85 \
    --output "$RUN_DIR/eval_greedy_6bench.json" 2>&1 | tee "$RUN_DIR/eval_greedy_6bench.log"
fi
