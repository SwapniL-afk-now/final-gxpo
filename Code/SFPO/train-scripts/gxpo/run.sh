#!/usr/bin/env bash
# GXPO paper protocol runner (Section 3.1): one script, parameterized by env vars.
#
#   METHOD=gxpo K=5 MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct MODEL_TAG=qwen1.5b ./run.sh
#
# Required: MODEL_PATH, MODEL_TAG, METHOD (grpo|sfpo|gxpo)
# Optional: K (default 5), ALPHA (0.5), TAU (0.5 gxpo / 2.0 sfpo), GPU_NUM (4),
#           LORA_RANK (128; 0 = full fine-tune), DATA_ROOT (~/data/gxpo),
#           MAX_STEPS (300), SEED (1), RUNS_ROOT (./runs), PROJECT (gxpo-paper)
set -euo pipefail

METHOD="${METHOD:?set METHOD=grpo|sfpo|gxpo}"
MODEL_PATH="${MODEL_PATH:?set MODEL_PATH}"
MODEL_TAG="${MODEL_TAG:?set MODEL_TAG}"
K="${K:-5}"
ALPHA="${ALPHA:-0.5}"
GPU_NUM="${GPU_NUM:-4}"
LORA_RANK="${LORA_RANK:-128}"
LORA_ALPHA="${LORA_ALPHA:-256}"
DATA_ROOT="${DATA_ROOT:-$HOME/data/gxpo}"
MAX_STEPS="${MAX_STEPS:-300}"
SEED="${SEED:-1}"
RUNS_ROOT="${RUNS_ROOT:-./runs}"
PROJECT="${PROJECT:-gxpo-paper}"

case "$METHOD" in
  grpo) TAU="${TAU:-0}"; EXP="${MODEL_TAG}_grpo_seed${SEED}" ;;
  sfpo) TAU="${TAU:-2.0}"; EXP="${MODEL_TAG}_sfpo_k${K}_seed${SEED}" ;;
  gxpo) TAU="${TAU:-0.5}"; EXP="${MODEL_TAG}_gxpo_k${K}_a${ALPHA}_tau${TAU}_seed${SEED}" ;;
  *) echo "unknown METHOD=$METHOD"; exit 1 ;;
esac

RUN_DIR="${RUNS_ROOT}/${EXP}"
mkdir -p "$RUN_DIR"

METHOD_FLAGS=()
case "$METHOD" in
  sfpo)
    # paper protocol: SFPO alpha_0 = 0.5, tau = 2.0
    METHOD_FLAGS+=(
      +actor_rollout_ref.actor.use_sfpo=True
      +actor_rollout_ref.actor.sfpo_inner_steps="$K"
      +actor_rollout_ref.actor.sfpo_step_size="$ALPHA"
      +actor_rollout_ref.actor.zscore_w=30
      +actor_rollout_ref.actor.zscore_threshold="$TAU"
    ) ;;
  gxpo)
    # paper protocol: alpha_0 = 0.5, delta = 1e-8, tau = 0.5, trajectory-aware shutoff
    METHOD_FLAGS+=(
      +actor_rollout_ref.actor.use_gxpo=True
      +actor_rollout_ref.actor.gxpo_k="$K"
      +actor_rollout_ref.actor.gxpo_alpha="$ALPHA"
      +actor_rollout_ref.actor.gxpo_delta=1e-8
      +actor_rollout_ref.actor.gxpo_tau="$TAU"
      +actor_rollout_ref.actor.gxpo_omega=0.1
      +actor_rollout_ref.actor.gxpo_shutoff_mode=trajectory_aware
      +actor_rollout_ref.actor.gxpo_recompute_old_log_probs=False
      +actor_rollout_ref.actor.gxpo_diag_freq=10
    ) ;;
esac

LORA_FLAGS=()
if [ "$LORA_RANK" -gt 0 ]; then
  LORA_FLAGS+=(
    +actor_rollout_ref.model.lora_rank="$LORA_RANK"
    +actor_rollout_ref.model.lora_alpha="$LORA_ALPHA"
  )
fi

export VLLM_ATTENTION_BACKEND=XFORMERS

python -u -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files="['$DATA_ROOT/train/train.parquet']" \
    data.val_files="['$DATA_ROOT/eval/math500/test.parquet']" \
    data.train_batch_size=128 \
    data.val_batch_size=8 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    actor_rollout_ref.model.path="$MODEL_PATH" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-7 \
    actor_rollout_ref.actor.ppo_mini_batch_size=128 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.actor.clip_ratio=0.2 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=5 \
    actor_rollout_ref.rollout.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.000 \
    trainer.critic_warmup=0 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="$PROJECT" \
    trainer.experiment_name="$EXP" \
    trainer.default_local_dir="$RUN_DIR" \
    trainer.n_gpus_per_node="$GPU_NUM" \
    trainer.nnodes=1 \
    trainer.save_freq=5 \
    trainer.test_freq=5 \
    +trainer.val_before_train=True \
    +trainer.max_steps="$MAX_STEPS" \
    trainer.total_epochs=100 \
    "${METHOD_FLAGS[@]}" \
    "${LORA_FLAGS[@]}" \
    | tee "$RUN_DIR/train.log"
