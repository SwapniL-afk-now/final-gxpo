#!/usr/bin/env bash
# Four-GPU FSDP verifier-RL comparison. Set METHOD=grpo or METHOD=gxpo.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PROJECT_ROOT="$(cd "$CODE_ROOT/.." && pwd)"
cd "$CODE_ROOT"
if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a; source "$PROJECT_ROOT/.env"; set +a
fi

export PYTHONPATH="$CODE_ROOT:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_PROJECT="${WANDB_PROJECT:-code-distillation}"
export RAY_ADDRESS=local

METHOD="${METHOD:-grpo}"
[[ "$METHOD" == grpo || "$METHOD" == gxpo ]] || { echo 'METHOD must be grpo or gxpo' >&2; exit 2; }
MODEL="${MODEL_QWEN25_CODER_1P5B:-Qwen/Qwen2.5-Coder-1.5B-Instruct}"
DATA_ROOT="${CODE_DISTILL_ROOT:-/workspace/jepa-grpo-cache/data/code_distill}"
EVAL_ROOT="${CODE_EVAL_ROOT:-/workspace/jepa-grpo-cache/eval_data/code_distill}"
RUN_DIR="${CODE_DISTILL_RUN_ROOT:-$CODE_ROOT/runs/code_distill_${METHOD}_rl}"
mkdir -p "$RUN_DIR"

if [[ "${1:-}" == "--dry-run" ]]; then
  printf 'method=%s model=%s train=%s/teacher_prompts_train.parquet\nGPUs=0,1,2,3 FSDP_SIZE=4 global_batch=256 ppo_minibatch=64\n' "$METHOD" "$MODEL" "$DATA_ROOT"
  exit 0
fi

for required in "$MODEL" "$DATA_ROOT/teacher_prompts_train.parquet" \
                "$EVAL_ROOT/humanevalplus.parquet" "$EVAL_ROOT/mbppplus.parquet" "$EVAL_ROOT/livecodebench.parquet"; do
  [[ -e "$required" ]] || { echo "Missing required study asset: $required" >&2; exit 2; }
done

GXPO_ARGS=()
if [[ "$METHOD" == gxpo ]]; then
  GXPO_ARGS=(+actor_rollout_ref.actor.use_gxpo=True +actor_rollout_ref.actor.gxpo_k=5
    +actor_rollout_ref.actor.gxpo_alpha=0.3
    +actor_rollout_ref.actor.gxpo_optimizer_state_mode=transactional
    +actor_rollout_ref.actor.gxpo_trigger_signal=entropy
    +actor_rollout_ref.actor.gxpo_zscore_w=30 +actor_rollout_ref.actor.gxpo_tau=1.5
    +actor_rollout_ref.actor.gxpo_trigger_patience=2
    +actor_rollout_ref.actor.gxpo_reset_entropy_after_warmup=False)
fi

python -u -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  data.train_files="['$DATA_ROOT/teacher_prompts_train.parquet']" \
  data.val_files="['$EVAL_ROOT/humanevalplus.parquet','$EVAL_ROOT/mbppplus.parquet','$EVAL_ROOT/livecodebench.parquet']" \
  data.train_batch_size=256 data.val_batch_size=64 data.max_prompt_length=1024 data.max_response_length=3072 \
  data.filter_overlong_prompts=True +data.seed="${TRAIN_SEED:-42}" \
  actor_rollout_ref.model.path="$MODEL" actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \

  actor_rollout_ref.actor.optim.lr=1e-6 +actor_rollout_ref.actor.optim.name=adamw \
  actor_rollout_ref.actor.ppo_mini_batch_size=64 actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=24576 actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.grad_clip=1.0 actor_rollout_ref.actor.entropy_coeff=0.0 \
  actor_rollout_ref.actor.use_kl_loss=False actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5 actor_rollout_ref.rollout.max_num_batched_tokens=65536 \
  actor_rollout_ref.rollout.max_num_seqs=512 actor_rollout_ref.rollout.enforce_eager=False \
  actor_rollout_ref.rollout.attention_backend=FLASHINFER actor_rollout_ref.rollout.n=8 \
  actor_rollout_ref.rollout.temperature=0.7 actor_rollout_ref.rollout.top_p=1.0 \
  actor_rollout_ref.rollout.val_kwargs.n=1 actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.ref.fsdp_config.param_offload=True algorithm.kl_ctrl.kl_coef=0.0 \
  trainer.critic_warmup=0 trainer.logger="['console','wandb']" trainer.project_name="$WANDB_PROJECT" \
  trainer.experiment_name="code_distill_${METHOD}_rl" trainer.default_local_dir="$RUN_DIR" \
  trainer.n_gpus_per_node=4 trainer.nnodes=1 trainer.save_freq=50 trainer.test_freq=25 \
  +trainer.val_before_train=True +trainer.max_steps="${MAX_STEPS:-400}" trainer.total_epochs=100 \
  +actor_rollout_ref.actor.filter_mixed_responses=True "${GXPO_ARGS[@]}" \
  2>&1 | tee -a "$RUN_DIR/train.log"
