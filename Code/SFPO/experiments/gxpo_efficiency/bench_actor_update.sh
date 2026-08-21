#!/usr/bin/env bash
set -euo pipefail

# Short, checkpoint-free actor-update benchmark.  It uses the same 1.5B
# Qwen/GXPO batch shape as the efficiency run and changes one performance
# switch at a time.  Outputs stay under results/perf_bench and never touch a
# training run or WandB.
VARIANT="${1:?usage: bench_actor_update.sh baseline|fused|no_ckpt|liger|cap64k_liger}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"

MODEL_PATH="$ROOT/models/Qwen2.5-Math-1.5B-Instruct"
TRAIN_FILES="['$ROOT/data/dapo_math/train.parquet','$ROOT/data/lighteval-math/train.parquet']"
OUT="$ROOT/results/perf_bench/qwen1p5b/$VARIANT"
[[ -d "$MODEL_PATH" ]] || { echo "Missing model: $MODEL_PATH" >&2; exit 2; }
[[ ! -e "$OUT/train_metrics.jsonl" ]] || { echo "Refusing to overwrite $OUT" >&2; exit 2; }
mkdir -p "$OUT" "$OUT/vllm_cache" "$OUT/flashinfer_autotune_cache"

export PYTHONPATH="$ROOT/.runtime_deps:/workspace/.gxpo_pydeps:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HOME="$ROOT/.hf_home"
export HF_HUB_CACHE="$HF_HOME/hub"
export VLLM_CACHE_ROOT="$OUT/vllm_cache"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$OUT/flashinfer_autotune_cache"
export WANDB_MODE=disabled WANDB_DISABLED=true WANDB_SILENT=true
export RAY_ADDRESS=local TOKENIZERS_PARALLELISM=false CUDA_DEVICE_MAX_CONNECTIONS=1
export VLLM_SLEEP_LEVEL=2

USE_LIGER=False
OPTIM_FUSED=False
ENABLE_GRADIENT_CHECKPOINTING=True
USE_TORCH_COMPILE=True
PPO_MAX_TOKEN_LEN_PER_GPU=24576
case "$VARIANT" in
  baseline) ;;
  fused) OPTIM_FUSED=True ;;
  no_ckpt) ENABLE_GRADIENT_CHECKPOINTING=False ;;
  liger|liger_2) USE_LIGER=True ;;
  no_compile) USE_TORCH_COMPILE=False ;;
  cap32k_liger) USE_LIGER=True; PPO_MAX_TOKEN_LEN_PER_GPU=32768 ;;
  cap64k_liger) USE_LIGER=True; PPO_MAX_TOKEN_LEN_PER_GPU=65536 ;;
  *) echo "Unknown variant: $VARIANT" >&2; exit 2 ;;
esac

ARGS=(
  algorithm.adv_estimator=grpo
  +algorithm.norm_adv_by_std_in_grpo=False
  +algorithm.use_kl_in_reward=False
  data.train_files="$TRAIN_FILES"
  data.val_files="[]"
  data.train_batch_size=256
  data.max_prompt_length=1024
  data.max_response_length=3072
  data.filter_overlong_prompts=True
  data.truncation=error
  +data.seed=3407
  actor_rollout_ref.model.path="$MODEL_PATH"
  actor_rollout_ref.model.use_remove_padding=True
  actor_rollout_ref.model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING"
  actor_rollout_ref.model.attn_implementation=flash_attention_3
  +actor_rollout_ref.model.use_liger="$USE_LIGER"
  actor_rollout_ref.actor.optim.lr=1e-6
  +actor_rollout_ref.actor.optim.name=adamw
  +actor_rollout_ref.actor.optim.fused="$OPTIM_FUSED"
  actor_rollout_ref.actor.use_torch_compile="$USE_TORCH_COMPILE"
  +actor_rollout_ref.actor.data_loader_seed=3407
  actor_rollout_ref.actor.ppo_mini_batch_size=64
  actor_rollout_ref.actor.use_dynamic_bsz=True
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="$PPO_MAX_TOKEN_LEN_PER_GPU"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.use_kl_loss=False
  actor_rollout_ref.actor.kl_loss_coef=0.0
  actor_rollout_ref.actor.fsdp_config.fsdp_size=1
  actor_rollout_ref.actor.fsdp_config.param_offload=False
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.rollout.tensor_model_parallel_size=1
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.gpu_memory_utilization=0.5
  actor_rollout_ref.rollout.max_num_batched_tokens=98304
  actor_rollout_ref.rollout.max_num_seqs=2048
  actor_rollout_ref.rollout.enable_chunked_prefill=True
  actor_rollout_ref.rollout.attention_backend=FLASHINFER
  actor_rollout_ref.rollout.n=8
  actor_rollout_ref.rollout.temperature=1.0
  actor_rollout_ref.rollout.top_p=1.0
  actor_rollout_ref.rollout.val_kwargs.n=1
  actor_rollout_ref.rollout.val_kwargs.do_sample=False
  actor_rollout_ref.rollout.val_kwargs.temperature=0
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8
  actor_rollout_ref.ref.fsdp_config.param_offload=True
  reward_model.reward_manager=naive
  algorithm.kl_ctrl.kl_coef=0.0
  trainer.logger="['console']"
  trainer.project_name=gxpo-perf-bench
  trainer.experiment_name="qwen1p5b_${VARIANT}"
  trainer.default_local_dir="$OUT"
  trainer.n_gpus_per_node=1
  trainer.nnodes=1
  trainer.save_freq=-1
  trainer.test_freq=-1
  trainer.resume_mode=disable
  +trainer.val_before_train=False
  +trainer.max_steps="${BENCH_STEPS:-1}"
  trainer.total_training_steps="${BENCH_STEPS:-1}"
  trainer.total_epochs=100
  +actor_rollout_ref.actor.use_gxpo=True
  +actor_rollout_ref.actor.gxpo_k=10
  +actor_rollout_ref.actor.gxpo_alpha=0.5
  +actor_rollout_ref.actor.gxpo_delta=1e-8
  +actor_rollout_ref.actor.gxpo_tau=3.0
  +actor_rollout_ref.actor.gxpo_zscore_w=30
  +actor_rollout_ref.actor.gxpo_trigger_signal=entropy
  +actor_rollout_ref.actor.gxpo_trigger_patience=3
  +actor_rollout_ref.actor.gxpo_trigger_granularity=outer
  +actor_rollout_ref.actor.gxpo_warmup_steps=50
  +actor_rollout_ref.actor.gxpo_reset_entropy_after_warmup=True
  +actor_rollout_ref.actor.gxpo_omega=0.1
  +actor_rollout_ref.actor.gxpo_shutoff_mode=trajectory_aware
  +actor_rollout_ref.actor.gxpo_recompute_old_log_probs=False
  +actor_rollout_ref.actor.gxpo_diag_freq=10
)

echo "[actor benchmark] variant=$VARIANT liger=$USE_LIGER fused_adamw=$OPTIM_FUSED checkpointing=$ENABLE_GRADIENT_CHECKPOINTING ppo_max_tokens=$PPO_MAX_TOKEN_LEN_PER_GPU"
/venv/main/bin/python -u -m verl.trainer.main_ppo "${ARGS[@]}" 2>&1 | tee "$OUT/train.log"
[[ -s "$OUT/train_metrics.jsonl" ]] || { echo "Missing metrics: $OUT/train_metrics.jsonl" >&2; exit 3; }
