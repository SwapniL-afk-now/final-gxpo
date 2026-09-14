#!/usr/bin/env bash
#
# qwen25_3b_onpolicy_kd.sh
#
# Per-step ON-POLICY KD (Option B: separate teacher Ray group), without GXPO.
# Student: Qwen2.5-3B-Instruct.  Teacher: frozen Qwen2.5-Math-7B
# (scores the student's own just-generated rollouts every step via
# verl/trainer/ppo/teacher_kd.py + TeacherScoringWorker -- no offline cache).
#
# Loop per step: 256 prompts -> student generates 256 responses
# (rollout.n=1) -> teacher scores those exact sequences -> KD-only policy update -> next batch. Two 2 GPU-locked engines never coexist:
# student generates -> student sleeps -> teacher wakes+scores -> teacher
# sleeps -> actor update (both asleep) -> student wakes for next rollout.
#
# This is a standalone entrypoint (common.sh is untouched -- its METHOD
# case only supports grpo|sfpo|gxpo and resets METHOD_FLAGS, so a bespoke
# KD/teacher-group launch cannot be layered on top of it). It duplicates the
# subset of common.sh's environment preamble and the shared launcher environment it
# needs, then adds the +actor_rollout_ref.actor.use_kd=True... KD flags.
#
# Usage:
#   bash qwen25_3b_onpolicy_kd.sh            # launch
#   bash qwen25_3b_onpolicy_kd.sh --dry-run  # preflight only
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"          # Code/SFPO
DRY_RUN=0
[[ "${1:-}" == "--dry-run" ]] && DRY_RUN=1
cd "$REPO_ROOT"

# --------------------------------------------------------- env preamble -----
# This checkout (final-gxpo) has never had its own .venv provisioned on this
# host -- the working one lives in the sibling final-gxpo-h200 checkout (see
# common.sh). Root-caused 2026-09-14: this script only checked its own .venv
# and fell through preflight with a bare "no venv python" failure.
GXPO_PROJECT_ROOT="$(cd -- "$REPO_ROOT/../.." && pwd)"
for _gxpo_venv in "$GXPO_PROJECT_ROOT/.venv" "/office/dev_workspace/swapnil/final-gxpo-h200/.venv"; do
  if [[ -x "$_gxpo_venv/bin/python" ]]; then
    export VIRTUAL_ENV="$_gxpo_venv"
    export PATH="$_gxpo_venv/bin:$PATH"
    break
  fi
done
unset _gxpo_venv
export PYTHONPATH="$REPO_ROOT/.runtime_deps:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_home}"
if [[ ! -w "$HF_HOME/hub" ]]; then
  export HF_HOME="$REPO_ROOT/.hf_home"
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# P2P/SHM stay ON: disabling them forces socket transport, measured 2.2 vs
# 10.2 GB/s for a 2-rank all_gather on this PCIe box -- a ~4.5x tax on every
# FSDP collective and every TP all-reduce. common.sh's 2-rank hybrid runs never
# set them and don't deadlock. Set both to 1 to restore the old workaround.
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_SHM_DISABLE="${NCCL_SHM_DISABLE:-0}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_2}"
export ATTN_IMPL

# --------------------------------------------- CUDA/JIT toolchain (venv) ----
# FlashInfer JIT-compiles kernels at first use (vLLM warmup) and needs nvcc +
# Python.h; the venv ships its own CUDA toolkit wheel (nvidia-cu13) instead of
# a system /usr/local/cuda, so point the toolchain at it explicitly.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
  KD_SITE_PACKAGES="$(ls -d "$VIRTUAL_ENV"/lib/python3.*/site-packages 2>/dev/null | head -1)"
  if [[ -n "$KD_SITE_PACKAGES" && -x "$KD_SITE_PACKAGES/nvidia/cu13/bin/nvcc" ]]; then
    export CUDA_HOME="${CUDA_HOME:-$KD_SITE_PACKAGES/nvidia/cu13}"
    export PATH="$CUDA_HOME/bin:$PATH"
    for _lib in "$KD_SITE_PACKAGES/nvidia/cu13/lib" "$KD_SITE_PACKAGES/torch/lib"; do
      [[ -d "$_lib" ]] && export LD_LIBRARY_PATH="$_lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
    done
    unset _lib
  fi
fi
KD_PY_INCLUDE="${KD_PY_INCLUDE:-/office/shared_cache/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12}"
[[ -f "$KD_PY_INCLUDE/Python.h" ]] && export CPATH="$KD_PY_INCLUDE${CPATH:+:$CPATH}"
export FLASHINFER_DISABLE_VERSION_CHECK="${FLASHINFER_DISABLE_VERSION_CHECK:-1}"
# Disable outbound telemetry/usage-stats calls: on this network they can hang
# in a blocking recv() (TCP handshake succeeds, response never completes)
# instead of failing fast, stalling vLLM/HF engine init for many minutes.
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export VLLM_DO_NOT_TRACK="${VLLM_DO_NOT_TRACK:-1}"
export DO_NOT_TRACK="${DO_NOT_TRACK:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
# OOM fix (mirrors the 1.5B launchers): expandable_segments crashes vLLM init,
# so cap split sizes instead to keep big blocks intact for large allocations.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
# Level 2 discards student weights and KV cache. The actor will sync fresh
# weights when the rollout phase starts again after the update.
export VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-2}"
# Throughput profile for this 256-prompt / n=1 rollout. These are deliberately
# explicit here instead of inheriting the conservative verl defaults.
# enforce_eager is forbidden for this run: CUDA graphs must remain enabled.
export VLLM_ATTENTION_BACKEND="${VLLM_ATTENTION_BACKEND:-FLASHINFER}"
export VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
export VLLM_MAX_NUM_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-65536}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-256}"
export VLLM_ENABLE_CHUNKED_PREFILL="${VLLM_ENABLE_CHUNKED_PREFILL:-True}"
export VLLM_ENABLE_PREFIX_CACHING="${VLLM_ENABLE_PREFIX_CACHING:-False}"
export VLLM_ENFORCE_EAGER=False
export VLLM_FREE_CACHE_ENGINE=False

# ---------------------------------------------------- locked 2-GPU setup ----
# Each verl rank owns one independent TP=1 student rollout engine. vLLM's
# external_launcher reuses verl's already-initialized torch process group, but
# the engine's model-parallel group remains size 1. This is the normal hybrid
# engine layout: rank-local generation, then sleep, then the FSDP actor update.
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
GPU_IDS="${GPU_IDS:-0,1,2}"
export CUDA_VISIBLE_DEVICES="$GPU_IDS"
GPU_COUNT="${GPU_COUNT:-2}"

# ---------------------------------------------------------- run identity ----
MODEL_ALIAS="qwen25-3b"
MODEL_ID="${MODEL_QWEN25_3B:-Qwen/Qwen2.5-3B-Instruct}"
RUN_NAME="${RUN_NAME:-${MODEL_ALIAS}_onpolicy_kd_b256_topk${KD_TOPK:-1024}$( [[ "${USE_GXPO:-True}" == "True" ]] && echo "_gxpo_k${K:-3}_a${REPOSITION_ALPHA:-0.8}" )}"
PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
RUN_DIR="${RUN_DIR:-$GXPO_PROJECT_ROOT/results/gxpo_efficiency/$RUN_NAME}"

# ---------------------------------------------------------- data / task -----
# Benchmark data defaults mirror common.sh: dapo + lighteval train lists and the
# 6-benchmark val set (math500, aime24/25, amc23, minerva, olympiad). Override
# individual lists from the environment; the preflight below fails loudly if a
# referenced parquet is absent.
# The prepared dapo_math/lighteval-math/benchmark parquets live in the sibling
# speed-audit checkout, not this one (see qwen25_1p5b_opd2_nogxpo_control.sh).
DATA_ROOT="${GXPO_DATA_ROOT:-/office/dev_workspace/swapnil/gradient-extrapolation-based-policy-optimization-gxpo-speed-audit/Code/SFPO/data}"
TRAIN_FILES="${TRAIN_FILES:-['$DATA_ROOT/dapo_math/train.parquet','$DATA_ROOT/lighteval-math/train.parquet']}"
VAL_FILES="${VAL_FILES:-['$DATA_ROOT/math500/test.parquet','$DATA_ROOT/aime2024/test.parquet','$DATA_ROOT/aime2025/test.parquet','$DATA_ROOT/amc/test.parquet','$DATA_ROOT/minervamath/test.parquet','$DATA_ROOT/olympiadbench/test.parquet']}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-2048}"    # 1024 prompt + 2048 response -> 3072 teacher max_model_len
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
TRAIN_SEED="${TRAIN_SEED:-3407}"
LR="${LR:-1e-5}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"

# --------------------------------------------------- per-step on-policy -----
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-256}"
ROLLOUT_N="${ROLLOUT_N:-1}"                           # single on-policy sample per prompt
PPO_EPOCHS="${PPO_EPOCHS:-1}"
# Root-caused 2026-09-14 (see verl/workers/actor/dp_actor.py _pool_kd_loss):
# the historical mini=64/token-mean pair let a flat, length-biased KD loss
# blow up entropy the same way it did for OPD^2 (qwen25-1p5b_onpolicy_kd_gxpo_k3_a0.1:
# entropy 0.84->4.2 by step 4, val collapsed to ~0.03). Default to the
# paper-faithful pairing (1 update/batch, every response weighted equally).
PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-256}"
LOSS_AGG_MODE="${LOSS_AGG_MODE:-seq-mean-token-mean}"

# --------------------------------------------------------------- GXPO -------
# GXPO is an optimizer wrapper (3-pass probe/reposition/correct) and composes
# with any advantage/loss source, KD included, with zero changes to the KD
# path itself. USE_GXPO=False reproduces the original KD-only launcher exactly.
USE_GXPO="${USE_GXPO:-True}"
K="${K:-3}"
REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.8}"
GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
GXPO_RETENTION_SPACE="${GXPO_RETENTION_SPACE:-auto}"
# 100-step run: scale the shutoff-gate warmup to the same 12.5% fraction the
# other 100-step GXPO launchers use (common.sh's 50-step default assumes 400).
GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-12}"
GXPO_RESET_ENTROPY_AFTER_WARMUP="${GXPO_RESET_ENTROPY_AFTER_WARMUP:-True}"
GXPO_MIN_EFFECTIVE_MULTIPLIER="${GXPO_MIN_EFFECTIVE_MULTIPLIER:-0}"
GXPO_TAU="${GXPO_TAU:-3.0}"
GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-entropy}"
GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-3}"
GXPO_FALLBACK_MODE="${GXPO_FALLBACK_MODE:-permanent}"
GXPO_FALLBACK_WINDOW="${GXPO_FALLBACK_WINDOW:-10}"
GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-trajectory_aware}"
GXPO_DIAG_FREQ="${GXPO_DIAG_FREQ:-10}"
GXPO_ACTOR_DUTY_CYCLE="${GXPO_ACTOR_DUTY_CYCLE:-0}"
GXPO_FLAGS=(+actor_rollout_ref.actor.use_gxpo=False)
if [[ "${USE_GXPO,,}" == "true" ]]; then
  GXPO_FLAGS=(
    +actor_rollout_ref.actor.use_gxpo=True
    +actor_rollout_ref.actor.zscore_w=0
    +actor_rollout_ref.actor.gxpo_k="$K"
    +actor_rollout_ref.actor.gxpo_alpha="$REPOSITION_ALPHA"
    +actor_rollout_ref.actor.gxpo_min_effective_multiplier="$GXPO_MIN_EFFECTIVE_MULTIPLIER"
    +actor_rollout_ref.actor.gxpo_delta=1e-8
    +actor_rollout_ref.actor.gxpo_tau="$GXPO_TAU"
    +actor_rollout_ref.actor.gxpo_zscore_w="$GXPO_ZSCORE_W"
    +actor_rollout_ref.actor.gxpo_trigger_signal="$GXPO_TRIGGER_SIGNAL"
    +actor_rollout_ref.actor.gxpo_trigger_patience="$GXPO_TRIGGER_PATIENCE"
    +actor_rollout_ref.actor.gxpo_fallback_mode="$GXPO_FALLBACK_MODE"
    +actor_rollout_ref.actor.gxpo_optimizer_state_mode="$GXPO_OPTIMIZER_STATE_MODE"
    +actor_rollout_ref.actor.gxpo_retention_space="$GXPO_RETENTION_SPACE"
    +actor_rollout_ref.actor.gxpo_fallback_window="$GXPO_FALLBACK_WINDOW"
    +actor_rollout_ref.actor.gxpo_trigger_granularity=outer
    +actor_rollout_ref.actor.gxpo_warmup_steps="$GXPO_WARMUP_STEPS"
    +actor_rollout_ref.actor.gxpo_reset_entropy_after_warmup="$GXPO_RESET_ENTROPY_AFTER_WARMUP"
    +actor_rollout_ref.actor.gxpo_omega=0.1
    +actor_rollout_ref.actor.gxpo_shutoff_mode="$GXPO_SHUTOFF_MODE"
    +actor_rollout_ref.actor.gxpo_recompute_old_log_probs=False
    +actor_rollout_ref.actor.gxpo_diag_freq="$GXPO_DIAG_FREQ"
    +actor_rollout_ref.actor.gxpo_actor_duty_cycle="$GXPO_ACTOR_DUTY_CYCLE"
  )
fi

# --------------------------------------------------------------- KD ---------
# The teacher is a frozen plain-HF model doing ONE teacher-forced forward + top-K
# per step. It is parked on CPU between steps; each phase transfers the full
# teacher, so this is an intentional per-step residency/PCIe cost (not free).
KD_TOPK="${KD_TOPK:-1024}"
KD_TEACHER_PATH="${KD_TEACHER_PATH:-$GXPO_PROJECT_ROOT/models/Qwen2.5-7B-Instruct}"
KD_TEACHER_DTYPE="${KD_TEACHER_DTYPE:-bfloat16}"
KD_TEACHER_NUM_REPLICAS="${KD_TEACHER_NUM_REPLICAS:-1}"
# Dedicated-GPU teacher (mirrors OPD^2's opd2_dedicated_gpu): the teacher gets
# its own GPU(s), additive to GPU_COUNT, and stays resident all run instead of
# CPU-parking between steps. Set GPU_IDS/GPU_COUNT to cover GPU_COUNT training
# GPUs + KD_TEACHER_NUM_REPLICAS dedicated ones when enabling this.
KD_TEACHER_DEDICATED_GPU="${KD_TEACHER_DEDICATED_GPU:-True}"
KD_TEACHER_STUDENT_VOCAB_SIZE="${KD_TEACHER_STUDENT_VOCAB_SIZE:-151936}"
KD_TEACHER_MICRO_BATCH_SIZE="${KD_TEACHER_MICRO_BATCH_SIZE:-8}"
KD_TEACHER_CHUNK_TOKENS="${KD_TEACHER_CHUNK_TOKENS:-2048}"
KD_TEACHER_ATTN_IMPL="${KD_TEACHER_ATTN_IMPL:-flash_attention_2}"

# ----------------------------------------------------------- preflight ------
MISSING=0
[[ -x "${VIRTUAL_ENV:-}/bin/python" ]] || { echo "PREFLIGHT FAIL: no venv python at ${VIRTUAL_ENV:-<unset>}/bin/python" >&2; MISSING=1; }
[[ -f "$KD_TEACHER_PATH/config.json" ]] || { echo "PREFLIGHT FAIL: teacher not found at $KD_TEACHER_PATH" >&2; MISSING=1; }
# TRAIN_FILES/VAL_FILES follow Hydra list syntax (e.g. "['a.parquet','b.parquet']")
# or a single bare path; check every path referenced either way.
_check_parquet_list() {
  local label="$1" raw="$2" stripped f
  stripped="${raw#\[}"; stripped="${stripped%\]}"
  stripped="${stripped//\'/}"
  IFS=',' read -ra _paths <<< "$stripped"
  for f in "${_paths[@]}"; do
    [[ -n "$f" ]] || continue
    [[ -f "$f" ]] || { echo "PREFLIGHT FAIL: $label component not found: $f" >&2; MISSING=1; }
  done
}
_check_parquet_list TRAIN_FILES "$TRAIN_FILES"
_check_parquet_list VAL_FILES "$VAL_FILES"
if [[ "$MISSING" -ne 0 ]]; then echo "Preflight failed - fix the items above and re-run." >&2; exit 2; fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  cat <<EOT
[dry-run] resolved on-policy KD-only launch configuration
  repo_root          : $REPO_ROOT
  student            : $MODEL_ID
  actor_dtype        : fp32 master params (FSDP bf16 compute, fp32 reduce/buffer; vLLM rollout bf16)
  teacher            : $KD_TEACHER_PATH (HF fwd, $KD_TEACHER_NUM_REPLICAS replicas, mb=$KD_TEACHER_MICRO_BATCH_SIZE, chunk=$KD_TEACHER_CHUNK_TOKENS, dedicated_gpu=$KD_TEACHER_DEDICATED_GPU)
  gpus (locked)      : $GPU_IDS   rollout attn: $VLLM_ATTENTION_BACKEND
  vllm scheduler     : max_num_batched_tokens=$VLLM_MAX_NUM_BATCHED_TOKENS max_num_seqs=$VLLM_MAX_NUM_SEQS enforce_eager=$VLLM_ENFORCE_EAGER free_cache_engine=$VLLM_FREE_CACHE_ENGINE
  train_batch        : $TRAIN_BATCH_SIZE  rollout.n=$ROLLOUT_N  mini=$PPO_MINI_BATCH_SIZE  ppo_epochs=$PPO_EPOCHS
  gxpo               : $( [[ "${USE_GXPO,,}" == "true" ]] && echo "enabled (K=$K, alpha=$REPOSITION_ALPHA, warmup=${GXPO_WARMUP_STEPS} steps)" || echo "disabled" )
  kd                 : topk=$KD_TOPK use_pg=False loss_agg_mode=$LOSS_AGG_MODE
  speed              : entropy_coeff=${ENTROPY_COEFF:-0} liger=${USE_LIGER:-True}
  run_dir            : $RUN_DIR
[dry-run] preflight OK - would launch now.
EOT
  exit 0
fi

mkdir -p "$RUN_DIR"
export WANDB_DIR="$RUN_DIR" WANDB_PROJECT="$PROJECT" WANDB_MODE="${WANDB_MODE:-online}"
export VLLM_CACHE_ROOT="$RUN_DIR/vllm_cache"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$RUN_DIR/flashinfer_autotune_cache"
mkdir -p "$VLLM_CACHE_ROOT" "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"

python -u -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  +algorithm.norm_adv_by_std_in_grpo=False \
  +algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  data.max_prompt_length=1024 \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.seed="$TRAIN_SEED" \
  data.system_prompt="$SYSTEM_PROMPT" \
  actor_rollout_ref.model.path="$MODEL_ID" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.model.attn_implementation="$ATTN_IMPL" \
  +actor_rollout_ref.model.use_liger="${USE_LIGER:-True}" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  +actor_rollout_ref.actor.optim.name=adamw \
  +actor_rollout_ref.actor.optim.foreach="${ADAMW_FOREACH:-True}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$PPO_MINI_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_epochs="$PPO_EPOCHS" \
  +actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-12288}" \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_kl_loss=False \
  actor_rollout_ref.actor.entropy_coeff="${ENTROPY_COEFF:-0}" \
  actor_rollout_ref.actor.kl_loss_coef=0.0 \
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD:-False}" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD:-False}" \
  +actor_rollout_ref.actor.fsdp_config.model_dtype=fp32 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-8}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE:-1}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="$VLLM_MAX_NUM_BATCHED_TOKENS" \
  actor_rollout_ref.rollout.max_num_seqs="$VLLM_MAX_NUM_SEQS" \
  actor_rollout_ref.rollout.enable_chunked_prefill="$VLLM_ENABLE_CHUNKED_PREFILL" \
  actor_rollout_ref.rollout.attention_backend="$VLLM_ATTENTION_BACKEND" \
  actor_rollout_ref.rollout.enforce_eager="$VLLM_ENFORCE_EAGER" \
  actor_rollout_ref.rollout.free_cache_engine="$VLLM_FREE_CACHE_ENGINE" \
  +actor_rollout_ref.rollout.enable_prefix_caching="$VLLM_ENABLE_PREFIX_CACHING" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.n=1 \
  actor_rollout_ref.rollout.val_kwargs.do_sample=False \
  actor_rollout_ref.rollout.val_kwargs.temperature=0 \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  reward_model.reward_manager=naive \
  algorithm.kl_ctrl.kl_coef=0.000 \
  trainer.critic_warmup=0 \
  trainer.logger="['console','wandb']" \
  trainer.project_name="$PROJECT" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$RUN_DIR" \
  trainer.n_gpus_per_node="$GPU_COUNT" \
  trainer.nnodes=1 \
  +trainer.max_colocate_count="${MAX_COLOCATE_COUNT:-10}" \
  trainer.save_freq="${SAVE_FREQ:-20}" \
  trainer.test_freq="${TRAINER_TEST_FREQ:-5}" \
  +trainer.keep_last_ckpts=1 \
  +trainer.keep_all_ckpts=False \
  +trainer.val_before_train="${VAL_BEFORE_TRAIN:-True}" \
  trainer.total_training_steps="${MAX_STEPS:-100}" \
  trainer.total_epochs=100 \
  "${GXPO_FLAGS[@]}" \
  +actor_rollout_ref.actor.use_kd=True \
  +actor_rollout_ref.actor.kd_use_pg=False \
  +actor_rollout_ref.actor.kd_topk="$KD_TOPK" \
  +actor_rollout_ref.actor.kd_teacher_path="$KD_TEACHER_PATH" \
  +actor_rollout_ref.actor.kd_teacher.path="$KD_TEACHER_PATH" \
  +actor_rollout_ref.actor.kd_teacher.dtype="$KD_TEACHER_DTYPE" \
  +actor_rollout_ref.actor.kd_teacher.num_replicas="$KD_TEACHER_NUM_REPLICAS" \
  +actor_rollout_ref.actor.kd_teacher.dedicated_gpu="$KD_TEACHER_DEDICATED_GPU" \
  +actor_rollout_ref.actor.kd_teacher.student_vocab_size="$KD_TEACHER_STUDENT_VOCAB_SIZE" \
  +actor_rollout_ref.actor.kd_teacher.micro_batch_size="$KD_TEACHER_MICRO_BATCH_SIZE" \
  +actor_rollout_ref.actor.kd_teacher.chunk_tokens="$KD_TEACHER_CHUNK_TOKENS" \
  +actor_rollout_ref.actor.kd_teacher.attn_implementation="$KD_TEACHER_ATTN_IMPL" \
  +actor_rollout_ref.actor.kd_teacher.pad_token_id=0 \
  2>&1 | tee "$RUN_DIR/train.log"
