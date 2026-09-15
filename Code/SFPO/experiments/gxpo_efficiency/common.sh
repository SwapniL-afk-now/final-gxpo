#!/usr/bin/env bash
# Shared launcher for the final 3-model x 3-method GXPO efficiency matrix.
# All fairness-critical settings live here; the nine entrypoints only select
# MODEL_ALIAS, MODEL_ID, and METHOD.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

# Make every launch reproducible from a fresh tmux shell.  The base image's
# venv is not automatically activated, and FA3/dependency wheels are kept in
# user-writable workspace paths on this unprivileged instance.
# This checkout (final-gxpo) has never had its own .venv provisioned on this
# host -- the working one lives in the sibling final-gxpo-h200 checkout. Try
# this checkout's own venv first (so a future provision here wins silently),
# then fall back to the known-working sibling.
GXPO_PROJECT_ROOT="$(cd -- "$REPO_ROOT/../.." && pwd)"
for _gxpo_venv in "$GXPO_PROJECT_ROOT/.venv" "/office/dev_workspace/swapnil/final-gxpo-h200/.venv"; do
  if [[ -x "$_gxpo_venv/bin/python" ]]; then
    export VIRTUAL_ENV="$_gxpo_venv"
    export PATH="$_gxpo_venv/bin:$PATH"
    # This host has no system /usr/local/cuda and no bare `nvcc` on PATH; the
    # only nvcc is the one the venv's nvidia-cuda-nvcc wheel unpacked into
    # site-packages. flashinfer's JIT compiler (used by the vLLM rollout
    # engine's FLASHINFER attention backend) needs CUDA_HOME to find it --
    # without this it fails at the first uncached kernel shape with
    # "Could not find nvcc and default cuda_home='/usr/local/cuda' doesn't exist".
    for _gxpo_cuda_home in "$_gxpo_venv"/lib/python*/site-packages/nvidia/cu13; do
      if [[ -x "$_gxpo_cuda_home/bin/nvcc" ]]; then
        export CUDA_HOME="$_gxpo_cuda_home"
        # The pip nvidia-cuda-nvcc wheel ships only lib/ (no lib64) and only
        # versioned .so.N files (no unversioned dev symlinks), so linking
        # (-lcudart) and loading (dlopen) both fail against it out of the box;
        # lib64 and the libcudart.so symlink are created once as a one-time
        # environment fixup (see gxpo-efficiency-flashinfer-jit-toolchain memory).
        # This also covers the runtime dlopen, which needs the .so found via
        # LD_LIBRARY_PATH, not just the link-time -L flag.
        export LD_LIBRARY_PATH="$_gxpo_cuda_home/lib:${LD_LIBRARY_PATH:-}"
        break
      fi
    done
    unset _gxpo_cuda_home
    break
  fi
done
unset _gxpo_venv
# Triton JIT-compiles a small C launcher shim for every kernel it builds, and that
# shim does `#include <Python.h>`. The venv's interpreter is the system python3.12
# (sys.base_prefix=/usr), so sysconfig points Triton at /usr/include/python3.12 --
# which does not exist here, since python3.12-dev is not installed and this user
# cannot install it. gcc also searches CPATH, so pointing that at a copy of the
# 3.12 headers is enough. Same fix the KD launchers in this directory already
# carry (qwen25_3b_onpolicy_kd.sh:71); it belongs here so every RL entrypoint that
# sources common.sh gets it instead of dying at the first uncached kernel.
GXPO_PY_INCLUDE="${GXPO_PY_INCLUDE:-/office/shared_cache/.local/share/uv/python/cpython-3.12.14-linux-x86_64-gnu/include/python3.12}"
if [[ -f "$GXPO_PY_INCLUDE/Python.h" ]]; then
  export CPATH="$GXPO_PY_INCLUDE${CPATH:+:$CPATH}"
else
  echo "PREFLIGHT WARN: no Python.h at $GXPO_PY_INCLUDE; Triton kernel compilation" >&2
  echo "                 will fail. Set GXPO_PY_INCLUDE to a python3.12 include dir." >&2
fi
# .runtime_deps is an optional extra-wheels shim (not present on this host);
# REPO_ROOT itself must always be on PYTHONPATH since that's where the verl
# package this launcher imports (`-m verl.trainer.main_ppo`) actually lives.
export PYTHONPATH="$REPO_ROOT/.runtime_deps:$REPO_ROOT:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-$REPO_ROOT/.hf_home}"
# A system-provided HF_HOME can exist but still be unwritable by the training
# user.  Fall back to the repository cache instead of failing inside Ray's
# remote main task with an opaque worker shutdown.
if [[ ! -w "$HF_HOME/hub" ]]; then
  export HF_HOME="$REPO_ROOT/.hf_home"
fi
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HF_HOME/hub}"
mkdir -p "$HF_HUB_CACHE"
ATTN_IMPL="${ATTN_IMPL:-flash_attention_3}"
export ATTN_IMPL
python - "$ATTN_IMPL" <<'PY'
import importlib.util
import sys

attn_impl = sys.argv[1]
required = ['pandas', 'wandb', 'tensordict']
if attn_impl == 'flash_attention_2':
    required.append('flash_attn')
else:
    required.extend(['flash_attn_3', 'flash_attn_interface'])
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit(f"Missing runtime dependencies for {attn_impl}: {', '.join(missing)}")
PY

REPOSITION_ALPHA="${REPOSITION_ALPHA:-0.5}"
K="${K:-5}"
TRAIN_SEED="${TRAIN_SEED:-3407}"
FINAL_EVAL_SEEDS="${FINAL_EVAL_SEEDS:-0 1 2 3}"
MAX_STEPS="${MAX_STEPS:-400}"
SAVE_FREQ="${SAVE_FREQ:-5}"
SFPO_WARMUP_STEPS="${SFPO_WARMUP_STEPS:-50}"
GXPO_WARMUP_STEPS="${GXPO_WARMUP_STEPS:-50}"
# Referenced unconditionally by the gxpo METHOD_FLAGS below but never defaulted
# here, so under `set -u` every entrypoint that did not export it died with
# "unbound variable" the moment it reached common.sh -- including
# qwen25_math_1p5b_gxpo_k10.sh and ..._gxpo_adamw_transactional_dir_k10.sh
# (their --dry-run exits before this file, which is why it stayed hidden).
# True matches SFPO_RESET_ENTROPY_AFTER_WARMUP's default and what most
# launchers that DO export it use.
GXPO_RESET_ENTROPY_AFTER_WARMUP="${GXPO_RESET_ENTROPY_AFTER_WARMUP:-True}"
GXPO_TAU="${GXPO_TAU:-3.0}"
GXPO_ZSCORE_W="${GXPO_ZSCORE_W:-30}"
GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-3}"
# Floor on GXPO's effective displacement multiplier alpha*scale. When the measured
# retention is low, alpha*scale can fall BELOW 1, which makes the 3-pass update
# land short of theta2 -- contracting instead of extrapolating, i.e. paying 3
# passes for a damped ordinary step. dp_actor warns once when that happens; set
# this to 1.0 to clamp instead. 0 (default) preserves the historical behavior.
GXPO_MIN_EFFECTIVE_MULTIPLIER="${GXPO_MIN_EFFECTIVE_MULTIPLIER:-0}"
# Entropy bonus subtracted from the policy loss (policy_loss = pg_loss - entropy*coeff).
# yaml default is 0.001; raise it to fight entropy collapse under a step size that's
# too large for the advantage's geometry (dp_actor.py:1061).
ENTROPY_COEFF="${ENTROPY_COEFF:-0.001}"
# GXPO optimizer state across the two probe steps. Parameters are repositioned to
# theta_tilde in both modes; only the optimizer state the slow correction starts from
# differs:
#   transactional            -- snapshot the optimizer state before probe 1 and roll back
#                               to it after repositioning, so the slow correction is taken
#                               from the moments the minibatch started with.
#   transactional_fast_state -- no refresh: the probe steps' moments and step counter are
#                               kept and the slow correction is taken from them, so Adam's
#                               step counter advances 3x per minibatch instead of 1x.
# OPT_STATE_TAG is appended to RUN_NAME below so the two arms never share a run dir or
# wandb run; transactional is left untagged because it is the established baseline.
GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
case "$GXPO_OPTIMIZER_STATE_MODE" in
  transactional)            OPT_STATE_TAG="" ;;
  transactional_fast_state) OPT_STATE_TAG="_optkeep" ;;
  *) echo "PREFLIGHT FAIL: GXPO_OPTIMIZER_STATE_MODE must be transactional or transactional_fast_state, got '$GXPO_OPTIMIZER_STATE_MODE'" >&2; exit 2 ;;
esac
# Which space the retention ratio is measured in. GXPO models OPTIMIZER-induced
# motion, so the right estimator is a property of the optimizer:
#   auto   -- optimizer-aware. Muon-owned matrices use update-space (per-matrix
#             scalar rho = <u0,u1>/<u0,u0>, read off the two real optimizer
#             steps); AdamW-owned parameters use the coordinatewise AdamW
#             optimizer-DIRECTION ratio r = d1/d0, where d_t is reconstructed
#             from the real probe displacement as (c*theta_t - theta_{t+1})/lr
#             with c = 1 - lr*weight_decay. An unrecognized optimizer falls back
#             to gradient space with a warning rather than being modelled wrong.
#   grad   -- force the legacy coordinatewise g1/g0 estimator everywhere. This is
#             the pre-optimizer-aware AdamW behavior and the A/B control arm.
#   update -- force update-space everywhere.
# Gradient ratios are meaningless for Muon: it normalizes the momentum matrix
# before Newton-Schulz and scales the write-back by parameter shape alone, so
# its step size does not depend on gradient magnitude at all. For AdamW they are
# merely an approximation -- AdamW moves along m_hat/(sqrt(v_hat)+eps), not along
# the gradient, so g1/g0 systematically overstates a gradient jump.
GXPO_RETENTION_SPACE="${GXPO_RETENTION_SPACE:-auto}"
case "$GXPO_RETENTION_SPACE" in
  auto|grad|update) ;;
  *) echo "PREFLIGHT FAIL: GXPO_RETENTION_SPACE must be auto, grad or update, got '$GXPO_RETENTION_SPACE'" >&2; exit 2 ;;
esac
GXPO_FALLBACK_MODE="${GXPO_FALLBACK_MODE:-permanent}"
GXPO_FALLBACK_WINDOW="${GXPO_FALLBACK_WINDOW:-10}"
GXPO_ACTOR_DUTY_CYCLE="${GXPO_ACTOR_DUTY_CYCLE:-0}"
GXPO_DIAG_FREQ="${GXPO_DIAG_FREQ:-10}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
# OOM fix: the gxpo_efficiency chain was missing the allocator setting the KD
# launchers already carry (qwen25_math_1p5b_onpolicy_kd_gxpo.sh and friends).
# Symptom here: vLLM generation died asking for 1.64GB with 94.60/94.97 GiB in
# use, while ~11.5GB per GPU sat in reserved-but-unusable segments (reserved
# 134.7GB vs allocated 111.8GB summed over both GPUs). Capping the split size
# keeps blocks >=128MB intact so large contiguous requests can be served.
# expandable_segments is NOT usable: vLLM's CuMemAllocator hard-crashes on it
# at engine init (see the comment in the KD launchers).
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:128}"
SFPO_ZSCORE_THRESHOLD="${SFPO_ZSCORE_THRESHOLD:-2.5}"
SFPO_TRIGGER_PATIENCE="${SFPO_TRIGGER_PATIENCE:-3}"
SFPO_RESET_ENTROPY_AFTER_WARMUP="${SFPO_RESET_ENTROPY_AFTER_WARMUP:-True}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-64}"
# Pre-generation curriculum filter (verl/trainer/ppo/presampling_selector.py): skips
# prompts the model has already solved recently ("easy") and, with decreasing
# probability the longer they've stayed unsolved, ones it keeps failing ("hard") --
# BEFORE generating for them, which is where the wall-clock savings actually come
# from (skipped prompts never pay for a rollout+reward+backward pass at all).
# Off by default -- preserves existing behavior for every entrypoint in this matrix
# that doesn't opt in.
GXPO_DYNAMIC_FILTERING="${GXPO_DYNAMIC_FILTERING:-False}"
# dynamic_filtering_strategy=linear_backoff is broken upstream: presampling_selector.py's
# filter_examples_linear_backoff() calls hard_linear_backoff_skip() without its required
# `k` argument, so it raises TypeError on the first example it evaluates as a hard-skip
# candidate. all_probabilistic is the only strategy that's actually wired end to end
# (self.p_easy/self.p_hard init'd from config, adaptively tuned every epoch, logged to
# examples/p_easy and examples/p_hard); keep_all is a documented no-op passthrough.
GXPO_DYNAMIC_FILTERING_STRATEGY="${GXPO_DYNAMIC_FILTERING_STRATEGY:-all_probabilistic}"
case "$GXPO_DYNAMIC_FILTERING_STRATEGY" in
  linear_backoff)
    echo "PREFLIGHT FAIL: GXPO_DYNAMIC_FILTERING_STRATEGY=linear_backoff is broken upstream" >&2
    echo "  (filter_examples_linear_backoff -> hard_linear_backoff_skip() missing its" >&2
    echo "  required k argument; TypeError on the first hard-skip candidate). Use" >&2
    echo "  all_probabilistic or keep_all instead." >&2
    exit 2
    ;;
  all_probabilistic|keep_all) ;;
  *)
    echo "PREFLIGHT FAIL: GXPO_DYNAMIC_FILTERING_STRATEGY must be all_probabilistic or" >&2
    echo "  keep_all, got '$GXPO_DYNAMIC_FILTERING_STRATEGY'" >&2
    exit 2
    ;;
esac
# Initial skip probabilities for the all_probabilistic strategy; these are the
# class methods' own default arguments (easy_probabilistic_skip/hard_probabilistic_skip
# in presampling_selector.py) and self-tune every epoch after that via p_easy/p_hard.
GXPO_P_EASY="${GXPO_P_EASY:-0.75}"
GXPO_P_HARD="${GXPO_P_HARD:-0.5}"
GXPO_TARGET_ZERO_VARIANCE="${GXPO_TARGET_ZERO_VARIANCE:-0.25}"
# Prompts accumulate across dataloader pulls (skipping filtered ones) until this many
# survive, then get trimmed to exactly this count -- so the trained batch size stays
# fixed at TRAIN_BATCH_SIZE regardless of how much the filter skips, instead of shrinking
# unpredictably. Must match TRAIN_BATCH_SIZE for that guarantee to hold.
GXPO_SAMPLING_BATCH_SIZE="${GXPO_SAMPLING_BATCH_SIZE:-$TRAIN_BATCH_SIZE}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-128}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-3072}"
VAL_BEFORE_TRAIN="${VAL_BEFORE_TRAIN:-True}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
ROLLOUT_N="${ROLLOUT_N:-8}"
# Responses per prompt for periodic/final VALIDATION decoding (val_kwargs.n).
# Independent of ROLLOUT_N above, which is the training-time num_generations.
VAL_N="${VAL_N:-1}"
# Validation sampling. Defaults preserve the historical arms (sampled, temp
# 0.7): VAL_DO_SAMPLE=False selects true greedy (temperature 0, n=1) via the
# rollout's do_sample=False branch.
VAL_DO_SAMPLE="${VAL_DO_SAMPLE:-True}"
VAL_TEMPERATURE="${VAL_TEMPERATURE:-0.7}"
LR="${LR:-1e-6}"
# LR schedule. `warmup_style` and `min_lr_ratio` are declared in
# verl/trainer/config/ppo_trainer.yaml but used to be DEAD -- fsdp_workers.py
# hardcoded the constant-with-warmup schedule. They are honoured now, so these
# defaults (constant, no warmup) preserve every existing arm exactly.
LR_WARMUP_STYLE="${LR_WARMUP_STYLE:-constant}"
LR_WARMUP_RATIO="${LR_WARMUP_RATIO:-0.0}"
LR_MIN_RATIO="${LR_MIN_RATIO:-0.0}"
case "$LR_WARMUP_STYLE" in
  constant|cosine) ;;
  *) echo "PREFLIGHT FAIL: LR_WARMUP_STYLE must be constant or cosine, got '$LR_WARMUP_STYLE'" >&2; exit 2 ;;
esac

# ------------------------------------------------------------------- OPD^2 ---
# On-Policy Delta Distillation (arXiv:2607.15161). Turning this on REPLACES the
# verifier advantage with the dense per-token teacher-delta signal:
#   signal = (teacher_gt - teacher_base_gt) - (E_base[teacher] - E_base[teacher_base])
# gated to zero wherever it disagrees with the (teacher - student) update
# direction. The math verifier keeps running (its reward still feeds GXPO's
# degenerate-batch guard and every reward dashboard) but no longer drives the
# update. Orthogonal to GXPO, which lives below the loss.
# Everything defaults off: with OPD2_ENABLED unset, every existing entrypoint
# resolves to exactly the config it resolved to before.
OPD2_ENABLED="${OPD2_ENABLED:-0}"
OPD2_TEACHER="${OPD2_TEACHER:-}"
OPD2_TEACHER_BASE="${OPD2_TEACHER_BASE:-}"
OPD2_TOPK="${OPD2_TOPK:-1024}"
OPD2_GEN_LOSS_WEIGHT="${OPD2_GEN_LOSS_WEIGHT:-0.1}"
# Offline audit (2026-09-14, scratchpad/opd2_audit*): the raw teacher-delta
# signal's within-prompt correct-vs-wrong AUC is ~0.48-0.52 (chance) for the
# 3B student / Math-7B pair -- see verl/trainer/ppo/ray_trainer.py near
# opd2_outcome_weight. 0 = paper behaviour (signal only, verifier metrics-only).
OPD2_OUTCOME_WEIGHT="${OPD2_OUTCOME_WEIGHT:-0.0}"
OPD2_REWARDS_BIAS="${OPD2_REWARDS_BIAS:-0.0}"
OPD2_TEACHER_TEMPLATE="${OPD2_TEACHER_TEMPLATE:-True}"
OPD2_MICRO_BATCH_SIZE="${OPD2_MICRO_BATCH_SIZE:-1}"
OPD2_CHUNK_TOKENS="${OPD2_CHUNK_TOKENS:-512}"
OPD2_KEEP_ON_GPU="${OPD2_KEEP_ON_GPU:-False}"
OPD2_ATTN_IMPL="${OPD2_ATTN_IMPL:-flash_attention_2}"
# True = teacher + teacher_base get their OWN GPU (a Ray actor holding one of
# GPU_IDS) and training uses GPU_COUNT of the rest, so set GPU_IDS to
# GPU_COUNT+1 devices. The models are loaded once and stay resident there.
OPD2_DEDICATED_GPU="${OPD2_DEDICATED_GPU:-False}"
OPD2_FLAGS=()
RETURN_RAW_CHAT="${RETURN_RAW_CHAT:-False}"
# Qwen3 chat_template switch for the student's prompt. null = template default
# (every non-Qwen3 arm); False = non-thinking mode.
ENABLE_THINKING="${ENABLE_THINKING:-null}"
# Each dataloader worker holds a full copy of the dataset (~0.8GB RSS here).
DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-8}"
case "${OPD2_ENABLED,,}" in
  1|true|yes|on) OPD2_ON=1 ;;
  0|false|no|off|"") OPD2_ON=0 ;;
  *) echo "PREFLIGHT FAIL: OPD2_ENABLED must be 0/1, got '$OPD2_ENABLED'" >&2; exit 2 ;;
esac
# ------------------------------------------------------- GRPO+SLED-Delta ---
# Auxiliary token-level self-distillation on top of unchanged GRPO:
#   L = grpo_coef * L_grpo + sled_loss_coef * L_sled,
# where L_sled is the OPD^2-style sign-gated SLED-Delta loss
# (verl/workers/actor/sled_delta.py). The GRPO advantage, ratio and clip are
# untouched; the gate acts only on the SLED term. Everything defaults off:
# with SLED_ENABLED unset, every existing entrypoint resolves to exactly the
# config it resolved to before. Parsed here (next to OPD2) so SLED_ON is
# defined before the RUN_NAME tags below (set -u would fail otherwise).
SLED_ENABLED="${SLED_ENABLED:-0}"
SLED_LOSS_COEF="${SLED_LOSS_COEF:-1.0}"
SLED_GRPO_COEF="${SLED_GRPO_COEF:-1.0}"
SLED_ALPHA="${SLED_ALPHA:-0.5}"
SLED_EARLY_LAYER="${SLED_EARLY_LAYER:--1}"
SLED_TOPK="${SLED_TOPK:-1024}"
SLED_MICRO_BATCH_SIZE="${SLED_MICRO_BATCH_SIZE:-2}"
SLED_CHUNK_TOKENS="${SLED_CHUNK_TOKENS:-512}"
SLED_FLAGS=()
case "${SLED_ENABLED,,}" in
  1|true|yes|on) SLED_ON=1 ;;
  0|false|no|off|"") SLED_ON=0 ;;
  *) echo "PREFLIGHT FAIL: SLED_ENABLED must be 0/1, got '$SLED_ENABLED'" >&2; exit 2 ;;
esac
if [[ "$SLED_ON" -eq 1 ]]; then
  if [[ "$OPD2_ON" -eq 1 ]]; then
    echo "PREFLIGHT FAIL: SLED_ENABLED=1 cannot be combined with OPD2_ENABLED=1:" >&2
    echo "  SLED-Delta composes with the GRPO advantage, which OPD^2 replaces." >&2
    exit 2
  fi
  python - "$SLED_LOSS_COEF" "$SLED_GRPO_COEF" "$SLED_ALPHA" "$SLED_TOPK" <<'PY' || exit 2
import sys
coef, gcoef, alpha, topk = float(sys.argv[1]), float(sys.argv[2]), float(sys.argv[3]), int(sys.argv[4])
assert coef >= 0.0, f'SLED_LOSS_COEF must be >= 0, got {coef}'
assert gcoef >= 0.0, f'SLED_GRPO_COEF must be >= 0, got {gcoef}'
assert alpha >= 0.0, f'SLED_ALPHA must be >= 0, got {alpha}'
assert topk > 0, f'SLED_TOPK must be > 0, got {topk}'
PY
  SLED_FLAGS+=(
    +actor_rollout_ref.actor.use_sled_delta=True
    +actor_rollout_ref.actor.sled_loss_coef="$SLED_LOSS_COEF"
    +actor_rollout_ref.actor.sled_grpo_coef="$SLED_GRPO_COEF"
    +actor_rollout_ref.actor.sled_alpha="$SLED_ALPHA"
    +actor_rollout_ref.actor.sled_early_layer="$SLED_EARLY_LAYER"
    +actor_rollout_ref.actor.sled_topk="$SLED_TOPK"
    +actor_rollout_ref.actor.sled_micro_batch_size="$SLED_MICRO_BATCH_SIZE"
    +actor_rollout_ref.actor.sled_chunk_tokens="$SLED_CHUNK_TOKENS"
  )
fi
# ---------------------------------------------------- multi-layer GRPO ---
# SELECTED_POLICY_LAYERS=12,14,16 averages the unchanged GRPO loss over those
# 1-based decoder layers (see actor.selected_policy_layers in ppo_trainer.yaml;
# range/duplicate validation happens in the actor). Unset/empty/null = the
# exact same command as before.
SELECTED_POLICY_LAYERS="${SELECTED_POLICY_LAYERS:-}"
[[ "${SELECTED_POLICY_LAYERS,,}" == "null" ]] && SELECTED_POLICY_LAYERS=""
SELECTED_POLICY_LAYERS="${SELECTED_POLICY_LAYERS// /}"
ML_FLAGS=()
if [[ -n "$SELECTED_POLICY_LAYERS" ]]; then
  if [[ ! "$SELECTED_POLICY_LAYERS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "PREFLIGHT FAIL: SELECTED_POLICY_LAYERS must be comma-separated ints, got '$SELECTED_POLICY_LAYERS'" >&2
    exit 2
  fi
  ML_FLAGS+=(actor_rollout_ref.actor.selected_policy_layers="[$SELECTED_POLICY_LAYERS]")
fi
# MULTILAYER_AUX_COEF=c: L = L_final + c * mean(L_intermediate). Unset = plain mean.
MULTILAYER_AUX_COEF="${MULTILAYER_AUX_COEF:-}"
if [[ -n "$MULTILAYER_AUX_COEF" ]]; then
  if [[ -z "$SELECTED_POLICY_LAYERS" || ! "$MULTILAYER_AUX_COEF" =~ ^[0-9]*\.?[0-9]+$ ]]; then
    echo "PREFLIGHT FAIL: MULTILAYER_AUX_COEF needs SELECTED_POLICY_LAYERS and a non-negative number, got '$MULTILAYER_AUX_COEF'" >&2
    exit 2
  fi
  ML_FLAGS+=(actor_rollout_ref.actor.multilayer_aux_coef="$MULTILAYER_AUX_COEF")
fi
# MULTILAYER_LOSS=kl: L = L_GRPO(final) + MULTILAYER_AUX_COEF * mean_l KL(sg[pi_final] || pi_l).
MULTILAYER_LOSS="${MULTILAYER_LOSS:-grpo}"
if [[ "$MULTILAYER_LOSS" != "grpo" && "$MULTILAYER_LOSS" != "kl" ]]; then
  echo "PREFLIGHT FAIL: MULTILAYER_LOSS must be grpo or kl, got '$MULTILAYER_LOSS'" >&2
  exit 2
fi
if [[ "$MULTILAYER_LOSS" == "kl" ]]; then
  if [[ -z "$SELECTED_POLICY_LAYERS" || -z "$MULTILAYER_AUX_COEF" ]]; then
    echo "PREFLIGHT FAIL: MULTILAYER_LOSS=kl needs SELECTED_POLICY_LAYERS and MULTILAYER_AUX_COEF" >&2
    exit 2
  fi
  ML_FLAGS+=(actor_rollout_ref.actor.multilayer_loss=kl)
fi
OPTIMIZER_NAME="${OPTIMIZER_NAME:-adamw}"
# AdamW weight decay. fsdp_workers defaults to 1e-2 when unset; HuggingFace
# TrainingArguments (and therefore every trl recipe that does not set it)
# defaults to 0.0, so recipe-faithful runs must pass it explicitly. Muon has its
# own MUON_WEIGHT_DECAY and ignores this.
ADAMW_WEIGHT_DECAY="${ADAMW_WEIGHT_DECAY:-1e-2}"
# How the per-token PPO loss is reduced to a scalar (verl/trainer/ppo/core_algos.py):
#   token-mean              -- one mean over every response token in the batch.
#                              verl's historical reduction; long responses carry
#                              proportionally more weight. DEFAULT: every existing
#                              GRPO/SFPO/GXPO arm keeps exactly this.
#   seq-mean-token-mean     -- mean within each sequence, then across sequences,
#                              so every response counts equally. This is trl's
#                              GRPOConfig loss_type="grpo", which the OPD^2 recipe
#                              pins -- use it for paper-faithful OPD^2 runs.
#   seq-mean-token-sum-norm -- trl's loss_type="dr_grpo".
# Verified by tools/opd2/test_loss_agg.py, which also checks that
# seq-mean-token-mean composes correctly across gradient accumulation.
LOSS_AGG_MODE="${LOSS_AGG_MODE:-token-mean}"
case "$LOSS_AGG_MODE" in
  token-mean|seq-mean-token-mean|seq-mean-token-sum-norm) ;;
  *) echo "PREFLIGHT FAIL: LOSS_AGG_MODE must be token-mean, seq-mean-token-mean or seq-mean-token-sum-norm, got '$LOSS_AGG_MODE'" >&2; exit 2 ;;
esac
USE_KL_LOSS="${USE_KL_LOSS:-False}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.0}"
MUON_MOMENTUM="${MUON_MOMENTUM:-0.95}"
MUON_NS_STEPS="${MUON_NS_STEPS:-5}"
MUON_NESTEROV="${MUON_NESTEROV:-True}"
MUON_WEIGHT_DECAY="${MUON_WEIGHT_DECAY:-1e-2}"
MUON_DISTRIBUTED_BACKEND="${MUON_DISTRIBUTED_BACKEND:-gather_scatter}"
USE_LIGER="${USE_LIGER:-True}"
OPTIM_FUSED="${OPTIM_FUSED:-False}"
ENABLE_GRADIENT_CHECKPOINTING="${ENABLE_GRADIENT_CHECKPOINTING:-True}"
USE_TORCH_COMPILE="${USE_TORCH_COMPILE:-True}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-False}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-False}"
ACTOR_MODEL_DTYPE="${ACTOR_MODEL_DTYPE:-}"
TRAINER_TEST_FREQ="${TRAINER_TEST_FREQ:-5}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-auto}"
TRAINER_RESUME_FROM_PATH="${TRAINER_RESUME_FROM_PATH:-False}"
GPU_COUNT="${GPU_COUNT:-${N_GPUS:-1}}"
PROJECT="${WANDB_PROJECT:-gxpo-efficiency-final}"
FINAL_EVAL_ENABLED="${FINAL_EVAL_ENABLED:-True}"

if [[ -z "${MODEL_ALIAS:-}" || -z "${MODEL_ID:-}" || -z "${METHOD:-}" ]]; then
  echo "common.sh requires MODEL_ALIAS, MODEL_ID, and METHOD" >&2
  exit 2
fi
case "$METHOD" in
  grpo|sfpo|gxpo) ;;
  *) echo "Unsupported METHOD=$METHOD" >&2; exit 2 ;;
esac

DATA_ROOT="${GXPO_DATA_ROOT:-$REPO_ROOT/data}"
DAPO_TRAIN="${DAPO_TRAIN:-$DATA_ROOT/dapo_math/train.parquet}"
LIGHTEVAL_TRAIN="${LIGHTEVAL_TRAIN:-$DATA_ROOT/lighteval-math/train.parquet}"
MATH500="${MATH500:-$DATA_ROOT/math500/test.parquet}"
AIME24="${AIME24:-$DATA_ROOT/aime2024/test.parquet}"
AIME25="${AIME25:-$DATA_ROOT/aime2025/test.parquet}"
AMC23="${AMC23:-$DATA_ROOT/amc/test.parquet}"
MINERVA="${MINERVA:-$DATA_ROOT/minervamath/test.parquet}"
OLYMPIAD="${OLYMPIAD:-$DATA_ROOT/olympiadbench/test.parquet}"

missing=0
for required in "$DAPO_TRAIN" "$LIGHTEVAL_TRAIN" "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD"; do
  if [[ ! -f "$required" ]]; then
    echo "Missing prepared dataset: $required" >&2
    missing=1
  fi
done
if [[ "$missing" -ne 0 ]]; then
  echo "Set GXPO_DATA_ROOT or the individual DAPO_TRAIN/LIGHTEVAL_TRAIN/benchmark variables." >&2
  exit 2
fi

if [[ "$MODEL_ID" == /* && ! -e "$MODEL_ID" ]]; then
  echo "Configured local model path does not exist: $MODEL_ID" >&2
  exit 2
fi

RUN_NAME="${GXPO_RUN_NAME:-${MODEL_ALIAS}_${METHOD}${METHOD:+_}$( [[ "$METHOD" == grpo ]] && echo "" || echo "k${K}_" )seed${TRAIN_SEED}}"
# Remove the accidental doubled separator for GRPO while keeping names explicit.
RUN_NAME="${RUN_NAME//__/_}"
# Tag the no-refresh optimizer-state arm. Applied after the name is resolved so it holds
# however RUN_NAME was derived, and guarded so re-entering common.sh cannot double-tag.
if [[ -n "$OPT_STATE_TAG" && "$RUN_NAME" != *"$OPT_STATE_TAG" ]]; then
  RUN_NAME="${RUN_NAME}${OPT_STATE_TAG}"
fi
# Tag the retention estimator whenever it is NOT what the run name has historically
# meant, so a new estimator can never resume an old run's wandb id, checkpoints or
# result directory (trainer.resume_mode=auto would otherwise splice new metrics into
# an old raw-gradient run). Same guard pattern as OPT_STATE_TAG.
#   adamw + auto   -> _adamwdir : NEW. auto used to be a no-op under AdamW (no
#                                 parameter is Muon-owned); it now selects the
#                                 optimizer-direction estimator r = d1/d0.
#   any   + update -> _updspace : forced update-space everywhere.
#   any   + grad   -> untagged  : the legacy estimator, i.e. what every existing
#                                 AdamW run name already means.
#   muon  + auto   -> untagged  : the established Muon arm; its run dirs exist.
RETENTION_TAG=""
if [[ "$GXPO_RETENTION_SPACE" == "update" ]]; then
  RETENTION_TAG="_updspace"
elif [[ "$GXPO_RETENTION_SPACE" == "auto" && "${OPTIMIZER_NAME,,}" == "adamw" ]]; then
  RETENTION_TAG="_adamwdir"
fi
if [[ -n "$RETENTION_TAG" && "$RUN_NAME" != *"$RETENTION_TAG"* ]]; then
  RUN_NAME="${RUN_NAME}${RETENTION_TAG}"
fi
# Tag runs with the pre-generation filter on, same guard pattern as OPT_STATE_TAG above --
# this is a methodology change (fewer/different prompts trained on per step), so it must
# get its own wandb run rather than resuming a pre-fix run's id/history.
if [[ "${GXPO_DYNAMIC_FILTERING,,}" == "true" && "$RUN_NAME" != *_dynfilt ]]; then
  RUN_NAME="${RUN_NAME}_dynfilt"
fi
# OPD^2 changes what the advantage MEANS, so it must never resume a reward-RL
# run's wandb id, checkpoints or result dir. Same guard pattern as above.
if [[ "$OPD2_ON" -eq 1 && "$RUN_NAME" != *_opd2* ]]; then
  RUN_NAME="${RUN_NAME}_opd2"
fi
# SLED adds an auxiliary loss, so it must never resume a plain-GRPO run's
# wandb id, checkpoints or result dir (trainer.resume_mode=auto would
# otherwise splice new metrics into an old run). Same guard pattern.
if [[ "$SLED_ON" -eq 1 && "$RUN_NAME" != *_sledsig* ]]; then
  RUN_NAME="${RUN_NAME}_sledsig"
fi
# Multi-layer GRPO is a different objective: never resume a plain-GRPO run.
if [[ -n "$SELECTED_POLICY_LAYERS" && "$RUN_NAME" != *_ml[0-9]* ]]; then
  RUN_NAME="${RUN_NAME}_ml${SELECTED_POLICY_LAYERS//,/-}"
fi
if [[ "$MULTILAYER_LOSS" == "kl" && "$RUN_NAME" != *_kl_aux* ]]; then
  RUN_NAME="${RUN_NAME}_kl"
fi
if [[ -n "$MULTILAYER_AUX_COEF" && "$RUN_NAME" != *_aux[0-9.]* ]]; then
  RUN_NAME="${RUN_NAME}_aux${MULTILAYER_AUX_COEF}"
fi
# A non-default loss reduction is a different objective, not a different setting.
# Tag it so it cannot resume a token-mean run's wandb id or checkpoints.
if [[ "$LOSS_AGG_MODE" != "token-mean" && "$RUN_NAME" != *_seqmean* && "$RUN_NAME" != *_drgrpo* ]]; then
  if [[ "$LOSS_AGG_MODE" == "seq-mean-token-mean" ]]; then
    RUN_NAME="${RUN_NAME}_seqmean"
  else
    RUN_NAME="${RUN_NAME}_drgrpo"
  fi
fi
RESULT_ROOT="${GXPO_RESULTS_ROOT:-$REPO_ROOT/results/gxpo_efficiency}"
RUN_DIR="$RESULT_ROOT/$RUN_NAME"
mkdir -p "$RUN_DIR"

# Keep vLLM and FlashInfer autotune artifacts writable and isolated per run.
export VLLM_CACHE_ROOT="$RUN_DIR/vllm_cache"
export VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR="$RUN_DIR/flashinfer_autotune_cache"
export VLLM_SLEEP_LEVEL="${VLLM_SLEEP_LEVEL:-2}"
mkdir -p "$VLLM_CACHE_ROOT" "$VLLM_FLASHINFER_AUTOTUNE_CACHE_DIR"

export GXPO_EFFICIENCY_RUN=1
export GXPO_RUN_NAME="$RUN_NAME"
export GXPO_MODEL_ALIAS="$MODEL_ALIAS"
export TRAIN_SEED
export FINAL_EVAL_SEEDS
export WANDB_PROJECT="$PROJECT"
if [[ ! -v WANDB_GROUP || -z "$WANDB_GROUP" ]]; then
  export WANDB_GROUP="$MODEL_ALIAS"
fi
if [[ ! -v WANDB_TAGS || -z "$WANDB_TAGS" ]]; then
  if [[ "$METHOD" == "grpo" ]]; then
    export WANDB_TAGS="model:$MODEL_ALIAS,method:$METHOD,experiment:final-efficiency"
  else
    export WANDB_TAGS="model:$MODEL_ALIAS,method:$METHOD,k:$K,alpha:$REPOSITION_ALPHA,experiment:final-efficiency"
  fi
fi
export WANDB_MODE="${WANDB_MODE:-online}"
export WANDB_DIR="$RUN_DIR"
export RAY_ADDRESS=local
export MPLBACKEND=Agg

if [[ -n "${GPU_IDS:-}" ]]; then
  export CUDA_VISIBLE_DEVICES="$GPU_IDS"
fi

# A duty-cycle sleep reduces sustained load but cannot guarantee an
# instantaneous board-power ceiling. For power-sensitive GXPO launches, fail
# closed unless the physical NVIDIA power limit is already at or below the
# configured maximum. Applying the limit requires an administrator.
if [[ "${GXPO_ENFORCE_POWER_LIMIT:-False}" == "True" || "${GXPO_ENFORCE_POWER_LIMIT:-0}" == "1" ]]; then
  GXPO_MAX_POWER_W="${GXPO_MAX_POWER_W:-500}"
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "GXPO power safety: nvidia-smi is unavailable; refusing to launch." >&2
    exit 2
  fi
  gxpo_power_limits="$(nvidia-smi --id="$GPU_IDS" \
    --query-gpu=index,power.limit --format=csv,noheader,nounits 2>/dev/null)" || {
    echo "GXPO power safety: unable to read power limits for GPUs $GPU_IDS; refusing to launch." >&2
    exit 2
  }
  while IFS=',' read -r gxpo_gpu_id gxpo_power_limit; do
    [[ -z "$gxpo_gpu_id" ]] && continue
    if awk -v limit="$gxpo_power_limit" -v maximum="$GXPO_MAX_POWER_W" \
      'BEGIN { exit !(limit+0 > maximum+0) }'; then
      echo "GXPO power safety: GPU ${gxpo_gpu_id//[[:space:]]/} limit=${gxpo_power_limit}W exceeds ${GXPO_MAX_POWER_W}W; refusing to launch." >&2
      echo "Ask an administrator to run: sudo nvidia-smi -i $GPU_IDS -pl $GXPO_MAX_POWER_W" >&2
      exit 2
    fi
  done <<< "$gxpo_power_limits"
  echo "GXPO power safety: physical GPU limits verified <= ${GXPO_MAX_POWER_W}W"
fi

TRAIN_FILES="['$DAPO_TRAIN','$LIGHTEVAL_TRAIN']"
VAL_FILES="['$MATH500','$AIME24','$AIME25','$AMC23','$MINERVA','$OLYMPIAD']"

if [[ "$OPD2_ON" -eq 1 ]]; then
  for _m in "$OPD2_TEACHER" "$OPD2_TEACHER_BASE"; do
    if [[ -z "$_m" ]]; then
      echo "PREFLIGHT FAIL: OPD2_ENABLED=1 requires OPD2_TEACHER and OPD2_TEACHER_BASE" >&2
      exit 2
    fi
    if [[ ! -f "$_m/config.json" ]]; then
      echo "PREFLIGHT FAIL: OPD^2 model not found at $_m (no config.json)" >&2
      exit 2
    fi
  done
  unset _m
  # OPD^2 assumes teacher, teacher_base and student share the base BPE vocab --
  # the student's sampled ids are scored directly by both frozen models. The
  # special-token block above the base vocab is allowed to differ in MEANING
  # (that is what the teacher chat-template render is for), but a differing base
  # vocab would silently score the wrong tokens.
  python - "$MODEL_ID" "$OPD2_TEACHER" "$OPD2_TEACHER_BASE" "$MAX_RESPONSE_LENGTH" <<'PY' || exit 2
import json, sys

def base_vocab(path):
    with open(f"{path}/tokenizer.json", encoding="utf-8") as f:
        return json.load(f)["model"]["vocab"]

student, teacher, teacher_base, max_resp = sys.argv[1:5]
sv = base_vocab(student)
for name, path in (("teacher", teacher), ("teacher_base", teacher_base)):
    v = base_vocab(path)
    if v != sv:
        raise SystemExit(
            f"PREFLIGHT FAIL: {name} base BPE vocab differs from the student "
            f"({len(v)} vs {len(sv)} entries or differing ids); OPD^2 needs a shared vocab.")
    # The frozen models score prompt + response in one HF forward; past
    # max_position_embeddings RoPE is out of range and the signal is silently
    # garbage (Qwen2.5-Math-* has only 4096). 1024 = data.max_prompt_length below.
    with open(f"{path}/config.json", encoding="utf-8") as f:
        max_pos = json.load(f).get("max_position_embeddings")
    if max_pos and 1024 + int(max_resp) > max_pos:
        raise SystemExit(
            f"PREFLIGHT FAIL: {name} has max_position_embeddings={max_pos}, but prompt 1024 + "
            f"MAX_RESPONSE_LENGTH {max_resp} exceeds it; set MAX_RESPONSE_LENGTH<={max_pos - 1024}.")
print(f"OPD^2 vocab check OK: {len(sv)} shared base BPE entries")
PY
  if [[ "${USE_KL_LOSS,,}" != "false" || "$KL_LOSS_COEF" != "0.0" ]]; then
    echo "PREFLIGHT FAIL: OPD^2 runs with the reference KL disabled (the paper's beta=0);" >&2
    echo "  set USE_KL_LOSS=False and KL_LOSS_COEF=0.0 (got $USE_KL_LOSS / $KL_LOSS_COEF)." >&2
    exit 2
  fi
  # Fail unless explicitly allowed: an entropy bonus's scale is not portable
  # from reward-RL to OPD^2. OPD^2's advantages are unnormalized (the reference
  # hardcodes disable_adv_norm) and land at std ~0.014, ~70x smaller than a
  # normalized GRPO advantage -- so entropy_loss*ENTROPY_COEFF becomes the
  # dominant term in policy_loss and the run just maximizes entropy. At 0.001
  # it drove gxpo-opd2 l4zbui0k / 4rwyfa3w from entropy 0.07 to 3.9 (gibberish).
  if [[ "$ENTROPY_COEFF" != "0" && "$ENTROPY_COEFF" != "0.0" ]]; then
    if [[ "${OPD2_ALLOW_ENTROPY:-0}" != "1" ]]; then
      echo "PREFLIGHT FAIL: ENTROPY_COEFF=$ENTROPY_COEFF with OPD^2 (reference uses 0)." >&2
      echo "  Set ENTROPY_COEFF=0, or OPD2_ALLOW_ENTROPY=1 to run it as a deliberate ablation." >&2
      exit 2
    fi
    echo "PREFLIGHT WARN: ENTROPY_COEFF=$ENTROPY_COEFF with OPD^2 (OPD2_ALLOW_ENTROPY=1)." >&2
  fi
  RETURN_RAW_CHAT=True
  OPD2_FLAGS+=(
    +actor_rollout_ref.actor.use_opd2=True
    +actor_rollout_ref.actor.opd2_teacher="$OPD2_TEACHER"
    +actor_rollout_ref.actor.opd2_teacher_base="$OPD2_TEACHER_BASE"
    +actor_rollout_ref.actor.opd2_topk="$OPD2_TOPK"
    +actor_rollout_ref.actor.opd2_gen_loss_weight="$OPD2_GEN_LOSS_WEIGHT"
    +actor_rollout_ref.actor.opd2_outcome_weight="$OPD2_OUTCOME_WEIGHT"
    +actor_rollout_ref.actor.opd2_rewards_bias="$OPD2_REWARDS_BIAS"
    +actor_rollout_ref.actor.opd2_teacher_template="$OPD2_TEACHER_TEMPLATE"
    +actor_rollout_ref.actor.opd2_micro_batch_size="$OPD2_MICRO_BATCH_SIZE"
    +actor_rollout_ref.actor.opd2_chunk_tokens="$OPD2_CHUNK_TOKENS"
    +actor_rollout_ref.actor.opd2_keep_on_gpu="$OPD2_KEEP_ON_GPU"
    +actor_rollout_ref.actor.opd2_attn_implementation="$OPD2_ATTN_IMPL"
    +actor_rollout_ref.actor.opd2_dedicated_gpu="$OPD2_DEDICATED_GPU"
  )
fi

METHOD_FLAGS=()
OPTIMIZER_FLAGS=()
case "${OPTIMIZER_NAME,,}" in
  adamw)
    OPTIMIZER_FLAGS+=(
      +actor_rollout_ref.actor.optim.weight_decay="$ADAMW_WEIGHT_DECAY"
    )
    ;;
  muon)
    OPTIMIZER_FLAGS+=(
      +actor_rollout_ref.actor.optim.muon_momentum="$MUON_MOMENTUM"
      +actor_rollout_ref.actor.optim.muon_ns_steps="$MUON_NS_STEPS"
      +actor_rollout_ref.actor.optim.muon_nesterov="$MUON_NESTEROV"
      +actor_rollout_ref.actor.optim.weight_decay="$MUON_WEIGHT_DECAY"
      +actor_rollout_ref.actor.optim.muon_distributed_backend="$MUON_DISTRIBUTED_BACKEND"
    )
    ;;
  *)
    echo "Unsupported OPTIMIZER_NAME=$OPTIMIZER_NAME (expected adamw or muon)" >&2
    exit 2
    ;;
esac
# Gate v2 (opt-in; see .audit/gxpo_algorithm_findings.md):
#   GXPO_TRIGGER_ROBUST=1   -> median/MAD z-score (resists early-warmup transient bursts)
#   GXPO_TRIGGER_MIN_OBS=N  -> gate cannot trip until N scored post-warmup observations
# Prediction-quality gating instead of the trainer entropy gate requires a custom
# entrypoint: gxpo_trigger_signal != entropy plus gxpo_shutoff_mode=cosine, which makes
# the actor gate on disagreement = 1 - |cos(g0, g_slow)|.

# Gate v2 passthroughs (defaults preserve the historical behavior exactly).
GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-entropy}"
GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-trajectory_aware}"
export GXPO_TRIGGER_SIGNAL GXPO_SHUTOFF_MODE

if [[ "$GXPO_SHUTOFF_MODE" == "cosine" && "$GXPO_TRIGGER_SIGNAL" == "entropy" ]]; then
  echo "WARNING: GXPO_SHUTOFF_MODE=cosine is INERT with GXPO_TRIGGER_SIGNAL=entropy:" >&2
  echo "         the trainer entropy gate makes the trip decision; cosine stats are logged only." >&2
fi
# Boolean footgun guard: ${VAR:+..} would fire on '0'/'false'; require an explicit yes.
case "${GXPO_TRIGGER_ROBUST:-}" in
  1|true|True|yes) METHOD_FLAGS+=(+actor_rollout_ref.actor.gxpo_trigger_robust=True) ;;
  ""|0|false|False|no) : ;;
  *) echo "WARNING: GXPO_TRIGGER_ROBUST='$GXPO_TRIGGER_ROBUST' not recognized; ignoring." >&2 ;;
esac
if [[ -n "${GXPO_TRIGGER_MIN_OBS:-}" && "$GXPO_TRIGGER_SIGNAL" == "entropy" ]]; then
  echo "WARNING: GXPO_TRIGGER_MIN_OBS only affects the actor-side gate; inert with signal=entropy." >&2
fi
if [[ -n "${GXPO_TRIGGER_ROBUST:-}" && "$GXPO_TRIGGER_SIGNAL" == "entropy" ]]; then
  echo "WARNING: GXPO_TRIGGER_ROBUST only affects the actor-side gate; inert with signal=entropy." >&2
fi
case "${GXPO_TRIGGER_ABS_THRESHOLD:-}" in
  ""|0|0.0) : ;;
  *) if [[ "$GXPO_TRIGGER_SIGNAL" == "entropy" || "$GXPO_SHUTOFF_MODE" != "cosine" ]]; then
       echo "WARNING: GXPO_TRIGGER_ABS_THRESHOLD only applies to cosine mode with signal!=entropy; inert here." >&2
     else
       METHOD_FLAGS+=(+actor_rollout_ref.actor.gxpo_trigger_abs_threshold="$GXPO_TRIGGER_ABS_THRESHOLD")
     fi ;;
esac
if [[ -n "${GXPO_TRIGGER_SUSTAIN_W:-}" ]]; then
  METHOD_FLAGS+=(+actor_rollout_ref.actor.gxpo_trigger_sustain_w="$GXPO_TRIGGER_SUSTAIN_W")
fi
if [[ -n "${GXPO_RELATIVE_THRESHOLD:-}" ]]; then
  METHOD_FLAGS+=(+actor_rollout_ref.actor.gxpo_relative_threshold="$GXPO_RELATIVE_THRESHOLD")
fi

case "$METHOD" in
  grpo)
    METHOD_FLAGS+=(+actor_rollout_ref.actor.use_gxpo=False)
    ;;
  sfpo)
    METHOD_FLAGS+=(
      +actor_rollout_ref.actor.use_sfpo=True
      +actor_rollout_ref.actor.sfpo_inner_steps="$K"
      +actor_rollout_ref.actor.sfpo_step_size="$REPOSITION_ALPHA"
      +actor_rollout_ref.actor.zscore_w=30
      +actor_rollout_ref.actor.zscore_threshold="$SFPO_ZSCORE_THRESHOLD"
      +actor_rollout_ref.actor.sfpo_warmup_steps="$SFPO_WARMUP_STEPS"
      +actor_rollout_ref.actor.sfpo_trigger_patience="$SFPO_TRIGGER_PATIENCE"
      +actor_rollout_ref.actor.sfpo_reset_entropy_after_warmup="$SFPO_RESET_ENTROPY_AFTER_WARMUP"
    )
    ;;
  gxpo)
    METHOD_FLAGS+=(
      +actor_rollout_ref.actor.use_gxpo=True
      # GXPO owns its trigger; disable the legacy trainer-side SFPO entropy gate.
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
      ${GXPO_TRIGGER_MIN_OBS:+\+actor_rollout_ref.actor.gxpo_trigger_min_obs="$GXPO_TRIGGER_MIN_OBS"}
      ${GXPO_MAX_ACTIVE_STEPS:+\+actor_rollout_ref.actor.gxpo_max_active_steps="$GXPO_MAX_ACTIVE_STEPS"}
    )
    ;;
esac

cat <<EOF
[fair comparison config]
model=$MODEL_ID
model_alias=$MODEL_ALIAS
method=$METHOD
K=$K
reposition_alpha=$REPOSITION_ALPHA
train_seed=$TRAIN_SEED
train_batch_size=$TRAIN_BATCH_SIZE
rollout_n=$ROLLOUT_N
learning_rate=$LR
lr_schedule=$LR_WARMUP_STYLE (warmup_ratio=$LR_WARMUP_RATIO, min_lr_ratio=$LR_MIN_RATIO)
opd2_enabled=$OPD2_ON$( [[ "$OPD2_ON" -eq 1 ]] && echo " (teacher=$OPD2_TEACHER, teacher_base=$OPD2_TEACHER_BASE, topk=$OPD2_TOPK, gen_loss_weight=$OPD2_GEN_LOSS_WEIGHT, outcome_weight=$OPD2_OUTCOME_WEIGHT, teacher_template=$OPD2_TEACHER_TEMPLATE, micro_bsz=$OPD2_MICRO_BATCH_SIZE)" )
sled_enabled=$SLED_ON$( [[ "$SLED_ON" -eq 1 ]] && echo " (loss_coef=$SLED_LOSS_COEF, grpo_coef=$SLED_GRPO_COEF, alpha=$SLED_ALPHA, early_layer=$SLED_EARLY_LAYER, topk=$SLED_TOPK, micro_bsz=$SLED_MICRO_BATCH_SIZE)" )
selected_policy_layers=${SELECTED_POLICY_LAYERS:-null}
multilayer_aux_coef=${MULTILAYER_AUX_COEF:-null}
multilayer_loss=$MULTILAYER_LOSS
loss_agg_mode=$LOSS_AGG_MODE
use_kl_loss=$USE_KL_LOSS
kl_loss_coef=$KL_LOSS_COEF
max_steps=$MAX_STEPS
save_freq=$SAVE_FREQ
sfpo_warmup_steps=$SFPO_WARMUP_STEPS
sfpo_zscore_threshold=$SFPO_ZSCORE_THRESHOLD
sfpo_trigger_patience=$SFPO_TRIGGER_PATIENCE
sfpo_reset_entropy_after_warmup=$SFPO_RESET_ENTROPY_AFTER_WARMUP
gxpo_warmup_steps=$GXPO_WARMUP_STEPS
gxpo_tau=$GXPO_TAU
gxpo_zscore_w=$GXPO_ZSCORE_W
gxpo_trigger_signal=$GXPO_TRIGGER_SIGNAL
gxpo_shutoff_mode=$GXPO_SHUTOFF_MODE
gxpo_optimizer_state_mode=$GXPO_OPTIMIZER_STATE_MODE
gxpo_retention_space=$GXPO_RETENTION_SPACE (run-name tag='${RETENTION_TAG:-none}')
optimizer=$OPTIMIZER_NAME
adamw_weight_decay=$ADAMW_WEIGHT_DECAY
muon_momentum=$MUON_MOMENTUM
muon_ns_steps=$MUON_NS_STEPS
muon_nesterov=$MUON_NESTEROV
muon_weight_decay=$MUON_WEIGHT_DECAY
muon_distributed_backend=$MUON_DISTRIBUTED_BACKEND
gxpo_trigger_robust=${GXPO_TRIGGER_ROBUST:-0}
gxpo_trigger_min_obs=${GXPO_TRIGGER_MIN_OBS:-0}
gxpo_max_active_steps=${GXPO_MAX_ACTIVE_STEPS:-0}
gxpo_trigger_patience=$GXPO_TRIGGER_PATIENCE
gxpo_fallback_mode=$GXPO_FALLBACK_MODE
gxpo_min_effective_multiplier=$GXPO_MIN_EFFECTIVE_MULTIPLIER
gxpo_fallback_window=$GXPO_FALLBACK_WINDOW
gxpo_actor_duty_cycle=$GXPO_ACTOR_DUTY_CYCLE
gxpo_diag_freq=$GXPO_DIAG_FREQ
gxpo_trigger_granularity=outer
dynamic_filtering=$GXPO_DYNAMIC_FILTERING (strategy=$GXPO_DYNAMIC_FILTERING_STRATEGY, p_easy=$GXPO_P_EASY, p_hard=$GXPO_P_HARD, target_zero_variance=$GXPO_TARGET_ZERO_VARIANCE, sampling_batch_size=$GXPO_SAMPLING_BATCH_SIZE)
validation_interval=5
validation_decoding=sampled temperature=0.7 do_sample=true n=$VAL_N
if [[ "$FINAL_EVAL_ENABLED" == "True" ]]; then
  final_decoding=stochastic temperature=1.0 top_p=0.7 do_sample=true n=4 seeds=$FINAL_EVAL_SEEDS
else
  final_decoding=disabled
fi
gpu_count=$GPU_COUNT
train_files=$TRAIN_FILES
validation_files=$VAL_FILES
run_dir=$RUN_DIR
EOF

python -u -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  +algorithm.norm_adv_by_std_in_grpo=False \
  +algorithm.use_kl_in_reward=False \
  data.train_files="$TRAIN_FILES" \
  data.val_files="$VAL_FILES" \
  data.train_batch_size="$TRAIN_BATCH_SIZE" \
  data.val_batch_size="$VAL_BATCH_SIZE" \
  +data.dynamic_filtering="$GXPO_DYNAMIC_FILTERING" \
  +data.dynamic_filtering_strategy="$GXPO_DYNAMIC_FILTERING_STRATEGY" \
  +data.p_easy="$GXPO_P_EASY" \
  +data.p_hard="$GXPO_P_HARD" \
  +data.target_zero_variance="$GXPO_TARGET_ZERO_VARIANCE" \
  +data.sampling_batch_size="$GXPO_SAMPLING_BATCH_SIZE" \
  data.max_prompt_length=1024 \
  data.max_response_length="$MAX_RESPONSE_LENGTH" \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  +data.seed="$TRAIN_SEED" \
  data.system_prompt="$SYSTEM_PROMPT" \
  data.return_raw_chat="$RETURN_RAW_CHAT" \
  +data.enable_thinking="$ENABLE_THINKING" \
  +data.dataloader_num_workers="$DATALOADER_NUM_WORKERS" \
  actor_rollout_ref.model.path="$MODEL_ID" \
  actor_rollout_ref.model.use_remove_padding=True \
  actor_rollout_ref.model.enable_gradient_checkpointing="$ENABLE_GRADIENT_CHECKPOINTING" \
  actor_rollout_ref.model.attn_implementation="$ATTN_IMPL" \
  +actor_rollout_ref.model.use_liger="$USE_LIGER" \
  actor_rollout_ref.actor.optim.lr="$LR" \
  actor_rollout_ref.actor.optim.warmup_style="$LR_WARMUP_STYLE" \
  actor_rollout_ref.actor.optim.lr_warmup_steps_ratio="$LR_WARMUP_RATIO" \
  actor_rollout_ref.actor.optim.min_lr_ratio="$LR_MIN_RATIO" \
  +actor_rollout_ref.actor.optim.name="$OPTIMIZER_NAME" \
  "${OPTIMIZER_FLAGS[@]}" \
  +actor_rollout_ref.actor.optim.fused="$OPTIM_FUSED" \
  actor_rollout_ref.actor.use_torch_compile="$USE_TORCH_COMPILE" \
  +actor_rollout_ref.actor.data_loader_seed="$TRAIN_SEED" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-16}" \
  actor_rollout_ref.actor.use_dynamic_bsz=True \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-24576}" \
  actor_rollout_ref.actor.clip_ratio=0.2 \
  actor_rollout_ref.actor.entropy_coeff="$ENTROPY_COEFF" \
  +actor_rollout_ref.actor.loss_agg_mode="$LOSS_AGG_MODE" \
  actor_rollout_ref.actor.grad_clip=1.0 \
  actor_rollout_ref.actor.use_kl_loss="$USE_KL_LOSS" \
  actor_rollout_ref.actor.kl_loss_coef="$KL_LOSS_COEF" \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE:-1}" \
  actor_rollout_ref.actor.fsdp_config.param_offload="$ACTOR_PARAM_OFFLOAD" \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="$ACTOR_OPTIMIZER_OFFLOAD" \
  ${ACTOR_MODEL_DTYPE:+\+actor_rollout_ref.actor.fsdp_config.model_dtype="$ACTOR_MODEL_DTYPE"} \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-8}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${TENSOR_PARALLEL_SIZE:-1}" \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEMORY_UTILIZATION:-0.5}" \
  actor_rollout_ref.rollout.max_num_batched_tokens="${VLLM_MAX_NUM_BATCHED_TOKENS:-98304}" \
  actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS:-1024}" \
  actor_rollout_ref.rollout.enable_chunked_prefill="${VLLM_ENABLE_CHUNKED_PREFILL:-True}" \
  actor_rollout_ref.rollout.attention_backend="${VLLM_ATTENTION_BACKEND:-FLASHINFER}" \
  actor_rollout_ref.rollout.n="$ROLLOUT_N" \
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-1.0}" \
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-1.0}" \
  actor_rollout_ref.rollout.val_kwargs.n="$VAL_N" \
  actor_rollout_ref.rollout.val_kwargs.do_sample="$VAL_DO_SAMPLE" \
  actor_rollout_ref.rollout.val_kwargs.temperature="$VAL_TEMPERATURE" \
  actor_rollout_ref.rollout.val_kwargs.top_p=1.0 \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu="${LOG_PROB_MICRO_BATCH_SIZE:-8}" \
  actor_rollout_ref.ref.fsdp_config.param_offload=True \
  reward_model.reward_manager=naive \
  algorithm.kl_ctrl.kl_coef=0.000 \
  trainer.critic_warmup=0 \
  trainer.logger="['console','wandb']" \
  trainer.project_name="$PROJECT" \
  trainer.experiment_name="$RUN_NAME" \
  trainer.default_local_dir="$RUN_DIR" \
  trainer.n_gpus_per_node="$GPU_COUNT" \
  trainer.nnodes=1 \
  trainer.save_freq="$SAVE_FREQ" \
  +trainer.keep_last_ckpts=1 \
  +trainer.keep_all_ckpts=False \
  trainer.resume_mode="$TRAINER_RESUME_MODE" \
  trainer.resume_from_path="$TRAINER_RESUME_FROM_PATH" \
  trainer.test_freq="$TRAINER_TEST_FREQ" \
  +trainer.validation_seeds='[0]' \
  +trainer.keep_last_validations=1 \
  +trainer.val_before_train="$VAL_BEFORE_TRAIN" \
  +trainer.max_steps="$MAX_STEPS" \
  trainer.total_training_steps="$MAX_STEPS" \
  trainer.total_epochs=100 \
  "${METHOD_FLAGS[@]}" \
  ${OPD2_FLAGS[@]+"${OPD2_FLAGS[@]}"} \
  ${SLED_FLAGS[@]+"${SLED_FLAGS[@]}"} \
  ${ML_FLAGS[@]+"${ML_FLAGS[@]}"} \
  2>&1 | tee "$RUN_DIR/train.log"

if [[ "$FINAL_EVAL_ENABLED" == "True" ]]; then
  TERMINAL_STEP="$MAX_STEPS"
  if [[ ! -d "$RUN_DIR/global_step_$TERMINAL_STEP" ]]; then
    TERMINAL_STEP="$(find "$RUN_DIR" -maxdepth 1 -type d -name 'global_step_*' -printf '%f\n' | sed 's/global_step_//' | sort -n | tail -1)"
  fi
  if [[ -z "$TERMINAL_STEP" || ! -d "$RUN_DIR/global_step_$TERMINAL_STEP" ]]; then
    echo "Training finished without a terminal checkpoint under $RUN_DIR" >&2
    exit 3
  fi

  python -u tools/evaluate_gxpo_terminal.py \
    --run-dir "$RUN_DIR" \
    --base-model "$MODEL_ID" \
    --data-files "$MATH500" "$AIME24" "$AIME25" "$AMC23" "$MINERVA" "$OLYMPIAD" \
    --seeds $FINAL_EVAL_SEEDS \
    --step "$TERMINAL_STEP" \
    --n 4 --temperature 1.0 --top-p 0.7 \
    --log-wandb \
    --wandb-project "$PROJECT" \
    --wandb-run "$RUN_NAME" \
    --wandb-id "$(cat "$RUN_DIR/wandb_id.txt" 2>/dev/null || true)"
else
  echo "Final evaluation disabled (FINAL_EVAL_ENABLED=False)."
fi
