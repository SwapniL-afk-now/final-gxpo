#!/usr/bin/env bash
# OFFLINE (off-policy) KD + GXPO, K=10 / alpha=0.3, locked 2-GPU (0,1).
#
# Thin wrapper over qwen25_math_1p5b_onpolicy_kd_gxpo.sh with
# KD_SOURCE=offpolicy_oss: fixed teacher-topk16 parquet
# (oss_low_kd/train_teacher_topk16.parquet), cached teacher, idle rollout
# engine that NEVER generates -- the offline counterpart of the previous
# offline KD run (results/gxpo_efficiency/qwen25-1p5b_offpolicy_kd_oss_b256_topk16).
#
# Note: the base script hard-sets forward KL for offpolicy fixed data
# ("forward KL's home turf"), matching the previous offline run
# (kd_reverse_kl=False in its log). No KD_REVERSE_KL export here on purpose.
#
# Validation mirrors the previous offline run: greedy (n=1, temp 0) on the 6
# benchmarks every 5 steps plus once before training; checkpoint every 5 steps
# (newest global_step_* + best_checkpoint kept). 200 steps.
#
# Launch only after GPUs 0 AND 1 are both free:
#   KD_TEACHER_PATH=<frozen 7B Math snapshot> bash qwen25_math_1p5b_offpolicy_kd_gxpo_k10_a03.sh
#   bash qwen25_math_1p5b_offpolicy_kd_gxpo_k10_a03.sh --dry-run   # preflight only
set -euo pipefail

export GPU_IDS="${GPU_IDS:-0,1}"
export KD_SOURCE=offpolicy_oss
export K=10
export REPOSITION_ALPHA=0.3
# Delegates to qwen25_math_1p5b_onpolicy_kd_gxpo.sh, which forwards this pin.
export GXPO_OPTIMIZER_STATE_MODE="${GXPO_OPTIMIZER_STATE_MODE:-transactional}"
# Mirror the previous offline run's teacher parallelism (its train.log shows
# num_replicas=1). Speed only, not math. Unset to use the base default of 2.
export KD_TEACHER_NUM_REPLICAS="${KD_TEACHER_NUM_REPLICAS:-1}"
export RUN_NAME="${RUN_NAME:-qwen25-1p5b_offpolicy_kd_gxpo_oss_k10_a03_b256_topk16}"
# Train + val data exactly as the previous offline run: teacher parquet from
# final-gxpo's data dir, 6-bench val from its eval_sft_6bench layout.
_FGXPO_DATA="${GXPO_DATA_ROOT:-/office/dev_workspace/swapnil/final-gxpo/Code/SFPO/data}"
export TRAIN_FILES="${TRAIN_FILES:-['$_FGXPO_DATA/oss_low_kd/train_teacher_topk16.parquet']}"
export MATH500="${MATH500:-$_FGXPO_DATA/eval_sft_6bench/math500.parquet}"
export AIME24="${AIME24:-$_FGXPO_DATA/eval_sft_6bench/aime24.parquet}"
export AIME25="${AIME25:-$_FGXPO_DATA/eval_sft_6bench/aime25.parquet}"
export AMC23="${AMC23:-$_FGXPO_DATA/eval_sft_6bench/amc23.parquet}"
export MINERVA="${MINERVA:-$_FGXPO_DATA/eval_sft_6bench/minervamath.parquet}"
export OLYMPIAD="${OLYMPIAD:-$_FGXPO_DATA/eval_sft_6bench/olympiadbench.parquet}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/qwen25_math_1p5b_onpolicy_kd_gxpo.sh" "$@"
