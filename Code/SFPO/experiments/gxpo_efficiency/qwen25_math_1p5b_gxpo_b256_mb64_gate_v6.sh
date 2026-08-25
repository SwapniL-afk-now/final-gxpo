#!/usr/bin/env bash
set -euo pipefail

# Gate-v2 variant of the b256 1.5B run: same compute config as
# qwen25_math_1p5b_gxpo_b256_a05.sh (batch 256, minibatch 64, k=5,
# alpha 0.5, 2 GPUs) but driven by the prediction-quality gate instead
# of the trainer entropy gate.
#
# Evidence (.audit/gxpo_algorithm_findings.md): replaying production
# curves through the cosine gate trips the failing k10 run at ~126
# (never tripped under entropy) while leaving healthy runs untripped.
# Robust median/MAD + sigma floor resists the early-warmup transient
# bursts that caused all observed production trips.
#
# Knobs (all overridable):
export GXPO_TRIGGER_SIGNAL="${GXPO_TRIGGER_SIGNAL:-grad}"      # != entropy -> actor-side gate owns decisions
export GXPO_SHUTOFF_MODE="${GXPO_SHUTOFF_MODE:-cosine}"        # disagreement = 1 - |cos(g0, g_slow)|
export GXPO_TRIGGER_ROBUST="${GXPO_TRIGGER_ROBUST:-1}"         # median/MAD z-score with sigma floor
export GXPO_TRIGGER_MIN_OBS="${GXPO_TRIGGER_MIN_OBS:-10}"      # no trip in the first volatile window
export GXPO_MAX_ACTIVE_STEPS="${GXPO_MAX_ACTIVE_STEPS:-150}"   # hard runtime ceiling regardless of gate
# tau/patience inherit from common.sh unless overridden here:
export GXPO_TAU="${GXPO_TAU:-2.0}"
export GXPO_TRIGGER_PATIENCE="${GXPO_TRIGGER_PATIENCE:-2}"

exec bash "$(cd "$(dirname "${BASH_SOURCE[0]}")")/qwen25_math_1p5b_gxpo_b256_a05.sh"
