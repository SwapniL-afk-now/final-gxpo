#!/usr/bin/env bash
# Compatibility wrapper for the local Qwen2.5-Math SFT audit entrypoint.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export GPU_IDS="${GPU_IDS:-${GPU:-2}}"
exec bash "$SCRIPT_DIR/../experiments/gxpo_efficiency/qwen2.5_1.5b_sft_baseline.sh" "$@"
