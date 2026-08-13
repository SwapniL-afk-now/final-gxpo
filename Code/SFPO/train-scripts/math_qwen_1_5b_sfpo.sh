#!/usr/bin/env bash
# SFPO rebuttal run with the same matched configuration as GXPO.
# Usage: GPU=1 ./math_qwen_1_5b_sfpo.sh
set -euo pipefail

GPU="${GPU:-1}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
METHOD=sfpo GPU="$GPU" exec "$SCRIPT_DIR/run_rebuttal.sh"
