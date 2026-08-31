#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
GPU_IDS="${GPU_IDS:-0}"; export GPU_IDS
exec bash "$SCRIPT_DIR/sft_common.sh" baseline "$@"
