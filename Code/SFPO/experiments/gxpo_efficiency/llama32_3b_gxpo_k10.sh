#!/usr/bin/env bash
set -euo pipefail
MODEL_ALIAS="llama32-3b"
MODEL_ID="${MODEL_LLAMA32_3B:-meta-llama/Llama-3.2-3B}"
METHOD="gxpo"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
