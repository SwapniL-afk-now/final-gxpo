#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export SFT_MODEL="${SFT_MODEL:-/office/shared_cache/.cache/huggingface/hub/models--meta-llama--Llama-3.2-3B/snapshots/13afe5124825b4f3751f836b40dafda64c1ed062}"
export SFT_FSDP_STRATEGY="${SFT_FSDP_STRATEGY:-fsdp2}"
export SFT_FSDP_CPU_OFFLOAD="${SFT_FSDP_CPU_OFFLOAD:-False}"
export GPU_IDS="${GPU_IDS:-1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-$GPU_IDS}"
exec bash "$SCRIPT_DIR/sft_common.sh" gxpo "$@"
