#!/usr/bin/env bash
set -euo pipefail

# Reproducible, local-only H200 setup for this customized verl/GXPO fork.
# This script never downloads a model or dataset. The validated Torch/vLLM
# binary stack is intentionally treated as a base-image contract because the
# image ships a custom Torch 2.13.0+cu130 build that is not published on PyPI.

usage() {
    cat <<'EOF'
Usage: setup_h200.sh [options]

Options:
  --verify-only       Run verification without installing anything.
  --rebuild-flash     Rebuild FA2 and FA3 even when imports already work.
  --skip-fa3-build    Install/check the rest but leave FA3 for a later build.
  --verbose           Print each setup command.
EOF
}

VERIFY_ONLY=0
REBUILD_FLASH=0
SKIP_FA3_BUILD=0
VERBOSE=0
while (($#)); do
    case "$1" in
        --verify-only) VERIFY_ONLY=1 ;;
        --rebuild-flash) REBUILD_FLASH=1 ;;
        --skip-fa3-build) SKIP_FA3_BUILD=1 ;;
        --verbose) VERBOSE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ((VERBOSE)); then
    set -x
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../../.." && pwd)"
SFPO_ROOT="$REPO_ROOT/Code/SFPO"
VENV="$REPO_ROOT/.venv-h200"
PYTHON="$VENV/bin/python"
UV_CACHE_DIR="${UV_CACHE_DIR:-}"
if [[ -z "$UV_CACHE_DIR" || ! -d "$UV_CACHE_DIR" || ! -w "$UV_CACHE_DIR" ]]; then
    UV_CACHE_DIR="$REPO_ROOT/.uv-cache"
fi
FA3_REF="${FA3_REF:-0251105a2fb19d2957484b7f023cd8c115286ced}"
FA3_SRC="$REPO_ROOT/.h200-build/flash-attention-$FA3_REF"
FA3_WHEELS="$REPO_ROOT/.h200-build/fa3-wheels"

die() { echo "H200 setup error: $*" >&2; exit 1; }

if ! command -v uv >/dev/null 2>&1; then
    die "uv is required. Install uv with the official installer, then rerun this script."
fi

if ((VERIFY_ONLY)); then
    [[ -x "$PYTHON" ]] || die "$VENV does not exist; run setup_h200.sh first"
    exec "$PYTHON" "$SFPO_ROOT/scripts/verify_h200_env.py"
fi

export UV_CACHE_DIR
mkdir -p "$UV_CACHE_DIR" "$REPO_ROOT/.h200-build"

if [[ ! -x "$PYTHON" ]]; then
    uv venv "$VENV" --python 3.12 --system-site-packages
fi

"$PYTHON" - <<'PY'
import sys
import torch

expected = {
    "torch": "2.13.0+cu130",
    "triton": "3.7.1",
    "vllm": "0.27.1",
    "transformers": "5.15.0",
}
print(f"Python: {sys.version.split()[0]} ({sys.executable})")
for package, wanted in expected.items():
    module = __import__(package)
    actual = getattr(module, "__version__", "unknown")
    if actual != wanted:
        raise SystemExit(
            f"Validated H200 core mismatch: {package}={actual}, expected {wanted}. "
            "Install a matching H200 base image; this script will not replace CUDA binaries."
        )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")
name = torch.cuda.get_device_name(0)
cc = torch.cuda.get_device_capability(0)
print(f"GPU: {name}; compute capability: {cc[0]}.{cc[1]}")
if cc < (9, 0):
    raise SystemExit("This setup is for Hopper (SM90+) and did not detect a compatible GPU")
PY

command -v nvidia-smi >/dev/null || die "nvidia-smi is unavailable"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
command -v nvcc >/dev/null || die "nvcc is required to compile FA2/FA3"
nvcc --version | tail -1

GPU_ARCH="$($PYTHON - <<'PY'
import torch
major, minor = torch.cuda.get_device_capability(0)
print(f"{major}{minor}")
PY
)"
MAX_JOBS="${MAX_JOBS:-8}"
echo "CUDA extension build: SM${GPU_ARCH}, MAX_JOBS=${MAX_JOBS}"

# Install the overlay without dependency resolution: resolving the published
# packages would try to replace the image's custom Torch wheel. FA2 is filtered
# here so its explicit build below receives the detected architecture flags.
OVERLAY_REQ="$REPO_ROOT/.h200-build/requirements-overlay.txt"
grep -Ev '^(flash-attn|liger-kernel)==' "$SFPO_ROOT/requirements-h200.txt" > "$OVERLAY_REQ"
uv pip install --python "$PYTHON" --no-deps -r "$OVERLAY_REQ"
uv pip install --python "$PYTHON" --no-deps \
    orjson==3.11.7 latex2sympy2-extended==1.10.2 \
    antlr4-python3-runtime==4.9.3 multiprocess==0.70.18 \
    xxhash==3.6.0 GitPython==3.1.46 gitdb==4.0.12 \
    pytz==2025.2 tzdata==2025.2 fsspec[http]==2025.10.0
uv pip install --python "$PYTHON" --no-deps --editable "$SFPO_ROOT"

if ((REBUILD_FLASH)) || ! "$PYTHON" -c 'import flash_attn' >/dev/null 2>&1; then
    MAX_JOBS="$MAX_JOBS" FLASH_ATTENTION_FORCE_BUILD=TRUE \
        FLASH_ATTN_CUDA_ARCHS="$GPU_ARCH" \
        uv pip install --python "$PYTHON" --no-build-isolation --no-deps \
        flash-attn==2.8.3.post1
fi

if ((SKIP_FA3_BUILD == 0)) && { ((REBUILD_FLASH)) || ! "$PYTHON" -c 'import flash_attn_interface' >/dev/null 2>&1; }; then
    command -v git >/dev/null || die "git is required to build FA3"
    if [[ ! -d "$FA3_SRC/.git" ]]; then
        rm -rf "$FA3_SRC"
        git clone --recursive --depth 1 https://github.com/Dao-AILab/flash-attention.git "$FA3_SRC"
        git -C "$FA3_SRC" fetch --depth 1 origin "$FA3_REF"
        git -C "$FA3_SRC" checkout --detach "$FA3_REF"
        git -C "$FA3_SRC" submodule update --init --recursive
    fi
    if ((REBUILD_FLASH)); then
        rm -rf "$FA3_SRC/hopper/build" "$FA3_WHEELS"
    fi
    mkdir -p "$FA3_WHEELS"
    (cd "$FA3_SRC/hopper" && \
        MAX_JOBS="$MAX_JOBS" FLASH_ATTENTION_FORCE_BUILD=TRUE \
        FLASH_ATTENTION_SKIP_CUDA_BUILD=FALSE FLASH_ATTENTION_DISABLE_SM80=TRUE \
        "$PYTHON" -m pip wheel --no-build-isolation --no-deps . -w "$FA3_WHEELS")
    FA3_WHEEL="$(find "$FA3_WHEELS" -maxdepth 1 -type f -name 'flash_attn_3-*.whl' -print -quit)"
    [[ -n "$FA3_WHEEL" ]] || die "FA3 wheel was not produced"
    uv pip install --python "$PYTHON" --no-deps "$FA3_WHEEL"
fi

# Liger is Triton-based and is installed without dependency resolution so it
# cannot pull a different Torch/Triton pair over the validated core.
if ! "$PYTHON" -c 'import liger_kernel' >/dev/null 2>&1; then
    uv pip install --python "$PYTHON" --no-deps liger-kernel==0.8.1
fi

"$PYTHON" "$SFPO_ROOT/scripts/verify_h200_env.py"

mkdir -p "$SFPO_ROOT/env"
"$PYTHON" - <<PY
from importlib.metadata import version
from pathlib import Path
import subprocess
import sys
import torch

root = Path(${SFPO_ROOT@Q})
env = root / "env"
freeze = subprocess.check_output([${PYTHON@Q}, "-m", "pip", "freeze", "--all"], text=True)
(env / "h200-freeze.txt").write_text("# Generated by setup_h200.sh\n" + "\n".join(sorted(freeze.splitlines())) + "\n")
driver = subprocess.check_output(["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()
nvcc = subprocess.check_output(["nvcc", "--version"], text=True).split("release ", 1)[-1].split(",", 1)[0]
lines = [
    "# H200 core versions recorded after verification",
    f"Python: {sys.version.split()[0]}",
    f"GPU: {torch.cuda.get_device_name(0)}",
    f"Compute capability: {'.'.join(map(str, torch.cuda.get_device_capability(0)))}",
    f"NVIDIA driver: {driver}",
    f"CUDA toolkit: {nvcc}",
]
for name in ["torch", "triton", "vllm", "transformers", "flash-attn", "flash-attn-3", "liger-kernel", "ray", "tensordict", "pyarrow"]:
    try:
        lines.append(f"{name}: {version(name)}")
    except Exception as exc:
        lines.append(f"{name}: unavailable ({exc})")
(env / "h200-core-versions.txt").write_text("\n".join(lines) + "\n")
PY

echo "H200 environment is ready. Activate with:"
echo "  source $VENV/bin/activate"
echo "Verify with:"
echo "  $PYTHON $SFPO_ROOT/scripts/verify_h200_env.py"
