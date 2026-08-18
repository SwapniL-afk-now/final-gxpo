# H200 setup for GXPO

This repository keeps a customized, older verl fork. The H200 setup adds a
dedicated environment and GPU-extension checks without changing GXPO/GRPO
math, optimizer behavior, gradient checkpointing, or data loading.

The setup never downloads a dataset, a tokenizer, a pretrained checkpoint, or
any model weights. The only source downloads it performs are Python packages
and the official Dao-AILab FlashAttention source used to compile CUDA
extensions.

## Detected and selected stack

The validated machine used for this setup has one NVIDIA H200 NVL (SM90),
driver 595.71.05, CUDA toolkit 13.0, and Python 3.12.13. The base image's
binary stack is the source of truth:

| Component | H200 selection | Why |
| --- | --- | --- |
| PyTorch | 2.13.0+cu130 | Already supplied by the H200 image; the custom wheel is not resolved from PyPI |
| Torch CUDA | 13.0 | Matches the installed Torch build and toolkit |
| Triton | 3.7.1 | The Triton paired with the selected Torch |
| vLLM | 0.27.1 | Latest available image release and imports through this fork's SPMD path |
| Transformers | 5.15.0 | Registers `flash_attention_3` and is the selected vLLM dependency |
| FlashAttention-2 | 2.8.3.post1 | Required by `flash_attn.bert_padding` in the actor and remove-padding utilities |
| FlashAttention-3 | official commit `0251105a2fb19d2957484b7f023cd8c115286ced` | Hopper FA3 source; built with SM80 disabled and SM90a kernels |
| Liger Kernel | 0.8.1 | Triton fused model kernels; installed without dependency resolution |
| Ray | 2.57.0 | Matches the base image and passes a local lifecycle test |
| TensorDict | 0.6.2 | Retains the repository metadata/API contract |
| pyarrow | 22.0.0 | Current pinned project data dependency |

The legacy definitions are inconsistent: `requirements.txt` leaves most
versions open and pins Transformers 4.57.1; the older CUDA Dockerfiles use
Torch 2.4/CUDA 12.4, vLLM 0.6.3, and old FA2 pins. Installing either legacy
metadata file normally could replace the validated H200 binary stack. The
H200 files therefore install the overlay with `--no-deps` and keep the core
constraints visible in `constraints-h200.txt`; the local `pyproject.toml`
vLLM requirement is pinned to the validated 0.27.1 release.

## Setup and verification

From the repository root:

```bash
./Code/SFPO/scripts/setup_h200.sh
source .venv-h200/bin/activate
python Code/SFPO/scripts/verify_h200_env.py
PYTHONPATH=Code/SFPO python -m pytest -q Code/SFPO/tests/gxpo/
python -m pip check
```

`setup_h200.sh` creates or reuses `.venv-h200`, detects the GPU capability,
uses Python 3.12, installs the pinned overlay, installs this local fork
editable with `--no-deps`, builds FA2 for the detected architecture, builds
official Hopper FA3, installs Liger, and writes the two environment manifests
under `Code/SFPO/env/`.

Build parallelism is bounded by `MAX_JOBS` (default 8). Override it for a
smaller machine, for example `MAX_JOBS=4 ./Code/SFPO/scripts/setup_h200.sh`.
`--verify-only`, `--rebuild-flash`, `--skip-fa3-build`, and `--verbose` are
available. `--skip-fa3-build` is useful for dependency-only debugging, but the
verifier intentionally reports FA3 as mandatory and fails until it is built.

The container option is:

```bash
docker build --gpus all --ipc=host \
  -f Code/SFPO/docker/Dockerfile.h200 \
  -t gxpo-h200:local .
docker run --rm --gpus all --ipc=host gxpo-h200:local
```

The Dockerfile copies only the source/configuration needed by setup and tests;
it does not copy `raw-data`, datasets, model directories, checkpoints, or
caches.

## Attention backend selection

FA2 and FA3 serve different roles here. FA2 remains installed because
`dp_actor.py` and the remove-padding path directly import
`flash_attn.bert_padding`. FA3 is the Hopper attention computation backend.
The shared resolver in `verl/utils/attention.py` defaults to FA2 and validates
FA3's Transformers registry entry, official interface, and SM90+ GPU before a
model is constructed. A requested FA3 backend therefore fails loudly instead
of silently becoming SDPA/eager attention.

The direct config field is:

```yaml
actor_rollout_ref.model.attn_implementation: flash_attention_2
```

Set it to `flash_attention_3` for an H200 run. Existing
`override_config.attn_implementation` launchers remain accepted for
backwards compatibility. The final GXPO/GRPO/SFPO launchers expose
`ATTN_IMPL`, for example:

```bash
ATTN_IMPL=flash_attention_3 bash 'Code/SFPO/train-scripts/final run scripts/run_gxpo.sh'
```

The default remains FA2. Gradient checkpointing remains enabled in the
intended PPO/SFT configurations and uses `use_reentrant=False`; changing an
attention backend must not disable it.

## vLLM compatibility

The fork dispatches releases newer than 0.6.3 to
`verl.workers.rollout.vllm_rollout.vllm_rollout_spmd`. vLLM 0.27.1 imports with
that mode and the verifier checks `LLM`, `SamplingParams`,
`vllm.distributed.parallel_state`, and the fork's
`FSDPVLLMShardingManager`. A real engine/model-load test is intentionally not
run because this setup forbids downloading a model; use a user-provided local
model path for that final test.

## Ray and single-GPU distributed checks

The verifier warns if `RAY_ADDRESS` is set, starts a local one-GPU Ray task,
and shuts it down. It also initializes and destroys a world-size-one NCCL
process group and runs a tiny BF16 FSDP1 model with checkpointing, clipping,
AdamW, and one optimizer step. No Ray cluster or process group is left behind.

## Common extension failures

- `CUDA_ERROR_UNSUPPORTED_PTX_VERSION`: use a toolkit/wheel compatible with
  the host driver; do not replace the host NVIDIA driver.
- FA2/FA3 compiler OOM: lower `MAX_JOBS`; the setup defaults to 8 rather than
  using every host CPU.
- `flash_attention_3` registry/interface error: verify the selected
  Transformers/FA3 versions and rerun with `--rebuild-flash`.
- A legacy dependency tries to install Torch or vLLM: stop and use the H200
  files with `--no-deps`; never install the local fork through dependency
  resolution.
- `pip check` reports `pygobject` from the base image: that optional desktop
  package is unrelated to GXPO. On this image it also reports an NCCL metadata
  mismatch (Torch expects 2.29.7 while the installed distribution metadata says
  2.30.7); the runtime probe reports NCCL (2, 29, 7). Do not replace CUDA/NCCL
  binaries to silence metadata warnings; revalidate if the base image changes.
  Other core runtime conflicts must not be ignored.

## Cleanup

To remove only artifacts created by this setup, after stopping any verification
processes:

```bash
rm -rf .venv-h200 .h200-build .uv-cache
rm -f Code/SFPO/env/h200-freeze.txt Code/SFPO/env/h200-core-versions.txt
```

This cleanup does not touch system CUDA, the host driver, model storage, or
dataset storage.

Official references: [FlashAttention Hopper installation](https://github.com/Dao-AILab/flash-attention/tree/main/hopper),
[Transformers attention interface](https://huggingface.co/docs/transformers/main/attention_interface),
and [vLLM documentation](https://docs.vllm.ai/).
