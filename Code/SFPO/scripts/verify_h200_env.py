#!/usr/bin/env python3
"""Synthetic, no-download verification for the repository's H200 environment."""

from __future__ import annotations

import importlib
import importlib.metadata as metadata
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
SFPO_ROOT = REPO_ROOT / "Code" / "SFPO"
if str(SFPO_ROOT) not in sys.path:
    sys.path.insert(0, str(SFPO_ROOT))

RESULTS: list[tuple[str, str, str]] = []


def check(name: str, function, mandatory: bool = True) -> None:
    try:
        detail = function() or "ok"
        RESULTS.append((name, "PASS", str(detail)))
    except Exception as exc:  # verification should report every check
        status = "FAIL" if mandatory else "SKIP"
        detail = f"{type(exc).__name__}: {str(exc).replace(chr(10), ' ')[:220]}"
        RESULTS.append((name, status, detail))


def package_version(name: str) -> str:
    return metadata.version(name)


def check_python() -> str:
    assert sys.version_info >= (3, 10), sys.version
    return f"{sys.version.split()[0]} ({sys.executable})"


def check_gpu() -> str:
    assert torch.cuda.is_available(), "torch.cuda.is_available() is false"
    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    assert capability >= (9, 0), capability
    x = torch.randn(8, 8, device="cuda", dtype=torch.bfloat16)
    assert torch.isfinite(x @ x.T).all()
    return f"{name}, SM{capability[0]}{capability[1]}, BF16"


def check_nvcc() -> str:
    output = subprocess.check_output(["nvcc", "--version"], text=True)
    assert "release 12." in output or "release 13." in output, output
    return next((line.strip() for line in output.splitlines() if "release" in line), "present")


def check_core_versions() -> str:
    expected = {
        "torch": "2.13.0+cu130",
        "triton": "3.7.1",
        "vllm": "0.27.1",
        "transformers": "5.15.0",
    }
    actual = {}
    for name, wanted in expected.items():
        module = importlib.import_module(name)
        value = getattr(module, "__version__", "unknown")
        actual[name] = value
        assert value == wanted, f"{name}: {value}, expected {wanted}"
    assert torch.version.cuda == "13.0", torch.version.cuda
    return ", ".join(f"{key}={value}" for key, value in actual.items())


def check_nccl_metadata() -> str:
    assert torch.distributed.is_nccl_available()
    return f"NCCL {torch.cuda.nccl.version()}"


def check_static_imports() -> str:
    modules = [
        "verl",
        "verl.workers.actor.dp_actor",
        "verl.workers.actor.gxpo_state",
        "verl.workers.fsdp_workers",
        "verl.workers.muon",
        "verl.workers.rollout.vllm_rollout",
        "verl.workers.rollout.vllm_rollout.vllm_rollout_spmd",
    ]
    for module in modules:
        importlib.import_module(module)
    return ", ".join(modules)


def check_vllm_integration() -> str:
    import vllm
    from verl.third_party.vllm import vllm_version
    from verl.workers.rollout.vllm_rollout import vllm_mode
    from verl.workers.sharding_manager.fsdp_vllm import FSDPVLLMShardingManager
    from vllm import LLM, SamplingParams
    from vllm.distributed import parallel_state

    assert vllm.__version__ == "0.27.1"
    assert vllm_mode == "spmd", vllm_mode
    assert LLM and SamplingParams and parallel_state
    assert FSDPVLLMShardingManager
    # This fork deliberately leaves vllm_version unset for modern releases;
    # its modern branches use `vllm_version not in old-version-tuples`.
    return f"vLLM {vllm.__version__}, rollout={vllm_mode}, fork-version-marker={vllm_version!r}"


def _fa3_module():
    try:
        from flash_attn_3 import flash_attn_interface
    except ImportError:
        import flash_attn_interface
    return flash_attn_interface


def check_attention_registry() -> str:
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

    assert "flash_attention_3" in ALL_ATTENTION_FUNCTIONS
    _fa3_module()
    return "Transformers registry contains flash_attention_3"


def _attention_tensors():
    shape = (1, 16, 4, 64)
    q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    return q, k, v


def check_fa2() -> str:
    from flash_attn import flash_attn_func
    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input

    q, k, v = _attention_tensors()
    output = flash_attn_func(q, k, v, dropout_p=0.0, causal=True)
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    assert index_first_axis and pad_input and unpad_input
    return f"{package_version('flash-attn')} BF16 forward/backward + bert_padding"


def check_fa3() -> str:
    interface = _fa3_module()
    q, k, v = _attention_tensors()
    output = interface.flash_attn_func(q, k, v, causal=True)
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    assert all(t.grad is not None and torch.isfinite(t.grad).all() for t in (q, k, v))
    return f"{package_version('flash-attn-3')} BF16 forward/backward"


def _tiny_qwen(attention: str = "flash_attention_3"):
    from transformers import Qwen2Config, Qwen2ForCausalLM

    config = Qwen2Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attn_implementation=attention,
    )
    config._attn_implementation = attention
    model = Qwen2ForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)
    return model


def check_fa3_checkpointing() -> str:
    model = _tiny_qwen()
    assert model.config._attn_implementation == "flash_attention_3"
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    assert model.is_gradient_checkpointing
    input_ids = torch.randint(0, 128, (2, 16), device="cuda")
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    return "tiny random Qwen2, FA3, BF16, checkpointing, use_reentrant=False"


def check_liger() -> str:
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

    model = _tiny_qwen("flash_attention_3")
    _apply_liger_kernel_to_instance(model=model)
    model.config.use_cache = False
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    input_ids = torch.randint(0, 128, (2, 16), device="cuda")
    loss = model(input_ids=input_ids, labels=input_ids).loss
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    return f"{package_version('liger-kernel')} tiny Qwen2 + FA3 + checkpointing"


def check_nccl_lifecycle() -> str:
    import torch.distributed as dist

    assert not dist.is_initialized()
    with tempfile.NamedTemporaryFile(prefix="gxpo-nccl-", delete=False) as handle:
        init_file = handle.name
    try:
        dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=0, world_size=1)
        value = torch.ones(1, device="cuda")
        dist.all_reduce(value)
        assert value.item() == 1
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        Path(init_file).unlink(missing_ok=True)
    return "world_size=1 all_reduce and clean destroy"


def check_fsdp() -> str:
    import torch.distributed as dist
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, MixedPrecision

    assert not dist.is_initialized()
    with tempfile.NamedTemporaryFile(prefix="gxpo-fsdp-", delete=False) as handle:
        init_file = handle.name
    try:
        dist.init_process_group("nccl", init_method=f"file://{init_file}", rank=0, world_size=1)
        model = _tiny_qwen("flash_attention_3")
        model.config.use_cache = False
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        wrapped = FSDP(
            model,
            device_id=torch.cuda.current_device(),
            use_orig_params=True,
            mixed_precision=MixedPrecision(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, buffer_dtype=torch.float32
            ),
        )
        input_ids = torch.randint(0, 128, (1, 8), device="cuda")
        wrapped(input_ids=input_ids, labels=input_ids).loss.backward()
        torch.nn.utils.clip_grad_norm_(wrapped.parameters(), 1.0)
        optimizer = torch.optim.AdamW(wrapped.parameters(), lr=1e-4)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
        Path(init_file).unlink(missing_ok=True)
    return "world_size=1, BF16 mixed precision, checkpointing, optimizer step"


def check_ray() -> str:
    import ray

    stale = os.environ.get("RAY_ADDRESS")
    if stale:
        print(f"WARNING: RAY_ADDRESS={stale!r} is set; local setup tests ignore it")
    # Ray 2.57 still consults RAY_ADDRESS when address=None.  Remove the
    # externally supplied cluster address for this isolated lifecycle probe,
    # then restore it after shutdown so the caller's environment is unchanged.
    os.environ.pop("RAY_ADDRESS", None)
    try:
        ray.init(address="local", num_gpus=1, include_dashboard=False, log_to_driver=False)

        @ray.remote
        def identity(value):
            return value

        assert ray.get(identity.remote(17)) == 17
    finally:
        ray.shutdown()
        if stale is not None:
            os.environ["RAY_ADDRESS"] = stale
    return f"Ray {package_version('ray')} local task and shutdown"


def check_remove_padding() -> str:
    from flash_attn.bert_padding import pad_input, unpad_input

    values = torch.randn(2, 8, 4, 16, device="cuda", dtype=torch.bfloat16)
    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]], device="cuda", dtype=torch.bool)
    # flash-attn 2.8 returns seqused as a fifth item for varlen callers.
    unpadded, indices, _, max_seqlen, _ = unpad_input(values, mask)
    restored = pad_input(unpadded, indices, values.shape[0], values.shape[1])
    assert restored.shape == values.shape
    assert torch.equal(restored[mask], values[mask])
    assert max_seqlen == 6
    return "unpad/pad shapes and valid-token reconstruction"


def check_dynamic_batching() -> str:
    from tensordict import TensorDict
    from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches

    mask = torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0], [1, 1, 1, 1, 1, 0, 0, 0], [1, 1, 1, 0, 0, 0, 0, 0], [1, 1, 1, 1, 1, 1, 0, 0]])
    batch = TensorDict({"attention_mask": mask, "input_ids": torch.arange(32).view(4, 8)}, batch_size=[4])
    micro_batches, indices = rearrange_micro_batches(batch, max_token_len=8)
    flat_indices = [idx for partition in indices for idx in partition]
    reverse = get_reverse_idx(flat_indices)
    assert len(micro_batches) == len(indices)
    assert sorted(flat_indices) == list(range(4))
    assert all(reverse[original] == position for position, original in enumerate(flat_indices))
    return f"{len(micro_batches)} balanced micro-batches and reverse index"


def check_muon() -> str:
    from verl.workers.muon import build_muon, zeropower_via_newtonschulz5
    from omegaconf import OmegaConf
    from torch import nn

    matrix = torch.randn(16, 8, device="cuda", dtype=torch.bfloat16)
    result = zeropower_via_newtonschulz5(matrix, steps=2)
    assert result.shape == matrix.shape and torch.isfinite(result).all()
    model = nn.Sequential(nn.Linear(8, 16), nn.Linear(16, 8)).cuda()
    optimizer = build_muon(model, OmegaConf.create({"lr": 1e-4}))
    model(torch.randn(2, 8, device="cuda")).sum().backward()
    optimizer.step()
    return "Newton-Schulz and build_muon routing"


def main() -> int:
    checks = [
        ("Python", check_python, True),
        ("GPU/BF16", check_gpu, True),
        ("CUDA toolkit/nvcc", check_nvcc, True),
        ("PyTorch/Triton/vLLM/Transformers", check_core_versions, True),
        ("NCCL availability", check_nccl_metadata, True),
        ("local verl/GXPO imports", check_static_imports, True),
        ("vLLM fork integration", check_vllm_integration, True),
        ("Transformers FA3 registry", check_attention_registry, True),
        ("FlashAttention-2 BF16", check_fa2, True),
        ("FlashAttention-3 BF16", check_fa3, True),
        ("FA3 + gradient checkpointing", check_fa3_checkpointing, True),
        ("Liger + FA3 + checkpointing", check_liger, True),
        ("NCCL world_size=1", check_nccl_lifecycle, True),
        ("FSDP world_size=1", check_fsdp, True),
        ("Ray lifecycle", check_ray, True),
        ("remove-padding path", check_remove_padding, True),
        ("dynamic batching path", check_dynamic_batching, True),
        ("Muon smoke", check_muon, True),
    ]
    for name, function, mandatory in checks:
        check(name, function, mandatory)

    print("\nH200 environment verification")
    print("=" * 100)
    print(f"{'CHECK':35} {'STATUS':8} DETAIL")
    print("-" * 100)
    for name, status, detail in RESULTS:
        print(f"{name:35} {status:8} {detail}")
    print("=" * 100)
    failed = [row for row in RESULTS if row[1] == "FAIL"]
    print(f"PASS={sum(row[1] == 'PASS' for row in RESULTS)} FAIL={len(failed)} SKIP={sum(row[1] == 'SKIP' for row in RESULTS)}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
