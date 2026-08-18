"""Low-overhead runtime helpers for the final GXPO efficiency runs.

This module intentionally contains only scalar/file helpers.  It never copies
model parameters or gradients to the driver and it does not synchronize CUDA.
"""
from __future__ import annotations

import importlib.metadata
import json
import math
import os
import subprocess
import time
from pathlib import Path
from typing import Any, Mapping

BENCHMARK_ORDER = ("math500", "aime24", "aime25", "amc23", "minerva", "olympiadbench")
BENCHMARK_DISPLAY = {
    "math500": "MATH-500",
    "aime24": "AIME24",
    "aime25": "AIME25",
    "amc23": "AMC23",
    "minerva": "Minerva",
    "olympiadbench": "OlympiadBench",
}
SOURCE_TO_BENCHMARK = {
    "HuggingFaceH4/MATH-500": "math500",
    "HuggingFaceH4/aime_2024": "aime24",
    "MathArena/aime_2025": "aime25",
    "AI-MO/aimo-validation-amc": "amc23",
    "math-ai/minervamath": "minerva",
    "math-ai/olympiadbench": "olympiadbench",
    # Useful aliases for already-prepared files from older experiment scripts.
    "math500": "math500",
    "aime24": "aime24",
    "aime25": "aime25",
    "amc23": "amc23",
    "minerva": "minerva",
    "olympiadbench": "olympiadbench",
}


def scalar(value: Any, default: float | None = None) -> float | None:
    """Convert a scalar/list-of-scalars metric to a finite Python float."""
    if isinstance(value, (list, tuple)):
        if not value:
            return default
        value = value[-1]
    try:
        value = value.item() if hasattr(value, "item") else value
        value = float(value)
    except (TypeError, ValueError):
        return default
    return value if math.isfinite(value) else default


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if hasattr(value, "item"):
        try:
            return json_safe(value.item())
        except Exception:
            pass
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_safe(value), indent=2, sort_keys=True) + "\n")


def append_jsonl(path: str | Path, value: Mapping[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(json_safe(value), sort_keys=True) + "\n")


def source_to_benchmark(source: Any) -> str | None:
    if source is None:
        return None
    source = str(source)
    if source in SOURCE_TO_BENCHMARK:
        return SOURCE_TO_BENCHMARK[source]
    lowered = source.lower()
    for key, benchmark in SOURCE_TO_BENCHMARK.items():
        if key.lower() in lowered or lowered in key.lower():
            return benchmark
    return None


def package_version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(repo_root: str | Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def gpu_inventory() -> dict[str, Any]:
    result: dict[str, Any] = {"count_visible": None, "devices": []}
    try:
        import torch
        result["count_visible"] = int(torch.cuda.device_count()) if torch.cuda.is_available() else 0
    except Exception:
        pass
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=3,
        )
        result["devices"] = [line.strip() for line in output.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        pass
    return result


def sample_gpu_telemetry() -> dict[str, float] | None:
    """Sample aggregate NVML data at caller-selected low frequency."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=utilization.gpu,power.draw", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
            timeout=2,
        )
        rows = []
        for line in output.splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 2:
                continue
            rows.append((float(parts[0]), float(parts[1])))
        if not rows:
            return None
        utils = [row[0] for row in rows]
        powers = [row[1] for row in rows]
        utils_sorted = sorted(utils)
        mid = len(utils_sorted) // 2
        p50 = utils_sorted[mid] if len(utils_sorted) % 2 else (utils_sorted[mid - 1] + utils_sorted[mid]) / 2.0
        p90 = utils_sorted[min(len(utils_sorted) - 1, int(math.ceil(0.90 * len(utils_sorted))) - 1)]
        return {
            "system/gpu_util_mean": sum(utils) / len(utils),
            "system/gpu_util_p50": p50,
            "system/gpu_util_p90": p90,
            "system/gpu_util_peak": max(utils),
            "system/gpu_power_mean_w": sum(powers) / len(powers),
            "system/gpu_power_peak_w": max(powers),
        }
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None


def resolved_package_versions() -> dict[str, Any]:
    versions = {
        "torch": package_version("torch"),
        "verl": package_version("verl"),
        "vllm": package_version("vllm"),
        "transformers": package_version("transformers"),
        "ray": package_version("ray"),
    }
    try:
        import torch
        versions["torch_cuda"] = torch.version.cuda
    except Exception:
        versions["torch_cuda"] = None
    return versions


def make_run_manifest(config: Any, run_dir: str | Path, repo_root: str | Path) -> dict[str, Any]:
    """Build a reproducibility manifest from the resolved Hydra config and env."""
    try:
        from omegaconf import OmegaConf
        resolved = OmegaConf.to_container(config, resolve=True)
    except Exception:
        resolved = config
    actor = resolved.get("actor_rollout_ref", {}).get("actor", {}) if isinstance(resolved, dict) else {}
    rollout = resolved.get("actor_rollout_ref", {}).get("rollout", {}) if isinstance(resolved, dict) else {}
    method = "gxpo" if actor.get("use_gxpo", False) else "sfpo" if actor.get("use_sfpo", False) else "grpo"
    manifest = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "run_dir": str(Path(run_dir).resolve()),
        "run_name": os.environ.get("GXPO_RUN_NAME", resolved.get("trainer", {}).get("experiment_name") if isinstance(resolved, dict) else None),
        "model_alias": os.environ.get("GXPO_MODEL_ALIAS"),
        "method": method,
        "k": actor.get("gxpo_k") if method == "gxpo" else actor.get("sfpo_inner_steps") if method == "sfpo" else None,
        "reposition_alpha": actor.get("gxpo_alpha") if method == "gxpo" else actor.get("sfpo_step_size") if method == "sfpo" else None,
        "train_seed": resolved.get("data", {}).get("seed", os.environ.get("TRAIN_SEED")) if isinstance(resolved, dict) else os.environ.get("TRAIN_SEED"),
        "gpu_count_configured": resolved.get("trainer", {}).get("n_gpus_per_node") if isinstance(resolved, dict) else None,
        "gpu": gpu_inventory(),
        "packages": resolved_package_versions(),
        "git_commit": git_commit(repo_root),
        "data": {
            "train_files": resolved.get("data", {}).get("train_files") if isinstance(resolved, dict) else None,
            "validation_files": resolved.get("data", {}).get("val_files") if isinstance(resolved, dict) else None,
        },
        "decoding": {
            "training": {"temperature": rollout.get("temperature"), "top_p": rollout.get("top_p"), "n": rollout.get("n")},
            "greedy_validation": {"temperature": rollout.get("val_kwargs", {}).get("temperature"), "top_p": rollout.get("val_kwargs", {}).get("top_p"), "do_sample": rollout.get("val_kwargs", {}).get("do_sample"), "n": rollout.get("val_kwargs", {}).get("n")},
            "final_stochastic": {"temperature": 1.0, "top_p": 0.7, "do_sample": True, "n": 4, "seeds": os.environ.get("FINAL_EVAL_SEEDS", "0 1 2 3")},
        },
        "hyperparameters": resolved,
        "instrumentation": {
            "policy_grad_eval_definition": "one full _backward_minibatch evaluation; SFPO/GXPO use actual executed control flow",
            "raw_backward_definition": "one loss.backward() invocation, including every gradient-accumulation microbatch",
            "headline_bp_speedup_uses": "policy_grad_evals",
            "headline_wall_speedup_uses": "cum_train_active_s",
            "validation": "greedy only during training; no interpolation",
            "offline_endpoint_diagnostic": "excluded",
        },
    }
    return manifest
