#!/usr/bin/env python3
"""Five-seed greedy evaluation for the Qwen 1.5B efficiency checkpoints.

The seed is recorded and replayed exactly as requested, although greedy
decoding (temperature=0, n=1) is deterministic and should therefore produce
zero seed variance for a fixed checkpoint and dataset. Correctness uses the
same verl reward checker used during training.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCHMARK_ORDER = ("math500", "aime24", "aime25", "amc23", "minerva", "olympiadbench")
CORRECT_THRESHOLD = 0.95


def benchmark_key(data_file: str, source: str) -> str:
    text = f"{data_file} {source}".lower()
    aliases = {
        "math500": ("math500", "math-500"),
        "aime24": ("aime24", "aime_2024", "aime-2024"),
        "aime25": ("aime25", "aime_2025", "aime-2025"),
        "amc23": ("amc23", "amc"),
        "minerva": ("minerva",),
        "olympiadbench": ("olympiad",),
    }
    for key, candidates in aliases.items():
        if any(candidate in text for candidate in candidates):
            return key
    return source.replace("/", "_")


def link_tree(source: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        destination = target / item.name
        if not destination.exists() and not destination.is_symlink():
            destination.symlink_to(item)


def materialize_actor(actor_dir: Path, workdir: Path) -> Path:
    """Make an HF model in workdir without duplicating the source checkpoint."""
    actor = workdir / "actor"
    actor.mkdir()
    for item in actor_dir.iterdir():
        if item.name == "huggingface" or item.name.startswith("optim_"):
            continue
        (actor / item.name).symlink_to(item)
    source_hf = actor_dir / "huggingface"
    if not source_hf.is_dir():
        raise FileNotFoundError(f"Missing checkpoint metadata: {source_hf}")
    link_tree(source_hf, actor / "huggingface")

    hf_dir = actor / "huggingface"
    weight_files = list(hf_dir.glob("*.safetensors")) + list(hf_dir.glob("*.bin"))
    if not weight_files:
        merger = ROOT / "scripts" / "model_merger.py"
        subprocess.run([sys.executable, str(merger), "--local_dir", str(actor)], check=True)
    return hf_dir


def download_grpo_actor(repo_id: str, step: int, destination: Path) -> Path:
    from huggingface_hub import snapshot_download

    destination.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        local_dir=str(destination),
        allow_patterns=[
            f"global_step_{step}/actor/model_world_size_1_rank_0.pt",
            f"global_step_{step}/actor/extra_state_world_size_1_rank_0.pt",
            f"global_step_{step}/actor/huggingface/*",
        ],
    )
    actor = destination / f"global_step_{step}" / "actor"
    if not (actor / "model_world_size_1_rank_0.pt").is_file():
        raise FileNotFoundError(f"GRPO actor was not downloaded: {actor}")
    return actor


def evaluate_model(model_dir: Path, tokenizer_dir: str, data_files: list[str], seeds: list[int], max_tokens: int, gpu_memory_utilization: float) -> dict:
    import torch
    from transformers import AutoTokenizer
    from verl.utils.reward_score import _default_compute_score
    from vllm import LLM, SamplingParams

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    llm = LLM(
        model=str(model_dir),
        tokenizer=tokenizer_dir,
        tensor_parallel_size=1,
        gpu_memory_utilization=gpu_memory_utilization,
        dtype="bfloat16",
    )
    per_seed: dict[str, dict[str, float]] = {}
    for seed in seeds:
        sampling = SamplingParams(
            n=1,
            temperature=0.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=max_tokens,
            seed=int(seed),
        )
        seed_metrics: dict[str, float] = {}
        for data_file in data_files:
            frame = pd.read_parquet(os.path.expanduser(data_file))
            prompts = [
                tokenizer.apply_chat_template(list(chat), tokenize=False, add_generation_prompt=True)
                for chat in frame["prompt"]
            ]
            outputs = llm.generate(prompts, sampling)
            data_source = str(frame["data_source"].iloc[0])
            correct = []
            for index, output in enumerate(outputs):
                ground_truth = frame["reward_model"].iloc[index]["ground_truth"]
                score = _default_compute_score(prompts[index], data_source, output.outputs[0].text, ground_truth)
                correct.append(float(score) >= CORRECT_THRESHOLD)
            key = benchmark_key(data_file, data_source)
            seed_metrics[key] = float(np.mean(correct)) if correct else float("nan")
            print(f"seed={seed} {key}: greedy_pass@1={seed_metrics[key]:.6f}", flush=True)
        seed_metrics["avg_pass1"] = float(np.mean([seed_metrics[key] for key in BENCHMARK_ORDER]))
        per_seed[str(seed)] = seed_metrics
        print(f"seed={seed} macro_avg_pass@1={seed_metrics['avg_pass1']:.6f}", flush=True)

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    benchmarks: dict[str, dict] = {}
    for key in (*BENCHMARK_ORDER, "avg_pass1"):
        values = [row[key] for row in per_seed.values()]
        benchmarks[key] = {
            "per_seed": {seed: row[key] for seed, row in per_seed.items()},
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
    return {"per_seed": per_seed, "benchmarks": benchmarks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-files", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--grpo-repo", required=True)
    parser.add_argument("--grpo-step", type=int, default=295)
    parser.add_argument("--sfpo-run-dir", required=True)
    parser.add_argument("--sfpo-step", type=int, default=190)
    parser.add_argument("--gxpo-run-dir", required=True)
    parser.add_argument("--gxpo-step", type=int, default=190)
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_files = [str(Path(item).expanduser().resolve()) for item in args.data_files]
    for data_file in data_files:
        if not Path(data_file).is_file():
            raise SystemExit(f"Evaluation dataset not found: {data_file}")

    grpo_source = download_grpo_actor(args.grpo_repo, args.grpo_step, output_dir / "grpo_hf_source")
    methods = [
        ("grpo", grpo_source, args.grpo_step),
        ("sfpo", Path(args.sfpo_run_dir).expanduser().resolve() / f"global_step_{args.sfpo_step}" / "actor", args.sfpo_step),
        ("gxpo", Path(args.gxpo_run_dir).expanduser().resolve() / f"global_step_{args.gxpo_step}" / "actor", args.gxpo_step),
    ]
    for method, actor_dir, step in methods:
        if not actor_dir.is_dir():
            raise SystemExit(f"{method} actor checkpoint not found: {actor_dir}")
        print(f"=== {method} step {step} ===", flush=True)
        with tempfile.TemporaryDirectory(prefix=f"greedy-{method}-", dir=output_dir) as workdir:
            hf_dir = materialize_actor(actor_dir, Path(workdir))
            result = evaluate_model(hf_dir, args.base_model, data_files, args.seeds, args.max_tokens, args.gpu_memory_utilization)
        result.update({
            "method": method,
            "checkpoint_step": step,
            "checkpoint_actor": str(actor_dir),
            "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "n": 1, "max_tokens": args.max_tokens},
            "seeds": args.seeds,
            "std_definition": "population standard deviation (numpy.std, ddof=0)",
        })
        (output_dir / f"{method}_greedy_5seed.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(f"wrote {output_dir / f'{method}_greedy_5seed.json'}", flush=True)

    summary = {}
    for method in ("grpo", "sfpo", "gxpo"):
        summary[method] = json.loads((output_dir / f"{method}_greedy_5seed.json").read_text())["benchmarks"]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
