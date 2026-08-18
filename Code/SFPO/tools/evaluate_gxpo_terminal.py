#!/usr/bin/env python3
"""Stochastic terminal-checkpoint evaluation for the final GXPO runs.

This is intentionally separate from time-to-target: only the trainer's
T=0/n=1 validation history is used for efficiency matching.  This script is
quality reporting at the terminal checkpoint with independent decoding seeds.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCHMARK_ORDER = ("math500", "aime24", "aime25", "amc23", "minerva", "olympiadbench")


def evaluate_seed(llm, tokenizer, data_files, seed, n, temperature, top_p, max_tokens):
    import pandas as pd
    from verl.utils.reward_score import _default_compute_score
    from vllm import SamplingParams

    per_benchmark = {}
    sampling = SamplingParams(
        n=n,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        seed=int(seed),
    )
    for data_file in data_files:
        frame = pd.read_parquet(os.path.expanduser(data_file))
        prompts = [
            tokenizer.apply_chat_template(list(chat), tokenize=False, add_generation_prompt=True)
            for chat in frame["prompt"]
        ]
        outputs = llm.generate(prompts, sampling)
        values = []
        data_source = str(frame["data_source"].iloc[0])
        for row_index, output in enumerate(outputs):
            ground_truth = frame["reward_model"].iloc[row_index]["ground_truth"]
            correct = [
                float(_default_compute_score(
                    prompts[row_index], data_source, candidate.text, ground_truth
                )) >= 0.95
                for candidate in output.outputs
            ]
            values.append(float(np.mean(correct)))
        key = _benchmark_key(data_file, data_source)
        per_benchmark[key] = float(np.mean(values)) if values else float("nan")
        print(f"seed={seed} {key}: Pass@1(avg@{n})={per_benchmark[key]:.6f}")
    return per_benchmark


def _benchmark_key(data_file, source):
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-files", nargs="+", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    actor_dir = run_dir / f"global_step_{args.step}" / "actor"
    if not actor_dir.is_dir():
        raise SystemExit(f"Terminal actor checkpoint not found: {actor_dir}")
    for data_file in args.data_files:
        if not Path(data_file).expanduser().is_file():
            raise SystemExit(f"Evaluation dataset not found: {data_file}")

    from scripts.eval_checkpoints import materialize_hf_model
    from transformers import AutoTokenizer
    from vllm import LLM

    with tempfile.TemporaryDirectory(prefix="gxpo-terminal-") as workdir:
        model_dir = materialize_hf_model(str(actor_dir), args.base_model, workdir)
        tokenizer = AutoTokenizer.from_pretrained(args.base_model or model_dir)
        llm = LLM(
            model=model_dir,
            tokenizer=args.base_model or model_dir,
            tensor_parallel_size=args.tp,
            gpu_memory_utilization=args.gpu_memory_utilization,
            dtype="bfloat16",
        )
        per_seed = {
            str(seed): evaluate_seed(
                llm, tokenizer, args.data_files, seed, args.n,
                args.temperature, args.top_p, args.max_tokens,
            )
            for seed in args.seeds
        }
        del llm
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    benchmarks = {}
    for benchmark in BENCHMARK_ORDER:
        values = [row[benchmark] for row in per_seed.values() if benchmark in row]
        benchmarks[benchmark] = {
            "per_seed": {seed: row.get(benchmark) for seed, row in per_seed.items()},
            "mean": float(np.mean(values)) if values else None,
            "std": float(np.std(values)) if values else None,
        }
    per_seed_avg = {
        seed: float(np.mean([row[b] for b in BENCHMARK_ORDER if b in row]))
        for seed, row in per_seed.items()
        if any(b in row for b in BENCHMARK_ORDER)
    }
    benchmarks["avg_pass1"] = {
        "per_seed": per_seed_avg,
        "mean": float(np.mean(list(per_seed_avg.values()))) if per_seed_avg else None,
        "std": float(np.std(list(per_seed_avg.values()))) if per_seed_avg else None,
    }
    flat_metrics = {}
    for benchmark, values in benchmarks.items():
        flat_metrics[f"final_eval/{benchmark}_mean"] = values.get("mean")
        flat_metrics[f"final_eval/{benchmark}_std"] = values.get("std")
    result = {
        "schema_version": 1,
        "checkpoint_step": args.step,
        "checkpoint": str(actor_dir),
        "seeds": args.seeds,
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "do_sample": True,
            "n": args.n,
            "pass_metric": f"Pass@1(avg@{args.n})",
        },
        "benchmarks": benchmarks,
        "per_seed": per_seed,
        # Flat names make the local artifact directly usable by W&B/table
        # tooling while retaining the structured per-benchmark records.
        "metrics": flat_metrics,
    }
    output = run_dir / "final_stochastic_eval.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({
        "terminal_step": args.step,
        "final_eval_pending": False,
        "final_eval": result,
    })
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
