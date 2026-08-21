#!/usr/bin/env python3
"""Evaluate terminal checkpoints with sampled pass@n and average@n.

This complements greedy pass@1 evaluation.  For each prompt, ``pass@n`` is
one when any of the n sampled completions is correct; ``average@n`` is the
fraction of correct completions.  The latter is useful for diagnosing why a
method's pass@n changed.
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

from tools.evaluate_greedy_5seeds import (  # noqa: E402
    BENCHMARK_ORDER,
    benchmark_key,
    download_grpo_actor,
    materialize_actor,
)


def evaluate_model(
    model_dir: Path,
    tokenizer_dir: str,
    data_files: list[str],
    seeds: list[int],
    n: int,
    temperature: float,
    top_p: float,
    max_tokens: int,
    gpu_memory_utilization: float,
) -> dict:
    import pandas as pd
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

    per_seed: dict[str, dict[str, dict[str, float]]] = {}
    for seed in seeds:
        sampling = SamplingParams(
            n=n,
            temperature=temperature,
            top_p=top_p,
            top_k=-1,
            max_tokens=max_tokens,
            seed=int(seed),
        )
        seed_metrics: dict[str, dict[str, float]] = {}
        for data_file in data_files:
            frame = pd.read_parquet(os.path.expanduser(data_file))
            prompts = [
                tokenizer.apply_chat_template(
                    list(chat), tokenize=False, add_generation_prompt=True
                )
                for chat in frame["prompt"]
            ]
            outputs = llm.generate(prompts, sampling)
            data_source = str(frame["data_source"].iloc[0])
            prompt_pass_n = []
            prompt_average_n = []
            for index, output in enumerate(outputs):
                ground_truth = frame["reward_model"].iloc[index]["ground_truth"]
                correct = [
                    float(
                        _default_compute_score(
                            prompts[index],
                            data_source,
                            candidate.text,
                            ground_truth,
                        )
                    )
                    >= 0.95
                    for candidate in output.outputs
                ]
                prompt_pass_n.append(float(any(correct)))
                prompt_average_n.append(float(np.mean(correct)))

            key = benchmark_key(data_file, data_source)
            seed_metrics[key] = {
                "pass_at_n": float(np.mean(prompt_pass_n)),
                "average_at_n": float(np.mean(prompt_average_n)),
            }
            print(
                f"seed={seed} {key}: pass@{n}={seed_metrics[key]['pass_at_n']:.6f} "
                f"average@{n}={seed_metrics[key]['average_at_n']:.6f}",
                flush=True,
            )
        observed_keys = [
            key for key in BENCHMARK_ORDER if key in seed_metrics
        ]
        if not observed_keys:
            raise RuntimeError("No recognized benchmark was evaluated")
        macro = float(
            np.mean([
                seed_metrics[key]["pass_at_n"] for key in observed_keys
            ])
        )
        seed_metrics["avg_pass_at_n"] = {"pass_at_n": macro}
        print(f"seed={seed} macro_pass@{n}={macro:.6f}", flush=True)
        per_seed[str(seed)] = seed_metrics

    del llm
    gc.collect()
    torch.cuda.empty_cache()

    benchmarks: dict[str, dict] = {}
    observed_keys = [
        key for key in BENCHMARK_ORDER if any(
            key in per_seed[str(seed)] for seed in seeds
        )
    ]
    for key in observed_keys:
        pass_values = [per_seed[str(seed)][key]["pass_at_n"] for seed in seeds]
        average_values = [per_seed[str(seed)][key]["average_at_n"] for seed in seeds]
        benchmarks[key] = {
            "pass_at_n": {
                "per_seed": {
                    str(seed): per_seed[str(seed)][key]["pass_at_n"]
                    for seed in seeds
                },
                "mean": float(np.mean(pass_values)),
                "std": float(np.std(pass_values)),
            },
            "average_at_n": {
                "per_seed": {
                    str(seed): per_seed[str(seed)][key]["average_at_n"]
                    for seed in seeds
                },
                "mean": float(np.mean(average_values)),
                "std": float(np.std(average_values)),
            },
        }

    macro_pass = [per_seed[str(seed)]["avg_pass_at_n"]["pass_at_n"] for seed in seeds]
    macro_average = [
        float(
            np.mean([
                per_seed[str(seed)][key]["average_at_n"]
                for key in observed_keys
            ])
        )
        for seed in seeds
    ]
    benchmarks["macro"] = {
        "pass_at_n": {
            "per_seed": {str(seed): value for seed, value in zip(seeds, macro_pass)},
            "mean": float(np.mean(macro_pass)),
            "std": float(np.std(macro_pass)),
        },
        "average_at_n": {
            "per_seed": {str(seed): value for seed, value in zip(seeds, macro_average)},
            "mean": float(np.mean(macro_average)),
            "std": float(np.std(macro_average)),
        },
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
    parser.add_argument("--n", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--grpo-repo", required=True)
    parser.add_argument("--grpo-step", type=int, required=True)
    parser.add_argument("--sfpo-run-dir", required=True)
    parser.add_argument("--sfpo-step", type=int, required=True)
    parser.add_argument("--gxpo-run-dir", required=True)
    parser.add_argument("--gxpo-step", type=int, required=True)
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=("grpo", "sfpo", "gxpo"),
        default=["grpo", "sfpo", "gxpo"],
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data_files = [str(Path(item).expanduser().resolve()) for item in args.data_files]
    for data_file in data_files:
        if not Path(data_file).is_file():
            raise SystemExit(f"Evaluation dataset not found: {data_file}")

    method_specs = {}
    if "grpo" in args.methods:
        grpo_source = download_grpo_actor(
            args.grpo_repo, args.grpo_step, output_dir / "grpo_hf_source"
        )
        method_specs["grpo"] = (grpo_source, args.grpo_step)
    method_specs["sfpo"] = (
        Path(args.sfpo_run_dir).expanduser().resolve()
        / f"global_step_{args.sfpo_step}"
        / "actor",
        args.sfpo_step,
    )
    method_specs["gxpo"] = (
        Path(args.gxpo_run_dir).expanduser().resolve()
        / f"global_step_{args.gxpo_step}"
        / "actor",
        args.gxpo_step,
    )
    methods = [
        (method, *method_specs[method]) for method in args.methods
    ]
    for method, actor_dir, step in methods:
        if not actor_dir.is_dir():
            raise SystemExit(f"{method} actor checkpoint not found: {actor_dir}")
        print(f"=== {method} step {step} ===", flush=True)
        with tempfile.TemporaryDirectory(
            prefix=f"sampled-{method}-", dir=output_dir
        ) as workdir:
            hf_dir = materialize_actor(actor_dir, Path(workdir))
            result = evaluate_model(
                hf_dir,
                args.base_model,
                data_files,
                args.seeds,
                args.n,
                args.temperature,
                args.top_p,
                args.max_tokens,
                args.gpu_memory_utilization,
            )
        result.update(
            {
                "method": method,
                "checkpoint_step": step,
                "checkpoint_actor": str(actor_dir),
                "decoding": {
                    "do_sample": args.temperature > 0,
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "n": args.n,
                    "max_tokens": args.max_tokens,
                    "primary_metric": f"pass@{args.n}",
                },
                "seeds": args.seeds,
                "std_definition": "population standard deviation (numpy.std, ddof=0)",
            }
        )
        (output_dir / f"{method}_sampled_{args.n}_5seed.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n"
        )

    summary = {
        method: json.loads(
            (output_dir / f"{method}_sampled_{args.n}_5seed.json").read_text()
        )
        for method in args.methods
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(f"wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
