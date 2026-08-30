#!/usr/bin/env python3
"""Evaluate a plain Hugging Face SFT checkpoint on the six GXPO benchmarks."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = {
    "math500": "math500/test.parquet",
    "aime24": "aime2024/test.parquet",
    "aime25": "aime2025/test.parquet",
    "amc23": "amc/test.parquet",
    "minerva": "minervamath/test.parquet",
    "olympiadbench": "olympiadbench/test.parquet",
}


def find_step(run_dir: Path, requested: int | None) -> int:
    if requested is not None:
        return requested
    steps = []
    for path in run_dir.glob("global_step_*"):
        if path.is_dir():
            try:
                steps.append(int(path.name.removeprefix("global_step_")))
            except ValueError:
                pass
    if not steps:
        raise SystemExit(f"No global_step_N SFT checkpoints found under {run_dir}")
    return max(steps)


def evaluate_seed(llm, tokenizer, data_files, seed, n, temperature, top_p, max_tokens):
    import pandas as pd
    from verl.utils.reward_score import _default_compute_score
    from vllm import SamplingParams

    results = {}
    for benchmark, data_file in data_files.items():
        frame = pd.read_parquet(data_file)
        prompts = [
            tokenizer.apply_chat_template(
                list(prompt), tokenize=False, add_generation_prompt=True
            )
            for prompt in frame["prompt"]
        ]
        ground_truths = [row["ground_truth"] for row in frame["reward_model"]]
        data_sources = frame["data_source"].tolist()

        flat_prompts = []
        sampling_params = []
        for prompt_index, prompt in enumerate(prompts):
            for sample_index in range(n):
                flat_prompts.append(prompt)
                sampling_params.append(
                    SamplingParams(
                        n=1,
                        temperature=temperature,
                        top_p=top_p,
                        max_tokens=max_tokens,
                        seed=int(seed) * 100003 + prompt_index * n + sample_index,
                    )
                )

        outputs = llm.generate(flat_prompts, sampling_params, use_tqdm=True)
        correct = np.zeros((len(prompts), n), dtype=np.float32)
        truncated = 0
        for prompt_index in range(len(prompts)):
            for sample_index in range(n):
                output = outputs[prompt_index * n + sample_index].outputs[0]
                truncated += int(output.finish_reason == "length")
                score = _default_compute_score(
                    prompts[prompt_index],
                    data_sources[prompt_index],
                    output.text,
                    ground_truths[prompt_index],
                )
                correct[prompt_index, sample_index] = float(score) >= 0.95

        avg_at_n = float(correct.mean()) if correct.size else float("nan")
        pass_at_n = float(correct.max(axis=1).mean()) if len(correct) else float("nan")
        results[benchmark] = {
            "pass_at_1": avg_at_n,
            "avg_at_n": avg_at_n,
            "pass_at_n": pass_at_n,
            "truncated_fraction": float(truncated / max(correct.size, 1)),
            "rows": len(frame),
        }
        print(
            f"seed={seed} {benchmark}: pass@1={avg_at_n:.6f} "
            f"avg@{n}={avg_at_n:.6f} pass@{n}={pass_at_n:.6f}"
        )
    return results


def aggregate(per_seed, seeds, n):
    benchmarks = {}
    for benchmark in BENCHMARKS:
        values = [per_seed[str(seed)][benchmark] for seed in seeds]
        benchmarks[benchmark] = {
            "per_seed": {str(seed): per_seed[str(seed)][benchmark] for seed in seeds},
            "mean": {
                "pass_at_1": float(np.mean([value["pass_at_1"] for value in values])),
                "avg_at_n": float(np.mean([value["avg_at_n"] for value in values])),
                "pass_at_n": float(np.mean([value["pass_at_n"] for value in values])),
            },
            "std": {
                "pass_at_1": float(np.std([value["pass_at_1"] for value in values])),
                "avg_at_n": float(np.std([value["avg_at_n"] for value in values])),
                "pass_at_n": float(np.std([value["pass_at_n"] for value in values])),
            },
        }

    per_seed_average = {}
    for seed in seeds:
        values = [per_seed[str(seed)][benchmark]["pass_at_1"] for benchmark in BENCHMARKS]
        per_seed_average[str(seed)] = float(np.mean(values))
    benchmarks["avg_pass_at_1"] = {
        "per_seed": per_seed_average,
        "mean": float(np.mean(list(per_seed_average.values()))),
        "std": float(np.std(list(per_seed_average.values()))),
    }
    return benchmarks


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--n", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be positive")
    run_dir = args.run_dir.expanduser().resolve()
    step = find_step(run_dir, args.step)
    checkpoint = run_dir / f"global_step_{step}"
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"SFT checkpoint is not a plain HF directory: {checkpoint}")

    data_root = args.data_root.expanduser().resolve()
    data_files = {
        name: data_root / relative_path for name, relative_path in BENCHMARKS.items()
    }
    missing = [str(path) for path in data_files.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing benchmark parquet files:\n" + "\n".join(missing))

    from transformers import AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    llm = LLM(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.prompt_length + args.max_tokens,
        dtype="bfloat16",
    )
    per_seed = {
        str(seed): evaluate_seed(
            llm,
            tokenizer,
            data_files,
            seed,
            args.n,
            args.temperature,
            args.top_p,
            args.max_tokens,
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

    result = {
        "schema_version": 1,
        "kind": "sft_terminal_eval",
        "checkpoint_step": step,
        "checkpoint": str(checkpoint),
        "seeds": args.seeds,
        "data_root": str(data_root),
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": args.max_tokens,
            "n": args.n,
            "do_sample": True,
        },
        "benchmarks": aggregate(per_seed, args.seeds, args.n),
        "per_seed": per_seed,
    }
    result["metrics"] = {
        f"final_eval/{benchmark}_pass_at_1_mean": values["mean"]["pass_at_1"]
        for benchmark, values in result["benchmarks"].items()
        if benchmark != "avg_pass_at_1"
    }
    result["metrics"]["final_eval/avg_pass_at_1_mean"] = result["benchmarks"]["avg_pass_at_1"]["mean"]

    output = run_dir / "final_sft_eval.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    summary.update({"terminal_step": step, "final_sft_eval": result})
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
