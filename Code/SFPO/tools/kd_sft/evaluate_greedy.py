#!/usr/bin/env python3
"""Greedy benchmark evaluation for a flat offline KD-SFT checkpoint."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


BENCHMARK_ORDER = ("math500", "aime24", "aime25", "amc23", "minerva", "olympiadbench")


def benchmark_key(path, source):
    text = f"{path} {source}".lower()
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint-dir", required=True)
    ap.add_argument("--base-model", required=True,
                    help="student tokenizer/config; checkpoint contains flat HF weights")
    ap.add_argument("--data-files", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--tp", type=int, default=1)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"flat HF checkpoint not found: {checkpoint}/config.json")
    for data_file in args.data_files:
        if not Path(data_file).expanduser().is_file():
            raise SystemExit(f"evaluation dataset not found: {data_file}")

    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score import _default_compute_score

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    llm = LLM(model=str(checkpoint), tokenizer=args.base_model,
              tensor_parallel_size=args.tp,
              gpu_memory_utilization=args.gpu_memory_utilization, dtype="bfloat16",
              trust_remote_code=True)
    sampling = SamplingParams(n=args.n, temperature=args.temperature,
                              top_p=args.top_p, max_tokens=args.max_tokens,
                              seed=args.seed)
    per_benchmark = {}
    for data_file in args.data_files:
        frame = pd.read_parquet(data_file)
        prompts = [tokenizer.apply_chat_template(
            list(chat), tokenize=False, add_generation_prompt=True) for chat in frame["prompt"]]
        outputs = llm.generate(prompts, sampling)
        source = str(frame["data_source"].iloc[0])
        values = []
        for row_index, output in enumerate(outputs):
            truth = frame["reward_model"].iloc[row_index]["ground_truth"]
            correct = [float(_default_compute_score(
                prompts[row_index], source, candidate.text, truth)) >= 0.95
                       for candidate in output.outputs]
            values.append(float(np.mean(correct)))
        key = benchmark_key(data_file, source)
        per_benchmark[key] = float(np.mean(values)) if values else float("nan")
        print(f"seed={args.seed} {key}: Pass@1(avg@{args.n})={per_benchmark[key]:.6f}", flush=True)

    result = {
        "schema_version": 1,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                      "n": args.n, "max_tokens": args.max_tokens},
        "benchmarks": per_benchmark,
        "avg_pass1": float(np.mean(list(per_benchmark.values()))) if per_benchmark else None,
    }
    output = Path(args.output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
