#!/usr/bin/env python3
"""Greedy MATH-500/AIME evaluation for the math-distillation SFT path."""

from __future__ import annotations

import argparse
import gc
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

BENCHMARKS = ("math500", "aime24", "aime25")
BENCHMARK_CANDIDATES = {
    "math500": ("math500.parquet", "math500/test.parquet", "math-500.parquet"),
    "aime24": ("aime2024.parquet", "aime24.parquet", "aime2024/test.parquet"),
    "aime25": ("aime2025.parquet", "aime25.parquet", "aime2025/test.parquet"),
}


def resolve_benchmark_files(data_root: Path) -> dict[str, Path]:
    files = {}
    for name, candidates in BENCHMARK_CANDIDATES.items():
        path = next((data_root / item for item in candidates if (data_root / item).is_file()), None)
        if path is None:
            raise SystemExit(f"Missing {name} under {data_root}; tried {candidates}")
        files[name] = path
    return files


def _messages(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)) and all(isinstance(item, dict) for item in value):
        return list(value)
    return [{"role": "user", "content": str(value)}]


def _ground_truth(row):
    reward = row.get("reward_model", {})
    if hasattr(reward, "tolist"):
        reward = reward.tolist()
    if isinstance(reward, dict) and "ground_truth" in reward:
        return reward["ground_truth"]
    if "ground_truth" in row:
        return row["ground_truth"]
    raise ValueError("Math benchmark row has no ground_truth")


def evaluate_benchmark(llm, tokenizer, frame, benchmark: str, max_tokens: int, max_examples: int, prompt_length: int):
    import pandas as pd
    from verl.utils.reward_score import math_verify
    from vllm import SamplingParams

    if max_examples > 0:
        frame = frame.head(max_examples)
    prompt_texts = [
        tokenizer.apply_chat_template(_messages(value), tokenize=False, add_generation_prompt=True)
        for value in frame["prompt"]
    ]
    prompt_ids = [
        tokenizer(text, add_special_tokens=False, truncation=True, max_length=prompt_length)["input_ids"]
        for text in prompt_texts
    ]
    outputs = llm.generate(
        [{"prompt_token_ids": ids} for ids in prompt_ids],
        SamplingParams(n=1, temperature=0.0, top_p=1.0, max_tokens=max_tokens),
        use_tqdm=True,
    )
    correct = []
    truncated = 0
    for row, output in zip(frame.to_dict("records"), outputs):
        completion = output.outputs[0]
        truncated += int(completion.finish_reason == "length")
        correct.append(float(math_verify.compute_score(completion.text, _ground_truth(row))) == 1.0)
    value = float(np.mean(correct)) if correct else float("nan")
    print(f"{benchmark}: greedy_pass@1={value:.6f} rows={len(correct)}", flush=True)
    return {
        "pass_at_1": value,
        "greedy_pass_at_1": value,
        "rows": len(correct),
        "truncated_fraction": truncated / max(len(correct), 1),
    }


def add_workspace_cuda_libs():
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--step", type=int, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.70)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--verifier-workers", type=int, default=0, help="Compatibility option; math verification is local and synchronous")
    args = parser.parse_args()
    if args.n != 1 or args.temperature != 0.0 or args.top_p != 1.0:
        raise SystemExit("Math distillation evaluation is fixed to greedy n=1, temperature=0, top_p=1")
    if args.data_parallel_size != 1:
        raise SystemExit("Math distillation evaluator uses one student vLLM process")

    run_dir = args.run_dir.expanduser().resolve()
    checkpoint = run_dir / f"global_step_{args.step}"
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"Missing Hugging Face checkpoint: {checkpoint}")
    data_files = resolve_benchmark_files(args.data_root.expanduser().resolve())

    add_workspace_cuda_libs()
    import pandas as pd
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model_config = AutoConfig.from_pretrained(str(checkpoint))
    model_context = getattr(model_config, "max_position_embeddings", None)
    effective_tokens = args.max_tokens
    if model_context is not None:
        effective_tokens = min(effective_tokens, int(model_context) - args.prompt_length)
    llm = LLM(
        model=str(checkpoint), tokenizer=str(checkpoint),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.prompt_length + effective_tokens,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        dtype="bfloat16",
        enforce_eager=os.environ.get("VLLM_ENFORCE_EAGER", "0").lower() in {"1", "true", "yes", "on"},
        attention_config={"backend": "FLASHINFER"},
    )
    try:
        frame_results = {}
        for benchmark, path in data_files.items():
            frame_results[benchmark] = evaluate_benchmark(
                llm, tokenizer, pd.read_parquet(path), benchmark,
                effective_tokens, args.max_examples, args.prompt_length,
            )
    finally:
        engine = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if getattr(engine, "shutdown", None):
            engine.shutdown()
        del llm
        gc.collect()

    average = float(np.mean([frame_results[name]["greedy_pass_at_1"] for name in BENCHMARKS]))
    result = {
        "schema_version": 1,
        "kind": "math_distillation_greedy_eval",
        "checkpoint_step": args.step,
        "checkpoint": str(checkpoint),
        "data_root": str(args.data_root),
        "seeds": [0],
        "decoding": {
            "do_sample": False, "temperature": 0.0, "top_p": 1.0,
            "n": 1, "max_tokens": effective_tokens, "mode": "greedy",
        },
        "benchmarks": {
            **frame_results,
            "average_greedy_pass_at_1": {"mean": average, "std": 0.0},
        },
        "metrics": {
            **{
                f"eval/{name}/greedy_pass_at_1": frame_results[name]["greedy_pass_at_1"]
                for name in BENCHMARKS
            },
            "eval/average_greedy_pass_at_1": average,
        },
    }
    output = run_dir / "final_sft_eval.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
