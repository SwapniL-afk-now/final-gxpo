#!/usr/bin/env python3
"""Evaluate a plain HF SFT checkpoint on K&K IID and OOD test splits."""
from __future__ import annotations

import argparse
import gc
import json
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from evaluate_sft_terminal import add_workspace_cuda_libs


GROUP_FILES = {
    "iid_3ppl": "iid_test.parquet",
    "iid_4ppl": "iid_test.parquet",
    "iid_5ppl": "iid_test.parquet",
    "iid_6ppl": "iid_test.parquet",
    "ood_7ppl": "ood_test.parquet",
    "ood_8ppl": "ood_test.parquet",
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


def extract_label(text: str, name: str) -> str | None:
    pattern = re.compile(
        rf"\b{re.escape(name)}\b\s*(?:is|:|-|=)\s*(?:a\s+)?(knight|knave)\b",
        re.IGNORECASE,
    )
    matches = pattern.findall(text)
    return matches[-1].lower() if matches else None


def correct(text: str, ground_truth: str) -> bool:
    truth = json.loads(ground_truth) if isinstance(ground_truth, str) else ground_truth
    return all(extract_label(text, name) == label for name, label in truth.items())


def format_prompt(tokenizer, prompt):
    if getattr(tokenizer, "chat_template", None):
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": str(prompt)}],
            tokenize=False,
            add_generation_prompt=True,
        )
    text = str(prompt)
    return text if text.endswith(("\n", " ")) else text + "\n"


def evaluate_group(llm, tokenizer, frame, seed, n, temperature, top_p, max_tokens, prompt_length):
    from vllm import SamplingParams

    prompts = []
    prompt_ids = []
    for prompt in frame["prompt"].tolist():
        text = format_prompt(tokenizer, prompt)
        ids = tokenizer(
            text, add_special_tokens=False, truncation=True, max_length=prompt_length
        )["input_ids"]
        prompts.append(text)
        prompt_ids.append(ids)

    requests = []
    params = []
    for prompt_index, ids in enumerate(prompt_ids):
        for sample_index in range(n):
            requests.append({"prompt_token_ids": ids})
            params.append(SamplingParams(
                n=1,
                temperature=temperature,
                top_p=top_p,
                max_tokens=max_tokens,
                seed=int(seed) * 100003 + prompt_index * n + sample_index,
            ))

    outputs = llm.generate(requests, params, use_tqdm=True)
    values = []
    truncated = 0
    for prompt_index in range(len(prompt_ids)):
        for sample_index in range(n):
            output = outputs[prompt_index * n + sample_index].outputs[0]
            truncated += int(output.finish_reason == "length")
            values.append(correct(output.text, frame.iloc[prompt_index]["ground_truth"]))
    correct_array = np.asarray(values, dtype=np.float32).reshape(len(frame), n)
    return {
        "pass_at_1": float(correct_array.mean()),
        "avg_at_n": float(correct_array.mean()),
        "pass_at_n": float(correct_array.max(axis=1).mean()),
        "truncated_fraction": float(truncated / max(correct_array.size, 1)),
        "rows": len(frame),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--step", type=int)
    parser.add_argument("--data-root", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.18)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()
    if args.n <= 0 or args.max_examples < 0:
        raise SystemExit("--n must be positive and --max-examples must be nonnegative")

    run_dir = args.run_dir.expanduser().resolve()
    step = find_step(run_dir, args.step)
    checkpoint = run_dir / f"global_step_{step}"
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"SFT checkpoint is not a plain HF directory: {checkpoint}")
    data_root = args.data_root.expanduser().resolve()
    paths = {name: data_root / filename for name, filename in {
        "iid": "iid_test.parquet", "ood": "ood_test.parquet"
    }.items()}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise SystemExit("Missing K&K evaluation parquet files:\n" + "\n".join(missing))

    add_workspace_cuda_libs()
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model_config = AutoConfig.from_pretrained(str(checkpoint))
    model_context = getattr(model_config, "max_position_embeddings", None)
    effective_max_tokens = args.max_tokens
    if model_context is not None:
        model_context = int(model_context)
        if args.prompt_length >= model_context:
            raise SystemExit("--prompt-length must be smaller than model context")
        effective_max_tokens = min(args.max_tokens, model_context - args.prompt_length)
    max_model_len = min(
        args.prompt_length + effective_max_tokens,
        model_context if model_context is not None else args.prompt_length + effective_max_tokens,
    )
    # CUDA graphs improve steady-state vLLM throughput. Keep an escape hatch
    # for environments that reproduce the earlier CUDA-graph startup hang.
    enforce_eager = os.environ.get("VLLM_ENFORCE_EAGER", "0").lower() in {
        "1", "true", "yes", "on"
    }
    print(f"vLLM enforce_eager={enforce_eager}", flush=True)
    llm = LLM(
        model=str(checkpoint),
        tokenizer=str(checkpoint),
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=max_model_len,
        dtype="bfloat16",
        enforce_eager=enforce_eager,
        attention_config={"backend": "FLASHINFER"},
    )
    try:
        per_seed = {}
        for seed in args.seeds:
            seed_results = {}
            for split_name, path in paths.items():
                frame = pd.read_parquet(path)
                for people in sorted(frame["people"].unique()):
                    group = frame[frame["people"] == people].reset_index(drop=True)
                    # Apply the smoke limit independently to each benchmark so a
                    # one-example smoke run still exercises every available K&K
                    # cardinality instead of silently dropping all but the first.
                    if args.max_examples > 0:
                        group = group.head(args.max_examples)
                    key = f"{'iid' if split_name == 'iid' else 'ood'}_{int(people)}ppl"
                    seed_results[key] = evaluate_group(
                        llm, tokenizer, group, seed, args.n, args.temperature,
                        args.top_p, effective_max_tokens, args.prompt_length,
                    )
                    print(f"seed={seed} {key}: pass@1={seed_results[key]['pass_at_1']:.6f}")
            per_seed[str(seed)] = seed_results
    finally:
        engine = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()
        del llm
        gc.collect()

    # Smoke runs may intentionally evaluate only a subset of rows. Aggregate
    # the groups that were actually present rather than indexing a fixed list
    # of names that may not have been reached.
    evaluated_names = [
        name for name in GROUP_FILES
        if all(name in per_seed[str(seed)] for seed in args.seeds)
    ]
    if not evaluated_names:
        raise SystemExit("No K&K benchmark groups were evaluated")

    benchmarks = {}
    for name in evaluated_names:
        values = [per_seed[str(seed)][name] for seed in args.seeds]
        benchmarks[name] = {
            "per_seed": {str(seed): per_seed[str(seed)][name] for seed in args.seeds},
            "mean": {metric: float(np.mean([value[metric] for value in values]))
                     for metric in ("pass_at_1", "avg_at_n", "pass_at_n")},
            "std": {metric: float(np.std([value[metric] for value in values]))
                    for metric in ("pass_at_1", "avg_at_n", "pass_at_n")},
        }
    averages = {
        str(seed): float(np.mean([per_seed[str(seed)][name]["pass_at_1"] for name in evaluated_names]))
        for seed in args.seeds
    }
    iid_average = float(np.mean([
        per_seed[str(seed)][name]["pass_at_1"]
        for seed in args.seeds for name in evaluated_names if name.startswith("iid_")
    ]))
    ood_average = float(np.mean([
        per_seed[str(seed)][name]["pass_at_1"]
        for seed in args.seeds for name in evaluated_names if name.startswith("ood_")
    ]))
    benchmarks["avg_pass_at_1"] = {
        "per_seed": averages, "mean": float(np.mean(list(averages.values()))),
        "std": float(np.std(list(averages.values()))),
    }
    result = {
        "schema_version": 1,
        "kind": "knights_and_knaves_sft_eval",
        "checkpoint_step": step,
        "checkpoint": str(checkpoint),
        "data_root": str(data_root),
        "seeds": args.seeds,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                      "max_tokens": effective_max_tokens, "n": args.n},
        "benchmarks": benchmarks,
        "per_seed": per_seed,
        "metrics": {
            **{f"final_eval/{name}_pass_at_1_mean": benchmarks[name]["mean"]["pass_at_1"]
               for name in evaluated_names},
            "final_eval/iid_avg_pass_at_1_mean": iid_average,
            "final_eval/ood_avg_pass_at_1_mean": ood_average,
            "final_eval/avg_pass_at_1_mean": benchmarks["avg_pass_at_1"]["mean"],
        },
    }
    output = run_dir / "final_sft_eval.json"
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(f"Wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
