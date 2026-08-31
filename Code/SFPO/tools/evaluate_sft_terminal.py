#!/usr/bin/env python3
"""Evaluate a plain Hugging Face SFT checkpoint on the six GXPO benchmarks."""
from __future__ import annotations

import argparse
import gc
import json
import os
import sys
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


def add_workspace_cuda_libs():
    os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
    os.environ["PATH"] = str(ROOT.parents[1] / ".venv" / "bin") + os.pathsep + os.environ.get("PATH", "")
    cuda_root = ROOT.parents[1] / ".venv" / "lib" / "python3.12" / "site-packages" / "nvidia"
    if not cuda_root.is_dir():
        return
    library_dirs = []
    for package_dir in sorted(cuda_root.iterdir()):
        candidate = package_dir / "lib"
        if candidate.is_dir():
            library_dirs.append(str(candidate))
    python_include = ROOT.parents[1] / ".python-dev" / "usr" / "include" / "python3.12"
    if (python_include / "Python.h").is_file():
        include_root = str(python_include.parent)
        existing_c = [path for path in os.environ.get("C_INCLUDE_PATH", "").split(":") if path]
        os.environ["C_INCLUDE_PATH"] = ":".join([include_root, str(python_include), *[path for path in existing_c if path not in {include_root, str(python_include)}]])
        existing_cpp = [path for path in os.environ.get("CPLUS_INCLUDE_PATH", "").split(":") if path]
        os.environ["CPLUS_INCLUDE_PATH"] = ":".join([include_root, str(python_include), *[path for path in existing_cpp if path not in {include_root, str(python_include)}]])
    cuda_home = ROOT.parents[1] / ".cuda-toolkit"
    if (cuda_home / "bin" / "nvcc").is_file():
        os.environ.setdefault("CUDA_HOME", str(cuda_home))
        os.environ.setdefault("CUDA_PATH", str(cuda_home))
        os.environ.setdefault("CUDACXX", str(cuda_home / "bin" / "nvcc"))
    existing = [path for path in os.environ.get("LD_LIBRARY_PATH", "").split(":") if path]
    merged = [path for path in library_dirs if path not in existing]
    merged.extend(path for path in existing if path)
    if merged:
        os.environ["LD_LIBRARY_PATH"] = ":".join(merged)
        if os.environ.get("SFT_EVAL_CUDA_REEXEC") != "1":
            os.environ["SFT_EVAL_CUDA_REEXEC"] = "1"
            os.execvpe(sys.executable, [sys.executable, *sys.argv], os.environ)

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


def evaluate_seed(
    llm, tokenizer, data_files, seed, n, temperature, top_p, max_tokens, max_examples, prompt_length
):
    import pandas as pd
    from verl.utils.reward_score import _default_compute_score
    from vllm import SamplingParams

    results = {}
    for benchmark, data_file in data_files.items():
        frame = pd.read_parquet(data_file)
        if max_examples > 0:
            frame = frame.head(max_examples)
        prompt_texts = [
            tokenizer.apply_chat_template(
                list(prompt), tokenize=False, add_generation_prompt=True
            )
            for prompt in frame["prompt"]
        ]
        # Keep the prompt within the requested context budget after adding
        # the chat template; this also makes tiny smoke contexts deterministic.
        prompt_token_ids = [
            tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=prompt_length,
            )["input_ids"]
            for prompt_text in prompt_texts
        ]
        ground_truths = [row["ground_truth"] for row in frame["reward_model"]]
        data_sources = frame["data_source"].tolist()

        flat_prompts = []
        sampling_params = []
        for prompt_index, prompt in enumerate(prompt_token_ids):
            for sample_index in range(n):
                flat_prompts.append({"prompt_token_ids": prompt})
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
        correct = np.zeros((len(prompt_token_ids), n), dtype=np.float32)
        truncated = 0
        for prompt_index in range(len(prompt_token_ids)):
            for sample_index in range(n):
                output = outputs[prompt_index * n + sample_index].outputs[0]
                truncated += int(output.finish_reason == "length")
                score = _default_compute_score(
                    prompt_texts[prompt_index],
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
    parser.add_argument(
        "--max-examples",
        type=int,
        default=0,
        help="Evaluate only the first N examples per benchmark (0 means all).",
    )
    parser.add_argument("--prompt-length", type=int, default=1024)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    args = parser.parse_args()

    if args.n <= 0:
        raise SystemExit("--n must be positive")
    if args.max_examples < 0:
        raise SystemExit("--max-examples must be zero or positive")
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

    add_workspace_cuda_libs()
    from transformers import AutoConfig, AutoTokenizer
    from vllm import LLM

    tokenizer = AutoTokenizer.from_pretrained(str(checkpoint))
    model_config = AutoConfig.from_pretrained(str(checkpoint))
    model_context = getattr(model_config, "max_position_embeddings", None)
    requested_max_model_len = args.prompt_length + args.max_tokens
    effective_max_tokens = args.max_tokens
    if model_context is not None:
        model_context = int(model_context)
        if args.prompt_length >= model_context:
            raise SystemExit(
                f"--prompt-length {args.prompt_length} must be smaller than "
                f"model context {model_context}"
            )
        effective_max_tokens = min(
            effective_max_tokens, model_context - args.prompt_length
        )
    max_model_len = (
        min(requested_max_model_len, model_context)
        if model_context is not None
        else requested_max_model_len
    )
    if effective_max_tokens != args.max_tokens:
        print(
            f"Clamping --max-tokens from {args.max_tokens} to "
            f"{effective_max_tokens} for model context {model_context}"
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
        # Make FlashInfer authoritative for vLLM V1; the environment alone
        # can otherwise be overridden by automatic backend selection.
        attention_config={"backend": "FLASHINFER"},
    )
    try:
        per_seed = {
            str(seed): evaluate_seed(
                llm,
                tokenizer,
                data_files,
                seed,
                args.n,
                args.temperature,
                args.top_p,
                effective_max_tokens,
                args.max_examples,
                args.prompt_length,
            )
            for seed in args.seeds
        }
    finally:
        # vLLM V1 owns a separate EngineCore process.  Explicitly shut it
        # down before returning so the trainer can reclaim the GPU and resume.
        engine = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        shutdown = getattr(engine, "shutdown", None)
        if shutdown is not None:
            shutdown()
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
        "max_examples": args.max_examples,
        "seeds": args.seeds,
        "data_root": str(data_root),
        "decoding": {
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_tokens": effective_max_tokens,
            "requested_max_tokens": args.max_tokens,
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
