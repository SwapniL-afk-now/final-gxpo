#!/usr/bin/env python3
"""Greedy benchmark evaluation for a flat offline KD-SFT checkpoint."""
from __future__ import annotations

import argparse
import json
import os
import sys
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


def _parse_devices(value):
    if not value:
        return []
    return [item.strip() for item in value.split(',') if item.strip()]


def _write_result(result, args, output):
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\\n")
    print(f"Wrote {output}", flush=True)
    if args.sft_output_dir:
        # Preserve the trainer readback schema alongside the multi-GPU result.
        sft_benchmarks = {}
        for key, value in result.get("pass_at_n", {}).items():
            average = result.get("average_at_n", {}).get(key)
            sft_benchmarks[key] = {
                "pass_at_n": {"per_seed": {str(args.seed): value},
                               "mean": value, "std": 0.0},
                "average_at_n": {"per_seed": {str(args.seed): average},
                                  "mean": average, "std": 0.0},
            }
        sft_benchmarks["macro"] = {
            "pass_at_n": {"mean": result["avg_pass_at_n"], "std": 0.0},
            "average_at_n": {"mean": result["avg_average_at_n"], "std": 0.0},
        }
        sft_output = Path(args.sft_output_dir).expanduser()
        sft_output.mkdir(parents=True, exist_ok=True)
        sft_path = sft_output / f"sft_sampled_{args.n}_1seed.json"
        sft_path.write_text(json.dumps({"benchmarks": sft_benchmarks},
                                       indent=2, sort_keys=True) + "\n")
        print(f"Wrote {sft_path}", flush=True)
    if not args.log_wandb:
        return
    import wandb
    run = wandb.init(project=args.wandb_project, name=args.wandb_run,
                     job_type="post_eval", resume="allow")
    sampled = args.n > 1 or args.temperature > 0
    prefix = "eval_sampled" if sampled else "eval_greedy"
    pass_suffix = f"pass{args.n}" if sampled else "pass1"
    payload = {}
    for key, value in result.get("pass_at_n", {}).items():
        payload[f"{prefix}/{key}_{pass_suffix}"] = value
    if sampled:
        for key, value in result.get("average_at_n", {}).items():
            payload[f"{prefix}/{key}_average{args.n}"] = value
        payload[f"{prefix}/avg_pass{args.n}"] = result["avg_pass_at_n"]
        payload[f"{prefix}/avg_average{args.n}"] = result["avg_average_at_n"]
    elif result.get("avg_pass1") is not None:
        payload[f"{prefix}/avg_pass1"] = result["avg_pass1"]
    log_kwargs = {"step": args.step} if args.step is not None else {}
    run.log(payload, **log_kwargs)
    run.finish()
    print(f"Logged {len(payload)} evaluation metrics to wandb", flush=True)


def _single_worker(args, data_files, output):
    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    import pandas as pd
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score import _default_compute_score

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    llm = LLM(model=str(checkpoint), tokenizer=args.base_model,
              tensor_parallel_size=1,
              gpu_memory_utilization=args.gpu_memory_utilization, dtype="bfloat16",
              max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
              max_num_batched_tokens=args.max_num_batched_tokens,
              trust_remote_code=True)
    sampling = SamplingParams(n=args.n, temperature=args.temperature,
                              top_p=args.top_p, max_tokens=args.max_tokens,
                              stop_token_ids=([tokenizer.eos_token_id]
                                              if tokenizer.eos_token_id is not None else None),
                              ignore_eos=False, seed=args.seed)
    benchmarks, pass_at_n, average_at_n = {}, {}, {}
    for data_file in data_files:
        frame = pd.read_parquet(data_file)
        prompts = [tokenizer.apply_chat_template(
            list(chat), tokenize=False, add_generation_prompt=True) for chat in frame["prompt"]]
        outputs = llm.generate(prompts, sampling)
        source = str(frame["data_source"].iloc[0])
        pass_values, average_values = [], []
        for row_index, generated in enumerate(outputs):
            truth = frame["reward_model"].iloc[row_index]["ground_truth"]
            correct = [float(_default_compute_score(
                prompts[row_index], source, candidate.text, truth)) >= 0.95
                       for candidate in generated.outputs]
            pass_values.append(float(any(correct)))
            average_values.append(float(np.mean(correct)))
        key = benchmark_key(data_file, source)
        pass_at_n[key] = float(np.mean(pass_values)) if pass_values else float("nan")
        average_at_n[key] = float(np.mean(average_values)) if average_values else float("nan")
        benchmarks[key] = average_at_n[key]
        print(f"seed={args.seed} {key}: pass@{args.n}={pass_at_n[key]:.6f} "
              f"average@{args.n}={average_at_n[key]:.6f}", flush=True)
    result = {
        "schema_version": 2,
        "checkpoint": str(checkpoint),
        "seed": args.seed,
        "train_step": args.step,
        "decoding": {"temperature": args.temperature, "top_p": args.top_p,
                      "n": args.n, "max_tokens": args.max_tokens},
        "vllm": {"max_model_len": args.max_model_len,
                 "max_num_seqs": args.max_num_seqs,
                 "max_num_batched_tokens": args.max_num_batched_tokens,
                 "tp": 1,
                 "gpu_memory_utilization": args.gpu_memory_utilization},
        "benchmarks": benchmarks,
        "pass_at_n": pass_at_n,
        "average_at_n": average_at_n,
        "avg_pass_at_n": float(np.mean(list(pass_at_n.values()))) if pass_at_n else None,
        "avg_average_at_n": float(np.mean(list(average_at_n.values()))) if average_at_n else None,
        "avg_pass1": float(np.mean(list(pass_at_n.values()))) if args.n == 1 and pass_at_n else None,
    }
    _write_result(result, args, output)
    return result


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
    ap.add_argument("--max-model-len", type=int, default=20480)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--max-num-batched-tokens", type=int, default=65536)
    ap.add_argument("--attention-backend", type=str, default=None)
    ap.add_argument("--gpu-devices", type=str, default=None,
                    help="comma-separated GPUs; launches one independent TP=1 worker per GPU")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--log-wandb", action="store_true")
    ap.add_argument("--wandb-project", type=str, default=None)
    ap.add_argument("--wandb-run", type=str, default=None)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sft-output-dir", default=None,
                    help="also write the trainer-compatible sft_sampled_N_1seed.json")
    args = ap.parse_args()
    if args.attention_backend is None:
        args.attention_backend = os.environ.get("VLLM_ATTENTION_BACKEND", "FLASH_ATTN")
    os.environ["VLLM_ATTENTION_BACKEND"] = args.attention_backend
    print(f"vLLM attention backend: {args.attention_backend}", flush=True)

    checkpoint = Path(args.checkpoint_dir).expanduser().resolve()
    if not (checkpoint / "config.json").is_file():
        raise SystemExit(f"flat HF checkpoint not found: {checkpoint}/config.json")
    data_files = [str(Path(item).expanduser().resolve()) for item in args.data_files]
    for data_file in data_files:
        if not Path(data_file).is_file():
            raise SystemExit(f"evaluation dataset not found: {data_file}")

    devices = _parse_devices(args.gpu_devices or os.environ.get("CUDA_VISIBLE_DEVICES"))
    if not args.worker and len(devices) > 1:
        output = Path(args.output).expanduser().resolve()
        worker_root = output.parent / f".{output.stem}.workers"
        if worker_root.exists():
            import shutil
            shutil.rmtree(worker_root)
        worker_root.mkdir(parents=True, exist_ok=True)
        chunks = [data_files[index::len(devices)] for index in range(len(devices))]
        jobs = []
        for index, (device, chunk) in enumerate(zip(devices, chunks)):
            if not chunk:
                continue
            worker_output = worker_root / f"worker{index}.json"
            command = [sys.executable, str(Path(__file__).resolve()),
                       "--checkpoint-dir", str(checkpoint), "--base-model", args.base_model,
                       "--data-files", *chunk, "--seed", str(args.seed), "--n", str(args.n),
                       "--temperature", str(args.temperature), "--top-p", str(args.top_p),
                       "--max-tokens", str(args.max_tokens), "--tp", "1",
                       "--gpu-memory-utilization", str(args.gpu_memory_utilization),
                       "--max-model-len", str(args.max_model_len), "--max-num-seqs", str(args.max_num_seqs),
                       "--max-num-batched-tokens", str(args.max_num_batched_tokens),
                       "--attention-backend", args.attention_backend, "--worker",
                       "--output", str(worker_output)]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = device
            for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "LOCAL_WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
                env.pop(key, None)
            jobs.append((device, worker_output, __import__("subprocess").Popen(command, env=env)))
        for device, worker_output, process in jobs:
            if process.wait() != 0:
                raise SystemExit(f"parallel evaluation worker on GPU {device} failed")
        merged = json.loads(jobs[0][1].read_text())
        for _, worker_output, _ in jobs[1:]:
            worker = json.loads(worker_output.read_text())
            for field in ("benchmarks", "pass_at_n", "average_at_n"):
                merged[field].update(worker.get(field, {}))
        merged["avg_pass_at_n"] = float(np.mean(list(merged["pass_at_n"].values())))
        merged["avg_average_at_n"] = float(np.mean(list(merged["average_at_n"].values())))
        merged["avg_pass1"] = merged["avg_pass_at_n"] if args.n == 1 else None
        merged["vllm"]["parallel_gpu_devices"] = devices
        _write_result(merged, args, output)
        import shutil
        shutil.rmtree(worker_root, ignore_errors=True)
        return

    _single_worker(args, data_files, args.output)


if __name__ == "__main__":
    main()
