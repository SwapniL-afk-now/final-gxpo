#!/usr/bin/env python3
"""Evaluate a Hugging Face checkpoint on code parquet benchmarks."""

from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from tqdm.auto import tqdm
except ImportError:  # pragma: no cover - optional progress display
    def tqdm(iterable, **_kwargs):
        return iterable


BENCHMARKS = ("humanevalplus", "mbppplus", "livecodebench")


def _score_job(payload):
    """Score one generated completion in a verifier worker."""
    prompt, data_source, completion, ground_truth, extra_info = payload
    from verl.utils.reward_score import _default_compute_score

    # Evaluation records binary correctness for pass@1.  Fail-fast is therefore
    # equivalent to the old full execution result while avoiding unnecessary
    # LiveCodeBench test cases after the first mismatch.
    return float(_default_compute_score(
        prompt, data_source, completion, ground_truth, extra_info=extra_info,
        stop_on_failure=True,
    ))


def _resolve_verifier_workers(requested: int) -> int:
    if requested and requested > 0:
        return requested
    return min(64, max(1, os.cpu_count() or 1))


def _verify_in_parallel(score_jobs, requested_workers: int):
    if not score_jobs:
        return []

    workers = min(_resolve_verifier_workers(requested_workers), len(score_jobs))
    try:
        context = mp.get_context("fork")
    except ValueError:  # pragma: no cover - non-POSIX fallback
        context = mp.get_context("spawn")

    print(
        f"Verifying {len(score_jobs)} generated candidates with "
        f"{workers} CPU workers (fail-fast)",
        flush=True,
    )
    scores = [None] * len(score_jobs)
    with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {
            pool.submit(_score_job, payload): index
            for index, payload in enumerate(score_jobs)
        }
        for future in tqdm(
            as_completed(futures),
            total=len(futures),
            desc="Verifying generated code",
            unit="candidate",
        ):
            scores[futures[future]] = future.result()
    return scores


def find_step(run_dir: Path, requested: int | None) -> int:
    if requested is not None:
        return requested
    steps = [
        int(path.name.removeprefix("global_step_"))
        for path in run_dir.glob("global_step_*")
        if path.is_dir() and path.name.removeprefix("global_step_").isdigit()
    ]
    if not steps:
        raise SystemExit(f"No checkpoints found under {run_dir}")
    return max(steps)


def resolve_files(root: Path) -> dict[str, Path]:
    candidates = {}
    for name in BENCHMARKS:
        paths = (
            root / f"{name}.parquet",
            root / name / "test.parquet",
            root / name / "data.parquet",
        )
        path = next((candidate for candidate in paths if candidate.is_file()), None)
        if path is not None:
            candidates[name] = path
    return candidates


def as_prompt(tokenizer, prompt):
    if isinstance(prompt, str):
        return prompt
    return tokenizer.apply_chat_template(
        list(prompt), tokenize=False, add_generation_prompt=True
    )


def _vllm_worker(
    rank,
    gpu_id,
    checkpoint,
    engine_kwargs,
    jobs,
    result_queue,
):
    """Run one TP=1 vLLM engine on one GPU for a data-parallel shard."""
    try:
        # Each spawned worker sees exactly one physical GPU. This is explicit
        # data parallelism; tensor_parallel_size remains 1.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
        from vllm import LLM, SamplingParams

        llm = LLM(**engine_kwargs)
        prompts = [{"prompt_token_ids": job["prompt_token_ids"]} for job in jobs]
        sampling_params = [
            SamplingParams(
                n=1,
                temperature=job["temperature"],
                top_p=job["top_p"],
                max_tokens=job["max_tokens"],
                seed=job["seed"],
            )
            for job in jobs
        ]
        outputs = llm.generate(prompts, sampling_params, use_tqdm=True)
        texts = [output.outputs[0].text for output in outputs]
        result_queue.put((rank, texts, None))
        # Let the engine worker processes flush before this process exits.
        sleep = getattr(__import__("time"), "sleep")
        sleep(1)
    except Exception as exc:  # propagate the actual worker failure to parent
        result_queue.put((rank, None, repr(exc)))
        raise


def _generate_data_parallel(
    checkpoint,
    tokenizer,
    files,
    seeds,
    n,
    temperature,
    top_p,
    max_tokens,
    max_examples,
    prompt_length,
    gpu_memory_utilization,
    tensor_parallel_size,
    data_parallel_size,
    max_num_batched_tokens,
    max_num_seqs,
    verifier_workers,
):
    if tensor_parallel_size != 1:
        raise ValueError("Code distillation evaluation requires tensor_parallel_size=1")
    if data_parallel_size < 1:
        raise ValueError("data_parallel_size must be positive")

    visible_gpus = [gpu.strip() for gpu in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if gpu.strip()]
    if len(visible_gpus) < data_parallel_size:
        raise RuntimeError(
            f"Need {data_parallel_size} visible GPUs for TP=1 data parallel evaluation; "
            f"CUDA_VISIBLE_DEVICES has {len(visible_gpus)}"
        )

    # Build one request stream for all benchmarks/seeds so the two engines are
    # loaded only once per evaluation and can batch across the full workload.
    jobs = []
    metadata = []
    for seed in seeds:
        for benchmark, path in files.items():
            frame = pd.read_parquet(path)
            if max_examples > 0:
                frame = frame.head(max_examples)
            prompt_texts = [as_prompt(tokenizer, prompt) for prompt in frame["prompt"]]
            prompt_ids = [
                tokenizer(
                    text,
                    add_special_tokens=False,
                    truncation=True,
                    max_length=prompt_length,
                )["input_ids"]
                for text in prompt_texts
            ]
            for problem_index, ids in enumerate(prompt_ids):
                for sample_index in range(n):
                    jobs.append({
                        "prompt_token_ids": ids,
                        "temperature": temperature,
                        "top_p": top_p,
                        "max_tokens": max_tokens,
                        "seed": seed * 100003 + problem_index * n + sample_index,
                    })
                    metadata.append((seed, benchmark, problem_index, sample_index))

    if not jobs:
        return {str(seed): {} for seed in seeds}

    engine_kwargs = {
        "model": str(checkpoint),
        "tokenizer": str(checkpoint),
        "tensor_parallel_size": 1,
        "gpu_memory_utilization": gpu_memory_utilization,
        "max_model_len": prompt_length + max_tokens,
        "max_num_batched_tokens": max_num_batched_tokens,
        "max_num_seqs": max_num_seqs,
        "enable_prefix_caching": True,
        "enable_chunked_prefill": True,
        "dtype": "bfloat16",
        "enforce_eager": os.environ.get("VLLM_ENFORCE_EAGER", "0").lower() in {"1", "true"},
        "attention_config": {"backend": "FLASHINFER"},
    }

    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []
    shards = []
    floor, remainder = divmod(len(jobs), data_parallel_size)
    start = 0
    for rank in range(data_parallel_size):
        size = floor + (1 if rank < remainder else 0)
        shards.append(jobs[start:start + size])
        start += size
    for rank, shard in enumerate(shards):
        process = ctx.Process(
            target=_vllm_worker,
            args=(rank, visible_gpus[rank], str(checkpoint), engine_kwargs, shard, result_queue),
        )
        process.start()
        processes.append(process)

    shard_outputs = {}
    worker_errors = []
    for _ in processes:
        rank, texts, error = result_queue.get()
        if error is not None:
            worker_errors.append(f"worker {rank}: {error}")
        else:
            shard_outputs[rank] = texts
    for process in processes:
        process.join()
    if worker_errors or any(process.exitcode for process in processes):
        details = "; ".join(worker_errors) or "vLLM worker exited nonzero"
        raise RuntimeError(details)

    all_texts = []
    for rank, texts in shard_outputs.items():
        all_texts.extend((rank, index, text) for index, text in enumerate(texts))
    # Shard-local indices are converted back to global request order.
    ordered_texts = [None] * len(jobs)
    offset = 0
    for rank, shard in enumerate(shards):
        texts = shard_outputs[rank]
        ordered_texts[offset:offset + len(texts)] = texts
        offset += len(texts)

    correct_by_key = {}
    score_jobs = []
    score_metadata = []
    cursor = 0
    for seed in seeds:
        for benchmark, path in files.items():
            frame = pd.read_parquet(path)
            if max_examples > 0:
                frame = frame.head(max_examples)
            key = (str(seed), benchmark)
            correct_by_key[key] = np.zeros((len(frame), n), dtype=np.float32)
            prompt_texts = [as_prompt(tokenizer, prompt) for prompt in frame["prompt"]]
            for problem_index in range(len(frame)):
                row = frame.iloc[problem_index]
                ground_truth = (
                    row["reward_model"]["ground_truth"]
                    if isinstance(row["reward_model"], dict)
                    else row["ground_truth"]
                )
                for sample_index in range(n):
                    score_jobs.append((
                        prompt_texts[problem_index],
                        row["data_source"],
                        ordered_texts[cursor],
                        ground_truth,
                        row.get("extra_info", None),
                    ))
                    score_metadata.append((str(seed), benchmark, problem_index, sample_index))
                    cursor += 1

    scores = _verify_in_parallel(score_jobs, verifier_workers)
    for metadata, score in zip(score_metadata, scores, strict=True):
        seed, benchmark, problem_index, sample_index = metadata
        correct_by_key[(seed, benchmark)][problem_index, sample_index] = (
            float(score) >= 1.0
        )

    per_seed = {str(seed): {} for seed in seeds}
    for seed in seeds:
        for benchmark in files:
            correct = correct_by_key[(str(seed), benchmark)]
            per_seed[str(seed)][benchmark] = {
                "pass_at_1": float(correct[:, 0].mean()) if len(correct) else float("nan"),
                "avg_at_n": float(correct.mean()) if correct.size else float("nan"),
                "pass_at_n": float(correct.max(axis=1).mean()) if len(correct) else float("nan"),
            }
    return per_seed


def evaluate(
    tokenizer,
    files,
    seeds,
    n,
    temperature,
    top_p,
    max_tokens,
    max_examples,
    prompt_length,
    checkpoint,
    gpu_memory_utilization,
    tensor_parallel_size,
    data_parallel_size,
    max_num_batched_tokens,
    max_num_seqs,
    verifier_workers,
):
    return _generate_data_parallel(
        checkpoint,
        tokenizer,
        files,
        seeds,
        n,
        temperature,
        top_p,
        max_tokens,
        max_examples,
        prompt_length,
        gpu_memory_utilization,
        tensor_parallel_size,
        data_parallel_size,
        max_num_batched_tokens,
        max_num_seqs,
        verifier_workers,
    )


def aggregate(per_seed, benchmarks):
    result = {}
    for benchmark in benchmarks:
        values = [per_seed[seed][benchmark] for seed in per_seed]
        result[benchmark] = {
            "mean": {key: float(np.mean([value[key] for value in values])) for key in values[0]},
            "std": {key: float(np.std([value[key] for value in values])) for key in values[0]},
        }
    pass1 = [
        float(np.mean([per_seed[seed][benchmark]["pass_at_1"] for benchmark in benchmarks]))
        for seed in per_seed
    ]
    result["avg_pass_at_1"] = {"mean": float(np.mean(pass1)), "std": float(np.std(pass1))}
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--step", type=int)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--max-examples", type=int, default=0)
    parser.add_argument("--prompt-length", type=int, default=2048)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.18)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--data-parallel-size", type=int, default=1)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=256)
    parser.add_argument(
        "--verifier-workers", type=int,
        default=int(os.environ.get("CODE_EVAL_WORKERS", "0")),
        help="CPU verification workers; 0 selects min(64, CPU count)",
    )
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    step = find_step(run_dir, args.step)
    checkpoint = run_dir / f"global_step_{step}"
    files = resolve_files(args.data_root.resolve())
    if not files:
        raise SystemExit(f"No code benchmark parquet files found under {args.data_root}")

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    per_seed = evaluate(
        tokenizer,
        files,
        args.seeds,
        args.n,
        args.temperature,
        args.top_p,
        args.max_tokens,
        args.max_examples,
        args.prompt_length,
        checkpoint,
        args.gpu_memory_utilization,
        args.tensor_parallel_size,
        args.data_parallel_size,
        args.max_num_batched_tokens,
        args.max_num_seqs,
        args.verifier_workers,
    )
    gc.collect()
    benchmarks = tuple(files)
    result = {
        "schema_version": 1,
        "kind": "code_distillation_eval",
        "checkpoint_step": step,
        "checkpoint": str(checkpoint),
        "data_root": str(args.data_root.resolve()),
        "benchmarks": aggregate(per_seed, benchmarks),
        "per_seed": per_seed,
        "decoding": vars(args),
    }
    serialized = json.dumps(result, indent=2, default=str) + "\n"
    (run_dir / "code_eval.json").write_text(serialized)
    # The SFT trainer consumes the same canonical filename as its math evaluator.
    (run_dir / "final_sft_eval.json").write_text(serialized)
    print(json.dumps(result["benchmarks"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
