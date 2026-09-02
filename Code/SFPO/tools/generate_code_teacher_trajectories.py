#!/usr/bin/env python3
"""Generate teacher coding trajectories and verify them in parallel."""
from __future__ import annotations

import argparse
import gc
import json
import multiprocessing as mp
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import pandas as pd


def verify_problem(payload: dict) -> list[dict]:
    """Verify one problem's candidates; keep at most two verified responses."""
    from verl.utils.reward_score.stdio_code import compute_score

    records = []
    for sample_index, response in enumerate(payload["responses"]):
        score_info = compute_score(
            response,
            payload["ground_truth"],
            continuous=False,
            stop_on_failure=True,
        )
        verified = bool(score_info.get("acc", False)) and float(score_info["score"]) >= 1.0
        if not verified:
            continue
        records.append(
            {
                "problem_id": payload["problem_id"],
                "problem_hash": payload["problem_hash"],
                "source": "taco_verified",
                "split": payload["split"],
                "sample_index": sample_index,
                "response": response,
                "score": float(score_info["score"]),
                "verified": True,
                "finish_reason": payload["finish_reasons"][sample_index],
            }
        )
        if len(records) >= 2:
            break
    return records


def write_candidates(frame: pd.DataFrame, outputs, path: Path) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w") as handle:
        for row, result in zip(frame.itertuples(index=False), outputs):
            payload = {
                "problem_id": str(row.problem_id),
                "problem_hash": str(row.problem_hash),
                "split": str(row.split),
                "ground_truth": row.ground_truth,
                "responses": [item.text for item in result.outputs],
                "finish_reasons": [
                    getattr(item, "finish_reason", None) for item in result.outputs
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
            count += 1
    return count


def load_candidates(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def verify_candidates(candidates: list[dict], output: Path, workers: int) -> tuple[int, int]:
    output.parent.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    verified_count = 0
    completed = 0
    problems_with_answer = 0
    context = mp.get_context("spawn")
    with output.open("w") as handle, ProcessPoolExecutor(
        max_workers=workers,
        mp_context=context,
        max_tasks_per_child=128,
    ) as pool:
        # Keep only a small bounded queue: candidate responses are large, and
        # submitting the whole corpus would unnecessarily duplicate them in
        # multiprocessing buffers.
        candidate_iter = iter(candidates)
        pending = {}

        def refill() -> None:
            while len(pending) < max(1, workers * 2):
                try:
                    payload = next(candidate_iter)
                except StopIteration:
                    return
                pending[pool.submit(verify_problem, payload)] = None

        refill()
        try:
            from tqdm import tqdm
            progress = tqdm(
                total=len(candidates),
                desc="Verifying teacher candidates",
                unit="problem",
                dynamic_ncols=True,
            )
        except ImportError:
            progress = None

        while pending:
            done, _ = wait(tuple(pending), return_when=FIRST_COMPLETED)
            for future in done:
                del pending[future]
                # A dead worker must fail the queue explicitly; never wait
                # forever for an ordered map result after the pool is broken.
                records = future.result()
                completed += 1
                if progress is not None:
                    progress.update(1)
                if records:
                    problems_with_answer += 1
                    verified_count += len(records)
                    for record in records:
                        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                if completed % 100 == 0:
                    elapsed = max(time.monotonic() - start, 1e-6)
                    rate = completed / elapsed
                    print(
                        f"[verifier] problems={completed}/{len(candidates)} "
                        f"rate={rate:.2f}/s verified={verified_count}",
                        flush=True,
                    )
            refill()
        if progress is not None:
            progress.close()
    return verified_count, problems_with_answer


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--candidates-output", type=Path)
    parser.add_argument("--model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=3072)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--verifier-workers", type=int, default=0)
    parser.add_argument("--split", choices=("all", "train", "validation"), default="all")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.num_samples <= 0:
        raise SystemExit("--num-samples must be positive")

    candidates_path = args.candidates_output or args.output.with_name("teacher_candidates.jsonl")
    if args.verify_only:
        candidates = load_candidates(candidates_path)
        if not candidates:
            raise SystemExit(f"No candidates found in {candidates_path}")
    else:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        frame = pd.read_parquet(args.input)
        if args.split != "all":
            frame = frame[frame["split"] == args.split].reset_index(drop=True)
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        prompts = [
            tokenizer.apply_chat_template(
                list(prompt), tokenize=False, add_generation_prompt=True
            )
            for prompt in frame["prompt"]
        ]
        llm = LLM(
            model=args.model,
            tokenizer=args.model,
            tensor_parallel_size=args.tensor_parallel_size,
            gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=1024 + args.max_tokens,
            dtype="bfloat16",
            enforce_eager=False,
            attention_config={"backend": "FLASHINFER"},
        )
        sampling = [
            SamplingParams(
                n=args.num_samples,
                temperature=args.temperature,
                top_p=args.top_p,
                max_tokens=args.max_tokens,
                seed=1729 + index,
            )
            for index in range(len(prompts))
        ]
        outputs = llm.generate(prompts, sampling, use_tqdm=True)
        count = write_candidates(frame, outputs, candidates_path)
        print(f"Saved {count} problem candidate groups to {candidates_path}", flush=True)

        engine = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if getattr(engine, "shutdown", None):
            engine.shutdown()
        del outputs, llm
        gc.collect()
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        candidates = load_candidates(candidates_path)

    workers = args.verifier_workers or min(64, max(1, (os.cpu_count() or 8) // 2))
    workers = max(1, workers)
    print(
        f"Verifying {len(candidates)} problem groups with {workers} workers; "
        f"keeping at most two correct responses per problem",
        flush=True,
    )
    verified, solved = verify_candidates(candidates, args.output, workers)
    print(
        f"Wrote {verified} verified trajectories from {len(candidates)} problems; "
        f"problems with a verified teacher answer={solved}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
