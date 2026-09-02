#!/usr/bin/env python3
"""Score every code-distillation target with offline top-K teacher distributions."""

from __future__ import annotations

import argparse
import gc
import json
import math
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from transformers import AutoTokenizer

from verl.utils.dataset.kd_utils import (
    KD_SCHEMA_VERSION,
    format_prompt_response,
    tokenizer_fingerprint,
)


def logprob_value(value) -> float:
    return float(getattr(value, "logprob", value))


def extract_topk(position, topk: int) -> tuple[np.ndarray, np.ndarray, int, float]:
    """Convert one vLLM prompt-logprob mapping to sorted fixed-width arrays."""
    if position is None or not position:
        raise ValueError("Teacher returned no distribution for a response token")
    entries = sorted(
        ((int(token_id), logprob_value(value)) for token_id, value in position.items()),
        key=lambda item: item[1],
        reverse=True,
    )[:topk]
    if not entries or any(not math.isfinite(logprob) for _, logprob in entries):
        raise ValueError("Teacher returned an empty or non-finite token distribution")
    ids = np.zeros(topk, dtype=np.int32)
    logprobs = np.full(topk, -np.inf, dtype=np.float16)
    count = len(entries)
    ids[:count] = [token_id for token_id, _ in entries]
    logprobs[:count] = np.asarray([logprob for _, logprob in entries], dtype=np.float16)
    mass = float(sum(math.exp(logprob) for _, logprob in entries))
    if not 0.0 < mass <= 1.0001:
        raise ValueError(f"Invalid captured teacher probability mass: {mass}")
    return ids, logprobs, count, mass


def tokenize_rows(frame: pd.DataFrame, tokenizer, max_length: int) -> list[dict]:
    tokenized = []
    failures = []
    for row_index, row in frame.iterrows():
        prompt_text, response_text = format_prompt_response(
            tokenizer, row["prompt"], row["response"]
        )
        prompt_ids = tokenizer(
            prompt_text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        response_ids = tokenizer(
            response_text, add_special_tokens=False, return_attention_mask=False
        )["input_ids"]
        full_ids = list(prompt_ids) + list(response_ids)
        if not response_ids or len(full_ids) > max_length:
            failures.append((int(row_index), len(prompt_ids), len(response_ids), len(full_ids)))
            continue
        tokenized.append({
            "prompt_length": len(prompt_ids),
            "response_ids": np.asarray(response_ids, dtype=np.int32),
            "full_ids": full_ids,
        })
    if failures:
        preview = ", ".join(
            f"row={row} prompt={prompt} response={response} total={total}"
            for row, prompt, response, total in failures[:10]
        )
        raise ValueError(
            f"KD scoring requires non-empty, untruncated responses <= {max_length} tokens; "
            f"failed rows={len(failures)} ({preview})"
        )
    return tokenized


def create_sidecar_arrays(root: Path, offsets: np.ndarray, topk: int) -> dict[str, np.memmap]:
    root.mkdir(parents=True, exist_ok=False)
    total_tokens = int(offsets[-1])
    np.save(root / "row_offsets.npy", offsets, allow_pickle=False)
    specs = {
        "token_ids": ((total_tokens,), np.int32),
        "topk_ids": ((total_tokens, topk), np.int32),
        "topk_logprobs": ((total_tokens, topk), np.float16),
        "topk_counts": ((total_tokens,), np.uint8),
        "topk_mass": ((total_tokens,), np.float16),
    }
    return {
        name: np.lib.format.open_memmap(root / f"{name}.npy", mode="w+", dtype=dtype, shape=shape)
        for name, (shape, dtype) in specs.items()
    }


def score_split(
    *,
    llm,
    sampling,
    tokenizer,
    tokenizer_hash: str,
    teacher_model: str,
    input_path: Path,
    output_path: Path,
    topk: int,
    max_length: int,
    request_batch_size: int,
    overwrite: bool,
) -> None:
    frame = pd.read_parquet(input_path).reset_index(drop=True)
    required = {"prompt", "response", "source"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"{input_path} is missing required columns: {sorted(missing)}")
    tokenized = tokenize_rows(frame, tokenizer, max_length)
    offsets = np.zeros(len(frame) + 1, dtype=np.int64)
    for index, item in enumerate(tokenized):
        offsets[index + 1] = offsets[index] + len(item["response_ids"])

    sidecar_target = output_path.with_suffix(".sidecar")
    if not overwrite and (output_path.exists() or sidecar_target.exists()):
        raise FileExistsError(
            f"Refusing to replace {output_path} or {sidecar_target}; pass --overwrite"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_root = Path(tempfile.mkdtemp(prefix=f".{output_path.stem}.tmp-", dir=output_path.parent))
    sidecar_tmp = temp_root / sidecar_target.name
    parquet_tmp = temp_root / output_path.name
    try:
        arrays = create_sidecar_arrays(sidecar_tmp, offsets, topk)
        arrays["token_ids"][:] = np.concatenate(
            [item["response_ids"] for item in tokenized], axis=0
        )
        for chunk_start in range(0, len(tokenized), request_batch_size):
            chunk = tokenized[chunk_start:chunk_start + request_batch_size]
            prompts = [{"prompt_token_ids": item["full_ids"]} for item in chunk]
            outputs = llm.generate(prompts, sampling, use_tqdm=False)
            if len(outputs) != len(chunk):
                raise RuntimeError("vLLM returned a different number of KD scoring outputs")
            for relative_index, (item, output) in enumerate(zip(chunk, outputs)):
                row_index = chunk_start + relative_index
                if list(output.prompt_token_ids or []) != item["full_ids"]:
                    raise ValueError(f"Teacher tokenization changed for KD row {row_index}")
                positions = output.prompt_logprobs
                if positions is None or len(positions) != len(item["full_ids"]):
                    raise ValueError(f"Missing or misaligned prompt logprobs for KD row {row_index}")
                start = int(offsets[row_index])
                for response_index, _target_id in enumerate(item["response_ids"]):
                    distribution = positions[item["prompt_length"] + response_index]
                    ids, logprobs, count, mass = extract_topk(distribution, topk)
                    token_index = start + response_index
                    arrays["topk_ids"][token_index] = ids
                    arrays["topk_logprobs"][token_index] = logprobs
                    arrays["topk_counts"][token_index] = count
                    arrays["topk_mass"][token_index] = mass
            completed = min(chunk_start + len(chunk), len(tokenized))
            print(f"[KD scorer] {output_path.name}: {completed}/{len(tokenized)} rows", flush=True)

        for array in arrays.values():
            array.flush()
        source_counts = {str(key): int(value) for key, value in frame["source"].value_counts().items()}
        manifest = {
            "schema_version": KD_SCHEMA_VERSION,
            "teacher_model": teacher_model,
            "tokenizer_fingerprint": tokenizer_hash,
            "topk": topk,
            "rows": len(frame),
            "tokens": int(offsets[-1]),
            "max_length": max_length,
            "source_counts": source_counts,
            "input": str(input_path),
            "storage": {
                "token_ids": "int32",
                "topk_ids": "int32",
                "topk_logprobs": "float16",
                "topk_counts": "uint8",
                "topk_mass": "float16",
            },
        }
        (sidecar_tmp / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        frame["kd_row_index"] = np.arange(len(frame), dtype=np.int64)
        frame["kd_token_start"] = offsets[:-1]
        frame["kd_token_count"] = np.diff(offsets)
        frame["kd_sidecar"] = sidecar_target.name
        frame["kd_schema_version"] = KD_SCHEMA_VERSION
        frame["teacher_model"] = teacher_model
        frame["teacher_tokenizer_fingerprint"] = tokenizer_hash
        frame["teacher_topk"] = topk
        frame.to_parquet(parquet_tmp, index=False)

        if overwrite:
            if sidecar_target.is_dir():
                shutil.rmtree(sidecar_target)
            if output_path.is_file():
                output_path.unlink()
        sidecar_tmp.rename(sidecar_target)
        parquet_tmp.rename(output_path)
        print(
            f"Wrote {len(frame)} rows and {int(offsets[-1])} KD tokens to {output_path}",
            flush=True,
        )
    finally:
        shutil.rmtree(temp_root, ignore_errors=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-input", type=Path, required=True)
    parser.add_argument("--val-input", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-model", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--student-tokenizer", default="Qwen/Qwen2.5-Coder-1.5B-Instruct")
    parser.add_argument("--topk", type=int, default=32)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--tensor-parallel-size", type=int, default=4)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    parser.add_argument("--max-num-batched-tokens", type=int, default=32768)
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--request-batch-size", type=int, default=128)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 1 <= args.topk <= 255:
        raise SystemExit("--topk must be in [1, 255]")
    if args.request_batch_size <= 0:
        raise SystemExit("--request-batch-size must be positive")

    teacher_tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    student_tokenizer = AutoTokenizer.from_pretrained(args.student_tokenizer)
    teacher_hash = tokenizer_fingerprint(teacher_tokenizer)
    student_hash = tokenizer_fingerprint(student_tokenizer)
    if teacher_hash != student_hash:
        raise RuntimeError(
            "Teacher and student tokenizers are not identical; sparse token KD cannot align them"
        )

    from vllm import LLM, SamplingParams

    llm = LLM(
        model=args.teacher_model,
        tokenizer=args.teacher_model,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_length + 1,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        dtype="bfloat16",
        enforce_eager=False,
        attention_config={"backend": "FLASHINFER"},
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        max_tokens=1,
        prompt_logprobs=args.topk,
    )
    try:
        score_split(
            llm=llm,
            sampling=sampling,
            tokenizer=teacher_tokenizer,
            tokenizer_hash=teacher_hash,
            teacher_model=args.teacher_model,
            input_path=args.train_input,
            output_path=args.output_dir / "teacher_kd_train.parquet",
            topk=args.topk,
            max_length=args.max_length,
            request_batch_size=args.request_batch_size,
            overwrite=args.overwrite,
        )
        score_split(
            llm=llm,
            sampling=sampling,
            tokenizer=teacher_tokenizer,
            tokenizer_hash=teacher_hash,
            teacher_model=args.teacher_model,
            input_path=args.val_input,
            output_path=args.output_dir / "teacher_kd_val.parquet",
            topk=args.topk,
            max_length=args.max_length,
            request_batch_size=args.request_batch_size,
            overwrite=args.overwrite,
        )
    finally:
        engine = getattr(getattr(llm, "llm_engine", None), "engine_core", None)
        if getattr(engine, "shutdown", None):
            engine.shutdown()
        del llm
        gc.collect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
