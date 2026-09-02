#!/usr/bin/env python3
"""Prepare deterministic, executable coding-distillation prompt/target data.

The script has two stages:

1. Build a prompt pool from TACO-verified and write ``teacher_prompts.parquet``.
2. After teacher generation, pass ``--teacher-jsonl`` to materialize the SFT
   corpus from only fully verified teacher trajectories.

Teacher generation is intentionally separate so the teacher never occupies GPU
memory during FSDP student training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer


DEFAULT_OUT = Path("/workspace/jepa-grpo-cache/data/code_distill")
DEFAULT_TOKENIZER = "Qwen/Qwen2.5-Coder-1.5B-Instruct"


def stable_id(value: object) -> str:
    return hashlib.sha256(str(value).strip().encode("utf-8")).hexdigest()


def canonical_prompt(value: object) -> str:
    if isinstance(value, str):
        text = value
    else:
        messages = value.tolist() if hasattr(value, "tolist") else value
        text = "\n".join(str(message.get("content", "")) for message in messages)
    text = text.replace(
        "Respond with a complete Python program in a single ```python code block.", ""
    )
    return " ".join(text.split())


def make_prompt(question: object) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "You are an expert competitive programmer. Solve the problem "
                "in Python and ensure the program reads stdin and writes stdout."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{str(question).strip()}\n\n"
                "Respond with a complete Python program in a single ```python code block."
            ),
        },
    ]


def parse_io(value: object) -> dict | None:
    if not value or not str(value).strip():
        return None
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return None
    if not parsed.get("inputs") or not parsed.get("outputs"):
        return None
    if parsed.get("fn_name"):
        return None
    return {
        "inputs": list(parsed["inputs"][:15]),
        "outputs": list(parsed["outputs"][:15]),
    }


def first_dataset_solution(value: object) -> str:
    """Return one canonical TACO solution for fallback distillation."""
    if value is None:
        return ""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        for solution in value:
            text = str(solution).strip()
            if text:
                return text
        return ""
    return str(value).strip()


def build_prompt_rows(args: argparse.Namespace) -> list[dict]:
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)
    taco_path = args.taco_parquet
    if taco_path is None:
        taco_path = hf_hub_download(
            "likaixin/TACO-verified",
            "default/train/0000.parquet",
            repo_type="dataset",
            revision="refs/convert/parquet",
        )

    frame = pd.read_parquet(taco_path)
    rows: list[dict] = []
    seen: set[str] = set()
    for _, row in frame.iterrows():
        question = str(row.get("question", "")).strip()
        io_spec = parse_io(row.get("input_output"))
        if not question or io_spec is None:
            continue
        problem_id = str(row.get("id", row.get("problem_id", len(rows))))
        problem_hash = stable_id(question)
        if problem_hash in seen:
            continue
        seen.add(problem_hash)
        prompt = make_prompt(question)
        prompt_tokens = len(
            tokenizer.apply_chat_template(
                prompt, add_generation_prompt=True, tokenize=True
            )
        )
        if prompt_tokens > args.max_prompt_tokens:
            continue
        split = "validation" if int(problem_hash[:8], 16) % 100 < args.val_percent else "train"
        rows.append(
            {
                "problem_id": problem_id,
                "problem_hash": problem_hash,
                "canonical_hash": stable_id(canonical_prompt(prompt)),
                "source": "taco_verified",
                "data_source": "taco",
                "split": split,
                "prompt": prompt,
                "ground_truth": json.dumps(io_spec, ensure_ascii=False),
                "reward_model": {"style": "rule", "ground_truth": json.dumps(io_spec, ensure_ascii=False)},
                "extra_info": {"problem_id": problem_id, "split": split},
                "prompt_tokens": prompt_tokens,
                "dataset_response": first_dataset_solution(row.get("solutions")),
            }
        )
        if args.max_problems and len(rows) >= args.max_problems:
            break
    return rows


def build_sft_rows(prompt_frame: pd.DataFrame, teacher_jsonl: Path) -> pd.DataFrame:
    prompts = {
        str(row.problem_id): row
        for row in prompt_frame.itertuples(index=False)
    }
    rows: list[dict] = []
    teacher_problem_ids: set[str] = set()
    with teacher_jsonl.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            record = json.loads(line)
            problem_id = str(record.get("problem_id", ""))
            prompt_row = prompts.get(problem_id)
            if prompt_row is None or not record.get("verified", False):
                continue
            rows.append(
                {
                    "problem_id": problem_id,
                    "problem_hash": prompt_row.problem_hash,
                    "canonical_hash": prompt_row.canonical_hash,
                    "source": "taco_verified_teacher",
                    "split": prompt_row.split,
                    "prompt": prompt_row.prompt,
                    "response": str(record["response"]),
                    "teacher_score": float(record.get("score", 1.0)),
                    "teacher_sample_index": int(record.get("sample_index", 0)),
                }
            )
            teacher_problem_ids.add(problem_id)

    # Keep every prompt in the distillation corpus. If sampling the teacher did
    # not produce a verified completion for a prompt, use TACO's own verified
    # solution as the single fallback target for that prompt.
    fallback_count = 0
    for prompt_row in prompt_frame.itertuples(index=False):
        problem_id = str(prompt_row.problem_id)
        if problem_id in teacher_problem_ids:
            continue
        response = str(getattr(prompt_row, "dataset_response", "") or "").strip()
        if not response:
            continue
        rows.append(
            {
                "problem_id": problem_id,
                "problem_hash": prompt_row.problem_hash,
                "canonical_hash": prompt_row.canonical_hash,
                "source": "taco_dataset_solution_fallback",
                "split": prompt_row.split,
                "prompt": prompt_row.prompt,
                "response": response,
                "teacher_score": 1.0,
                "teacher_sample_index": -1,
            }
        )
        fallback_count += 1
    if not rows:
        raise RuntimeError(f"No verified teacher trajectories found in {teacher_jsonl}")
    print(f"Added {fallback_count} TACO solution fallback rows", flush=True)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--tokenizer", default=DEFAULT_TOKENIZER)
    parser.add_argument("--taco-parquet", type=Path)
    parser.add_argument("--teacher-jsonl", type=Path)
    parser.add_argument("--max-problems", type=int, default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=1024)
    parser.add_argument("--val-percent", type=int, default=10)
    args = parser.parse_args()
    if not 0 <= args.val_percent < 100:
        raise SystemExit("--val-percent must be in [0, 100)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    prompt_frame = pd.DataFrame(build_prompt_rows(args))
    if prompt_frame.empty:
        raise SystemExit("No valid stdin/stdout TACO prompts were produced")
    prompt_path = args.output_dir / "teacher_prompts.parquet"
    prompt_frame.to_parquet(prompt_path, index=False)
    prompt_frame[prompt_frame["split"] == "train"].reset_index(drop=True).to_parquet(
        args.output_dir / "teacher_prompts_train.parquet", index=False
    )
    prompt_frame[prompt_frame["split"] == "validation"].reset_index(drop=True).to_parquet(
        args.output_dir / "teacher_prompts_val.parquet", index=False
    )

    manifest = {
        "schema_version": 1,
        "source": "likaixin/TACO-verified",
        "source_revision": "refs/convert/parquet",
        "tokenizer": args.tokenizer,
        "max_prompt_tokens": args.max_prompt_tokens,
        "validation_percent": args.val_percent,
        "rows": len(prompt_frame),
        "train_rows": int((prompt_frame["split"] == "train").sum()),
        "validation_rows": int((prompt_frame["split"] == "validation").sum()),
        "problem_hashes": sorted(prompt_frame["problem_hash"].tolist()),
        "canonical_hashes": sorted(prompt_frame["canonical_hash"].tolist()),
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {len(prompt_frame)} prompt rows to {prompt_path}")

    if args.teacher_jsonl:
        sft_frame = build_sft_rows(prompt_frame, args.teacher_jsonl)
        for split in ("train", "validation"):
            split_frame = sft_frame[sft_frame["split"] == split].reset_index(drop=True)
            sft_path = args.output_dir / f"teacher_sft_{'val' if split == 'validation' else 'train'}.parquet"
            split_frame.to_parquet(sft_path, index=False)
            print(f"Wrote {len(split_frame)} distillation rows to {sft_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
