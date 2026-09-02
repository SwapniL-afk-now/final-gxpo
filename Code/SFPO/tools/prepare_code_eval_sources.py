#!/usr/bin/env python3
"""Download and normalize coding benchmark sources for the local evaluator."""
from __future__ import annotations
import argparse, ast, json
import re
from pathlib import Path
import pandas as pd
from huggingface_hub import hf_hub_download

HUMAN_REPO = "evalplus/humanevalplus"
HUMAN_FILE = "data/test-00000-of-00001-5973903632b82d40.parquet"
MBPP_REPO = "evalplus/mbppplus"
MBPP_FILE = "data/test-00000-of-00001-d5781c9c51e02795.parquet"
LCB_REPO = "livecodebench/code_generation_lite"

def msg_prompt(text):
    return [
        {"role": "system", "content": "You are an expert Python programmer. Return only correct Python code."},
        {"role": "user", "content": str(text).strip()},
    ]

def mbpp_entry_point(code: str, test: str | None = None) -> str:
    """Return MBPP's tested function name from canonical code and tests."""
    tree = ast.parse(str(code))
    function_names = [
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]
    if test is not None and len(function_names) > 1:
        counts = {
            name: len(re.findall(r"\b" + re.escape(name) + r"\s*\(", str(test)))
            for name in function_names
        }
        used = [name for name in function_names if counts[name] > 0]
        if used:
            return max(used, key=lambda name: counts[name])
    if len(function_names) == 1:
        return function_names[0]
    raise ValueError(f"could not infer tested MBPP function from {function_names!r}")

def make_eval_rows(out: Path):
    raw = out / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    hp = hf_hub_download(HUMAN_REPO, HUMAN_FILE, repo_type="dataset", local_dir=str(raw))
    mp = hf_hub_download(MBPP_REPO, MBPP_FILE, repo_type="dataset", local_dir=str(raw))
    human = pd.read_parquet(hp)
    mbpp = pd.read_parquet(mp)
    human_rows = []
    for row in human.itertuples(index=False):
        gt = {"prompt": row.prompt, "test": row.test, "entry_point": row.entry_point}
        human_rows.append({
            "prompt": msg_prompt("Complete this function:\n\n" + row.prompt),
            "data_source": "humanevalplus",
            "ground_truth": json.dumps(gt),
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {"task_id": row.task_id, "entry_point": row.entry_point},
        })
    mbpp_rows = []
    for row in mbpp.itertuples(index=False):
        gt = {"prompt": "", "test": str(row.test), "entry_point": mbpp_entry_point(row.code, row.test)}
        mbpp_rows.append({
            "prompt": msg_prompt("Write a Python function for this task:\n\n" + row.prompt),
            "data_source": "mbppplus",
            "ground_truth": json.dumps(gt),
            "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
            "extra_info": {"task_id": row.task_id},
        })
    pd.DataFrame(human_rows).to_parquet(raw / "humanevalplus.parquet", index=False)
    pd.DataFrame(mbpp_rows).to_parquet(raw / "mbppplus.parquet", index=False)

    lcb_rows = []
    for index in range(1, 7):
        path = hf_hub_download(
            LCB_REPO, "test.jsonl" if index == 1 else f"test{index}.jsonl",
            repo_type="dataset", local_dir=str(raw),
        )
        with open(path) as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                try:
                    tests = json.loads(record["public_test_cases"])
                except (TypeError, json.JSONDecodeError, KeyError):
                    continue
                tests = [item for item in tests if item.get("testtype") == "stdin"]
                if not tests:
                    continue
                gt = {"inputs": [item["input"] for item in tests], "outputs": [item["output"] for item in tests]}
                prompt = (
                    "Solve this competitive-programming problem in Python. "
                    "Read from stdin and write to stdout. Return only a complete program.\n\n"
                    + str(record.get("question_content", ""))
                )
                lcb_rows.append({
                    "prompt": msg_prompt(prompt),
                    "data_source": "livecodebench",
                    "ground_truth": json.dumps(gt),
                    "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
                    "extra_info": {"question_id": record.get("question_id"), "contest_date": record.get("contest_date"), "platform": record.get("platform")},
                    "contest_date": record.get("contest_date"),
                })
    pd.DataFrame(lcb_rows).drop_duplicates(subset=["prompt"]).to_parquet(raw / "livecodebench.parquet", index=False)
    print(f"raw benchmarks: humanevalplus={len(human_rows)} mbppplus={len(mbpp_rows)} livecodebench={len(lcb_rows)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("/workspace/jepa-grpo-cache/eval_data/code_distill"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    make_eval_rows(args.output_dir)
