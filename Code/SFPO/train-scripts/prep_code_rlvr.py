#!/usr/bin/env python
"""Build verl-format parquets for code RLVR (GRPO/GXPO rebuttal, non-math run).

TRAIN: likaixin/TACO-verified + codeparrot/apps (all/train), all difficulties.
       data_source in {'taco','apps'} -> prime_code (unit-test exec reward), unchanged.
EVAL : APPS introductory test, 200 subsample (disjoint from apps train split & from TACO).

STDIN-ONLY: call-based (fn_name) problems are dropped. TACO/APPS reference solutions
for call-based items are frequently stdin-style, so prime_code's function-call harness
scores even correct solutions 0 (verified: 4/4 ref sols -> 0.0, while 4/4 stdin -> 1.0).
FIXED 1024 PROMPT BUDGET: prompts are tokenized with the Qwen2.5-1.5B chat template and
only those <= MAX_PROMPT_TOK kept, so the runs use max_prompt_length=1024 with (near) zero
runtime drops. TACO alone is 9,799 <=1024, so APPS train is added to clear >=10k.
Tests capped (prime_code scores only the first 10 anyway; caps reward cost).
"""
import json, pandas as pd
from pathlib import Path
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

OUT = Path("/workspace/jepa-grpo-cache/data/code_rlvr"); OUT.mkdir(parents=True, exist_ok=True)
MAX_TESTS = 15
MAX_PROMPT_TOK = 1024
EVAL_N = 200
SYS = "You are an expert competitive programmer. Solve the problem below in Python."
TOK = AutoTokenizer.from_pretrained("/workspace/models/Qwen2.5-1.5B-Instruct")


def make_prompt(q):
    instr = ("Read from standard input and write the answer to standard output.\n"
             "Respond with a complete Python program in a single ```python code block.")
    return [{"role": "system", "content": SYS},
            {"role": "user", "content": f"{str(q).strip()}\n\n{instr}"}]


def row_to_rec(question, io_str, data_source, split, pid):
    if not io_str or not str(io_str).strip():
        return None
    try:
        d = json.loads(io_str)
    except Exception:
        return None
    if not d.get("inputs"):
        return None
    if d.get("fn_name"):                         # stdin-only (call-based scoring unreliable)
        return None
    prompt = make_prompt(question)
    n_tok = len(TOK.apply_chat_template(prompt, add_generation_prompt=True, tokenize=True))
    if n_tok > MAX_PROMPT_TOK:                    # fixed 1024 budget
        return None
    gt = {"inputs": d["inputs"][:MAX_TESTS], "outputs": d["outputs"][:MAX_TESTS]}
    return {
        "data_source": data_source,
        "prompt": prompt,
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": json.dumps(gt)},
        "extra_info": {"split": split, "problem_id": str(pid), "n_tests": len(gt["inputs"]),
                       "prompt_tok": n_tok},
    }


# ---- TRAIN: TACO-verified + APPS all/train (stdin-only, <=1024 tok) ----
train_rows, seen = [], set()

def add(question, io_str, source, split, pid):
    key = hash(str(question).strip())          # full-text dedup (prefix key over-merged)
    if key in seen:
        return
    rec = row_to_rec(question, io_str, source, split, pid)
    if rec:
        seen.add(key)
        train_rows.append(rec)

taco = pd.read_parquet(hf_hub_download("likaixin/TACO-verified", "default/train/0000.parquet",
                                       repo_type="dataset", revision="refs/convert/parquet"))
for _, r in taco.iterrows():
    add(r["question"], r["input_output"], "taco", "train", r["id"])

apps_tr = pd.read_parquet(hf_hub_download("codeparrot/apps", "all/train/0000.parquet",
                                          repo_type="dataset", revision="refs/convert/parquet"))
for _, r in apps_tr.iterrows():
    add(r["question"], r["input_output"], "apps", "train", r["problem_id"])

train = pd.DataFrame(train_rows).sample(frac=1, random_state=42).reset_index(drop=True)  # shuffle mix

# ---- EVAL: APPS introductory test ----
apps_te = pd.read_parquet(hf_hub_download("codeparrot/apps", "introductory/test/0000.parquet",
                                          repo_type="dataset", revision="refs/convert/parquet"))
eval_rows = []
for _, r in apps_te.iterrows():
    rec = row_to_rec(r["question"], r["input_output"], "apps", "test", r["problem_id"])
    if rec:
        eval_rows.append(rec)
test = pd.DataFrame(eval_rows)
if len(test) > EVAL_N:
    test = test.sample(n=EVAL_N, random_state=42).reset_index(drop=True)

train.to_parquet(OUT / "train.parquet")
test.to_parquet(OUT / "test.parquet")
print(f"TRAIN stdin<=1024 {len(train)} (taco+apps) -> {OUT/'train.parquet'}")
print(f"  by source: {train['data_source'].value_counts().to_dict()}")
print(f"EVAL  apps stdin<=1024 {len(test)} -> {OUT/'test.parquet'}")
