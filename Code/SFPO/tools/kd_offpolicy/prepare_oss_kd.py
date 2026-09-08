#!/usr/bin/env python3
"""W0: gpt-oss-low reasoning JSONL -> EOS-enforced off-policy KD parquet.

Reads math-oss-low-17k.filtered_resp3072.jsonl, re-verifies every length with
the REAL student tokenizer (Qwen2.5-1.5B-Instruct snapshot), enforces a single
trailing EOS (151645 <|im_end|>), and writes token-id columns plus texts and
reward metadata. No truncation of kept rows: violators are dropped and counted.

Output columns (lists are plain python lists of ints, parquet list<int64>):
  prompt_ids      : chat-templated prompt token ids (no EOS appended)
  response_ids    : response token ids, ALWAYS ending with exactly one EOS
  prompt_text     : chat-templated prompt string (debug)
  response_text   : response string as tokenized (debug)
  data_source     : e.g. math_dapo (verifier dispatch)
  ground_truth    : standard_answer (verifier)
  reward          : precomputed verifier score from the source file
  source_index    : extra_info index (provenance)
"""
from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

PROMPT_MAX = 1024
RESPONSE_MAX = 3072
TOTAL_MAX = 4096  # teacher context window: prompt + response must fit


def parse_prompt(raw) -> list:
    if isinstance(raw, list):
        return raw
    if isinstance(raw, str):
        try:
            msgs = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            raise ValueError(f"unparseable prompt field: {raw[:120]!r}")
        if isinstance(msgs, list):
            return msgs
    raise ValueError(f"unexpected prompt type {type(raw)}: {str(raw)[:120]!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--input", required=True, help="filtered_resp3072 jsonl")
    ap.add_argument("--out", required=True, help="output parquet path")
    ap.add_argument("--tokenizer-path", required=True)
    ap.add_argument("--eos-id", type=int, default=151645)
    a = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.tokenizer_path, trust_remote_code=True)
    assert tok.eos_token_id == a.eos_id, f"tokenizer eos {tok.eos_token_id} != {a.eos_id}"

    rows = {
        "prompt_ids": [], "response_ids": [], "prompt_text": [], "response_text": [],
        "data_source": [], "ground_truth": [], "reward": [], "source_index": [],
    }
    stats = {"read": 0, "kept": 0, "drop_prompt_long": 0, "drop_response_long": 0,
             "drop_total_long": 0, "drop_bad_row": 0, "eos_appended": 0,
             "eos_truncated_mid": 0, "eos_present": 0}
    resp_lens, prompt_lens = [], []

    with open(a.input, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            stats["read"] += 1
            try:
                r = json.loads(line)
                msgs = parse_prompt(r["prompt"])
                prompt_text = tok.apply_chat_template(
                    msgs, add_generation_prompt=True, tokenize=False)
                prompt_ids = tok.encode(prompt_text, add_special_tokens=False)
                resp_text = r["gpt-oss-120b-response"]
                if not resp_text or not resp_text.strip():
                    raise ValueError("empty response")
                resp_ids = tok.encode(resp_text, add_special_tokens=False)
            except (KeyError, ValueError, TypeError) as e:
                stats["drop_bad_row"] += 1
                continue

            if len(prompt_ids) > PROMPT_MAX:
                stats["drop_prompt_long"] += 1
                continue
            # Single termination: cut at first EOS when the trace already ends.
            if a.eos_id in resp_ids:
                first = resp_ids.index(a.eos_id)
                if first < len(resp_ids) - 1:
                    stats["eos_truncated_mid"] += 1
                resp_ids = resp_ids[:first + 1]
                stats["eos_present"] += 1
            else:
                # Reserve one slot for EOS instead of overflowing the cap.
                resp_ids = resp_ids[:RESPONSE_MAX - 1] + [a.eos_id]
                stats["eos_appended"] += 1
            if len(resp_ids) > RESPONSE_MAX:
                stats["drop_response_long"] += 1
                continue
            if len(prompt_ids) + len(resp_ids) > TOTAL_MAX:
                stats["drop_total_long"] += 1
                continue

            stats["kept"] += 1
            prompt_lens.append(len(prompt_ids))
            resp_lens.append(len(resp_ids))
            rows["prompt_ids"].append(prompt_ids)
            rows["response_ids"].append(resp_ids)
            rows["prompt_text"].append(prompt_text)
            rows["response_text"].append(resp_text)
            # Verifier dispatch only knows 'dapo_math' (math_verify); the oss
            # rows label the same DAPO distribution as 'math_dapo'.
            src = r.get("data_source", "math_dapo")
            rows["data_source"].append("dapo_math" if src == "math_dapo" else src)
            rows["ground_truth"].append(str(r.get("standard_answer", "")))
            rows["reward"].append(float(r.get("reward", 0.0)))
            try:
                extra = ast.literal_eval(r.get("extra_info", "{}"))
                rows["source_index"].append(str(extra.get("index", "")))
            except (SyntaxError, ValueError):
                rows["source_index"].append("")

    assert stats["kept"] > 0, "no rows kept"
    assert all(ids[-1] == a.eos_id for ids in rows["response_ids"]), "EOS missing!"

    import statistics
    print(json.dumps({"counts": stats,
                      "prompt_len": {"mean": statistics.mean(prompt_lens),
                                     "p50": statistics.median(prompt_lens),
                                     "max": max(prompt_lens)},
                      "response_len": {"mean": statistics.mean(resp_lens),
                                       "p50": statistics.median(resp_lens),
                                       "max": max(resp_lens)}}, indent=2))

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.table({k: pa.array(v) for k, v in rows.items()})
    pq.write_table(table, out)
    print(f"wrote {stats['kept']} rows -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
