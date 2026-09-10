#!/usr/bin/env python
"""Cache DeepSeek-R1-Distill-Qwen-32B reasoning traces as train/val parquets.

Each upstream row holds ONE problem plus a LIST of sampled traces with a
per-trace correctness flag. This explodes each row into one training row per
CORRECT, non-empty trace: (problem, solution=trace). A problem with several
correct traces therefore yields several rows sharing one prompt -- the SFT
trainer consumes rows independently (row-based SFTDataset, no prompt-uniqueness
assumption), so that is natively supported.

Length gate mirrors the old DeepScaleR prep: prompt (chat-templated) + trace +
eos in student tokens must fit MAX_LENGTH, else the row is dropped (the trainer
runs truncation=error and would raise instead).

Validation holdout is at PROBLEM level (seed-shuffled), so no prompt appears in
both splits.

Usage: prep_32b_traces_kd.py DATASET_ID OUT_DIR TOKENIZER MAX_LENGTH VAL_SIZE SEED
"""
import os
import random
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

dataset_id, out_dir, tokenizer_path, max_length, val_size, seed = sys.argv[1:7]
max_length, val_size, seed = int(max_length), int(val_size), int(seed)

train_out = os.path.join(out_dir, f'train_32b_correct_max{max_length}.parquet')
val_out = os.path.join(out_dir, f'val_32b_correct_max{max_length}_{val_size}.parquet')
if os.path.exists(train_out) and os.path.exists(val_out):
    print(f'reusing {train_out} and {val_out}')
    raise SystemExit(0)

dataset = load_dataset(dataset_id, split='train')
if not len(dataset):
    raise SystemExit(f'dataset {dataset_id} has no rows')

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
eos = tokenizer.eos_token or ''

TRACE_KEY = 'DeepSeek-R1-Distill-Qwen-32B_reasoning'
FLAG_KEY = 'DeepSeek-R1-Distill-Qwen-32B_correct'

kept = []  # (problem_text, answer, trace)
dropped_no_correct = 0
dropped_empty = 0
dropped_long = 0
for problem_idx, row in enumerate(dataset):
    traces = row[TRACE_KEY]
    flags = row[FLAG_KEY]
    if not isinstance(traces, list):
        traces, flags = [traces], [flags]
    problem_rows = []
    for trace, flag in zip(traces, flags):
        if not flag:
            continue
        if not isinstance(trace, str) or not trace.strip():
            dropped_empty += 1
            continue
        prompt = tokenizer.apply_chat_template(
            [{'role': 'user', 'content': row['problem']}],
            add_generation_prompt=True, tokenize=False)
        length = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
        response = trace if eos and trace.endswith(eos) else trace + eos
        length += len(tokenizer(response, add_special_tokens=False)['input_ids'])
        if length > max_length:
            dropped_long += 1
            continue
        problem_rows.append((row['problem'], row['answer'], trace))
    if not problem_rows:
        dropped_no_correct += 1
    kept.extend(problem_rows)
    if (problem_idx + 1) % 5000 == 0:
        print(f'...{problem_idx + 1}/{len(dataset)} kept={len(kept)}', flush=True)

# Problem-level holdout: no prompt in both splits.
prompts = sorted({p for p, _, _ in kept})
random.Random(seed).shuffle(prompts)
val_prompts, val_count = set(), 0
counts = {}
for p, _, _ in kept:
    counts[p] = counts.get(p, 0) + 1
for p in prompts:
    if val_count >= val_size:
        break
    val_prompts.add(p)
    val_count += counts[p]

from datasets import Dataset as HFDS


def rows_for(pred):
    cols = {'problem': [], 'answer': [], 'solution': []}
    for p, a, t in kept:
        if pred(p):
            cols['problem'].append(p)
            cols['answer'].append(a)
            cols['solution'].append(t)
    return cols


os.makedirs(out_dir, exist_ok=True)
sizes = {}
for output, pred in ((val_out, val_prompts.__contains__),
                     (train_out, lambda p: p not in val_prompts)):
    subset = HFDS.from_dict(rows_for(pred))
    temporary = f'{output}.tmp.{os.getpid()}'
    subset.to_parquet(temporary)
    os.replace(temporary, output)
    sizes[output] = len(subset)
print(f'cached {dataset_id}: kept={len(kept)} '
      f'dropped_no_correct_problem={dropped_no_correct} '
      f'dropped_empty={dropped_empty} dropped_over_{max_length}={dropped_long} '
      f'-> train={sizes[train_out]} val={sizes[val_out]}')
