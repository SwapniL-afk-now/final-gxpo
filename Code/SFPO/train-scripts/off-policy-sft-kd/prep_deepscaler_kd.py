#!/usr/bin/env python
"""Cache DeepScaleR problem/solution pairs as a train/validation parquet pair.

Shared by both off-policy-sft-kd launchers, which each used to carry their own
copy of this as an inline heredoc that built the train split only.

Only 7,391 of DeepScaleR's 40,315 rows carry a `solution`; the other 32,924 hold
a bare final answer ('27', '-4') with solution == ''. Those rows have no response
text, so there is nothing to compute CE or a matched-token KL against and they are
dropped -- the teacher here only runs forward passes, it does not generate targets.

Usage: prep_deepscaler_kd.py DATASET_ID OUT_DIR TOKENIZER MAX_LENGTH VAL_SIZE SEED
"""
import os
import random
import sys

from datasets import load_dataset
from transformers import AutoTokenizer

dataset_id, out_dir, tokenizer_path, max_length, val_size, seed = sys.argv[1:7]
max_length, val_size, seed = int(max_length), int(val_size), int(seed)

train_out = os.path.join(out_dir, f'train_solution_max{max_length}.parquet')
val_out = os.path.join(out_dir, f'val_solution_max{max_length}_{val_size}.parquet')
if os.path.exists(train_out) and os.path.exists(val_out):
    print(f'reusing {train_out} and {val_out}')
    raise SystemExit(0)

dataset = load_dataset(dataset_id, split='train')
missing = {'problem', 'solution'}.difference(dataset.column_names)
if missing:
    raise SystemExit(f'dataset {dataset_id} is missing columns: {sorted(missing)}')
if not len(dataset):
    raise SystemExit(f'dataset {dataset_id} has no rows')

tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
eos = tokenizer.eos_token or ''
kept = []
dropped_empty = 0
dropped_long = 0
for index, row in enumerate(dataset):
    problem, solution = row['problem'], row['solution']
    if (not isinstance(problem, str) or not problem.strip()
            or not isinstance(solution, str) or not solution.strip()):
        dropped_empty += 1
        continue
    prompt = tokenizer.apply_chat_template(
        [{'role': 'user', 'content': problem}], add_generation_prompt=True, tokenize=False)
    length = len(tokenizer(prompt, add_special_tokens=False)['input_ids'])
    response = solution if eos and solution.endswith(eos) else solution + eos
    length += len(tokenizer(response, add_special_tokens=False)['input_ids'])
    if length > max_length:
        dropped_long += 1
        continue
    kept.append(index)
if len(kept) <= val_size:
    raise SystemExit(
        f'dataset {dataset_id} kept only {len(kept)} rows at max_length={max_length}, '
        f'which does not leave a train split after a {val_size}-row holdout')

# Held-out validation, so val/loss measures generalization rather than memorization.
random.Random(seed).shuffle(kept)
splits = {val_out: kept[:val_size], train_out: kept[val_size:]}
for output, indices in splits.items():
    subset = dataset.select(sorted(indices))
    temporary = f'{output}.tmp.{os.getpid()}'
    subset.to_parquet(temporary)
    os.replace(temporary, output)
print(f'cached {dataset_id}: kept={len(kept)} dropped_empty={dropped_empty} '
      f'dropped_over_{max_length}={dropped_long} -> '
      f'train={len(splits[train_out])} val={len(splits[val_out])}')
