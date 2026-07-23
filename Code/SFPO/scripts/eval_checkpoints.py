# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Offline benchmark evaluation of saved verl checkpoints (GXPO paper Tables 1/3/4).

For each global_step_* checkpoint in a run directory, loads the policy
(merging the LoRA adapter into the base model when present), generates n
samples per prompt with vLLM at temperature 1.0, scores them with the SAME
reward functions used during training (verl.utils.reward_score), and appends
per-benchmark avg pass@1 and pass@n rows to a CSV. Budget columns
(cumulative backward passes, elapsed hours) are joined from the run's
metrics.jsonl so iso-BP / iso-wall-clock tables can be built directly.

Usage:
  python scripts/eval_checkpoints.py \
      --run_dir checkpoints/my_gxpo_run \
      --base_model Qwen/Qwen2.5-1.5B-Instruct \
      --data ~/data/gxpo/eval/math500/test.parquet ~/data/gxpo/eval/gsm8k/test.parquet \
      --n 16 --out eval_results.csv
"""

import argparse
import csv
import gc
import glob
import json
import os
import re
import subprocess
import sys
import tempfile

import pandas as pd
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

CORRECT_THRESHOLD = 0.95  # same correctness cut as ray_trainer._validate


def load_budgets(run_dir):
    """step -> (cumulative_bp, elapsed_hours) from the run's metrics.jsonl."""
    budgets = {}
    path = os.path.join(run_dir, 'metrics.jsonl')
    if not os.path.exists(path):
        return budgets
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            budgets[int(row['step'])] = (row.get('actor/cumulative_bp'), row.get('train/elapsed_hours'))
    return budgets


def find_checkpoints(run_dir, steps=None):
    ckpts = []
    for d in glob.glob(os.path.join(run_dir, 'global_step_*')):
        m = re.match(r'.*global_step_(\d+)$', d)
        if m:
            ckpts.append((int(m.group(1)), os.path.join(d, 'actor')))
    ckpts.sort()
    if steps:
        keep = set(steps)
        ckpts = [(s, p) for s, p in ckpts if s in keep]
    return ckpts


def materialize_hf_model(actor_dir, base_model, workdir):
    """Return an HF model dir for this checkpoint (merge LoRA or merge FSDP shards)."""
    adapter_dir = os.path.join(actor_dir, 'lora_adapter')
    if os.path.isdir(adapter_dir):
        from transformers import AutoModelForCausalLM
        from peft import PeftModel
        assert base_model, '--base_model is required for LoRA checkpoints'
        model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.bfloat16)
        model = PeftModel.from_pretrained(model, adapter_dir)
        model = model.merge_and_unload()
        out_dir = os.path.join(workdir, 'merged')
        model.save_pretrained(out_dir)
        del model
        gc.collect()
        return out_dir

    hf_dir = os.path.join(actor_dir, 'huggingface')
    weight_files = glob.glob(os.path.join(hf_dir, '*.safetensors')) + glob.glob(os.path.join(hf_dir, '*.bin'))
    if not weight_files:
        # full-FT sharded checkpoint: merge DTensor shards into hf_dir
        merger = os.path.join(os.path.dirname(__file__), 'model_merger.py')
        subprocess.run([sys.executable, merger, '--local_dir', actor_dir], check=True)
    return hf_dir


def evaluate_checkpoint(model_dir, tokenizer_dir, data_files, n, temperature, max_tokens, tp,
                        gpu_memory_utilization):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from verl.utils.reward_score import _default_compute_score

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    llm = LLM(model=model_dir,
              tokenizer=tokenizer_dir,
              tensor_parallel_size=tp,
              gpu_memory_utilization=gpu_memory_utilization,
              dtype='bfloat16')
    sampling = SamplingParams(n=n, temperature=temperature, top_p=1.0, max_tokens=max_tokens)

    results = {}
    for data_file in data_files:
        df = pd.read_parquet(os.path.expanduser(data_file))
        prompts = [
            tokenizer.apply_chat_template(list(chat), tokenize=False, add_generation_prompt=True)
            for chat in df['prompt']
        ]
        outputs = llm.generate(prompts, sampling)

        pass1_sum, passn_sum = 0.0, 0.0
        data_source = df['data_source'].iloc[0]
        for row_idx, out in enumerate(outputs):
            ground_truth = df['reward_model'].iloc[row_idx]['ground_truth']
            prompt_str = prompts[row_idx]
            correct = [
                float(_default_compute_score(prompt_str, data_source, o.text, ground_truth)) >= CORRECT_THRESHOLD
                for o in out.outputs
            ]
            pass1_sum += sum(correct) / len(correct)
            passn_sum += float(any(correct))
        num_prompts = len(outputs)
        results[data_source] = (pass1_sum / num_prompts, passn_sum / num_prompts, num_prompts)
        print(f'  {data_source}: pass@1={results[data_source][0]:.4f} '
              f'pass@{n}={results[data_source][1]:.4f} ({num_prompts} prompts)')

    del llm
    gc.collect()
    torch.cuda.empty_cache()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', required=True)
    parser.add_argument('--base_model', default=None, help='HF base model (required for LoRA checkpoints)')
    parser.add_argument('--data', nargs='+', required=True, help='eval parquet files')
    parser.add_argument('--n', type=int, default=16)
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--max_tokens', type=int, default=3072)
    parser.add_argument('--tp', type=int, default=1)
    parser.add_argument('--gpu_memory_utilization', type=float, default=0.85)
    parser.add_argument('--steps', type=int, nargs='*', default=None, help='only these global steps')
    parser.add_argument('--out', default=None, help='output CSV (default: <run_dir>/eval_results.csv)')
    args = parser.parse_args()

    run_dir = os.path.expanduser(args.run_dir)
    out_csv = args.out or os.path.join(run_dir, 'eval_results.csv')
    budgets = load_budgets(run_dir)
    ckpts = find_checkpoints(run_dir, args.steps)
    if not ckpts:
        sys.exit(f'No global_step_* checkpoints found under {run_dir}')
    print(f'Evaluating {len(ckpts)} checkpoints from {run_dir}')

    write_header = not os.path.exists(out_csv)
    with open(out_csv, 'a', newline='') as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(['run', 'step', 'cumulative_bp', 'elapsed_hours', 'benchmark',
                             f'pass1_avg', f'pass{args.n}', 'num_prompts'])

        for step, actor_dir in ckpts:
            print(f'== global_step_{step} ==')
            with tempfile.TemporaryDirectory() as workdir:
                model_dir = materialize_hf_model(actor_dir, args.base_model, workdir)
                tokenizer_dir = args.base_model or model_dir
                results = evaluate_checkpoint(model_dir, tokenizer_dir, args.data, args.n,
                                              args.temperature, args.max_tokens, args.tp,
                                              args.gpu_memory_utilization)
            bp, hours = budgets.get(step, (None, None))
            for benchmark, (p1, pn, num) in results.items():
                writer.writerow([os.path.basename(run_dir), step, bp, hours, benchmark, p1, pn, num])
            f.flush()

    print(f'Wrote {out_csv}')


if __name__ == '__main__':
    main()
