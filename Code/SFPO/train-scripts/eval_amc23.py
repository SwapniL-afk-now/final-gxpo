"""Score an HF checkpoint on AMC23 with avg@n / pass@n over several seeds.

Produces the same quantities the RL runs log as val/avg_at_8 and val/pass_at_8, using the
same per-sample seeding scheme as verl's validation rollout
(`vllm_rollout_spmd.py`: sp.seed = gen_seed * 100003 + i), so SFT checkpoints can be put
straight next to the RL arms.

verl's own main_eval.py is not usable here: it hardcodes pass@5, has no avg@n, no seeds,
and its select_reward_fn raises NotImplementedError for data_source 'amc23'.

Usage:
  python eval_amc23.py --model <hf_dir_or_name> [--n 8] [--seeds 0,1,2]
"""
import argparse
import os

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, help='HF checkpoint dir (or hub id)')
    ap.add_argument('--data', default='/workspace/jepa-grpo-cache/eval_data/amc23.parquet')
    ap.add_argument('--n', type=int, default=8, help='responses per prompt')
    ap.add_argument('--seeds', default='0,1,2')
    ap.add_argument('--temperature', type=float, default=1.0)
    ap.add_argument('--top-p', type=float, default=1.0)
    ap.add_argument('--prompt-length', type=int, default=1024)
    ap.add_argument('--response-length', type=int, default=3072)
    ap.add_argument('--gpu-memory-utilization', type=float, default=0.85)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from verl.utils.reward_score import _default_compute_score

    df = pd.read_parquet(args.data)
    seeds = [int(s) for s in args.seeds.split(',')]
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    # same chat formatting the trainers use
    prompts = [
        tokenizer.apply_chat_template(list(p), add_generation_prompt=True, tokenize=False)
        for p in df['prompt']
    ]
    ground_truths = [r['ground_truth'] for r in df['reward_model']]
    data_sources = df['data_source'].tolist()

    llm = LLM(model=args.model,
              dtype='bfloat16',
              gpu_memory_utilization=args.gpu_memory_utilization,
              max_model_len=args.prompt_length + args.response_length,
              enforce_eager=True,
              seed=0)

    print(f'{len(prompts)} prompts x n={args.n} x seeds={seeds}')
    per_seed = {}
    for gen_seed in seeds:
        # one SamplingParams per (prompt, sample) so the n copies stay diverse -- a single
        # shared seed collapses pass@n to pass@1
        flat_prompts, params = [], []
        for i, prompt in enumerate(prompts):
            for j in range(args.n):
                flat_prompts.append(prompt)
                params.append(
                    SamplingParams(n=1,
                                   temperature=args.temperature,
                                   top_p=args.top_p,
                                   max_tokens=args.response_length,
                                   seed=gen_seed * 100003 + i * args.n + j))

        outputs = llm.generate(flat_prompts, params, use_tqdm=True)
        texts = [o.outputs[0].text for o in outputs]

        # (num_prompts, n) score matrix
        correct = np.zeros((len(prompts), args.n))
        for i in range(len(prompts)):
            for j in range(args.n):
                score = _default_compute_score(prompts[i], data_sources[i],
                                               texts[i * args.n + j], ground_truths[i])
                correct[i, j] = float(score) > 0.95

        avg_at_n = float(correct.mean())
        pass_at_n = float(correct.max(axis=1).mean())
        per_seed[gen_seed] = (avg_at_n, pass_at_n)
        print(f'  seed {gen_seed}: avg@{args.n}={avg_at_n:.4f}  pass@{args.n}={pass_at_n:.4f}')

    avgs = [v[0] for v in per_seed.values()]
    passes = [v[1] for v in per_seed.values()]
    print(f'\nmodel: {args.model}')
    print(f'avg@{args.n}  mean={np.mean(avgs):.4f}  std={np.std(avgs):.4f}')
    print(f'pass@{args.n} mean={np.mean(passes):.4f}  std={np.std(passes):.4f}')


if __name__ == '__main__':
    main()
