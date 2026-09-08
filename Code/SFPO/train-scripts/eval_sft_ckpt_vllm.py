#!/usr/bin/env python3
"""vLLM 6-benchmark eval for one SFT HF checkpoint (baseline or +GXPO arm).

Same harness the KD/RL arms use -- tools/evaluate_greedy_5seeds.py (greedy
pass@1) and tools/evaluate_sampled_5seeds.py (sampled pass@n / average@n):
vLLM LLM.generate + verl's _default_compute_score + the same JSON schema, so
SFT numbers sit straight next to the RL/KD arms. Only the decoding flags
differ (the SFT pair's locked metrics: n=8, seeds 0,1,2, temp 0.6, top_p 0.95).

Usage:
  python train-scripts/eval_sft_ckpt_vllm.py --ckpt <hf_dir> \
      --data-files a.parquet ... --output-dir <out> [...]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1]
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

from tools.evaluate_greedy_5seeds import (  # noqa: E402
    BENCHMARK_ORDER,
    evaluate_model as evaluate_greedy,
)
from tools.evaluate_sampled_5seeds import (  # noqa: E402
    evaluate_model as evaluate_sampled,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="HF checkpoint dir (fsdp_sft_trainer global_step_*)")
    ap.add_argument("--tokenizer", default=None, help="tokenizer dir; defaults to --ckpt")
    ap.add_argument("--data-files", nargs="+", required=True)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--max-model-len", type=int, default=None,
                     help="Explicit vLLM context cap (prompt + max-tokens headroom). "
                          "Much smaller than the model's full native context lets vLLM "
                          "fit far more concurrent sequences in the same KV cache budget "
                          "-- the main throughput lever for short-response evaluation.")
    ap.add_argument("--max-num-seqs", type=int, default=None,
                     help="vLLM scheduler concurrency cap; raise once max-model-len is small.")
    ap.add_argument("--skip-greedy", action="store_true")
    args = ap.parse_args()

    ckpt = Path(args.ckpt).expanduser().resolve()
    if not ckpt.is_dir():
        raise SystemExit(f"checkpoint not found: {ckpt}")
    tokenizer = args.tokenizer or str(ckpt)
    data_files = [str(Path(f).expanduser().resolve()) for f in args.data_files]
    for f in data_files:
        if not Path(f).is_file():
            raise SystemExit(f"evaluation dataset not found: {f}")
    out = Path(args.output_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    meta = {
        "method": "sft",
        "checkpoint": str(ckpt),
        "seeds": args.seeds,
        "std_definition": "population standard deviation (numpy.std, ddof=0)",
    }

    nseed = len(args.seeds)
    if not args.skip_greedy:
        print("=== greedy pass@1 (vLLM, temp 0) ===", flush=True)
        greedy = evaluate_greedy(ckpt, tokenizer, data_files, args.seeds, args.max_tokens, args.gpu_memory_utilization,
                                  max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs)
        greedy.update(meta | {"decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0, "n": 1, "max_tokens": args.max_tokens}})
        (out / f"sft_greedy_{nseed}seed.json").write_text(json.dumps(greedy, indent=2, sort_keys=True) + "\n")

    print(f"=== sampled pass@{args.n} / average@{args.n} (temp {args.temperature}, top_p {args.top_p}) ===", flush=True)
    sampled = evaluate_sampled(
        ckpt, tokenizer, data_files, args.seeds, args.n,
        args.temperature, args.top_p, args.max_tokens, args.gpu_memory_utilization,
        max_model_len=args.max_model_len, max_num_seqs=args.max_num_seqs,
    )
    sampled.update(meta | {"decoding": {
        "do_sample": args.temperature > 0, "temperature": args.temperature,
        "top_p": args.top_p, "n": args.n, "max_tokens": args.max_tokens,
        "primary_metric": f"pass@{args.n}",
    }})
    (out / f"sft_sampled_{args.n}_{nseed}seed.json").write_text(json.dumps(sampled, indent=2, sort_keys=True) + "\n")

    keys = [k for k in BENCHMARK_ORDER if k in sampled["benchmarks"]]
    print(f"\n================ 6-benchmark summary (mean +- std over seeds {args.seeds}) ================")
    for key in keys:
        b = sampled["benchmarks"][key]
        print(f"{key:<16}pass@{args.n} {b['pass_at_n']['mean']:.4f} +- {b['pass_at_n']['std']:.4f}  "
              f"average@{args.n} {b['average_at_n']['mean']:.4f} +- {b['average_at_n']['std']:.4f}")
    print(f"wrote {out}", flush=True)


if __name__ == "__main__":
    main()
