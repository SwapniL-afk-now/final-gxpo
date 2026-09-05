#!/usr/bin/env python3
"""Build offline KD teacher top-K caches for final-gxpo SFT+GXPO KD training.

Single-pass design: the teacher GENERATES with vLLM (CUDA graphs on, FlashInfer
attention, TP=1) and the generation's own top-K logprobs are harvested straight
from the sampler -- no second HF rescoring pass. For an autoregressive teacher,
the per-token generation logprobs ARE the teacher-forced top-K distributions.

For each prompt-only training row (DAPO / lighteval-math / benchmark schema: a
``prompt`` column holding ``[{role, content}]``):

Output parquet columns (exactly what ``KDSFTDataset`` consumes):
  prompt                  : str   (raw user content, unchanged)
  response                : str   (teacher generation, for readability)
  response_ids            : list[int] (teacher token ids incl. trailing eos --
                            the EXACT ids the top-K rows align to)
  teacher_topk_log_probs  : list[[R, K] float]  (R = len(response_ids))
  teacher_topk_ids        : list[[R, K] int]
  kd_*_tokenizer_fingerprint: cache-time tokenizer identity metadata

Alignment is exact by construction: KDSFTDataset consumes ``response_ids``
verbatim (``response_ids_key``) instead of re-tokenizing ``response`` text,
so BPE detokenize/retokenize drift is impossible. Rows are kept whenever
every generated token has a top-K map (length-capped sequences included:
each teacher token is a valid per-token KD target); only logprob gaps drop.

Text and ids both come from the student tokenizer + chat template given by
``--student-tokenizer`` (must equal the SFT training tokenizer; KDSFTDataset
fail-closes with the rebuild instruction on any mismatch).

Throughput: run two shards in parallel (one TP=1 engine per GPU):
  CUDA_VISIBLE_DEVICES=0 ... --num-shards 2 --shard 0 &
  CUDA_VISIBLE_DEVICES=1 ... --num-shards 2 --shard 1 &

Usage:
  python tools/kd_sft/build_teacher_topk.py \
    --train-parquet <prompt-parquet> --teacher-path <7B> \
    --student-tokenizer <1.5B> --out data/kd/<name>_topk32.parquet
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Throughput profile: FlashInfer vLLM attention (never enforce_eager), CUDA
# graphs on. The version-check bypass mirrors the proven GXPO launchers:
# flashinfer-cubin tops out below the pinned wheel, so backend ranking would
# otherwise die on FlashInfer before selecting it.
os.environ.setdefault("VLLM_ATTENTION_BACKEND", "FLASHINFER")
os.environ.setdefault("FLASHINFER_DISABLE_VERSION_CHECK", "1")
os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")

import pandas as pd

PAD_LOGPROB = -1e4


def tokenizer_fingerprint(tokenizer) -> str:
    """Stable identity for token IDs, special tokens, and chat formatting."""
    payload = {
        "vocab": sorted((str(k), int(v)) for k, v in tokenizer.get_vocab().items()),
        "vocab_size": int(getattr(tokenizer, "vocab_size", 0)),
        "length": int(len(tokenizer)),
        "special_tokens_map": tokenizer.special_tokens_map,
        "all_special_ids": [int(v) for v in tokenizer.all_special_ids],
        "chat_template": getattr(tokenizer, "chat_template", None),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def validate_tokenizer_identity(student_tok, teacher_tok):
    student_fp = tokenizer_fingerprint(student_tok)
    teacher_fp = tokenizer_fingerprint(teacher_tok)
    if student_fp != teacher_fp:
        raise SystemExit(
            "student and teacher tokenizers are not identical; offline top-K KD "
            "cannot safely use teacher token IDs as student IDs. "
            f"student={student_fp[:12]} teacher={teacher_fp[:12]}"
        )
    return student_fp


def _part_paths(work: str):
    import glob
    paths = glob.glob(f"{work}.part*.parquet")
    return sorted(paths, key=lambda p: int(p.rsplit('.part', 1)[1].split('.')[0]))


def _next_part_index(work: str) -> int:
    paths = _part_paths(work)
    if not paths:
        return 0
    return max(int(p.rsplit('.part', 1)[1].split('.')[0]) for p in paths) + 1


def merge_shards(out: str, num_shards: int):
    """Fan in completed shard work files into the cache consumed by KD-SFT."""
    import glob
    shard_paths = []
    for shard in range(num_shards):
        work = f"{out}.shard{shard}.work.parquet"
        if os.path.exists(work):
            shard_paths.append(work)
            continue
        parts = _part_paths(work)
        if parts:
            shard_paths.append(work)
            pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True).to_parquet(work)
            continue
        raise SystemExit(f"missing shard output: {work}")
    frames = [pd.read_parquet(path) for path in shard_paths]
    merged = pd.concat(frames, ignore_index=True)
    # A resumed shard can contain a row from both a prior part and a rerun.
    # Keep one exact response per prompt, preserving shard order.
    if {'prompt', 'response_ids'}.issubset(merged.columns):
        merged['_kd_row_key'] = merged.apply(
            lambda row: (str(row['prompt']), json.dumps(row['response_ids'], default=str)), axis=1)
        merged = merged.drop_duplicates(subset=['_kd_row_key'], keep='first').drop(
            columns=['_kd_row_key'])
    merged.to_parquet(out, index=False)
    print(f"[kd-cache] merged {len(shard_paths)} shards / {len(merged)} rows -> {out}", flush=True)


def resolve_student_vocab_size(student_tokenizer_path: str, override=None) -> int:
    """Model embedding dim the student forward can legally index (< V)."""
    if override:
        return int(override)
    import json
    cfg = os.path.join(student_tokenizer_path, "config.json")
    if os.path.exists(cfg):
        with open(cfg) as f:
            return int(json.load(f)["vocab_size"])
    raise SystemExit(f"no config.json under {student_tokenizer_path}; pass --student-vocab-size")


def read_prompts(parquet: str, max_prompt_tokens=None, tokenizer=None) -> list:
    df = pd.read_parquet(parquet)
    if "prompt" not in df.columns:
        raise SystemExit(f"{parquet} has no 'prompt' column (got {list(df.columns)})")
    rows = []
    for idx, row in df.iterrows():
        prompt = row["prompt"]
        content = prompt[0]["content"] if isinstance(prompt, list) else str(prompt)
        if tokenizer is not None and max_prompt_tokens is not None:
            chat = tokenizer.apply_chat_template([{"role": "user", "content": content}],
                                                 add_generation_prompt=True, tokenize=False)
            if len(tokenizer(chat, add_special_tokens=False)["input_ids"]) > max_prompt_tokens:
                continue  # same filter as KD training (filter_overlong_prompts)
        rows.append({"dataset_index": int(idx), "prompt": content})
    return rows


def topk_from_vllm_logprobs(logprobs, topk: int, pad_id: int):
    """Convert one position's {token_id: Logprob} map to top-K (ids, logprobs)."""
    if not logprobs:
        return [pad_id] * topk, [PAD_LOGPROB] * topk
    ranked = sorted(logprobs.items(), key=lambda kv: kv[1].logprob, reverse=True)[:topk]
    ids = [int(tid) for tid, _ in ranked]
    lps = [float(lp.logprob) for _, lp in ranked]
    while len(ids) < topk:  # defensive: short maps pad inert, never NaN
        ids.append(pad_id)
        lps.append(PAD_LOGPROB)
    return ids, lps


def run_score_mode(args, llm, student_tok, pad_id, student_V, rows):
    """Teacher scores (prompt, response, response_ids) pairs via prompt_logprobs.

    This is the on-policy half of the loop: the STUDENT generated `response`
    (see tools/kd_sft/generate_student_responses.py); the teacher teacher-forces the exact
    same id sequence and returns top-K logprobs per response position.
    Positions come from the student template + stored response_ids, so rows
    align exactly (both tokenizers share merges/vocab).
    """
    import glob as _glob
    from vllm import SamplingParams

    df = pd.read_parquet(args.train_parquet)
    for col in ("prompt", "response", "response_ids"):
        if col not in df.columns:
            raise SystemExit(f"score mode needs (prompt, response, response_ids); "
                             f"{args.train_parquet} has {list(df.columns)}")
    pairs = []
    for _, row in df.iterrows():
        p = row["prompt"]
        content = p[0]["content"] if isinstance(p, list) else str(p)
        import numpy as np
        rids = np.asarray(row["response_ids"]).flatten().astype(int).tolist()
        pairs.append({"prompt": content, "response": str(row["response"]),
                      "response_ids": rids})
    pairs = pairs[args.shard::args.num_shards]
    print(f"[kd-score] {len(pairs)} pairs (shard {args.shard}/{args.num_shards})", flush=True)

    work = f"{args.out}.shard{args.shard}.work.parquet"
    done = set()
    done_paths = _part_paths(work)
    if args.num_shards == 1 and os.path.exists(args.out):
        done_paths.append(args.out)
    for _pf in done_paths:
        try:
            _d = pd.read_parquet(_pf, columns=["prompt", "response"])
            done.update(zip(_d["prompt"].tolist(), _d["response"].tolist()))
        except Exception:
            pass
    pairs = [x for x in pairs if (x["prompt"], x["response"]) not in done]
    if done:
        print(f"[kd-score] resume: {len(done)} rows cached", flush=True)

    params = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk)
    reqs, meta = [], []
    for x in pairs:
        p_ids = student_tok.apply_chat_template([{"role": "user", "content": x["prompt"]}],
                                                add_generation_prompt=True, tokenize=True,
                                                add_special_tokens=False)
        full = p_ids + x["response_ids"]
        if len(full) > args.max_model_len:
            continue  # over budget (matches truncation='error' downstream)
        reqs.append({"prompt_token_ids": full})
        meta.append((len(p_ids), len(x["response_ids"]), x))
    outs = llm.generate(reqs, params)

    scored, dropped, drop_reasons = [], 0, {}
    pending, part_idx = [], _next_part_index(work)
    for (P, R, x), o in zip(meta, outs):
        plp = o.prompt_logprobs or []
        span = plp[P:P + R]
        if len(span) != R or any(m is None for m in span):
            dropped += 1
            drop_reasons["missing_prompt_logprob"] = drop_reasons.get("missing_prompt_logprob", 0) + 1
            continue
        t_ids_rows, t_lp_rows, remapped = [], [], 0
        for m in span:
            ids, lps = topk_from_vllm_logprobs(m, args.topk, pad_id)
            fids, flps = [], []
            for tid, tlp in zip(ids, lps):
                if tid >= student_V:
                    tid, tlp = pad_id, PAD_LOGPROB
                    remapped += 1
                fids.append(tid)
                flps.append(tlp)
            t_ids_rows.append(fids)
            t_lp_rows.append(flps)
        if remapped:
            drop_reasons["topk_entries_remapped_inert"] = \
                drop_reasons.get("topk_entries_remapped_inert", 0) + remapped
        row = {"prompt": x["prompt"], "response": x["response"],
               "response_ids": x["response_ids"],
               "teacher_topk_log_probs": t_lp_rows, "teacher_topk_ids": t_ids_rows,
               "kd_student_tokenizer_fingerprint": tokenizer_fp,
               "kd_teacher_tokenizer_fingerprint": tokenizer_fp}
        scored.append(row)
        pending.append(row)
        if len(pending) >= args.flush_every:
            pd.DataFrame(pending).to_parquet(f"{work}.part{part_idx}.parquet")
            pending.clear()
            part_idx += 1
    if pending:
        pd.DataFrame(pending).to_parquet(f"{work}.part{part_idx}.parquet")
    parts = sorted(_glob.glob(f"{work}.part*.parquet"))
    if parts:
        pd.concat([pd.read_parquet(p) for p in parts],
                  ignore_index=True).to_parquet(args.out if args.num_shards == 1 else work)
    print(f"[kd-score] scored={len(scored)} dropped={dropped} -> {args.out}", flush=True)
    print(f"[kd-score] reasons={drop_reasons}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="gen", choices=["gen", "score"],
                    help="gen: teacher generates+scores prompts (offline cache). "
                         "score: teacher scores given (prompt, response, response_ids) "
                         "pairs via prompt_logprobs (on-policy student generations).")
    ap.add_argument("--train-parquet", required=False)
    ap.add_argument("--teacher-path", required=False)
    ap.add_argument("--student-tokenizer", required=False)
    ap.add_argument("--out", required=True)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--student-vocab-size", type=int, default=None)
    ap.add_argument("--no-ban-above-vocab", action="store_true",
                    help="allow teacher-only ids >= student vocab (will crash the student forward)")
    ap.add_argument("--max-prompt-tokens", type=int, default=1023)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--gen-temperature", type=float, default=1.0)
    ap.add_argument("--gen-top-p", type=float, default=1.0)
    ap.add_argument("--gen-max-tokens", type=int, default=3072)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--max-num-seqs", type=int, default=256)
    ap.add_argument("--flush-every", type=int, default=512)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--merge-shards", action="store_true",
                    help="fan in <out>.shardN.work.parquet files and exit")
    args = ap.parse_args()
    if args.merge_shards:
        if args.num_shards < 2:
            raise SystemExit("--merge-shards requires --num-shards >= 2")
        merge_shards(args.out, args.num_shards)
        return
    for name in ("train_parquet", "teacher_path", "student_tokenizer"):
        if not getattr(args, name):
            ap.error(f"--{name.replace('_', '-')} is required unless --merge-shards is used")

    from transformers import AutoTokenizer
    student_tok = AutoTokenizer.from_pretrained(args.student_tokenizer, trust_remote_code=True)
    teacher_tok = AutoTokenizer.from_pretrained(args.teacher_path, trust_remote_code=True)
    tokenizer_fp = validate_tokenizer_identity(student_tok, teacher_tok)
    print(f"[kd-cache] tokenizer identity OK: {tokenizer_fp[:12]}", flush=True)
    pad_id = student_tok.pad_token_id if student_tok.pad_token_id is not None else 0
    student_V = resolve_student_vocab_size(args.student_tokenizer, args.student_vocab_size)
    print(f"[kd-cache] student vocab bound V={student_V} (ids must be < V)", flush=True)
    teacher_V = resolve_student_vocab_size(args.teacher_path, None)
    ban = {} if args.no_ban_above_vocab else {i: -100.0 for i in range(student_V, teacher_V)}
    if ban:
        print(f"[kd-cache] banning {len(ban)} teacher-only ids from generation", flush=True)
    eos = student_tok.eos_token

    rows = read_prompts(args.train_parquet, args.max_prompt_tokens, student_tok)
    rows = rows[args.shard::args.num_shards]
    print(f"[kd-cache] {len(rows)} prompts (shard {args.shard}/{args.num_shards})", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(model=args.teacher_path, dtype="bfloat16",
              tensor_parallel_size=args.tensor_parallel_size,
              gpu_memory_utilization=0.90, max_model_len=args.max_model_len,
              max_num_seqs=args.max_num_seqs, enable_chunked_prefill=True,
              max_logprobs=args.topk,
              trust_remote_code=True, enforce_eager=False)

    if args.mode == "score":
        return run_score_mode(args, llm, student_tok, pad_id, student_V, rows)

    work = f"{args.out}.shard{args.shard}.work.parquet"
    done_prompts = set()
    import glob as _glob
    done_paths = _part_paths(work)
    if args.num_shards == 1 and os.path.exists(args.out):
        done_paths.append(args.out)
    for _pf in done_paths:
        try:
            done_prompts.update(pd.read_parquet(_pf, columns=["prompt"])["prompt"].tolist())
        except Exception:
            pass
    if done_prompts:
        print(f"[kd-cache] resume: {len(done_prompts)} rows already cached", flush=True)
    rows = [r for r in rows if r["prompt"] not in done_prompts]

    params = SamplingParams(temperature=args.gen_temperature, top_p=args.gen_top_p,
                            max_tokens=args.gen_max_tokens, logprobs=args.topk,
                            logit_bias=ban or None)
    chats = [student_tok.apply_chat_template([{"role": "user", "content": r["prompt"]}],
                                             add_generation_prompt=True, tokenize=False) for r in rows]
    outs = llm.generate(chats, params)

    scored, dropped = [], 0
    drop_reasons: dict = {}
    pending, part_idx = [], _next_part_index(work)

    def flush(final=False):
        nonlocal part_idx
        if not pending and not final:
            return
        if pending:
            part = f"{work}.part{part_idx}.parquet"
            pd.DataFrame(pending).to_parquet(part)
            pending.clear()
            part_idx += 1

    for r, o in zip(rows, outs):
        gen = o.outputs[0]
        gen_ids = list(gen.token_ids)
        text = gen.text
        # Keep every generation with full per-token top-K coverage, including
        # length-capped ones (each teacher token is a valid KD target).
        # response_ids ARE the generated ids, so rows align exactly.
        if not gen_ids:
            dropped += 1
            drop_reasons["empty_generation"] = drop_reasons.get("empty_generation", 0) + 1
            continue
        if max(gen_ids) >= student_V:
            # Ban leakage (should not happen): the student embedding table
            # has no such row; indexing it aborts the CUDA kernel (SIGABRT).
            dropped += 1
            drop_reasons["response_id_above_student_vocab"] = \
                drop_reasons.get("response_id_above_student_vocab", 0) + 1
            continue
        t_ids_rows, t_lp_rows = [], []
        remapped = 0
        for lp_map in (gen.logprobs or []):
            ids, lps = topk_from_vllm_logprobs(lp_map, args.topk, pad_id)
            # Residual teacher-only ids (logit bias makes them ~zero mass):
            # remap to an inert (pad, -1e4) entry so the student gather stays
            # in-bounds without changing the KL.
            fixed_ids, fixed_lps = [], []
            for tid, tlp in zip(ids, lps):
                if tid >= student_V:
                    tid, tlp = pad_id, PAD_LOGPROB
                    remapped += 1
                fixed_ids.append(tid)
                fixed_lps.append(tlp)
            t_ids_rows.append(fixed_ids)
            t_lp_rows.append(fixed_lps)
        if remapped:
            drop_reasons["topk_entries_remapped_inert"] = \
                drop_reasons.get("topk_entries_remapped_inert", 0) + remapped
        if len(t_ids_rows) != len(gen_ids):
            dropped += 1
            drop_reasons[f"logprob_rows={len(t_ids_rows)}_vs_ids={len(gen_ids)}"] = \
                drop_reasons.get(f"logprob_rows={len(t_ids_rows)}_vs_ids={len(gen_ids)}", 0) + 1
            continue
        scored.append({"prompt": r["prompt"], "response": text,
                       "response_ids": gen_ids,
                       "teacher_topk_log_probs": t_lp_rows,
                       "teacher_topk_ids": t_ids_rows,
                       "kd_student_tokenizer_fingerprint": tokenizer_fp,
                       "kd_teacher_tokenizer_fingerprint": tokenizer_fp})
        pending.append(scored[-1])
        if len(pending) >= args.flush_every:
            flush()
    flush(final=True)

    import glob
    parts = sorted(glob.glob(f"{work}.part*.parquet"))
    if parts:
        out_df = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        out_df.to_parquet(args.out if args.num_shards == 1 else work)
        if args.num_shards != 1:
            print(f"[kd-cache] shard part files: {parts}", flush=True)
    else:
        pd.DataFrame([]).to_parquet(args.out if args.num_shards == 1 else work)
    total = len(pd.read_parquet(args.out if args.num_shards == 1 else work))
    print(f"[kd-cache] scored={len(scored)} dropped={dropped} total_on_disk={total} -> {args.out}", flush=True)
    print(f"[kd-cache] drop_reasons={drop_reasons}", flush=True)


if __name__ == "__main__":
    main()
