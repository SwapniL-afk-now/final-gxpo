#!/usr/bin/env python3
"""Compare the terminal step of the two Qwen 1.5B performance smokes."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "results" / "perf_smoke" / "qwen1p5b"
OUT_CSV = BASE / "comparison.csv"
OUT_JSON = BASE / "comparison.json"


def read_step(name: str, step: int = 2) -> dict[str, Any]:
    path = BASE / name / "train_metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    matches = [row for row in rows if int(row.get("train/global_step", row.get("step", -1))) == step]
    if not matches:
        raise SystemExit(f"No terminal step {step} in {path}")
    return matches[-1]


def nested(row: dict[str, Any], key: str, child: str) -> Any:
    value = row.get(key, {})
    return value.get(child) if isinstance(value, dict) else None


def fields(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": row.get("train/global_step"),
        "total_step_s": row.get("perf/total_step_time_s"),
        "rollout_s": row.get("perf/rollout_time_s"),
        "rollout_tokens_per_s": row.get("perf/rollout_tokens_per_second"),
        "rollout_sequences_per_s": row.get("perf/rollout_sequences_per_second"),
        "logprob_s": row.get("perf/logprob_time_s"),
        "actor_update_s": row.get("perf/actor_update_time_s"),
        "actor_tokens_per_s": row.get("perf/actor_tokens_per_second"),
        "weight_sync_s": row.get("perf/weight_sync_time_s"),
        "gpu_util_mean": row.get("system/gpu_util_mean"),
        "gpu_util_p50": row.get("system/gpu_util_p50"),
        "gpu_util_p90": row.get("system/gpu_util_p90"),
        "gpu_util_peak": row.get("system/gpu_util_peak"),
        "rollout_peak_allocated_gb": nested(row, "system/rollout_peak_memory", "allocated_gb"),
        "after_vllm_sleep_allocated_gb": nested(row, "system/after_vllm_sleep_memory", "allocated_gb"),
        "actor_peak_allocated_gb": nested(row, "system/actor_update_peak_memory", "allocated_gb"),
        "actor_peak_free_device_gb": nested(row, "system/actor_update_peak_memory", "free_device_gb"),
        "reward_mean": row.get("train/reward_mean"),
        "vllm_sleep_level": row.get("perf/vllm_sleep_level"),
    }


def speedup(a: Any, b: Any) -> Any:
    if not isinstance(a, (int, float)) or not isinstance(b, (int, float)) or b == 0:
        return None
    return a / b


def main() -> None:
    a = fields(read_step("smoke_a"))
    b = fields(read_step("smoke_b"))
    comparison = {
        "step": 2,
        "smoke_a": a,
        "smoke_b": b,
        "speedup_a_over_b": {
            "total_step": speedup(a["total_step_s"], b["total_step_s"]),
            "rollout_tokens_per_s": speedup(b["rollout_tokens_per_s"], a["rollout_tokens_per_s"]),
            "actor_tokens_per_s": speedup(b["actor_tokens_per_s"], a["actor_tokens_per_s"]),
            "weight_sync_time": speedup(a["weight_sync_s"], b["weight_sync_s"]),
        },
    }
    BASE.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    columns = sorted(set(a) | set(b))
    with OUT_CSV.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "smoke_a", "smoke_b", "smoke_b_minus_a"])
        for key in columns:
            av, bv = a.get(key), b.get(key)
            delta = bv - av if isinstance(av, (int, float)) and isinstance(bv, (int, float)) else None
            writer.writerow([key, av, bv, delta])
    print(f"Wrote {OUT_CSV}")
    print(f"Wrote {OUT_JSON}")
    print(json.dumps(comparison["speedup_a_over_b"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
