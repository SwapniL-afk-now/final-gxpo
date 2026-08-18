#!/usr/bin/env python3
"""Verify fairness-critical fields match across one model's three runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

FIELDS = {
    "model": lambda c: c["actor_rollout_ref"]["model"]["path"],
    "data": lambda c: (c["data"]["train_files"], c["data"]["val_files"]),
    "train_batch_size": lambda c: c["data"]["train_batch_size"],
    "rollout_n": lambda c: c["actor_rollout_ref"]["rollout"]["n"],
    "learning_rate": lambda c: c["actor_rollout_ref"]["actor"]["optim"]["lr"],
    "optimizer": lambda c: c["actor_rollout_ref"]["actor"]["optim"].get("name", "adamw"),
    "prompt_length": lambda c: c["data"]["max_prompt_length"],
    "response_length": lambda c: c["data"]["max_response_length"],
    "mini_batch": lambda c: c["actor_rollout_ref"]["actor"]["ppo_mini_batch_size"],
    "micro_batch": lambda c: c["actor_rollout_ref"]["actor"].get("ppo_micro_batch_size_per_gpu"),
    "reward": lambda c: c["reward_model"].get("reward_manager", "naive"),
    "advantage_estimator": lambda c: c["algorithm"]["adv_estimator"],
    "rollout_temperature": lambda c: c["actor_rollout_ref"]["rollout"]["temperature"],
    "rollout_top_p": lambda c: c["actor_rollout_ref"]["rollout"].get("top_p"),
    "training_seed": lambda c: c["data"].get("seed"),
    "gpu_count": lambda c: c["trainer"]["n_gpus_per_node"],
    "validation_interval": lambda c: c["trainer"]["test_freq"],
    "validation_datasets": lambda c: c["data"]["val_files"],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="+", help="three local result directories")
    args = parser.parse_args()
    if len(args.run_dirs) != 3:
        raise SystemExit("Pass exactly three run directories for one model: GRPO, SFPO, GXPO.")
    manifests = []
    for raw in args.run_dirs:
        path = Path(raw).expanduser()
        manifest_path = path / "run_manifest.json"
        if not manifest_path.exists():
            raise SystemExit(f"Missing manifest: {manifest_path}")
        manifests.append((path, json.loads(manifest_path.read_text())))
    resolved = []
    for path, manifest in manifests:
        config = manifest.get("hyperparameters", {})
        values = {}
        for name, getter in FIELDS.items():
            try:
                values[name] = getter(config)
            except (KeyError, TypeError):
                values[name] = "<missing>"
        resolved.append((path, manifest.get("method"), values))
    failures = []
    baseline = resolved[0][2]
    for path, method, values in resolved:
        for name in FIELDS:
            if values[name] != baseline[name]:
                failures.append((name, path.name, values[name], baseline[name]))
    print("fair comparison config")
    for name in FIELDS:
        print(f"  {name}: {baseline[name]}")
    if failures:
        print("MISMATCHES:")
        for name, path, got, expected in failures:
            print(f"  {name}: {path}={got!r}, expected={expected!r}")
        raise SystemExit(1)
    methods = [method for _, method, _ in resolved]
    if set(methods) != {"grpo", "sfpo", "gxpo"}:
        raise SystemExit(f"Expected one grpo/sfpo/gxpo run, got {methods}")
    print(f"OK: matched {', '.join(methods)}")


if __name__ == "__main__":
    main()
