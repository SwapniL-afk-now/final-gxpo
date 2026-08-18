#!/usr/bin/env python3
"""Build paper-ready GXPO efficiency tables from local run directories.

Primary input is local JSON/JSONL output, so W&B/network access is not needed.
Efficiency targets use only exact greedy validation checkpoints; no
interpolation or extrapolation is performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Any

BENCHMARKS = ("math500", "aime24", "aime25", "amc23", "minerva", "olympiadbench")
DISPLAY_BENCHMARKS = {
    "math500": "MATH-500",
    "aime24": "AIME24",
    "aime25": "AIME25",
    "amc23": "AMC23",
    "minerva": "Minerva",
    "olympiadbench": "OlympiadBench",
}
MODEL_DISPLAY = {
    "qwen25-math-1p5b": "Qwen2.5-Math-1.5B-Instruct",
    "llama32-3b": "Llama-3.2-3B",
    "qwen25-math-7b": "Qwen2.5-Math-7B-Instruct",
}
METHODS = ("grpo", "sfpo", "gxpo")
METHOD_DISPLAY = {"grpo": "GRPO", "sfpo": "SFPO (K=10)", "gxpo": "GXPO (K=10)"}


def read_json(path: Path, default=None):
    if not path.exists():
        return default
    return json.loads(path.read_text())


def read_jsonl(path: Path):
    if not path.exists():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSONL at {path}:{line_no}: {exc}") from exc
    return rows


def number(value):
    if value is None or isinstance(value, bool):
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def step_of(row):
    value = row.get("step", row.get("eval_greedy/global_step"))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def run_key(manifest, run_dir):
    model = manifest.get("model_alias")
    method = manifest.get("method")
    if not model:
        model = str(manifest.get("hyperparameters", {}).get("actor_rollout_ref", {}).get("model", {}).get("path", ""))
    if not method:
        method = "gxpo" if "gxpo" in run_dir.name.lower() else "sfpo" if "sfpo" in run_dir.name.lower() else "grpo"
    return str(model), str(method).lower()


def discover(args):
    paths = [Path(path).expanduser() for path in args.run_dirs]
    if args.results_root:
        root = Path(args.results_root).expanduser()
        if root.exists():
            paths.extend(path for path in sorted(root.iterdir()) if path.is_dir())
    runs = {}
    for path in paths:
        manifest_path = path / "run_manifest.json"
        if not manifest_path.exists():
            continue
        manifest = read_json(manifest_path, {})
        model, method = run_key(manifest, path)
        runs[(model, method)] = {"dir": path, "manifest": manifest}
    if not runs:
        raise SystemExit("No run_manifest.json files found; pass --results-root or run directories.")
    return runs


def validation_rows(run):
    # The efficiency launcher keeps greedy_validation.jsonl bounded to the latest
    # row. Historical validation rows are already present in train_metrics.jsonl
    # (the same file used for all efficiency accounting), so recover them there
    # without creating a second growing validation artifact.
    result = {}
    rows = read_jsonl(run["dir"] / "greedy_validation.jsonl")
    for row in rows:
        step = step_of(row)
        if step is not None:
            result[step] = row

    rows = read_jsonl(run["dir"] / "train_metrics.jsonl")
    if not rows:
        rows = read_jsonl(run["dir"] / "metrics.jsonl")
    for row in rows:
        step = step_of(row)
        if step is not None and "eval_greedy/avg_pass1" in row:
            result[step] = row
    return result


def train_rows(run):
    rows = read_jsonl(run["dir"] / "train_metrics.jsonl")
    if not rows:
        rows = read_jsonl(run["dir"] / "metrics.jsonl")
    result = {}
    for row in rows:
        step = step_of(row)
        if step is not None:
            result[step] = row
    return result


def terminal_quality(run):
    data = read_json(run["dir"] / "final_stochastic_eval.json", {}) or {}
    benchmarks = data.get("benchmarks", {})
    result = {}
    for benchmark in BENCHMARKS + ("avg_pass1",):
        item = benchmarks.get(benchmark, {}) or {}
        result[benchmark] = (number(item.get("mean")), number(item.get("std")))
    return result


def crossing(model, method_runs, target):
    if target is None:
        return None
    rows = validation_rows(method_runs)
    for step in sorted(rows):
        value = number(rows[step].get("eval_greedy/avg_pass1"))
        if value is not None and value >= target:
            return step
    return None


def metric_at(run, step, key):
    if step is None:
        return None
    return number(train_rows(run).get(step, {}).get(key))


def max_metric_through(run, step, key):
    if step is None:
        return None
    values = [number(row.get(key)) for current, row in train_rows(run).items() if current <= step]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def fmt(value, digits=2, suffix=""):
    value = number(value)
    return "N/R" if value is None else f"{value:.{digits}f}{suffix}"


def accuracy_cell(pair):
    mean, std = pair
    if mean is None:
        return "N/R"
    return f"{100 * mean:.2f} ± {100 * (std or 0.0):.2f}"


def model_label(model, run):
    return MODEL_DISPLAY.get(model, model)


def collect_rows(runs):
    models = sorted({model for model, _ in runs}, key=lambda value: (list(MODEL_DISPLAY).index(value) if value in MODEL_DISPLAY else 99, value))
    targets = {}
    crossings = {}
    efficiency = {}
    for model in models:
        grpo = runs.get((model, "grpo"))
        if not grpo:
            targets[model] = None
            continue
        greedy = validation_rows(grpo)
        values = [number(row.get("eval_greedy/avg_pass1")) for row in greedy.values()]
        values = [value for value in values if value is not None]
        target = max(values) if values else None
        targets[model] = target
        crossings[model] = {"target": target, "methods": {}}
        for method in METHODS:
            run = runs.get((model, method))
            step = crossing(model, run, target) if run else None
            crossings[model]["methods"][method] = {
                "status": "reached" if step is not None else "N/R",
                "step": step,
            }
            if run:
                efficiency[(model, method)] = build_efficiency(run, step, model, method)
    return models, targets, crossings, efficiency


def build_efficiency(run, step, model, method):
    manifest = run["manifest"]
    row = train_rows(run).get(step, {}) if step is not None else {}
    num_gpus = number(manifest.get("gpu_count_configured")) or number(manifest.get("hyperparameters", {}).get("trainer", {}).get("n_gpus_per_node")) or 1.0
    active_wall = number(row.get("time/cum_train_active_s"))
    bp = number(row.get("eff/cum_policy_grad_evals"))
    return {
        "step": step,
        "responses": number(row.get("eff/cum_responses")),
        "prompt_tokens": number(row.get("eff/cum_prompt_tokens")),
        "completion_tokens": number(row.get("eff/cum_completion_tokens")),
        "total_tokens": number(row.get("eff/cum_total_tokens")),
        "policy_grad_evals": bp,
        "raw_backward_calls": number(row.get("eff/cum_raw_backward_calls")),
        "actor_update_time_s": number(row.get("time/cum_actor_update_s")),
        "wall_time_s": active_wall,
        "gpu_hours": num_gpus * active_wall / 3600.0 if active_wall is not None else None,
        "peak_vram_gb": max_metric_through(run, step, "system/peak_vram_allocated_gb"),
        "reward_time_s": number(row.get("time/cum_reward_s")),
        "ref_logprob_time_s": number(row.get("time/cum_ref_logprob_s")),
        "old_logprob_time_s": number(row.get("time/cum_old_logprob_s")),
        "rollout_time_s": number(row.get("time/cum_rollout_s")),
        "other_time_s": number(row.get("time/cum_data_sync_other_s")),
        "end_to_end_elapsed_s": number(row.get("time/cum_end_to_end_elapsed_s")),
        "gpu_util_mean": number(row.get("system/gpu_util_mean")),
        "gpu_util_peak": number(row.get("system/gpu_util_peak")),
        "power_mean_w": number(row.get("system/gpu_power_mean_w")),
        "power_peak_w": number(row.get("system/gpu_power_peak_w")),
        "energy_kwh": number(row.get("system/energy_kwh")),
        "fallback_step": number(row.get("gxpo/fallback_step", row.get("sfpo/fallback_step"))),
        "method_active_fraction": active_fraction(run, step, method),
    }


def active_fraction(run, step, method):
    if step is None:
        return None
    key = "gxpo/prediction_active" if method == "gxpo" else "sfpo/fast_phase_active" if method == "sfpo" else None
    if key is None:
        return 1.0
    values = [number(row.get(key)) for current, row in train_rows(run).items() if current <= step]
    values = [value for value in values if value is not None]
    return sum(values) / len(values) if values else None


def write_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def latex_escape(value):
    return str(value).replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def write_latex(path, caption, label, headers, rows):
    columns = "l" * len(headers)
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        f"\\begin{{tabular}}{{{columns}}}",
        r"\toprule",
        " & ".join(latex_escape(header) for header in headers) + r" \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(latex_escape(row.get(header, "N/R")) for header in headers) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}", ""])
    path.write_text("\n".join(lines))


def build_long_rows(runs):
    rows = []
    for (model, method), run in sorted(runs.items()):
        validation = validation_rows(run)
        for step, train in sorted(train_rows(run).items()):
            val = validation.get(step, {})
            rows.append({
                "model": model,
                "method": method,
                "step": step,
                "greedy_avg_pass1": number(val.get("eval_greedy/avg_pass1")),
                "cum_train_wall_s": number(train.get("time/cum_train_active_s")),
                "cum_policy_grad_evals": number(train.get("eff/cum_policy_grad_evals")),
                "cum_raw_backward_calls": number(train.get("eff/cum_raw_backward_calls")),
                "cum_completion_tokens": number(train.get("eff/cum_completion_tokens")),
                "cum_total_tokens": number(train.get("eff/cum_total_tokens")),
                "cum_rollout_s": number(train.get("time/cum_rollout_s")),
                "cum_actor_update_s": number(train.get("time/cum_actor_update_s")),
                "cum_other_s": number(train.get("time/cum_data_sync_other_s")),
            })
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dirs", nargs="*", help="local run directories")
    parser.add_argument("--results-root", default="results/gxpo_efficiency")
    parser.add_argument("--output-dir", default="paper_results")
    args = parser.parse_args()
    runs = discover(args)
    expected = {(model, method) for model in MODEL_DISPLAY for method in METHODS}
    missing = sorted(expected - set(runs))
    if missing:
        formatted = ", ".join(f"{model}/{method}" for model, method in missing)
        raise SystemExit(f"Expected exactly the 9 final runs; missing: {formatted}")
    models, targets, crossing_data, efficiency = collect_rows(runs)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    table1_rows = []
    table2_rows = []
    extended_rows = []
    for model in models:
        for method in METHODS:
            run = runs.get((model, method))
            if not run:
                continue
            quality = terminal_quality(run)
            eff = efficiency.get((model, method), {})
            target_info = crossing_data.get(model, {}).get("methods", {}).get(method, {})
            step = target_info.get("step")
            grpo_eff = efficiency.get((model, "grpo"), {})
            wall_speedup = (grpo_eff.get("wall_time_s") / eff.get("wall_time_s")
                            if method != "grpo" and grpo_eff.get("wall_time_s") and eff.get("wall_time_s") else 1.0 if method == "grpo" else None)
            bp_speedup = (grpo_eff.get("policy_grad_evals") / eff.get("policy_grad_evals")
                          if method != "grpo" and grpo_eff.get("policy_grad_evals") and eff.get("policy_grad_evals") else 1.0 if method == "grpo" else None)
            label = model_label(model, run)
            table1_rows.append({
                "Model": label,
                "Method": METHOD_DISPLAY[method],
                "MATH-500": accuracy_cell(quality["math500"]),
                "AIME24": accuracy_cell(quality["aime24"]),
                "AIME25": accuracy_cell(quality["aime25"]),
                "AMC23": accuracy_cell(quality["amc23"]),
                "Minerva": accuracy_cell(quality["minerva"]),
                "OlympiadBench": accuracy_cell(quality["olympiadbench"]),
                "Avg. Pass@1 ↑": accuracy_cell(quality["avg_pass1"]),
                "Wall-clock Speedup ↑": fmt(wall_speedup, 2, "×"),
                "BP Speedup ↑": fmt(bp_speedup, 2, "×"),
            })
            table2_rows.append({
                "Model": label,
                "Method": METHOD_DISPLAY[method],
                "Steps ↓": step if step is not None else "N/R",
                "Responses ↓": fmt(eff.get("responses")),
                "Completion Tokens ↓": fmt(eff.get("completion_tokens"), 0),
                "Policy-Gradient BPs ↓": fmt(eff.get("policy_grad_evals"), 0),
                "Actor Update Time (s) ↓": fmt(eff.get("actor_update_time_s")),
                "Wall Time (s) ↓": fmt(eff.get("wall_time_s")),
                "GPU-hours ↓": fmt(eff.get("gpu_hours")),
                "Peak VRAM (GB) ↓": fmt(eff.get("peak_vram_gb")),
            })
            extended = {"Model": label, "Method": METHOD_DISPLAY[method], "Steps to Target": step if step is not None else "N/R"}
            extended.update({
                "Generated Responses": eff.get("responses"),
                "Prompt Tokens": eff.get("prompt_tokens"),
                "Completion Tokens": eff.get("completion_tokens"),
                "Total Generated Tokens": eff.get("total_tokens"),
                "Policy-Gradient BPs": eff.get("policy_grad_evals"),
                "Raw backward() Calls": eff.get("raw_backward_calls"),
                "Actor Update Time (s)": eff.get("actor_update_time_s"),
                "Training Wall Time (s)": eff.get("wall_time_s"),
                "GPU-hours": eff.get("gpu_hours"),
                "Peak VRAM (GB)": eff.get("peak_vram_gb"),
                "Reward Time (s)": eff.get("reward_time_s"),
                "Ref-logprob Time (s)": eff.get("ref_logprob_time_s"),
                "Old-logprob Time (s)": eff.get("old_logprob_time_s"),
                "Rollout Time (s)": eff.get("rollout_time_s"),
                "Other Time (s)": eff.get("other_time_s"),
                "End-to-end Elapsed (s)": eff.get("end_to_end_elapsed_s"),
                "GPU Util Mean": eff.get("gpu_util_mean"),
                "GPU Util Peak": eff.get("gpu_util_peak"),
                "Power Mean (W)": eff.get("power_mean_w"),
                "Power Peak (W)": eff.get("power_peak_w"),
                "Energy (kWh)": eff.get("energy_kwh"),
                "Fallback Step": eff.get("fallback_step"),
                "Method-specific Active Fraction": eff.get("method_active_fraction"),
            })
            extended_rows.append(extended)

    table1_fields = list(table1_rows[0]) if table1_rows else ["Model", "Method"]
    table2_fields = list(table2_rows[0]) if table2_rows else ["Model", "Method"]
    extended_fields = list(extended_rows[0]) if extended_rows else ["Model", "Method"]
    write_csv(output / "table1_main.csv", table1_rows, table1_fields)
    write_csv(output / "table2_secondary.csv", table2_rows, table2_fields)
    write_csv(output / "table2_secondary_extended.csv", extended_rows, extended_fields)
    write_latex(output / "table1_main.tex", "Main quality and headline efficiency at the matched greedy GRPO target.", "tab:gxpo-main", table1_fields, table1_rows)
    write_latex(output / "table2_secondary.tex", "Secondary efficiency accounting at the matched greedy GRPO target.", "tab:gxpo-efficiency", table2_fields, table2_rows)
    write_csv(output / "all_metrics_long.csv", build_long_rows(runs), [
        "model", "method", "step", "greedy_avg_pass1", "cum_train_wall_s",
        "cum_policy_grad_evals", "cum_raw_backward_calls", "cum_completion_tokens",
        "cum_total_tokens", "cum_rollout_s", "cum_actor_update_s", "cum_other_s",
    ])
    (output / "target_crossings.json").write_text(json.dumps({"targets": targets, "runs": crossing_data}, indent=2, sort_keys=True) + "\n")
    print(f"Wrote paper tables to {output}")


if __name__ == "__main__":
    main()
