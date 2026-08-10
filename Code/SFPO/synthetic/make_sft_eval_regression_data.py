#!/usr/bin/env python3
"""Generate SYNTHETIC (config -> amc23 pass@1/pass@8) rows for training a regression model.

NOT MEASUREMENTS. Every row is simulated. Nothing here may be reported as an eval result;
the `synthetic` column and the `syn-*` run ids exist so a stray row is obvious on sight.

Why simulate rather than sample noise around a mean: pass@1 and pass@8 come from the SAME
per-problem outcome matrix, so they are strongly coupled (a config cannot have high pass@8
and near-zero pass@1). A regressor trained on independently-jittered targets learns that
coupling wrong. So this reproduces the actual measurement process:

    40 problems x n=8 samples x 3 decode seeds -> pass@1 = mean(correct)
                                                 pass@8 = mean(max over the 8)
    reported mean/std = over the 3 decode seeds  (np.std ddof=0, as eval_amc23.py does)

Per-problem difficulty profile is calibrated to the real seed42 SFT baseline, which measured
pass@1=5.10% and pass@8=31.67%: a fraction `solvable_frac` of problems are reachable at rate
p, the rest at 0. Fitting both observed numbers gives 28/40 solvable at p=0.0729, i.e.
sd across problems 3.34pp -- the term that dominates the true standard error.

Usage:
    python make_sft_eval_regression_data.py --rows 400 --out sft_eval_synth.csv
"""
import argparse
import csv

import numpy as np

# Calibrated to the real amc23 SFT baseline (pass@1 5.10%, pass@8 31.67%). See docstring.
N_PROBLEMS, N_SAMPLES, N_SEEDS = 40, 8, 3
SOLVABLE_FRAC = 0.70


def simulate_eval(latent_pass1, rng, solvable_frac=SOLVABLE_FRAC,
                  n_problems=N_PROBLEMS, n_samples=N_SAMPLES, n_seeds=N_SEEDS):
    """Run the real measurement process on a model whose true pass@1 is `latent_pass1`.

    Returns (pass1_mean, pass1_std, pass8_mean, pass8_std) exactly as eval_amc23.py would
    print them -- std is over decode seeds with ddof=0, which is what that script reports.
    """
    n_solvable = max(1, int(round(solvable_frac * n_problems)))
    p = np.clip(latent_pass1 * n_problems / n_solvable, 0.0, 1.0)
    # Per-problem rates: solvable ones jitter around p, the rest are unreachable.
    rates = np.zeros(n_problems)
    rates[:n_solvable] = np.clip(rng.normal(p, p * 0.6, n_solvable), 0.0, 1.0)
    rng.shuffle(rates)

    p1, p8 = [], []
    for _ in range(n_seeds):
        correct = rng.random((n_problems, n_samples)) < rates[:, None]
        p1.append(correct.mean())
        p8.append(correct.max(axis=1).mean())
    return (float(np.mean(p1)), float(np.std(p1)),
            float(np.mean(p8)), float(np.std(p8)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rows", type=int, default=400)
    ap.add_argument("--out", default="sft_eval_synth.csv")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--base-pass1", type=float, default=0.0510,
                    help="latent pass@1 of the plain-SFT arm (real measured value)")
    ap.add_argument("--max-effect-pp", type=float, default=1.0,
                    help="max |GXPO-SFT - SFT| in percentage points. Default 1.0 keeps the "
                         "arms close, well inside the +-2.06pp band where the real eval "
                         "cannot distinguish them.")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    fields = ["run_id", "synthetic", "method", "k", "alpha", "tau", "warmup", "lr",
              "train_steps", "train_seed", "latent_pass1", "effect_pp",
              "pass1_mean", "pass1_std", "pass8_mean", "pass8_std"]
    with open(args.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for i in range(args.rows):
            is_gxpo = bool(rng.integers(2))
            # Config knobs spanning the ranges the SFT arm actually sweeps.
            k = int(rng.choice([1, 3, 5, 10])) if is_gxpo else 0
            alpha = float(rng.choice([0.05, 0.1, 0.2, 0.5])) if is_gxpo else 0.0
            tau = float(rng.choice([1.0, 2.0, 5.0])) if is_gxpo else 0.0
            warmup = int(rng.choice([0, 3, 5])) if is_gxpo else 0
            lr = float(rng.choice([5e-6, 1e-5, 2e-5]))
            steps = int(rng.choice([250, 500, 750]))
            train_seed = int(rng.choice([42, 123, 777]))

            # Effect is small and centred on zero: the real comparison could not resolve
            # anything below ~2.9pp, so a plausible generator should not manufacture more.
            effect_pp = (rng.normal(0.0, args.max_effect_pp / 2.0) if is_gxpo else 0.0)
            effect_pp = float(np.clip(effect_pp, -args.max_effect_pp, args.max_effect_pp))
            # Mild, monotone config sensitivity so the target is actually learnable.
            if is_gxpo:
                effect_pp += 0.15 * (tau >= 5.0) + 0.10 * (warmup >= 3) - 0.20 * (alpha >= 0.5)
            latent = max(0.001, args.base_pass1 + effect_pp / 100.0
                         + rng.normal(0, 0.0015))          # train-seed variability

            p1m, p1s, p8m, p8s = simulate_eval(latent, rng)
            w.writerow({
                "run_id": f"syn-{i:04d}", "synthetic": "true",
                "method": "gxpo-sft" if is_gxpo else "sft",
                "k": k, "alpha": alpha, "tau": tau, "warmup": warmup, "lr": lr,
                "train_steps": steps, "train_seed": train_seed,
                "latent_pass1": round(latent, 6), "effect_pp": round(effect_pp, 4),
                "pass1_mean": round(p1m, 6), "pass1_std": round(p1s, 6),
                "pass8_mean": round(p8m, 6), "pass8_std": round(p8s, 6),
            })
    print(f"wrote {args.rows} SYNTHETIC rows -> {args.out}")


if __name__ == "__main__":
    main()
