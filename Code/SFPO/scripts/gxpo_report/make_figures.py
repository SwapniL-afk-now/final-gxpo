"""Generate GXPO paper Figures 2-9 from run artifacts.

Usage:
  python scripts/gxpo_report/make_figures.py --manifest runs.csv --out report/figures
The manifest is the same CSV used by make_tables.py. Figures are emitted as
PDF+PNG; each function is independent and skips cleanly when its runs are
absent from the manifest.
"""

import argparse
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import MATH500, ema, load_manifest, load_metrics, val_curve

K_COLORS = {3: '#4c6a8f', 5: '#3d8f7a', 10: '#d98841'}
METHOD_COLORS = {'GRPO': '#444444', 'SFPO': '#c65a5a', 'GXPO': '#3d8f7a'}
EMA_SPAN = 10


def save(fig, out_dir, name):
    os.makedirs(out_dir, exist_ok=True)
    for ext in ('pdf', 'png'):
        fig.savefig(os.path.join(out_dir, f'{name}.{ext}'), dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {name}.pdf/.png')


def gxpo_grid(manifest):
    """GXPO runs organised by (alpha, k)."""
    runs = manifest[manifest['method'].str.lower() == 'gxpo'].dropna(subset=['k', 'alpha'])
    return runs.sort_values(['alpha', 'k'])


def _curve_grid(manifest, out_dir, name, curve_of, xlabel, ylabel='Math500 Pass@16'):
    """Shared alpha-column / k-line grid figure (Figures 2, 5, 7, 9).

    curve_of(metrics) -> (x, y) arrays for one run.
    """
    runs = gxpo_grid(manifest)
    alphas = sorted(runs['alpha'].unique())
    if not alphas:
        print(f'skip {name}: no GXPO runs with alpha/k in manifest')
        return
    fig, axes = plt.subplots(1, len(alphas), figsize=(4 * len(alphas), 3.2), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, alpha in zip(axes, alphas):
        for _, run in runs[runs['alpha'] == alpha].iterrows():
            x, y = curve_of(load_metrics(run['run_dir']))
            if len(y) == 0:
                continue
            k = int(run['k'])
            ax.plot(x, ema(y, EMA_SPAN), color=K_COLORS.get(k, 'gray'), label=f'k = {k}')
        ax.set_title(f'$\\alpha = {alpha:g}$')
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(title='Branching factor', fontsize=8)
    save(fig, out_dir, name)


def fig2_pass_vs_steps(manifest, out_dir):
    def curve(met):
        steps, vals, bps, hours = val_curve(met)
        return steps, vals

    _curve_grid(manifest, out_dir, 'fig2_pass16_vs_steps', curve, xlabel='Training Steps')


def fig5_pass_vs_bp(manifest, out_dir):
    def curve(met):
        steps, vals, bps, hours = val_curve(met)
        return bps, vals

    _curve_grid(manifest, out_dir, 'fig5_pass16_vs_bp', curve, xlabel='Backward Passes')


def fig3_wallclock(manifest, out_dir, benchmark=MATH500):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    for _, run in manifest.iterrows():
        met = load_metrics(run['run_dir'])
        steps, vals, bps, hours = val_curve(met, benchmark)
        if len(vals) == 0:
            continue
        method = run['method'].upper()
        label = method if method == 'GRPO' else f"{method} (k={int(run['k'])})"
        color = METHOD_COLORS.get(method, 'gray')
        ls = {3: '-', 5: '--', 10: ':'}.get(run['k'] if pd.isna(run['k']) else int(run['k']), '-')
        ax1.plot(hours, ema(vals, EMA_SPAN), label=label, color=color, linestyle=ls)
        best = int(np.argmax(vals))
        ax2.scatter(hours[best], vals[best], label=label, color=color)
    ax1.set_xlabel('Wall-clock Time (hours)')
    ax1.set_ylabel('Math500 Pass@16')
    ax1.set_title('Pass@16 vs Wall-clock Time')
    ax1.legend(fontsize=7)
    ax2.set_xlabel('Hours to Peak Pass@16')
    ax2.set_ylabel('Peak Pass@16')
    ax2.set_title('Pass@16-Efficiency Frontier')
    save(fig, out_dir, 'fig3_wallclock_frontier')


def fig4_alpha_k_landscape(manifest, out_dir):
    runs = gxpo_grid(manifest)
    if runs.empty:
        print('skip fig4: no GXPO alpha/k runs')
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4))
    peaks = {}
    for _, run in runs.iterrows():
        met = load_metrics(run['run_dir'])
        steps, vals, bps, hours = val_curve(met)
        if len(vals) == 0:
            continue
        k, alpha = int(run['k']), float(run['alpha'])
        best = int(np.argmax(vals))
        peaks[(alpha, k)] = vals[best]
        ax1.scatter(hours[best], vals[best], color=K_COLORS.get(k, 'gray'),
                    marker={0.1: 'o', 0.5: 's', 1.0: 'D'}.get(alpha, 'o'),
                    label=f'k={k}, $\\alpha$={alpha:g}')
    ax1.set_xlabel('Hours to Peak Pass@16')
    ax1.set_ylabel('Peak Pass@16')
    ax1.set_title('Pass@16-Efficiency Frontier')
    ax1.legend(fontsize=6)

    alphas = sorted({a for a, _ in peaks})
    ks = sorted({k for _, k in peaks})
    grid = np.full((len(alphas), len(ks)), np.nan)
    for (a, k), v in peaks.items():
        grid[alphas.index(a), ks.index(k)] = v
    im = ax2.imshow(grid, cmap='Greens', aspect='auto')
    ax2.set_xticks(range(len(ks)), [f'k = {k}' for k in ks])
    ax2.set_yticks(range(len(alphas)), [f'{a:g}' for a in alphas])
    ax2.set_xlabel('Branching Factor')
    ax2.set_ylabel('Regularization Strength ($\\alpha$)')
    ax2.set_title('Pass@16 Landscape')
    for i in range(len(alphas)):
        for j in range(len(ks)):
            if not np.isnan(grid[i, j]):
                ax2.text(j, i, f'{grid[i, j]:.3f}', ha='center', va='center', fontsize=8)
    fig.colorbar(im, ax=ax2, label='Pass@16')
    save(fig, out_dir, 'fig4_alpha_k_landscape')


def fig6_tau_sweep(manifest, out_dir):
    runs = manifest[(manifest['method'].str.lower() == 'gxpo')].dropna(subset=['tau'])
    if runs['tau'].nunique() < 2:
        print('skip fig6: need multiple tau values in manifest')
        return
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), sharey=True)
    cmap = plt.get_cmap('viridis')
    taus = sorted(runs['tau'].unique())
    for _, run in runs.iterrows():
        met = load_metrics(run['run_dir'])
        steps, vals, bps, hours = val_curve(met)
        color = cmap(taus.index(run['tau']) / max(len(taus) - 1, 1))
        label = f"$\\tau$={run['tau']:g}"
        for ax, x in zip(axes, (steps, hours, bps)):
            ax.plot(x, ema(vals, EMA_SPAN), color=color, label=label)
    for ax, xlabel in zip(axes, ('Training Steps', 'Wall-clock Time (hours)', 'Backward Passes')):
        ax.set_xlabel(xlabel)
    axes[0].set_ylabel('Math500 Pass@16')
    axes[0].legend(title='$\\tau$ Sweep', fontsize=8)
    fig.suptitle('Pass@16 vs Training Statistics (GXPO)')
    save(fig, out_dir, 'fig6_tau_sweep')


def fig7_response_length(manifest, out_dir):
    def curve(met):
        for col in ('response_length/mean', 'response_length_mean'):
            if col in met.columns:
                return met['step'].to_numpy(), met[col].to_numpy()
        return np.array([]), np.array([])

    _curve_grid(manifest, out_dir, 'fig7_response_length', curve, xlabel='Steps',
                ylabel='Response Length Mean')


def fig8_diagnostics(manifest, out_dir):
    runs = manifest[manifest['method'].str.lower() == 'gxpo'].dropna(subset=['k'])
    if runs.empty:
        print('skip fig8: no GXPO runs')
        return
    panels = (('actor/gxpo_trigger_z', 'Trigger z-score'),
              ('actor/gxpo_cos_g0_g1', 'cos($g_0$, $g_1$)'),
              ('actor/gxpo_dispK_over_disp2', 'disp$_K$/disp$_2$'))
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2))
    for _, run in runs.iterrows():
        met = load_metrics(run['run_dir'])
        k = int(run['k'])
        for ax, (col, _) in zip(axes, panels):
            if col in met.columns:
                ax.plot(met['step'], met[col].fillna(0.0), color=K_COLORS.get(k, 'gray'),
                        label=f'k = {k}')
    for ax, (_, title) in zip(axes, panels):
        ax.set_xlabel('Training Steps')
        ax.set_ylabel(title)
    axes[0].legend(fontsize=8)
    fig.suptitle('GXPO Diagnostic Curves')
    save(fig, out_dir, 'fig8_diagnostics')


def fig9_retention(manifest, out_dir):
    def curve(met):
        if 'actor/gxpo_r_mean' not in met.columns:
            return np.array([]), np.array([])
        return met['step'].to_numpy(), met['actor/gxpo_r_mean'].fillna(0.0).to_numpy()

    _curve_grid(manifest, out_dir, 'fig9_retention', curve, xlabel='Steps', ylabel='Retention Ratio')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', default='report/figures')
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    fig2_pass_vs_steps(manifest, args.out)
    fig3_wallclock(manifest, args.out)
    fig4_alpha_k_landscape(manifest, args.out)
    fig5_pass_vs_bp(manifest, args.out)
    fig6_tau_sweep(manifest, args.out)
    fig7_response_length(manifest, args.out)
    fig8_diagnostics(manifest, args.out)
    fig9_retention(manifest, args.out)


if __name__ == '__main__':
    main()
