"""Generate GXPO paper Tables 1-8 from run artifacts.

Usage:
  python scripts/gxpo_report/make_tables.py --manifest runs.csv --out report/tables
Optional: --benchmark <data_source> for Tables 2/5 (default Math-500).
"""

import argparse
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from common import (BENCHMARKS, MATH500, ema, first_at_least, load_eval, load_manifest,
                    load_metrics, nearest_row, val_curve, write_table)


def method_label(row):
    return row['method'].upper()


def table1(manifest, out_dir):
    """Final-checkpoint benchmark accuracy (pass@1 avg) per model x method x k."""
    rows = []
    for _, run in manifest.iterrows():
        ev = load_eval(run['run_dir'])
        if ev is None:
            continue
        last = ev[ev['step'] == ev['step'].max()]
        met = load_metrics(run['run_dir'])
        active_bp = met.get('actor/gxpo_enabled')
        # BP per step during the active phase (3 for GXPO, k+1 for SFPO, 1 for GRPO)
        bp_per_step = {'grpo': 1, 'sfpo': int(run['k']) + 1 if pd.notna(run['k']) else None,
                       'gxpo': 3}.get(run['method'].lower())
        row = {'Model': run['model'], 'Method': method_label(run), 'k': run['k'], 'BP': bp_per_step}
        scores = []
        for src, label in BENCHMARKS.items():
            hit = last[last['benchmark'] == src]
            if not hit.empty:
                score = 100.0 * hit['pass1_avg'].iloc[0]
                row[label] = score
                scores.append(score)
        row['Avg.'] = np.mean(scores) if scores else np.nan
        rows.append(row)
    write_table(pd.DataFrame(rows), out_dir, 'table1_benchmarks')


def table2(manifest, out_dir, benchmark):
    """Convergence efficiency: peak acc, steps/hours/BPs to match GRPO peak."""
    rows = []
    for model, group in manifest.groupby('model'):
        grpo = group[group['method'].str.lower() == 'grpo']
        if grpo.empty:
            continue
        g_steps, g_vals, g_bp, g_hours = val_curve(load_metrics(grpo.iloc[0]['run_dir']), benchmark)
        grpo_peak = g_vals.max()
        for _, run in group.iterrows():
            steps, vals, bps, hours = val_curve(load_metrics(run['run_dir']), benchmark)
            idx = first_at_least(steps, vals, grpo_peak)
            rows.append({
                'Model': model, 'Method': method_label(run), 'k': run['k'],
                'Peak Acc.': vals.max(),
                'Steps to match GRPO': steps[idx] if idx is not None else np.nan,
                'Hours to match GRPO': hours[idx] if idx is not None else np.nan,
                'BPs to match GRPO': bps[idx] if idx is not None else np.nan,
            })
    df = pd.DataFrame(rows)
    for src_col, speed_col in (('Steps to match GRPO', 'Step-up'),
                               ('Hours to match GRPO', 'Time-up'),
                               ('BPs to match GRPO', 'BP-up')):
        base = df[df['Method'] == 'GRPO'].set_index('Model')[src_col]
        df[speed_col] = df.apply(lambda r: base.get(r['Model'], np.nan) / r[src_col]
                                 if pd.notna(r[src_col]) else np.nan, axis=1)
    write_table(df, out_dir, 'table2_convergence', float_format='%.4f')


def tables34(manifest, out_dir, bp_budgets=(108, 204, 300), hour_budgets=(4, 8, 12)):
    """Iso-backward-pass (T3) and iso-wall-clock (T4) benchmark comparisons."""
    view = [MATH500, 'openai/gsm8k', 'math-ai/minervamath']
    for name, col, budgets in (('table3_iso_bp', 'cumulative_bp', bp_budgets),
                               ('table4_iso_hours', 'elapsed_hours', hour_budgets)):
        rows = []
        for _, run in manifest.iterrows():
            ev = load_eval(run['run_dir'])
            if ev is None:
                continue
            row = {'Method': method_label(run), 'k': run['k'], 'Model': run['model']}
            for budget in budgets:
                for src in view:
                    sub = ev[ev['benchmark'] == src]
                    hit = nearest_row(sub, col, budget)
                    if hit is not None:
                        row[f'{BENCHMARKS[src]}@{budget}'] = 100.0 * hit['pass1_avg']
            rows.append(row)
        write_table(pd.DataFrame(rows), out_dir, name)


def table5(manifest, out_dir, benchmark):
    """Alpha-sweep sensitivity (GXPO runs only)."""
    rows = []
    for _, run in manifest[manifest['method'].str.lower() == 'gxpo'].iterrows():
        met = load_metrics(run['run_dir'])
        steps, vals, bps, hours = val_curve(met, benchmark)
        if len(vals) == 0:
            continue
        best = int(np.argmax(vals))
        shutoff = met['actor/gxpo_shutoff_step'].dropna().iloc[-1] \
            if 'actor/gxpo_shutoff_step' in met.columns and met['actor/gxpo_shutoff_step'].notna().any() else np.nan
        rows.append({
            'k': run['k'], 'alpha': run['alpha'], 'Shutoff step': shutoff,
            'Total BP': met['actor/cumulative_bp'].max() if 'actor/cumulative_bp' in met.columns else np.nan,
            'Total hours': met['train/elapsed_hours'].max() if 'train/elapsed_hours' in met.columns else np.nan,
            'Step to best': steps[best], 'BP to best': bps[best], 'Hours to best': hours[best],
            'Best pass@16': 100.0 * vals[best],
        })
    write_table(pd.DataFrame(rows).sort_values(['k', 'alpha']), out_dir, 'table5_alpha_sweep')


def table6(manifest, out_dir):
    """Surrogate-displacement diagnostics (medians over diag checkpoints)."""
    rows = []
    for _, run in manifest[manifest['method'].str.lower() == 'gxpo'].iterrows():
        met = load_metrics(run['run_dir'])
        cols = ['actor/gxpo_diag_thetaK_abs_err', 'actor/gxpo_diag_thetatilde_abs_err',
                'actor/gxpo_diag_disp_cosine_err', 'actor/gxpo_cos_g0_gslow']
        if not all(c in met.columns for c in cols[:3]):
            continue
        active = met[met.get('actor/gxpo_enabled', 0) > 0.5]
        rows.append({
            'k': run['k'],
            'Med. thetaK abs err': active[cols[0]].median(),
            'Med. thetatilde abs err': active[cols[1]].median(),
            'Med. disp cosine err': active[cols[2]].median(),
            'Med. active cos(g0,gslow)': active[cols[3]].median(),
        })
    write_table(pd.DataFrame(rows).sort_values('k'), out_dir, 'table6_displacement_diag',
                float_format='%.3e')


def table7(manifest, out_dir):
    """Active-phase GXPO diagnostics (medians over active steps)."""
    rows = []
    for _, run in manifest[manifest['method'].str.lower() == 'gxpo'].iterrows():
        met = load_metrics(run['run_dir'])
        if 'actor/gxpo_enabled' not in met.columns:
            continue
        a = met[met['actor/gxpo_enabled'] > 0.5]
        if a.empty:
            continue
        rows.append({
            'k': run['k'], 'Policy passes': 3,
            'Med. active ||g0||': a['actor/gxpo_g0_norm'].median(),
            'Med. active ||g1||': a['actor/gxpo_g1_norm'].median(),
            'Med. active ||gslow||': a['actor/gxpo_gslow_norm'].median(),
            'Med. cos(g0,gslow)': a['actor/gxpo_cos_g0_gslow'].median(),
            'Retention ratio': f"{a['actor/gxpo_r_mean'].median():.3f} ± {a['actor/gxpo_r_std'].median():.3f}",
            '||dK||/||d2||': a['actor/gxpo_dispK_over_disp2'].median(),
            'Scale mean': a['actor/gxpo_scale_mean'].median(),
            'Inactive frac': a['actor/gxpo_inactive_frac'].median(),
        })
    write_table(pd.DataFrame(rows).sort_values('k'), out_dir, 'table7_active_phase',
                float_format='%.4g')


def table8(manifest, out_dir):
    """KL and clipping diagnostics per method."""
    rows = []
    for _, run in manifest.iterrows():
        met = load_metrics(run['run_dir'])
        clip = met.get('actor/pg_clipfrac')
        kl = met.get('actor/kl_loss')
        rows.append({
            'Method': method_label(run), 'k': run['k'], 'Model': run['model'],
            'Mean clip fraction': clip.mean() if clip is not None else np.nan,
            'Max clip fraction': clip.max() if clip is not None else np.nan,
            'Mean KL penalty': kl.mean() if kl is not None else np.nan,
            'Max KL penalty': kl.max() if kl is not None else np.nan,
        })
    write_table(pd.DataFrame(rows), out_dir, 'table8_kl_clip', float_format='%.3e')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--out', default='report/tables')
    parser.add_argument('--benchmark', default=MATH500)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    table1(manifest, args.out)
    table2(manifest, args.out, args.benchmark)
    tables34(manifest, args.out)
    table5(manifest, args.out, args.benchmark)
    table6(manifest, args.out)
    table7(manifest, args.out)
    table8(manifest, args.out)


if __name__ == '__main__':
    main()
