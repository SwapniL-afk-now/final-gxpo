"""Shared loading/utilities for GXPO paper tables and figures.

Run manifest: a CSV with columns
    run_dir,model,method,k,alpha,tau,seed
one row per training run. `run_dir` must contain metrics.jsonl (written by
ray_trainer.fit) and, for benchmark tables, eval_results.csv (written by
scripts/eval_checkpoints.py). Missing k/alpha/tau cells may be left empty
(e.g. for GRPO).
"""

import json
import os

import numpy as np
import pandas as pd

# canonical benchmark display order / names (keys = data_source in parquets)
BENCHMARKS = {
    'HuggingFaceH4/MATH-500': 'Math-500',
    'AI-MO/aimo-validation-amc': 'AMC23',
    'openai/gsm8k': 'GSM8k',
    'math-ai/minervamath': 'Minerva',
    'math-ai/olympiadbench': 'Olympiad',
}

MATH500 = 'HuggingFaceH4/MATH-500'


def load_manifest(path):
    df = pd.read_csv(path)
    required = {'run_dir', 'model', 'method'}
    missing = required - set(df.columns)
    assert not missing, f'manifest missing columns: {missing}'
    for col in ('k', 'alpha', 'tau', 'seed'):
        if col not in df.columns:
            df[col] = np.nan
    return df


def load_metrics(run_dir):
    path = os.path.join(os.path.expanduser(run_dir), 'metrics.jsonl')
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return pd.DataFrame(rows).sort_values('step').reset_index(drop=True)


def load_eval(run_dir):
    path = os.path.join(os.path.expanduser(run_dir), 'eval_results.csv')
    if not os.path.exists(path):
        return None
    return pd.read_csv(path).sort_values('step').reset_index(drop=True)


def pass16_col(metrics: pd.DataFrame, benchmark: str = MATH500):
    """Best-available in-training pass@n column for a benchmark."""
    for col in (f'val/pass_at_16/{benchmark}', f'val/pass_at_8/{benchmark}',
                f'val/pass_at_4/{benchmark}', f'val/pass_at_1/{benchmark}',
                f'val/test_score/{benchmark}'):
        if col in metrics.columns:
            return col
    raise KeyError(f'no validation metric for {benchmark} in metrics.jsonl')


def val_curve(metrics: pd.DataFrame, benchmark: str = MATH500):
    """(steps, values, cumulative_bp, elapsed_hours) at validation points."""
    col = pass16_col(metrics, benchmark)
    m = metrics.dropna(subset=[col])
    return (m['step'].to_numpy(), m[col].to_numpy(),
            m.get('actor/cumulative_bp', pd.Series(np.nan, index=m.index)).to_numpy(),
            m.get('train/elapsed_hours', pd.Series(np.nan, index=m.index)).to_numpy())


def ema(values, span=10):
    if len(values) == 0:
        return values
    return pd.Series(values).ewm(span=span, adjust=False).mean().to_numpy()


def first_at_least(steps, values, threshold):
    """First index where value >= threshold, or None."""
    idx = np.argmax(np.asarray(values) >= threshold)
    if len(values) == 0 or values[idx] < threshold:
        return None
    return idx


def nearest_row(df, col, target):
    """Row of df whose df[col] is nearest to target (NaNs dropped)."""
    d = df.dropna(subset=[col])
    if d.empty:
        return None
    return d.iloc[(d[col] - target).abs().argmin()]


def write_table(df: pd.DataFrame, out_dir, name, float_format='%.2f'):
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, f'{name}.csv'), index=False)
    with open(os.path.join(out_dir, f'{name}.tex'), 'w') as f:
        f.write(df.to_latex(index=False, float_format=lambda x: float_format % x, na_rep='--'))
    print(f'wrote {name}.csv / {name}.tex ({len(df)} rows)')
