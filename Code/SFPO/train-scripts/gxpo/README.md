# GXPO paper reproduction pipeline

End-to-end workflow reproducing every table (1-8) and figure (2-9) of
*Gradient Extrapolation-Based Policy Optimization* on top of the SFPO verl fork.

## 1. Data

```bash
./train-scripts/prepare_gxpo_data.sh          # DATA_ROOT defaults to ~/data/gxpo
```

Train: Hendrycks MATH Level 3-5. Eval: Math-500, AMC23, GSM8K, Minerva, OlympiadBench.

## 2. Training runs

`run.sh` implements the paper protocol (batch 128 x 5 responses, lr 1e-7,
clip 0.2, KL beta 0.001, bf16, grad clip 1.0, 300 steps, LoRA r=128 alpha=256
on q/k/v/o; set `LORA_RANK=0` for full fine-tuning):

```bash
cd train-scripts/gxpo
# Table 1/2 main grid, per model:
for METHOD in grpo sfpo gxpo; do for K in 3 5 10; do
  METHOD=$METHOD K=$K MODEL_PATH=Qwen/Qwen2.5-1.5B-Instruct MODEL_TAG=qwen1.5b ./run.sh
done; done
./sweep_alpha.sh    # Tables 5-7, Figures 2/4/5/9
./sweep_tau.sh      # Figure 6
```

Each run writes `runs/<exp>/metrics.jsonl` (per-step: all actor/gxpo_* diagnostics,
cumulative backward passes, elapsed hours, val pass@1 / pass@16) plus periodic
checkpoints with `lora_adapter/` exports.

GXPO knobs (`+actor_rollout_ref.actor.*`): `use_gxpo, gxpo_k, gxpo_alpha,
gxpo_delta, gxpo_tau, gxpo_omega, gxpo_shutoff_mode (trajectory_aware|legacy_g0|never),
gxpo_recompute_old_log_probs, gxpo_diag_freq`.

## 3. Offline benchmark evaluation (Tables 1/3/4)

```bash
python scripts/eval_checkpoints.py \
  --run_dir train-scripts/gxpo/runs/qwen1.5b_gxpo_k5_a0.5_tau0.5_seed1 \
  --base_model Qwen/Qwen2.5-1.5B-Instruct \
  --data ~/data/gxpo/eval/*/test.parquet --n 16
```

Scoring uses the exact reward functions from training
(`verl.utils.reward_score._default_compute_score`), so train and eval
correctness criteria are identical.

## 4. Tables and figures

Write a manifest CSV (`runs.csv`):

```csv
run_dir,model,method,k,alpha,tau,seed
train-scripts/gxpo/runs/qwen1.5b_grpo_seed1,Qwen2.5-1.5B,grpo,,,,1
train-scripts/gxpo/runs/qwen1.5b_gxpo_k5_a0.5_tau0.5_seed1,Qwen2.5-1.5B,gxpo,5,0.5,0.5,1
```

```bash
python scripts/gxpo_report/make_tables.py  --manifest runs.csv --out report/tables
python scripts/gxpo_report/make_figures.py --manifest runs.csv --out report/figures
```

Emits Tables 1-8 as CSV + LaTeX and Figures 2-9 as PDF + PNG.

## Artifact -> paper mapping

| Paper artifact | Source |
|---|---|
| Table 1 (benchmarks) | eval_results.csv final step |
| Table 2 (convergence) | metrics.jsonl val curves + budgets |
| Tables 3/4 (iso-BP / iso-hours) | eval_results.csv joined on budgets |
| Table 5 (alpha sweep) | metrics.jsonl (shutoff step, budgets, best pass@16) |
| Table 6 (surrogate displacement) | actor/gxpo_diag_* (every `gxpo_diag_freq` active steps) |
| Table 7 (active-phase diagnostics) | actor/gxpo_* medians over active steps |
| Table 8 (KL/clip) | actor/pg_clipfrac, actor/kl_loss |
| Figures 2-9 | metrics.jsonl curves |
