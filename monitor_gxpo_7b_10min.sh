#!/usr/bin/env bash
set -u
# Monitor Qwen2.5-Math-7B GXPO training every 10 minutes until 400 steps.
# Session: gxpo-qwen25-7b-k10-a03
# Log: /workspace/final-gxpo/results/gxpo_efficiency/qwen25_math_7b_k10_a03_queued.log
# Run dir: /workspace/final-gxpo/Code/SFPO/results/gxpo_efficiency/qwen25_math_7b_instruct_gxpo_b256_mb64_a0.3_k10_entropy_txopt_tau1.5_p2_m150_tp10_v048

SESSION="gxpo-qwen25-7b-k10-a03"
QUEUED_LOG="/workspace/final-gxpo/results/gxpo_efficiency/qwen25_math_7b_k10_a03_queued.log"
RUN_DIR="/workspace/final-gxpo/Code/SFPO/results/gxpo_efficiency/qwen25_math_7b_instruct_gxpo_b256_mb64_a0.3_k10_entropy_txopt_tau1.5_p2_m150_tp10_v048"
TRAIN_LOG="$RUN_DIR/train.log"
LAUNCHER="/workspace/final-gxpo/Code/SFPO/experiments/gxpo_efficiency/qwen25_math_7b_gxpo_b256_a03_k10.sh"
STATUS_LOG="/workspace/final-gxpo/results/gxpo_efficiency/monitor_7b_10min.log"
MAX_STEPS=400
INTERVAL=600

mkdir -p "$(dirname "$STATUS_LOG")"

log_status() {
  echo "[$(date -u +%FT%TZ)] $*" | tee -a "$STATUS_LOG"
}

check_trainer_alive() {
  if pgrep -f "verl.trainer.main_ppo.*Qwen.*7B" >/dev/null 2>&1 || pgrep -f "verl.trainer.main_ppo.*qwen25_math_7b" >/dev/null 2>&1; then
    echo 1
  else
    # also check main pid 597396 if still alive
    if ps -p 597396 >/dev/null 2>&1; then echo 1; else echo 0; fi
  fi
}

get_last_step() {
  # Try multiple patterns: "step:" "global_step" "Global_steps done"
  local step=""
  if [[ -f "$QUEUED_LOG" ]]; then
    step=$(grep -a -o "global_step:[[:space:]]*[0-9]\+" "$QUEUED_LOG" 2>/dev/null | tail -1 | grep -o "[0-9]\+" || true)
    if [[ -z "$step" ]]; then
      step=$(grep -a -o "step:[0-9]\+" "$QUEUED_LOG" 2>/dev/null | tail -1 | grep -o "[0-9]\+" || true)
    fi
    if [[ -z "$step" ]]; then
      step=$(grep -a -o "Global_steps done: [0-9]\+" "$QUEUED_LOG" 2>/dev/null | tail -1 | grep -o "[0-9]\+" || true)
    fi
    if [[ -z "$step" ]]; then
      step=$(grep -a -o "train/global_step:[[:space:]]*[0-9.]\+" "$QUEUED_LOG" 2>/dev/null | tail -1 | grep -o "[0-9]\+" | head -1 || true)
    fi
  fi
  if [[ -z "$step" && -f "$TRAIN_LOG" ]]; then
    step=$(grep -a -o "train/global_step:[[:space:]]*[0-9.]\+" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -o "[0-9]\+" | head -1 || true)
    if [[ -z "$step" ]]; then
      step=$(grep -a -o "global_step:[[:space:]]*[0-9]\+" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -o "[0-9]\+" || true)
    fi
  fi
  echo "${step:-0}"
}

check_log_stale() {
  local log="$1"
  local stale_sec=1800  # 30 min considered stalled for 7B
  if [[ ! -f "$log" ]]; then echo "missing"; return; fi
  local mtime=$(stat -c %Y "$log" 2>/dev/null || echo 0)
  local now=$(date +%s)
  local diff=$((now - mtime))
  echo "$diff"
}

detect_errors() {
  local log="$1"
  local errs=""
  if [[ -f "$log" ]]; then
    # Precise system error detection - avoid matching truncation error or prediction content
    # Only flag if actual system CUDA OOM, not just word OOM in prediction gibberish
    if grep -a -q "RuntimeError: CUDA out of memory\|torch.cuda.OutOfMemoryError\|CUDA_ERROR.*out of memory\|an illegal memory access was encountered" "$log" 2>/dev/null; then errs+=" CUDA_OOM;"; fi
    if grep -a -q "FSDP.*error\|fsdp.*failed" "$log" 2>/dev/null; then errs+=" FSDP_ERROR;"; fi
    if grep -a -q "vllm.*error\|VLLM.*error" "$log" 2>/dev/null; then errs+=" VLLM_ERROR;"; fi
    # Traceback should be counted only if not inside the huge prediction dump that contains Traceback-like text? Check for python Traceback with following File line
    if grep -a -q "Traceback (most recent call last)" "$log" 2>/dev/null && grep -a -q 'File "' "$log" 2>/dev/null; then
      # Ensure it's not just the warning inside Pred text - check if line count of Traceback is >1 and associated with File?
      # Count occurrences - if only warnings about prediction format, ignore
      if grep -a -c "Traceback (most recent call last)" "$log" 2>/dev/null | grep -qv "^0$"; then
        # Check that next lines contain .py files, not just prediction text
        if grep -A2 "Traceback (most recent call last)" "$log" 2>/dev/null | grep -q 'File ".*\.py' ; then
          errs+=" TRACEBACK;"
        fi
      fi
    fi
    if grep -a -q "Missing.*parquet\|FileNotFoundError.*parquet\|Missing prepared dataset" "$log" 2>/dev/null; then errs+=" DATASET_ERROR;"; fi
    if grep -a -q "wandb.*ERROR\|Failed to.*wandb" "$log" 2>/dev/null; then errs+=" WANDB_ERROR;"; fi
    if grep -a -q "NCCL.*error" "$log" 2>/dev/null; then errs+=" NCCL_ERROR;"; fi
  fi
  echo "${errs:-None}"
}

gpu_status() {
  nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader 2>&1 | tr -d '\r' || echo "nvidia-smi failed"
}

wandb_status() {
  if [[ -d "$RUN_DIR/wandb" ]]; then
    local wandb_files=$(find "$RUN_DIR/wandb" -name "*.wandb" -type f 2>/dev/null | head -5)
    local wandb_size=$(du -sh "$RUN_DIR/wandb" 2>/dev/null | cut -f1)
    local latest=$(ls -t "$RUN_DIR/wandb"/run-*/run-*.wandb 2>/dev/null | head -1)
    local mtime="unknown"
    if [[ -n "$latest" && -f "$latest" ]]; then
      mtime=$(stat -c %y "$latest" 2>/dev/null)
    fi
    echo "wandb_dir_exists size=$wandb_size latest=$latest mtime=$mtime"
  else
    echo "wandb_missing"
  fi
}

tmux_status() {
  if tmux has-session -t "$SESSION" 2>/dev/null; then echo "tmux_session_exists"; else echo "tmux_missing"; fi
  tmux list-panes -t "$SESSION" 2>&1 | head -20
  # check tee processes
  pgrep -a tee 2>&1 | head -20
}

per_step_metrics() {
  # Extract last step's metrics if available
  if [[ -f "$QUEUED_LOG" ]]; then
    grep -a "step:" "$QUEUED_LOG" 2>/dev/null | tail -1 | cut -c1-800 || echo "no step line yet"
  else
    echo "no log"
  fi
}

first_six_check() {
  local log="$QUEUED_LOG"
  if [[ ! -f "$log" ]]; then echo "log_missing"; return; fi
  local head_lines=$(head -n 800 "$log" 2>/dev/null)
  # Check first six steps would be steps 0-5 or 1-6
  local steps_found=$(grep -a -c "step:" "$log" 2>/dev/null || echo 0)
  echo "steps_found=$steps_found"
  # Look for errors in first 6 steps window (approx first 400KB)
  local early=$(head -c 4000000 "$log" 2>/dev/null | grep -a -i "Traceback\|CUDA out of memory\|unbound variable\|launcher exited with code [1-9]\|ERROR\|Failed" | head -20 || echo "none")
  echo "early_errors: $early"
}

restart_if_needed() {
  local reason="$1"
  log_status "!!! RESTART TRIGGERED: $reason"
  # Identify root cause
  log_status "Root cause search:"
  grep -a -n "Traceback\|CUDA out of memory\|OOM\|ERROR\|Failed\|wandb.*error\|FSDP.*error\|vllm.*error" "$QUEUED_LOG" 2>/dev/null | tail -n 100 | tee -a "$STATUS_LOG" || true
  grep -a -n "Traceback\|CUDA out of memory\|OOM\|ERROR\|Failed" "$TRAIN_LOG" 2>/dev/null | tail -n 100 | tee -a "$STATUS_LOG" || true
  tmux capture-pane -t "$SESSION:0" -p 2>&1 | tail -n 200 | tee -a "$STATUS_LOG" || true

  log_status "Stopping failed run..."
  # Find trainer pids
  local pids=$(pgrep -f "verl.trainer.main_ppo" 2>/dev/null || true)
  if [[ -n "$pids" ]]; then
    for pid in $pids; do
      log_status "Killing trainer pid $pid"
      kill -TERM "$pid" 2>/dev/null || true
    done
    sleep 10
    for pid in $pids; do
      if kill -0 "$pid" 2>/dev/null; then
        log_status "Force kill $pid"
        kill -KILL "$pid" 2>/dev/null || true
      fi
    done
  fi
  # Also kill Ray if needed
  pkill -f "ray::" 2>/dev/null || true
  sleep 5

  log_status "Deleting only previous 7B checkpoint directories to free space, preserving unrelated checkpoints..."
  # Preserve 1.5B and llama, delete only 7B
  # List what will be deleted first
  echo "Will delete: " | tee -a "$STATUS_LOG"
  ls -d "$RUN_DIR" 2>&1 | tee -a "$STATUS_LOG" || true
  ls -d /workspace/final-gxpo/Code/SFPO/results/gxpo_efficiency/qwen25_math_7b* 2>&1 | tee -a "$STATUS_LOG" || true
  ls -d /workspace/final-gxpo/results/gxpo_efficiency/qwen25_math_7b* 2>&1 | tee -a "$STATUS_LOG" || true
  # Actually delete only the specific 7B run dir and its checkpoints, not whole results
  # Keep other models safe
  if [[ -d "$RUN_DIR" ]]; then
    # Find global_step directories
    ls -d "$RUN_DIR"/global_step_* 2>&1 | head -20 | tee -a "$STATUS_LOG" || true
    rm -rf "$RUN_DIR"/global_step_* 2>&1 | tee -a "$STATUS_LOG" || true
    rm -rf "$RUN_DIR"/checkpoint* 2>&1 | tee -a "$STATUS_LOG" || true
    # Also remove latest_checkpointed files to force fresh start
    rm -f "$RUN_DIR"/latest_checkpointed_iteration.txt 2>&1 | tee -a "$STATUS_LOG" || true
  fi
  # Clean queued log? Keep for forensic but truncate? The task says delete only checkpoint dirs, not logs. So keep logs but start fresh log via restart.
  # Do not delete unrelated: ensure we don't rm 1.5b
  if ls /workspace/final-gxpo/Code/SFPO/results/gxpo_efficiency/qwen25_math_1p5b* >/dev/null 2>&1; then
    log_status "Preserved 1.5B checkpoints: $(ls -d /workspace/final-gxpo/Code/SFPO/results/gxpo_efficiency/qwen25_math_1p5b* 2>&1 | tr '\n' ' ')"
  fi
  if ls /workspace/final-gxpo/Code/SFPO/results/gxpo_efficiency/llama* >/dev/null 2>&1; then
    log_status "Preserved llama checkpoints: $(ls -d /workspace/final-gxpo/Code/SFPO/results/gxpo_efficiency/llama* 2>&1 | tr '\n' ' ')"
  fi

  log_status "Fixing configuration/ launcher - verifying settings (do not enable entropy reset, preserve 7B settings)..."
  # Verify launcher settings
  grep -E "GXPO_RESET_ENTROPY_AFTER_WARMUP|GXPO_TAU|GXPO_ZSCORE_W|GXPO_TRIGGER_PATIENCE|K=|REPOSITION_ALPHA|TRAIN_BATCH_SIZE|PPO_MINI|FSDP_SIZE|ATTN_IMPL|GPU_COUNT|VLLM_GPU_MEMORY" "$LAUNCHER" 2>&1 | tee -a "$STATUS_LOG" || true
  # Ensure entropy reset is False (as required)
  if grep -q 'GXPO_RESET_ENTROPY_AFTER_WARMUP.*True' "$LAUNCHER" 2>&1; then
    log_status "ERROR: launcher has entropy reset True, patching to False"
    sed -i 's/GXPO_RESET_ENTROPY_AFTER_WARMUP.*True/GXPO_RESET_ENTROPY_AFTER_WARMUP:-False/' "$LAUNCHER" || true
  fi
  # Ensure WANDB online
  export WANDB_MODE=online
  if [[ -f /workspace/.env ]]; then
    source /workspace/.env 2>/dev/null || true
  fi
  if [[ -z "${WANDB_API_KEY:-}" ]]; then
    log_status "WARNING: WANDB_API_KEY empty, trying to load from env files"
    # Try repo .env
    if [[ -f /workspace/final-gxpo/.env ]]; then source /workspace/final-gxpo/.env 2>/dev/null || true; fi
  fi
  log_status "WANDB_MODE=$WANDB_MODE WANDB_API_KEY set? $([[ -n "${WANDB_API_KEY:-}" ]] && echo yes || echo no)"

  log_status "Restart fresh in same tmux session $SESSION with logs attached and W&B online..."
  # Ensure tmux session exists
  if ! tmux has-session -t "$SESSION" >/dev/null 2>&1; then
    log_status "Recreating tmux session $SESSION"
    tmux new-session -d -s "$SESSION" -c /workspace/final-gxpo
  fi
  # Kill any stale pane process that may hold the session
  tmux send-keys -t "$SESSION:0" C-c 2>&1 || true
  sleep 2
  # Clear pane and restart
  tmux send-keys -t "$SESSION:0" "cd /workspace/final-gxpo; export WANDB_MODE=online; bash \"$LAUNCHER\" 2>&1 | tee \"$QUEUED_LOG\"; rc=\${PIPESTATUS[0]}; echo \"7B launcher exited with code \$rc\"" Enter 2>&1 || true
  log_status "Restart command sent to tmux pane 0. Waiting 90s for first logs..."
  sleep 90
  log_status "Rechecking first six steps after restart..."
  # Re-check
  tmux capture-pane -t "$SESSION:0" -p 2>&1 | head -n 200 | tee -a "$STATUS_LOG" || true
  tail -n 500 "$QUEUED_LOG" 2>&1 | tee -a "$STATUS_LOG" || true
  # Check for errors in early window
  if grep -a -qi "Traceback\|CUDA out of memory\|unbound variable" "$QUEUED_LOG" 2>&1; then
    log_status "ERROR still present after restart, need manual fix"
  else
    log_status "Restart appears clean, monitoring continues"
  fi
}

log_status "=== GXPO 7B 10-min monitor started $(date -u) ==="
log_status "Session=$SESSION Launcher=$LAUNCHER RunDir=$RUN_DIR MaxSteps=$MAX_STEPS"
log_status "Preserving: transactional optimizer, entropy trigger, z=30, tau=1.5, patience=2, K=10, alpha=0.3, batch256, minibatch64, FSDP4, FA3, vLLM, 4 GPUs, no entropy reset"

ITER=0
while true; do
  ITER=$((ITER+1))
  log_status "==== ITER $ITER $(date -u +%FT%TZ) ===="

  ALIVE=$(check_trainer_alive)
  LAST_STEP=$(get_last_step)
  QUEUED_STALE=$(check_log_stale "$QUEUED_LOG")
  TRAIN_STALE=$(check_log_stale "$TRAIN_LOG")
  ERRORS_Q=$(detect_errors "$QUEUED_LOG")
  ERRORS_T=$(detect_errors "$TRAIN_LOG")
  GPU=$(gpu_status)
  WANDB=$(wandb_status)
  TMUX=$(tmux_status)
  METRICS=$(per_step_metrics)

  log_status "Trainer alive=$ALIVE last_step=$LAST_STEP / $MAX_STEPS"
  log_status "Queued log stale=${QUEUED_STALE}s Train log stale=${TRAIN_STALE}s"
  log_status "Errors queued: $ERRORS_Q"
  log_status "Errors train: $ERRORS_T"
  log_status "Per-step metrics: $METRICS"
  # Detailed gpu
  log_status "GPU: $GPU"
  log_status "W&B: $WANDB"
  log_status "Tmux: $TMUX"

  # First six steps check
  first_six_check | tee -a "$STATUS_LOG" || true

  # Training metrics detail if steps >0
  if [[ "$LAST_STEP" != "0" && -f "$QUEUED_LOG" ]]; then
    # Try to extract last reward/entropy/GXPO
    grep -a -o "reward/mean:[0-9.]*\|actor/entropy_loss:[0-9.]*\|train/accuracy:[0-9.]*\|gxpo/.*:[0-9.]*\|perf/max_memory.*_gb:[0-9.]*" "$QUEUED_LOG" 2>/dev/null | tail -n 30 | tee -a "$STATUS_LOG" || true
    # Also check W&B wandb file growth
    wandb_file=$(ls -t "$RUN_DIR"/wandb/run-*/run-*.wandb 2>/dev/null | head -1)
    if [[ -n "$wandb_file" ]]; then
      wandb_sz=$(stat -c %s "$wandb_file" 2>/dev/null || echo 0)
      log_status "W&B wandb file size=$wandb_sz"
      if [[ "$wandb_sz" -lt 1000 && "$LAST_STEP" -gt 2 ]]; then
        log_status "WARNING: W&B file suspiciously small, possible missing logging"
      fi
    fi
  else
    log_status "No steps yet - checking initial pipeline (CUDA, FSDP, vLLM, W&B, dataset)"
    # Ensure initial validation passed
    if grep -a -q "validation generation end" "$QUEUED_LOG" 2>/dev/null; then
      log_status "Initial validation completed OK"
    else
      log_status "Initial validation not yet complete or log buffering"
    fi
    if grep -a -q "After building vllm rollout" "$QUEUED_LOG" 2>/dev/null; then
      log_status "vLLM built OK"
    fi
    if grep -a -q "After building sharding manager" "$QUEUED_LOG" 2>/dev/null; then
      log_status "FSDP sharding manager built OK"
    fi
    if grep -a -q "wandb:.*View run at" "$QUEUED_LOG" 2>/dev/null; then
      log_status "W&B online OK"
    else
      log_status "W&B not yet confirmed"
    fi
  fi

  # Check tmux pane continues displaying logs
  if ! tmux has-session -t "$SESSION" >/dev/null 2>&1; then
    log_status "CRITICAL: tmux session missing!"
    restart_if_needed "tmux session missing"
    sleep $INTERVAL
    continue
  fi
  # Check if pane 0 dead
  pane_dead=$(tmux display-message -t "$SESSION:0" -p "#{pane_dead}" 2>&1 || echo 1)
  if [[ "$pane_dead" == "1" ]]; then
    log_status "CRITICAL: tmux pane 0 dead"
    restart_if_needed "tmux pane dead"
    sleep $INTERVAL
    continue
  fi

  # Check for stalled progress: log not updating for >1800s and trainer alive
  if [[ "$ALIVE" == "1" && "$QUEUED_STALE" != "missing" && "$QUEUED_STALE" -gt 1800 ]]; then
    # But allow initial generation time: if last_step=0, give more grace (up to 3600s)
    if [[ "$LAST_STEP" == "0" && "$QUEUED_STALE" -lt 3600 ]]; then
      log_status "Stale but within initial generation grace period (last_step=0, stale=${QUEUED_STALE}s <3600s)"
    else
      log_status "STALLED: log stale ${QUEUED_STALE}s with trainer alive and last_step=$LAST_STEP"
      restart_if_needed "stalled progress stale=${QUEUED_STALE}s"
      sleep $INTERVAL
      continue
    fi
  fi

  # Check for OOM/crash in logs
  if [[ "$ERRORS_Q" != "None" || "$ERRORS_T" != "None" ]]; then
    # Filter out harmless warnings
    if echo "$ERRORS_Q $ERRORS_T" | grep -q "CUDA_ERROR\|OOM\|TRACEBACK"; then
      log_status "ERROR detected: $ERRORS_Q $ERRORS_T"
      restart_if_needed "error detected $ERRORS_Q $ERRORS_T"
      sleep $INTERVAL
      continue
    else
      log_status "Non-critical warnings: $ERRORS_Q $ERRORS_T"
    fi
  fi

  # Check for W&B missing
  if echo "$WANDB" | grep -q "missing"; then
    log_status "W&B logging missing!"
    # Don't restart immediately if still early (<5 min), but warn
    if [[ "$LAST_STEP" -gt 2 ]]; then
      restart_if_needed "missing W&B logging"
      sleep $INTERVAL
      continue
    fi
  fi

  # Check for completion
  if [[ "$LAST_STEP" -ge "$MAX_STEPS" ]]; then
    log_status "SUCCESS: Reached $LAST_STEP / $MAX_STEPS steps! Verifying final checkpoint..."
    if [[ -d "$RUN_DIR/global_step_$MAX_STEPS" || -d "$RUN_DIR/global_step_$LAST_STEP" ]]; then
      log_status "Final checkpoint exists. Run completed successfully."
      log_status "Verifying final status..."
      ls -lh "$RUN_DIR"/global_step_* 2>&1 | tail -n 20 | tee -a "$STATUS_LOG" || true
      # Check wandb final
      wandb_status | tee -a "$STATUS_LOG" || true
      log_status "Monitoring complete. Exiting."
      exit 0
    else
      log_status "Steps reached but checkpoint missing, waiting..."
    fi
  fi

  # Also check if trainer exited before completion
  if [[ "$ALIVE" == "0" && "$LAST_STEP" -lt "$MAX_STEPS" ]]; then
    # Check if launcher exited with error code
    if tmux capture-pane -t "$SESSION:0" -p 2>&1 | grep -q "launcher exited with code [1-9]"; then
      log_status "Trainer dead before completion (launcher error). Triggering restart."
      restart_if_needed "trainer exited before completion"
      sleep $INTERVAL
      continue
    fi
    # If still within boot time (<10 min), don't restart yet - maybe starting up
    if [[ "$QUEUED_STALE" -lt 600 ]]; then
      log_status "Trainer not yet visible but log recently updated, waiting for startup"
    else
      log_status "Trainer dead and log stale, triggering restart"
      restart_if_needed "trainer dead"
      sleep $INTERVAL
      continue
    fi
  fi

  # Estimate per-step time if we have at least 2 steps
  if [[ "$LAST_STEP" -ge 2 ]]; then
    # Try to compute from cum_train_active_s if available
    last_time=$(grep -a "time/cum_train_active_s" "$QUEUED_LOG" 2>/dev/null | tail -1 | grep -o "[0-9.]\+" | tail -1 || echo "unknown")
    log_status "Cumulative train active s: $last_time"
  fi

  log_status "Sleeping $INTERVAL s until next check..."
  sleep $INTERVAL
done
