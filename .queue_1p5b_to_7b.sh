#!/usr/bin/env bash
set -u

CURRENT_PID=485987
CURRENT_SESSION="gxpo-k10-a08-disagreement-sampling"
CURRENT_LOG="/workspace/final-gxpo/results/gxpo_efficiency/qwen25_math_1p5b_k10_a08_fixed_transactional_disagreement_sampling_400step.log"
SEVEN_SESSION="gxpo-qwen25-7b-k10-a03"
SEVEN_LAUNCHER="/workspace/final-gxpo/Code/SFPO/experiments/gxpo_efficiency/qwen25_math_7b_gxpo_b256_a03_k10.sh"
SEVEN_LOG="/workspace/final-gxpo/results/gxpo_efficiency/qwen25_math_7b_k10_a03_queued.log"

echo "queue watcher started $(date -u +%FT%TZ); waiting for 1.5B PID $CURRENT_PID"
while kill -0 "$CURRENT_PID" 2>/dev/null; do
  last_step="$(tail -c 2000000 "$CURRENT_LOG" 2>/dev/null | rg -o "Global_steps done: [0-9]+" | tail -1 | sed "s/Global_steps done: //" || true)"
  echo "$(date -u +%FT%TZ) 1.5B active; last logged step=${last_step:-unknown}"
  sleep 60
done

echo "$(date -u +%FT%TZ) 1.5B process exited; waiting for log flush"
sleep 15
if ! tail -c 2000000 "$CURRENT_LOG" 2>/dev/null | rg -q "Global_steps done: 400"; then
  echo "1.5B did not reach step 400; 7B launch cancelled"
  exec bash
fi

echo "$(date -u +%FT%TZ) verified 1.5B step 400; starting 7B"
if tmux has-session -t "$SEVEN_SESSION" 2>/dev/null; then
  echo "7B session $SEVEN_SESSION already exists; refusing duplicate launch"
  exec bash
fi

tmux new-session -d -s "$SEVEN_SESSION" bash -lc "
cd /workspace/final-gxpo
export WANDB_MODE=online
bash \"$SEVEN_LAUNCHER\" 2>&1 | tee \"$SEVEN_LOG\"
rc=\${PIPESTATUS[0]}
echo \"7B launcher exited with code \$rc\"
exec bash
"
tmux set-option -t "$SEVEN_SESSION" remain-on-exit on
echo "$(date -u +%FT%TZ) 7B started in tmux session $SEVEN_SESSION; log=$SEVEN_LOG"
exec bash
