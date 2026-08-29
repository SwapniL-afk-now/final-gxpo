#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 || $# -gt 2 ]]; then
  echo "usage: $0 PID [CSV_LOG_PATH]" >&2
  exit 2
fi

TARGET_PID="$1"
LOG_PATH="${2:-}"
GPU_IDS="${POWER_WATCH_GPU_IDS:-0,1}"
INTERVAL="${POWER_WATCH_INTERVAL:-1}"
OVER_WATTS="${POWER_WATCH_OVER_WATTS:-500}"

if [[ -n "$LOG_PATH" ]]; then
  mkdir -p "$(dirname -- "$LOG_PATH")"
  printf '%s\n' 'timestamp,index,power.draw [W],utilization.gpu [%],memory.used [MiB]' > "$LOG_PATH"
fi

echo "[power-watch] pid=$TARGET_PID gpus=$GPU_IDS interval=${INTERVAL}s over=${OVER_WATTS}W"
while kill -0 "$TARGET_PID" 2>/dev/null; do
  sample="$(nvidia-smi --id="$GPU_IDS" \
    --query-gpu=timestamp,index,power.draw,utilization.gpu,memory.used \
    --format=csv,noheader,nounits 2>/dev/null || true)"
  now="$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
  while IFS= read -r row; do
    [[ -z "$row" ]] && continue
    echo "$now,$row"
    if [[ -n "$LOG_PATH" ]]; then
      printf '%s,%s\n' "$now" "$row" >> "$LOG_PATH"
    fi
    if [[ "$OVER_WATTS" != 0 ]]; then
      power="$(awk -F',' '{gsub(/[[:space:]]/, "", $3); print $3}' <<< "$row")"
      if awk -v power="$power" -v limit="$OVER_WATTS" 'BEGIN { exit !(power+0 > limit) }'; then
        echo "[power-watch] OVER LIMIT: gpu row=$row" >&2
      fi
    fi
  done <<< "$sample"
  sleep "$INTERVAL"
done
echo "[power-watch] target pid $TARGET_PID exited"
