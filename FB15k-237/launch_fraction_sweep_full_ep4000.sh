#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/logs/fraction_sweep_full_ep4000_s23_20260913}"
PID_FILE="$OUTPUT_DIR/sweep.pid"
DRIVER_LOG="$OUTPUT_DIR/sweep_driver.log"

mkdir -p "$OUTPUT_DIR"
if [[ -f "$PID_FILE" ]]; then
  previous_pid="$(cat "$PID_FILE")"
  if kill -0 "$previous_pid" 2>/dev/null; then
    echo "Sweep is already running with PID $previous_pid" >&2
    exit 2
  fi
fi

source /home/xtront/miniconda3/etc/profile.d/conda.sh
conda activate intersection_feature_cu128

nohup bash "$PROJECT_DIR/run_fraction_sweep_full_ep4000.sh" \
  > "$DRIVER_LOG" 2>&1 < /dev/null &
sweep_pid=$!
printf '%s\n' "$sweep_pid" > "$PID_FILE"
echo "Started fraction sweep with PID $sweep_pid"
echo "Driver log: $DRIVER_LOG"
