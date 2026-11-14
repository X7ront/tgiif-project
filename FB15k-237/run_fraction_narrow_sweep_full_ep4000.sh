#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR
export RUN_BASELINE=0
export FRACTIONS="${FRACTIONS:-0.0175 0.018 0.019 0.021 0.022 0.025}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/logs/fraction_sweep_full_ep4000_s23_20260913}"

exec bash "$PROJECT_DIR/run_fraction_fine_sweep_full_ep4000.sh"
