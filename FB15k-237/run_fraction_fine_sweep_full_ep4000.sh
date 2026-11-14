#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_DIR
export RUN_BASELINE=0
export FRACTIONS="${FRACTIONS:-0.002 0.003 0.005 0.007 0.015 0.02 0.03}"
export OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_DIR/logs/fraction_sweep_full_ep4000_s23_20260913}"

exec bash "$PROJECT_DIR/run_fraction_sweep_full_ep4000.sh"
