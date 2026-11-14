#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/xtront/miniconda3/etc/profile.d/conda.sh
conda activate intersection_feature_cu128
exec bash "$PROJECT_DIR/run_fraction_narrow_sweep_full_ep4000.sh"
