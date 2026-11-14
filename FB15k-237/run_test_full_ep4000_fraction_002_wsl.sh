#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/xtront/miniconda3/etc/profile.d/conda.sh
conda activate intersection_feature_cu128
cd "$PROJECT_DIR"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"

CHECKPOINT="checkpoint/ctsnet_tgiif_fb15k237_full_ep4000_s23_FB15k-237_h1_0.005_Adam_ep4000.ckpt"
OUTPUT="logs/fraction_sweep_full_ep4000_s23_20260913/eval_test_f0.02_m0.75.log"

exec env \
  CHECKPOINT="$CHECKPOINT" \
  SPLIT=test \
  M=0.75 \
  INVARIANT_MARGIN_FRACTION=0.02 \
  OUTPUT="$OUTPUT" \
  PYTHON_BIN=python \
  bash eval_tgiif_fb15k237.sh
