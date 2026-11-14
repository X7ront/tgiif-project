#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to a TGIIF NELL-995 checkpoint path}"
SPLIT="${SPLIT:-valid}"
M="${M:-1.0}"
FUSION_MODE="${FUSION_MODE:-invariant-margin}"
# Latest NELL-995 stagewise adapter setting.  Training and evaluation both
# use the invariant branch without the learned treatment gate.
INVARIANT_MARGIN_FRACTION="${INVARIANT_MARGIN_FRACTION:-0.05}"
INVARIANT_MARGIN_UNGATED="${INVARIANT_MARGIN_UNGATED:-1}"
DELTA_NORMALIZATION="${DELTA_NORMALIZATION:-auto}"
DELTA_TARGET_RMS="${DELTA_TARGET_RMS:-auto}"
INVARIANT_BRANCH_MULTIPLIER="${INVARIANT_BRANCH_MULTIPLIER:-1.0}"
BIAS_BRANCH_MULTIPLIER="${BIAS_BRANCH_MULTIPLIER:-1.0}"
MAX_GAP_RATIO="${MAX_GAP_RATIO:-none}"
FLOOR_AT_BASE_MIN="${FLOOR_AT_BASE_MIN:-0}"
TIE_POLICY_OVERRIDE="${TIE_POLICY_OVERRIDE:-checkpoint}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-8}"
OMP_THREADS="${OMP_THREADS:-8}"
OUTPUT="${OUTPUT:-./logs/v5c_manual_eval_${SPLIT}_m${M}.log}"

cd "$PROJECT_DIR"
mkdir -p "$(dirname "$OUTPUT")"
export OMP_NUM_THREADS="$OMP_THREADS"
export MKL_NUM_THREADS="$OMP_THREADS"

args=(
  "$CHECKPOINT"
  --architecture treatment-aware
  --data-path ./benchmarks/NELL-995/
  --split "$SPLIT"
  --hash-hops 1
  --feature-mode triple
  --hidden-dim 128
  --branch-dim 64
  --branch-activation relu
  --branch-negative-slope 0.01
  --delta-temperature 1.0
  --delta-normalization "$DELTA_NORMALIZATION"
  --gate-temperature 1.0
  --fusion-mode "$FUSION_MODE"
  --invariant-margin-fraction "$INVARIANT_MARGIN_FRACTION"
  --invariant-branch-multiplier "$INVARIANT_BRANCH_MULTIPLIER"
  --bias-branch-multiplier "$BIAS_BRANCH_MULTIPLIER"
  --tie-policy-override "$TIE_POLICY_OVERRIDE"
  --residual-multiplier "$M"
  --threads "$THREADS"
  --device "$DEVICE"
  --skip-diagnostics
  --filtered-ranking
)
[[ "$DELTA_TARGET_RMS" != "auto" ]] && args+=(--delta-target-rms "$DELTA_TARGET_RMS")
[[ "$INVARIANT_MARGIN_UNGATED" == "1" ]] && args+=(--invariant-margin-ungated)
[[ "$FLOOR_AT_BASE_MIN" == "1" ]] && args+=(--floor-residual-at-baseline-min)
[[ "$MAX_GAP_RATIO" != "none" ]] && args+=(--residual-max-top-gap-ratio "$MAX_GAP_RATIO")

"$PYTHON_BIN" examples/evaluate_tgiif_checkpoint_NELL995.py "${args[@]}" \
  2>&1 | tee "$OUTPUT"
