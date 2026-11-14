#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-checkpoint/ctsnet_tgiif_fb15k237_full_ep4000_s23_FB15k-237_h1_0.005_Adam_ep4000.ckpt}"
SPLIT="${SPLIT:-valid}"
M="${M:-0.75}"
FRACTIONS="${FRACTIONS:-0.000001 0.00001 0.0001 0.001 0.01 0.1 0.25}"
RUN_BASELINE="${RUN_BASELINE:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-logs/fraction_sweep_full_ep4000_s23_20260913}"
DEVICE="${DEVICE:-cuda}"
THREADS="${THREADS:-8}"
OMP_THREADS="${OMP_THREADS:-8}"

cd "$PROJECT_DIR"
mkdir -p "$OUTPUT_DIR"
test -f "$CHECKPOINT"

export OMP_NUM_THREADS="$OMP_THREADS"
export MKL_NUM_THREADS="$OMP_THREADS"

common_args=(
  "$CHECKPOINT"
  --architecture treatment-aware
  --data-path ./benchmarks/FB15k-237/
  --split "$SPLIT"
  --hash-hops 1
  --feature-mode triple
  --hidden-dim 128
  --branch-dim 64
  --branch-activation relu
  --branch-negative-slope 0.01
  --delta-temperature 1.0
  --delta-normalization none
  --delta-target-rms 0.25
  --gate-temperature 1.0
  --fusion-mode invariant-margin
  --invariant-branch-multiplier 1.0
  --bias-branch-multiplier 1.0
  --threads "$THREADS"
  --device "$DEVICE"
  --skip-diagnostics
  --filtered-ranking
)

manifest="$OUTPUT_DIR/sweep_manifest.tsv"
if [[ ! -f "$manifest" ]]; then
  printf 'split\tm\tfraction\tlog\n' > "$manifest"
fi

prepare_log() {
  local output_log="$1"
  if [[ -f "$output_log" ]] && grep -q 'Filtered ranking' "$output_log"; then
    grep 'Filtered ranking' "$output_log"
    return 1
  fi
  if [[ -f "$output_log" ]]; then
    mv "$output_log" "${output_log}.interrupted.$(date +%Y%m%d-%H%M%S)"
  fi
  return 0
}

record_manifest() {
  local line="$1"
  grep -Fqx "$line" "$manifest" || printf '%s\n' "$line" >> "$manifest"
}

if [[ "$RUN_BASELINE" == "1" ]]; then
  baseline_log="$OUTPUT_DIR/eval_${SPLIT}_f0.0001_m0.log"
  if prepare_log "$baseline_log"; then
    echo "START split=$SPLIT baseline m=0"
    "$PYTHON_BIN" examples/evaluate_tgiif_checkpoint_FB15k237.py \
      "${common_args[@]}" \
      --invariant-margin-fraction 0.0001 \
      --residual-multiplier 0 \
      > "$baseline_log" 2>&1
    grep 'Filtered ranking' "$baseline_log"
  else
    echo "SKIP completed split=$SPLIT baseline m=0"
  fi
  record_manifest "$SPLIT"$'\t'"0"$'\t'"0.0001"$'\t'"$baseline_log"
fi

for fraction in $FRACTIONS; do
  output_log="$OUTPUT_DIR/eval_${SPLIT}_f${fraction}_m${M}.log"
  if prepare_log "$output_log"; then
    echo "START split=$SPLIT fraction=$fraction m=$M"
    "$PYTHON_BIN" examples/evaluate_tgiif_checkpoint_FB15k237.py \
      "${common_args[@]}" \
      --invariant-margin-fraction "$fraction" \
      --residual-multiplier "$M" \
      > "$output_log" 2>&1
    grep 'Filtered ranking' "$output_log"
  else
    echo "SKIP completed split=$SPLIT fraction=$fraction m=$M"
  fi
  record_manifest "$SPLIT"$'\t'"$M"$'\t'"$fraction"$'\t'"$output_log"
done

echo "FRACTION_SWEEP_DONE output=$OUTPUT_DIR"
