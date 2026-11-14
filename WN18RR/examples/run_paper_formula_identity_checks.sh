#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:-./paper_evidence/smoke/checkpoints/ctsnet_paper_formula_smoke_full_ep20_WN18RR_h1_0.005_Adam_ep20.ckpt}"
OUTPUT_DIR="${OUTPUT_DIR:-./paper_evidence/smoke/identity_checks}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
mkdir -p "$OUTPUT_DIR"

COMMON_ARGS=(
  "$CHECKPOINT"
  --architecture treatment-aware
  --data-path ./benchmarks/WN18RR/
  --split valid
  --hash-hops 1
  --feature-mode triple
  --hidden-dim 128
  --branch-dim 64
  --branch-activation relu
  --branch-negative-slope 0.01
  --delta-normalization none
  --delta-temperature 1.0
  --gate-temperature 1.0
  --fusion-mode invariant-margin
  --invariant-margin-fraction 0.25
  --nbatches 100
  --neg-ent 25
  --threads 8
  --seed 23
  --device cuda
  --skip-diagnostics
  --filtered-ranking
)

run_eval() {
  local name="$1"
  shift
  echo "===== ${name} ====="
  "$PYTHON_BIN" examples/evaluate_ctsnet_checkpoint_WN18RR.py \
    "${COMMON_ARGS[@]}" "$@" 2>&1 | tee "$OUTPUT_DIR/${name}.log"
}

# Separate processes are required because OpenKE's Base.so evaluator keeps
# dataset state globally within a process.
run_eval full_m0p75 --residual-multiplier 0.75
run_eval official_m0 --residual-multiplier 0.0
run_eval disable_invariant_m0p75 \
  --residual-multiplier 0.75 --disable-invariant-branch
run_eval disable_bias_m0p75 \
  --residual-multiplier 0.75 --disable-bias-branch
