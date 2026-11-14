#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"
PYTHON_BIN="${PYTHON_BIN:-python}"
EPOCHS=1000
SEED=23
LR=0.005
TRAIN_FRACTION=0.25
EVAL_FRACTION=0.0001

export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8

run_one() {
  local nbatches="$1"
  local tag="tgiif_batch_nb${nbatches}_ep${EPOCHS}_s${SEED}"
  local log_dir="./logs/${tag}"
  local manifest="${log_dir}/run_manifest.log"

  if [[ -e "$log_dir" ]]; then
    echo "Refusing to overwrite existing run directory: $log_dir" >&2
    return 2
  fi

  mkdir -p "$log_dir"
  {
    printf 'tag=%s\nnbatches=%s\npositive_batch_size=%s\n' \
      "$tag" "$nbatches" "$((86835 / nbatches))"
    printf 'epochs=%s\nlr=%s\nseed=%s\ntrain_fraction=%s\neval_fraction=%s\n' \
      "$EPOCHS" "$LR" "$SEED" "$TRAIN_FRACTION" "$EVAL_FRACTION"
    sha256sum benchmarks/WN18RR/train2id.txt benchmarks/WN18RR/valid2id.txt \
      benchmarks/WN18RR/test2id.txt counterfactual_cache/WN18RR_popularity_h1.pt
    nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
  } > "${log_dir}/experiment_provenance.txt"

  echo "START training nbatches=${nbatches} tag=${tag}"
  PROJECT_DIR="$PROJECT_DIR" PYTHON_BIN="$PYTHON_BIN" ABLATION=full \
    EPOCHS="$EPOCHS" SEED="$SEED" RUN_TAG="$tag" \
    NBATCHES="$nbatches" NEG_ENT=25 LR="$LR" \
    FUSION_MODE=invariant-margin INVARIANT_MARGIN_FRACTION="$TRAIN_FRACTION" \
    RUN_EVAL=0 EVAL_SPLIT=valid bash ./run_v5c_wn18rr_cloud.sh \
    > "${log_dir}/runner_console.log" 2>&1 || {
      echo "FAILED training nbatches=${nbatches}; tail follows" >&2
      tail -n 30 "${log_dir}/runner_console.log" >&2
      return 1
    }

  local checkpoint
  checkpoint="$(sed -n 's/^checkpoint=//p' "$manifest" | tail -n 1)"
  if [[ -z "$checkpoint" || ! -f "$checkpoint" ]]; then
    echo "Missing checkpoint recorded in $manifest" >&2
    return 3
  fi
  printf 'checkpoint=%s\n' "$checkpoint" >> "${log_dir}/experiment_provenance.txt"
  sha256sum "$checkpoint" >> "${log_dir}/experiment_provenance.txt"

  local eval_args=(
    --architecture treatment-aware --data-path ./benchmarks/WN18RR/
    --split valid --hash-hops 1 --feature-mode triple
    --hidden-dim 128 --branch-dim 64 --branch-activation relu
    --branch-negative-slope 0.01 --delta-temperature 1.0
    --delta-normalization none --delta-target-rms 0.25
    --gate-temperature 1.0 --fusion-mode invariant-margin
    --invariant-margin-fraction "$EVAL_FRACTION"
    --threads 8 --device cuda --skip-diagnostics --filtered-ranking
  )

  echo "START validation nbatches=${nbatches} m=0"
  "$PYTHON_BIN" examples/evaluate_ctsnet_checkpoint_WN18RR.py \
    "$checkpoint" "${eval_args[@]}" --residual-multiplier 0 \
    > "${log_dir}/eval_valid_m0.log" 2>&1
  grep 'Filtered ranking (valid)' "${log_dir}/eval_valid_m0.log" | tail -n 1

  echo "START validation nbatches=${nbatches} m=0.75"
  "$PYTHON_BIN" examples/evaluate_ctsnet_checkpoint_WN18RR.py \
    "$checkpoint" "${eval_args[@]}" --residual-multiplier 0.75 \
    > "${log_dir}/eval_valid_m0.75.log" 2>&1
  grep 'Filtered ranking (valid)' "${log_dir}/eval_valid_m0.75.log" | tail -n 1

  echo "COMPLETE nbatches=${nbatches} log_dir=${log_dir}"
}

run_one 200
run_one 50
