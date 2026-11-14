#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ABLATION="${ABLATION:-full}"
EPOCHS="${EPOCHS:-1000}"
SEED="${SEED:-23}"
RUN_TAG="${RUN_TAG:-tgiif_fb15k237_${ABLATION}_ep${EPOCHS}_s${SEED}}"

DATASET="${DATASET:-FB15k-237}"
DATA_PATH="${DATA_PATH:-./benchmarks/FB15k-237/}"
CF_CACHE="${CF_CACHE:-./counterfactual_cache/FB15k-237_popularity_triple_h1.pt}"
DEVICE="${DEVICE:-cuda}"
CUDA_DEVICE="${CUDA_DEVICE:-0}"
THREADS="${THREADS:-8}"
OMP_THREADS="${OMP_THREADS:-8}"
NBATCHES="${NBATCHES:-100}"
NEG_ENT="${NEG_ENT:-25}"

LR="${LR:-0.005}"
OPTIMIZER="${OPTIMIZER:-Adam}"
MARGIN="${MARGIN:-5.0}"
HIDDEN_DIM="${HIDDEN_DIM:-128}"
BRANCH_DIM="${BRANCH_DIM:-64}"
RESIDUAL_SCALE="${RESIDUAL_SCALE:-0.1}"
BRANCH_ACTIVATION="${BRANCH_ACTIVATION:-relu}"
BRANCH_NEGATIVE_SLOPE="${BRANCH_NEGATIVE_SLOPE:-0.01}"
DELTA_TEMPERATURE="${DELTA_TEMPERATURE:-1.0}"
DELTA_NORMALIZATION="${DELTA_NORMALIZATION:-none}"
DELTA_TARGET_RMS="${DELTA_TARGET_RMS:-0.25}"
GATE_TEMPERATURE="${GATE_TEMPERATURE:-1.0}"
FUSION_MODE="${FUSION_MODE:-invariant-margin}"
INVARIANT_MARGIN_FRACTION="${INVARIANT_MARGIN_FRACTION:-0.25}"

RESIDUAL_HARD_NEGATIVES="${RESIDUAL_HARD_NEGATIVES:-5}"
RESIDUAL_NEAR_TIE_NEGATIVES="${RESIDUAL_NEAR_TIE_NEGATIVES:-0}"
RESIDUAL_START_EPOCH="${RESIDUAL_START_EPOCH:-100}"
RESIDUAL_WARMUP_EPOCHS="${RESIDUAL_WARMUP_EPOCHS:-100}"
RESIDUAL_LOSS_SCHEDULE="${RESIDUAL_LOSS_SCHEDULE:-auxiliary}"
TREATMENT_SUPERVISION="${TREATMENT_SUPERVISION:-factual}"
SHUFFLE_TREATMENT_LABELS="${SHUFFLE_TREATMENT_LABELS:-0}"
WARMUP_START_EPOCH="${WARMUP_START_EPOCH:-5}"
WARMUP_EPOCHS="${WARMUP_EPOCHS:-10}"

profile_lambda_cf=0.1
profile_lambda_treat=0.1
profile_lambda_orth=0.01
profile_lambda_gate=0.0
profile_lambda_residual_rank=1.0
profile_lambda_invariant_rank=0.0
profile_lambda_bias_rank=0.0
profile_lambda_residual_preserve=1.0
profile_invariant_branch_multiplier=1.0
profile_bias_branch_multiplier=1.0
profile_baseline_lr_multiplier=1.0
profile_shared_lr_multiplier=1.0
profile_invariant_lr_multiplier=1.0
profile_bias_lr_multiplier=1.0
profile_gate_lr_multiplier=1.0

case "$ABLATION" in
  full)
    ;;
  cf_off)
    profile_lambda_cf=0.0
    ;;
  treatment_off)
    profile_lambda_treat=0.0
    ;;
  treatment_shuffled)
    ;;
  orth_off)
    profile_lambda_orth=0.0
    ;;
  gate_fixed)
    profile_gate_lr_multiplier=0.0
    ;;
  invariant_only)
    profile_lambda_treat=0.0
    profile_lambda_orth=0.0
    profile_bias_branch_multiplier=0.0
    profile_bias_lr_multiplier=0.0
    profile_gate_lr_multiplier=0.0
    ;;
  bias_only)
    profile_lambda_cf=0.0
    profile_lambda_orth=0.0
    profile_invariant_branch_multiplier=0.0
    profile_invariant_lr_multiplier=0.0
    profile_gate_lr_multiplier=0.0
    ;;
  baseline)
    profile_lambda_cf=0.0
    profile_lambda_treat=0.0
    profile_lambda_orth=0.0
    profile_lambda_gate=0.0
    profile_lambda_residual_rank=0.0
    profile_lambda_residual_preserve=0.0
    profile_invariant_branch_multiplier=0.0
    profile_bias_branch_multiplier=0.0
    profile_shared_lr_multiplier=0.0
    profile_invariant_lr_multiplier=0.0
    profile_bias_lr_multiplier=0.0
    profile_gate_lr_multiplier=0.0
    ;;
  *)
    echo "Unknown ABLATION=$ABLATION" >&2
    echo "Use: full, cf_off, treatment_off, treatment_shuffled, orth_off, gate_fixed, invariant_only, bias_only, baseline" >&2
    exit 2
    ;;
esac

LAMBDA_CF="${LAMBDA_CF:-$profile_lambda_cf}"
LAMBDA_TREAT="${LAMBDA_TREAT:-$profile_lambda_treat}"
LAMBDA_ORTH="${LAMBDA_ORTH:-$profile_lambda_orth}"
LAMBDA_GATE="${LAMBDA_GATE:-$profile_lambda_gate}"
LAMBDA_GATE_VARIANCE="${LAMBDA_GATE_VARIANCE:-0.0}"
LAMBDA_INVARIANT_VARIANCE="${LAMBDA_INVARIANT_VARIANCE:-0.0}"
LAMBDA_BIAS_VARIANCE="${LAMBDA_BIAS_VARIANCE:-0.0}"
LAMBDA_BIAS_COUNTERFACTUAL="${LAMBDA_BIAS_COUNTERFACTUAL:-0.0}"
LAMBDA_BRANCH_BALANCE="${LAMBDA_BRANCH_BALANCE:-0.0}"
LAMBDA_RESIDUAL_RANK="${LAMBDA_RESIDUAL_RANK:-$profile_lambda_residual_rank}"
LAMBDA_INVARIANT_RANK="${LAMBDA_INVARIANT_RANK:-$profile_lambda_invariant_rank}"
LAMBDA_BIAS_RANK="${LAMBDA_BIAS_RANK:-$profile_lambda_bias_rank}"
LAMBDA_RESIDUAL_PRESERVE="${LAMBDA_RESIDUAL_PRESERVE:-$profile_lambda_residual_preserve}"

INVARIANT_BRANCH_MULTIPLIER="${INVARIANT_BRANCH_MULTIPLIER:-$profile_invariant_branch_multiplier}"
BIAS_BRANCH_MULTIPLIER="${BIAS_BRANCH_MULTIPLIER:-$profile_bias_branch_multiplier}"
BASELINE_LR_MULTIPLIER="${BASELINE_LR_MULTIPLIER:-$profile_baseline_lr_multiplier}"
SHARED_LR_MULTIPLIER="${SHARED_LR_MULTIPLIER:-$profile_shared_lr_multiplier}"
INVARIANT_LR_MULTIPLIER="${INVARIANT_LR_MULTIPLIER:-$profile_invariant_lr_multiplier}"
BIAS_LR_MULTIPLIER="${BIAS_LR_MULTIPLIER:-$profile_bias_lr_multiplier}"
GATE_LR_MULTIPLIER="${GATE_LR_MULTIPLIER:-$profile_gate_lr_multiplier}"

GATE_REGULARIZER="${GATE_REGULARIZER:-sample-kl}"
GATE_TARGET="${GATE_TARGET:-0.5}"
GATE_STD_FLOOR="${GATE_STD_FLOOR:-0.05}"
REPRESENTATION_STD_FLOOR="${REPRESENTATION_STD_FLOOR:-0.5}"
TREATMENT_LABEL_SMOOTHING="${TREATMENT_LABEL_SMOOTHING:-0.0}"
TREATMENT_LOGIT_TEMPERATURE="${TREATMENT_LOGIT_TEMPERATURE:-1.0}"
BIAS_COUNTERFACTUAL_MARGIN="${BIAS_COUNTERFACTUAL_MARGIN:-0.1}"
BRANCH_BALANCE_TARGET="${BRANCH_BALANCE_TARGET:-0.5}"
RESIDUAL_TIE_BAND="${RESIDUAL_TIE_BAND:-0.0}"

CENTER_RESIDUAL="${CENTER_RESIDUAL:-0}"
PRESERVE_BASE_TIES="${PRESERVE_BASE_TIES:-0}"
FLOOR_AT_BASE_MIN="${FLOOR_AT_BASE_MIN:-0}"
MAX_GAP_RATIO="${MAX_GAP_RATIO:-none}"
DELTA_LIMIT="${DELTA_LIMIT:-none}"
MAX_GRAD_NORM="${MAX_GRAD_NORM:-0.0}"
LR_SCHEDULER="${LR_SCHEDULER:-none}"
MIN_LR_RATIO="${MIN_LR_RATIO:-0.0}"
CHECKPOINT_INTERVAL="${CHECKPOINT_INTERVAL:-0}"
DIAGNOSTIC_INTERVAL="${DIAGNOSTIC_INTERVAL:-0}"

RUN_EVAL="${RUN_EVAL:-1}"
EVAL_SPLIT="${EVAL_SPLIT:-valid}"
EVAL_RESIDUAL_MULTIPLIER="${EVAL_RESIDUAL_MULTIPLIER:-0.75}"

if [[ ! "$RUN_TAG" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "RUN_TAG may contain only letters, digits, dot, underscore, and dash" >&2
  exit 2
fi

cd "$PROJECT_DIR"
[[ -f "$DATA_PATH/train2id.txt" ]] || { echo "Missing FB15k-237 data at $DATA_PATH" >&2; exit 2; }
[[ -f "$CF_CACHE" ]] || { echo "Missing CF cache at $CF_CACHE" >&2; exit 2; }

export CUDA_VISIBLE_DEVICES="$CUDA_DEVICE"
export OMP_NUM_THREADS="$OMP_THREADS"
export MKL_NUM_THREADS="$OMP_THREADS"

LOG_DIR="./logs/${RUN_TAG}"
mkdir -p "$LOG_DIR" ./checkpoint

train_args=(
  --dataset "$DATASET"
  --data-path "$DATA_PATH"
  --cf-cache "$CF_CACHE"
  --epochs "$EPOCHS"
  --lr "$LR"
  --optimizer "$OPTIMIZER"
  --margin "$MARGIN"
  --hash-hops 1
  --hidden-dim "$HIDDEN_DIM"
  --branch-dim "$BRANCH_DIM"
  --residual-scale "$RESIDUAL_SCALE"
  --branch-activation "$BRANCH_ACTIVATION"
  --branch-negative-slope "$BRANCH_NEGATIVE_SLOPE"
  --delta-temperature "$DELTA_TEMPERATURE"
  --delta-normalization "$DELTA_NORMALIZATION"
  --delta-target-rms "$DELTA_TARGET_RMS"
  --gate-temperature "$GATE_TEMPERATURE"
  --fusion-mode "$FUSION_MODE"
  --invariant-margin-fraction "$INVARIANT_MARGIN_FRACTION"
  --invariant-branch-multiplier "$INVARIANT_BRANCH_MULTIPLIER"
  --bias-branch-multiplier "$BIAS_BRANCH_MULTIPLIER"
  --treatment-aware
  --residual-hard-negatives "$RESIDUAL_HARD_NEGATIVES"
  --residual-near-tie-negatives "$RESIDUAL_NEAR_TIE_NEGATIVES"
  --residual-start-epoch "$RESIDUAL_START_EPOCH"
  --residual-warmup-epochs "$RESIDUAL_WARMUP_EPOCHS"
  --residual-loss-schedule "$RESIDUAL_LOSS_SCHEDULE"
  --treatment-supervision "$TREATMENT_SUPERVISION"
  --lambda-cf "$LAMBDA_CF"
  --lambda-treat "$LAMBDA_TREAT"
  --lambda-orth "$LAMBDA_ORTH"
  --lambda-gate "$LAMBDA_GATE"
  --gate-regularizer "$GATE_REGULARIZER"
  --gate-target "$GATE_TARGET"
  --lambda-gate-variance "$LAMBDA_GATE_VARIANCE"
  --gate-std-floor "$GATE_STD_FLOOR"
  --lambda-invariant-variance "$LAMBDA_INVARIANT_VARIANCE"
  --lambda-bias-variance "$LAMBDA_BIAS_VARIANCE"
  --representation-std-floor "$REPRESENTATION_STD_FLOOR"
  --treatment-label-smoothing "$TREATMENT_LABEL_SMOOTHING"
  --treatment-logit-temperature "$TREATMENT_LOGIT_TEMPERATURE"
  --lambda-bias-counterfactual "$LAMBDA_BIAS_COUNTERFACTUAL"
  --bias-counterfactual-margin "$BIAS_COUNTERFACTUAL_MARGIN"
  --lambda-branch-balance "$LAMBDA_BRANCH_BALANCE"
  --branch-balance-target "$BRANCH_BALANCE_TARGET"
  --lambda-residual-rank "$LAMBDA_RESIDUAL_RANK"
  --residual-tie-band "$RESIDUAL_TIE_BAND"
  --lambda-invariant-rank "$LAMBDA_INVARIANT_RANK"
  --lambda-bias-rank "$LAMBDA_BIAS_RANK"
  --lambda-residual-preserve "$LAMBDA_RESIDUAL_PRESERVE"
  --warmup-start-epoch "$WARMUP_START_EPOCH"
  --warmup-epochs "$WARMUP_EPOCHS"
  --nbatches "$NBATCHES"
  --neg-ent "$NEG_ENT"
  --threads "$THREADS"
  --seed "$SEED"
  --device "$DEVICE"
  --baseline-lr-multiplier "$BASELINE_LR_MULTIPLIER"
  --shared-lr-multiplier "$SHARED_LR_MULTIPLIER"
  --invariant-lr-multiplier "$INVARIANT_LR_MULTIPLIER"
  --bias-lr-multiplier "$BIAS_LR_MULTIPLIER"
  --gate-lr-multiplier "$GATE_LR_MULTIPLIER"
  --max-grad-norm "$MAX_GRAD_NORM"
  --lr-scheduler "$LR_SCHEDULER"
  --min-lr-ratio "$MIN_LR_RATIO"
  --checkpoint-dir ./checkpoint
  --checkpoint-interval "$CHECKPOINT_INTERVAL"
  --log-dir "$LOG_DIR"
  --architecture-diagnostic-interval "$DIAGNOSTIC_INTERVAL"
  --run-tag "$RUN_TAG"
  --eval-split "$EVAL_SPLIT"
  --eval-residual-multiplier "$EVAL_RESIDUAL_MULTIPLIER"
  --skip-diagnostics
  --skip-test
)

[[ "$CENTER_RESIDUAL" == "1" ]] && train_args+=(--center-residual)
[[ "$PRESERVE_BASE_TIES" == "1" ]] && train_args+=(--preserve-base-ties)
[[ "$FLOOR_AT_BASE_MIN" == "1" ]] && train_args+=(--floor-residual-at-baseline-min)
[[ "$DELTA_LIMIT" != "none" ]] && train_args+=(--delta-limit "$DELTA_LIMIT")
[[ "$MAX_GAP_RATIO" != "none" ]] && train_args+=(--residual-max-top-gap-ratio "$MAX_GAP_RATIO")
[[ "$SHUFFLE_TREATMENT_LABELS" == "1" ]] && train_args+=(--shuffle-treatment-labels)

printf 'TGIIF FB15k-237 run: tag=%s ablation=%s epochs=%s split=%s\n' \
  "$RUN_TAG" "$ABLATION" "$EPOCHS" "$EVAL_SPLIT" | tee "$LOG_DIR/run_manifest.log"
printf 'branch multipliers: invariant=%s bias=%s; losses: cf=%s treat=%s orth=%s residual_rank=%s\n' \
  "$INVARIANT_BRANCH_MULTIPLIER" "$BIAS_BRANCH_MULTIPLIER" \
  "$LAMBDA_CF" "$LAMBDA_TREAT" "$LAMBDA_ORTH" "$LAMBDA_RESIDUAL_RANK" \
  | tee -a "$LOG_DIR/run_manifest.log"
printf 'fusion: mode=%s invariant_margin_fraction=%s\n' \
  "$FUSION_MODE" "$INVARIANT_MARGIN_FRACTION" | tee -a "$LOG_DIR/run_manifest.log"

"$PYTHON_BIN" examples/train_tgiif_from_scratch_FB15k237.py "${train_args[@]}" \
  2>&1 | tee "$LOG_DIR/train.log"

checkpoint_pattern="ctsnet_${RUN_TAG}_FB15k-237_h1_*_${OPTIMIZER}_ep${EPOCHS}.ckpt"
FINAL_CHECKPOINT="$(find ./checkpoint -maxdepth 1 -type f -name "$checkpoint_pattern" -printf '%T@ %p\n' | sort -nr | awk 'NR == 1 {print $2}')"
[[ -n "$FINAL_CHECKPOINT" && -f "$FINAL_CHECKPOINT" ]] || {
  echo "Could not locate final checkpoint matching $checkpoint_pattern" >&2
  exit 3
}
printf 'checkpoint=%s\n' "$FINAL_CHECKPOINT" | tee -a "$LOG_DIR/run_manifest.log"

if [[ "$RUN_EVAL" == "1" ]]; then
  eval_args=(
    --architecture treatment-aware
    --data-path "$DATA_PATH"
    --split "$EVAL_SPLIT"
    --hash-hops 1
    --feature-mode triple
    --hidden-dim "$HIDDEN_DIM"
    --branch-dim "$BRANCH_DIM"
    --branch-activation "$BRANCH_ACTIVATION"
    --branch-negative-slope "$BRANCH_NEGATIVE_SLOPE"
    --delta-temperature "$DELTA_TEMPERATURE"
    --delta-normalization "$DELTA_NORMALIZATION"
    --delta-target-rms "$DELTA_TARGET_RMS"
    --gate-temperature "$GATE_TEMPERATURE"
    --fusion-mode "$FUSION_MODE"
    --invariant-margin-fraction "$INVARIANT_MARGIN_FRACTION"
    --invariant-branch-multiplier "$INVARIANT_BRANCH_MULTIPLIER"
    --bias-branch-multiplier "$BIAS_BRANCH_MULTIPLIER"
    --threads "$THREADS"
    --device "$DEVICE"
    --skip-diagnostics
    --filtered-ranking
  )
  [[ "$DELTA_LIMIT" != "none" ]] && eval_args+=(--delta-limit "$DELTA_LIMIT")
  [[ "$FLOOR_AT_BASE_MIN" == "1" ]] && eval_args+=(--floor-residual-at-baseline-min)
  [[ "$MAX_GAP_RATIO" != "none" ]] && eval_args+=(--residual-max-top-gap-ratio "$MAX_GAP_RATIO")

  "$PYTHON_BIN" examples/evaluate_tgiif_checkpoint_FB15k237.py \
    "$FINAL_CHECKPOINT" "${eval_args[@]}" --residual-multiplier 0 \
    2>&1 | tee "$LOG_DIR/eval_${EVAL_SPLIT}_m0.log"
  "$PYTHON_BIN" examples/evaluate_tgiif_checkpoint_FB15k237.py \
    "$FINAL_CHECKPOINT" "${eval_args[@]}" \
    --residual-multiplier "$EVAL_RESIDUAL_MULTIPLIER" \
    2>&1 | tee "$LOG_DIR/eval_${EVAL_SPLIT}_m${EVAL_RESIDUAL_MULTIPLIER}.log"
fi

echo "Completed V5c run: $RUN_TAG"
