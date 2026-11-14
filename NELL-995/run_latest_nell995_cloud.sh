#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ABLATION="${ABLATION:-treatment_off}"
SEED="${SEED:-23}"
ADAPTER_EPOCHS="${ADAPTER_EPOCHS:-500}"
RESIDUAL_START_EPOCH="${RESIDUAL_START_EPOCH:-100}"
RESIDUAL_WARMUP_EPOCHS="${RESIDUAL_WARMUP_EPOCHS:-150}"
EVAL_SPLIT="${EVAL_SPLIT:-valid}"
RUN_TEST="${RUN_TEST:-0}"
M="${M:-1.0}"
INVARIANT_MARGIN_FRACTION="${INVARIANT_MARGIN_FRACTION:-0.05}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-./checkpoint/ctsnet_tgiif_nell995_full_ep4000_cloud_s23_NELL-995_h1_0.005_Adam_ep4000.ckpt}"
RUN_TAG="${RUN_TAG:-tgiif_nell995_latest_${ABLATION}_adapter${ADAPTER_EPOCHS}_s${SEED}}"
HYBRID_CHECKPOINT="${HYBRID_CHECKPOINT:-./checkpoint/${RUN_TAG}_hybrid_base4000.ckpt}"

case "$ABLATION" in
  full|treatment_off|cf_off|treatment_shuffled) ;;
  *)
    echo "ABLATION must be full, treatment_off, cf_off, or treatment_shuffled" >&2
    exit 2
    ;;
esac

cd "$PROJECT_DIR"
[[ -f "$BASE_CHECKPOINT" ]] || { echo "Missing base checkpoint: $BASE_CHECKPOINT" >&2; exit 2; }
[[ -f ./benchmarks/NELL-995/train2id.txt ]] || { echo "Missing NELL-995 dataset" >&2; exit 2; }
[[ -f ./counterfactual_cache/NELL-995_popularity_triple_h1.pt ]] || { echo "Missing CF cache" >&2; exit 2; }

# Train only the adapter source.  The mature scorer is composed afterwards,
# which exactly matches the validated stagewise protocol.
PROJECT_DIR="$PROJECT_DIR" \
PYTHON_BIN="$PYTHON_BIN" \
ABLATION="$ABLATION" \
EPOCHS="$ADAPTER_EPOCHS" \
SEED="$SEED" \
RUN_TAG="$RUN_TAG" \
NELL_SAFE_MODE=1 \
TREATMENT_SHARED_GRADIENT_SCALE=0.01 \
CF_DISTANCE_TEMPERATURE=0.05 \
RESIDUAL_START_EPOCH="$RESIDUAL_START_EPOCH" \
RESIDUAL_WARMUP_EPOCHS="$RESIDUAL_WARMUP_EPOCHS" \
RUN_EVAL=0 \
bash ./run_tgiif_nell995.sh

adapter_pattern="./checkpoint/ctsnet_${RUN_TAG}_NELL-995_h1_*_Adam_ep${ADAPTER_EPOCHS}.ckpt"
ADAPTER_CHECKPOINT="$(compgen -G "$adapter_pattern" | head -n 1 || true)"
[[ -n "$ADAPTER_CHECKPOINT" && -f "$ADAPTER_CHECKPOINT" ]] || {
  echo "Could not locate adapter checkpoint: $adapter_pattern" >&2
  exit 3
}

"$PYTHON_BIN" examples/compose_tgiif_nell995_hybrid.py \
  --official "$BASE_CHECKPOINT" \
  --adapter "$ADAPTER_CHECKPOINT" \
  --output "$HYBRID_CHECKPOINT"

EVAL_DIR="./logs/${RUN_TAG}/hybrid_eval"
mkdir -p "$EVAL_DIR"
for eval_m in 0 "$M"; do
  PROJECT_DIR="$PROJECT_DIR" PYTHON_BIN="$PYTHON_BIN" \
  CHECKPOINT="$HYBRID_CHECKPOINT" SPLIT="$EVAL_SPLIT" M="$eval_m" \
  INVARIANT_MARGIN_FRACTION="$INVARIANT_MARGIN_FRACTION" \
  INVARIANT_MARGIN_UNGATED=1 \
  OUTPUT="$EVAL_DIR/eval_${EVAL_SPLIT}_m${eval_m}.log" \
  bash ./eval_tgiif_nell995.sh
done

if [[ "$RUN_TEST" == "1" ]]; then
  PROJECT_DIR="$PROJECT_DIR" PYTHON_BIN="$PYTHON_BIN" \
  CHECKPOINT="$HYBRID_CHECKPOINT" SPLIT=test M="$M" \
  INVARIANT_MARGIN_FRACTION="$INVARIANT_MARGIN_FRACTION" \
  INVARIANT_MARGIN_UNGATED=1 \
  OUTPUT="$EVAL_DIR/eval_test_m${M}.log" \
  bash ./eval_tgiif_nell995.sh
fi

echo "Completed latest NELL-995 stagewise run"
echo "adapter=$ADAPTER_CHECKPOINT"
echo "hybrid=$HYBRID_CHECKPOINT"
echo "evaluation=$EVAL_DIR"
