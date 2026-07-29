#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 MODEL_DIR OUTPUT_DIR TARGET_AUDIT_ROOT STOCK_QUANTILES_ROOT" >&2
  exit 2
fi

MODEL_DIR=$1
OUTPUT_DIR=$2
TARGET_AUDIT_ROOT=$3
STOCK_QUANTILES_ROOT=$4
CONTRACT=configs/family-query-trajectory-diagnostic-v2-20260716.json
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ -e "$OUTPUT_DIR/summary.json" || -e "$OUTPUT_DIR/DIAGNOSTIC_COMPLETE" ]]; then
  echo "refusing to overwrite an existing family-query diagnostic: $OUTPUT_DIR" >&2
  exit 3
fi

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" scripts/benchmark_market_transition_head.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --horizons 1,2,3,5,10 \
  --validation-days 126 \
  --epochs 80 \
  --patience 10 \
  --projection-dim 128 \
  --family-query-pooling \
  --stock-quantile-pooling \
  --hidden-dim 256 \
  --layers 2 \
  --heads 8 \
  --dropout 0.10 \
  --learning-rate 0.0003 \
  --weight-decay 0.001 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --edge-cache-workers 16 \
  --device cuda \
  --seed 2701 \
  --cache-dir data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv \
  --external-cache-dir data/external_cache

"$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
  --target-audit-root "$TARGET_AUDIT_ROOT" \
  --prediction-root "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR/major_trajectory" \
  --major-event-quantile 0.90

"$PYTHON_BIN" scripts/audit_family_query_trajectory.py \
  --contract "$CONTRACT" \
  --candidate-root "$OUTPUT_DIR" \
  --stock-quantiles-root "$STOCK_QUANTILES_ROOT" \
  --source-root . \
  --output-dir "$OUTPUT_DIR/contract_audit"

touch "$OUTPUT_DIR/DIAGNOSTIC_COMPLETE"
