#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
DEVICE="${DEVICE:-cuda}"
FEATURE_WORKERS="${FEATURE_WORKERS:-16}"
FOLD_ID="${1:?fold id is required}"
TRAIN_END="${2:?train end is required}"
PANEL_END="${3:?panel end is required}"
TEST_END="${4:?test end is required}"
RUN_NAME="final_aligned_jepa_market_head_historical_v1_seed17"
FOLD_NAME="${RUN_NAME}_${FOLD_ID}_${TRAIN_END//-/}_to_${PANEL_END//-/}"

cd "$ROOT"
export PYTHONUNBUFFERED=1

"$PYTHON_BIN" scripts/evaluate_auxiliary_trading_policy.py \
  --model-dir "models/${RUN_NAME}/${FOLD_NAME}" \
  --output-dir "reports/${RUN_NAME}_h2_cash_gate_data" \
  --prediction-cache-dir "artifacts/prediction_caches/${RUN_NAME}_${FOLD_ID}" \
  --fold "$FOLD_ID" \
  --test-end "$TEST_END" \
  --policy-horizon 2 \
  --top-k 10 \
  --liquidity-top-n 300 \
  --cost-bps 30 \
  --stress-cost-bps 50 \
  --device "$DEVICE" \
  --feature-workers "$FEATURE_WORKERS" \
  --skip-direct-probes \
  --save-cash-gate-dataset
