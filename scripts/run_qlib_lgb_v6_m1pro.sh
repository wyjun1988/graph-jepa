#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/work/stock-v2}"
FEATURE_PYTHON="${FEATURE_PYTHON:-$ROOT/.venv-mps/bin/python}"
QLIB_PYTHON="${QLIB_PYTHON:-$ROOT/.venv-qlib/bin/python}"
RUN_NAME="broad_transition_jepa_v6_systemic_seed17_rtx4000ada_20260714"
OHLCV="$ROOT/data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
OUTPUT_ROOT="$ROOT/reports/qlib_lgb_${RUN_NAME}_20260714"
LOG="$ROOT/logs/qlib_lgb_${RUN_NAME}_m1pro.log"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG")"

run_fold() {
  local fold="$1"
  local model_name="$2"
  local fold_root="$OUTPUT_ROOT/$fold"
  local bundle="$fold_root/pit_bundle"
  local result="$fold_root/result"
  mkdir -p "$fold_root"
  if [[ ! -f "$bundle/bundle_contract.json" ]]; then
    "$FEATURE_PYTHON" scripts/export_qlib_pit_bundle.py \
      --model-dir "$ROOT/models/$RUN_NAME/$model_name" \
      --output-dir "$bundle" \
      --horizons 1,2,3,5,10 \
      --validation-days 126 \
      --feature-workers 10 \
      --cache-dir "$OHLCV" \
      --external-cache-dir "$ROOT/data/external_cache"
  fi
  if [[ ! -f "$result/EXPERIMENT_COMPLETE" ]]; then
    "$QLIB_PYTHON" scripts/benchmark_qlib_lgb.py \
      --bundle-dir "$bundle" \
      --output-dir "$result" \
      --horizons 1,2,3,5,10 \
      --num-boost-round 500 \
      --early-stopping-rounds 50 \
      --num-threads 10 \
      --liquidity-top-k 300 \
      --seed 17
  fi
}

{
  run_fold \
    fold1 \
    "${RUN_NAME}_fold1_20231229_to_20241230"
  run_fold \
    fold2 \
    "${RUN_NAME}_fold2_20241230_to_20260710"
  touch "$OUTPUT_ROOT/EXPERIMENT_COMPLETE"
} 2>&1 | tee -a "$LOG"
