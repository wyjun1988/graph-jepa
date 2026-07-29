#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
MODEL="models/milestones/broad_transition_v5_seed43_fold1_epoch008"
FIRST="reports/cached_pooled_market_transition_head_v5_epoch008_seed2701_20260714"
SECOND="reports/cached_pooled_market_transition_head_v5_epoch008_seed4301_20260714"
CACHE="$FIRST/frozen_transition_pool.npz"
TARGET="reports/market_transition_target_audit_v5_systemic_impact_metric_20260714/fold1"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
cd "$ROOT"
export PYTORCH_ENABLE_MPS_FALLBACK=1

until [[ -f "$FIRST/summary.json" ]]; do
  sleep 30
done

if [[ ! -f "$SECOND/summary.json" ]]; then
  "$PYTHON_BIN" scripts/benchmark_cached_pooled_market_transition_head.py \
    --model-dir "$MODEL" \
    --output-dir "$SECOND" \
    --pooled-cache "$CACHE" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --epochs 80 \
    --patience 10 \
    --hidden-dim 256 \
    --layers 2 \
    --heads 8 \
    --dropout 0.10 \
    --learning-rate 0.0003 \
    --weight-decay 0.001 \
    --batch-size 128 \
    --eval-batch-size 512 \
    --device mps \
    --seed 4301 \
    --cache-dir "$OHLCV" \
    --external-cache-dir data/external_cache
fi

for run in "$FIRST" "$SECOND"; do
  output="$run/major_trajectory"
  if [[ -f "$output/summary.json" ]]; then
    continue
  fi
  "$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
    --target-audit-root "$TARGET" \
    --prediction-root "$run" \
    --output-dir "$output" \
    --major-event-quantile 0.90
done
