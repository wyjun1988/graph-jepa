#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps-max/bin/python}"
MODEL_NAME="broad_transition_jepa_v5_systemic_seed43_20260714"
MODEL_ROOT="models/$MODEL_NAME"
CACHE_ROOT="reports/cached_raw_market_transition_head_v6_seed43_20260714"
RUN_NAME="cached_raw_market_transition_head_v6_seed43_head7301_20260714"
RUN_ROOT="reports/$RUN_NAME"
TARGET_ROOT="reports/market_transition_target_audit_v6_systemic_impact_metric_20260714"
LOG="ops/training/${RUN_NAME}_m1max.log"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
export PYTORCH_ENABLE_MPS_FALLBACK=1

printf '%s\n' \
  '{"scope":"posthoc_head_seed_stability_only","encoder_seed":43,"head_seed":7301,"frozen_latent_cache":true,"test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

run_fold() {
  local fold="$1"
  local model="$2"
  local output="$RUN_ROOT/$fold"
  local cache="$CACHE_ROOT/$fold/frozen_raw_transition_pool.npz"
  if [[ ! -f "$output/summary.json" ]]; then
    "$PYTHON_BIN" scripts/benchmark_cached_pooled_market_transition_head.py \
      --model-dir "$MODEL_ROOT/$model" \
      --output-dir "$output" \
      --pooled-cache "$cache" \
      --pooling-mode raw \
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
      --cache-batch-size 4 \
      --batch-size 64 \
      --eval-batch-size 256 \
      --edge-cache-workers 8 \
      --device mps \
      --seed 7301 \
      --cache-dir "$OHLCV" \
      --external-cache-dir data/external_cache \
      2>&1 | tee -a "$LOG"
  fi
  if [[ ! -f "$output/major_trajectory/summary.json" ]]; then
    "$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
      --target-audit-root "$TARGET_ROOT/$fold" \
      --prediction-root "$output" \
      --output-dir "$output/major_trajectory" \
      --major-event-quantile 0.90 \
      2>&1 | tee -a "$LOG"
  fi
}

run_fold fold1 "${MODEL_NAME}_fold1_20231229_to_20241230"
run_fold fold2 "${MODEL_NAME}_fold2_20241230_to_20260710"

touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "$RUN_NAME complete" | tee -a "$LOG"
