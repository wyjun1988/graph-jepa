#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps-max/bin/python}"
MODEL_NAME="broad_transition_jepa_v5_systemic_seed43_20260714"
MODEL_ROOT="models/$MODEL_NAME"
RUN_NAME="cached_raw_market_transition_head_v6_seed43_20260714"
RUN_ROOT="reports/$RUN_NAME"
TARGET_ROOT="reports/market_transition_target_audit_v6_systemic_impact_metric_20260714"
LOG="ops/training/${RUN_NAME}_m1max.log"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
export PYTORCH_ENABLE_MPS_FALLBACK=1

actual_target="$($PYTHON_BIN -c 'from stock_v2.market_transition import MARKET_TRANSITION_TARGET_VERSION; print(MARKET_TRANSITION_TARGET_VERSION)')"
if [[ "$actual_target" != "market_transition_v6_systemic_impact_20260714" ]]; then
  printf 'refusing posthoc run with target %s\n' "$actual_target" >&2
  exit 2
fi

printf '%s\n' \
  '{"scope":"posthoc_research_only","target_version":"market_transition_v6_systemic_impact_20260714","impact_metric_version":"market_transition_systemic_impact_mass_v2_20260714","representation":"raw_context_and_latent_delta_robust_pool","impact_weighted_event_loss":true,"head_seeds":[2701,4301],"test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

run_head() {
  local fold="$1"
  local model="$2"
  local seed="$3"
  local output="$RUN_ROOT/${fold}_seed${seed}"
  local cache="$RUN_ROOT/$fold/frozen_raw_transition_pool.npz"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
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
    --seed "$seed" \
    --cache-dir "$OHLCV" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

run_major() {
  local fold="$1"
  local seed="$2"
  local prediction="$RUN_ROOT/${fold}_seed${seed}"
  local output="$prediction/major_trajectory"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
    --target-audit-root "$TARGET_ROOT/$fold" \
    --prediction-root "$prediction" \
    --output-dir "$output" \
    --major-event-quantile 0.90 \
    2>&1 | tee -a "$LOG"
}

run_fold() {
  local fold="$1"
  local model="$2"
  mkdir -p "$RUN_ROOT/$fold"
  run_head "$fold" "$model" 2701
  run_head "$fold" "$model" 4301
  run_major "$fold" 2701
  run_major "$fold" 4301
}

run_fold fold1 "${MODEL_NAME}_fold1_20231229_to_20241230"
run_fold fold2 "${MODEL_NAME}_fold2_20241230_to_20260710"

touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "$RUN_NAME complete" | tee -a "$LOG"
