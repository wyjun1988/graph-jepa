#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON_BIN="${PYTHON:-/workspace/venvs/stock-v2-cu128/bin/python}"
MODEL_NAME="${MODEL_NAME:-broad_transition_jepa_v4_robust_seed17_20260714}"
TARGET_VERSION="${TARGET_VERSION:-market_transition_v4_robust_breadth_20260714}"
IMPACT_METRIC_VERSION="${IMPACT_METRIC_VERSION:-legacy_raw_family_energy}"
DEVICE="${DEVICE:-cuda}"
EDGE_CACHE_WORKERS="${EDGE_CACHE_WORKERS:-16}"
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-64}"
HEAD_EVAL_BATCH_SIZE="${HEAD_EVAL_BATCH_SIZE:-128}"
MODEL_ROOT="models/$MODEL_NAME"
RUN_NAME="${RUN_NAME:-market_transition_head_jepa_v4_20260714}"
RUN_ROOT="reports/$RUN_NAME"
LOG="ops/training/${RUN_NAME}_a5000.log"
CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
TARGET_AUDIT_ROOT="${TARGET_AUDIT_ROOT:-reports/market_transition_target_audit_v4_robust_20260714}"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
printf \
  '{"scope":"posthoc_research_only","target_version":"%s","impact_metric_version":"%s","objective":"joint-horizon robust broad market transition prediction","family_event_quantile":0.95,"test_used_for_selection":false,"live_orders_allowed":false}\n' \
  "$TARGET_VERSION" \
  "$IMPACT_METRIC_VERSION" \
  > "$RUN_ROOT/experiment_contract.json"

run_fold() {
  local fold="$1"
  local model="$2"
  local seed="$3"
  local output="$RUN_ROOT/$fold"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/benchmark_market_transition_head.py \
    --model-dir "$MODEL_ROOT/$model" \
    --output-dir "$output" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --epochs 20 \
    --patience 5 \
    --projection-dim 128 \
    --hidden-dim 256 \
    --layers 2 \
    --heads 8 \
    --dropout 0.10 \
    --learning-rate 0.0003 \
    --weight-decay 0.001 \
    --batch-size "$HEAD_BATCH_SIZE" \
    --eval-batch-size "$HEAD_EVAL_BATCH_SIZE" \
    --edge-cache-workers "$EDGE_CACHE_WORKERS" \
    --device "$DEVICE" \
    --seed "$seed" \
    --cache-dir "$CACHE" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

run_fold \
  fold1 \
  "${MODEL_NAME}_fold1_20231229_to_20241230" \
  2701
run_fold \
  fold2 \
  "${MODEL_NAME}_fold2_20241230_to_20260710" \
  2702

run_major_trajectory() {
  local fold="$1"
  local prediction_root="$RUN_ROOT/$fold"
  local output="$prediction_root/major_trajectory"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
    --target-audit-root "$TARGET_AUDIT_ROOT/$fold" \
    --prediction-root "$prediction_root" \
    --output-dir "$output" \
    --major-event-quantile 0.90 \
    2>&1 | tee -a "$LOG"
}

run_major_trajectory fold1
run_major_trajectory fold2

touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "$RUN_NAME experiment complete" | tee -a "$LOG"
