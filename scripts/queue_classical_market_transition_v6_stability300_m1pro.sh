#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
MODEL_ROOT="models/walk_forward_causal453_path_v2_20260713"
TARGET_ROOT="reports/market_transition_target_audit_v6_systemic_impact_metric_20260714"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
WAIT_MARKER="reports/classical_extra_trees_market_transition_v6_by_event_history_lags125_event_tuned_v1_two_seed_20260714_COMPLETE"
RUN_PREFIX="classical_extra_trees_market_transition_v6_by_event_history_lags125_stability300"
cd "$ROOT"

until [[ -f "$WAIT_MARKER" ]]; do
  sleep 15
done

run_fold() {
  local seed="$1"
  local fold="$2"
  local model="$3"
  local output="reports/${RUN_PREFIX}_seed${seed}_20260714/$fold"
  if [[ ! -f "$output/summary.json" ]]; then
    "$PYTHON_BIN" scripts/benchmark_classical_market_transition_head.py \
      --model-dir "$MODEL_ROOT/$model" \
      --output-dir "$output" \
      --horizons 1,2,3,5,10 \
      --event-classifier-mode by_event \
      --event-model-selection joint_bundle \
      --transition-history-lags 1,2,5 \
      --validation-days 126 \
      --estimators 300 \
      --jobs 6 \
      --seed "$seed" \
      --cache-dir "$OHLCV" \
      --external-cache-dir data/external_cache
  fi
  if [[ ! -f "$output/major_trajectory/summary.json" ]]; then
    "$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
      --target-audit-root "$TARGET_ROOT/$fold" \
      --prediction-root "$output" \
      --output-dir "$output/major_trajectory" \
      --major-event-quantile 0.90
  fi
}

for seed in 2701 4301 7301; do
  run_fold \
    "$seed" fold1 \
    strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold1_20231229_to_20241230
  run_fold \
    "$seed" fold2 \
    strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold2_20241230_to_20260710
done

"$PYTHON_BIN" scripts/aggregate_market_transition_stability.py \
  --run-prefix "reports/${RUN_PREFIX}" \
  --seeds 2701,4301,7301 \
  --output-dir "reports/${RUN_PREFIX}_aggregate_20260714"

touch "reports/${RUN_PREFIX}_three_seed_20260714_COMPLETE"
