#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
MODEL_ROOT="models/walk_forward_causal453_path_v2_20260713"
TARGET_ROOT="reports/market_transition_target_audit_v6_systemic_impact_metric_20260714"
OUTPUT_ROOT="reports/market_transition_v6_state_feature_contribution_audit_20260714"
WAIT_MARKER="reports/classical_extra_trees_market_transition_v6_history_lags125_two_seed_20260714_COMPLETE"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
cd "$ROOT"

until [[ -f "$WAIT_MARKER" ]]; do
  sleep 15
done

run_fold() {
  local fold="$1"
  local model="$2"
  local output="$OUTPUT_ROOT/$fold"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/audit_market_transition_state_feature_contributions.py \
    --model-dir "$MODEL_ROOT/$model" \
    --target-audit-root "$TARGET_ROOT/$fold" \
    --output-dir "$output" \
    --major-event-quantile 0.90 \
    --cache-dir "$OHLCV" \
    --external-cache-dir data/external_cache
}

run_fold \
  fold1 \
  strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold1_20231229_to_20241230
run_fold \
  fold2 \
  strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold2_20241230_to_20260710

touch "$OUTPUT_ROOT/EXPERIMENT_COMPLETE"
