#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
RUN_ROOT="reports/market_transition_target_audit_v6_systemic_impact_metric_20260714"
LOG="ops/training/market_transition_target_audit_v6_systemic_impact_metric_20260714_m1pro.log"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"

printf '%s\n' \
  '{"scope":"target_audit_only","target_version":"market_transition_v6_systemic_impact_20260714","impact_metric_version":"market_transition_systemic_impact_mass_v2_20260714","topology_change_direction":"increase_only","test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

run_fold() {
  local fold="$1"
  local model="$2"
  local output="$RUN_ROOT/$fold"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/audit_market_transition_targets.py \
    --model-dir "$model" \
    --output-dir "$output" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --component-scale-quantile 0.90 \
    --family-event-quantile 0.95 \
    --top-events 30 \
    --cache-dir "$OHLCV" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

run_fold \
  fold1 \
  models/walk_forward_causal453_path_v2_20260713/strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold1_20231229_to_20241230
run_fold \
  fold2 \
  models/walk_forward_causal453_path_v2_20260713/strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold2_20241230_to_20260710

touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "market transition v6 target audit complete" | tee -a "$LOG"
