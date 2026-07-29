#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON_BIN="$ROOT/.venv-mps/bin/python"
RUN_ROOT="reports/systemic_transition_target_audit_20260714"
LOG_PATH="ops/training/systemic_transition_target_audit_m1pro_20260714.log"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG_PATH")"

on_error() {
  local status=$?
  printf '{"status":"failed","exit_status":%d,"live_orders_allowed":false}\n' \
    "$status" > "$RUN_ROOT/PIPELINE_FAILED"
  exit "$status"
}
trap on_error ERR

printf '%s\n' \
  '{"scope":"posthoc_research_only","target":"broad market systemic transition rather than individual absolute return","event_quantile":0.90,"selection_status":"requires_future_confirmation","live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

COMMON_ARGS=(
  --horizons 1,2,3,5,10
  --validation-days 126
  --event-quantile 0.90
  --top-events 30
  --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv
  --external-cache-dir data/external_cache
)

if [[ ! -s "$RUN_ROOT/fold1/summary.json" ]]; then
  "$PYTHON_BIN" scripts/audit_systemic_transition_targets.py \
    --model-dir models/walk_forward_causal453_path_v2_20260713/strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold1_20231229_to_20241230 \
    --output-dir "$RUN_ROOT/fold1" \
    "${COMMON_ARGS[@]}"
fi

if [[ ! -s "$RUN_ROOT/fold2/summary.json" ]]; then
  "$PYTHON_BIN" scripts/audit_systemic_transition_targets.py \
    --model-dir models/walk_forward_causal453_path_v2_20260713/strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold2_20241230_to_20260710 \
    --output-dir "$RUN_ROOT/fold2" \
    "${COMMON_ARGS[@]}"
fi

rm -f "$RUN_ROOT/PIPELINE_FAILED"
touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "systemic transition target audit complete"
