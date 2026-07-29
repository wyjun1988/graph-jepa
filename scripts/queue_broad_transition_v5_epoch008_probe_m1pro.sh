#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
MODEL="models/milestones/broad_transition_v5_seed43_fold1_epoch008"
RUN_ROOT="reports/broad_transition_v5_seed43_fold1_epoch008_probe_20260714"
LOG="ops/training/broad_transition_v5_seed43_fold1_epoch008_probe_20260714_m1pro.log"
CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
cd "$ROOT"

until [[ -f reports/robust_direct_market_transition_head_v5_impact_mass_seed4301_20260714/EXPERIMENT_COMPLETE ]]; do
  sleep 30
done

mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
printf '%s\n' \
  '{"scope":"milestone_failure_probe_only","checkpoint_epoch":8,"test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

if [[ ! -f "$RUN_ROOT/trained_transition/summary.json" ]]; then
  "$PYTHON_BIN" scripts/evaluate_trained_market_transition_auxiliary.py \
    --model-dir "$MODEL" \
    --output-dir "$RUN_ROOT/trained_transition" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --batch-size 2 \
    --max-test-steps 30 \
    --device mps \
    --cache-dir "$CACHE" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
fi

if [[ ! -f "$RUN_ROOT/major_node_shape/summary.json" ]]; then
  "$PYTHON_BIN" scripts/evaluate_major_node_transition_shape.py \
    --model-dir "$MODEL" \
    --target-audit-root reports/market_transition_target_audit_v5_systemic_impact_metric_20260714/fold1 \
    --output-dir "$RUN_ROOT/major_node_shape" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --batch-size 1 \
    --edge-cache-workers 4 \
    --max-test-steps 20 \
    --device mps \
    --cache-dir "$CACHE" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
fi

touch "$RUN_ROOT/PROBE_COMPLETE"
