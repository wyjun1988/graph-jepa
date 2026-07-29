#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps-max/bin/python}"
RUN_ROOT="reports/major_node_transition_shape_v5_impact_mass_seed43_20260714"
LOG="ops/training/major_node_transition_shape_v5_impact_mass_seed43_20260714_m1max.log"
MODEL_NAME="broad_transition_jepa_v5_systemic_seed43_20260714"
MODEL_ROOT="models/$MODEL_NAME"
TARGET_ROOT="reports/market_transition_target_audit_v5_systemic_impact_metric_20260714"
CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
export PYTORCH_ENABLE_MPS_FALLBACK=1
printf '%s\n' \
  '{"scope":"posthoc_research_only","target_version":"market_transition_v5_systemic_impact_20260714","impact_metric_version":"market_transition_systemic_impact_mass_v1_20260714","objective":"full 453-stock node-state shape on major systemic trajectories","baseline":"same decoder without latent rollout","test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

run_fold() {
  local fold="$1"
  local model="$2"
  local output="$RUN_ROOT/$fold"
  if [[ -f "$output/summary.json" ]]; then
    printf 'skip completed %s\n' "$fold" | tee -a "$LOG"
    return
  fi
  "$PYTHON_BIN" scripts/evaluate_major_node_transition_shape.py \
    --model-dir "$MODEL_ROOT/$model" \
    --target-audit-root "$TARGET_ROOT/$fold" \
    --output-dir "$output" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --batch-size 4 \
    --edge-cache-workers 8 \
    --device mps \
    --cache-dir "$CACHE" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

run_fold \
  fold1 \
  "${MODEL_NAME}_fold1_20231229_to_20241230"
run_fold \
  fold2 \
  "${MODEL_NAME}_fold2_20241230_to_20260710"

touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "v5 seed43 major node-transition shape evaluation complete" | tee -a "$LOG"
