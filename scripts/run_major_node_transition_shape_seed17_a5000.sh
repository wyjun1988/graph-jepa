#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON_BIN="${PYTHON:-/workspace/venvs/stock-v2-cu128/bin/python}"
RUN_ROOT="reports/major_node_transition_shape_seed17_v3_20260714"
LOG="ops/training/major_node_transition_shape_seed17_a5000_20260714.log"
MODEL_ROOT="models/walk_forward_causal453_path_v2_20260713"
TARGET_ROOT="reports/market_transition_target_audit_v3_20260714"
CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
printf '%s\n' \
  '{"scope":"posthoc_research_only","target_version":"market_transition_v3_breadth_20260714","objective":"full 453-stock node-state shape on major trajectories","baseline":"same decoder without latent rollout","test_used_for_selection":false,"live_orders_allowed":false}' \
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
    --batch-size 16 \
    --edge-cache-workers 16 \
    --device cuda \
    --cache-dir "$CACHE" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

run_fold \
  fold1 \
  strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold1_20231229_to_20241230
run_fold \
  fold2 \
  strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold2_20241230_to_20260710

touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "seed17 major node-transition shape evaluation complete" | tee -a "$LOG"
