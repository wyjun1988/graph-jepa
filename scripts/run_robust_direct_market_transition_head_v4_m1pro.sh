#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
RUN_NAME="${RUN_NAME:-robust_direct_market_transition_head_v4_20260714}"
TARGET_VERSION="${TARGET_VERSION:-market_transition_v4_robust_breadth_20260714}"
IMPACT_METRIC_VERSION="${IMPACT_METRIC_VERSION:-legacy_raw_family_energy}"
HEAD_SEED="${HEAD_SEED:-2701}"
RUN_ROOT="reports/$RUN_NAME"
LOG="ops/training/${RUN_NAME}_m1pro.log"
MODEL_ROOT="models/walk_forward_causal453_path_v2_20260713"
CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
printf \
  '{"scope":"posthoc_research_only","target_version":"%s","impact_metric_version":"%s","role":"same-objective robust causal direct comparator","causal_stock_statistics":["mean","std","q25","median","q75","availability"],"test_used_for_selection":false,"live_orders_allowed":false}\n' \
  "$TARGET_VERSION" \
  "$IMPACT_METRIC_VERSION" \
  > "$RUN_ROOT/experiment_contract.json"

run_fold() {
  local fold="$1"
  local model="$2"
  local output="$RUN_ROOT/$fold"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/benchmark_direct_market_transition_head.py \
    --model-dir "$MODEL_ROOT/$model" \
    --output-dir "$output" \
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
    --batch-size 128 \
    --eval-batch-size 512 \
    --device mps \
    --seed "$HEAD_SEED" \
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
printf '%s\n' "$RUN_NAME experiment complete" | tee -a "$LOG"
