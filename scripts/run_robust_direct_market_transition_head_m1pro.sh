#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
RUN_ROOT="reports/robust_direct_market_transition_head_seed17_v3_20260714"
LOG="ops/training/robust_direct_market_transition_head_seed17_m1pro_20260714.log"
MODEL_ROOT="models/walk_forward_causal453_path_v2_20260713"
CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
printf '%s\n' \
  '{"scope":"posthoc_research_only","target_version":"market_transition_v3_breadth_20260714","role":"strengthened same-objective direct comparator","causal_stock_statistics":["mean","std","q25","median","q75","availability"],"test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

run_fold() {
  local fold="$1"
  local model="$2"
  local output="$RUN_ROOT/$fold"
  if [[ -f "$output/summary.json" ]]; then
    printf 'skip completed %s\n' "$fold" | tee -a "$LOG"
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
    --seed 2701 \
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
printf '%s\n' "robust direct joint market-transition experiment complete" | tee -a "$LOG"
