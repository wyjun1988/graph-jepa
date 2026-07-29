#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON_BIN="${PYTHON:-/workspace/venvs/stock-v2-cu128/bin/python}"
RUN_ROOT="reports/separated_major_path_head_seed17_v32_20260714"
LOG="ops/training/separated_major_path_head_seed17_a5000_20260714.log"
MODEL_ROOT="models/walk_forward_causal453_path_v2_20260713"
CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
printf '%s\n' \
  '{"scope":"posthoc_research_only","target_version":"market_transition_v3_breadth_20260714","objective_version":"separated_major_path_v32_20260714","test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

run_fold() {
  local fold="$1"
  local model="$2"
  local output="$RUN_ROOT/$fold"
  if [[ -f "$output/summary.json" ]]; then
    printf 'skip completed %s\n' "$fold" | tee -a "$LOG"
    return
  fi
  "$PYTHON_BIN" scripts/benchmark_separated_major_path_head.py \
    --model-dir "$MODEL_ROOT/$model" \
    --output-dir "$output" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --major-event-quantile 0.90 \
    --epochs 20 \
    --patience 5 \
    --projection-dim 128 \
    --hidden-dim 256 \
    --layers 2 \
    --heads 8 \
    --dropout 0.10 \
    --learning-rate 0.0003 \
    --weight-decay 0.001 \
    --batch-size 64 \
    --eval-batch-size 128 \
    --edge-cache-workers 16 \
    --device cuda \
    --seed 3201 \
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
printf '%s\n' "seed17 separated-major v3.2 experiment complete" | tee -a "$LOG"
