#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps-max/bin/python}"
RUN_ROOT="reports/market_transition_head_seed43_v3_20260714"
LOG="ops/training/market_transition_head_seed43_m1max_20260714.log"
MODEL_ROOT="models/walk_forward_causal453_path_multiseed_seed43_20260714"
CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
printf '%s\n' \
  '{"scope":"posthoc_research_only","target_version":"market_transition_v3_breadth_20260714","objective":"multiseed joint-horizon broad market transition prediction","seed":43,"test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

run_fold() {
  local fold="$1"
  local model="$2"
  local eval_seed="$3"
  local output="$RUN_ROOT/$fold"
  if [[ -f "$output/summary.json" ]]; then
    printf 'skip completed %s\n' "$fold" | tee -a "$LOG"
    return
  fi
  "$PYTHON_BIN" scripts/benchmark_market_transition_head.py \
    --model-dir "$MODEL_ROOT/$model" \
    --output-dir "$output" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --epochs 16 \
    --patience 4 \
    --projection-dim 128 \
    --hidden-dim 256 \
    --layers 2 \
    --heads 8 \
    --dropout 0.10 \
    --learning-rate 0.0003 \
    --weight-decay 0.001 \
    --batch-size 32 \
    --eval-batch-size 64 \
    --edge-cache-workers 8 \
    --device mps \
    --seed "$eval_seed" \
    --cache-dir "$CACHE" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

run_fold \
  fold1 \
  strict_causal453_path_v2_path_w12_p005_l025_skip_seed43_fold1_20231229_to_20241230 \
  4301
run_fold \
  fold2 \
  strict_causal453_path_v2_path_w12_p005_l025_skip_seed43_fold2_20241230_to_20260710 \
  4301

touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "seed43 joint market-transition experiment complete" | tee -a "$LOG"
