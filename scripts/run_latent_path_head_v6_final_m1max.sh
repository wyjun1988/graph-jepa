#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
RUN_NAME="broad_transition_jepa_v6_systemic_seed17_rtx4000ada_20260714"
SEED="${SEED:-2701}"
OUTPUT_ROOT="$ROOT/reports/latent_path_head_${RUN_NAME}_final_seed${SEED}_20260714"
OHLCV="$ROOT/data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
LOG="$ROOT/logs/latent_path_head_${RUN_NAME}_final_seed${SEED}_m1max.log"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT" "$(dirname "$LOG")"
printf \
  '{"scope":"exploratory_frozen_v6_final_encoder_path_head","head_seed":%s,"selection_data":"fit_and_validation_only","fold2_used_for_selection":false,"live_orders_allowed":false}\n' \
  "$SEED" \
  > "$OUTPUT_ROOT/experiment_contract.json"

run_fold() {
  local fold="$1"
  local model_name="$2"
  local output="$OUTPUT_ROOT/$fold"
  if [[ -f "$output/EXPERIMENT_COMPLETE" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/benchmark_latent_trajectory_path_head.py \
    --model-dir "$ROOT/models/$RUN_NAME/$model_name" \
    --output-dir "$output" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --epochs 8 \
    --patience 2 \
    --hidden-dim 256 \
    --dropout 0.05 \
    --learning-rate 0.0003 \
    --batch-size 8 \
    --liquidity-top-k 300 \
    --latent-blend-weight 1.0 \
    --edge-cache-workers 8 \
    --device mps \
    --seed "$SEED" \
    --cache-dir "$OHLCV" \
    --external-cache-dir "$ROOT/data/external_cache" \
    2>&1 | tee -a "$LOG"
  "$PYTHON_BIN" - "$output/summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["live_orders_allowed"] is False
assert payload["fold2_used_for_selection"] is False
assert set(payload["horizons"]) == {"1", "2", "3", "5", "10"}
PY
  touch "$output/EXPERIMENT_COMPLETE"
}

run_fold \
  fold1 \
  "${RUN_NAME}_fold1_20231229_to_20241230"
run_fold \
  fold2 \
  "${RUN_NAME}_fold2_20241230_to_20260710"

touch "$OUTPUT_ROOT/EXPERIMENT_COMPLETE"
