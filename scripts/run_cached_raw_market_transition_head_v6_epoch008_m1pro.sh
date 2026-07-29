#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
MODEL="models/milestones/broad_transition_v5_seed43_fold1_epoch008"
RUN_ROOT="reports/cached_raw_market_transition_head_v6_epoch008_20260714"
TARGET="reports/market_transition_target_audit_v6_systemic_impact_metric_20260714/fold1"
CACHE="$RUN_ROOT/frozen_raw_transition_pool.npz"
LOG="ops/training/cached_raw_market_transition_head_v6_epoch008_20260714_m1pro.log"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
export PYTORCH_ENABLE_MPS_FALLBACK=1

until [[ -f "$TARGET/summary.json" ]]; do
  sleep 15
done

printf '%s\n' \
  '{"scope":"frozen_epoch8_diagnostic","target_version":"market_transition_v6_systemic_impact_20260714","impact_metric_version":"market_transition_systemic_impact_mass_v2_20260714","representation":"raw_context_and_latent_delta_robust_pool","impact_weighted_event_loss":true,"head_seeds":[2701,4301],"test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

run_head() {
  local seed="$1"
  local output="$RUN_ROOT/seed${seed}"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/benchmark_cached_pooled_market_transition_head.py \
    --model-dir "$MODEL" \
    --output-dir "$output" \
    --pooled-cache "$CACHE" \
    --pooling-mode raw \
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
    --cache-batch-size 2 \
    --batch-size 64 \
    --eval-batch-size 256 \
    --edge-cache-workers 8 \
    --device mps \
    --seed "$seed" \
    --cache-dir "$OHLCV" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

run_major() {
  local seed="$1"
  local prediction="$RUN_ROOT/seed${seed}"
  local output="$prediction/major_trajectory"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
    --target-audit-root "$TARGET" \
    --prediction-root "$prediction" \
    --output-dir "$output" \
    --major-event-quantile 0.90 \
    2>&1 | tee -a "$LOG"
}

run_head 2701
run_head 4301
run_major 2701
run_major 4301

touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "cached raw v6 epoch8 diagnostic complete" | tee -a "$LOG"
