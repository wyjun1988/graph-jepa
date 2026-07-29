#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
MODEL="models/milestones/broad_transition_v5_seed43_fold1_epoch008"
RUN_ROOT="reports/cached_projected_market_transition_head_v6_epoch008_20260714"
TARGET="reports/market_transition_target_audit_v6_systemic_impact_metric_20260714/fold1"
CACHE="reports/cached_pooled_market_transition_head_v5_epoch008_seed2701_20260714/frozen_transition_pool.npz"
RAW_MARKER="reports/cached_raw_market_transition_head_v6_epoch008_20260714/EXPERIMENT_COMPLETE"
LOG="ops/training/cached_projected_market_transition_head_v6_epoch008_20260714_m1pro.log"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
cd "$ROOT"
mkdir -p "$RUN_ROOT" "$(dirname "$LOG")"
export PYTORCH_ENABLE_MPS_FALLBACK=1

until [[ -f "$RAW_MARKER" ]]; do
  sleep 30
done

printf '%s\n' \
  '{"scope":"frozen_epoch8_diagnostic","target_version":"market_transition_v6_systemic_impact_20260714","impact_metric_version":"market_transition_systemic_impact_mass_v2_20260714","representation":"trained_transition_projector_robust_pool","impact_weighted_event_loss":true,"head_seeds":[2701,4301],"test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

for seed in 2701 4301; do
  output="$RUN_ROOT/seed${seed}"
  if [[ ! -f "$output/summary.json" ]]; then
    "$PYTHON_BIN" scripts/benchmark_cached_pooled_market_transition_head.py \
      --model-dir "$MODEL" \
      --output-dir "$output" \
      --pooled-cache "$CACHE" \
      --pooling-mode projected \
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
      --seed "$seed" \
      --cache-dir "$OHLCV" \
      --external-cache-dir data/external_cache \
      2>&1 | tee -a "$LOG"
  fi
  if [[ ! -f "$output/major_trajectory/summary.json" ]]; then
    "$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
      --target-audit-root "$TARGET" \
      --prediction-root "$output" \
      --output-dir "$output/major_trajectory" \
      --major-event-quantile 0.90 \
      2>&1 | tee -a "$LOG"
  fi
done

touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "cached projected v6 epoch8 diagnostic complete" | tee -a "$LOG"
