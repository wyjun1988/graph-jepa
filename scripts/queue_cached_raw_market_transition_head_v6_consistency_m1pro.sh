#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
RUN_ROOT="reports/cached_raw_market_transition_head_v6_epoch008_20260714"
OUTPUT="$RUN_ROOT/seed2701_weighted_rerun"
TARGET="reports/market_transition_target_audit_v6_systemic_impact_metric_20260714/fold1"
CACHE="$RUN_ROOT/frozen_raw_transition_pool.npz"
WAIT_MARKER="reports/classical_extra_trees_market_transition_v6_two_seed_20260714_COMPLETE"
LOG="ops/training/cached_raw_market_transition_head_v6_seed2701_weighted_rerun_20260714_m1pro.log"

cd "$ROOT"
mkdir -p "$OUTPUT" "$(dirname "$LOG")"
export PYTORCH_ENABLE_MPS_FALLBACK=1

until [[ -f "$WAIT_MARKER" ]]; do
  sleep 15
done

if [[ ! -f "$OUTPUT/summary.json" ]]; then
  "$PYTHON_BIN" scripts/benchmark_cached_pooled_market_transition_head.py \
    --model-dir models/milestones/broad_transition_v5_seed43_fold1_epoch008 \
    --output-dir "$OUTPUT" \
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
    --batch-size 64 \
    --eval-batch-size 256 \
    --device mps \
    --seed 2701 \
    --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
fi

if [[ ! -f "$OUTPUT/major_trajectory/summary.json" ]]; then
  "$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
    --target-audit-root "$TARGET" \
    --prediction-root "$OUTPUT" \
    --output-dir "$OUTPUT/major_trajectory" \
    --major-event-quantile 0.90 \
    2>&1 | tee -a "$LOG"
fi

"$PYTHON_BIN" - "$OUTPUT/summary.json" <<'PY'
import json
import sys

summary = json.load(open(sys.argv[1], encoding="utf-8"))
if summary.get("impact_weighted_event_loss") is not True:
    raise SystemExit("rerun did not record impact-weighted event loss")
if summary.get("live_orders_allowed") is not False:
    raise SystemExit("unsafe live order contract")
PY

touch "$RUN_ROOT/WEIGHTED_SEED2701_RERUN_COMPLETE"
printf '%s\n' "cached raw v6 weighted seed2701 consistency rerun complete" | tee -a "$LOG"
