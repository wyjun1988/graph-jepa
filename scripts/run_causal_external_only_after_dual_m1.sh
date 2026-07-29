#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 WAIT_PID" >&2
  exit 2
fi

WAIT_PID="$1"
ROOT="/Users/wooyeol/work/stock-v2"
RUNNER="$ROOT/scripts/run_causal_edge_candidate_m1.sh"
QUEUE_ROOT="$ROOT/reports/causal453_edge_candidates_m1_20260713"
PYTHON="/Users/wooyeol/work/stock/venv/bin/python"
NAME="signed_k6_external_only_exog"

cd "$ROOT"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done
if [[ ! -f "$QUEUE_ROOT/DUAL_PATH_QUEUE_COMPLETE" ]]; then
  echo "dual-path queue exited without completion marker" >&2
  exit 3
fi

if [[ ! -f "$QUEUE_ROOT/$NAME/CANDIDATE_COMPLETE" ]]; then
  TEMPORAL_STOCK_EDGE_SCALE=0.0 \
    "$RUNNER" "$NAME" signed 6 1.0 1.0 news_ fund_
fi

for baseline in signed_k6_t0_exog signed_k6_t025_exog; do
  "$PYTHON" scripts/compare_node_run_ablation.py \
    --baseline-daily "$QUEUE_ROOT/$baseline/walk_forward/node_eval/${baseline}_fold1_20231229_to_20241230/future_rollout.csv" \
    --candidate-daily "$QUEUE_ROOT/$NAME/walk_forward/node_eval/${NAME}_fold1_20231229_to_20241230/future_rollout.csv" \
    --baseline-label "$baseline" \
    --candidate-label "$NAME" \
    --output-dir "$QUEUE_ROOT/comparisons/${baseline}_vs_${NAME}"
done

touch "$QUEUE_ROOT/EXTERNAL_ONLY_QUEUE_COMPLETE"
