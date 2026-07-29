#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
WAIT_ROOT="reports/impact_head_fixed_k_seed17_20260714"
DIRECT_ROOT="reports/direct_impact_fixed_k_m1pro_v1_20260714"
DIRECT_ARCHIVE="reports/direct_impact_fixed_k_m1pro_per_horizon_v2_20260714"
LOG_PATH="ops/training/direct_cross_horizon_refresh_m1pro_20260714.log"
PID_PATH="ops/training/direct_cross_horizon_refresh_m1pro_20260714.pid"

cd "$ROOT"
mkdir -p ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "direct cross-horizon refresh already queued: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup caffeinate -dimsu bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "direct cross-horizon refresh queued: $!"
  exit 0
fi

while ! grep -q 'cross_horizon_metrics' "$WAIT_ROOT/fold1/summary.json" 2>/dev/null; do
  if [[ -f "$WAIT_ROOT/EVALUATION_FAILED" ]]; then
    echo "JEPA fixed-k cross-horizon evaluation failed" >&2
    exit 4
  fi
  sleep 30
done

if ! grep -q 'cross_horizon_metrics' "$DIRECT_ROOT/graph/fold1/summary.json"; then
  test ! -e "$DIRECT_ARCHIVE"
  mv "$DIRECT_ROOT" "$DIRECT_ARCHIVE"
  bash scripts/run_direct_impact_fixed_k_m1pro.sh --worker
fi

test -f "$DIRECT_ROOT/EVALUATION_COMPLETE"
"$ROOT/.venv-mps/bin/python" scripts/aggregate_impact_validation.py \
  --root "$ROOT" \
  --output-dir reports/impact_validation_aggregate_20260714

echo "direct cross-horizon metrics refresh complete"
