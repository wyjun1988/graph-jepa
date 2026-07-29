#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
WAIT_ROOT="reports/direct_impact_fixed_k_m1pro_v1_20260714"
IMPACT_ROOT="reports/impact_head_weight95_seed17_20260714"
IMPACT_ARCHIVE="reports/impact_head_weight95_seed17_count_metrics_v1_20260714"
FIXED_ROOT="reports/impact_head_fixed_k_seed17_20260714"
FIXED_ARCHIVE="reports/impact_head_fixed_k_seed17_count_metrics_v1_20260714"
LOG_PATH="ops/training/impact_mass_refresh_m1pro_20260714.log"
PID_PATH="ops/training/impact_mass_refresh_m1pro_20260714.pid"

cd "$ROOT"
mkdir -p ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "impact mass refresh already queued: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup caffeinate -dimsu bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "impact mass refresh queued: $!"
  exit 0
fi

while [[ ! -f "$WAIT_ROOT/EVALUATION_COMPLETE" ]]; do
  if [[ -f "$WAIT_ROOT/EVALUATION_FAILED" ]]; then
    echo "direct fixed-k refresh failed" >&2
    exit 4
  fi
  sleep 30
done

if ! grep -q 'signed_realized_tail_mass_capture' "$IMPACT_ROOT/fold1/summary.json"; then
  test ! -e "$IMPACT_ARCHIVE"
  mv "$IMPACT_ROOT" "$IMPACT_ARCHIVE"
  bash scripts/run_jepa_impact_seed17_m1pro.sh --worker
fi

test -f "$IMPACT_ROOT/IMPACT_COMPLETE"

if ! grep -q 'signed_realized_tail_mass_capture_at_k' "$FIXED_ROOT/fold1/summary.json"; then
  test ! -e "$FIXED_ARCHIVE"
  mv "$FIXED_ROOT" "$FIXED_ARCHIVE"
  bash scripts/run_impact_fixed_k_m1pro.sh --worker
fi

test -f "$FIXED_ROOT/EVALUATION_COMPLETE"
"$ROOT/.venv-mps/bin/python" scripts/aggregate_impact_validation.py \
  --root "$ROOT" \
  --output-dir reports/impact_validation_aggregate_20260714

echo "impact magnitude-weighted metrics refresh complete"
