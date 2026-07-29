#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-etf-ablation-v1}"
PYTHON_BIN="${PYTHON:-/root/venvs/stock-v2-cu128/bin/python}"
PREFLIGHT_LOG="ops/training/us_etf_node_ablation_v1_baseline_exact_preflight_20260716.log"
REPORT_ROOT="reports/us_etf_node_ablation_v1_edge_connectivity_20260716"
LOG="ops/training/us_etf_node_ablation_v1_edge_connectivity_20260716.log"

cd "$ROOT"
mkdir -p "$REPORT_ROOT" "$(dirname "$LOG")"
exec 9>"$REPORT_ROOT/.run.lock"
if ! flock -n 9; then
  printf '%s\n' "US ETF edge connectivity audit is already running"
  exit 0
fi

"$PYTHON_BIN" scripts/audit_us_etf_edge_connectivity.py \
  --source-log "$PREFLIGHT_LOG" \
  --train-end 2024-01-03 \
  --output "$REPORT_ROOT/fold3.json" \
  --scratch-root "$REPORT_ROOT/scratch_fold3" \
  --require-cross-edges \
  2>&1 | tee "$LOG"

"$PYTHON_BIN" scripts/audit_us_etf_edge_connectivity.py \
  --source-log "$PREFLIGHT_LOG" \
  --train-end 2024-11-05 \
  --output "$REPORT_ROOT/fold4.json" \
  --scratch-root "$REPORT_ROOT/scratch_fold4" \
  --require-cross-edges \
  2>&1 | tee -a "$LOG"

touch "$REPORT_ROOT/COMPLETE"
