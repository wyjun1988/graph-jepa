#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
TARGET_ROOT="reports/market_transition_target_audit_v5_systemic_impact_metric_20260714"
cd "$ROOT"

runs=(
  robust_direct_market_transition_head_v5_impact_mass_seed2701_20260714
  robust_direct_market_transition_head_v5_impact_mass_seed4301_20260714
)

for run in "${runs[@]}"; do
  until [[ -f "reports/$run/EXPERIMENT_COMPLETE" ]]; do
    sleep 30
  done
  for fold in fold1 fold2; do
    output="reports/$run/$fold/major_trajectory"
    if [[ -f "$output/summary.json" ]]; then
      continue
    fi
    "$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
      --target-audit-root "$TARGET_ROOT/$fold" \
      --prediction-root "reports/$run/$fold" \
      --output-dir "$output" \
      --major-event-quantile 0.90
  done
done
