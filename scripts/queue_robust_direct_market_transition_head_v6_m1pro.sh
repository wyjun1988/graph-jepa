#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
TARGET_ROOT="reports/market_transition_target_audit_v6_systemic_impact_metric_20260714"
PROJECTED_DIAGNOSTIC="reports/cached_projected_market_transition_head_v6_epoch008_20260714"
cd "$ROOT"

until [[ -f "$PROJECTED_DIAGNOSTIC/EXPERIMENT_COMPLETE" ]]; do
  sleep 30
done

for seed in 2701 4301 7301; do
  run="robust_direct_market_transition_head_v6_impact_mass_seed${seed}_20260714"
  if [[ ! -f "reports/$run/EXPERIMENT_COMPLETE" ]]; then
    RUN_NAME="$run" \
    TARGET_VERSION="market_transition_v6_systemic_impact_20260714" \
    IMPACT_METRIC_VERSION="market_transition_systemic_impact_mass_v2_20260714" \
    HEAD_SEED="$seed" \
      bash scripts/run_robust_direct_market_transition_head_v4_m1pro.sh
  fi
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

touch reports/robust_direct_market_transition_head_v6_three_seed_20260714_COMPLETE
