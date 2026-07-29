#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
MODEL_PREFIX="${MODEL_PREFIX:-reports/classical_extra_trees_market_transition_v6_by_event}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reports/classical_extra_trees_market_transition_v6_by_event_feature_importance_20260714}"
WAIT_MARKER="${WAIT_MARKER:-reports/classical_extra_trees_market_transition_v6_by_event_two_seed_20260714_COMPLETE}"
FORCE_REBUILD="${FORCE_REBUILD:-0}"
cd "$ROOT"

until [[ -f "$WAIT_MARKER" ]]; do
  sleep 15
done

for seed in 2701 4301; do
  for fold in fold1 fold2; do
    input="${MODEL_PREFIX}_seed${seed}_20260714/$fold"
    output="$OUTPUT_ROOT/seed${seed}/$fold"
    if [[ "$FORCE_REBUILD" == "1" || ! -f "$output/summary.json" ]]; then
      "$PYTHON_BIN" scripts/audit_classical_market_transition_feature_importance.py \
        --model-root "$input" \
        --output-dir "$output" \
        --top-k 30
    fi
  done
done

touch "$OUTPUT_ROOT/EXPERIMENT_COMPLETE"
