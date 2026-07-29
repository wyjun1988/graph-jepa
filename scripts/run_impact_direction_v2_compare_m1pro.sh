#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="$ROOT/.venv-mps/bin/python"
WAIT_ROOT="reports/impact_direction_v2_jepa_m1pro_20260714"
OUTPUT_ROOT="reports/impact_direction_v2_vs_v1_seed17_20260714"
LOG_PATH="ops/training/impact_direction_v2_compare_m1pro_20260714.log"
PID_PATH="ops/training/impact_direction_v2_compare_m1pro_20260714.pid"

cd "$ROOT"
mkdir -p ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "impact-direction v2 comparison already queued: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup caffeinate -dimsu bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "impact-direction v2 comparison queued: $!"
  exit 0
fi

while [[ ! -f "$WAIT_ROOT/EXPERIMENT_COMPLETE" ]]; do
  if [[ -f "$WAIT_ROOT/EXPERIMENT_FAILED" ]]; then
    echo "JEPA v2 dependency failed; refusing version comparison" >&2
    exit 4
  fi
  sleep 30
done

"$PYTHON" scripts/compare_impact_direction_versions.py \
  --v1-root reports/impact_head_weight95_seed17_20260714 \
  --v2-root reports/impact_direction_v2_seed17_20260714 \
  --v1-fixed-root reports/impact_head_fixed_k_seed17_20260714 \
  --v2-fixed-root reports/impact_direction_v2_fixed_k_seed17_20260714 \
  --output-dir "$OUTPUT_ROOT"

ssh wooyeol@mac-pro 'mkdir -p /Users/wooyeol/work/stock-v2/reports/impact_direction_v2_vs_v1_seed17_20260714'
scp "$OUTPUT_ROOT/summary.json" "$OUTPUT_ROOT/summary.md" \
  wooyeol@mac-pro:/Users/wooyeol/work/stock-v2/reports/impact_direction_v2_vs_v1_seed17_20260714/

ssh -i /Users/wooyeol/.ssh/id_ed25519 -p 22008 root@194.68.245.170 \
  'mkdir -p /workspace/stock-v2/reports/impact_direction_v2_vs_v1_seed17_20260714'
scp -i /Users/wooyeol/.ssh/id_ed25519 -P 22008 \
  "$OUTPUT_ROOT/summary.json" "$OUTPUT_ROOT/summary.md" \
  root@194.68.245.170:/workspace/stock-v2/reports/impact_direction_v2_vs_v1_seed17_20260714/

echo "impact-direction v2 versus v1 paired comparison complete and synced"
