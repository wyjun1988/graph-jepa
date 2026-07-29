#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
WAIT_ROOT="reports/impact_direction_v2_jepa_m1pro_20260714"
JEPA_ROOT="reports/impact_direction_v2_seed17_20260714"
FIXED_ROOT="reports/impact_direction_v2_fixed_k_seed17_20260714"
LOG_PATH="ops/training/sync_impact_direction_v2_from_m1pro_20260714.log"
PID_PATH="ops/training/sync_impact_direction_v2_from_m1pro_20260714.pid"

cd "$ROOT"
mkdir -p ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "impact-direction v2 sync already queued: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup caffeinate -dimsu bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "impact-direction v2 sync queued: $!"
  exit 0
fi

while [[ ! -f "$WAIT_ROOT/EXPERIMENT_COMPLETE" ]]; do
  if [[ -f "$WAIT_ROOT/EXPERIMENT_FAILED" ]]; then
    echo "M1 Pro impact-direction v2 failed; refusing artifact sync" >&2
    exit 4
  fi
  sleep 30
done

test -f "$JEPA_ROOT/fold1/summary.json"
test -f "$JEPA_ROOT/fold2/summary.json"
test -f "$FIXED_ROOT/fold1/summary.json"
test -f "$FIXED_ROOT/fold2/summary.json"

ssh wooyeol@mac-pro 'mkdir -p /Users/wooyeol/work/stock-v2/reports'
scp -r "$JEPA_ROOT" "$FIXED_ROOT" "$WAIT_ROOT" \
  wooyeol@mac-pro:/Users/wooyeol/work/stock-v2/reports/

RUNPOD_SSH=(
  ssh -i /Users/wooyeol/.ssh/id_ed25519 -p 22008 root@194.68.245.170
)
RUNPOD_SCP=(
  scp -i /Users/wooyeol/.ssh/id_ed25519 -P 22008
)
"${RUNPOD_SSH[@]}" 'mkdir -p /workspace/stock-v2/reports'
"${RUNPOD_SCP[@]}" -r "$JEPA_ROOT" "$FIXED_ROOT" "$WAIT_ROOT" \
  root@194.68.245.170:/workspace/stock-v2/reports/

echo "impact-direction v2 artifacts synced to M1 Max and RunPod"
