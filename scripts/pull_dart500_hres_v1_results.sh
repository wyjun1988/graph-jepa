#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
POD="root@194.68.245.170"
SSH_PORT="22045"
SSH_KEY="/Users/wooyeol/.ssh/id_ed25519"
REMOTE_ROOT="/workspace/stock-v2"
REMOTE_PID_FILE="reports/causal453_hres_v2_driver_20260713.pid"
COMPLETE_FILE="reports/walk_forward_causal453_hres_v2_20260713/PIPELINE_COMPLETE"
LOG="$ROOT/logs/pull_causal453_hres_v2_results_20260713.log"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

remote() {
  ssh -p "$SSH_PORT" -i "$SSH_KEY" -o BatchMode=yes "$POD" "$@"
}

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

echo "[$(timestamp)] waiting for RunPod model pipeline"
while ! remote "test -f '$REMOTE_ROOT/$COMPLETE_FILE'"; do
  remote "pid=\$(cat '$REMOTE_ROOT/$REMOTE_PID_FILE' 2>/dev/null || true); test -n \"\$pid\" && kill -0 \"\$pid\" 2>/dev/null" || {
    echo "[$(timestamp)] RunPod model pipeline exited without completion marker" >&2
    exit 1
  }
  sleep 300
done

echo "[$(timestamp)] pipeline complete; pulling reports and checkpoints"
mkdir -p "$ROOT/reports" "$ROOT/models"
RSYNC_SSH="ssh -p $SSH_PORT -i $SSH_KEY -o BatchMode=yes"

for path in \
  reports/walk_forward_causal453_hres_v2_20260713 \
  reports/direct_state_mlp_causal453_hres_v2 \
  reports/direct_state_mlp_causal453_hres_v2_nograph \
  reports/direct_vs_jepa_causal453_hres_v2 \
  reports/shadow_gate_causal453_hres_v2; do
  rsync -rt --partial -e "$RSYNC_SSH" \
    "$POD:$REMOTE_ROOT/$path" "$ROOT/reports/"
done

rsync -rt --partial -e "$RSYNC_SSH" \
  "$POD:$REMOTE_ROOT/models/walk_forward_causal453_hres_v2_20260713" \
  "$ROOT/models/"
rsync -rt --partial -e "$RSYNC_SSH" \
  "$POD:$REMOTE_ROOT/reports/causal453_hres_v2_driver_20260713.log" \
  "$ROOT/reports/"

touch "$ROOT/reports/walk_forward_causal453_hres_v2_20260713/PULLED_FROM_RUNPOD"
echo "[$(timestamp)] result pull complete"
