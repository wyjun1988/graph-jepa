#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
REMOTE="root@194.68.245.170"
RSH=(ssh -p 22045 -i /Users/wooyeol/.ssh/id_ed25519 -o BatchMode=yes)
REMOTE_ROOT="/workspace/stock-v2"
REPORT_NAME="walk_forward_causal453_path_v2_20260713"
MODEL_NAME="walk_forward_causal453_path_v2_20260713"

cd "$ROOT"
while true; do
  if "${RSH[@]}" "$REMOTE" "test -f '$REMOTE_ROOT/reports/$REPORT_NAME/PIPELINE_COMPLETE'"; then
    break
  fi
  if "${RSH[@]}" "$REMOTE" "test -f '$REMOTE_ROOT/reports/$REPORT_NAME/PIPELINE_FAILED'"; then
    rsync -az -e "${RSH[*]}" \
      "$REMOTE:$REMOTE_ROOT/reports/$REPORT_NAME" reports/
    exit 3
  fi
  sleep 120
done

rsync -az -e "${RSH[*]}" \
  "$REMOTE:$REMOTE_ROOT/reports/$REPORT_NAME" reports/
rsync -az -e "${RSH[*]}" \
  "$REMOTE:$REMOTE_ROOT/models/$MODEL_NAME" models/
touch reports/causal453_path_v2_pull_complete_20260713
