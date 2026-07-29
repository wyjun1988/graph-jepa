#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
REMOTE="root@194.68.245.170"
RSYNC_RSH="ssh -p 22045 -i /Users/wooyeol/.ssh/id_ed25519 -o BatchMode=yes"
MARKER="/workspace/stock-v2/reports/walk_forward_causal453_path_v1_20260713/PIPELINE_COMPLETE"

cd "$ROOT"
until ssh -p 22045 -i /Users/wooyeol/.ssh/id_ed25519 -o BatchMode=yes \
  "$REMOTE" "test -f '$MARKER'"; do
  sleep 120
done

rsync -az -e "$RSYNC_RSH" \
  "$REMOTE:/workspace/stock-v2/reports/path_objective_screen_causal453_v1_20260713" \
  reports/
rsync -az -e "$RSYNC_RSH" \
  "$REMOTE:/workspace/stock-v2/reports/walk_forward_causal453_path_v1_20260713" \
  reports/
rsync -az -e "$RSYNC_RSH" \
  "$REMOTE:/workspace/stock-v2/models/path_objective_screen_causal453_v1_20260713" \
  models/
rsync -az -e "$RSYNC_RSH" \
  "$REMOTE:/workspace/stock-v2/models/walk_forward_causal453_path_v1_20260713" \
  models/
touch reports/causal453_path_v1_pull_complete_20260713
