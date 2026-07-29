#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Users/wooyeol/work/stock-v2}"
REMOTE_HOST="${REMOTE_HOST:-157.157.221.29}"
REMOTE_PORT="${REMOTE_PORT:-24856}"
REMOTE_USER="${REMOTE_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/workspace/stock-v2-seed-stability}"
STABILITY_NAME="${STABILITY_NAME:-seed29_stability_v1_20260715}"
RUN_NAME="${RUN_NAME:-seed29_open_innovation_schema4_v1_20260715}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/stock-v2-runtime}"
STABILITY_ROOT="$PERSISTENT_ROOT/reports/$STABILITY_NAME"
OUTPUT_ROOT="$PERSISTENT_ROOT/reports/$RUN_NAME"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
LOCK_DIR="$PROJECT_ROOT/ops/seed29_open_innovation_watcher.lock"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -i "$SSH_KEY" -p "$REMOTE_PORT")
SCP=(scp -o BatchMode=yes -o ConnectTimeout=15 -i "$SSH_KEY" -P "$REMOTE_PORT")

mkdir -p "$PROJECT_ROOT/ops"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf '%s\n' "seed29 open-innovation watcher is already running"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

printf '%s\n' "waiting for seed29 stability evaluation"
while ! "${SSH[@]}" "$REMOTE" \
  "test -f '$STABILITY_ROOT/EVALUATION_COMPLETE'"; do
  sleep "$POLL_SECONDS"
done

printf '%s\n' "stability evaluation complete; syncing frozen replication files"
"${SSH[@]}" "$REMOTE" "mkdir -p '$PERSISTENT_ROOT/contracts'"
"${SCP[@]}" \
  "$PROJECT_ROOT/configs/seed29-open-innovation-schema4-v1-20260715.json" \
  "$REMOTE:$PERSISTENT_ROOT/contracts/seed29-open-innovation-schema4-v1-20260715.json"
"${SCP[@]}" "$PROJECT_ROOT/stock_v2/open_innovation_replication.py" \
  "$REMOTE:$REMOTE_ROOT/stock_v2/open_innovation_replication.py"
"${SCP[@]}" \
  "$PROJECT_ROOT/scripts/verify_seed29_open_innovation_replication.py" \
  "$REMOTE:$REMOTE_ROOT/scripts/verify_seed29_open_innovation_replication.py"
"${SCP[@]}" \
  "$PROJECT_ROOT/scripts/run_seed29_open_innovation_schema4_rtx4000ada.sh" \
  "$REMOTE:$REMOTE_ROOT/scripts/run_seed29_open_innovation_schema4_rtx4000ada.sh"

"${SSH[@]}" "$REMOTE" \
  "chmod +x '$REMOTE_ROOT/scripts/run_seed29_open_innovation_schema4_rtx4000ada.sh'; \
   mkdir -p '$PERSISTENT_ROOT/launcher'; \
   if test -f '$OUTPUT_ROOT/REPLICATION_COMPLETE'; then \
     printf '%s\\n' 'open-innovation replication already complete'; \
   elif pgrep -f '[r]un_seed29_open_innovation_schema4_rtx4000ada.sh' >/dev/null; then \
     printf '%s\\n' 'open-innovation replication already running'; \
   else \
     cd '$REMOTE_ROOT'; \
     nohup bash scripts/run_seed29_open_innovation_schema4_rtx4000ada.sh \
       >'$PERSISTENT_ROOT/launcher/seed29_open_innovation.nohup' 2>&1 </dev/null & \
     echo \$! >'$PERSISTENT_ROOT/launcher/seed29_open_innovation.pid'; \
   fi"

missing_checks=0
while ! "${SSH[@]}" "$REMOTE" \
  "test -f '$OUTPUT_ROOT/REPLICATION_COMPLETE'"; do
  if "${SSH[@]}" "$REMOTE" \
    "pgrep -f '[r]un_seed29_open_innovation_schema4_rtx4000ada.sh' >/dev/null"; then
    missing_checks=0
  else
    missing_checks=$((missing_checks + 1))
    if [[ "$missing_checks" -ge 3 ]]; then
      printf '%s\n' "open-innovation replication stopped without completion" >&2
      "${SSH[@]}" "$REMOTE" \
        "tail -100 '$PERSISTENT_ROOT/launcher/seed29_open_innovation.nohup'" >&2 || true
      exit 8
    fi
  fi
  sleep "$POLL_SECONDS"
done

printf '%s\n' "seed29 open-innovation replication complete; live orders remain disabled"
