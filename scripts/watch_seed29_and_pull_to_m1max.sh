#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Users/wooyeol/work/stock-v2}"
REMOTE_HOST="${REMOTE_HOST:-157.157.221.29}"
REMOTE_PORT="${REMOTE_PORT:-24856}"
REMOTE_USER="${REMOTE_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
POLL_SECONDS="${POLL_SECONDS:-60}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/workspace/stock-v2-seed-stability}"
TRAIN_RUN="${TRAIN_RUN:-broad_transition_jepa_v6_lifecycle500_v4_seed29_diagnostic_rtx4000ada_20260715}"
STABILITY_NAME="${STABILITY_NAME:-seed29_stability_v1_20260715}"
REPLICATION_NAME="${REPLICATION_NAME:-seed29_open_innovation_schema4_v1_20260715}"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
LOCK_DIR="$PROJECT_ROOT/ops/seed29_m1max_pull_watcher.lock"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -i "$SSH_KEY" -p "$REMOTE_PORT")
RSYNC_RSH="ssh -o BatchMode=yes -o ConnectTimeout=15 -i $SSH_KEY -p $REMOTE_PORT"

mkdir -p "$PROJECT_ROOT/ops"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf '%s\n' "seed29 M1 Max pull watcher is already running"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

wait_for_marker() {
  local marker="$1"
  local process_pattern="$2"
  local label="$3"
  local missing_checks=0

  printf 'waiting for %s\n' "$label"
  while ! "${SSH[@]}" "$REMOTE" "test -f '$marker'"; do
    if "${SSH[@]}" "$REMOTE" "pgrep -f '$process_pattern' >/dev/null"; then
      missing_checks=0
    else
      missing_checks=$((missing_checks + 1))
      if [[ "$missing_checks" -ge 3 ]]; then
        printf '%s stopped without marker %s\n' "$label" "$marker" >&2
        exit 6
      fi
    fi
    sleep "$POLL_SECONDS"
  done
}

sync_training() {
  mkdir -p "$PROJECT_ROOT/models/$TRAIN_RUN" "$PROJECT_ROOT/reports/$TRAIN_RUN"
  rsync -a --partial -e "$RSYNC_RSH" \
    "$REMOTE:$PERSISTENT_ROOT/models/$TRAIN_RUN/" \
    "$PROJECT_ROOT/models/$TRAIN_RUN/"
  rsync -a --partial -e "$RSYNC_RSH" \
    "$REMOTE:$PERSISTENT_ROOT/reports/$TRAIN_RUN/" \
    "$PROJECT_ROOT/reports/$TRAIN_RUN/"
}

verify_training_checkpoints() {
  local local_root="$PROJECT_ROOT/models/$TRAIN_RUN"
  local remote_hashes
  local local_hashes

  remote_hashes="$("${SSH[@]}" "$REMOTE" \
    "find '$PERSISTENT_ROOT/models/$TRAIN_RUN' -name graph_jepa_real.pt -type f -exec sha256sum {} + | awk '{print \$1}' | sort")"
  local_hashes="$(find "$local_root" -name graph_jepa_real.pt -type f -print0 \
    | xargs -0 shasum -a 256 | awk '{print $1}' | sort)"
  if [[ "$(printf '%s\n' "$local_hashes" | sed '/^$/d' | wc -l | tr -d ' ')" -ne 5 ]]; then
    printf '%s\n' "M1 Max does not contain five seed29 checkpoints" >&2
    exit 7
  fi
  if [[ "$remote_hashes" != "$local_hashes" ]]; then
    printf '%s\n' "seed29 checkpoint hashes differ between RunPod and M1 Max" >&2
    exit 8
  fi
}

TRAIN_REPORT="$PERSISTENT_ROOT/reports/$TRAIN_RUN"
STABILITY_REPORT="$PERSISTENT_ROOT/reports/$STABILITY_NAME"
REPLICATION_REPORT="$PERSISTENT_ROOT/reports/$REPLICATION_NAME"

wait_for_marker "$TRAIN_REPORT/PIPELINE_COMPLETE" \
  "[r]un_walk_forward_node_eval.py.*$TRAIN_RUN" "seed29 five-fold training"
sync_training
verify_training_checkpoints

wait_for_marker "$STABILITY_REPORT/EVALUATION_COMPLETE" \
  "[r]un_seed29_stability_eval_rtx4000ada.sh" "seed29 stability evaluation"
mkdir -p "$PROJECT_ROOT/reports/$STABILITY_NAME"
rsync -a --partial -e "$RSYNC_RSH" \
  "$REMOTE:$STABILITY_REPORT/" "$PROJECT_ROOT/reports/$STABILITY_NAME/"

wait_for_marker "$REPLICATION_REPORT/REPLICATION_COMPLETE" \
  "[r]un_seed29_open_innovation_schema4_rtx4000ada.sh" \
  "seed29 schema4 open-innovation replication"
mkdir -p "$PROJECT_ROOT/reports/$REPLICATION_NAME"
rsync -a --partial --exclude 'caches' --exclude 'caches/**' -e "$RSYNC_RSH" \
  "$REMOTE:$REPLICATION_REPORT/" "$PROJECT_ROOT/reports/$REPLICATION_NAME/"

mkdir -p "$PROJECT_ROOT/configs/runpod-frozen-contracts-20260715"
rsync -a --partial -e "$RSYNC_RSH" \
  "$REMOTE:$PERSISTENT_ROOT/contracts/" \
  "$PROJECT_ROOT/configs/runpod-frozen-contracts-20260715/"
touch "$PROJECT_ROOT/reports/$REPLICATION_NAME/M1MAX_SYNC_COMPLETE"
printf '%s\n' "seed29 artifacts verified and synchronized to M1 Max; live orders remain disabled"
