#!/usr/bin/env bash
set -Eeuo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/Users/wooyeol/work/stock-v2}"
REMOTE_HOST="${REMOTE_HOST:-157.157.221.29}"
REMOTE_PORT="${REMOTE_PORT:-24856}"
REMOTE_USER="${REMOTE_USER:-root}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519}"
POLL_SECONDS="${POLL_SECONDS:-60}"
TRAIN_RUN="${TRAIN_RUN:-broad_transition_jepa_v6_lifecycle500_v4_seed29_diagnostic_rtx4000ada_20260715}"
EVAL_NAME="${EVAL_NAME:-seed29_stability_v1_20260715}"
REMOTE_ROOT="${REMOTE_ROOT:-/root/stock-v2-runtime}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/workspace/stock-v2-seed-stability}"
TRAIN_ROOT="$PERSISTENT_ROOT/reports/$TRAIN_RUN"
EVAL_ROOT="$PERSISTENT_ROOT/reports/$EVAL_NAME"
REMOTE="$REMOTE_USER@$REMOTE_HOST"
LOCK_DIR="$PROJECT_ROOT/ops/seed29_eval_watcher.lock"
SSH=(ssh -o BatchMode=yes -o ConnectTimeout=15 -i "$SSH_KEY" -p "$REMOTE_PORT")
SCP=(scp -o BatchMode=yes -o ConnectTimeout=15 -i "$SSH_KEY" -P "$REMOTE_PORT")

mkdir -p "$PROJECT_ROOT/ops"
if ! mkdir "$LOCK_DIR" 2>/dev/null; then
  printf '%s\n' "seed29 evaluation watcher is already running"
  exit 0
fi
trap 'rmdir "$LOCK_DIR" 2>/dev/null || true' EXIT

printf 'waiting for source-frozen training: %s\n' "$TRAIN_RUN"
missing_training_checks=0
while ! "${SSH[@]}" "$REMOTE" "test -f '$TRAIN_ROOT/PIPELINE_COMPLETE'"; do
  if "${SSH[@]}" "$REMOTE" \
    "pgrep -f '[r]un_walk_forward_node_eval.py.*$TRAIN_RUN' >/dev/null"; then
    missing_training_checks=0
  else
    missing_training_checks=$((missing_training_checks + 1))
    if [[ "$missing_training_checks" -ge 3 ]]; then
      printf '%s\n' "training stopped without PIPELINE_COMPLETE" >&2
      exit 6
    fi
  fi
  sleep "$POLL_SECONDS"
done

printf '%s\n' "training complete; syncing audited evaluation-only sources"
"${SCP[@]}" "$PROJECT_ROOT/stock_v2/seed_stability.py" \
  "$REMOTE:$REMOTE_ROOT/stock_v2/seed_stability.py"
"${SCP[@]}" "$PROJECT_ROOT/scripts/evaluate_seed_stability.py" \
  "$REMOTE:$REMOTE_ROOT/scripts/evaluate_seed_stability.py"
"${SCP[@]}" "$PROJECT_ROOT/scripts/run_seed29_stability_eval_rtx4000ada.sh" \
  "$REMOTE:$REMOTE_ROOT/scripts/run_seed29_stability_eval_rtx4000ada.sh"

"${SSH[@]}" "$REMOTE" \
  "chmod +x '$REMOTE_ROOT/scripts/run_seed29_stability_eval_rtx4000ada.sh'; \
   mkdir -p '$PERSISTENT_ROOT/launcher'; \
   if test -f '$EVAL_ROOT/EVALUATION_COMPLETE'; then \
     printf '%s\\n' 'evaluation already complete'; \
   elif pgrep -f '[r]un_seed29_stability_eval_rtx4000ada.sh' >/dev/null; then \
     printf '%s\\n' 'evaluation already running'; \
   else \
     cd '$REMOTE_ROOT'; \
     nohup bash scripts/run_seed29_stability_eval_rtx4000ada.sh \
       >'$PERSISTENT_ROOT/launcher/seed29_eval.nohup' 2>&1 </dev/null & \
     echo \$! >'$PERSISTENT_ROOT/launcher/seed29_eval.pid'; \
   fi"

missing_eval_checks=0
while ! "${SSH[@]}" "$REMOTE" "test -f '$EVAL_ROOT/EVALUATION_COMPLETE'"; do
  if "${SSH[@]}" "$REMOTE" \
    "pgrep -f '[r]un_seed29_stability_eval_rtx4000ada.sh' >/dev/null"; then
    missing_eval_checks=0
  else
    missing_eval_checks=$((missing_eval_checks + 1))
    if [[ "$missing_eval_checks" -ge 3 ]]; then
      printf '%s\n' "evaluation stopped without EVALUATION_COMPLETE" >&2
      "${SSH[@]}" "$REMOTE" \
        "tail -80 '$PERSISTENT_ROOT/launcher/seed29_eval.nohup'" >&2 || true
      exit 7
    fi
  fi
  sleep "$POLL_SECONDS"
done

printf '%s\n' "seed29 evaluation completed; live orders remain disabled"
