#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
WAIT_ROOT="reports/walk_forward_causal453_path_multiseed_seed53_20260714/postprocess_latent_head"
LOG_PATH="ops/training/jepa_impact_after_seed53_20260714.log"
PID_PATH="ops/training/jepa_impact_after_seed53_20260714.pid"

cd "$ROOT"
mkdir -p ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "post-seed53 impact queue already running: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "seed 29 and 53 impact jobs queued: $!"
  exit 0
fi

while [[ ! -f "$WAIT_ROOT/POSTPROCESS_COMPLETE" && ! -f "$WAIT_ROOT/POSTPROCESS_FAILED" ]]; do
  sleep 30
done

if [[ -f "$WAIT_ROOT/POSTPROCESS_FAILED" ]]; then
  echo "seed 53 standard postprocess failed; skipping dependent impact jobs" >&2
  exit 4
fi

bash scripts/run_jepa_impact_postprocess.sh 29 --worker
bash scripts/run_jepa_impact_postprocess.sh 53 --worker
