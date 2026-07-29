#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
SEED29_POST="reports/walk_forward_causal453_path_multiseed_seed29_20260714/postprocess_latent_head"
LOG_PATH="ops/training/jepa_seed53_after_seed29_20260714.log"
PID_PATH="ops/training/jepa_seed53_after_seed29_20260714.pid"

cd "$ROOT"
mkdir -p ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "seed 53 queue already running: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "seed 53 queued after seed 29 postprocess: $!"
  exit 0
fi

while [[ ! -f "$SEED29_POST/POSTPROCESS_COMPLETE" && ! -f "$SEED29_POST/POSTPROCESS_FAILED" ]]; do
  sleep 30
done

if [[ -f "$SEED29_POST/POSTPROCESS_FAILED" ]]; then
  echo "seed 29 postprocess failed; preserving GPU utilization by continuing to seed 53"
fi

exec bash scripts/run_jepa_multiseed_seed53_a5000.sh --worker
