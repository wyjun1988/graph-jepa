#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="$ROOT/.venv-mps/bin/python"
REPORT_DIR="reports/jepa_multiseed_preflight_m1pro_20260714"
LOG_PATH="ops/training/jepa_multiseed_preflight_m1pro_20260714.log"
PID_PATH="ops/training/jepa_multiseed_preflight_m1pro_20260714.pid"

cd "$ROOT"
mkdir -p "$REPORT_DIR" ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "M1 Pro preflight already running: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup caffeinate -dimsu bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "M1 Pro preflight started: $!"
  exit 0
fi

export PYTHONUNBUFFERED=1

status=0
{
  date '+%Y-%m-%dT%H:%M:%S%z'
  system_profiler SPHardwareDataType | head -20
  "$PYTHON" -c 'import torch; print(torch.__version__); print("mps", torch.backends.mps.is_available())'
  shasum -a 256 \
    stock_v2/graph_jepa.py \
    stock_v2/latent_path_head.py \
    scripts/run_walk_forward_node_eval.py \
    scripts/evaluate_node_prediction.py \
    scripts/benchmark_direct_state_mlp.py \
    scripts/gate_shadow_candidate.py \
    data/universes/krx500_pit_20191231.json \
    data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl \
    data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl
} > "$REPORT_DIR/preflight.txt" 2>&1 || status=$?

if [[ "$status" -eq 0 ]]; then
  "$PYTHON" -m pytest -q tests > "$REPORT_DIR/pytest.txt" 2>&1 || status=$?
fi

printf '%s\n' "$status" > "$REPORT_DIR/exit_status.txt"
date '+%Y-%m-%dT%H:%M:%S%z' > "$REPORT_DIR/FINISHED_AT"
exit "$status"
