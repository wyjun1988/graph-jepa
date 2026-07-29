#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

REMOTE="${REMOTE:-wooyeol@wooyeol}"
REMOTE_ROOT="${REMOTE_ROOT:-/Users/wooyeol/work/stock-v2}"
POLL_SEC="${POLL_SEC:-120}"

PIPELINE_PATTERN="${PIPELINE_PATTERN:-run_event_edge_v2_after_qwen.sh}"
LATENCY_OUT="${LATENCY_OUT:-reports/latency/event_edge_v2_krx100_h384_l5_e8_mps.json}"

echo "[$(date)] waiting for M1 Pro event-edge v2 pipeline"
while ssh "${REMOTE}" "pgrep -f '${PIPELINE_PATTERN}' >/dev/null"; do
  echo "[$(date)] M1 Pro v2 pipeline still running"
  sleep "${POLL_SEC}"
done

echo "[$(date)] M1 Pro pipeline process ended; verifying completion artifact"
ssh "${REMOTE}" "test -f '${REMOTE_ROOT}/${LATENCY_OUT}'"

echo "[$(date)] pulling M1 Pro event-edge v2 artifacts"
mkdir -p data/events reports/latency reports/node_prediction_eval_event_edge_v2 models
rsync -az "${REMOTE}:${REMOTE_ROOT}/data/events/news_calibration_qwen_v2_800_krx100_20200101_20260710.jsonl" data/events/
rsync -az "${REMOTE}:${REMOTE_ROOT}/data/events/news_backfill_qwen_calibrated_v2_krx100_20200101_20260710.jsonl" data/events/
rsync -az "${REMOTE}:${REMOTE_ROOT}/reports/news_qwen_calibration_v2_800_report.json" reports/
rsync -az "${REMOTE}:${REMOTE_ROOT}/reports/event_edge_v2_krx100_h384_l5_e8" reports/
rsync -az "${REMOTE}:${REMOTE_ROOT}/reports/node_prediction_eval_event_edge_v2" reports/
rsync -az "${REMOTE}:${REMOTE_ROOT}/reports/latency/event_edge_v2_krx100_h384_l5_e8_mps.json" reports/latency/
rsync -az "${REMOTE}:${REMOTE_ROOT}/models/event_edge_v2_krx100_h384_l5_e8" models/

echo "[$(date)] M1 Pro event-edge v2 artifacts copied to iMac"
