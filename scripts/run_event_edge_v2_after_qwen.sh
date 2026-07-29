#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

PYTHON_BIN="${PYTHON_BIN:-.venv/bin/python}"
POLL_SEC="${POLL_SEC:-60}"
TARGET_ROWS="${TARGET_ROWS:-800}"

BASE_EVENTS="${BASE_EVENTS:-data/events/news_backfill_google_krx100_20200101_20260710.jsonl}"
QWEN_SAMPLE="${QWEN_SAMPLE:-data/events/news_calibration_qwen_v2_800_krx100_20200101_20260710.jsonl}"
CALIBRATED_EVENTS="${CALIBRATED_EVENTS:-data/events/news_backfill_qwen_calibrated_v2_krx100_20200101_20260710.jsonl}"
CALIBRATION_REPORT="${CALIBRATION_REPORT:-reports/news_qwen_calibration_v2_800_report.json}"

MODEL_DIR="${MODEL_DIR:-models/event_edge_v2_krx100_h384_l5_e8}"
REPORTS_DIR="${REPORTS_DIR:-reports/event_edge_v2_krx100_h384_l5_e8}"
NODE_EVAL_DIR="${NODE_EVAL_DIR:-reports/node_prediction_eval_event_edge_v2}"
LATENCY_OUT="${LATENCY_OUT:-reports/latency/event_edge_v2_krx100_h384_l5_e8_mps.json}"

echo "[$(date)] waiting for Qwen v2 sample: ${QWEN_SAMPLE}"
while true; do
  rows=0
  if [[ -f "${QWEN_SAMPLE}" ]]; then
    rows="$(wc -l < "${QWEN_SAMPLE}" | tr -d ' ')"
  fi
  echo "[$(date)] qwen_rows=${rows}/${TARGET_ROWS}"
  if [[ "${rows}" -ge "${TARGET_ROWS}" ]]; then
    break
  fi
  if ! pgrep -f "rescore_news_events_qwen.py --input data/events/news_calibration_candidates_v2_800" >/dev/null; then
    echo "[$(date)] Qwen rescore process stopped before target rows: ${rows}/${TARGET_ROWS}" >&2
    exit 2
  fi
  sleep "${POLL_SEC}"
done

echo "[$(date)] Qwen v2 sample complete; stopping Qwen server to free memory"
pgrep -f "llama_cpp.server" | xargs kill -TERM 2>/dev/null || true
sleep 5

echo "[$(date)] applying Qwen v2 calibration"
"${PYTHON_BIN}" scripts/apply_news_score_calibration.py \
  --base "${BASE_EVENTS}" \
  --qwen-sample "${QWEN_SAMPLE}" \
  --output "${CALIBRATED_EVENTS}" \
  --report "${CALIBRATION_REPORT}"

echo "[$(date)] training event-edge v2 KRX100 model"
"${PYTHON_BIN}" scripts/run_real_backtest.py \
  --universe krx \
  --max-tickers 100 \
  --start 2020-01-01 \
  --train-end 2023-12-29 \
  --horizon 5 \
  --top-k 5 \
  --epochs 8 \
  --hidden-dim 384 \
  --layers 5 \
  --hide-ratio 0.35 \
  --mask-strategy mixed \
  --pretrain-task temporal \
  --temporal-offset 5 \
  --latent-rollout-steps 5 \
  --rollout-offsets 3,5,10 \
  --path-horizons 1,3,5,10 \
  --lr 0.0002 \
  --state-loss-weight 0.40 \
  --event-path "${CALIBRATED_EVENTS}" \
  --event-edge-top-k 4 \
  --event-edge-scale 0.25 \
  --event-edge-min-weight 0.05 \
  --event-edge-max-themes 96 \
  --event-edge-min-theme-count 2 \
  --device mps \
  --reports-dir "${REPORTS_DIR}" \
  --models-dir "${MODEL_DIR}"

echo "[$(date)] evaluating node-state prediction"
"${PYTHON_BIN}" scripts/evaluate_node_prediction.py \
  --model-dir "${MODEL_DIR}" \
  --output-dir "${NODE_EVAL_DIR}" \
  --max-steps 180 \
  --device mps \
  --horizons 1,3,5,10

echo "[$(date)] measuring latency"
"${PYTHON_BIN}" scripts/measure_jepa_latency.py \
  --model-dir "${MODEL_DIR}" \
  --device mps \
  --cycles 50 \
  --warmup 5 \
  --rollout-steps 10 \
  --output "${LATENCY_OUT}"

echo "[$(date)] event-edge v2 pipeline complete"
