#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
MODEL_DIR="models/lead_lag_v2_500_krx100_h384_l5_e8"
REPORT_DIR="reports/lead_lag_v2_500_krx100_h384_l5_e8"
NODE_EVAL_DIR="reports/node_prediction_eval_lead_lag_v2_500"
LATENCY_OUT="reports/latency/lead_lag_v2_500_krx100_h384_l5_e8_mps.json"
EVENT_PATH="data/events/news_backfill_qwen_calibrated_v2_500_krx100_20200101_20260710.jsonl"

mkdir -p ops/training reports/latency

echo "[$(date)] lead_lag_v2_500 M1 Pro run started"
echo "python=$PYTHON"
echo "model_dir=$MODEL_DIR"
echo "event_path=$EVENT_PATH"

set +e
"$PYTHON" scripts/run_real_backtest.py \
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
  --event-path "$EVENT_PATH" \
  --edge-correlation-mode none \
  --edge-top-k 0 \
  --lead-lag-top-k 4 \
  --lead-lag-mode abs \
  --lead-lag-days 1 \
  --lead-lag-min-abs-corr 0.06 \
  --lead-lag-scale 0.50 \
  --device mps \
  --reports-dir "$REPORT_DIR" \
  --models-dir "$MODEL_DIR"
TRAIN_RC=$?
set -e

echo "[$(date)] run_real_backtest exited rc=$TRAIN_RC"

if [[ -f "$MODEL_DIR/graph_jepa_real.pt" ]]; then
  echo "[$(date)] checkpoint exists; running node prediction eval"
  "$PYTHON" scripts/evaluate_node_prediction.py \
    --model-dir "$MODEL_DIR" \
    --output-dir "$NODE_EVAL_DIR" \
    --max-steps 180 \
    --device mps \
    --horizons 1,3,5,10

  echo "[$(date)] running latency measurement"
  "$PYTHON" scripts/measure_jepa_latency.py \
    --model-dir "$MODEL_DIR" \
    --device mps \
    --cycles 50 \
    --warmup 5 \
    --rollout-steps 10 \
    --output "$LATENCY_OUT"
else
  echo "[$(date)] checkpoint missing; skipping node eval and latency"
  exit "$TRAIN_RC"
fi

echo "[$(date)] lead_lag_v2_500 M1 Pro run complete"
