#!/usr/bin/env bash
set -u

cd "$(dirname "$0")/.."

PYTHON="${PYTHON:-.venv/bin/python}"
CANDIDATE="${CANDIDATE:-corr_signed}"
DEVICE="${DEVICE:-mps}"
EVENT_PATH="${EVENT_PATH:-data/events/news_backfill_qwen_calibrated_v2_500_krx100_20200101_20260710.jsonl}"
RUN_SUFFIX="${RUN_SUFFIX:-}"

case "$CANDIDATE" in
  corr_signed)
    MODEL_DIR="models/qwen_v2_500_corr_signed_krx100_h256_l4_e12${RUN_SUFFIX}"
    REPORT_DIR="reports/qwen_v2_500_corr_signed_krx100_h256_l4_e12${RUN_SUFFIX}"
    NODE_EVAL_DIR="reports/node_prediction_eval_qwen_v2_500_corr_signed_h256${RUN_SUFFIX}"
    LATENCY_OUT="reports/latency/qwen_v2_500_corr_signed_krx100_h256_l4_e12${RUN_SUFFIX}_mps.json"
    EDGE_ARGS=(
      --edge-correlation-mode signed
      --edge-top-k 12
      --min-abs-corr 0.2
      --lead-lag-top-k 0
    )
    ;;
  lead_lag_abs)
    MODEL_DIR="models/qwen_v2_500_lead_lag_abs_krx100_h256_l4_e12${RUN_SUFFIX}"
    REPORT_DIR="reports/qwen_v2_500_lead_lag_abs_krx100_h256_l4_e12${RUN_SUFFIX}"
    NODE_EVAL_DIR="reports/node_prediction_eval_qwen_v2_500_lead_lag_abs_h256${RUN_SUFFIX}"
    LATENCY_OUT="reports/latency/qwen_v2_500_lead_lag_abs_krx100_h256_l4_e12${RUN_SUFFIX}_mps.json"
    EDGE_ARGS=(
      --edge-correlation-mode none
      --edge-top-k 0
      --lead-lag-top-k 4
      --lead-lag-mode abs
      --lead-lag-days 1
      --lead-lag-min-abs-corr 0.06
      --lead-lag-scale 0.50
    )
    ;;
  *)
    echo "unknown CANDIDATE=$CANDIDATE; expected corr_signed or lead_lag_abs" >&2
    exit 2
    ;;
esac

mkdir -p ops/training reports/latency

echo "[$(date)] qwen_v2_h256 candidate started"
echo "candidate=$CANDIDATE"
echo "run_suffix=$RUN_SUFFIX"
echo "python=$PYTHON"
echo "model_dir=$MODEL_DIR"
echo "event_path=$EVENT_PATH"

set +e
"$PYTHON" scripts/run_real_backtest.py \
  --universe krx \
  --max-tickers 100 \
  --start 2020-01-01 \
  --end 2026-07-10 \
  --train-end 2023-12-29 \
  --horizon 5 \
  --top-k 10 \
  --epochs 12 \
  --hidden-dim 256 \
  --layers 4 \
  --hide-ratio 0.30 \
  --mask-strategy mixed \
  --pretrain-task temporal \
  --temporal-offset 5 \
  --latent-rollout-steps 5 \
  --rollout-offsets 1,2,3,5 \
  --path-horizons 1,2,3,5,10 \
  --lr 0.0003 \
  --state-loss-weight 0.35 \
  --min-train-rows 850 \
  --event-path "$EVENT_PATH" \
  --device "$DEVICE" \
  --reports-dir "$REPORT_DIR" \
  --models-dir "$MODEL_DIR" \
  "${EDGE_ARGS[@]}"
TRAIN_RC=$?
set -e

echo "[$(date)] run_real_backtest exited rc=$TRAIN_RC"

if [[ -f "$MODEL_DIR/graph_jepa_real.pt" ]]; then
  echo "[$(date)] checkpoint exists; running node prediction eval"
  "$PYTHON" scripts/evaluate_node_prediction.py \
    --model-dir "$MODEL_DIR" \
    --output-dir "$NODE_EVAL_DIR" \
    --max-steps 180 \
    --device "$DEVICE" \
    --horizons 1,2,3,5,10

  echo "[$(date)] running latency measurement"
  "$PYTHON" scripts/measure_jepa_latency.py \
    --model-dir "$MODEL_DIR" \
    --device "$DEVICE" \
    --cycles 50 \
    --warmup 5 \
    --rollout-steps 10 \
    --output "$LATENCY_OUT"
else
  echo "[$(date)] checkpoint missing; skipping node eval and latency"
  exit "$TRAIN_RC"
fi

echo "[$(date)] qwen_v2_h256 candidate complete"
