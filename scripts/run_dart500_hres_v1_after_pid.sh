#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 WAIT_PID" >&2
  exit 2
fi

WAIT_PID="$1"
ROOT="/workspace/stock-v2"
PYTHON="/root/venvs/news-vllm-cu128/bin/python"
RUN_NAME="strict_causal453_hres_v2_seed17"
REPORTS_ROOT="reports/walk_forward_causal453_hres_v2_20260713"
MODELS_ROOT="models/walk_forward_causal453_hres_v2_20260713"
DRIVER_LOG="reports/causal453_hres_v2_driver_20260713.log"
EVENT_PATH="data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl"
EVENT_SHA256="70eb7a753855de89bdc19607683d902547bf7ef9421415a784dbae37838107a7"
OHLCV_CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
OHLCV_MANIFEST="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/manifest.json"
OHLCV_MANIFEST_SHA256="cb87aec3cc0b9cbf76fff1886430549b3d1eb30ced7350c755b449da07fefd18"

cd "$ROOT"
mkdir -p reports "$REPORTS_ROOT" "$MODELS_ROOT"
exec > >(tee -a "$DRIVER_LOG") 2>&1

echo "[$(date -Is)] waiting for GPU pipeline pid=$WAIT_PID"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 60
done

echo "[$(date -Is)] GPU pipeline exited; validating causal inputs"
printf '%s  %s\n' "$EVENT_SHA256" "$EVENT_PATH" | sha256sum -c -
printf '%s  %s\n' "$OHLCV_MANIFEST_SHA256" "$OHLCV_MANIFEST" | sha256sum -c -
"$PYTHON" -m py_compile \
  stock_v2/data_contract.py \
  stock_v2/graph_jepa.py \
  scripts/audit_causal_ohlcv_release.py \
  scripts/run_real_backtest.py \
  scripts/run_walk_forward_node_eval.py \
  scripts/evaluate_node_prediction.py \
  scripts/benchmark_direct_state_mlp.py \
  scripts/compare_direct_state_mlp.py \
  scripts/gate_shadow_candidate.py
"$PYTHON" scripts/audit_causal_ohlcv_release.py \
  --manifest "$OHLCV_MANIFEST" \
  --output reports/ohlcv_causal453_release_audit_20260713.json \
  --min-tickers 450 \
  --min-rows 1500
nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader

export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "[$(date -Is)] starting strict two-fold 500-stock Graph-JEPA"
"$PYTHON" scripts/run_walk_forward_node_eval.py \
  --name "$RUN_NAME" \
  --fold 2023-12-29:2024-12-30 \
  --fold 2024-12-30:2026-07-10 \
  --start 2020-01-01 \
  --universe krx \
  --universe-manifest data/universes/krx500_pit_20191231.json \
  --max-tickers 500 \
  --epochs 24 \
  --checkpoint-epochs 8,16 \
  --hidden-dim 1024 \
  --layers 10 \
  --horizon 10 \
  --top-k 5 \
  --edge-top-k 6 \
  --edge-correlation-mode signed \
  --lr 0.0003 \
  --ema-decay 0.9995 \
  --state-loss-weight 1.0 \
  --current-imputation-loss-weight 1.0 \
  --return-correlation-loss-weight 0.0 \
  --normalize-predictor-output \
  --temporal-state-mode horizon_residual_heads \
  --temporal-offset 10 \
  --latent-rollout-steps 10 \
  --rollout-offsets 1,2,3,5,10 \
  --rollout-loss-weights 2,2,1,1,1 \
  --path-horizons 1,2,3,5,10 \
  --mask-strategy mixed \
  --train-batch-size 8 \
  --snapshot-workers 16 \
  --device cuda \
  --eval-device cuda \
  --max-steps 0 \
  --seed 17 \
  --event-path "$EVENT_PATH" \
  --event-coverage-mode mask_uncovered \
  --require-event-sensors \
  --min-event-coverage 0.95 \
  --fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
  --fundamental-lag-days 1 \
  --require-fundamental-sensors \
  --min-fundamental-coverage 0.86 \
  --investor-cache-dir data/kiwoom_investor_cache \
  --investor-flow-lag-days 1 \
  --require-investor-sensors \
  --min-investor-coverage 0.89 \
  --external-preset kr_global_rates \
  --external-node-mode nodes \
  --external-lag-days 1 \
  --cache-dir "$OHLCV_CACHE" \
  --external-cache-dir data/external_cache \
  --require-all-external-factors \
  --expected-training-manifest-sha256 8be0e0e1f1775459e85681729a41614f83f2e30f5dea6ed97ec721c183c6c74c \
  --expected-training-manifest-sha256 bb61fc283b69e84ca7a9d1798f3971aedbfbe90207fc7a8fcb43bf082ae89835 \
  --expected-training-edge-manifest-sha256 c66077ecbc91c3996204dfb95b1b5e12b2542ac143036c8659d3363519669c2b \
  --expected-training-edge-manifest-sha256 a85f939144cec194ef35fb927603b69bc111ce3bbedf12bcc159757c713d7870 \
  --reports-root "$REPORTS_ROOT" \
  --models-root "$MODELS_ROOT" \
  --summary-output "$REPORTS_ROOT/summary.json"

for fold in \
  "fold1_20231229_to_20241230" \
  "fold2_20241230_to_20260710"; do
  MODEL_NAME="${RUN_NAME}_${fold}"
  MODEL_DIR="${MODELS_ROOT}/${MODEL_NAME}"
  JEPA_DAILY="${REPORTS_ROOT}/node_eval/${MODEL_NAME}/future_rollout.csv"
  DIRECT_DIR="reports/direct_state_mlp_causal453_hres_v2/${MODEL_NAME}"
  DIRECT_NOGRAPH_DIR="reports/direct_state_mlp_causal453_hres_v2_nograph/${MODEL_NAME}"

  echo "[$(date -Is)] direct residual MLP baseline model=$MODEL_NAME"
  "$PYTHON" scripts/benchmark_direct_state_mlp.py \
    --model-dir "$MODEL_DIR" \
    --output-dir "$DIRECT_DIR" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --hidden-dim 512 \
    --layers 3 \
    --epochs 16 \
    --patience 4 \
    --batch-size 16384 \
    --device cuda \
    --cache-dir "$OHLCV_CACHE" \
    --external-cache-dir data/external_cache \
    --context-cache "data/cache/direct_context_${MODEL_NAME}.npy"

  "$PYTHON" scripts/compare_direct_state_mlp.py \
    --direct-daily "$DIRECT_DIR/daily_metrics.csv" \
    --jepa-daily "$JEPA_DAILY" \
    --output-dir "reports/direct_vs_jepa_causal453_hres_v2/${MODEL_NAME}"

  echo "[$(date -Is)] no-graph direct baseline model=$MODEL_NAME"
  "$PYTHON" scripts/benchmark_direct_state_mlp.py \
    --model-dir "$MODEL_DIR" \
    --output-dir "$DIRECT_NOGRAPH_DIR" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --hidden-dim 512 \
    --layers 3 \
    --epochs 16 \
    --patience 4 \
    --batch-size 16384 \
    --device cuda \
    --without-graph \
    --cache-dir "$OHLCV_CACHE" \
    --external-cache-dir data/external_cache \
    --context-cache "data/cache/direct_context_${MODEL_NAME}.npy"
done

echo "[$(date -Is)] applying strict read-only shadow gate"
set +e
"$PYTHON" scripts/gate_shadow_candidate.py \
  --walk-forward-summary "$REPORTS_ROOT/summary.json" \
  --node-summary "$REPORTS_ROOT/node_eval/${RUN_NAME}_fold1_20231229_to_20241230/summary.json" \
  --node-summary "$REPORTS_ROOT/node_eval/${RUN_NAME}_fold2_20241230_to_20260710/summary.json" \
  --direct-comparison "reports/direct_vs_jepa_causal453_hres_v2/${RUN_NAME}_fold1_20231229_to_20241230/comparison.json" \
  --direct-comparison "reports/direct_vs_jepa_causal453_hres_v2/${RUN_NAME}_fold2_20241230_to_20260710/comparison.json" \
  --dataset-audit reports/news_krx500_dart_pit_v2_integrity_20260712.json \
  --ohlcv-audit reports/ohlcv_causal453_release_audit_20260713.json \
  --output-dir reports/shadow_gate_causal453_hres_v2
GATE_STATUS=$?
set -e
echo "[$(date -Is)] shadow gate exit_status=$GATE_STATUS"

touch "$REPORTS_ROOT/PIPELINE_COMPLETE"
echo "[$(date -Is)] strict model and direct-baseline pipeline complete"
