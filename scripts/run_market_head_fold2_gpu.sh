#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/stock-v2"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
MODE="${1:-train}"
RUN_NAME="final_aligned_jepa_market_head_v1_seed17"
FOLD_NAME="${RUN_NAME}_fold2_20241230_to_20251230"
REPORTS_DIR="reports/${RUN_NAME}/${FOLD_NAME}"
MODELS_DIR="models/${RUN_NAME}/${FOLD_NAME}"
EXPECTED_DATA_SHA256="${EXPECTED_DATA_SHA256:-4ae8bdfb8e6f13af77dcb9847974f2c74694768a46667ff9debab136b0f96452}"
EXPECTED_EDGE_SHA256="${EXPECTED_EDGE_SHA256:-a85f939144cec194ef35fb927603b69bc111ce3bbedf12bcc159757c713d7870}"

cd "$ROOT"
mkdir -p "$REPORTS_DIR" "$MODELS_DIR"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

on_error() {
  local status=$?
  printf '{"status":"failed","mode":"%s","exit_status":%d,"live_orders_allowed":false}\n' \
    "$MODE" "$status" > "$REPORTS_DIR/PIPELINE_FAILED"
  exit "$status"
}
trap on_error ERR

case "$MODE" in
  manifest)
    MODE_ARGS=(--manifest-only)
    ;;
  edge-manifest)
    if [[ -z "$EXPECTED_DATA_SHA256" ]]; then
      echo "EXPECTED_DATA_SHA256 is required for edge-manifest mode" >&2
      exit 2
    fi
    MODE_ARGS=(
      --expected-training-manifest-sha256 "$EXPECTED_DATA_SHA256"
      --edge-manifest-only
    )
    ;;
  train)
    if [[ -z "$EXPECTED_DATA_SHA256" || -z "$EXPECTED_EDGE_SHA256" ]]; then
      echo "EXPECTED_DATA_SHA256 and EXPECTED_EDGE_SHA256 are required" >&2
      exit 2
    fi
    MODE_ARGS=(
      --expected-training-manifest-sha256 "$EXPECTED_DATA_SHA256"
      --expected-training-edge-manifest-sha256 "$EXPECTED_EDGE_SHA256"
    )
    ;;
  *)
    echo "usage: $0 [manifest|edge-manifest|train]" >&2
    exit 2
    ;;
esac

printf '%s\n' \
  '{"scope":"research_only","live_orders_allowed":false,"purpose":"frozen h2 policy Fold2 confirmation","development_fold":"2024","confirmation_fold":"2025","policy_horizon":2,"policy_selection":"predicted market return at least roundtrip cost","hyperparameters_frozen":true}' \
  > "$REPORTS_DIR/experiment_contract.json"

ARGS=(
  scripts/run_real_backtest.py
  --start 2020-01-01
  --end 2025-12-30
  --train-end 2024-12-30
  --universe krx
  --universe-manifest data/universes/krx500_pit_20191231.json
  --max-tickers 500
  --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv
  --event-path data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl
  --event-coverage-mode mask_uncovered
  --require-event-sensors
  --min-event-coverage 0.95
  --fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl
  --fundamental-lag-days 1
  --require-fundamental-sensors
  --min-fundamental-coverage 0.86
  --investor-cache-dir data/kiwoom_investor_cache
  --investor-flow-lag-days 1
  --require-investor-sensors
  --min-investor-coverage 0.89
  --external-preset kr_global_rates
  --external-node-mode nodes
  --external-lag-days 1
  --external-cache-dir data/external_cache
  --require-all-external-factors
  --horizon 10
  --top-k 5
  --epochs 24
  --hidden-dim 1024
  --layers 10
  --edge-top-k 6
  --edge-correlation-mode signed
  --graph-neighbor-scale 1.0
  --temporal-graph-neighbor-scale 0.0
  --temporal-stock-edge-scale 1.0
  --lr 0.0003
  --ema-decay 0.9995
  --latent-loss-weight 0.25
  --state-loss-weight 1.0
  --current-imputation-loss-weight 1.0
  --normalize-predictor-output
  --temporal-state-mode horizon_residual_heads
  --temporal-state-context-skip
  --temporal-residual-short-steps 2
  --pretrain-task temporal
  --temporal-offset 10
  --latent-rollout-steps 10
  --rollout-offsets 1,2,3,5,10
  --rollout-loss-weights 2,2,1,1,1
  --path-horizons 1,2,3,5,10
  --mask-strategy operational_mixed
  --partial-corr-top-k 0
  --lead-lag-top-k 0
  --policy-rate-edge-scale 0.0
  --event-edge-top-k 0
  --temporal-exclude-feature-prefix news_
  --temporal-exclude-feature-prefix fund_
  --return-correlation-loss-weight 0.0
  --entry-path-correlation-loss-weight 0.05
  --downstream-auxiliary-loss-weight 0.10
  --downstream-path-weight 1.0
  --downstream-mfe-weight 0.25
  --downstream-mae-weight 0.25
  --downstream-volatility-weight 1.0
  --downstream-market-loss-weight 0.10
  --downstream-market-cost-bps 50
  --state-feature-weight return_1d=12
  --state-feature-weight return_2d=12
  --state-feature-weight return_3d=12
  --state-feature-weight return_5d=12
  --state-feature-weight return_10d=12
  --state-feature-weight gap_open=12
  --state-feature-weight intraday_return=12
  --train-batch-size 28
  --snapshot-workers 16
  --device cuda
  --seed 17
  --training-manifest-schema-version 4
  --skip-return-backtest
  --reports-dir "$REPORTS_DIR"
  --models-dir "$MODELS_DIR"
)

"$PYTHON_BIN" "${ARGS[@]}" "${MODE_ARGS[@]}" 2>&1 | tee "$REPORTS_DIR/${MODE}.log"

rm -f "$REPORTS_DIR/PIPELINE_FAILED"
if [[ "$MODE" == "train" ]]; then
  touch "$REPORTS_DIR/TRAINING_COMPLETE"
fi
