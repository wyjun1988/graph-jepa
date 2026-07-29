#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/stock-v2"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
RUN_NAME="final_aligned_jepa_market_head_v1_seed17"
FOLD_NAME="${RUN_NAME}_fold1_20231229_to_20241230"
REPORTS_DIR="reports/${RUN_NAME}/${FOLD_NAME}"
MODELS_DIR="models/${RUN_NAME}/${FOLD_NAME}"

cd "$ROOT"
mkdir -p "$REPORTS_DIR" "$MODELS_DIR"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

on_error() {
  local status=$?
  printf '{"status":"failed","exit_status":%d,"live_orders_allowed":false}\n' \
    "$status" > "$REPORTS_DIR/PIPELINE_FAILED"
  exit "$status"
}
trap on_error ERR

printf '%s\n' \
  '{"scope":"research_only","live_orders_allowed":false,"purpose":"absolute market head Fold1 development","fold2_opened":false,"market_target":"mean liquid-stock next-open path return and 50bps exceedance"}' \
  > "$REPORTS_DIR/experiment_contract.json"

"$PYTHON_BIN" scripts/run_real_backtest.py \
  --start 2020-01-01 \
  --end 2024-12-30 \
  --train-end 2023-12-29 \
  --universe krx \
  --universe-manifest data/universes/krx500_pit_20191231.json \
  --max-tickers 500 \
  --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
  --event-path data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl \
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
  --external-cache-dir data/external_cache \
  --require-all-external-factors \
  --horizon 10 \
  --top-k 5 \
  --epochs 24 \
  --hidden-dim 1024 \
  --layers 10 \
  --edge-top-k 6 \
  --edge-correlation-mode signed \
  --graph-neighbor-scale 1.0 \
  --temporal-graph-neighbor-scale 0.0 \
  --temporal-stock-edge-scale 1.0 \
  --lr 0.0003 \
  --ema-decay 0.9995 \
  --latent-loss-weight 0.25 \
  --state-loss-weight 1.0 \
  --current-imputation-loss-weight 1.0 \
  --normalize-predictor-output \
  --temporal-state-mode horizon_residual_heads \
  --temporal-state-context-skip \
  --temporal-residual-short-steps 2 \
  --pretrain-task temporal \
  --temporal-offset 10 \
  --latent-rollout-steps 10 \
  --rollout-offsets 1,2,3,5,10 \
  --rollout-loss-weights 2,2,1,1,1 \
  --path-horizons 1,2,3,5,10 \
  --mask-strategy operational_mixed \
  --partial-corr-top-k 0 \
  --lead-lag-top-k 0 \
  --policy-rate-edge-scale 0.0 \
  --event-edge-top-k 0 \
  --temporal-exclude-feature-prefix news_ \
  --temporal-exclude-feature-prefix fund_ \
  --return-correlation-loss-weight 0.0 \
  --entry-path-correlation-loss-weight 0.05 \
  --downstream-auxiliary-loss-weight 0.10 \
  --downstream-path-weight 1.0 \
  --downstream-mfe-weight 0.25 \
  --downstream-mae-weight 0.25 \
  --downstream-volatility-weight 1.0 \
  --downstream-market-loss-weight 0.10 \
  --downstream-market-cost-bps 50 \
  --state-feature-weight return_1d=12 \
  --state-feature-weight return_2d=12 \
  --state-feature-weight return_3d=12 \
  --state-feature-weight return_5d=12 \
  --state-feature-weight return_10d=12 \
  --state-feature-weight gap_open=12 \
  --state-feature-weight intraday_return=12 \
  --train-batch-size 28 \
  --snapshot-workers 16 \
  --device cuda \
  --seed 17 \
  --expected-training-manifest-sha256 00cbf81903f5c4bba5ef6ddc39e8a243c63b8445b8ed90bc1e8cea7f2ada630e \
  --expected-training-edge-manifest-sha256 c66077ecbc91c3996204dfb95b1b5e12b2542ac143036c8659d3363519669c2b \
  --training-manifest-schema-version 4 \
  --skip-return-backtest \
  --reports-dir "$REPORTS_DIR" \
  --models-dir "$MODELS_DIR" \
  2>&1 | tee "$REPORTS_DIR/train.log"

touch "$REPORTS_DIR/TRAINING_COMPLETE"
