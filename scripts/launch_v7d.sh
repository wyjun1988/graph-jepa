#!/bin/bash
# v7d: return-weight pilot (intent 1).
# Contract: configs/pilot-v7d-return-weight-v1-20260716.json
#   SHA-256 11046e7ef3d7b9855cf11d355134fe5de618ab804f3a23568cfda50efb75a287
#
# Arm 1 drops the 12x return weighting entirely (uniform); arm 2 reduces it to 4.
# Everything else is the exact v6 fold3 command, including the news_/fund_ target
# exclusion -- v7c tests that separately.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python

base_args() {
  echo "--start 2020-01-01 --end 2024-11-05 --train-end 2024-01-03 --universe krx \
--max-tickers 500 --epochs 24 --hidden-dim 1024 --layers 10 --horizon 10 --top-k 5 \
--edge-top-k 6 --graph-neighbor-scale 1.0 --lr 0.0003 --ema-decay 0.9995 \
--latent-loss-weight 0.25 --state-loss-weight 1.0 --current-imputation-loss-weight 1.0 \
--return-correlation-loss-weight 0.0 --entry-path-correlation-loss-weight 0.05 \
--downstream-auxiliary-loss-weight 0.0 --downstream-path-weight 1.0 \
--downstream-mfe-weight 0.25 --downstream-mae-weight 0.25 --downstream-volatility-weight 1.0 \
--downstream-market-loss-weight 0.0 --downstream-market-cost-bps 50.0 \
--downstream-transition-loss-weight 0.1 --downstream-transition-pooling robust_projected \
--temporal-impact-loss-mix 0.0 --temporal-state-mode horizon_residual_heads \
--temporal-residual-short-steps 2 --pretrain-task temporal --temporal-offset 10 \
--latent-rollout-steps 10 --rollout-offsets 1,2,3,5,10 --mask-strategy mixed \
--train-batch-size 16 --snapshot-workers 16 --amp-dtype bfloat16 --max-train-steps 0 \
--path-horizons 1,2,3,5,10 --device cuda --seed 17 \
--cache-dir data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv \
--training-manifest-schema-version 4 --skip-return-backtest --min-train-rows 1 \
--checkpoint-epochs 8,16 --temporal-graph-neighbor-scale 0.0 --temporal-stock-edge-scale 1.0 \
--rollout-loss-weights 2,2,1,1,1 \
--temporal-exclude-feature-prefix news_ --temporal-exclude-feature-prefix fund_ \
--temporal-state-context-skip --normalize-predictor-output \
--universe-manifest data/universes/krx500_pit_20191231.json \
--edge-correlation-mode signed --partial-corr-top-k 0 --partial-corr-min-abs 0.1 \
--partial-corr-mode signed --partial-corr-scale 0.5 --lead-lag-top-k 0 --lead-lag-days 1 \
--lead-lag-min-abs-corr 0.08 --lead-lag-mode signed --lead-lag-scale 0.5 \
--policy-rate-edge-scale 0.0 --factor-sensitivity-top-k 0 \
--event-edge-top-k 0 --event-edge-min-weight 0.05 \
--event-edge-scale 0.25 --industry-prefix-length 2 --industry-edge-scale 0.2 \
--event-path data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl \
--event-coverage-mode mask_uncovered --require-event-sensors --min-event-coverage 0.99 \
--fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
--fundamental-lag-days 1 --require-fundamental-sensors --min-fundamental-coverage 0.79 \
--investor-cache-dir data/kiwoom_investor_cache --investor-flow-lag-days 1 \
--require-investor-sensors --min-investor-coverage 0.95 --external-preset kr_global_rates \
--require-all-external-factors --external-node-mode nodes --external-lag-days 1 \
--external-cache-dir data/external_cache"
}

echo "=== ARM 1/2: uniform (no return weighting)  $(date -u) ==="
$PY scripts/run_real_backtest.py $(base_args) \
  --reports-dir reports/pilot_v7d_uniform_weight_seed17_20260716 \
  --models-dir models/pilot_v7d_uniform_weight_seed17_20260716 \
  > logs/pilot_v7d_uniform_weight_seed17_20260716.log 2>&1
echo "uniform exit=$? $(date -u)"

echo "=== ARM 2/2: return weight 4  $(date -u) ==="
$PY scripts/run_real_backtest.py $(base_args) \
  --state-feature-weight return_1d=4 --state-feature-weight return_2d=4 \
  --state-feature-weight return_3d=4 --state-feature-weight return_5d=4 \
  --state-feature-weight return_10d=4 --state-feature-weight gap_open=4 \
  --state-feature-weight intraday_return=4 \
  --reports-dir reports/pilot_v7d_weight4_seed17_20260716 \
  --models-dir models/pilot_v7d_weight4_seed17_20260716 \
  > logs/pilot_v7d_weight4_seed17_20260716.log 2>&1
echo "weight4 exit=$? $(date -u)"

touch reports/PILOT_V7D_BOTH_ARMS_COMPLETE
echo "=== BOTH ARMS DONE $(date -u) ==="
