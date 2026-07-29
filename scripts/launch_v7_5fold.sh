#!/bin/bash
# v7 five-fold: news_ added to the temporal targets (intent 1).
# Contract: configs/rolling-v7-news-targets-v1-20260717.json
#   SHA-256 fcc175c41bb24ad0c8df0ea6c8351653a08951025891bdd2f35aa539a77e3bdb
#
# Only --temporal-exclude-feature-prefix changes versus v6: fund_ only, news_ removed.
# Preflight (--edge-manifest-only) runs first for every fold so a data or edge
# mismatch is caught before any GPU is spent on training.
# Waits for the GPU so it queues behind v7d and the W5 sweep.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v7_news_targets_seed17_20260717

fold_args() {  # $1=end $2=train_end $3=tag
  echo "--start 2020-01-01 --end $1 --train-end $2 --universe krx \
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
--state-feature-weight return_1d=12 --state-feature-weight return_2d=12 \
--state-feature-weight return_3d=12 --state-feature-weight return_5d=12 \
--state-feature-weight return_10d=12 --state-feature-weight gap_open=12 \
--state-feature-weight intraday_return=12 \
--temporal-exclude-feature-prefix fund_ \
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
--external-cache-dir data/external_cache \
--reports-dir reports/${RUN}/$3 --models-dir models/${RUN}/$3"
}

FOLDS=("2023-03-06:2022-05-09:r1" "2024-01-03:2023-03-06:r2" "2024-11-05:2024-01-03:r3" \
       "2025-09-05:2024-11-05:r4" "2026-07-10:2025-09-05:r5")

echo "=== waiting for the GPU (v7d, then W5) ==="
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 2000 ]; do sleep 120; done
echo "GPU free at $(date -u)"

echo "=== PREFLIGHT: edge manifests for all five folds ==="
for f in "${FOLDS[@]}"; do
  IFS=: read END TE TAG <<< "$f"
  $PY scripts/run_real_backtest.py $(fold_args "$END" "$TE" "preflight_$TAG") --edge-manifest-only \
    > "logs/${RUN}_preflight_${TAG}.log" 2>&1
  RC=$?
  echo "preflight $TAG exit=$RC  $(grep -oE 'training (data manifest|edge cache):.*' logs/${RUN}_preflight_${TAG}.log | tr '\n' ' ')"
  if [ $RC -ne 0 ]; then echo "=== PREFLIGHT FAILED on $TAG — aborting before training ==="; tail -20 "logs/${RUN}_preflight_${TAG}.log"; exit 1; fi
done
echo "=== preflight complete; five folds verified ==="

for f in "${FOLDS[@]}"; do
  IFS=: read END TE TAG <<< "$f"
  echo "--- TRAIN $TAG  $(date -u) ---"
  $PY scripts/run_real_backtest.py $(fold_args "$END" "$TE" "$TAG") > "logs/${RUN}_${TAG}.log" 2>&1
  echo "$TAG exit=$?  $(date -u)"
done

touch reports/${RUN}/ALL_FOLDS_COMPLETE
echo "=== V7 FIVE-FOLD DONE $(date -u) ==="
