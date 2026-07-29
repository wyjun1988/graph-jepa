#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON_BIN="${PYTHON:-/workspace/venvs/stock-v2-cu128/bin/python}"
RUN_NAME="${RUN_NAME:-broad_transition_aux_preflight_h64_seed1701_20260714}"
DEVICE="${DEVICE:-cuda}"
POOLING="${POOLING:-mean}"
TARGET_VERSION="${TARGET_VERSION:-market_transition_v5_systemic_impact_20260714}"
SNAPSHOT_WORKERS="${SNAPSHOT_WORKERS:-16}"
REPORTS="reports/$RUN_NAME"
MODELS="models/$RUN_NAME"
LOG="ops/training/${RUN_NAME}.log"

cd "$ROOT"
mkdir -p "$REPORTS" "$MODELS" "$(dirname "$LOG")"
printf \
  '{"scope":"research_preflight_only","target":"%s","transition_pooling":"%s","test_used_for_selection":false,"live_orders_allowed":false}\n' \
  "$TARGET_VERSION" "$POOLING" \
  > "$REPORTS/experiment_contract.json"

"$PYTHON_BIN" scripts/run_real_backtest.py \
  --start 2020-01-01 \
  --end 2024-12-30 \
  --train-end 2023-12-29 \
  --universe krx \
  --universe-manifest data/universes/krx500_pit_20191231.json \
  --max-tickers 500 \
  --horizon 10 \
  --top-k 5 \
  --epochs 1 \
  --hidden-dim 64 \
  --layers 2 \
  --hide-ratio 0.30 \
  --mask-strategy mixed \
  --edge-window 60 \
  --edge-top-k 6 \
  --min-abs-corr 0.20 \
  --edge-correlation-mode signed \
  --graph-neighbor-scale 1.0 \
  --temporal-graph-neighbor-scale 0.0 \
  --temporal-stock-edge-scale 1.0 \
  --partial-corr-top-k 0 \
  --lead-lag-top-k 0 \
  --policy-rate-edge-scale 0.0 \
  --lr 0.0003 \
  --train-batch-size 16 \
  --snapshot-workers "$SNAPSHOT_WORKERS" \
  --ema-decay 0.9995 \
  --latent-loss-weight 0.25 \
  --state-loss-weight 1.0 \
  --current-imputation-loss-weight 1.0 \
  --entry-path-correlation-loss-weight 0.05 \
  --downstream-transition-loss-weight 0.10 \
  --downstream-transition-pooling "$POOLING" \
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
  --state-feature-weight return_1d=12 \
  --state-feature-weight return_2d=12 \
  --state-feature-weight return_3d=12 \
  --state-feature-weight return_5d=12 \
  --state-feature-weight return_10d=12 \
  --state-feature-weight gap_open=12 \
  --state-feature-weight intraday_return=12 \
  --temporal-exclude-feature-prefix news_ \
  --temporal-exclude-feature-prefix fund_ \
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
  --risk-free-source bok_base_rate \
  --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
  --training-manifest-schema-version 4 \
  --expected-training-manifest-sha256 00cbf81903f5c4bba5ef6ddc39e8a243c63b8445b8ed90bc1e8cea7f2ada630e \
  --expected-training-edge-manifest-sha256 c66077ecbc91c3996204dfb95b1b5e12b2542ac143036c8659d3363519669c2b \
  --device "$DEVICE" \
  --seed 1701 \
  --reports-dir "$REPORTS" \
  --models-dir "$MODELS" \
  --skip-return-backtest \
  2>&1 | tee "$LOG"

touch "$REPORTS/PREFLIGHT_COMPLETE"
