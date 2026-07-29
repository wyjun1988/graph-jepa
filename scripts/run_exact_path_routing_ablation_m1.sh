#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="/Users/wooyeol/work/stock/venv/bin/python"
REPORT_ROOT="reports/exact_path_routing_ablation_m1_20260713"
MODEL_ROOT="models/exact_path_routing_ablation_m1_20260713"
PANEL_SHA="00cbf81903f5c4bba5ef6ddc39e8a243c63b8445b8ed90bc1e8cea7f2ada630e"
EDGE_SHA="c66077ecbc91c3996204dfb95b1b5e12b2542ac143036c8659d3363519669c2b"

cd "$ROOT"
mkdir -p "$REPORT_ROOT" "$MODEL_ROOT"
export PYTHONUNBUFFERED=1

run_candidate() {
  local name="$1"
  local temporal_scale="$2"
  local stock_edge_scale="$3"
  local report_dir="$REPORT_ROOT/$name"
  local model_dir="$MODEL_ROOT/$name"
  if [[ -f "$report_dir/CANDIDATE_COMPLETE" ]]; then
    echo "candidate already complete: $name"
    return
  fi

  "$PYTHON" scripts/run_real_backtest.py \
    --start 2020-01-01 \
    --end 2024-12-30 \
    --train-end 2023-12-29 \
    --epochs 6 \
    --hidden-dim 256 \
    --layers 4 \
    --train-batch-size 8 \
    --snapshot-workers 8 \
    --device mps \
    --seed 1907 \
    --skip-return-backtest \
    --reports-dir "$report_dir/train" \
    --models-dir "$model_dir" \
    --training-manifest-schema-version 4 \
    --expected-training-manifest-sha256 "$PANEL_SHA" \
    --expected-training-edge-manifest-sha256 "$EDGE_SHA" \
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
    --edge-top-k 6 \
    --edge-correlation-mode signed \
    --graph-neighbor-scale 1.0 \
    --temporal-graph-neighbor-scale "$temporal_scale" \
    --temporal-stock-edge-scale "$stock_edge_scale" \
    --lr 0.0003 \
    --ema-decay 0.9995 \
    --state-loss-weight 1.0 \
    --current-imputation-loss-weight 1.0 \
    --return-correlation-loss-weight 0.0 \
    --entry-path-correlation-loss-weight 0.025 \
    --latent-loss-weight 0.25 \
    --state-feature-weight return_1d=8 \
    --state-feature-weight return_2d=8 \
    --state-feature-weight return_3d=8 \
    --state-feature-weight return_5d=8 \
    --state-feature-weight return_10d=8 \
    --state-feature-weight gap_open=8 \
    --state-feature-weight intraday_return=8 \
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
    --mask-strategy mixed \
    --partial-corr-top-k 0 \
    --lead-lag-top-k 0 \
    --policy-rate-edge-scale 0.0 \
    --event-edge-top-k 0 \
    --temporal-exclude-feature-prefix news_ \
    --temporal-exclude-feature-prefix fund_

  "$PYTHON" scripts/evaluate_node_prediction.py \
    --model-dir "$model_dir" \
    --output-dir "$report_dir/node_eval" \
    --horizons 1,2,3,5,10 \
    --mask-strategy mixed \
    --max-steps 180 \
    --edge-cache-workers 8 \
    --device mps \
    --seed 1907 \
    --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
    --external-cache-dir data/external_cache
  touch "$report_dir/CANDIDATE_COMPLETE"
}

run_candidate global0 0.0 1.0
run_candidate external_only 1.0 0.0

"$PYTHON" scripts/select_path_objective_candidates.py \
  --candidate "global0=$REPORT_ROOT/global0/node_eval/global0/summary.json" \
  --candidate "external_only=$REPORT_ROOT/external_only/node_eval/external_only/summary.json" \
  --output-dir "$REPORT_ROOT/selection"
touch "$REPORT_ROOT/ABLATION_COMPLETE"
