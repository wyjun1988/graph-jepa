#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="/Users/wooyeol/work/stock/venv/bin/python"
REPORTS_ROOT="reports/dart_event_edge_ablation_m1_20260713"
MODELS_ROOT="models/dart_event_edge_ablation_m1_20260713"

cd "$ROOT"
mkdir -p "$REPORTS_ROOT" "$MODELS_ROOT"

run_variant() {
  local name="$1"
  local event_top_k="$2"

  "$PYTHON" scripts/run_walk_forward_node_eval.py \
    --name "$name" \
    --fold 2023-12-29:2024-12-30 \
    --start 2020-01-01 \
    --universe krx \
    --universe-manifest data/universes/krx500_pit_20191231.json \
    --max-tickers 500 \
    --epochs 6 \
    --hidden-dim 256 \
    --layers 4 \
    --horizon 10 \
    --top-k 5 \
    --edge-top-k 6 \
    --edge-correlation-mode signed \
    --event-edge-top-k "$event_top_k" \
    --event-edge-min-weight 0.05 \
    --event-edge-scale 0.25 \
    --lr 0.0003 \
    --ema-decay 0.9995 \
    --state-loss-weight 0.35 \
    --return-correlation-loss-weight 0.0 \
    --normalize-predictor-output \
    --temporal-state-mode horizon_residual_heads \
    --temporal-offset 10 \
    --latent-rollout-steps 10 \
    --rollout-offsets 1,2,3,5,10 \
    --rollout-loss-weights 2,2,1,1,1 \
    --path-horizons 1,2,3,5,10 \
    --mask-strategy mixed \
    --train-batch-size 4 \
    --snapshot-workers 8 \
    --device mps \
    --eval-device mps \
    --max-steps 120 \
    --seed 1704 \
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
    --reports-root "$REPORTS_ROOT" \
    --models-root "$MODELS_ROOT" \
    --summary-output "$REPORTS_ROOT/${name}_summary.json"
}

run_variant dart500_no_event_edges_h256_l4_e6 0
run_variant dart500_event_edges_k2_h256_l4_e6 2
touch "$REPORTS_ROOT/ABLATION_COMPLETE"
