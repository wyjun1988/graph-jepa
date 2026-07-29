#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 NAME CORRELATION_MODE TOP_K [NEIGHBOR_SCALE] [TEMPORAL_NEIGHBOR_SCALE] [EXCLUDED_PREFIX ...]" >&2
  exit 2
fi

NAME="$1"
CORRELATION_MODE="$2"
TOP_K="$3"
NEIGHBOR_SCALE="${4:-1.0}"
TEMPORAL_NEIGHBOR_SCALE="${5:-$NEIGHBOR_SCALE}"
TEMPORAL_STOCK_EDGE_SCALE="${TEMPORAL_STOCK_EDGE_SCALE:-1.0}"
EXCLUDED_PREFIXES=()
if [[ $# -gt 5 ]]; then
  EXCLUDED_PREFIXES=("${@:6}")
fi
case "$CORRELATION_MODE" in
  signed|abs|positive|negative) ;;
  *) echo "unsupported correlation mode: $CORRELATION_MODE" >&2; exit 2 ;;
esac
if [[ ! "$TOP_K" =~ ^[1-9][0-9]*$ ]]; then
  echo "TOP_K must be a positive integer" >&2
  exit 2
fi
if [[ ! "$NEIGHBOR_SCALE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "NEIGHBOR_SCALE must be a non-negative decimal" >&2
  exit 2
fi
if [[ ! "$TEMPORAL_NEIGHBOR_SCALE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "TEMPORAL_NEIGHBOR_SCALE must be a non-negative decimal" >&2
  exit 2
fi
if [[ ! "$TEMPORAL_STOCK_EDGE_SCALE" =~ ^[0-9]+([.][0-9]+)?$ ]]; then
  echo "TEMPORAL_STOCK_EDGE_SCALE must be a non-negative decimal" >&2
  exit 2
fi

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="/Users/wooyeol/work/stock/venv/bin/python"
RUN_ROOT="reports/causal453_edge_candidates_m1_20260713/$NAME"
REPORTS_ROOT="$RUN_ROOT/walk_forward"
MODELS_ROOT="models/causal453_edge_candidates_m1_20260713/$NAME"
PREFLIGHT_DIR="$RUN_ROOT/preflight"
OHLCV_CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
FOLD1_SHA256="8be0e0e1f1775459e85681729a41614f83f2e30f5dea6ed97ec721c183c6c74c"

cd "$ROOT"
if [[ -e "$RUN_ROOT" || -e "$MODELS_ROOT" ]]; then
  echo "refusing to overwrite edge candidate: $NAME" >&2
  exit 3
fi
mkdir -p "$REPORTS_ROOT" "$MODELS_ROOT"

COMMON_SENSOR_ARGS=(
  --start 2020-01-01
  --end 2024-12-30
  --train-end 2023-12-29
  --universe krx
  --universe-manifest data/universes/krx500_pit_20191231.json
  --max-tickers 500
  --horizon 10
  --top-k 5
  --edge-top-k "$TOP_K"
  --edge-correlation-mode "$CORRELATION_MODE"
  --graph-neighbor-scale "$NEIGHBOR_SCALE"
  --temporal-graph-neighbor-scale "$TEMPORAL_NEIGHBOR_SCALE"
  --temporal-stock-edge-scale "$TEMPORAL_STOCK_EDGE_SCALE"
  --event-edge-top-k 0
  --pretrain-task temporal
  --temporal-offset 10
  --latent-rollout-steps 10
  --rollout-offsets 1,2,3,5,10
  --rollout-loss-weights 2,2,1,1,1
  --path-horizons 1,2,3,5,10
  --mask-strategy mixed
  --snapshot-workers 8
  --seed 1704
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
  --cache-dir "$OHLCV_CACHE"
  --external-cache-dir data/external_cache
  --require-all-external-factors
  --expected-training-manifest-sha256 "$FOLD1_SHA256"
)
TEMPORAL_EXCLUDE_ARGS=()
EXCLUDED_PREFIX_CSV=""
if [[ ${#EXCLUDED_PREFIXES[@]} -gt 0 ]]; then
  for prefix in "${EXCLUDED_PREFIXES[@]}"; do
    if [[ -z "$prefix" ]]; then
      echo "excluded feature prefixes must be non-empty" >&2
      exit 2
    fi
    TEMPORAL_EXCLUDE_ARGS+=(--temporal-exclude-feature-prefix "$prefix")
  done
  COMMON_SENSOR_ARGS+=("${TEMPORAL_EXCLUDE_ARGS[@]}")
  EXCLUDED_PREFIX_CSV=$(IFS=,; printf '%s' "${EXCLUDED_PREFIXES[*]}")
fi

"$PYTHON" scripts/run_real_backtest.py \
  "${COMMON_SENSOR_ARGS[@]}" \
  --device cpu \
  --edge-manifest-only \
  --reports-dir "$PREFLIGHT_DIR" \
  --models-dir "$MODELS_ROOT/preflight_unused"

EDGE_SHA256=$(jq -er .sha256 "$PREFLIGHT_DIR/training_edge_manifest.json")

"$PYTHON" scripts/run_walk_forward_node_eval.py \
  --name "$NAME" \
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
  --edge-top-k "$TOP_K" \
  --edge-correlation-mode "$CORRELATION_MODE" \
  --graph-neighbor-scale "$NEIGHBOR_SCALE" \
  --temporal-graph-neighbor-scale "$TEMPORAL_NEIGHBOR_SCALE" \
  --temporal-stock-edge-scale "$TEMPORAL_STOCK_EDGE_SCALE" \
  --event-edge-top-k 0 \
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
  --cache-dir "$OHLCV_CACHE" \
  --external-cache-dir data/external_cache \
  --require-all-external-factors \
  ${TEMPORAL_EXCLUDE_ARGS[@]+"${TEMPORAL_EXCLUDE_ARGS[@]}"} \
  --expected-training-manifest-sha256 "$FOLD1_SHA256" \
  --expected-training-edge-manifest-sha256 "$EDGE_SHA256" \
  --reports-root "$REPORTS_ROOT" \
  --models-root "$MODELS_ROOT" \
  --summary-output "$REPORTS_ROOT/summary.json"

jq -n \
  --arg name "$NAME" \
  --arg mode "$CORRELATION_MODE" \
  --arg panel_sha "$FOLD1_SHA256" \
  --arg edge_sha "$EDGE_SHA256" \
  --argjson neighbor_scale "$NEIGHBOR_SCALE" \
  --argjson temporal_neighbor_scale "$TEMPORAL_NEIGHBOR_SCALE" \
  --argjson temporal_stock_edge_scale "$TEMPORAL_STOCK_EDGE_SCALE" \
  --arg excluded_prefixes "$EXCLUDED_PREFIX_CSV" \
  --argjson top_k "$TOP_K" \
  '{
    name: $name,
    edge_correlation_mode: $mode,
    edge_top_k: $top_k,
    graph_neighbor_scale: $neighbor_scale,
    temporal_graph_neighbor_scale: $temporal_neighbor_scale,
    temporal_stock_edge_scale: $temporal_stock_edge_scale,
    temporal_excluded_feature_prefixes: (
      $excluded_prefixes | split(",") | map(select(length > 0))
    ),
    seed: 1704,
    training_panel_sha256: $panel_sha,
    training_edge_manifest_sha256: $edge_sha,
    promotion_evidence: false
  }' > "$RUN_ROOT/config.json"
touch "$RUN_ROOT/CANDIDATE_COMPLETE"
