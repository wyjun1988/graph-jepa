#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
PYTHON_BIN="${PYTHON:-/root/venvs/stock-v2-cu128/bin/python}"
MODE="${MODE:-preflight}"
CONTRACT="configs/us-etf-node-ablation-v1-20260716.json"
PREFLIGHT_NAME="us_etf_node_ablation_v1_baseline_exact_preflight_20260716"
RUN_NAME="us_etf_node_ablation_v1_baseline_exact_seed17_rtx4000ada_20260716"
PREFLIGHT_ROOT="reports/$PREFLIGHT_NAME"
FROZEN="$PREFLIGHT_ROOT/frozen_edge_manifests.json"
ETF_PANEL="data/us_etf_consensus_daily/etf_v2_34nodes_20191231_20260714_v2"
OHLCV="data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv"

if [[ "$MODE" == "preflight" ]]; then
  ACTIVE_NAME="$PREFLIGHT_NAME"
  REPORT_ROOT="$PREFLIGHT_ROOT"
  MODEL_ROOT="models/$PREFLIGHT_NAME"
  LOG="ops/training/${PREFLIGHT_NAME}.log"
elif [[ "$MODE" == "train" ]]; then
  ACTIVE_NAME="$RUN_NAME"
  REPORT_ROOT="reports/$RUN_NAME"
  MODEL_ROOT="models/$RUN_NAME"
  LOG="ops/training/${RUN_NAME}.log"
else
  printf '%s\n' "MODE must be preflight or train" >&2
  exit 2
fi

cd "$ROOT"
mkdir -p "$REPORT_ROOT" "$MODEL_ROOT" "$(dirname "$LOG")"
exec 9>"$REPORT_ROOT/.run.lock"
if ! flock -n 9; then
  printf '%s\n' "$ACTIVE_NAME is already running"
  exit 0
fi

COMMON_ARGS=(
  --name "$ACTIVE_NAME"
  --fold 2024-01-03:2024-11-05
  --fold 2024-11-05:2025-09-05
  --start 2020-01-01
  --epochs 24
  --hidden-dim 1024
  --layers 10
  --train-batch-size 16
  --snapshot-workers 16
  --amp-dtype bfloat16
  --device cuda
  --eval-device cuda
  --max-steps 0
  --seed 17
  --training-manifest-schema-version 4
  --universe krx
  --universe-manifest data/universes/krx500_pit_20191231.json
  --max-tickers 500
  --cache-dir "$OHLCV"
  --min-train-rows 1
  --horizon 10
  --top-k 5
  --edge-top-k 6
  --edge-correlation-mode signed
  --graph-neighbor-scale 1.0
  --temporal-graph-neighbor-scale 0.0
  --temporal-stock-edge-scale 1.0
  --partial-corr-top-k 0
  --lead-lag-top-k 0
  --policy-rate-edge-scale 0.0
  --lr 0.0003
  --ema-decay 0.9995
  --latent-loss-weight 0.25
  --state-loss-weight 1.0
  --current-imputation-loss-weight 1.0
  --entry-path-correlation-loss-weight 0.05
  --downstream-auxiliary-loss-weight 0.0
  --downstream-transition-loss-weight 0.10
  --downstream-transition-pooling robust_projected
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
  --mask-strategy mixed
  --state-feature-weight return_1d=12
  --state-feature-weight return_2d=12
  --state-feature-weight return_3d=12
  --state-feature-weight return_5d=12
  --state-feature-weight return_10d=12
  --state-feature-weight gap_open=12
  --state-feature-weight intraday_return=12
  --temporal-exclude-feature-prefix news_
  --temporal-exclude-feature-prefix fund_
  --event-path data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl
  --event-coverage-mode mask_uncovered
  --require-event-sensors
  --min-event-coverage 0.99
  --fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl
  --fundamental-lag-days 1
  --require-fundamental-sensors
  --min-fundamental-coverage 0.79
  --investor-cache-dir data/kiwoom_investor_cache
  --investor-flow-lag-days 1
  --require-investor-sensors
  --min-investor-coverage 0.95
  --external-preset kr_global_rates
  --external-node-mode nodes
  --external-lag-days 1
  --external-cache-dir data/external_cache
  --require-all-external-factors
  --external-etf-panel "$ETF_PANEL"
  --expected-training-manifest-sha256 6361518b3de63f8760a6bea653f4bcac7eb3f943083b2108eb5aad496719c896
  --expected-training-manifest-sha256 0b6df01f186f95a5fa11c5364218cbbb80d5f76e959f3e40281c42e4d63ac49b
  --reports-root "$REPORT_ROOT"
  --models-root "$MODEL_ROOT"
  --summary-output "$REPORT_ROOT/summary.json"
)

if [[ "$MODE" == "preflight" ]]; then
  "$PYTHON_BIN" scripts/run_walk_forward_node_eval.py \
    "${COMMON_ARGS[@]}" \
    --edge-manifest-only \
    2>&1 | tee "$LOG"
  "$PYTHON_BIN" scripts/freeze_us_etf_node_ablation.py \
    --contract "$CONTRACT" \
    --reports-root "$REPORT_ROOT" \
    --run-name "$PREFLIGHT_NAME" \
    --output "$FROZEN" \
    2>&1 | tee -a "$LOG"
  touch "$REPORT_ROOT/PREFLIGHT_COMPLETE"
  exit 0
fi

if [[ ! -f "$PREFLIGHT_ROOT/PREFLIGHT_COMPLETE" || ! -f "$FROZEN" ]]; then
  printf '%s\n' "frozen ETF preflight is incomplete" >&2
  exit 3
fi
"$PYTHON_BIN" scripts/freeze_us_etf_node_ablation.py \
  --contract "$CONTRACT" \
  --reports-root "$PREFLIGHT_ROOT" \
  --run-name "$PREFLIGHT_NAME" \
  --output "$FROZEN" \
  --verify

readarray -t EDGE_SHAS < <(
  "$PYTHON_BIN" -c \
    'import json,sys; [print(row["training_edge_manifest_sha256"]) for row in json.load(open(sys.argv[1]))["fold_manifests"]]' \
    "$FROZEN"
)
if [[ "${#EDGE_SHAS[@]}" -ne 2 ]]; then
  printf '%s\n' "frozen ETF preflight must contain two edge manifests" >&2
  exit 4
fi

"$PYTHON_BIN" scripts/run_walk_forward_node_eval.py \
  "${COMMON_ARGS[@]}" \
  --expected-training-edge-manifest-sha256 "${EDGE_SHAS[0]}" \
  --expected-training-edge-manifest-sha256 "${EDGE_SHAS[1]}" \
  --resume-complete-folds \
  2>&1 | tee "$LOG"
touch "$REPORT_ROOT/EXPERIMENT_COMPLETE"
