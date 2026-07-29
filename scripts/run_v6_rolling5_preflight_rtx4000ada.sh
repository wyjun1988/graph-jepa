#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
PYTHON_BIN="${PYTHON:-/root/venvs/stock-v2-cu128/bin/python}"
RUN_NAME="${RUN_NAME:-broad_transition_jepa_v6_rolling5_v3_recipe16_preflight_20260714}"
CONTRACT="${CONTRACT:-configs/rolling-v6-shadow-qualification-v3-20260714.json}"
MIN_EVENT_COVERAGE="${MIN_EVENT_COVERAGE:-0.99}"
MIN_FUNDAMENTAL_COVERAGE="${MIN_FUNDAMENTAL_COVERAGE:-0.79}"
MIN_INVESTOR_COVERAGE="${MIN_INVESTOR_COVERAGE:-0.95}"
DISCOVERY_ONLY="${DISCOVERY_ONLY:-0}"
REPORT_ROOT="reports/$RUN_NAME"
MODEL_ROOT="models/$RUN_NAME"
LOG="ops/training/${RUN_NAME}.log"
OHLCV="${OHLCV:-data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv}"
MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-}"
GLOBAL_STOCK_CONTEXT="${GLOBAL_STOCK_CONTEXT:-0}"
DOWNSTREAM_AUXILIARY_LOSS_WEIGHT="${DOWNSTREAM_AUXILIARY_LOSS_WEIGHT:-0.0}"
ROLLOUT_LOSS_WEIGHTS="${ROLLOUT_LOSS_WEIGHTS:-2,2,1,1,1}"

cd "$ROOT"
mkdir -p "$REPORT_ROOT" "$MODEL_ROOT" "$(dirname "$LOG")"

exec 9>"$REPORT_ROOT/.preflight.lock"
if ! flock -n 9; then
  printf '%s\n' "rolling preflight is already running"
  exit 0
fi

"$PYTHON_BIN" scripts/audit_rolling_validation_contract.py \
  --contract "$CONTRACT" \
  --output "$REPORT_ROOT/contract_audit.json"

readarray -t FOLD_SPECS < <(
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); [print(f"{r['"'"'train_end'"'"']}:{r['"'"'eval_end'"'"']}") for r in p['"'"'folds'"'"']]' \
    "$CONTRACT"
)
FOLD_ARGS=()
for fold in "${FOLD_SPECS[@]}"; do
  FOLD_ARGS+=(--fold "$fold")
done
MIN_TRAIN_ARGS=()
if [[ -n "$MIN_TRAIN_ROWS" ]]; then
  MIN_TRAIN_ARGS+=(--min-train-rows "$MIN_TRAIN_ROWS")
fi
GLOBAL_STOCK_CONTEXT_ARGS=()
if [[ "$GLOBAL_STOCK_CONTEXT" == "1" ]]; then
  GLOBAL_STOCK_CONTEXT_ARGS+=(--global-stock-context)
fi

"$PYTHON_BIN" scripts/run_walk_forward_node_eval.py \
  --name "$RUN_NAME" \
  "${FOLD_ARGS[@]}" \
  --start 2020-01-01 \
  --epochs 24 \
  --hidden-dim 1024 \
  --layers 10 \
  --train-batch-size 16 \
  --snapshot-workers 16 \
  --amp-dtype bfloat16 \
  --device cuda \
  --eval-device cuda \
  --max-steps 0 \
  --seed 17 \
  --checkpoint-epochs 8,16 \
  --training-manifest-schema-version 4 \
  --universe krx \
  --universe-manifest data/universes/krx500_pit_20191231.json \
  --max-tickers 500 \
  --cache-dir "$OHLCV" \
  "${MIN_TRAIN_ARGS[@]}" \
  --horizon 10 \
  --top-k 5 \
  --edge-top-k 6 \
  --edge-correlation-mode signed \
  --graph-neighbor-scale 1.0 \
  --temporal-graph-neighbor-scale 0.0 \
  --temporal-stock-edge-scale 1.0 \
  "${GLOBAL_STOCK_CONTEXT_ARGS[@]}" \
  --partial-corr-top-k 0 \
  --lead-lag-top-k 0 \
  --policy-rate-edge-scale 0.0 \
  --lr 0.0003 \
  --ema-decay 0.9995 \
  --latent-loss-weight 0.25 \
  --state-loss-weight 1.0 \
  --current-imputation-loss-weight 1.0 \
  --entry-path-correlation-loss-weight 0.05 \
  --downstream-auxiliary-loss-weight "$DOWNSTREAM_AUXILIARY_LOSS_WEIGHT" \
  --downstream-transition-loss-weight 0.10 \
  --downstream-transition-pooling robust_projected \
  --normalize-predictor-output \
  --temporal-state-mode horizon_residual_heads \
  --temporal-state-context-skip \
  --temporal-residual-short-steps 2 \
  --pretrain-task temporal \
  --temporal-offset 10 \
  --latent-rollout-steps 10 \
  --rollout-offsets 1,2,3,5,10 \
  --rollout-loss-weights "$ROLLOUT_LOSS_WEIGHTS" \
  --path-horizons 1,2,3,5,10 \
  --mask-strategy mixed \
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
  --min-event-coverage "$MIN_EVENT_COVERAGE" \
  --fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
  --fundamental-lag-days 1 \
  --require-fundamental-sensors \
  --min-fundamental-coverage "$MIN_FUNDAMENTAL_COVERAGE" \
  --investor-cache-dir data/kiwoom_investor_cache \
  --investor-flow-lag-days 1 \
  --require-investor-sensors \
  --min-investor-coverage "$MIN_INVESTOR_COVERAGE" \
  --external-preset kr_global_rates \
  --external-node-mode nodes \
  --external-lag-days 1 \
  --external-cache-dir data/external_cache \
  --require-all-external-factors \
  --edge-manifest-only \
  --reports-root "$REPORT_ROOT" \
  --models-root "$MODEL_ROOT" \
  --summary-output "$REPORT_ROOT/summary.json" \
  2>&1 | tee "$LOG"

if [[ "$DISCOVERY_ONLY" == "1" ]]; then
  touch "$REPORT_ROOT/DISCOVERY_COMPLETE"
  printf '%s\n' "$RUN_NAME sensor discovery complete" | tee -a "$LOG"
  exit 0
fi

"$PYTHON_BIN" scripts/freeze_rolling_v6_preflight.py \
  --contract "$CONTRACT" \
  --reports-root "$REPORT_ROOT" \
  --run-name "$RUN_NAME" \
  --output "$REPORT_ROOT/frozen_manifest_contract.json"

touch "$REPORT_ROOT/PREFLIGHT_COMPLETE"
printf '%s\n' "$RUN_NAME preflight complete" | tee -a "$LOG"
