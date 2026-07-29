#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
PYTHON_BIN="${PYTHON:-/root/venvs/stock-v2-cu128/bin/python}"
RUN_NAME="${RUN_NAME:-broad_transition_jepa_v6_rolling5_v3_seed17_rtx4000ada_20260714}"
PREFLIGHT_NAME="${PREFLIGHT_NAME:-broad_transition_jepa_v6_rolling5_v3_recipe16_preflight_20260714}"
CONTRACT="${CONTRACT:-configs/rolling-v6-shadow-qualification-v3-20260714.json}"
PREFLIGHT_ROOT="reports/$PREFLIGHT_NAME"
FROZEN_CONTRACT="$PREFLIGHT_ROOT/frozen_manifest_contract.json"
REPORTS_BASE="${REPORTS_BASE:-reports}"
MODELS_BASE="${MODELS_BASE:-models}"
REPORT_ROOT="$REPORTS_BASE/$RUN_NAME"
MODEL_ROOT="$MODELS_BASE/$RUN_NAME"
LOG="${LOG:-ops/training/${RUN_NAME}.log}"
OHLCV="${OHLCV:-data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv}"
MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-}"
GLOBAL_STOCK_CONTEXT="${GLOBAL_STOCK_CONTEXT:-0}"
DOWNSTREAM_AUXILIARY_LOSS_WEIGHT="${DOWNSTREAM_AUXILIARY_LOSS_WEIGHT:-0.0}"
ROLLOUT_LOSS_WEIGHTS="${ROLLOUT_LOSS_WEIGHTS:-2,2,1,1,1}"
SEED="${SEED:-17}"
ALLOW_DIAGNOSTIC_SEED_OVERRIDE="${ALLOW_DIAGNOSTIC_SEED_OVERRIDE:-0}"

cd "$ROOT"
mkdir -p "$REPORT_ROOT" "$MODEL_ROOT" "$(dirname "$LOG")"

exec 9>"$REPORT_ROOT/.training.lock"
if ! flock -n 9; then
  printf '%s\n' "rolling training is already running"
  exit 0
fi
if [[ ! -f "$PREFLIGHT_ROOT/PREFLIGHT_COMPLETE" ]]; then
  printf '%s\n' "frozen rolling preflight is incomplete" >&2
  exit 3
fi

"$PYTHON_BIN" scripts/verify_rolling_v6_preflight.py \
  --frozen-contract "$FROZEN_CONTRACT" \
  --contract "$CONTRACT" \
  --reports-root "$PREFLIGHT_ROOT" \
  --output "$REPORT_ROOT/preflight_verification.json"

CONTRACT_SEED="$("$PYTHON_BIN" -c \
  'import json,sys; print(json.load(open(sys.argv[1]))["base_contract"]["architecture"]["seed"])' \
  "$FROZEN_CONTRACT")"
if [[ "$SEED" != "$CONTRACT_SEED" ]]; then
  if [[ "$ALLOW_DIAGNOSTIC_SEED_OVERRIDE" != "1" ]]; then
    printf '%s\n' \
      "runtime seed $SEED differs from frozen contract seed $CONTRACT_SEED; diagnostic override required" >&2
    exit 5
  fi
  "$PYTHON_BIN" -c \
    'import json,pathlib,sys; path=pathlib.Path(sys.argv[1]); payload={"schema_version":1,"role":"diagnostic_seed_stability_only","contract_seed":int(sys.argv[2]),"runtime_seed":int(sys.argv[3]),"promotion_eligible":False,"live_orders_allowed":False}; path.write_text(json.dumps(payload,sort_keys=True,indent=2)+"\n")' \
    "$REPORT_ROOT/diagnostic_seed_override.json" "$CONTRACT_SEED" "$SEED"
  touch "$REPORT_ROOT/DIAGNOSTIC_ONLY"
fi

readarray -t FOLD_SPECS < <(
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); [print(f"{r['"'"'train_end'"'"']}:{r['"'"'eval_end'"'"']}") for r in p['"'"'base_contract'"'"']['"'"'folds'"'"']]' \
    "$FROZEN_CONTRACT"
)
readarray -t DATA_SHAS < <(
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); [print(r['"'"'training_data_manifest_sha256'"'"']) for r in p['"'"'fold_manifests'"'"']]' \
    "$FROZEN_CONTRACT"
)
readarray -t EDGE_SHAS < <(
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); [print(r['"'"'training_edge_manifest_sha256'"'"']) for r in p['"'"'fold_manifests'"'"']]' \
    "$FROZEN_CONTRACT"
)
if [[ "${#FOLD_SPECS[@]}" -ne 5 || "${#DATA_SHAS[@]}" -ne 5 || "${#EDGE_SHAS[@]}" -ne 5 ]]; then
  printf '%s\n' "frozen rolling contract must contain exactly five folds" >&2
  exit 4
fi
FOLD_ARGS=()
MANIFEST_ARGS=()
for index in "${!FOLD_SPECS[@]}"; do
  FOLD_ARGS+=(--fold "${FOLD_SPECS[$index]}")
  MANIFEST_ARGS+=(--expected-training-manifest-sha256 "${DATA_SHAS[$index]}")
  MANIFEST_ARGS+=(--expected-training-edge-manifest-sha256 "${EDGE_SHAS[$index]}")
done
MIN_TRAIN_ARGS=()
if [[ -n "$MIN_TRAIN_ROWS" ]]; then
  MIN_TRAIN_ARGS+=(--min-train-rows "$MIN_TRAIN_ROWS")
fi
GLOBAL_STOCK_CONTEXT_ARGS=()
if [[ "$GLOBAL_STOCK_CONTEXT" == "1" ]]; then
  GLOBAL_STOCK_CONTEXT_ARGS+=(--global-stock-context)
fi

if [[ ! -f "$REPORT_ROOT/WALK_FORWARD_COMPLETE" ]]; then
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
    --seed "$SEED" \
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
    --min-event-coverage 0.99 \
    --fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
    --fundamental-lag-days 1 \
    --require-fundamental-sensors \
    --min-fundamental-coverage 0.79 \
    --investor-cache-dir data/kiwoom_investor_cache \
    --investor-flow-lag-days 1 \
    --require-investor-sensors \
    --min-investor-coverage 0.95 \
    --external-preset kr_global_rates \
    --external-node-mode nodes \
    --external-lag-days 1 \
    --external-cache-dir data/external_cache \
    --require-all-external-factors \
    --resume-complete-folds \
    "${MANIFEST_ARGS[@]}" \
    --reports-root "$REPORT_ROOT" \
    --models-root "$MODEL_ROOT" \
    --summary-output "$REPORT_ROOT/summary.json" \
    2>&1 | tee "$LOG"
  touch "$REPORT_ROOT/WALK_FORWARD_COMPLETE"
fi

readarray -t MODEL_DIRS < <(
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); [print(r['"'"'model_dir'"'"']) for r in p['"'"'folds'"'"']]' \
    "$REPORT_ROOT/summary.json"
)
for index in "${!MODEL_DIRS[@]}"; do
  fold_number=$((index + 1))
  output="$REPORT_ROOT/trained_transition_eval/fold${fold_number}"
  if [[ -f "$output/summary.json" ]]; then
    continue
  fi
  "$PYTHON_BIN" scripts/evaluate_trained_market_transition_auxiliary.py \
    --model-dir "${MODEL_DIRS[$index]}" \
    --output-dir "$output" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --batch-size 64 \
    --device cuda \
    --cache-dir "$OHLCV" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
done

touch "$REPORT_ROOT/PIPELINE_COMPLETE"
printf '%s\n' "$RUN_NAME rolling pipeline complete" | tee -a "$LOG"
