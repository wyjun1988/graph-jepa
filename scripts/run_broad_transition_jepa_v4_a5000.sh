#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON_BIN="${PYTHON:-/workspace/venvs/stock-v2-cu128/bin/python}"
RUN_NAME="${RUN_NAME:-broad_transition_jepa_v4_robust_seed17_20260714}"
TARGET_VERSION="${TARGET_VERSION:-market_transition_v4_robust_breadth_20260714}"
DEVICE="${DEVICE:-cuda}"
EVAL_DEVICE="${EVAL_DEVICE:-$DEVICE}"
TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-8}"
SNAPSHOT_WORKERS="${SNAPSHOT_WORKERS:-16}"
AMP_DTYPE="${AMP_DTYPE:-none}"
MAX_TRAIN_STEPS="${MAX_TRAIN_STEPS:-0}"
TRAINING_ONLY="${TRAINING_ONLY:-0}"
EDGE_MANIFEST_ONLY="${EDGE_MANIFEST_ONLY:-0}"
SEED="${SEED:-17}"
FOLD1_EDGE_SHA256="${FOLD1_EDGE_SHA256:-c66077ecbc91c3996204dfb95b1b5e12b2542ac143036c8659d3363519669c2b}"
FOLD2_EDGE_SHA256="${FOLD2_EDGE_SHA256:-a85f939144cec194ef35fb927603b69bc111ce3bbedf12bcc159757c713d7870}"
TRANSITION_EVAL_BATCH_SIZE="${TRANSITION_EVAL_BATCH_SIZE:-64}"
REPORTS_ROOT="reports/$RUN_NAME"
MODELS_ROOT="models/$RUN_NAME"
LOG="ops/training/${RUN_NAME}_a5000.log"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$REPORTS_ROOT" "$MODELS_ROOT" "$(dirname "$LOG")"
FOLD_ARGS=(
  --fold 2023-12-29:2024-12-30
  --fold 2024-12-30:2026-07-10
)
MANIFEST_ARGS=(
  --expected-training-manifest-sha256 00cbf81903f5c4bba5ef6ddc39e8a243c63b8445b8ed90bc1e8cea7f2ada630e
  --expected-training-manifest-sha256 4ae8bdfb8e6f13af77dcb9847974f2c74694768a46667ff9debab136b0f96452
  --expected-training-edge-manifest-sha256 "$FOLD1_EDGE_SHA256"
  --expected-training-edge-manifest-sha256 "$FOLD2_EDGE_SHA256"
)
TRAINING_ONLY_ARGS=()
EDGE_MANIFEST_ONLY_ARGS=()
if [[ "$TRAINING_ONLY" == "1" && "$EDGE_MANIFEST_ONLY" == "1" ]]; then
  printf '%s\n' "TRAINING_ONLY and EDGE_MANIFEST_ONLY are mutually exclusive" >&2
  exit 2
fi
if [[ "$TRAINING_ONLY" == "1" ]]; then
  FOLD_ARGS=(--fold 2023-12-29:2024-12-30)
  MANIFEST_ARGS=(
    --expected-training-manifest-sha256 00cbf81903f5c4bba5ef6ddc39e8a243c63b8445b8ed90bc1e8cea7f2ada630e
    --expected-training-edge-manifest-sha256 "$FOLD1_EDGE_SHA256"
  )
  TRAINING_ONLY_ARGS=(--training-only)
fi
if [[ "$EDGE_MANIFEST_ONLY" == "1" ]]; then
  EDGE_MANIFEST_ONLY_ARGS=(--edge-manifest-only)
fi
printf \
  '{"scope":"research_only","target":"%s","objective":"joint_robust_broad_transition_auxiliary","transition_pooling":"robust_projected","explicit_broad_selloff":true,"test_used_for_selection":false,"live_orders_allowed":false}\n' \
  "$TARGET_VERSION" \
  > "$REPORTS_ROOT/experiment_contract.json"

if [[ ! -f "$REPORTS_ROOT/WALK_FORWARD_COMPLETE" ]]; then
  "$PYTHON_BIN" scripts/run_walk_forward_node_eval.py \
    --name "$RUN_NAME" \
    "${FOLD_ARGS[@]}" \
    --start 2020-01-01 \
    --epochs 24 \
    --hidden-dim 1024 \
    --layers 10 \
    --train-batch-size "$TRAIN_BATCH_SIZE" \
    --snapshot-workers "$SNAPSHOT_WORKERS" \
    --amp-dtype "$AMP_DTYPE" \
    --max-train-steps "$MAX_TRAIN_STEPS" \
    --device "$DEVICE" \
    --eval-device "$EVAL_DEVICE" \
    --max-steps 0 \
    --seed "$SEED" \
    --checkpoint-epochs 8,16 \
    --training-manifest-schema-version 4 \
    --universe krx \
    --universe-manifest data/universes/krx500_pit_20191231.json \
    --max-tickers 500 \
    --cache-dir "$OHLCV" \
    --horizon 10 \
    --top-k 5 \
    --edge-top-k 6 \
    --edge-correlation-mode signed \
    --graph-neighbor-scale 1.0 \
    --temporal-graph-neighbor-scale 0.0 \
    --temporal-stock-edge-scale 1.0 \
    --partial-corr-top-k 0 \
    --lead-lag-top-k 0 \
    --policy-rate-edge-scale 0.0 \
    --lr 0.0003 \
    --ema-decay 0.9995 \
    --latent-loss-weight 0.25 \
    --state-loss-weight 1.0 \
    --current-imputation-loss-weight 1.0 \
    --entry-path-correlation-loss-weight 0.05 \
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
    --rollout-loss-weights 2,2,1,1,1 \
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
    "${MANIFEST_ARGS[@]}" \
    "${TRAINING_ONLY_ARGS[@]}" \
    "${EDGE_MANIFEST_ONLY_ARGS[@]}" \
    --reports-root "$REPORTS_ROOT" \
    --models-root "$MODELS_ROOT" \
    --summary-output "$REPORTS_ROOT/summary.json" \
    2>&1 | tee "$LOG"
  touch "$REPORTS_ROOT/WALK_FORWARD_COMPLETE"
fi

if [[ "$TRAINING_ONLY" == "1" ]]; then
  touch "$REPORTS_ROOT/TRAINING_PROBE_COMPLETE"
  exit 0
fi
if [[ "$EDGE_MANIFEST_ONLY" == "1" ]]; then
  touch "$REPORTS_ROOT/EDGE_MANIFEST_COMPLETE"
  exit 0
fi

evaluate_transition() {
  local fold="$1"
  local model="$2"
  local output="$REPORTS_ROOT/trained_transition_eval/$fold"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/evaluate_trained_market_transition_auxiliary.py \
    --model-dir "$MODELS_ROOT/$model" \
    --output-dir "$output" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --batch-size "$TRANSITION_EVAL_BATCH_SIZE" \
    --device "$EVAL_DEVICE" \
    --cache-dir "$OHLCV" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

evaluate_transition \
  fold1 \
  "${RUN_NAME}_fold1_20231229_to_20241230"
evaluate_transition \
  fold2 \
  "${RUN_NAME}_fold2_20241230_to_20260710"

touch "$REPORTS_ROOT/PIPELINE_COMPLETE"
printf '%s\n' "$RUN_NAME pipeline complete" | tee -a "$LOG"
