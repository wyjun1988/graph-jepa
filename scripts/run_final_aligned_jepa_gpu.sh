#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/stock-v2"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
RUN_NAME="final_aligned_jepa_opmask_aux_v1_seed17"
TRAIN_REPORTS="reports/${RUN_NAME}"
TRAIN_MODELS="models/${RUN_NAME}"
PROBE_NAME="${RUN_NAME}_frozen_downstream"
PROBE_REPORTS="reports/${PROBE_NAME}"
PROBE_CACHE="data/cache/${PROBE_NAME}"
AUX_REPORTS="reports/${RUN_NAME}_trained_auxiliary"
FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"
FOLD1_PANEL="00cbf81903f5c4bba5ef6ddc39e8a243c63b8445b8ed90bc1e8cea7f2ada630e"
FOLD2_PANEL="4ae8bdfb8e6f13af77dcb9847974f2c74694768a46667ff9debab136b0f96452"
FOLD1_EDGE="c66077ecbc91c3996204dfb95b1b5e12b2542ac143036c8659d3363519669c2b"
FOLD2_EDGE="a85f939144cec194ef35fb927603b69bc111ce3bbedf12bcc159757c713d7870"
EVENT_PATH="data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl"
OHLCV_CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
UNIVERSE="data/universes/krx500_pit_20191231.json"
FUNDAMENTALS="data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl"

cd "$ROOT"
mkdir -p \
  "$TRAIN_REPORTS" \
  "$TRAIN_MODELS" \
  "$PROBE_REPORTS" \
  "$PROBE_CACHE" \
  "$AUX_REPORTS"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

on_error() {
  local status=$?
  printf '{"status":"failed","exit_status":%d,"live_orders_allowed":false}\n' \
    "$status" > "$TRAIN_REPORTS/PIPELINE_FAILED"
  exit "$status"
}
trap on_error ERR

printf '%s\n' \
  '{"scope":"research_only","live_orders_allowed":false,"mask_policy":"operational_mixed","downstream_auxiliary_tasks":["path_return","max_favorable_excursion","max_adverse_excursion","realized_volatility"]}' \
  > "$TRAIN_REPORTS/experiment_contract.json"

common_data_args=(
  --training-manifest-schema-version 4
  --universe krx
  --universe-manifest "$UNIVERSE"
  --max-tickers 500
  --cache-dir "$OHLCV_CACHE"
  --event-path "$EVENT_PATH"
  --event-coverage-mode mask_uncovered
  --require-event-sensors
  --min-event-coverage 0.95
  --fundamental-path "$FUNDAMENTALS"
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
  --external-cache-dir data/external_cache
  --require-all-external-factors
)

model_args=(
  --horizon 10
  --top-k 5
  --edge-top-k 6
  --edge-correlation-mode signed
  --graph-neighbor-scale 1.0
  --temporal-graph-neighbor-scale 0.0
  --temporal-stock-edge-scale 1.0
  --lr 0.0003
  --ema-decay 0.9995
  --state-loss-weight 1.0
  --current-imputation-loss-weight 1.0
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
  --mask-strategy operational_mixed
  --partial-corr-top-k 0
  --lead-lag-top-k 0
  --policy-rate-edge-scale 0.0
  --event-edge-top-k 0
  --temporal-exclude-feature-prefix news_
  --temporal-exclude-feature-prefix fund_
)

objective_args=(
  --return-correlation-loss-weight 0.0
  --entry-path-correlation-loss-weight 0.05
  --downstream-auxiliary-loss-weight 0.10
  --downstream-path-weight 1.0
  --downstream-mfe-weight 0.25
  --downstream-mae-weight 0.25
  --downstream-volatility-weight 1.0
  --latent-loss-weight 0.25
  --state-feature-weight return_1d=12
  --state-feature-weight return_2d=12
  --state-feature-weight return_3d=12
  --state-feature-weight return_5d=12
  --state-feature-weight return_10d=12
  --state-feature-weight gap_open=12
  --state-feature-weight intraday_return=12
)

if [[ ! -f "$TRAIN_REPORTS/TRAINING_COMPLETE" ]]; then
  "$PYTHON_BIN" scripts/run_walk_forward_node_eval.py \
    --name "$RUN_NAME" \
    --fold 2023-12-29:2024-12-30 \
    --fold 2024-12-30:2026-07-10 \
    --start 2020-01-01 \
    --epochs 24 \
    --hidden-dim 1024 \
    --layers 10 \
    --train-batch-size 8 \
    --snapshot-workers 16 \
    --device cuda \
    --eval-device cuda \
    --max-steps 0 \
    --seed 17 \
    --expected-training-manifest-sha256 "$FOLD1_PANEL" \
    --expected-training-manifest-sha256 "$FOLD2_PANEL" \
    --expected-training-edge-manifest-sha256 "$FOLD1_EDGE" \
    --expected-training-edge-manifest-sha256 "$FOLD2_EDGE" \
    --reports-root "$TRAIN_REPORTS" \
    --models-root "$TRAIN_MODELS" \
    --summary-output "$TRAIN_REPORTS/summary.json" \
    "${objective_args[@]}" \
    "${common_data_args[@]}" \
    "${model_args[@]}"
  touch "$TRAIN_REPORTS/TRAINING_COMPLETE"
fi

probe_args=(
  --horizons 1,2,3,5,10
  --variants raw,latent,raw_latent,raw_shuffled_latent
  --modes single,multi
  --validation-days 126
  --epochs 8
  --patience 2
  --batch-size 8192
  --hidden-dim 256
  --layers 2
  --dropout 0.05
  --learning-rate 0.0003
  --weight-decay 0.0001
  --feature-workers 16
  --device cuda
  --amp
)

run_probe() {
  local fold_name="$1"
  local model_dir="$2"
  local test_end="$3"
  if [[ -f "$PROBE_REPORTS/$fold_name/PROBE_COMPLETE" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/benchmark_frozen_downstream.py \
    --model-dir "$model_dir" \
    --output-dir "$PROBE_REPORTS/$fold_name" \
    --raw-context-cache "$PROBE_CACHE/${fold_name}_raw.npy" \
    --latent-cache-dir "$PROBE_CACHE/${fold_name}_latent" \
    --test-end "$test_end" \
    "${probe_args[@]}" \
    2>&1 | tee "$PROBE_REPORTS/${fold_name}.log"
  touch "$PROBE_REPORTS/$fold_name/PROBE_COMPLETE"
}

run_auxiliary_eval_and_cleanup() {
  local fold_name="$1"
  local model_dir="$2"
  local test_end="$3"
  local model_name
  local output
  model_name="$(basename "$model_dir")"
  output="$AUX_REPORTS/${model_name}.json"
  if [[ ! -f "$output" ]]; then
    "$PYTHON_BIN" scripts/evaluate_trained_auxiliary_heads.py \
      --model-dir "$model_dir" \
      --latent-cache-dir "$PROBE_CACHE/${fold_name}_latent" \
      --output "$output" \
      --horizons 1,2,3,5,10 \
      --validation-days 126 \
      --test-end "$test_end" \
      --batch-size 8192 \
      --device cuda \
      --cache-dir "$OHLCV_CACHE" \
      --external-cache-dir data/external_cache
  fi
  rm -rf \
    "$PROBE_CACHE/${fold_name}_latent" \
    "$PROBE_CACHE/${fold_name}_raw.npy.parts"
  rm -f \
    "$PROBE_CACHE/${fold_name}_raw.npy" \
    "$PROBE_CACHE/${fold_name}_raw.npy.json"
}

run_probe \
  fold1 \
  "$TRAIN_MODELS/${RUN_NAME}_${FOLD1}" \
  2024-12-30
run_auxiliary_eval_and_cleanup \
  fold1 \
  "$TRAIN_MODELS/${RUN_NAME}_${FOLD1}" \
  2024-12-30
run_probe \
  fold2 \
  "$TRAIN_MODELS/${RUN_NAME}_${FOLD2}" \
  2026-07-10
run_auxiliary_eval_and_cleanup \
  fold2 \
  "$TRAIN_MODELS/${RUN_NAME}_${FOLD2}" \
  2026-07-10

"$PYTHON_BIN" scripts/summarize_frozen_downstream.py \
  --fold "$PROBE_REPORTS/fold1/summary.json" \
  --fold "$PROBE_REPORTS/fold2/summary.json" \
  --output-dir "$PROBE_REPORTS"

CLEAN_CACHE_AFTER_EVAL=1 \
  "$ROOT/scripts/run_final_aligned_auxiliary_eval_gpu.sh"

touch "$PROBE_REPORTS/PIPELINE_COMPLETE"
touch "$TRAIN_REPORTS/PIPELINE_COMPLETE"
