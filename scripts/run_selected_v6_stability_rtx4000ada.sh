#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
PYTHON_BIN="${PYTHON:-/root/venvs/stock-v2-cu128/bin/python}"
BASE_RUN="${BASE_RUN:-broad_transition_jepa_v6_systemic_seed17_rtx4000ada_20260714}"
DEVICE="${DEVICE:-cuda}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
CACHE_BATCH_SIZE="${CACHE_BATCH_SIZE:-8}"
HEAD_BATCH_SIZE="${HEAD_BATCH_SIZE:-128}"
HEAD_EVAL_BATCH_SIZE="${HEAD_EVAL_BATCH_SIZE:-512}"
EDGE_CACHE_WORKERS="${EDGE_CACHE_WORKERS:-16}"
BASE_REPORT="reports/$BASE_RUN"
BASE_MODEL="models/$BASE_RUN"
SELECTION_PATH="${SELECTION_PATH:-$BASE_REPORT/deployment_checkpoint_selection/selection.json}"
RUN_PREFIX="reports/broad_transition_jepa_v6_selected_raw_head"
AGGREGATE_ROOT="reports/broad_transition_jepa_v6_selected_raw_head_stability_aggregate_20260714"
SHAPE_ROOT="reports/broad_transition_jepa_v6_selected_major_node_shape_20260714"
CACHE_ROOT="reports/broad_transition_jepa_v6_selected_raw_head_cache_20260714"
TARGET_ROOT="reports/market_transition_target_audit_v6_systemic_impact_metric_20260714"
LOG="ops/training/broad_transition_jepa_v6_selected_stability_rtx4000ada_20260714.log"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$AGGREGATE_ROOT" "$SHAPE_ROOT" "$CACHE_ROOT" "$(dirname "$LOG")"

exec 9>"$AGGREGATE_ROOT/.stability.lock"
if ! flock -n 9; then
  printf '%s\n' "selected v6 stability evaluation is already queued or running"
  exit 0
fi

until [[ -f "$BASE_REPORT/CHECKPOINT_VALIDATION_COMPLETE" ]]; do
  sleep "$WAIT_SECONDS"
done

SELECTION="$SELECTION_PATH"
if [[ ! -f "$SELECTION" ]]; then
  printf '%s\n' "deployment checkpoint selection is missing" >&2
  exit 3
fi
if [[ "$($PYTHON_BIN -c "import json; print(json.load(open('$SELECTION'))['live_orders_allowed'])")" != "False" ]]; then
  printf '%s\n' "checkpoint selection is not research-only" >&2
  exit 4
fi
SELECTED_LABEL="$($PYTHON_BIN -c "import json; print(json.load(open('$SELECTION'))['selected_label'])")"

FOLD1="${BASE_RUN}_fold1_20231229_to_20241230"
FOLD2="${BASE_RUN}_fold2_20241230_to_20260710"

model_dir() {
  local model_name="$1"
  case "$SELECTED_LABEL" in
    epoch8)
      printf '%s\n' "$BASE_MODEL/$model_name/epoch_008"
      ;;
    epoch16)
      printf '%s\n' "$BASE_MODEL/$model_name/epoch_016"
      ;;
    epoch24)
      printf '%s\n' "$BASE_MODEL/$model_name"
      ;;
    *)
      printf 'unknown selected checkpoint: %s\n' "$SELECTED_LABEL" >&2
      return 5
      ;;
  esac
}

printf \
  '{"scope":"posthoc_selected_encoder_stability","base_run":"%s","selected_label":"%s","selection_fold":"fold1_only","fold2_used_for_selection":false,"head_seeds":[2701,4301,7301],"live_orders_allowed":false}\n' \
  "$BASE_RUN" "$SELECTED_LABEL" \
  > "$AGGREGATE_ROOT/experiment_contract.json"

run_head() {
  local seed="$1"
  local fold="$2"
  local model_name="$3"
  local model
  model="$(model_dir "$model_name")"
  local run_root="${RUN_PREFIX}_seed${seed}_20260714"
  local output="$run_root/$fold"
  local cache="$CACHE_ROOT/$fold/frozen_raw_transition_pool.npz"
  mkdir -p "$run_root"
  printf \
    '{"scope":"posthoc_selected_encoder_head_seed","base_run":"%s","selected_label":"%s","head_seed":%s,"fold1_used_for_checkpoint_selection":true,"fold2_used_for_selection":false,"head_test_used_for_selection":false,"live_orders_allowed":false}\n' \
    "$BASE_RUN" "$SELECTED_LABEL" "$seed" \
    > "$run_root/experiment_contract.json"
  if [[ ! -f "$output/summary.json" ]]; then
    "$PYTHON_BIN" scripts/benchmark_cached_pooled_market_transition_head.py \
      --model-dir "$model" \
      --output-dir "$output" \
      --pooled-cache "$cache" \
      --pooling-mode raw \
      --horizons 1,2,3,5,10 \
      --validation-days 126 \
      --epochs 80 \
      --patience 10 \
      --hidden-dim 256 \
      --layers 2 \
      --heads 8 \
      --dropout 0.10 \
      --learning-rate 0.0003 \
      --weight-decay 0.001 \
      --cache-batch-size "$CACHE_BATCH_SIZE" \
      --batch-size "$HEAD_BATCH_SIZE" \
      --eval-batch-size "$HEAD_EVAL_BATCH_SIZE" \
      --edge-cache-workers "$EDGE_CACHE_WORKERS" \
      --device "$DEVICE" \
      --seed "$seed" \
      --cache-dir "$OHLCV" \
      --external-cache-dir data/external_cache \
      2>&1 | tee -a "$LOG"
  fi
  if [[ ! -f "$output/major_trajectory/summary.json" ]]; then
    "$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
      --target-audit-root "$TARGET_ROOT/$fold" \
      --prediction-root "$output" \
      --output-dir "$output/major_trajectory" \
      --major-event-quantile 0.90 \
      2>&1 | tee -a "$LOG"
  fi
}

for seed in 2701 4301 7301; do
  run_head "$seed" fold1 "$FOLD1"
  run_head "$seed" fold2 "$FOLD2"
done

"$PYTHON_BIN" scripts/aggregate_market_transition_stability.py \
  --run-prefix "$RUN_PREFIX" \
  --seeds 2701,4301,7301 \
  --output-dir "$AGGREGATE_ROOT" \
  2>&1 | tee -a "$LOG"

printf \
  '{"scope":"selected_encoder_major_node_shape","base_run":"%s","selected_label":"%s","fold2_used_for_selection":false,"live_orders_allowed":false}\n' \
  "$BASE_RUN" "$SELECTED_LABEL" \
  > "$SHAPE_ROOT/experiment_contract.json"

run_shape() {
  local fold="$1"
  local model_name="$2"
  local output="$SHAPE_ROOT/$fold"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/evaluate_major_node_transition_shape.py \
    --model-dir "$(model_dir "$model_name")" \
    --target-audit-root "$TARGET_ROOT/$fold" \
    --output-dir "$output" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --batch-size 8 \
    --edge-cache-workers "$EDGE_CACHE_WORKERS" \
    --device "$DEVICE" \
    --cache-dir "$OHLCV" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

run_shape fold1 "$FOLD1"
run_shape fold2 "$FOLD2"

touch "$AGGREGATE_ROOT/EXPERIMENT_COMPLETE"
touch "$SHAPE_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "selected v6 stability and node-shape evaluation complete" | tee -a "$LOG"
