#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
PYTHON_BIN="${PYTHON:-/root/venvs/stock-v2-cu128/bin/python}"
RUN_NAME="${RUN_NAME:-broad_transition_jepa_v6_systemic_seed17_rtx4000ada_20260714}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-17}"
EDGE_CACHE_WORKERS="${EDGE_CACHE_WORKERS:-16}"
TRANSITION_BATCH_SIZE="${TRANSITION_BATCH_SIZE:-48}"
WAIT_SECONDS="${WAIT_SECONDS:-30}"
REPORT_ROOT="reports/$RUN_NAME"
MODEL_ROOT="models/$RUN_NAME"
LOG="ops/training/${RUN_NAME}_checkpoint_validation_rtx4000ada.log"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
DIRECT_REPORT_ROOT="${DIRECT_REPORT_ROOT:-reports/direct_state_mlp_broad_transition_jepa_v6_systemic_seed17_rtx4000ada_20260714_20260714}"

cd "$ROOT"
mkdir -p "$REPORT_ROOT" "$(dirname "$LOG")"

exec 9>"$REPORT_ROOT/.checkpoint_validation.lock"
if ! flock -n 9; then
  printf '%s\n' "checkpoint validation is already queued or running"
  exit 0
fi

printf '%s\n' \
  '{"scope":"research_only_checkpoint_validation","selection_fold":"fold1_only","fold2_used_for_selection":false,"direct_mlp_challenge_required_at_every_horizon":true,"test_used_for_head_selection":false,"live_orders_allowed":false}' \
  > "$REPORT_ROOT/checkpoint_validation_contract.json"

until [[ -f "$REPORT_ROOT/PIPELINE_COMPLETE" ]]; do
  if [[ -f "$REPORT_ROOT/PIPELINE_FAILED" ]]; then
    printf '%s\n' "main pipeline failed; checkpoint validation will not run" >&2
    exit 3
  fi
  sleep "$WAIT_SECONDS"
done

evaluate_node_checkpoint() {
  local fold="$1"
  local model_name="$2"
  local epoch="$3"
  local model_dir="$MODEL_ROOT/$model_name/epoch_${epoch}"
  local output_root="$REPORT_ROOT/checkpoint_eval/$fold"
  local summary="$output_root/epoch_${epoch}/summary.json"
  if [[ -f "$summary" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/evaluate_node_prediction.py \
    --model-dir "$model_dir" \
    --output-dir "$output_root" \
    --horizons 1,2,3,5,10 \
    --mask-strategy mixed \
    --max-steps 0 \
    --edge-cache-workers "$EDGE_CACHE_WORKERS" \
    --device "$DEVICE" \
    --seed "$SEED" \
    --cache-dir "$OHLCV" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

evaluate_transition_checkpoint() {
  local fold="$1"
  local model_name="$2"
  local epoch="$3"
  local model_dir="$MODEL_ROOT/$model_name/epoch_${epoch}"
  local output="$REPORT_ROOT/checkpoint_transition_eval/$fold/epoch_${epoch}"
  if [[ -f "$output/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/evaluate_trained_market_transition_auxiliary.py \
    --model-dir "$model_dir" \
    --output-dir "$output" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --batch-size "$TRANSITION_BATCH_SIZE" \
    --device "$DEVICE" \
    --cache-dir "$OHLCV" \
    --external-cache-dir data/external_cache \
    2>&1 | tee -a "$LOG"
}

FOLD1="${RUN_NAME}_fold1_20231229_to_20241230"
FOLD2="${RUN_NAME}_fold2_20241230_to_20260710"

"$PYTHON_BIN" scripts/mark_json_research_only.py \
  "$REPORT_ROOT/summary.json" \
  "$REPORT_ROOT/node_eval/$FOLD1/summary.json" \
  "$REPORT_ROOT/node_eval/$FOLD2/summary.json" \
  2>&1 | tee -a "$LOG"

for epoch in 008 016; do
  evaluate_node_checkpoint fold1 "$FOLD1" "$epoch"
  evaluate_transition_checkpoint fold1 "$FOLD1" "$epoch"
done
for epoch in 008 016; do
  evaluate_node_checkpoint fold2 "$FOLD2" "$epoch"
  evaluate_transition_checkpoint fold2 "$FOLD2" "$epoch"
done

"$PYTHON_BIN" scripts/summarize_checkpoint_epochs.py \
  --fold1 "epoch8=$REPORT_ROOT/checkpoint_eval/fold1/epoch_008/summary.json" \
  --fold1 "epoch16=$REPORT_ROOT/checkpoint_eval/fold1/epoch_016/summary.json" \
  --fold1 "epoch24=$REPORT_ROOT/node_eval/$FOLD1/summary.json" \
  --fold2 "epoch8=$REPORT_ROOT/checkpoint_eval/fold2/epoch_008/summary.json" \
  --fold2 "epoch16=$REPORT_ROOT/checkpoint_eval/fold2/epoch_016/summary.json" \
  --fold2 "epoch24=$REPORT_ROOT/node_eval/$FOLD2/summary.json" \
  --output-dir "$REPORT_ROOT/checkpoint_selection" \
  2>&1 | tee -a "$LOG"

"$PYTHON_BIN" scripts/select_systemic_checkpoint.py \
  --fold1-node "epoch8=$REPORT_ROOT/checkpoint_eval/fold1/epoch_008/summary.json" \
  --fold1-node "epoch16=$REPORT_ROOT/checkpoint_eval/fold1/epoch_016/summary.json" \
  --fold1-node "epoch24=$REPORT_ROOT/node_eval/$FOLD1/summary.json" \
  --fold1-transition "epoch8=$REPORT_ROOT/checkpoint_transition_eval/fold1/epoch_008/summary.json" \
  --fold1-transition "epoch16=$REPORT_ROOT/checkpoint_transition_eval/fold1/epoch_016/summary.json" \
  --fold1-transition "epoch24=$REPORT_ROOT/trained_transition_eval/fold1/summary.json" \
  --fold2-node "epoch8=$REPORT_ROOT/checkpoint_eval/fold2/epoch_008/summary.json" \
  --fold2-node "epoch16=$REPORT_ROOT/checkpoint_eval/fold2/epoch_016/summary.json" \
  --fold2-node "epoch24=$REPORT_ROOT/node_eval/$FOLD2/summary.json" \
  --fold2-transition "epoch8=$REPORT_ROOT/checkpoint_transition_eval/fold2/epoch_008/summary.json" \
  --fold2-transition "epoch16=$REPORT_ROOT/checkpoint_transition_eval/fold2/epoch_016/summary.json" \
  --fold2-transition "epoch24=$REPORT_ROOT/trained_transition_eval/fold2/summary.json" \
  --fold1-direct-summary "$DIRECT_REPORT_ROOT/fold1/summary.json" \
  --fold2-direct-summary "$DIRECT_REPORT_ROOT/fold2/summary.json" \
  --output-dir "$REPORT_ROOT/deployment_checkpoint_selection" \
  2>&1 | tee -a "$LOG"

touch "$REPORT_ROOT/CHECKPOINT_VALIDATION_COMPLETE"
printf '%s\n' "$RUN_NAME checkpoint validation complete" | tee -a "$LOG"
