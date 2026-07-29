#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON_BIN="${PYTHON_BIN:-/usr/local/bin/python}"
DEVICE="${DEVICE:-cuda}"
RUN_NAME="final_aligned_jepa_opmask_aux_v1_seed17"
OLD_ROOT="models/walk_forward_causal453_path_v2_20260713"
NEW_ROOT="models/${RUN_NAME}"
REPORT_ROOT="reports/${RUN_NAME}/mask_controls"
EVENT_PATH="data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl"
FUNDAMENTALS="data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl"
FOLD1_OLD="strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold1_20231229_to_20241230"
FOLD2_OLD="strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold2_20241230_to_20260710"
FOLD1_NEW="${RUN_NAME}_fold1_20231229_to_20241230"
FOLD2_NEW="${RUN_NAME}_fold2_20241230_to_20260710"

cd "$ROOT"
mkdir -p "$REPORT_ROOT"
export PYTHONUNBUFFERED=1

evaluate() {
  local model_dir="$1"
  local output_dir="$2"
  local mask="$3"
  local model_name
  model_name="$(basename "$model_dir")"
  if [[ -f "$output_dir/$model_name/summary.json" ]]; then
    return
  fi
  "$PYTHON_BIN" scripts/evaluate_node_prediction.py \
    --model-dir "$model_dir" \
    --output-dir "$output_dir" \
    --horizons 1,2,3,5,10 \
    --mask-strategy "$mask" \
    --max-steps 0 \
    --edge-cache-workers 16 \
    --device "$DEVICE" \
    --seed 17 \
    --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
    --event-path "$EVENT_PATH" \
    --fundamental-path "$FUNDAMENTALS" \
    --fundamental-lag-days 1 \
    --investor-cache-dir data/kiwoom_investor_cache \
    --investor-flow-lag-days 1 \
    --external-preset kr_global_rates \
    --external-lag-days 1 \
    --external-cache-dir data/external_cache
}

evaluate "$NEW_ROOT/$FOLD1_NEW" "$REPORT_ROOT/candidate_mixed" mixed
evaluate "$NEW_ROOT/$FOLD2_NEW" "$REPORT_ROOT/candidate_mixed" mixed
evaluate "$OLD_ROOT/$FOLD1_OLD" "$REPORT_ROOT/baseline_operational" operational_mixed
evaluate "$OLD_ROOT/$FOLD2_OLD" "$REPORT_ROOT/baseline_operational" operational_mixed

"$PYTHON_BIN" scripts/compare_final_aligned_node.py \
  --baseline-mixed "reports/walk_forward_causal453_path_v2_20260713/node_eval/$FOLD1_OLD/summary.json" \
  --baseline-mixed "reports/walk_forward_causal453_path_v2_20260713/node_eval/$FOLD2_OLD/summary.json" \
  --candidate-mixed "$REPORT_ROOT/candidate_mixed/$FOLD1_NEW/summary.json" \
  --candidate-mixed "$REPORT_ROOT/candidate_mixed/$FOLD2_NEW/summary.json" \
  --baseline-operational "$REPORT_ROOT/baseline_operational/$FOLD1_OLD/summary.json" \
  --baseline-operational "$REPORT_ROOT/baseline_operational/$FOLD2_OLD/summary.json" \
  --candidate-operational "reports/$RUN_NAME/node_eval/$FOLD1_NEW/summary.json" \
  --candidate-operational "reports/$RUN_NAME/node_eval/$FOLD2_NEW/summary.json" \
  --output "$REPORT_ROOT/comparison.json"

touch "$REPORT_ROOT/CONTROLS_COMPLETE"
