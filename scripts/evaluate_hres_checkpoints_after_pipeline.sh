#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/stock-v2"
PYTHON="/root/venvs/news-vllm-cu128/bin/python"
RUN_NAME="strict_causal453_hres_v2_seed17"
REPORTS_ROOT="reports/walk_forward_causal453_hres_v2_20260713"
MODELS_ROOT="models/walk_forward_causal453_hres_v2_20260713"
OUTPUT_ROOT="reports/checkpoint_selection_causal453_hres_v2_20260713"
PIPELINE_MARKER="$REPORTS_ROOT/PIPELINE_COMPLETE"
DRIVER_PID_FILE="reports/causal453_hres_v2_driver_20260713.pid"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

while [[ ! -f "$PIPELINE_MARKER" ]]; do
  if [[ -f "$DRIVER_PID_FILE" ]]; then
    DRIVER_PID="$(cat "$DRIVER_PID_FILE")"
    if ! kill -0 "$DRIVER_PID" 2>/dev/null; then
      echo "strict pipeline exited without completion marker" >&2
      exit 3
    fi
  fi
  sleep 60
done

evaluate_checkpoint() {
  local fold="$1"
  local epoch="$2"
  local model_dir="$MODELS_ROOT/${RUN_NAME}_${fold}/epoch_${epoch}"
  local output_dir="$OUTPUT_ROOT/$fold/epoch_${epoch}"
  local summary="$output_dir/epoch_${epoch}/summary.json"
  local reference="$REPORTS_ROOT/node_eval/${RUN_NAME}_${fold}/summary.json"
  if [[ -f "$summary" ]] && "$PYTHON" - "$summary" "$reference" <<'PY'
import json
import sys

candidate = json.load(open(sys.argv[1]))
reference = json.load(open(sys.argv[2]))
fields = ("eval_start", "eval_end", "eval_steps")
raise SystemExit(
    0 if all(candidate.get(field) == reference.get(field) for field in fields) else 1
)
PY
  then
    echo "checkpoint already evaluated on full matched window: $fold epoch=$epoch"
    return
  fi
  "$PYTHON" scripts/evaluate_node_prediction.py \
    --model-dir "$model_dir" \
    --output-dir "$output_dir" \
    --horizons 1,2,3,5,10 \
    --mask-strategy mixed \
    --max-steps 0 \
    --edge-cache-workers 16 \
    --device cuda \
    --seed 17 \
    --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
    --external-cache-dir data/external_cache
}

for fold in fold1_20231229_to_20241230 fold2_20241230_to_20260710; do
  evaluate_checkpoint "$fold" 008
  evaluate_checkpoint "$fold" 016
done

FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"
"$PYTHON" scripts/summarize_checkpoint_epochs.py \
  --fold1 "epoch8=$OUTPUT_ROOT/$FOLD1/epoch_008/epoch_008/summary.json" \
  --fold1 "epoch16=$OUTPUT_ROOT/$FOLD1/epoch_016/epoch_016/summary.json" \
  --fold1 "epoch24=$REPORTS_ROOT/node_eval/${RUN_NAME}_${FOLD1}/summary.json" \
  --fold2 "epoch8=$OUTPUT_ROOT/$FOLD2/epoch_008/epoch_008/summary.json" \
  --fold2 "epoch16=$OUTPUT_ROOT/$FOLD2/epoch_016/epoch_016/summary.json" \
  --fold2 "epoch24=$REPORTS_ROOT/node_eval/${RUN_NAME}_${FOLD2}/summary.json" \
  --output-dir "$OUTPUT_ROOT"

touch "$OUTPUT_ROOT/CHECKPOINT_EVAL_COMPLETE"
echo "checkpoint epoch evaluation complete"
