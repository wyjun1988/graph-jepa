#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
RUN_NAME="broad_transition_jepa_v6_systemic_seed17_rtx4000ada_20260714"
FOLD="${FOLD:-fold1}"
case "$FOLD" in
  fold1)
    FOLD_NAME="${RUN_NAME}_fold1_20231229_to_20241230"
    ;;
  fold2)
    FOLD_NAME="${RUN_NAME}_fold2_20241230_to_20260710"
    ;;
  *)
    printf 'unsupported fold: %s\n' "$FOLD" >&2
    exit 2
    ;;
esac
MODEL_DIR="$ROOT/models/$RUN_NAME/$FOLD_NAME"
OUTPUT_DIR="$ROOT/reports/direct_state_mlp_${RUN_NAME}_20260714/$FOLD"
LOG="$ROOT/logs/direct_state_mlp_${RUN_NAME}_${FOLD}_m1max_20260714.log"
CONTEXT_CACHE="$OUTPUT_DIR/direct_context_graph.npz"
OHLCV="$ROOT/data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$OUTPUT_DIR" "$(dirname "$LOG")"

printf '%s\n' \
  "{\"scope\":\"research_only_direct_return_state_baseline\",\"fold\":\"$FOLD\",\"selection_data\":\"fit_and_validation_only\",\"test_used_for_selection\":false,\"live_orders_allowed\":false}" \
  > "$OUTPUT_DIR/experiment_contract.json"

if [[ -f "$OUTPUT_DIR/EXPERIMENT_COMPLETE" ]]; then
  printf '%s\n' "direct state $FOLD baseline already complete"
  exit 0
fi

"$PYTHON_BIN" scripts/benchmark_direct_state_mlp.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --horizons 1,2,3,5,10 \
  --validation-days 126 \
  --hidden-dim 512 \
  --layers 3 \
  --dropout 0.05 \
  --epochs 16 \
  --patience 4 \
  --batch-size 16384 \
  --learning-rate 0.0003 \
  --weight-decay 0.0001 \
  --device mps \
  --seed 17 \
  --feature-workers 8 \
  --cache-dir "$OHLCV" \
  --external-cache-dir "$ROOT/data/external_cache" \
  --context-cache "$CONTEXT_CACHE" \
  2>&1 | tee -a "$LOG"

"$PYTHON_BIN" - "$OUTPUT_DIR/summary.json" <<'PY'
import json
import sys

payload = json.load(open(sys.argv[1], encoding="utf-8"))
assert payload["live_orders_allowed"] is False
assert payload["test_used_for_selection"] is False
assert set(payload["horizons"]) == {"1", "2", "3", "5", "10"}
PY

touch "$OUTPUT_DIR/EXPERIMENT_COMPLETE"
