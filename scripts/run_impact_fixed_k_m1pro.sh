#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="$ROOT/.venv-mps/bin/python"
WAIT_ROOT="reports/direct_state_impact_v1_20260714"
MODEL_ROOT="models/walk_forward_causal453_path_v2_20260713"
HEAD_ROOT="reports/impact_head_weight95_seed17_20260714"
OUTPUT_ROOT="reports/impact_head_fixed_k_seed17_20260714"
LOG_PATH="ops/training/impact_fixed_k_seed17_m1pro_20260714.log"
PID_PATH="ops/training/impact_fixed_k_seed17_m1pro_20260714.pid"
RUN_NAME="strict_causal453_path_v2_path_w12_p005_l025_skip_seed17"
FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT" ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "fixed-k impact evaluation already queued: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup caffeinate -dimsu bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "fixed-k impact evaluation queued: $!"
  exit 0
fi

while [[ ! -f "$WAIT_ROOT/EVALUATION_COMPLETE" ]]; do
  if [[ -f "$WAIT_ROOT/EVALUATION_FAILED" ]]; then
    echo "direct impact comparison failed; refusing dependent fixed-k run" >&2
    exit 4
  fi
  sleep 30
done

export PYTHONUNBUFFERED=1
export PYTORCH_ENABLE_MPS_FALLBACK=1
rm -f "$OUTPUT_ROOT/EVALUATION_COMPLETE" "$OUTPUT_ROOT/EVALUATION_FAILED"
on_exit() {
  status=$?
  trap - EXIT
  date '+%Y-%m-%dT%H:%M:%S%z' > "$OUTPUT_ROOT/FINISHED_AT"
  printf '%s\n' "$status" > "$OUTPUT_ROOT/exit_status.txt"
  if [[ "$status" -eq 0 ]]; then
    touch "$OUTPUT_ROOT/EVALUATION_COMPLETE"
  else
    touch "$OUTPUT_ROOT/EVALUATION_FAILED"
  fi
  exit "$status"
}
trap on_exit EXIT

{
  date '+%Y-%m-%dT%H:%M:%S%z'
  shasum -a 256 scripts/evaluate_impact_head_fixed_k.py "$0"
} > "$OUTPUT_ROOT/source_sha256.txt"

for fold in "$FOLD1" "$FOLD2"; do
  short_fold="${fold%%_*}"
  output_dir="$OUTPUT_ROOT/$short_fold"
  if [[ ! -f "$output_dir/summary.json" ]]; then
    "$PYTHON" scripts/evaluate_impact_head_fixed_k.py \
      --model-dir "$MODEL_ROOT/${RUN_NAME}_${fold}" \
      --head-path "$HEAD_ROOT/$short_fold/impact_trajectory_head.pt" \
      --output-dir "$output_dir" \
      --horizons 1,2,3,5,10 \
      --counts 1,3,5 \
      --liquidity-top-k 300 \
      --batch-size 8 \
      --edge-cache-workers 8 \
      --device mps \
      --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
      --external-cache-dir data/external_cache
  fi
done

printf '%s\n' \
  '{"scope":"read_only_fixed_count_diagnostic","live_orders_allowed":false}' \
  > "$OUTPUT_ROOT/safety_contract.json"
