#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/stock-v2"
PYTHON="/workspace/venvs/stock-v2-cu128/bin/python"
WAIT_ROOT="reports/walk_forward_causal453_path_multiseed_seed53_20260714/impact_fixed_k"
MODEL_ROOT="models/walk_forward_causal453_path_v2_20260713"
OUTPUT_ROOT="reports/direct_impact_equal_objective_v1_20260714"
LOG_PATH="ops/training/direct_impact_equal_a5000_20260714.log"
PID_PATH="ops/training/direct_impact_equal_a5000_20260714.pid"
RUN_NAME="strict_causal453_path_v2_path_w12_p005_l025_skip_seed17"
FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT" data/context_cache ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "equal-objective direct impact job already queued: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "equal-objective direct impact job queued: $!"
  exit 0
fi

while [[ ! -f "$WAIT_ROOT/EVALUATION_COMPLETE" ]]; do
  if [[ -f "$WAIT_ROOT/EVALUATION_FAILED" ]]; then
    echo "seed 53 fixed-k run failed; refusing dependent direct challenger" >&2
    exit 4
  fi
  sleep 30
done

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2701
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -f "$OUTPUT_ROOT/TRAINING_COMPLETE" "$OUTPUT_ROOT/TRAINING_FAILED"
on_exit() {
  status=$?
  trap - EXIT
  date '+%Y-%m-%dT%H:%M:%S%z' > "$OUTPUT_ROOT/FINISHED_AT"
  printf '%s\n' "$status" > "$OUTPUT_ROOT/exit_status.txt"
  if [[ "$status" -eq 0 ]]; then
    touch "$OUTPUT_ROOT/TRAINING_COMPLETE"
  else
    touch "$OUTPUT_ROOT/TRAINING_FAILED"
  fi
  exit "$status"
}
trap on_exit EXIT

{
  date '+%Y-%m-%dT%H:%M:%S%z'
  sha256sum scripts/benchmark_direct_impact_head.py "$0"
} > "$OUTPUT_ROOT/source_sha256.txt"

for fold in "$FOLD1" "$FOLD2"; do
  short_fold="${fold%%_*}"
  for mode in graph nograph; do
    output_dir="$OUTPUT_ROOT/$mode/$short_fold"
    extra_arg=""
    if [[ "$mode" == "nograph" ]]; then
      extra_arg="--without-graph"
    fi
    if [[ ! -f "$output_dir/summary.json" ]]; then
      "$PYTHON" scripts/benchmark_direct_impact_head.py \
        --model-dir "$MODEL_ROOT/${RUN_NAME}_${fold}" \
        --output-dir "$output_dir" \
        --horizons 1,2,3,5,10 \
        --impact-fractions 0.05,0.10,0.20 \
        --train-impact-fraction 0.10 \
        --validation-days 126 \
        --epochs 8 \
        --patience 2 \
        --hidden-dim 256 \
        --dropout 0.05 \
        --learning-rate 0.0003 \
        --weight-decay 0.0001 \
        --batch-dates 32 \
        --eval-batch-dates 64 \
        --liquidity-top-k 300 \
        --impact-rank-weight 0.30 \
        --impact-focal-weight 0.25 \
        --tail-rank-weight 0.30 \
        --tail-direction-weight 0.10 \
        --all-rank-weight 0.05 \
        --feature-workers 16 \
        --device cuda \
        --amp \
        --seed 2701 \
        --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
        --external-cache-dir data/external_cache \
        --context-cache "data/context_cache/direct_impact_equal_${short_fold}_20260714.npy" \
        $extra_arg
    fi
  done
done

printf '%s\n' \
  '{"scope":"read_only_equal_objective_direct_challenger","live_orders_allowed":false}' \
  > "$OUTPUT_ROOT/safety_contract.json"
