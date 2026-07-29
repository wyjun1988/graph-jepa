#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="$ROOT/.venv-mps/bin/python"
MODELS_ROOT="models/walk_forward_causal453_path_v2_20260713"
OUTPUT_ROOT="reports/impact_head_weight95_seed17_20260714"
LOG_PATH="ops/training/jepa_impact_seed17_m1pro_20260714.log"
PID_PATH="ops/training/jepa_impact_seed17_m1pro_20260714.pid"
RUN_NAME="strict_causal453_path_v2_path_w12_p005_l025_skip_seed17"
FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT" ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "seed 17 impact job already running: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup caffeinate -dimsu bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "seed 17 impact job started: $!"
  exit 0
fi

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2701
export PYTORCH_ENABLE_MPS_FALLBACK=1
rm -f "$OUTPUT_ROOT/IMPACT_COMPLETE" "$OUTPUT_ROOT/IMPACT_FAILED"

on_exit() {
  status=$?
  trap - EXIT
  date '+%Y-%m-%dT%H:%M:%S%z' > "$OUTPUT_ROOT/FINISHED_AT"
  printf '%s\n' "$status" > "$OUTPUT_ROOT/exit_status.txt"
  if [[ "$status" -eq 0 ]]; then
    touch "$OUTPUT_ROOT/IMPACT_COMPLETE"
  else
    touch "$OUTPUT_ROOT/IMPACT_FAILED"
  fi
  exit "$status"
}
trap on_exit EXIT

{
  date '+%Y-%m-%dT%H:%M:%S%z'
  shasum -a 256 scripts/benchmark_impact_trajectory_head.py "$0"
} > "$OUTPUT_ROOT/source_sha256.txt"

for fold in "$FOLD1" "$FOLD2"; do
  model_dir="$MODELS_ROOT/${RUN_NAME}_${fold}"
  output_dir="$OUTPUT_ROOT/${fold%%_*}"
  test -f "$model_dir/graph_jepa_real.pt"
  if [[ ! -f "$output_dir/summary.json" ]]; then
    "$PYTHON" scripts/benchmark_impact_trajectory_head.py \
      --model-dir "$model_dir" \
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
      --batch-size 8 \
      --liquidity-top-k 300 \
      --latent-blend-weight 0.5 \
      --impact-rank-weight 0.30 \
      --impact-focal-weight 0.25 \
      --tail-rank-weight 0.30 \
      --tail-direction-weight 0.10 \
      --all-rank-weight 0.05 \
      --edge-cache-workers 8 \
      --device mps \
      --seed 2701 \
      --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
      --external-cache-dir data/external_cache
  fi
done

printf '%s\n' \
  '{"scope":"read_only_impact_research","live_orders_allowed":false}' \
  > "$OUTPUT_ROOT/safety_contract.json"
