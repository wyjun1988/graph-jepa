#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="$ROOT/.venv-mps/bin/python"
WAIT_ROOT="reports/direct_impact_fixed_k_m1pro_v1_20260714"
MODEL_ROOT="models/walk_forward_causal453_path_v2_20260713"
JEPA_ROOT="reports/impact_direction_v2_seed17_20260714"
FIXED_ROOT="reports/impact_direction_v2_fixed_k_seed17_20260714"
RUN_ROOT="reports/impact_direction_v2_jepa_m1pro_20260714"
LOG_PATH="ops/training/impact_direction_v2_m1pro_20260714.log"
PID_PATH="ops/training/impact_direction_v2_m1pro_20260714.pid"
RUN_NAME="strict_causal453_path_v2_path_w12_p005_l025_skip_seed17"
FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$JEPA_ROOT" "$FIXED_ROOT" ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "impact-direction v2 M1 Pro run already queued: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup caffeinate -dimsu bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "impact-direction v2 M1 Pro run queued: $!"
  exit 0
fi

while [[ ! -f "$WAIT_ROOT/EVALUATION_COMPLETE" ]]; do
  if [[ -f "$WAIT_ROOT/EVALUATION_FAILED" ]]; then
    echo "direct fixed-K dependency failed; refusing v2" >&2
    exit 4
  fi
  sleep 30
done

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2701
rm -f "$RUN_ROOT/EXPERIMENT_COMPLETE" "$RUN_ROOT/EXPERIMENT_FAILED"
on_exit() {
  status=$?
  trap - EXIT
  date '+%Y-%m-%dT%H:%M:%S%z' > "$RUN_ROOT/FINISHED_AT"
  printf '%s\n' "$status" > "$RUN_ROOT/exit_status.txt"
  if [[ "$status" -eq 0 ]]; then
    touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
  else
    touch "$RUN_ROOT/EXPERIMENT_FAILED"
  fi
  exit "$status"
}
trap on_exit EXIT
trap 'exit 143' TERM
trap 'exit 130' INT

shasum -a 256 \
  scripts/benchmark_impact_trajectory_head.py \
  scripts/evaluate_impact_head_fixed_k.py \
  "$0" > "$RUN_ROOT/source_sha256.txt"

for fold in "$FOLD1" "$FOLD2"; do
  short_fold="${fold%%_*}"
  model_dir="$MODEL_ROOT/${RUN_NAME}_${fold}"
  if [[ ! -f "$JEPA_ROOT/$short_fold/summary.json" ]]; then
    "$PYTHON" scripts/benchmark_impact_trajectory_head.py \
      --model-dir "$model_dir" \
      --output-dir "$JEPA_ROOT/$short_fold" \
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
      --impact-rank-weight 0.25 \
      --impact-focal-weight 0.20 \
      --tail-rank-weight 0.25 \
      --tail-direction-weight 0.25 \
      --all-rank-weight 0.05 \
      --tail-direction-magnitude-power 1.0 \
      --validation-score-mode magnitude_v2 \
      --edge-cache-workers 8 \
      --device mps \
      --seed 2701 \
      --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
      --external-cache-dir data/external_cache
  fi
  if [[ ! -f "$FIXED_ROOT/$short_fold/summary.json" ]]; then
    "$PYTHON" scripts/evaluate_impact_head_fixed_k.py \
      --model-dir "$model_dir" \
      --head-path "$JEPA_ROOT/$short_fold/impact_trajectory_head.pt" \
      --output-dir "$FIXED_ROOT/$short_fold" \
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
  '{"scope":"read_only_posthoc_impact_direction_v2","live_orders_allowed":false}' \
  > "$RUN_ROOT/safety_contract.json"
echo "impact-direction v2 JEPA M1 Pro experiment complete"
