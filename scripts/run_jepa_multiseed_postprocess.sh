#!/usr/bin/env bash
set -Eeuo pipefail

SEED="${1:?usage: $0 SEED [--worker]}"
MODE="${2:-}"
ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON="${PYTHON:-/workspace/venvs/stock-v2-cu128/bin/python}"
DEVICE="${DEVICE:-cuda}"
EDGE_CACHE_WORKERS="${EDGE_CACHE_WORKERS:-16}"
RUN_NAME="strict_causal453_path_v2_path_w12_p005_l025_skip_seed${SEED}"
REPORTS_ROOT="reports/walk_forward_causal453_path_multiseed_seed${SEED}_20260714"
MODELS_ROOT="models/walk_forward_causal453_path_multiseed_seed${SEED}_20260714"
POST_ROOT="$REPORTS_ROOT/postprocess_latent_head"
DIRECT_ROOT="reports/walk_forward_causal453_path_v2_20260713/direct_recovery_chunked_v1"
LOG_PATH="ops/training/jepa_multiseed_seed${SEED}_postprocess_20260714.log"
PID_PATH="ops/training/jepa_multiseed_seed${SEED}_postprocess_20260714.pid"
FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"

cd "$ROOT"
mkdir -p "$POST_ROOT" ops/training

if [[ "$MODE" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "seed $SEED postprocess already running: $(cat "$PID_PATH")"
    exit 0
  fi
  if [[ "$(uname -s)" == "Darwin" ]]; then
    nohup caffeinate -dimsu env \
      ROOT="$ROOT" PYTHON="$PYTHON" DEVICE="$DEVICE" \
      EDGE_CACHE_WORKERS="$EDGE_CACHE_WORKERS" \
      bash "$0" "$SEED" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  else
    nohup env \
      ROOT="$ROOT" PYTHON="$PYTHON" DEVICE="$DEVICE" \
      EDGE_CACHE_WORKERS="$EDGE_CACHE_WORKERS" \
      bash "$0" "$SEED" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  fi
  echo "$!" > "$PID_PATH"
  echo "seed $SEED postprocess queued: $!"
  exit 0
fi

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=2701
if [[ "$DEVICE" == "cuda" ]]; then
  export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
else
  export PYTORCH_ENABLE_MPS_FALLBACK=1
fi

rm -f \
  "$POST_ROOT/POSTPROCESS_COMPLETE" \
  "$POST_ROOT/POSTPROCESS_FAILED" \
  "$POST_ROOT/FINISHED_AT" \
  "$POST_ROOT/exit_status.txt"

on_exit() {
  status=$?
  trap - EXIT
  date '+%Y-%m-%dT%H:%M:%S%z' > "$POST_ROOT/FINISHED_AT"
  printf '%s\n' "$status" > "$POST_ROOT/exit_status.txt"
  if [[ "$status" -eq 0 ]]; then
    touch "$POST_ROOT/POSTPROCESS_COMPLETE"
  else
    touch "$POST_ROOT/POSTPROCESS_FAILED"
  fi
  exit "$status"
}
trap on_exit EXIT

while [[ ! -f "$REPORTS_ROOT/TRAINING_COMPLETE" ]]; do
  if [[ -f "$REPORTS_ROOT/TRAINING_FAILED" ]]; then
    echo "seed $SEED training failed; refusing postprocess" >&2
    exit 4
  fi
  sleep 30
done

test -x "$PYTHON"
test -f "$DIRECT_ROOT/direct/$FOLD1/daily_metrics.csv"
test -f "$DIRECT_ROOT/direct_nograph/$FOLD2/daily_metrics.csv"

{
  date '+%Y-%m-%dT%H:%M:%S%z'
  if command -v sha256sum >/dev/null 2>&1; then
    sha256sum \
      scripts/benchmark_latent_trajectory_path_head.py \
      scripts/attach_latent_path_head_summary.py \
      scripts/compare_latent_path_head_direct.py \
      scripts/compare_direct_state_mlp.py \
      scripts/combine_direct_state_challenges.py \
      scripts/gate_shadow_candidate.py \
      "$0"
  else
    shasum -a 256 \
      scripts/benchmark_latent_trajectory_path_head.py \
      scripts/attach_latent_path_head_summary.py \
      scripts/compare_latent_path_head_direct.py \
      scripts/compare_direct_state_mlp.py \
      scripts/combine_direct_state_challenges.py \
      scripts/gate_shadow_candidate.py \
      "$0"
  fi
} > "$POST_ROOT/source_sha256.txt"

for fold in "$FOLD1" "$FOLD2"; do
  model_name="${RUN_NAME}_${fold}"
  model_dir="$MODELS_ROOT/$model_name"
  node_dir="$REPORTS_ROOT/node_eval/$model_name"
  head_dir="$POST_ROOT/${fold%%_*}"
  raw_graph="$POST_ROOT/direct_vs_jepa/$fold/graph"
  raw_nograph="$POST_ROOT/direct_vs_jepa/$fold/nograph"
  raw_combined="$POST_ROOT/direct_vs_jepa/$fold/combined"
  adapted_dir="$POST_ROOT/gate_inputs/$fold"
  mkdir -p "$head_dir" "$raw_graph" "$raw_nograph" "$raw_combined" "$adapted_dir"

  test -f "$model_dir/graph_jepa_real.pt"
  test -f "$node_dir/summary.json"
  if [[ ! -f "$head_dir/summary.json" ]]; then
    "$PYTHON" scripts/benchmark_latent_trajectory_path_head.py \
      --model-dir "$model_dir" \
      --output-dir "$head_dir" \
      --horizons 1,2,3,5,10 \
      --validation-days 126 \
      --epochs 8 \
      --patience 2 \
      --hidden-dim 256 \
      --dropout 0.05 \
      --learning-rate 0.0003 \
      --batch-size 8 \
      --liquidity-top-k 300 \
      --latent-blend-weight 0.5 \
      --max-test-steps 0 \
      --edge-cache-workers "$EDGE_CACHE_WORKERS" \
      --device "$DEVICE" \
      --seed 2701 \
      --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
      --external-cache-dir data/external_cache
  fi

  "$PYTHON" scripts/compare_direct_state_mlp.py \
    --direct-daily "$DIRECT_ROOT/direct/$fold/daily_metrics.csv" \
    --jepa-daily "$node_dir/future_rollout.csv" \
    --output-dir "$raw_graph"
  "$PYTHON" scripts/compare_direct_state_mlp.py \
    --direct-daily "$DIRECT_ROOT/direct_nograph/$fold/daily_metrics.csv" \
    --jepa-daily "$node_dir/future_rollout.csv" \
    --output-dir "$raw_nograph"
  "$PYTHON" scripts/combine_direct_state_challenges.py \
    --challenger "graph=$raw_graph/comparison.json" \
    --challenger "nograph=$raw_nograph/comparison.json" \
    --output-dir "$raw_combined"
  "$PYTHON" scripts/attach_latent_path_head_summary.py \
    --node-summary "$node_dir/summary.json" \
    --head-summary "$head_dir/summary.json" \
    --output "$adapted_dir/node_summary.json"
  "$PYTHON" scripts/compare_latent_path_head_direct.py \
    --original-combined "$raw_combined/comparison.json" \
    --head-daily "$head_dir/daily_metrics.csv" \
    --challenger "graph=$DIRECT_ROOT/direct/$fold/daily_metrics.csv" \
    --challenger "nograph=$DIRECT_ROOT/direct_nograph/$fold/daily_metrics.csv" \
    --output "$adapted_dir/direct_comparison.json"
done

printf '%s\n' \
  '{"scope":"read_only_shadow_multiseed_validation","live_orders_allowed":false}' \
  > "$POST_ROOT/safety_contract.json"

set +e
"$PYTHON" scripts/gate_shadow_candidate.py \
  --walk-forward-summary "$REPORTS_ROOT/summary.json" \
  --node-summary "$POST_ROOT/gate_inputs/$FOLD1/node_summary.json" \
  --node-summary "$POST_ROOT/gate_inputs/$FOLD2/node_summary.json" \
  --direct-comparison "$POST_ROOT/gate_inputs/$FOLD1/direct_comparison.json" \
  --direct-comparison "$POST_ROOT/gate_inputs/$FOLD2/direct_comparison.json" \
  --dataset-audit reports/news_krx500_dart_pit_v2_integrity_20260712.json \
  --ohlcv-audit reports/ohlcv_causal453_release_audit_20260713.json \
  --output-dir "$POST_ROOT/shadow_gate"
GATE_STATUS=$?
set -e
printf '%s\n' "$GATE_STATUS" > "$POST_ROOT/shadow_gate/exit_status.txt"
touch "$POST_ROOT/GATE_COMPLETE"
echo "seed $SEED postprocess complete gate_status=$GATE_STATUS"
