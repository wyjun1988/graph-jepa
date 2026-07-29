#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/stock-v2-liquidity"
PRIMARY="/workspace/stock-v2"
PYTHON="/root/venvs/news-vllm-cu128/bin/python"
RUN_NAME="strict_causal453_path_v2_path_w12_p005_l025_skip_seed17"
REPORT_NAME="walk_forward_causal453_path_v2_20260713"
MODEL_NAME="walk_forward_causal453_path_v2_20260713"
REPORT_ROOT="$PRIMARY/reports/$REPORT_NAME"
MODEL_ROOT="$PRIMARY/models/$MODEL_NAME"
RECOVERY_ROOT="$REPORT_ROOT/direct_recovery_chunked_v1"
FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"

cd "$ROOT"
mkdir -p "$RECOVERY_ROOT"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

on_error() {
  local status=$?
  printf '{"status":"failed","exit_status":%d,"live_orders_allowed":false}\n' \
    "$status" > "$RECOVERY_ROOT/RECOVERY_FAILED"
  exit "$status"
}
trap on_error ERR

SELECTION="$REPORT_ROOT/checkpoint_selection/selection.json"
SELECTED_EPOCH="$($PYTHON -c "import json; print(json.load(open('$SELECTION'))['selected_label'])")"
if [[ "$SELECTED_EPOCH" != "epoch24" ]]; then
  echo "recovery contract expected epoch24, found $SELECTED_EPOCH" >&2
  exit 4
fi

printf '%s\n' \
  '{"scope":"direct_challenge_and_gate_recovery_only","training_mutated":false,"fold2_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RECOVERY_ROOT/safety_contract.json"
sha256sum \
  scripts/benchmark_direct_baselines.py \
  scripts/benchmark_direct_state_mlp.py \
  scripts/compare_direct_state_mlp.py \
  scripts/combine_direct_state_challenges.py \
  scripts/gate_shadow_candidate.py \
  scripts/recover_path_confirmation_direct_gpu.sh \
  "$REPORT_ROOT/run_contract.json" \
  "$SELECTION" \
  > "$RECOVERY_ROOT/source_and_parent_sha256.txt"

for fold in "$FOLD1" "$FOLD2"; do
  model_name="${RUN_NAME}_${fold}"
  model_dir="$MODEL_ROOT/$model_name"
  node_dir="$REPORT_ROOT/node_eval/$model_name"
  direct_dir="$RECOVERY_ROOT/direct/$fold"
  direct_nograph_dir="$RECOVERY_ROOT/direct_nograph/$fold"
  comparison_dir="$RECOVERY_ROOT/direct_vs_jepa/$fold/graph"
  comparison_nograph_dir="$RECOVERY_ROOT/direct_vs_jepa/$fold/nograph"
  combined_dir="$RECOVERY_ROOT/direct_vs_jepa/$fold/combined"
  context_cache="/root/stock-v2-cache/direct_context_${model_name}_chunked_v1.npy"

  sha256sum "$model_dir/graph_jepa_real.pt" > "$RECOVERY_ROOT/${fold}_model_sha256.txt"
  if [[ ! -f "$direct_dir/summary.json" ]]; then
    "$PYTHON" scripts/benchmark_direct_state_mlp.py \
      --model-dir "$model_dir" \
      --output-dir "$direct_dir" \
      --horizons 1,2,3,5,10 \
      --validation-days 126 \
      --hidden-dim 512 \
      --layers 3 \
      --epochs 16 \
      --patience 4 \
      --batch-size 16384 \
      --device cuda \
      --cache-dir "$PRIMARY/data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv" \
      --external-cache-dir "$PRIMARY/data/external_cache" \
      --context-cache "$context_cache"
  fi
  if [[ ! -f "$direct_nograph_dir/summary.json" ]]; then
    "$PYTHON" scripts/benchmark_direct_state_mlp.py \
      --model-dir "$model_dir" \
      --output-dir "$direct_nograph_dir" \
      --horizons 1,2,3,5,10 \
      --validation-days 126 \
      --hidden-dim 512 \
      --layers 3 \
      --epochs 16 \
      --patience 4 \
      --batch-size 16384 \
      --device cuda \
      --without-graph \
      --cache-dir "$PRIMARY/data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv" \
      --external-cache-dir "$PRIMARY/data/external_cache" \
      --context-cache "$context_cache"
  fi
  "$PYTHON" scripts/compare_direct_state_mlp.py \
    --direct-daily "$direct_dir/daily_metrics.csv" \
    --jepa-daily "$node_dir/future_rollout.csv" \
    --output-dir "$comparison_dir"
  "$PYTHON" scripts/compare_direct_state_mlp.py \
    --direct-daily "$direct_nograph_dir/daily_metrics.csv" \
    --jepa-daily "$node_dir/future_rollout.csv" \
    --output-dir "$comparison_nograph_dir"
  "$PYTHON" scripts/combine_direct_state_challenges.py \
    --challenger "graph=$comparison_dir/comparison.json" \
    --challenger "nograph=$comparison_nograph_dir/comparison.json" \
    --output-dir "$combined_dir"
done

set +e
"$PYTHON" scripts/gate_shadow_candidate.py \
  --walk-forward-summary "$REPORT_ROOT/summary.json" \
  --node-summary "$REPORT_ROOT/node_eval/${RUN_NAME}_${FOLD1}/summary.json" \
  --node-summary "$REPORT_ROOT/node_eval/${RUN_NAME}_${FOLD2}/summary.json" \
  --direct-comparison "$RECOVERY_ROOT/direct_vs_jepa/$FOLD1/combined/comparison.json" \
  --direct-comparison "$RECOVERY_ROOT/direct_vs_jepa/$FOLD2/combined/comparison.json" \
  --dataset-audit "$PRIMARY/reports/news_krx500_dart_pit_v2_integrity_20260712.json" \
  --ohlcv-audit "$PRIMARY/reports/ohlcv_causal453_release_audit_20260713.json" \
  --output-dir "$RECOVERY_ROOT/shadow_gate"
GATE_STATUS=$?
set -e
printf '%s\n' "$GATE_STATUS" > "$RECOVERY_ROOT/shadow_gate/exit_status.txt"
touch "$RECOVERY_ROOT/RECOVERY_COMPLETE"
echo "direct challenge recovery complete gate_status=$GATE_STATUS"
