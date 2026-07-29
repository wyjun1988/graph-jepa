#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON_BIN="$ROOT/.venv-mps/bin/python"
RUN_ROOT="reports/direct_systemic_transition_head_seed17_v2_20260714"

cd "$ROOT"
mkdir -p "$RUN_ROOT" ops/training

on_error() {
  local status=$?
  printf '{"status":"failed","exit_status":%d,"live_orders_allowed":false}\n' \
    "$status" > "$RUN_ROOT/PIPELINE_FAILED"
  exit "$status"
}
trap on_error ERR

printf '%s\n' \
  '{"scope":"posthoc_research_only","target_version":"broad_systemic_v2_robust_20260714","objective":"same loss and labels as JEPA systemic head","input":"causal stock moments plus external nodes","test_used_for_selection":false,"live_orders_allowed":false}' \
  > "$RUN_ROOT/experiment_contract.json"

COMMON_ARGS=(
  --horizons 1,2,3,5,10
  --validation-days 126
  --epochs 80
  --patience 10
  --hidden-dim 256
  --horizon-dim 16
  --dropout 0.10
  --learning-rate 0.0003
  --weight-decay 0.001
  --batch-size 128
  --eval-batch-size 512
  --device mps
  --seed 2701
  --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv
  --external-cache-dir data/external_cache
)

for fold in fold1 fold2; do
  if [[ "$fold" == "fold1" ]]; then
    MODEL_DIR="models/walk_forward_causal453_path_v2_20260713/strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold1_20231229_to_20241230"
  else
    MODEL_DIR="models/walk_forward_causal453_path_v2_20260713/strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold2_20241230_to_20260710"
  fi
  if [[ ! -s "$RUN_ROOT/$fold/summary.json" ]]; then
    "$PYTHON_BIN" scripts/benchmark_direct_systemic_transition_head.py \
      --model-dir "$MODEL_DIR" \
      --output-dir "$RUN_ROOT/$fold" \
      "${COMMON_ARGS[@]}"
  fi
done

rm -f "$RUN_ROOT/PIPELINE_FAILED"
touch "$RUN_ROOT/EXPERIMENT_COMPLETE"
printf '%s\n' "direct systemic transition head experiment complete"
