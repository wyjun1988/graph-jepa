#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PYTHON_BIN="${PYTHON_BIN:-/workspace/venvs/stock-v2-cu128/bin/python}"
RUN_NAME="${RUN_NAME:-frozen_downstream_multitask_v1_20260713}"
OUTPUT_ROOT="reports/${RUN_NAME}"
CACHE_ROOT="${CACHE_ROOT:-/tmp/${RUN_NAME}}"
COMMON_ARGS=(
  --horizons 1,2,3,5,10
  --variants raw,latent,raw_latent,raw_shuffled_latent
  --modes single,multi
  --validation-days 126
  --epochs 8
  --patience 2
  --batch-size 8192
  --hidden-dim 256
  --layers 2
  --dropout 0.05
  --learning-rate 0.0003
  --weight-decay 0.0001
  --feature-workers 16
  --device cuda
  --amp
)

mkdir -p "$OUTPUT_ROOT" "$CACHE_ROOT"

run_fold() {
  local fold_name="$1"
  local model_dir="$2"
  local test_end="$3"
  "$PYTHON_BIN" scripts/benchmark_frozen_downstream.py \
    --model-dir "$model_dir" \
    --output-dir "$OUTPUT_ROOT/$fold_name" \
    --raw-context-cache "$CACHE_ROOT/${fold_name}_raw.npy" \
    --latent-cache-dir "$CACHE_ROOT/${fold_name}_latent" \
    --test-end "$test_end" \
    "${COMMON_ARGS[@]}" \
    2>&1 | tee "$OUTPUT_ROOT/${fold_name}.log"
}

run_fold \
  fold1 \
  models/walk_forward_causal453_path_v2_20260713/strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold1_20231229_to_20241230 \
  2024-12-30

run_fold \
  fold2 \
  models/walk_forward_causal453_path_v2_20260713/strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold2_20241230_to_20260710 \
  2026-07-10

"$PYTHON_BIN" scripts/summarize_frozen_downstream.py \
  --fold "$OUTPUT_ROOT/fold1/summary.json" \
  --fold "$OUTPUT_ROOT/fold2/summary.json" \
  --output-dir "$OUTPUT_ROOT"

touch "$OUTPUT_ROOT/PIPELINE_COMPLETE"
