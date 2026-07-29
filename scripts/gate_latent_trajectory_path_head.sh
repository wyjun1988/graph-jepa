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
DIRECT_ROOT="$REPORT_ROOT/direct_recovery_chunked_v1"
HEAD_ROOT="$ROOT/reports/latent_trajectory_path_head_blend05_v1_20260713"
OUTPUT_ROOT="$REPORT_ROOT/latent_head_blend05_gate_v1"
FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"
printf '%s\n' \
  '{"scope":"read_only_shadow_gate_research","training_mutated":false,"live_orders_allowed":false}' \
  > "$OUTPUT_ROOT/safety_contract.json"
sha256sum \
  scripts/benchmark_latent_trajectory_path_head.py \
  scripts/attach_latent_path_head_summary.py \
  scripts/compare_latent_path_head_direct.py \
  scripts/gate_shadow_candidate.py \
  scripts/gate_latent_trajectory_path_head.sh \
  > "$OUTPUT_ROOT/source_sha256.txt"

for fold in "$FOLD1" "$FOLD2"; do
  model_name="${RUN_NAME}_${fold}"
  head_dir="$HEAD_ROOT/${fold%%_*}"
  expected_sha="$($PYTHON -c "import json; print(json.load(open('$head_dir/summary.json'))['parent_model_sha256'])")"
  actual_sha="$(sha256sum "$MODEL_ROOT/$model_name/graph_jepa_real.pt" | cut -d' ' -f1)"
  if [[ "$expected_sha" != "$actual_sha" ]]; then
    echo "latent head parent SHA mismatch for $fold" >&2
    exit 4
  fi
  mkdir -p "$OUTPUT_ROOT/$fold"
  "$PYTHON" scripts/attach_latent_path_head_summary.py \
    --node-summary "$REPORT_ROOT/node_eval/$model_name/summary.json" \
    --head-summary "$head_dir/summary.json" \
    --output "$OUTPUT_ROOT/$fold/node_summary.json"
  "$PYTHON" scripts/compare_latent_path_head_direct.py \
    --original-combined "$DIRECT_ROOT/direct_vs_jepa/$fold/combined/comparison.json" \
    --head-daily "$head_dir/daily_metrics.csv" \
    --challenger "graph=$DIRECT_ROOT/direct/$fold/daily_metrics.csv" \
    --challenger "nograph=$DIRECT_ROOT/direct_nograph/$fold/daily_metrics.csv" \
    --output "$OUTPUT_ROOT/$fold/direct_comparison.json"
done

set +e
"$PYTHON" scripts/gate_shadow_candidate.py \
  --walk-forward-summary "$REPORT_ROOT/summary.json" \
  --node-summary "$OUTPUT_ROOT/$FOLD1/node_summary.json" \
  --node-summary "$OUTPUT_ROOT/$FOLD2/node_summary.json" \
  --direct-comparison "$OUTPUT_ROOT/$FOLD1/direct_comparison.json" \
  --direct-comparison "$OUTPUT_ROOT/$FOLD2/direct_comparison.json" \
  --dataset-audit "$PRIMARY/reports/news_krx500_dart_pit_v2_integrity_20260712.json" \
  --ohlcv-audit "$PRIMARY/reports/ohlcv_causal453_release_audit_20260713.json" \
  --output-dir "$OUTPUT_ROOT/shadow_gate"
GATE_STATUS=$?
set -e
printf '%s\n' "$GATE_STATUS" > "$OUTPUT_ROOT/shadow_gate/exit_status.txt"
touch "$OUTPUT_ROOT/GATE_COMPLETE"
echo "latent trajectory path-head gate complete status=$GATE_STATUS"
