#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
BASE_RUN="${BASE_RUN:-broad_transition_jepa_v6_systemic_seed17_rtx4000ada_20260714}"
HEAD_SELECTION="${HEAD_SELECTION:-reports/latent_path_head_v6_final_seed_selection_20260714/selection.json}"
CHECKPOINT_SELECTION="${CHECKPOINT_SELECTION:-reports/$BASE_RUN/deployment_checkpoint_selection/selection.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reports/shadow_gate_broad_transition_v6_latent_head_selected_20260714}"
DIRECT_ROOT="reports/direct_state_mlp_${BASE_RUN}_20260714"
DIRECT_COMPARISON_ROOT="reports/direct_vs_jepa_broad_transition_v6_seed17_20260714"
DATASET_AUDIT="reports/news_krx500_dart_pit_v2_integrity_20260712.json"
OHLCV_AUDIT="reports/ohlcv_causal453_release_audit_20260713.json"
FOLD1="${BASE_RUN}_fold1_20231229_to_20241230"
FOLD2="${BASE_RUN}_fold2_20241230_to_20260710"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

read_json_field() {
  "$PYTHON_BIN" -c \
    "import json; print(json.load(open('$1', encoding='utf-8'))['$2'])"
}

if [[ "$(read_json_field "$CHECKPOINT_SELECTION" live_orders_allowed)" != "False" ]]; then
  printf '%s\n' "checkpoint selection is not research-only" >&2
  exit 3
fi
if [[ "$(read_json_field "$CHECKPOINT_SELECTION" selected_label)" != "epoch24" ]]; then
  printf '%s\n' "deployment checkpoint selection did not choose epoch24" >&2
  exit 4
fi
if [[ "$(read_json_field "$HEAD_SELECTION" live_orders_allowed)" != "False" ]]; then
  printf '%s\n' "head selection is not research-only" >&2
  exit 5
fi

SELECTED_LABEL="$(read_json_field "$HEAD_SELECTION" selected_label)"
HEAD_ROOT="reports/latent_path_head_${BASE_RUN}_final_${SELECTED_LABEL}_20260714"

printf \
  '{"scope":"read_only_shadow_gate_research","checkpoint_selection":"%s","head_selection":"%s","selected_checkpoint":"epoch24","selected_head":"%s","live_orders_allowed":false}\n' \
  "$CHECKPOINT_SELECTION" "$HEAD_SELECTION" "$SELECTED_LABEL" \
  > "$OUTPUT_ROOT/safety_contract.json"
shasum -a 256 \
  scripts/select_latent_path_head_seed.py \
  scripts/attach_latent_path_head_summary.py \
  scripts/compare_latent_path_head_direct.py \
  scripts/gate_shadow_candidate.py \
  scripts/qualify_selected_v6_latent_head_m1pro.sh \
  "$CHECKPOINT_SELECTION" \
  "$HEAD_SELECTION" \
  > "$OUTPUT_ROOT/input_sha256.txt"

for fold in fold1 fold2; do
  if [[ "$fold" == "fold1" ]]; then
    model_name="$FOLD1"
  else
    model_name="$FOLD2"
  fi
  head_dir="$HEAD_ROOT/$fold"
  expected_sha="$(read_json_field "$head_dir/summary.json" parent_model_sha256)"
  actual_sha="$(shasum -a 256 "models/$BASE_RUN/$model_name/graph_jepa_real.pt" | awk '{print $1}')"
  if [[ "$expected_sha" != "$actual_sha" ]]; then
    printf 'latent head parent SHA mismatch for %s\n' "$fold" >&2
    exit 6
  fi
  mkdir -p "$OUTPUT_ROOT/$fold"
  "$PYTHON_BIN" scripts/attach_latent_path_head_summary.py \
    --node-summary "reports/$BASE_RUN/node_eval/$model_name/summary.json" \
    --head-summary "$head_dir/summary.json" \
    --output "$OUTPUT_ROOT/$fold/node_summary.json"
  "$PYTHON_BIN" scripts/compare_latent_path_head_direct.py \
    --original-combined "$DIRECT_COMPARISON_ROOT/$fold/comparison.json" \
    --head-daily "$head_dir/daily_metrics.csv" \
    --challenger "graph=$DIRECT_ROOT/$fold/daily_metrics.csv" \
    --output "$OUTPUT_ROOT/$fold/direct_comparison.json"
done

set +e
"$PYTHON_BIN" scripts/gate_shadow_candidate.py \
  --walk-forward-summary "reports/$BASE_RUN/summary.json" \
  --node-summary "$OUTPUT_ROOT/fold1/node_summary.json" \
  --node-summary "$OUTPUT_ROOT/fold2/node_summary.json" \
  --direct-comparison "$OUTPUT_ROOT/fold1/direct_comparison.json" \
  --direct-comparison "$OUTPUT_ROOT/fold2/direct_comparison.json" \
  --dataset-audit "$DATASET_AUDIT" \
  --ohlcv-audit "$OHLCV_AUDIT" \
  --output-dir "$OUTPUT_ROOT/gate"
GATE_STATUS=$?
set -e

printf '%s\n' "$GATE_STATUS" > "$OUTPUT_ROOT/gate/exit_status.txt"
touch "$OUTPUT_ROOT/GATE_COMPLETE"
printf 'selected latent-head gate complete status=%s head=%s\n' \
  "$GATE_STATUS" "$SELECTED_LABEL"
