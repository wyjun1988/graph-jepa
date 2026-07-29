#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/stock-v2"
PYTHON="/root/venvs/news-vllm-cu128/bin/python"
TEMPORAL_SCALE="${1:-0.0}"
TEMPORAL_STOCK_EDGE_SCALE="${2:-1.0}"
EVENT_PATH="data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl"
OHLCV_CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
UNIVERSE="data/universes/krx500_pit_20191231.json"
FUNDAMENTALS="data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl"
FOLD1_PANEL="00cbf81903f5c4bba5ef6ddc39e8a243c63b8445b8ed90bc1e8cea7f2ada630e"
FOLD2_PANEL="4ae8bdfb8e6f13af77dcb9847974f2c74694768a46667ff9debab136b0f96452"
FOLD1_EDGE="c66077ecbc91c3996204dfb95b1b5e12b2542ac143036c8659d3363519669c2b"
FOLD2_EDGE="a85f939144cec194ef35fb927603b69bc111ce3bbedf12bcc159757c713d7870"
SCREEN_ROOT="reports/path_objective_screen_causal453_v1_20260713"
SCREEN_MODELS="models/path_objective_screen_causal453_v1_20260713"
BIG_REPORTS="reports/walk_forward_causal453_path_v1_20260713"
BIG_MODELS="models/walk_forward_causal453_path_v1_20260713"

cd "$ROOT"
mkdir -p "$SCREEN_ROOT" "$SCREEN_MODELS" "$BIG_REPORTS" "$BIG_MODELS"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
"$PYTHON" scripts/ensure_path_run_contract.py \
  --root "$ROOT" \
  --output "$SCREEN_ROOT/run_contract.json" \
  --output "$BIG_REPORTS/run_contract.json" \
  --temporal-graph-neighbor-scale "$TEMPORAL_SCALE" \
  --temporal-stock-edge-scale "$TEMPORAL_STOCK_EDGE_SCALE" \
  --fold-panel-sha256 "$FOLD1_PANEL" \
  --fold-panel-sha256 "$FOLD2_PANEL" \
  --fold-edge-sha256 "$FOLD1_EDGE" \
  --fold-edge-sha256 "$FOLD2_EDGE"
sha256sum \
  stock_v2/graph_jepa.py \
  stock_v2/ops/signals.py \
  scripts/run_real_backtest.py \
  scripts/run_walk_forward_node_eval.py \
  scripts/evaluate_node_prediction.py \
  scripts/benchmark_direct_state_mlp.py \
  scripts/compare_direct_state_mlp.py \
  scripts/combine_direct_state_challenges.py \
  scripts/gate_shadow_candidate.py \
  scripts/select_path_objective_candidates.py \
  scripts/summarize_checkpoint_epochs.py \
  scripts/ensure_path_run_contract.py \
  scripts/run_path_objective_pipeline_gpu.sh \
  > "$BIG_REPORTS/source_sha256.txt"
printf '%s\n' \
  '{"scope":"read_only_shadow_validation","live_orders_allowed":false}' \
  > "$BIG_REPORTS/safety_contract.json"

common_data_args=(
  --training-manifest-schema-version 4
  --universe krx
  --universe-manifest "$UNIVERSE"
  --max-tickers 500
  --cache-dir "$OHLCV_CACHE"
  --event-path "$EVENT_PATH"
  --event-coverage-mode mask_uncovered
  --require-event-sensors
  --min-event-coverage 0.95
  --fundamental-path "$FUNDAMENTALS"
  --fundamental-lag-days 1
  --require-fundamental-sensors
  --min-fundamental-coverage 0.86
  --investor-cache-dir data/kiwoom_investor_cache
  --investor-flow-lag-days 1
  --require-investor-sensors
  --min-investor-coverage 0.89
  --external-preset kr_global_rates
  --external-node-mode nodes
  --external-lag-days 1
  --external-cache-dir data/external_cache
  --require-all-external-factors
)

common_model_args=(
  --horizon 10
  --top-k 5
  --edge-top-k 6
  --edge-correlation-mode signed
  --graph-neighbor-scale 1.0
  --temporal-graph-neighbor-scale "$TEMPORAL_SCALE"
  --temporal-stock-edge-scale "$TEMPORAL_STOCK_EDGE_SCALE"
  --lr 0.0003
  --ema-decay 0.9995
  --state-loss-weight 1.0
  --current-imputation-loss-weight 1.0
  --normalize-predictor-output
  --temporal-state-mode horizon_residual_heads
  --temporal-residual-short-steps 2
  --pretrain-task temporal
  --temporal-offset 10
  --latent-rollout-steps 10
  --rollout-offsets 1,2,3,5,10
  --rollout-loss-weights 2,2,1,1,1
  --path-horizons 1,2,3,5,10
  --mask-strategy mixed
  --partial-corr-top-k 0
  --lead-lag-top-k 0
  --policy-rate-edge-scale 0.0
  --event-edge-top-k 0
  --temporal-exclude-feature-prefix news_
  --temporal-exclude-feature-prefix fund_
)

weight_args() {
  local weight="$1"
  printf '%s\n' \
    --state-feature-weight "return_1d=$weight" \
    --state-feature-weight "return_2d=$weight" \
    --state-feature-weight "return_3d=$weight" \
    --state-feature-weight "return_5d=$weight" \
    --state-feature-weight "return_10d=$weight" \
    --state-feature-weight "gap_open=$weight" \
    --state-feature-weight "intraday_return=$weight"
}

run_screen() {
  local name="$1"
  local return_weight="$2"
  local path_corr_weight="$3"
  local latent_weight="$4"
  local context_skip="$5"
  local report_dir="$SCREEN_ROOT/$name"
  local model_dir="$SCREEN_MODELS/$name"
  if [[ -f "$report_dir/CANDIDATE_COMPLETE" ]]; then
    echo "screen already complete: $name"
    return
  fi
  mkdir -p "$report_dir"
  mapfile -t return_args < <(weight_args "$return_weight")
  context_skip_args=()
  if [[ "$context_skip" == "true" ]]; then
    context_skip_args+=(--temporal-state-context-skip)
  fi
  "$PYTHON" scripts/run_real_backtest.py \
    --start 2020-01-01 \
    --end 2024-12-30 \
    --train-end 2023-12-29 \
    --epochs 8 \
    --hidden-dim 512 \
    --layers 6 \
    --train-batch-size 16 \
    --snapshot-workers 16 \
    --device cuda \
    --seed 1704 \
    --skip-return-backtest \
    --reports-dir "$report_dir/train" \
    --models-dir "$model_dir" \
    --return-correlation-loss-weight 0.0 \
    --entry-path-correlation-loss-weight "$path_corr_weight" \
    --latent-loss-weight "$latent_weight" \
    --expected-training-manifest-sha256 "$FOLD1_PANEL" \
    --expected-training-edge-manifest-sha256 "$FOLD1_EDGE" \
    "${return_args[@]}" \
    "${context_skip_args[@]}" \
    "${common_data_args[@]}" \
    "${common_model_args[@]}"
  "$PYTHON" scripts/evaluate_node_prediction.py \
    --model-dir "$model_dir" \
    --output-dir "$report_dir/node_eval" \
    --horizons 1,2,3,5,10 \
    --mask-strategy mixed \
    --edge-cache-workers 16 \
    --device cuda \
    --seed 1704 \
    --cache-dir "$OHLCV_CACHE" \
    --external-cache-dir data/external_cache
  touch "$report_dir/CANDIDATE_COMPLETE"
}

run_screen control_w1_c0_l1_noskip 1 0.0 1.0 false
run_screen control_w1_c0_l1_skip 1 0.0 1.0 true
run_screen control_w1_c0_l025_skip 1 0.0 0.25 true
run_screen path_w4_p001_l025_skip 4 0.01 0.25 true
run_screen path_w8_p0025_l025_noskip 8 0.025 0.25 false
run_screen path_w8_p0025_l025_skip 8 0.025 0.25 true
run_screen path_w12_p005_l025_skip 12 0.05 0.25 true

selection_args=()
for name in control_w1_c0_l1_noskip control_w1_c0_l1_skip control_w1_c0_l025_skip path_w4_p001_l025_skip path_w8_p0025_l025_noskip path_w8_p0025_l025_skip path_w12_p005_l025_skip; do
  selection_args+=(
    --candidate "$name=$SCREEN_ROOT/$name/node_eval/$name/summary.json"
  )
done
"$PYTHON" scripts/select_path_objective_candidates.py \
  "${selection_args[@]}" \
  --output-dir "$SCREEN_ROOT/selection"
SELECTED_LABEL="$($PYTHON - <<'PY'
import json
print(json.load(open('reports/path_objective_screen_causal453_v1_20260713/selection/selection.json'))['selected_label'])
PY
)"
case "$SELECTED_LABEL" in
  control_w1_c0_l1_noskip) RETURN_WEIGHT=1; PATH_CORR_WEIGHT=0.0; LATENT_WEIGHT=1.0; CONTEXT_SKIP=false ;;
  control_w1_c0_l1_skip) RETURN_WEIGHT=1; PATH_CORR_WEIGHT=0.0; LATENT_WEIGHT=1.0; CONTEXT_SKIP=true ;;
  control_w1_c0_l025_skip) RETURN_WEIGHT=1; PATH_CORR_WEIGHT=0.0; LATENT_WEIGHT=0.25; CONTEXT_SKIP=true ;;
  path_w4_p001_l025_skip) RETURN_WEIGHT=4; PATH_CORR_WEIGHT=0.01; LATENT_WEIGHT=0.25; CONTEXT_SKIP=true ;;
  path_w8_p0025_l025_noskip) RETURN_WEIGHT=8; PATH_CORR_WEIGHT=0.025; LATENT_WEIGHT=0.25; CONTEXT_SKIP=false ;;
  path_w8_p0025_l025_skip) RETURN_WEIGHT=8; PATH_CORR_WEIGHT=0.025; LATENT_WEIGHT=0.25; CONTEXT_SKIP=true ;;
  path_w12_p005_l025_skip) RETURN_WEIGHT=12; PATH_CORR_WEIGHT=0.05; LATENT_WEIGHT=0.25; CONTEXT_SKIP=true ;;
  *) echo "unknown selected objective: $SELECTED_LABEL" >&2; exit 4 ;;
esac

RUN_NAME="strict_causal453_path_v1_${SELECTED_LABEL}_seed17"
mapfile -t selected_return_args < <(weight_args "$RETURN_WEIGHT")
selected_context_args=()
if [[ "$CONTEXT_SKIP" == "true" ]]; then
  selected_context_args+=(--temporal-state-context-skip)
fi
if [[ ! -f "$BIG_REPORTS/TRAINING_COMPLETE" ]]; then
  "$PYTHON" scripts/run_walk_forward_node_eval.py \
    --name "$RUN_NAME" \
    --fold 2023-12-29:2024-12-30 \
    --fold 2024-12-30:2026-07-10 \
    --start 2020-01-01 \
    --epochs 24 \
    --checkpoint-epochs 8,16 \
    --hidden-dim 1024 \
    --layers 10 \
    --train-batch-size 8 \
    --snapshot-workers 16 \
    --device cuda \
    --eval-device cuda \
    --max-steps 0 \
    --seed 17 \
    --return-correlation-loss-weight 0.0 \
    --entry-path-correlation-loss-weight "$PATH_CORR_WEIGHT" \
    --latent-loss-weight "$LATENT_WEIGHT" \
    --expected-training-manifest-sha256 "$FOLD1_PANEL" \
    --expected-training-manifest-sha256 "$FOLD2_PANEL" \
    --expected-training-edge-manifest-sha256 "$FOLD1_EDGE" \
    --expected-training-edge-manifest-sha256 "$FOLD2_EDGE" \
    --reports-root "$BIG_REPORTS" \
    --models-root "$BIG_MODELS" \
    --summary-output "$BIG_REPORTS/summary.json" \
    "${selected_return_args[@]}" \
    "${selected_context_args[@]}" \
    "${common_data_args[@]}" \
    "${common_model_args[@]}"
  touch "$BIG_REPORTS/TRAINING_COMPLETE"
fi

FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"
for fold in "$FOLD1" "$FOLD2"; do
  model_name="${RUN_NAME}_${fold}"
  checkpoint_output="$BIG_REPORTS/checkpoint_eval/$fold"
  for epoch in 008 016; do
    checkpoint_dir="$BIG_MODELS/$model_name/epoch_${epoch}"
    if [[ ! -f "$checkpoint_output/epoch_${epoch}/summary.json" ]]; then
      "$PYTHON" scripts/evaluate_node_prediction.py \
        --model-dir "$checkpoint_dir" \
        --output-dir "$checkpoint_output" \
        --horizons 1,2,3,5,10 \
        --mask-strategy mixed \
        --max-steps 0 \
        --edge-cache-workers 16 \
        --device cuda \
        --seed 17 \
        --cache-dir "$OHLCV_CACHE" \
        --external-cache-dir data/external_cache
    fi
  done
done

"$PYTHON" scripts/summarize_checkpoint_epochs.py \
  --fold1 "epoch8=$BIG_REPORTS/checkpoint_eval/$FOLD1/epoch_008/summary.json" \
  --fold1 "epoch16=$BIG_REPORTS/checkpoint_eval/$FOLD1/epoch_016/summary.json" \
  --fold1 "epoch24=$BIG_REPORTS/node_eval/${RUN_NAME}_${FOLD1}/summary.json" \
  --fold2 "epoch8=$BIG_REPORTS/checkpoint_eval/$FOLD2/epoch_008/summary.json" \
  --fold2 "epoch16=$BIG_REPORTS/checkpoint_eval/$FOLD2/epoch_016/summary.json" \
  --fold2 "epoch24=$BIG_REPORTS/node_eval/${RUN_NAME}_${FOLD2}/summary.json" \
  --output-dir "$BIG_REPORTS/checkpoint_selection"
SELECTED_EPOCH="$($PYTHON - <<'PY'
import json
print(json.load(open('reports/walk_forward_causal453_path_v1_20260713/checkpoint_selection/selection.json'))['selected_label'])
PY
)"

for fold in "$FOLD1" "$FOLD2"; do
  model_name="${RUN_NAME}_${fold}"
  case "$SELECTED_EPOCH" in
    epoch8)
      model_dir="$BIG_MODELS/$model_name/epoch_008"
      node_dir="$BIG_REPORTS/checkpoint_eval/$fold/epoch_008"
      ;;
    epoch16)
      model_dir="$BIG_MODELS/$model_name/epoch_016"
      node_dir="$BIG_REPORTS/checkpoint_eval/$fold/epoch_016"
      ;;
    epoch24)
      model_dir="$BIG_MODELS/$model_name"
      node_dir="$BIG_REPORTS/node_eval/$model_name"
      ;;
    *)
      echo "unknown selected checkpoint: $SELECTED_EPOCH" >&2
      exit 5
      ;;
  esac
  direct_dir="$BIG_REPORTS/direct/$fold"
  direct_nograph_dir="$BIG_REPORTS/direct_nograph/$fold"
  comparison_dir="$BIG_REPORTS/direct_vs_jepa/$fold/graph"
  comparison_nograph_dir="$BIG_REPORTS/direct_vs_jepa/$fold/nograph"
  combined_comparison_dir="$BIG_REPORTS/direct_vs_jepa/$fold/combined"
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
      --cache-dir "$OHLCV_CACHE" \
      --external-cache-dir data/external_cache \
      --context-cache "data/cache/direct_context_${RUN_NAME}_${fold}.npy"
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
      --cache-dir "$OHLCV_CACHE" \
      --external-cache-dir data/external_cache \
      --context-cache "data/cache/direct_context_${RUN_NAME}_${fold}.npy"
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
    --output-dir "$combined_comparison_dir"
done

case "$SELECTED_EPOCH" in
  epoch8)
    NODE1="$BIG_REPORTS/checkpoint_eval/$FOLD1/epoch_008/summary.json"
    NODE2="$BIG_REPORTS/checkpoint_eval/$FOLD2/epoch_008/summary.json"
    ;;
  epoch16)
    NODE1="$BIG_REPORTS/checkpoint_eval/$FOLD1/epoch_016/summary.json"
    NODE2="$BIG_REPORTS/checkpoint_eval/$FOLD2/epoch_016/summary.json"
    ;;
  epoch24)
    NODE1="$BIG_REPORTS/node_eval/${RUN_NAME}_${FOLD1}/summary.json"
    NODE2="$BIG_REPORTS/node_eval/${RUN_NAME}_${FOLD2}/summary.json"
    ;;
  *)
    echo "unknown selected checkpoint: $SELECTED_EPOCH" >&2
    exit 5
    ;;
esac
set +e
"$PYTHON" scripts/gate_shadow_candidate.py \
  --walk-forward-summary "$BIG_REPORTS/summary.json" \
  --node-summary "$NODE1" \
  --node-summary "$NODE2" \
  --direct-comparison "$BIG_REPORTS/direct_vs_jepa/$FOLD1/combined/comparison.json" \
  --direct-comparison "$BIG_REPORTS/direct_vs_jepa/$FOLD2/combined/comparison.json" \
  --dataset-audit reports/news_krx500_dart_pit_v2_integrity_20260712.json \
  --ohlcv-audit reports/ohlcv_causal453_release_audit_20260713.json \
  --output-dir "$BIG_REPORTS/shadow_gate"
GATE_STATUS=$?
set -e
printf '%s\n' "$GATE_STATUS" > "$BIG_REPORTS/shadow_gate/exit_status.txt"
touch "$BIG_REPORTS/PIPELINE_COMPLETE"
echo "path objective pipeline complete gate_status=$GATE_STATUS"
