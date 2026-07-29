#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/workspace/stock-v2"
PYTHON="/root/venvs/news-vllm-cu128/bin/python"
SCREEN_ROOT="reports/path_objective_screen_causal453_v1_20260713"
SCREEN_SELECTION="$SCREEN_ROOT/selection/selection.json"
SCREEN_CONTRACT="$SCREEN_ROOT/run_contract.json"
BIG_REPORTS="reports/walk_forward_causal453_path_v2_20260713"
BIG_MODELS="models/walk_forward_causal453_path_v2_20260713"
RUN_NAME="strict_causal453_path_v2_path_w12_p005_l025_skip_seed17"
SELECTED_LABEL="path_w12_p005_l025_skip"
FOLD1="fold1_20231229_to_20241230"
FOLD2="fold2_20241230_to_20260710"
FOLD1_PANEL="00cbf81903f5c4bba5ef6ddc39e8a243c63b8445b8ed90bc1e8cea7f2ada630e"
FOLD2_PANEL="4ae8bdfb8e6f13af77dcb9847974f2c74694768a46667ff9debab136b0f96452"
FOLD1_EDGE="c66077ecbc91c3996204dfb95b1b5e12b2542ac143036c8659d3363519669c2b"
FOLD2_EDGE="a85f939144cec194ef35fb927603b69bc111ce3bbedf12bcc159757c713d7870"
EVENT_PATH="data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl"
OHLCV_CACHE="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"
UNIVERSE="data/universes/krx500_pit_20191231.json"
FUNDAMENTALS="data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl"

cd "$ROOT"
mkdir -p "$BIG_REPORTS" "$BIG_MODELS"
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

on_error() {
  local status=$?
  printf '{"status":"failed","exit_status":%d,"live_orders_allowed":false}\n' \
    "$status" > "$BIG_REPORTS/PIPELINE_FAILED"
  exit "$status"
}
trap on_error ERR

if [[ -f "$BIG_REPORTS/PIPELINE_FAILED" ]]; then
  mv "$BIG_REPORTS/PIPELINE_FAILED" \
    "$BIG_REPORTS/PIPELINE_FAILED.previous.$(date +%Y%m%dT%H%M%S)"
fi

"$PYTHON" scripts/ensure_path_confirmation_contract.py \
  --root "$ROOT" \
  --output "$BIG_REPORTS/run_contract.json" \
  --screen-selection "$SCREEN_SELECTION" \
  --screen-contract "$SCREEN_CONTRACT" \
  --expected-selected-label "$SELECTED_LABEL" \
  --fold-panel-sha256 "$FOLD1_PANEL" \
  --fold-panel-sha256 "$FOLD2_PANEL" \
  --fold-edge-sha256 "$FOLD1_EDGE" \
  --fold-edge-sha256 "$FOLD2_EDGE" \
  --temporal-graph-neighbor-scale 0.0 \
  --temporal-stock-edge-scale 1.0
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
  --temporal-graph-neighbor-scale 0.0
  --temporal-stock-edge-scale 1.0
  --lr 0.0003
  --ema-decay 0.9995
  --state-loss-weight 1.0
  --current-imputation-loss-weight 1.0
  --normalize-predictor-output
  --temporal-state-mode horizon_residual_heads
  --temporal-state-context-skip
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

objective_args=(
  --return-correlation-loss-weight 0.0
  --entry-path-correlation-loss-weight 0.05
  --latent-loss-weight 0.25
  --state-feature-weight return_1d=12
  --state-feature-weight return_2d=12
  --state-feature-weight return_3d=12
  --state-feature-weight return_5d=12
  --state-feature-weight return_10d=12
  --state-feature-weight gap_open=12
  --state-feature-weight intraday_return=12
)

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
    --expected-training-manifest-sha256 "$FOLD1_PANEL" \
    --expected-training-manifest-sha256 "$FOLD2_PANEL" \
    --expected-training-edge-manifest-sha256 "$FOLD1_EDGE" \
    --expected-training-edge-manifest-sha256 "$FOLD2_EDGE" \
    --reports-root "$BIG_REPORTS" \
    --models-root "$BIG_MODELS" \
    --summary-output "$BIG_REPORTS/summary.json" \
    "${objective_args[@]}" \
    "${common_data_args[@]}" \
    "${common_model_args[@]}"
  touch "$BIG_REPORTS/TRAINING_COMPLETE"
fi

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
SELECTED_EPOCH="$($PYTHON -c "import json; print(json.load(open('$BIG_REPORTS/checkpoint_selection/selection.json'))['selected_label'])")"

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
echo "path confirmation pipeline complete gate_status=$GATE_STATUS"
