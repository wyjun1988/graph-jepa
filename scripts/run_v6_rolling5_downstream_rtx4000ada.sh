#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
PYTHON_BIN="${PYTHON:-/root/venvs/stock-v2-cu128/bin/python}"
RUN_NAME="${RUN_NAME:-broad_transition_jepa_v6_rolling5_v3_seed17_rtx4000ada_20260714}"
REPORT_ROOT="reports/$RUN_NAME"
MODEL_ROOT="models/$RUN_NAME"
DIRECT_ROOT="reports/direct_state_mlp_${RUN_NAME}_20260714"
HEAD_ROOT="reports/latent_path_head_${RUN_NAME}_seed2701_20260714"
COMPARE_ROOT="reports/direct_vs_jepa_${RUN_NAME}_20260714"
GATE_INPUT_ROOT="reports/shadow_gate_inputs_${RUN_NAME}_seed2701_20260714"
LOG="ops/training/${RUN_NAME}_downstream.log"
OHLCV="${OHLCV:-data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv}"

cd "$ROOT"
mkdir -p "$DIRECT_ROOT" "$HEAD_ROOT" "$COMPARE_ROOT" "$GATE_INPUT_ROOT" "$(dirname "$LOG")"

exec 9>"$GATE_INPUT_ROOT/.downstream.lock"
if ! flock -n 9; then
  printf '%s\n' "rolling downstream evaluation is already running"
  exit 0
fi
if [[ ! -f "$REPORT_ROOT/PIPELINE_COMPLETE" ]]; then
  printf '%s\n' "rolling encoder pipeline is incomplete" >&2
  exit 3
fi

readarray -t MODEL_DIRS < <(
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); [print(r['"'"'model_dir'"'"']) for r in p['"'"'folds'"'"']]' \
    "$REPORT_ROOT/summary.json"
)
if [[ "${#MODEL_DIRS[@]}" -ne 5 ]]; then
  printf '%s\n' "rolling downstream requires exactly five encoder folds" >&2
  exit 4
fi

for index in "${!MODEL_DIRS[@]}"; do
  fold_number=$((index + 1))
  fold="fold${fold_number}"
  model_dir="${MODEL_DIRS[$index]}"
  model_name="$(basename "$model_dir")"
  direct_output="$DIRECT_ROOT/$fold"
  head_output="$HEAD_ROOT/$fold"
  compare_output="$COMPARE_ROOT/$fold"
  gate_output="$GATE_INPUT_ROOT/$fold"
  node_summary="$REPORT_ROOT/node_eval/$model_name/summary.json"
  jepa_daily="$REPORT_ROOT/node_eval/$model_name/future_rollout.csv"

  if [[ ! -f "$direct_output/EXPERIMENT_COMPLETE" ]]; then
    "$PYTHON_BIN" scripts/benchmark_direct_state_mlp.py \
      --model-dir "$model_dir" \
      --output-dir "$direct_output" \
      --horizons 1,2,3,5,10 \
      --validation-days 126 \
      --hidden-dim 512 \
      --layers 3 \
      --dropout 0.05 \
      --epochs 16 \
      --patience 4 \
      --batch-size 16384 \
      --learning-rate 0.0003 \
      --weight-decay 0.0001 \
      --device cuda \
      --seed 17 \
      --feature-workers 16 \
      --cache-dir "$OHLCV" \
      --external-cache-dir data/external_cache \
      --context-cache "$direct_output/direct_context_graph.npz" \
      2>&1 | tee -a "$LOG"
    touch "$direct_output/EXPERIMENT_COMPLETE"
  fi

  if [[ ! -f "$compare_output/comparison.json" ]]; then
    "$PYTHON_BIN" scripts/compare_direct_state_mlp.py \
      --direct-daily "$direct_output/daily_metrics.csv" \
      --jepa-daily "$jepa_daily" \
      --output-dir "$compare_output" \
      2>&1 | tee -a "$LOG"
  fi

  if [[ ! -f "$head_output/EXPERIMENT_COMPLETE" ]]; then
    "$PYTHON_BIN" scripts/benchmark_latent_trajectory_path_head.py \
      --model-dir "$model_dir" \
      --output-dir "$head_output" \
      --horizons 1,2,3,5,10 \
      --validation-days 126 \
      --epochs 8 \
      --patience 2 \
      --hidden-dim 256 \
      --dropout 0.05 \
      --learning-rate 0.0003 \
      --batch-size 8 \
      --liquidity-top-k 300 \
      --latent-blend-weight 1.0 \
      --edge-cache-workers 16 \
      --device cuda \
      --seed 2701 \
      --cache-dir "$OHLCV" \
      --external-cache-dir data/external_cache \
      2>&1 | tee -a "$LOG"
    touch "$head_output/EXPERIMENT_COMPLETE"
  fi

  mkdir -p "$gate_output"
  "$PYTHON_BIN" scripts/attach_latent_path_head_summary.py \
    --node-summary "$node_summary" \
    --head-summary "$head_output/summary.json" \
    --output "$gate_output/node_summary.json"
  "$PYTHON_BIN" scripts/compare_latent_path_head_direct.py \
    --original-combined "$compare_output/comparison.json" \
    --head-daily "$head_output/daily_metrics.csv" \
    --challenger "graph=$direct_output/daily_metrics.csv" \
    --output "$gate_output/direct_comparison.json"
done

touch "$GATE_INPUT_ROOT/DOWNSTREAM_COMPLETE"
printf '%s\n' "$RUN_NAME rolling downstream evaluation complete" | tee -a "$LOG"
