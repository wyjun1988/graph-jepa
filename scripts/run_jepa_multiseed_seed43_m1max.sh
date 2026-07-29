#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="$ROOT/.venv-mps-max/bin/python"
RUN_NAME="strict_causal453_path_v2_path_w12_p005_l025_skip_seed43"
REPORTS_ROOT="reports/walk_forward_causal453_path_multiseed_seed43_20260714"
MODELS_ROOT="models/walk_forward_causal453_path_multiseed_seed43_20260714"
LOG_PATH="ops/training/jepa_multiseed_seed43_m1max_20260714.log"
PID_PATH="ops/training/jepa_multiseed_seed43_m1max_20260714.pid"

cd "$ROOT"
mkdir -p "$REPORTS_ROOT" "$MODELS_ROOT" ops/training

if [[ "${1:-}" != "--worker" ]]; then
  if [[ -f "$PID_PATH" ]] && kill -0 "$(cat "$PID_PATH")" 2>/dev/null; then
    echo "seed 43 M1 Max job already running: $(cat "$PID_PATH")"
    exit 0
  fi
  nohup caffeinate -dimsu bash "$0" --worker > "$LOG_PATH" 2>&1 < /dev/null &
  echo "$!" > "$PID_PATH"
  echo "seed 43 M1 Max job started: $!"
  exit 0
fi

export PYTHONUNBUFFERED=1
export PYTHONHASHSEED=43
export PYTORCH_ENABLE_MPS_FALLBACK=1
rm -f \
  "$REPORTS_ROOT/FINISHED_AT" \
  "$REPORTS_ROOT/TRAINING_COMPLETE" \
  "$REPORTS_ROOT/TRAINING_FAILED" \
  "$REPORTS_ROOT/exit_status.txt"

on_exit() {
  status=$?
  trap - EXIT
  date '+%Y-%m-%dT%H:%M:%S%z' > "$REPORTS_ROOT/FINISHED_AT"
  printf '%s\n' "$status" > "$REPORTS_ROOT/exit_status.txt"
  if [[ "$status" -eq 0 ]]; then
    touch "$REPORTS_ROOT/TRAINING_COMPLETE"
  else
    touch "$REPORTS_ROOT/TRAINING_FAILED"
  fi
  exit "$status"
}
trap on_exit EXIT

test -x "$PYTHON"
"$PYTHON" -c 'import torch; assert torch.backends.mps.is_available()'
test -f data/universes/krx500_pit_20191231.json
test -f data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl
test -f data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl
test -d data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv

{
  date '+%Y-%m-%dT%H:%M:%S%z'
  system_profiler SPHardwareDataType | head -20
  shasum -a 256 \
    stock_v2/graph_jepa.py \
    scripts/run_walk_forward_node_eval.py \
    scripts/evaluate_node_prediction.py \
    "$0"
} > "$REPORTS_ROOT/preflight_sha256.txt"

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
  --snapshot-workers 8 \
  --device mps \
  --eval-device mps \
  --max-steps 0 \
  --seed 43 \
  --expected-training-manifest-sha256 00cbf81903f5c4bba5ef6ddc39e8a243c63b8445b8ed90bc1e8cea7f2ada630e \
  --expected-training-manifest-sha256 4ae8bdfb8e6f13af77dcb9847974f2c74694768a46667ff9debab136b0f96452 \
  --expected-training-edge-manifest-sha256 c66077ecbc91c3996204dfb95b1b5e12b2542ac143036c8659d3363519669c2b \
  --expected-training-edge-manifest-sha256 a85f939144cec194ef35fb927603b69bc111ce3bbedf12bcc159757c713d7870 \
  --reports-root "$REPORTS_ROOT" \
  --models-root "$MODELS_ROOT" \
  --summary-output "$REPORTS_ROOT/summary.json" \
  --training-manifest-schema-version 4 \
  --universe krx \
  --universe-manifest data/universes/krx500_pit_20191231.json \
  --max-tickers 500 \
  --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
  --event-path data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl \
  --event-coverage-mode mask_uncovered \
  --require-event-sensors \
  --min-event-coverage 0.95 \
  --fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
  --fundamental-lag-days 1 \
  --require-fundamental-sensors \
  --min-fundamental-coverage 0.86 \
  --investor-cache-dir data/kiwoom_investor_cache \
  --investor-flow-lag-days 1 \
  --require-investor-sensors \
  --min-investor-coverage 0.89 \
  --external-preset kr_global_rates \
  --external-node-mode nodes \
  --external-lag-days 1 \
  --external-cache-dir data/external_cache \
  --require-all-external-factors \
  --horizon 10 \
  --top-k 5 \
  --edge-top-k 6 \
  --edge-correlation-mode signed \
  --graph-neighbor-scale 1.0 \
  --temporal-graph-neighbor-scale 0.0 \
  --temporal-stock-edge-scale 1.0 \
  --lr 0.0003 \
  --ema-decay 0.9995 \
  --state-loss-weight 1.0 \
  --current-imputation-loss-weight 1.0 \
  --normalize-predictor-output \
  --temporal-state-mode horizon_residual_heads \
  --temporal-state-context-skip \
  --temporal-residual-short-steps 2 \
  --pretrain-task temporal \
  --temporal-offset 10 \
  --latent-rollout-steps 10 \
  --rollout-offsets 1,2,3,5,10 \
  --rollout-loss-weights 2,2,1,1,1 \
  --path-horizons 1,2,3,5,10 \
  --mask-strategy mixed \
  --partial-corr-top-k 0 \
  --lead-lag-top-k 0 \
  --policy-rate-edge-scale 0.0 \
  --event-edge-top-k 0 \
  --temporal-exclude-feature-prefix news_ \
  --temporal-exclude-feature-prefix fund_ \
  --return-correlation-loss-weight 0.0 \
  --entry-path-correlation-loss-weight 0.05 \
  --latent-loss-weight 0.25 \
  --state-feature-weight return_1d=12 \
  --state-feature-weight return_2d=12 \
  --state-feature-weight return_3d=12 \
  --state-feature-weight return_5d=12 \
  --state-feature-weight return_10d=12 \
  --state-feature-weight gap_open=12 \
  --state-feature-weight intraday_return=12
