#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/work/stock-v2}"
FEATURE_PYTHON="${FEATURE_PYTHON:-$ROOT/.venv-mps/bin/python}"
QLIB_PYTHON="${QLIB_PYTHON:-$ROOT/.venv-qlib/bin/python}"
RUN_NAME="${RUN_NAME:-broad_transition_jepa_v6_rolling5_v3_seed17_rtx4000ada_20260714}"
REPORT_ROOT="$ROOT/reports/$RUN_NAME"
QLIB_ROOT="$ROOT/reports/qlib_lgb_${RUN_NAME}_20260714"
HEAD_ROOT="$ROOT/reports/latent_path_head_${RUN_NAME}_seed2701_20260714"
COMPARE_ROOT="$ROOT/reports/qlib_vs_jepa_${RUN_NAME}_seed2701_20260714"
OHLCV="${OHLCV:-$ROOT/data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv}"
LOG="$ROOT/logs/qlib_lgb_${RUN_NAME}_m1pro.log"

cd "$ROOT"
mkdir -p "$QLIB_ROOT" "$COMPARE_ROOT" "$(dirname "$LOG")"
if [[ ! -f "$REPORT_ROOT/PIPELINE_COMPLETE" ]]; then
  printf '%s\n' "rolling encoder reports are not available under ROOT" >&2
  exit 3
fi

readarray_compat() {
  local output
  output="$($FEATURE_PYTHON -c 'import json,sys; p=json.load(open(sys.argv[1])); [print(r['"'"'model_dir'"'"']) for r in p['"'"'folds'"'"']]' "$REPORT_ROOT/summary.json")"
  local fold_number=0
  while IFS= read -r remote_model_dir; do
    [[ -n "$remote_model_dir" ]] || continue
    fold_number=$((fold_number + 1))
    fold="fold${fold_number}"
    model_dir="$ROOT/models/$RUN_NAME/$(basename "$remote_model_dir")"
    fold_root="$QLIB_ROOT/$fold"
    bundle="$fold_root/pit_bundle"
    result="$fold_root/result"
    comparison="$COMPARE_ROOT/$fold/comparison.json"
    mkdir -p "$fold_root" "$(dirname "$comparison")"
    if [[ ! -f "$bundle/bundle_contract.json" ]]; then
      "$FEATURE_PYTHON" scripts/export_qlib_pit_bundle.py \
        --model-dir "$model_dir" \
        --output-dir "$bundle" \
        --horizons 1,2,3,5,10 \
        --validation-days 126 \
        --feature-workers 10 \
        --cache-dir "$OHLCV" \
        --external-cache-dir "$ROOT/data/external_cache"
    fi
    if [[ ! -f "$result/EXPERIMENT_COMPLETE" ]]; then
      "$QLIB_PYTHON" scripts/benchmark_qlib_lgb.py \
        --bundle-dir "$bundle" \
        --output-dir "$result" \
        --horizons 1,2,3,5,10 \
        --num-boost-round 500 \
        --early-stopping-rounds 50 \
        --num-threads 10 \
        --liquidity-top-k 300 \
        --seed 17
    fi
    if [[ ! -f "$comparison" ]]; then
      "$FEATURE_PYTHON" scripts/compare_qlib_jepa_path.py \
        --qlib-daily "$result/daily_metrics.csv" \
        --jepa-daily "$HEAD_ROOT/$fold/daily_metrics.csv" \
        --output "$comparison" \
        --horizons 1,2,3,5,10 \
        --superiority-t 1.96
    fi
  done <<< "$output"
  if [[ "$fold_number" -ne 5 ]]; then
    printf 'expected five folds, found %s\n' "$fold_number" >&2
    exit 4
  fi
}

{
  readarray_compat
  touch "$QLIB_ROOT/EXPERIMENT_COMPLETE" "$COMPARE_ROOT/EXPERIMENT_COMPLETE"
} 2>&1 | tee -a "$LOG"
