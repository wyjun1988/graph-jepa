#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/work/stock-v2}"
PYTHON_BIN="${PYTHON:-$ROOT/.venv-mps/bin/python}"
CONTRACT="${CONTRACT:-configs/rolling-v6-shadow-qualification-v3-20260714.json}"
OUTPUT_ROOT="${OUTPUT_ROOT:-reports/rolling_v6_sensor_audit_v3_20260714}"
OHLCV="data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv"

cd "$ROOT"
mkdir -p "$OUTPUT_ROOT"

readarray_compat() {
  local output
  output="$($PYTHON_BIN -c 'import json,sys; p=json.load(open(sys.argv[1])); [print(f"{i + 1}\t{r['"'"'train_end'"'"']}\t{r['"'"'eval_end'"'"']}") for i,r in enumerate(p['"'"'folds'"'"'])]' "$CONTRACT")"
  while IFS=$'\t' read -r fold_number train_end eval_end; do
    [[ -n "$fold_number" ]] || continue
    output_path="$OUTPUT_ROOT/fold${fold_number}.json"
    if [[ -f "$output_path" ]]; then
      continue
    fi
    "$PYTHON_BIN" -m stock_v2.validate_stock_dataset \
      --universe krx \
      --universe-manifest data/universes/krx500_pit_20191231.json \
      --max-tickers 500 \
      --start 2020-01-01 \
      --end "$eval_end" \
      --train-end "$train_end" \
      --cache-dir "$OHLCV" \
      --event-path data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl \
      --fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
      --fundamental-lag-days 1 \
      --investor-cache-dir data/kiwoom_investor_cache \
      --investor-flow-lag-days 1 \
      --external-preset kr_global_rates \
      --external-node-mode nodes \
      --external-lag-days 1 \
      --external-cache-dir data/external_cache \
      --path-horizons 1,2,3,5,10 \
      --horizon 10 \
      --min-train-feature-rows 500 \
      --min-test-feature-rows 190 \
      --output "$output_path"
  done <<< "$output"
}

readarray_compat
touch "$OUTPUT_ROOT/AUDIT_COMPLETE"
