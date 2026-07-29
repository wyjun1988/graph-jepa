#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
export ROOT
export RUN_NAME="${RUN_NAME:-broad_transition_jepa_v6_lifecycle500_v4_recipe16_preflight_20260714}"
export CONTRACT="${CONTRACT:-configs/rolling-v6-shadow-qualification-v4-20260714.json}"
export OHLCV="${OHLCV:-data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv}"
export MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-1}"

exec bash "$ROOT/scripts/run_v6_rolling5_preflight_rtx4000ada.sh"
