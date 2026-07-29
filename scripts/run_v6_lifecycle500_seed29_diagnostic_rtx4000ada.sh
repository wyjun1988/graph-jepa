#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
export ROOT
export RUN_NAME="${RUN_NAME:-broad_transition_jepa_v6_lifecycle500_v4_seed29_diagnostic_rtx4000ada_20260715}"
export PREFLIGHT_NAME="${PREFLIGHT_NAME:-broad_transition_jepa_v6_lifecycle500_v4_sourcefreeze_20260715}"
export CONTRACT="${CONTRACT:-configs/rolling-v6-shadow-qualification-v4-20260714.json}"
export OHLCV="${OHLCV:-data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv}"
export MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-1}"
export SEED="${SEED:-29}"
export ALLOW_DIAGNOSTIC_SEED_OVERRIDE=1
export REPORTS_BASE="${REPORTS_BASE:-/workspace/stock-v2-seed-stability/reports}"
export MODELS_BASE="${MODELS_BASE:-/workspace/stock-v2-seed-stability/models}"
export LOG="${LOG:-/workspace/stock-v2-seed-stability/logs/${RUN_NAME}.log}"

exec bash "$ROOT/scripts/run_v6_rolling5_train_rtx4000ada.sh"
