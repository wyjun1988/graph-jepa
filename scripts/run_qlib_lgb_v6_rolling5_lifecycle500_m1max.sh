#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/work/stock-v2}"
export ROOT
export FEATURE_PYTHON="${FEATURE_PYTHON:-$ROOT/.venv-mps-max/bin/python}"
export QLIB_PYTHON="${QLIB_PYTHON:-$ROOT/.venv-qlib/bin/python}"
export RUN_NAME="${RUN_NAME:-broad_transition_jepa_v6_lifecycle500_v4_seed17_rtx4000ada_20260714}"
export OHLCV="${OHLCV:-$ROOT/data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv}"

exec bash "$ROOT/scripts/run_qlib_lgb_v6_rolling5_m1pro.sh"
