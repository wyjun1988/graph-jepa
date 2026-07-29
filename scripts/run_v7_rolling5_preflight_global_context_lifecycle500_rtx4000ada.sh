#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
export ROOT
export RUN_NAME="${RUN_NAME:-broad_transition_jepa_v7_globalctx_aux_lifecycle500_recipe_preflight_20260715}"
export CONTRACT="${CONTRACT:-configs/rolling-v7-shadow-qualification-v5-20260715.json}"
export OHLCV="${OHLCV:-data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv}"
export MIN_TRAIN_ROWS="${MIN_TRAIN_ROWS:-1}"
export GLOBAL_STOCK_CONTEXT=1
export DOWNSTREAM_AUXILIARY_LOSS_WEIGHT=0.10
export ROLLOUT_LOSS_WEIGHTS=2,2,2,1,1

exec bash "$ROOT/scripts/run_v6_rolling5_preflight_rtx4000ada.sh"
