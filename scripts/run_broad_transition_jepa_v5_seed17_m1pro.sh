#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
export ROOT
export PYTHON="${PYTHON:-$ROOT/.venv-mps/bin/python}"
export RUN_NAME="broad_transition_jepa_v5_systemic_seed17_m1pro_20260714"
export TARGET_VERSION="market_transition_v5_systemic_impact_20260714"
export DEVICE="mps"
export EVAL_DEVICE="mps"
export TRAIN_BATCH_SIZE="8"
export SNAPSHOT_WORKERS="8"
export SEED="17"
export TRANSITION_EVAL_BATCH_SIZE="32"
export PYTORCH_ENABLE_MPS_FALLBACK=1
exec bash "$(dirname "$0")/run_broad_transition_jepa_v4_a5000.sh"
