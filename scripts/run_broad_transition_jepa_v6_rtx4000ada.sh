#!/usr/bin/env bash
set -Eeuo pipefail

export RUN_NAME="${RUN_NAME:-broad_transition_jepa_v6_systemic_seed17_20260714}"
export TARGET_VERSION="market_transition_v6_systemic_impact_20260714"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
export SNAPSHOT_WORKERS="${SNAPSHOT_WORKERS:-16}"
export AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
export TRANSITION_EVAL_BATCH_SIZE="${TRANSITION_EVAL_BATCH_SIZE:-48}"
export SEED="${SEED:-17}"
export FOLD1_EDGE_SHA256="${FOLD1_EDGE_SHA256:-d78855280a6b1cf69ba1850bf556b69e9b157c66ae59293534a59541a42ea71d}"
export FOLD2_EDGE_SHA256="${FOLD2_EDGE_SHA256:-fecf7d9e21c131248b37e99d47afe84cfc1112deebfb84b94808b5e925c84f9f}"

exec bash "$(dirname "$0")/run_broad_transition_jepa_v4_a5000.sh"
