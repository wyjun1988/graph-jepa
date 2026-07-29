#!/usr/bin/env bash
set -Eeuo pipefail

export RUN_NAME="${RUN_NAME:-broad_transition_jepa_v6_systemic_seed17_rtxpro4000blackwell_20260714}"
export TARGET_VERSION="market_transition_v6_systemic_impact_20260714"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-12}"
export SNAPSHOT_WORKERS="${SNAPSHOT_WORKERS:-12}"
export AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
export TRANSITION_EVAL_BATCH_SIZE="${TRANSITION_EVAL_BATCH_SIZE:-48}"
export SEED="${SEED:-17}"

exec bash "$(dirname "$0")/run_broad_transition_jepa_v4_a5000.sh"
