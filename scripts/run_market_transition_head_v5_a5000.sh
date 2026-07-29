#!/usr/bin/env bash
set -Eeuo pipefail

export MODEL_NAME="broad_transition_jepa_v5_systemic_seed17_20260714"
export RUN_NAME="market_transition_head_jepa_v5_20260714"
export TARGET_VERSION="market_transition_v5_systemic_impact_20260714"
exec bash "$(dirname "$0")/run_market_transition_head_v4_a5000.sh"
