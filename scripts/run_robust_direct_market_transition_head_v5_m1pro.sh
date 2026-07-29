#!/usr/bin/env bash
set -Eeuo pipefail

export RUN_NAME="robust_direct_market_transition_head_v5_20260714"
export TARGET_VERSION="market_transition_v5_systemic_impact_20260714"
exec bash "$(dirname "$0")/run_robust_direct_market_transition_head_v4_m1pro.sh"
