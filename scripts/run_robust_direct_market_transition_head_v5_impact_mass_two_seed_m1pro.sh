#!/usr/bin/env bash
set -Eeuo pipefail

bash "$(dirname "$0")/run_robust_direct_market_transition_head_v5_impact_mass_m1pro.sh"
bash "$(dirname "$0")/run_robust_direct_market_transition_head_v5_impact_mass_seed43_m1pro.sh"
