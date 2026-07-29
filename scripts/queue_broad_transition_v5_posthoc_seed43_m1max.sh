#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
cd "$ROOT"

until [[ -f reports/broad_transition_jepa_v5_systemic_seed43_20260714/PIPELINE_COMPLETE ]]; do
  sleep 60
done

bash scripts/run_market_transition_head_v5_impact_mass_seed43_m1max.sh
bash scripts/run_major_node_transition_shape_v5_impact_mass_seed43_m1max.sh
