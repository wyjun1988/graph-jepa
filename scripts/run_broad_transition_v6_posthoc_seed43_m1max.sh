#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/Users/wooyeol/work/stock-v2}"
cd "$ROOT"

if [[ ! -f reports/broad_transition_jepa_v5_systemic_seed43_20260714/PIPELINE_COMPLETE ]]; then
  printf '%s\n' "main JEPA pipeline is not complete" >&2
  exit 2
fi

bash scripts/run_cached_raw_market_transition_head_v6_seed43_m1max.sh
bash scripts/run_major_node_transition_shape_v6_seed43_m1max.sh
