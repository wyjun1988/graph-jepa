#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/work/stock-v2}"
FOLD1_COMPLETE="$ROOT/reports/direct_state_mlp_broad_transition_jepa_v6_systemic_seed17_rtx4000ada_20260714_20260714/fold1/EXPERIMENT_COMPLETE"

until [[ -f "$FOLD1_COMPLETE" ]]; do
  sleep 30
done

FOLD=fold2 exec "$ROOT/scripts/run_direct_state_mlp_v6_m1max.sh"
