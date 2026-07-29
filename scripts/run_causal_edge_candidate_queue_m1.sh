#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
RUNNER="$ROOT/scripts/run_causal_edge_candidate_m1.sh"
QUEUE_ROOT="$ROOT/reports/causal453_edge_candidates_m1_20260713"

cd "$ROOT"
mkdir -p "$QUEUE_ROOT"

run_candidate() {
  local name="$1"
  local mode="$2"
  local top_k="$3"
  if [[ -f "$QUEUE_ROOT/$name/CANDIDATE_COMPLETE" ]]; then
    echo "candidate already complete; skipping: $name"
    return
  fi
  "$RUNNER" "$name" "$mode" "$top_k"
}

run_candidate signed_k2 signed 2
run_candidate signed_k4 signed 4
run_candidate positive_k4 positive 4
run_candidate negative_k4 negative 4
run_candidate abs_k4 abs 4

touch "$QUEUE_ROOT/QUEUE_COMPLETE"
