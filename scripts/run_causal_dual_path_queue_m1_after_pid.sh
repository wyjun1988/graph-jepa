#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 1 ]]; then
  echo "usage: $0 WAIT_PID" >&2
  exit 2
fi

WAIT_PID="$1"
ROOT="/Users/wooyeol/work/stock-v2"
RUNNER="$ROOT/scripts/run_causal_edge_candidate_m1.sh"
QUEUE_ROOT="$ROOT/reports/causal453_edge_candidates_m1_20260713"

cd "$ROOT"
while kill -0 "$WAIT_PID" 2>/dev/null; do
  sleep 30
done
if [[ ! -f "$QUEUE_ROOT/QUEUE_COMPLETE" ]]; then
  echo "base edge-candidate queue exited without completion marker" >&2
  exit 3
fi

run_candidate() {
  local name="$1"
  shift
  if [[ -f "$QUEUE_ROOT/$name/CANDIDATE_COMPLETE" ]]; then
    echo "candidate already complete; skipping: $name"
    return
  fi
  "$RUNNER" "$name" "$@"
}

run_candidate signed_k6_t1_exog signed 6 1.0 1.0 news_ fund_
run_candidate signed_k6_t0_all signed 6 1.0 0.0
run_candidate signed_k6_t0_exog signed 6 1.0 0.0 news_ fund_
run_candidate signed_k6_t025_exog signed 6 1.0 0.25 news_ fund_
run_candidate signed_k6_t05_exog signed 6 1.0 0.5 news_ fund_

touch "$QUEUE_ROOT/DUAL_PATH_QUEUE_COMPLETE"
