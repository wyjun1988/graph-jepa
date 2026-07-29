#!/bin/bash
# H6-1 placebo arm. Contract: configs/pilot-h61-plan-timing-v1-20260717.json
#
#   "each node's plan is scored against another node's realized path. Gradient
#    path, loss magnitude and weight distribution are preserved; only the
#    node-to-plan pairing breaks. A full-arm advantage that does not survive
#    this is not timing skill."
#
# Conditional in the contract on the full arm training and preserving state
# skill. Both happened (24/24, rule 3 PASS at every horizon), so it runs.
#
# --plan-permute-seed 43 is the contract's value. base() is extracted verbatim
# from launch_h61.sh so the arm differs from `full` in exactly one flag.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
sed -n '/^base() {/,/^}/p' scripts/launch_h61.sh > /tmp/h61_base_placebo.sh
grep -q "external-cache-dir" /tmp/h61_base_placebo.sh || { echo "ABORT: base() extraction failed"; exit 1; }
source /tmp/h61_base_placebo.sh
grep -q "metrics.update(plan_diagnostics)" stock_v2/graph_jepa.py || { echo "ABORT: plan diagnostics not wired"; exit 1; }
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 4000 ]; do sleep 60; done
echo "=== PLACEBO (node permutation, seed 43) $(date -u) ==="
rm -rf reports/pilot_h61_placebo_seed17_20260717 models/pilot_h61_placebo_seed17_20260717
$PY scripts/run_real_backtest.py \
  $(base "--downstream-auxiliary-loss-weight 0.25 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01 --plan-permute-seed 43" pilot_h61_placebo_seed17_20260717) \
  > logs/pilot_h61_placebo_seed17_20260717.log 2>&1
echo "placebo exit=$? $(date -u)"
grep -E "^epoch=(01|24) " logs/pilot_h61_placebo_seed17_20260717.log | grep -o "epoch=[0-9]*\|plan_adv=[-+0-9.]*\|plan_oracle_adv=[-+0-9.]*\|downstream_aux=[0-9.]*" | tr "\n" " "; echo
touch reports/H61_PLACEBO_COMPLETE
