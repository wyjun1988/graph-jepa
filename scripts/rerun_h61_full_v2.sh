#!/bin/bash
# Re-run the H6-1 full arm after the NaN-gradient fix.  Contract:
# configs/pilot-h61-plan-timing-v1-20260717.json
#
# Supersedes rerun_h61_full.sh, which had two defects that combined into a
# silent no-op:
#
#   1. It rebuilt the flag list with
#        ARGS=$(grep -o "\-\-start .*--external-cache-dir data/external_cache" ...)
#      but grep -o matches within a single line, and launch_h61.sh spreads base()
#      across backslash-continued lines. No line holds both anchors, so ARGS was
#      the EMPTY STRING and both runs used run_real_backtest defaults -- 28 manual
#      tickers, 41 features -- dying in StockGraphJEPA.__init__ on an empty
#      temporal_head_steps.
#   2. The smoke gate read `LINE=$(grep -m1 "^epoch" ...)` and aborted only if
#      LINE contained "nan". A crashed smoke emits no epoch line at all, so LINE
#      was empty and the gate reported "smoke clean" for a run that never
#      started. Absence of evidence passed as evidence of absence.
#
# Both are fixed here: base() is extracted verbatim from launch_h61.sh rather
# than transcribed or pattern-scraped -- the aux_only arm that trained correctly
# used that exact function, so the arms stay comparable by construction -- and
# the smoke gate now fails closed.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python

# The NaN fix must actually be in the tree; without it this run repeats the
# 25-minute all-NaN burn that reported exit=0.
grep -q "safe_deviation" stock_v2/graph_jepa.py || {
  echo "ABORT: the NaN-gradient fix is not in stock_v2/graph_jepa.py"; exit 1; }

sed -n '/^base() {/,/^}/p' scripts/launch_h61.sh > /tmp/h61_base.sh
grep -q "external-cache-dir" /tmp/h61_base.sh || { echo "ABORT: base() extraction failed"; exit 1; }
# shellcheck disable=SC1091
source /tmp/h61_base.sh
echo "base() extracted: $(base X TAG | wc -w) flags"

while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 4000 ]; do sleep 60; done
echo "GPU free $(date -u)"

# 2 epochs, not 1: base() carries --checkpoint-epochs 8,16 and the trainer
# requires those to be < --epochs, so a 1-epoch smoke cannot parse. Overriding
# to --checkpoint-epochs 1 needs --epochs 2 to stay legal. Two epochs also show
# whether the loss MOVES, which one cannot.
echo "=== SMOKE: 2 epochs with the plan loss on $(date -u) ==="
rm -rf reports/h61_smoke models/h61_smoke
$PY scripts/run_real_backtest.py \
  $(base "--downstream-auxiliary-loss-weight 0.25 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01 --epochs 2 --checkpoint-epochs 1" h61_smoke) \
  > logs/h61_smoke.log 2>&1
echo "smoke exit=$?"

EPOCHS=$(grep "^epoch" logs/h61_smoke.log || true)
if [ -z "$EPOCHS" ]; then
  echo "=== SMOKE PRODUCED NO EPOCH LINE — the run died. Not burning the card. ==="
  tail -12 logs/h61_smoke.log
  exit 1
fi
echo "$EPOCHS" | cut -c1-150
case "$EPOCHS" in
  *nan*) echo "=== SMOKE STILL NaN — aborting ==="; exit 1 ;;
esac
# Decision rule 2: the switches must actually be live. A silently inert loss
# trains clean and proves nothing, which is the whole reason the rule exists.
echo "$EPOCHS" | grep -q "downstream_aux=0.0000" && {
  echo "=== SMOKE: downstream_aux is 0.0000 — the aux loss is inert. Aborting. ==="; exit 1; }
echo "$EPOCHS" | grep -q "plan_adv=" || {
  echo "=== SMOKE: no plan diagnostics — the plan loss is not reporting. Aborting. ==="; exit 1; }
echo "$EPOCHS" | grep -qE "plan_adv=[+-]?nan" && {
  echo "=== SMOKE: plan_adv is nan. Aborting. ==="; exit 1; }
echo "=== smoke clean; running the full arm $(date -u) ==="

rm -rf reports/pilot_h61_full_seed17_20260717 models/pilot_h61_full_seed17_20260717
$PY scripts/run_real_backtest.py \
  $(base "--downstream-auxiliary-loss-weight 0.25 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01" pilot_h61_full_seed17_20260717) \
  > logs/pilot_h61_full_seed17_20260717.log 2>&1
echo "full exit=$? $(date -u)"
grep -E "^epoch=(1|12|24) " logs/pilot_h61_full_seed17_20260717.log | cut -c1-150
touch reports/H61_FULL_RERUN_COMPLETE
echo "=== H6-1 full arm done $(date -u) ==="
