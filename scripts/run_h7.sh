#!/bin/bash
# H7: the full intent stack -- intent 2's continuation head + the causal plan loss.
# Contract: configs/pilot-h7-continuation-plan-v1-20260717.json (c1b32cad)
#
# Waits for the W5 030 repair to finish and leave the card. Running these
# concurrently is exactly what killed W5's latent_030 and placebo_030 with CUDA
# OOM earlier today: this training peaks near 12.7 GiB and W5 holds ~3.7 GiB of a
# 19.67 GiB card, which fits until it does not. Not repeating that.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python

grep -q "continuation_rate" stock_v2/graph_jepa.py || { echo "ABORT: continuation task missing"; exit 1; }
grep -q "target_downstream_causal_scale" stock_v2/graph_jepa.py || { echo "ABORT: causal plan scale missing"; exit 1; }
grep -q "metrics.update(plan_diagnostics)" stock_v2/graph_jepa.py || { echo "ABORT: plan diagnostics not wired"; exit 1; }

sed -n '/^fold_args() {/,/^}/p' scripts/launch_v7_5fold.sh \
  | sed 's|reports/${RUN}|reports/h7_tmp|g; s|models/${RUN}|models/h7_tmp|g' > /tmp/h7_base.sh
grep -q "external-cache-dir" /tmp/h7_base.sh || { echo "ABORT: fold_args extraction failed"; exit 1; }
# shellcheck disable=SC1091
source /tmp/h7_base.sh

h7_args() {  # $1 = tag, $2 = extra
  fold_args 2024-11-05 2024-01-03 "$1" \
    | sed 's|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |' \
    | sed "s|reports/h7_tmp/$1|reports/$1|; s|models/h7_tmp/$1|models/$1|" \
    | sed "s|\$| --downstream-continuation-weight 1.0 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01 $2|"
}

PROBE=$(h7_args probe "")
echo "$PROBE" | grep -q -- "--downstream-auxiliary-loss-weight 0.25" || { echo "ABORT: aux substitution failed"; exit 1; }
echo "$PROBE" | grep -q -- "--downstream-continuation-weight 1.0" || { echo "ABORT: continuation flag missing"; exit 1; }
echo "$PROBE" | grep -q -- "--downstream-plan-loss-weight 0.25" || { echo "ABORT: plan flag missing"; exit 1; }
echo "$PROBE" | grep -q -- "--max-tickers 500" || { echo "ABORT: fold_args broken"; exit 1; }
echo "$PROBE" | grep -q "reports/probe" || { echo "ABORT: reports path substitution failed"; exit 1; }
echo "h7 args verified: $(echo "$PROBE" | wc -w) flags"

echo "=== waiting for the card to have room (need ~13G of 20G) $(date -u) ==="
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 6000 ]; do sleep 60; done
echo "GPU free $(date -u)"

echo "=== SMOKE: 2 epochs, 5 heads + causal plan $(date -u) ==="
rm -rf reports/h7_smoke models/h7_smoke
$PY scripts/run_real_backtest.py $(h7_args h7_smoke "--epochs 2 --checkpoint-epochs 1") \
  > logs/h7_smoke.log 2>&1
echo "smoke exit=$?"
EPOCHS=$(grep "^epoch" logs/h7_smoke.log || true)
[ -z "$EPOCHS" ] && { echo "=== SMOKE DIED ==="; tail -14 logs/h7_smoke.log; exit 1; }
echo "$EPOCHS" | cut -c1-185
case "$EPOCHS" in *nan*) echo "=== SMOKE NaN — abort ==="; exit 1 ;; esac
echo "$EPOCHS" | grep -q "plan_adv=" || { echo "=== SMOKE: no plan diagnostics — abort ==="; exit 1; }
echo "$EPOCHS" | grep -q "downstream_aux=0.0000" && { echo "=== SMOKE: aux inert — abort ==="; exit 1; }

echo "=== FULL RUN $(date -u) ==="
rm -rf reports/pilot_h7_continuation_plan_seed17_20260717 models/pilot_h7_continuation_plan_seed17_20260717
$PY scripts/run_real_backtest.py $(h7_args pilot_h7_continuation_plan_seed17_20260717 "") \
  > logs/pilot_h7_continuation_plan_seed17_20260717.log 2>&1
RC=$?
N=$(grep -c "^epoch" logs/pilot_h7_continuation_plan_seed17_20260717.log)
echo "h7 exit=$RC epochs=$N/24 $(date -u)"
for e in 01 08 16 24; do
  grep "^epoch=$e " logs/pilot_h7_continuation_plan_seed17_20260717.log \
    | grep -o "epoch=[0-9]*\|downstream_aux=[0-9.]*\|plan_adv=[-+0-9.]*\|plan_oracle_adv=[-+0-9.]*\|state=[0-9.]*" | tr "\n" " "; echo
done
[ "$N" -ge 24 ] && touch reports/H7_COMPLETE
echo "=== H7 DONE $(date -u) ==="
