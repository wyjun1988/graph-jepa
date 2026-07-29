#!/bin/bash
# H6-2: does the plan learn real timing once the leak is out of the loss?
# Contract: configs/pilot-h62-causal-plan-scale-v1-20260717.json (8afcd8df...)
#
# Same command line as the H6-1 full arm -- base() extracted verbatim from
# launch_h61.sh -- so the ONLY difference is the patched causal plan scale.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python

grep -q "target_downstream_causal_scale" stock_v2/graph_jepa.py || { echo "ABORT: causal scale patch missing from the model"; exit 1; }
grep -q "target_downstream_causal_scale" scripts/run_real_backtest.py || { echo "ABORT: causal scale patch missing from the trainer"; exit 1; }
sed -n '/^base() {/,/^}/p' scripts/launch_h61.sh > /tmp/h62_base.sh
grep -q "external-cache-dir" /tmp/h62_base.sh || { echo "ABORT: base() extraction failed"; exit 1; }
source /tmp/h62_base.sh

echo "=== waiting for the GPU (W5) $(date -u) ==="
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 5000 ]; do sleep 60; done
echo "GPU free $(date -u)"

echo "=== SMOKE: 2 epochs with the causal plan scale $(date -u) ==="
rm -rf reports/h62_smoke models/h62_smoke
$PY scripts/run_real_backtest.py \
  $(base "--downstream-auxiliary-loss-weight 0.25 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01 --epochs 2 --checkpoint-epochs 1" h62_smoke) \
  > logs/h62_smoke.log 2>&1
echo "smoke exit=$?"
EPOCHS=$(grep "^epoch" logs/h62_smoke.log || true)
[ -z "$EPOCHS" ] && { echo "=== SMOKE DIED — not burning the card ==="; tail -12 logs/h62_smoke.log; exit 1; }
echo "$EPOCHS" | cut -c1-170
case "$EPOCHS" in *nan*) echo "=== SMOKE NaN — abort ==="; exit 1 ;; esac
echo "$EPOCHS" | grep -q "plan_adv=" || { echo "=== SMOKE: no plan diagnostics — abort ==="; exit 1; }

# Decision rule 2: landing back on the model-free leak value would mean the
# leak survived the patch and the whole run would be meaningless.
LEAK=$(echo "$EPOCHS" | tail -1 | grep -o "plan_adv=[-+0-9.]*" | cut -d= -f2)
echo "  smoke plan_adv=$LEAK  (model-free leak value was +0.01404; landing there means the leak survived)"

echo "=== FULL RUN $(date -u) ==="
rm -rf reports/pilot_h62_causal_seed17_20260717 models/pilot_h62_causal_seed17_20260717
$PY scripts/run_real_backtest.py \
  $(base "--downstream-auxiliary-loss-weight 0.25 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01" pilot_h62_causal_seed17_20260717) \
  > logs/pilot_h62_causal_seed17_20260717.log 2>&1
echo "h62 exit=$? $(date -u)"
for e in 01 08 16 24; do grep "^epoch=$e " logs/pilot_h62_causal_seed17_20260717.log | grep -o "epoch=[0-9]*\|downstream_aux=[0-9.]*\|plan_adv=[-+0-9.]*\|plan_oracle_adv=[-+0-9.]*\|plan_entropy=[0-9.]*" | tr "\n" " "; echo; done
touch reports/H62_COMPLETE
echo "=== H6-2 DONE $(date -u) ==="
