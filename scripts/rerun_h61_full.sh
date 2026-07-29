#!/bin/bash
# Re-run the H6-1 full arm after the NaN-gradient fix.
# A 1-epoch smoke runs first: the previous attempt burned 25 minutes producing
# nothing but NaN while reporting exit=0, so prove the loss is finite before
# spending the card again.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python

echo "=== waiting for the GPU (aux_only arm) $(date -u) ==="
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 4000 ]; do sleep 60; done
echo "GPU free $(date -u)"

ARGS=$(grep -o "\-\-start .*--external-cache-dir data/external_cache" scripts/launch_h61.sh | head -1)

echo "=== SMOKE: 1 epoch with the plan loss on ==="
rm -rf reports/h61_smoke models/h61_smoke
$PY scripts/run_real_backtest.py $ARGS \
  --downstream-auxiliary-loss-weight 0.25 --downstream-plan-loss-weight 0.25 \
  --plan-temperature 0.01 --epochs 1 \
  --reports-dir reports/h61_smoke --models-dir models/h61_smoke \
  > logs/h61_smoke.log 2>&1
echo "smoke exit=$?"
LINE=$(grep -m1 "^epoch" logs/h61_smoke.log)
echo "  $LINE"
if echo "$LINE" | grep -q "nan"; then
  echo "=== SMOKE STILL NaN — aborting, do not burn the card ==="
  exit 1
fi
echo "=== smoke clean; running the full arm $(date -u) ==="
rm -rf reports/pilot_h61_full_seed17_20260717 models/pilot_h61_full_seed17_20260717
$PY scripts/run_real_backtest.py $ARGS \
  --downstream-auxiliary-loss-weight 0.25 --downstream-plan-loss-weight 0.25 \
  --plan-temperature 0.01 \
  --reports-dir reports/pilot_h61_full_seed17_20260717 \
  --models-dir models/pilot_h61_full_seed17_20260717 \
  > logs/pilot_h61_full_seed17_20260717.log 2>&1
echo "full exit=$? $(date -u)"
grep "^epoch=24" logs/pilot_h61_full_seed17_20260717.log | cut -c1-120
touch reports/H61_FULL_RERUN_COMPLETE
