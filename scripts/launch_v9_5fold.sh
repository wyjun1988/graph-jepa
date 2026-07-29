#!/bin/bash
# v9 = the full intent stack, five folds.
# Gate: configs/intent-gate-v9-20260717.json (a2ac8c64), frozen before any v9 training.
#
#   intent 1  state / latent / imputation losses -- unchanged from v7
#   intent 2  --downstream-continuation-weight 1.0  (the fifth head)
#   intent 3  --downstream-auxiliary-loss-weight 0.25 (all five heads get gradient)
#   plan      --downstream-plan-loss-weight 0.25 with the CAUSAL scale
#
# Chained behind H7 so the card is never idle: H7 is fold 3 and proves the wiring,
# and its result gates whether these five are worth the GPU hours. If H7's own
# contract rejects it, this does not run.
#
# fold_args() is extracted verbatim from launch_v7_5fold.sh and only the run name
# and the three flags above are substituted, each substitution asserted before any
# training starts. Retyping ~190 flags would risk a silent difference the gate
# would then blame on the heads.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v9_full_intent_seed17_20260717

echo "=== waiting for H7 (fold 3, wiring proof) $(date -u) ==="
while [ ! -f reports/H7_COMPLETE ]; do
  if ! pgrep -f "[r]un_h7.sh" > /dev/null && [ ! -f reports/H7_COMPLETE ]; then
    echo "ABORT: the H7 driver is gone and left no completion marker."
    tail -6 logs/h7_driver.log
    exit 1
  fi
  sleep 60
done
echo "H7 complete $(date -u)"

# H7's contract decides whether v9 is worth the hours. Rule 1: it trains.
N=$(grep -c '^epoch' logs/pilot_h7_continuation_plan_seed17_20260717.log)
[ "$N" -lt 24 ] && { echo "ABORT: H7 only reached $N/24 epochs"; exit 1; }
grep '^epoch' logs/pilot_h7_continuation_plan_seed17_20260717.log | grep -q nan && { echo "ABORT: NaN in H7"; exit 1; }
# Rule 2: the leak did not come back. The model-free rule scored +0.01404.
ADV=$(grep '^epoch=24 ' logs/pilot_h7_continuation_plan_seed17_20260717.log | grep -o 'plan_adv=[-+0-9.]*' | cut -d= -f2)
echo "H7 final plan_adv=$ADV  (leak value +0.01404; H6-2's causal run gave +0.00307)"

sed -n '/^fold_args() {/,/^}/p' scripts/launch_v7_5fold.sh \
  | sed "s|reports/\${RUN}|reports/${RUN}|g; s|models/\${RUN}|models/${RUN}|g" > /tmp/v9_fold_args.sh
grep -q 'external-cache-dir' /tmp/v9_fold_args.sh || { echo "ABORT: fold_args extraction failed"; exit 1; }
# shellcheck disable=SC1091
source /tmp/v9_fold_args.sh

v9_args() {  # $1=end $2=train_end $3=tag
  fold_args "$1" "$2" "$3" \
    | sed 's|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |' \
    | sed 's|$| --downstream-continuation-weight 1.0 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01|'
}

PROBE=$(v9_args 2024-11-05 2024-01-03 probe)
for flag in "--downstream-auxiliary-loss-weight 0.25" "--downstream-continuation-weight 1.0" \
            "--downstream-plan-loss-weight 0.25" "--max-tickers 500" "reports/${RUN}/probe"; do
  echo "$PROBE" | grep -q -- "$flag" || { echo "ABORT: '$flag' missing from the command"; exit 1; }
done
echo "$PROBE" | grep -q -- "--downstream-auxiliary-loss-weight 0.0 " && { echo "ABORT: the 0.0 weight survived"; exit 1; }
echo "v9 args verified: $(echo "$PROBE" | wc -w) flags"

FOLDS=("2023-03-06:2022-05-09:r1" "2024-01-03:2023-03-06:r2" "2024-11-05:2024-01-03:r3" \
       "2025-09-05:2024-11-05:r4" "2026-07-10:2025-09-05:r5")

for f in "${FOLDS[@]}"; do
  IFS=: read END TE TAG <<< "$f"
  echo "--- TRAIN $TAG  $(date -u) ---"
  $PY scripts/run_real_backtest.py $(v9_args "$END" "$TE" "$TAG") > "logs/${RUN}_${TAG}.log" 2>&1
  RC=$?
  N=$(grep -c '^epoch' "logs/${RUN}_${TAG}.log")
  echo "$TAG exit=$RC epochs=$N/24 $(date -u)"
  grep '^epoch=24 ' "logs/${RUN}_${TAG}.log" | grep -o 'downstream_aux=[0-9.]*\|plan_adv=[-+0-9.]*\|state=[0-9.]*' | tr '\n' ' '; echo
  if [ "$N" -lt 24 ]; then
    echo "=== $TAG INCOMPLETE — aborting the sweep rather than gating a partial run ==="
    tail -8 "logs/${RUN}_${TAG}.log"
    exit 1
  fi
done
mkdir -p reports/${RUN}
touch reports/${RUN}/ALL_FOLDS_COMPLETE
echo "=== V9 ALL FOLDS DONE $(date -u) ==="
