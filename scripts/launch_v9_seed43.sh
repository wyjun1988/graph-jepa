#!/bin/bash
# v9 seed-43 replication. Same configuration as v9_full_intent_seed17 in every
# flag but the seed; run BEFORE the seed-17 gate verdict exists, so it is a
# robustness replication, not tuning. A five-fold result that only one seed can
# produce is a seed anecdote; two seeds passing the same frozen gate is a model
# property.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v9_full_intent_seed43_20260717

sed -n "/^fold_args() {/,/^}/p" scripts/launch_v7_5fold.sh \
  | sed "s|reports/\${RUN}|reports/${RUN}|g; s|models/\${RUN}|models/${RUN}|g" > /tmp/v9s43.sh
source /tmp/v9s43.sh
v9_args() {
  fold_args "$1" "$2" "$3" \
    | sed "s|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |" \
    | sed "s|--seed 17 |--seed 43 |" \
    | sed "s|\$| --downstream-continuation-weight 1.0 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01|"
}
P=$(v9_args 2024-11-05 2024-01-03 probe)
echo "$P" | grep -q -- "--seed 43 " || { echo "ABORT: seed substitution failed"; exit 1; }
echo "$P" | grep -q -- "--seed 17 " && { echo "ABORT: seed 17 survived"; exit 1; }
echo "$P" | grep -q -- "--downstream-continuation-weight 1.0" || { echo "ABORT: continuation flag missing"; exit 1; }
echo "$P" | grep -q -- "--max-tickers 500" || { echo "ABORT: fold_args broken"; exit 1; }
echo "seed-43 args verified: $(echo "$P" | wc -w) flags"

FOLDS=("2023-03-06:2022-05-09:r1" "2024-01-03:2023-03-06:r2" "2024-11-05:2024-01-03:r3" \
       "2025-09-05:2024-11-05:r4" "2026-07-10:2025-09-05:r5")
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 6000 ]; do sleep 60; done
for f in "${FOLDS[@]}"; do
  IFS=: read END TE TAG <<< "$f"
  echo "--- TRAIN $TAG  $(date -u) ---"
  $PY scripts/run_real_backtest.py $(v9_args "$END" "$TE" "$TAG") > "logs/${RUN}_${TAG}.log" 2>&1
  RC=$?; N=$(grep -c "^epoch" "logs/${RUN}_${TAG}.log")
  echo "$TAG exit=$RC epochs=$N/24 $(date -u)"
  grep "^epoch=24 " "logs/${RUN}_${TAG}.log" | grep -o "state=[0-9.]*\|downstream_aux=[0-9.]*\|plan_adv=[-+0-9.]*" | tr "\n" " "; echo
  [ "$N" -lt 24 ] && { echo "=== $TAG INCOMPLETE — abort ==="; exit 1; }
done
mkdir -p reports/${RUN}; touch reports/${RUN}/ALL_FOLDS_COMPLETE
echo "=== SEED-43 ALL FOLDS DONE $(date -u) ==="
