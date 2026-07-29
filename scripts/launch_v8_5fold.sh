#!/bin/bash
# v8 = v7 news targets + the four downstream heads switched on.
# Contract: configs/rolling-v8-aux-heads-qualification-v1-20260717.json (b89351b3...)
#
# fold_args() is extracted VERBATIM from launch_v7_5fold.sh and then two strings
# are substituted: the run name, and --downstream-auxiliary-loss-weight 0.0 ->
# 0.25. Retyping the ~190 flags would risk a silent difference that the gate
# would then attribute to the heads. Both substitutions are asserted below; if
# either fails to bite, the run aborts rather than training v7 again under a v8
# name.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v8_aux_heads_seed17_20260717

sed -n '/^fold_args() {/,/^}/p' scripts/launch_v7_5fold.sh \
  | sed "s|reports/\${RUN}|reports/${RUN}|g; s|models/\${RUN}|models/${RUN}|g" \
  > /tmp/v8_fold_args.sh
source /tmp/v8_fold_args.sh

v8_args() {
  fold_args "$1" "$2" "$3" | sed "s|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |"
}

PROBE=$(v8_args 2024-11-05 2024-01-03 probe)
echo "$PROBE" | grep -q -- "--downstream-auxiliary-loss-weight 0.25" || { echo "ABORT: aux substitution did not bite"; exit 1; }
echo "$PROBE" | grep -q -- "--downstream-auxiliary-loss-weight 0.0 " && { echo "ABORT: the 0.0 weight survived"; exit 1; }
echo "$PROBE" | grep -q -- "--max-tickers 500" || { echo "ABORT: fold_args extraction is broken"; exit 1; }
echo "$PROBE" | grep -q "reports/${RUN}/probe" || { echo "ABORT: run-name substitution did not bite"; exit 1; }
echo "v8 args verified: $(echo "$PROBE" | wc -w) flags, aux=0.25"

FOLDS=("2023-03-06:2022-05-09:r1" "2024-01-03:2023-03-06:r2" "2024-11-05:2024-01-03:r3" \
       "2025-09-05:2024-11-05:r4" "2026-07-10:2025-09-05:r5")

echo "=== waiting for the GPU $(date -u) ==="
while [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -gt 5000 ]; do sleep 120; done
echo "GPU free $(date -u)"

for f in "${FOLDS[@]}"; do
  IFS=: read END TE TAG <<< "$f"
  echo "--- TRAIN $TAG  $(date -u) ---"
  $PY scripts/run_real_backtest.py $(v8_args "$END" "$TE" "$TAG") > "logs/${RUN}_${TAG}.log" 2>&1
  RC=$?
  N=$(grep -c "^epoch" "logs/${RUN}_${TAG}.log")
  echo "$TAG exit=$RC epochs=$N/24  $(date -u)"
  grep "^epoch=24 " "logs/${RUN}_${TAG}.log" | grep -o "downstream_aux=[0-9.]*\|state=[0-9.]*\|latent=[0-9.]*" | tr "\n" " "; echo
  if [ "$N" -lt 24 ]; then echo "=== $TAG INCOMPLETE — aborting the sweep ==="; tail -8 "logs/${RUN}_${TAG}.log"; exit 1; fi
done
touch reports/${RUN}/ALL_FOLDS_COMPLETE
echo "=== V8 ALL FOLDS DONE $(date -u) ==="
