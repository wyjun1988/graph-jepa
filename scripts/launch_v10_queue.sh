#!/bin/bash
# Two v10 runs, queued back to back. The RunPod slot is scarce and the card is
# otherwise idle, so both are launched BEFORE the fold-3 verdict rather than
# after: if the verdict rejects v10, these are discarded and nothing is lost but
# electricity already being paid for. Judgement is by contracts frozen before any
# v10 number existed (483bfc8a and the 5-fold contract below), so running early
# is not selecting on results.
#
# [1] 48 epochs, fold 3. v9 is CONVERGED at 24 (loss -0.0026 over epochs 20-24);
#     v10 is not (-0.0103 over the same span, 4x steeper). Judging an
#     undertrained v10 against a converged v9 is a rigged comparison, and this is
#     the run that removes the excuse.
# [2] 24 epochs, five folds. Needed only if the fold-3 verdict passes -- but it
#     cannot be run later if the slot is gone.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python

# The quota that silently truncated a checkpoint at 256MB killed two runs before.
dd if=/dev/zero of=/workspace/_wcheck.bin bs=1M count=400 2>/dev/null
SZ=$(stat -c%s /workspace/_wcheck.bin 2>/dev/null || echo 0); rm -f /workspace/_wcheck.bin
[ "$SZ" -lt 419430400 ] && { echo "ABORT: /workspace truncates writes ($SZ)"; exit 1; }
echo "write check OK"

base_args() {  # $1=end $2=train_end $3=tag $4=RUN $5=extra
  sed -n "/^fold_args() {/,/^}/p" scripts/launch_v7_5fold.sh \
    | sed "s|reports/\${RUN}|reports/$4|g; s|models/\${RUN}|models/$4|g" > /tmp/_fa_$4.sh
  source /tmp/_fa_$4.sh
  fold_args "$1" "$2" "$3" \
    | sed "s|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |" \
    | sed "s|--mask-strategy mixed |--mask-strategy operational_mixed |" \
    | sed "s|\$| --downstream-continuation-weight 1.0 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01 $5|"
}

echo "=== [1/2] v10 48 epochs, fold3 $(date -u) ==="
R1=pilot_v10_opmask_48ep_seed17_20260717
A=$(base_args 2024-11-05 2024-01-03 "$R1" "$R1" "--epochs 48 --checkpoint-epochs 16,32")
echo "$A" | grep -q -- "--mask-strategy operational_mixed" || { echo "ABORT: mask flag lost"; exit 1; }
echo "$A" | grep -q -- "--epochs 48" || { echo "ABORT: epoch flag lost"; exit 1; }
rm -rf reports/$R1 models/$R1
$PY scripts/run_real_backtest.py $A > "logs/${R1}.log" 2>&1
echo "48ep exit=$? epochs=$(grep -c "^epoch" logs/${R1}.log)/48 $(date -u)"
for e in 24 32 40 48; do grep "^epoch=$e " "logs/${R1}.log" | grep -o "epoch=[0-9]*\|loss=[0-9.]*\|state=[0-9.]*\|current_impute=[0-9.]*\|plan_adv=[-+0-9.]*" | tr "\n" " "; echo; done

echo "=== [2/2] v10 five folds, 24 epochs $(date -u) ==="
R2=v10_opmask_5fold_seed17_20260717
FOLDS=("2023-03-06:2022-05-09:r1" "2024-01-03:2023-03-06:r2" "2024-11-05:2024-01-03:r3" \
       "2025-09-05:2024-11-05:r4" "2026-07-10:2025-09-05:r5")
for f in "${FOLDS[@]}"; do
  IFS=: read END TE TAG <<< "$f"
  echo "--- TRAIN $TAG $(date -u) ---"
  A=$(base_args "$END" "$TE" "$TAG" "$R2" "")
  $PY scripts/run_real_backtest.py $A > "logs/${R2}_${TAG}.log" 2>&1
  N=$(grep -c "^epoch" "logs/${R2}_${TAG}.log")
  echo "$TAG exit=$? epochs=$N/24 $(date -u)"
  grep "^epoch=24 " "logs/${R2}_${TAG}.log" | grep -o "state=[0-9.]*\|downstream_aux=[0-9.]*\|plan_adv=[-+0-9.]*" | tr "\n" " "; echo
  [ "$N" -lt 24 ] && { echo "=== $TAG INCOMPLETE — abort ==="; exit 1; }
done
mkdir -p reports/$R2; touch reports/$R2/ALL_FOLDS_COMPLETE
echo "=== V10 QUEUE DONE $(date -u) ==="
