#!/bin/bash
# v10 five folds. Contract: configs/rolling-v10-operational-mask-5fold-v1-20260717.json (02f4080f)
#
# The previous attempt died between the 48-epoch run and this sweep: /workspace
# hit its quota and every write returned "Disk quota exceeded". That failure is
# SILENT -- torch.save truncates and the process dies with no error in the log --
# and it has now cost three runs today. So the quota is checked before EVERY fold,
# not just at the start, and the sweep stops loudly rather than burning GPU on
# checkpoints that cannot be written.
#
# One flag against v9: --mask-strategy mixed -> operational_mixed.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v10_opmask_5fold_seed17_20260717

check_quota() {
  dd if=/dev/zero of=/workspace/_q.bin bs=1M count=400 2>/dev/null
  local sz; sz=$(stat -c%s /workspace/_q.bin 2>/dev/null || echo 0)
  rm -f /workspace/_q.bin
  if [ "$sz" -lt 419430400 ]; then
    echo "ABORT: /workspace truncates writes ($sz/419430400). Free space before spending GPU."
    df -h /workspace | tail -1
    exit 1
  fi
}
check_quota; echo "write check OK $(date -u)"

sed -n "/^fold_args() {/,/^}/p" scripts/launch_v7_5fold.sh \
  | sed "s|reports/\${RUN}|reports/${RUN}|g; s|models/\${RUN}|models/${RUN}|g" > /tmp/v10_5f.sh
source /tmp/v10_5f.sh
v10_args() {  # $1=end $2=train_end $3=tag  -- tag is the FOLD, not the run name
  fold_args "$1" "$2" "$3" \
    | sed "s|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |" \
    | sed "s|--mask-strategy mixed |--mask-strategy operational_mixed |" \
    | sed "s|\$| --downstream-continuation-weight 1.0 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01|"
}
P=$(v10_args 2024-11-05 2024-01-03 r3)
echo "$P" | grep -q -- "--mask-strategy operational_mixed" || { echo "ABORT: mask flag lost"; exit 1; }
echo "$P" | grep -q -- "--mask-strategy mixed " && { echo "ABORT: mixed survived"; exit 1; }
echo "$P" | grep -q -- "--downstream-continuation-weight 1.0" || { echo "ABORT: continuation flag lost"; exit 1; }
echo "$P" | grep -q -- "--max-tickers 500" || { echo "ABORT: fold_args broken"; exit 1; }
echo "$P" | grep -q "models/${RUN}/r3" || { echo "ABORT: path substitution wrong -- $(echo "$P" | grep -o "models/[a-z0-9_/]*")"; exit 1; }
echo "v10 5fold args verified: $(echo "$P" | wc -w) flags"

FOLDS=("2023-03-06:2022-05-09:r1" "2024-01-03:2023-03-06:r2" "2024-11-05:2024-01-03:r3" \
       "2025-09-05:2024-11-05:r4" "2026-07-10:2025-09-05:r5")
for f in "${FOLDS[@]}"; do
  IFS=: read END TE TAG <<< "$f"
  check_quota
  echo "--- TRAIN $TAG $(date -u) ---"
  $PY scripts/run_real_backtest.py $(v10_args "$END" "$TE" "$TAG") > "logs/${RUN}_${TAG}.log" 2>&1
  RC=$?; N=$(grep -c "^epoch" "logs/${RUN}_${TAG}.log")
  echo "$TAG exit=$RC epochs=$N/24 $(date -u)"
  grep "^epoch=24 " "logs/${RUN}_${TAG}.log" | grep -o "state=[0-9.]*\|downstream_aux=[0-9.]*\|plan_adv=[-+0-9.]*" | tr "\n" " "; echo
  if [ "$N" -lt 24 ]; then
    echo "=== $TAG INCOMPLETE — aborting rather than gating a partial sweep ==="
    tail -6 "logs/${RUN}_${TAG}.log"; exit 1
  fi
  ls "models/${RUN}/${TAG}/graph_jepa_real.pt" >/dev/null 2>&1 || { echo "=== $TAG has no checkpoint — abort ==="; exit 1; }
done
mkdir -p reports/${RUN}; touch reports/${RUN}/ALL_FOLDS_COMPLETE
echo "=== V10 5FOLD DONE $(date -u) ==="
