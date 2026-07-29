#!/bin/bash
# v10: train on the failures production actually has.
# Contract: configs/pilot-v10-operational-mask-v1-20260717.json
#
# One flag against v9: --mask-strategy mixed -> operational_mixed. Everything
# else is extracted verbatim from launch_v7_5fold.sh so a difference in outcome
# is attributable to the mask and nothing else.
#
# The disk quota that silently truncated the seed-43 checkpoint at exactly 256MB
# and killed two runs is guarded before any GPU is spent.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=pilot_v10_opmask_seed17_20260717

dd if=/dev/zero of=/workspace/_wcheck.bin bs=1M count=400 2>/dev/null
SZ=$(stat -c%s /workspace/_wcheck.bin 2>/dev/null || echo 0)
rm -f /workspace/_wcheck.bin
[ "$SZ" -lt 419430400 ] && { echo "ABORT: /workspace truncates writes ($SZ/419430400)"; exit 1; }
echo "write check OK"

sed -n "/^fold_args() {/,/^}/p" scripts/launch_v7_5fold.sh \
  | sed "s|reports/\${RUN}|reports/${RUN}|g; s|models/\${RUN}|models/${RUN}|g" > /tmp/v10.sh
source /tmp/v10.sh
v10_args() {
  fold_args "$1" "$2" "$3" \
    | sed "s|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |" \
    | sed "s|--mask-strategy mixed |--mask-strategy operational_mixed |" \
    | sed "s|\$| --downstream-continuation-weight 1.0 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01|"
}
P=$(v10_args 2024-11-05 2024-01-03 probe)
echo "$P" | grep -q -- "--mask-strategy operational_mixed" || { echo "ABORT: mask substitution failed"; exit 1; }
echo "$P" | grep -q -- "--mask-strategy mixed " && { echo "ABORT: mixed survived"; exit 1; }
echo "$P" | grep -q -- "--downstream-continuation-weight 1.0" || { echo "ABORT: continuation flag missing"; exit 1; }
echo "$P" | grep -q -- "--max-tickers 500" || { echo "ABORT: fold_args broken"; exit 1; }
echo "v10 args verified: $(echo "$P" | wc -w) flags, operational_mixed"

echo "--- TRAIN fold3 $(date -u) ---"
rm -rf reports/${RUN} models/${RUN}
$PY scripts/run_real_backtest.py $(v10_args 2024-11-05 2024-01-03 "${RUN}") > "logs/${RUN}.log" 2>&1
RC=$?; N=$(grep -c "^epoch" "logs/${RUN}.log")
echo "v10 exit=$RC epochs=$N/24 $(date -u)"
for e in 01 08 16 24; do
  grep "^epoch=$e " "logs/${RUN}.log" | grep -o "epoch=[0-9]*\|state=[0-9.]*\|downstream_aux=[0-9.]*\|plan_adv=[-+0-9.]*" | tr "\n" " "; echo
done
[ "$N" -ge 24 ] && touch reports/V10_COMPLETE
echo "=== V10 DONE $(date -u) ==="
