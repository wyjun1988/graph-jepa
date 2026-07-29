#!/bin/bash
# Runs after the H6-1 arms: off-parity proof, then the deferred W5 sweep, then v8.
# The v7 five-fold step is gone: r1-r5 are already trained and judged (3 of 4
# untouched folds passed), so re-running it would only burn the card.
set -u
cd /workspace/stock-v2-pilot-v7
while [ ! -f reports/H61_ARMS_COMPLETE ]; do sleep 120; done
echo "=== [1/3] plan-loss off-parity $(date -u) ==="
bash scripts/verify_plan_loss_offparity.sh
echo "=== [2/3] W5 data-efficiency sweep (deferred earlier) $(date -u) ==="
cd /workspace/stock-v2-post-impact && bash scripts/launch_w5.sh > logs/w5_driver.log 2>&1
echo "w5 exit=$? $(date -u)"
cd /workspace/stock-v2-pilot-v7
touch reports/POST_H61_QUEUE_COMPLETE
echo "=== POST-H61 QUEUE DONE $(date -u) ==="
