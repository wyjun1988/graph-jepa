#!/bin/bash
# One-shot: after the original five-fold driver exits (marker appears), make
# launch_v7_5fold.sh a no-op so the master queue does not re-train five folds.
# Editing the script only AFTER its running instance has exited is what makes
# the in-place edit safe (bash reads scripts incrementally).
M=/workspace/stock-v2-pilot-v7/reports/v7_news_targets_seed17_20260717/ALL_FOLDS_COMPLETE
while [ ! -f "$M" ]; do sleep 60; done
sleep 10
S=/workspace/stock-v2-pilot-v7/scripts/launch_v7_5fold.sh
grep -q "ALREADY_COMPLETE_GUARD" "$S" || sed -i "s|^set -u$|set -u\n# ALREADY_COMPLETE_GUARD\nif [ -f $M ]; then echo \"five-fold already complete; skipping\"; exit 0; fi|" "$S"
echo "guard installed at $(date -u)" >> /workspace/stock-v2-pilot-v7/logs/guard_after_5fold.log
