#!/bin/bash
# Evaluate v9 on the pod. It is the only host that can.
#
# M1 Max carries a FOUR-task graph_jepa.py and cannot load a five-task v9
# checkpoint. It cannot be patched either: it holds the immutable prospective
# ledger, and its 06:05 chain loads a four-task checkpoint on 2026-07-20. M1 Pro
# was patched and could do this, but it is rebooting. That leaves the pod, whose
# tree is already at five tasks because it trained them.
#
# Runs after each fold rather than after the sweep, so the work overlaps the next
# fold's training. These are inference passes -- they cost far less than the
# training they share the card with, and the alternative is leaving them until
# the end and paying for an idle card.
#
# Each fold is scored against ITS OWN frontier: a different train_end means a
# different pre-train-end window and a different bar, and configs/intent-gate-v9
# pins all five.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v9_full_intent_seed17_20260717

for f in r1 r2 r3 r4 r5; do
  echo "=== $f: waiting for training $(date -u) ==="
  while true; do
    N=$(grep -c '^epoch' "logs/${RUN}_${f}.log" 2>/dev/null || echo 0)
    [ "${N:-0}" -ge 24 ] && break
    pgrep -f '[l]aunch_v9_5fold.sh' > /dev/null || { echo "sweep gone; stopping at $f"; exit 1; }
    sleep 60
  done
  M=models/${RUN}/${f}
  [ -f "$M/graph_jepa_real.pt" ] || { echo "$f: no checkpoint"; continue; }
  echo "$f trained $(date -u)"

  echo "--- $f node eval (intent 1) ---"
  $PY scripts/evaluate_node_prediction.py --model-dir "$M" \
    --output-dir reports/v9_node_eval_${f}_20260717 --horizons 1,2,3,5,10 \
    --state-target-scope all --mask-strategy mixed --max-steps 194 --seed 17 \
    --device cuda > logs/v9_node_eval_${f}.log 2>&1
  echo "  exit=$?"

  echo "--- $f head quality (intent 3) ---"
  $PY scripts/evaluate_downstream_heads.py --model-dir "$M" \
    --output-dir reports/v9_head_quality_${f}_20260717 --device cuda \
    --max-steps 194 --seed 17 > logs/v9_head_quality_${f}.log 2>&1
  echo "  exit=$?"

  F=reports/daily_continuation_frontier_${f}_20260717/summary.json
  [ -f "$F" ] || F=reports/daily_continuation_frontier_r3_20260717/summary.json
  echo "--- $f continuation (intent 2), frontier=$(basename "$(dirname "$F")") ---"
  $PY scripts/evaluate_daily_continuation.py --model-dir "$M" \
    --output-dir reports/v9_continuation_${f}_20260717 --frontier "$F" \
    --device cuda --max-steps 194 --seed 17 > logs/v9_continuation_${f}.log 2>&1
  echo "  exit=$?"
  grep -E "^ +[0-9]+ +[0-9]+" logs/v9_continuation_${f}.log | head -3

  echo "--- $f plan timing ---"
  $PY scripts/evaluate_plan_timing.py --model-dir "$M" \
    --output-dir reports/v9_plan_timing_${f}_20260717 --scale-source causal \
    --scale-lookback 60 --parity-summary reports/v9_node_eval_${f}_20260717/${f}/summary.json \
    --device cuda --max-steps 194 --seed 17 > logs/v9_plan_${f}.log 2>&1
  echo "  exit=$?"
  touch reports/V9_FOLD_${f}_EVALUATED
done
touch reports/V9_POD_EVAL_DONE
echo "=== v9 pod evaluation complete $(date -u) ==="
