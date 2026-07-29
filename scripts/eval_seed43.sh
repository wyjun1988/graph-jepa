#!/bin/bash
# Evaluate the seed-43 folds against the same frozen intent gate as seed 17.
# Training reproduced seed 17 to three decimals on all five folds; the question
# now is whether the VERDICTS reproduce, which needs the same four evaluations.
set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v9_full_intent_seed43_20260717
for f in r1 r2 r3 r4 r5; do
  M=models/${RUN}/${f}
  [ -f "$M/graph_jepa_real.pt" ] || { echo "$f: no checkpoint"; continue; }
  echo "--- $f $(date -u) ---"
  $PY scripts/evaluate_node_prediction.py --model-dir "$M" \
    --output-dir reports/s43_node_eval_${f}_20260717 --horizons 1,2,3,5,10 \
    --state-target-scope all --mask-strategy mixed --max-steps 194 --seed 17 \
    --device cuda > logs/s43_node_${f}.log 2>&1
  echo "  node exit=$?"
  $PY scripts/evaluate_downstream_heads.py --model-dir "$M" \
    --output-dir reports/s43_head_quality_${f}_20260717 --device cuda \
    --max-steps 194 --seed 17 > logs/s43_head_${f}.log 2>&1
  echo "  head exit=$?"
  F=reports/daily_continuation_frontier_${f}_20260717/summary.json
  $PY scripts/evaluate_daily_continuation.py --model-dir "$M" \
    --output-dir reports/s43_continuation_${f}_20260717 --frontier "$F" \
    --device cuda --max-steps 194 --seed 17 > logs/s43_cont_${f}.log 2>&1
  echo "  cont exit=$?"
  grep -E "^ +[0-9]+ +[0-9]+" logs/s43_cont_${f}.log | head -3
  $PY scripts/evaluate_plan_timing.py --model-dir "$M" \
    --output-dir reports/s43_plan_${f}_20260717 --scale-source causal \
    --scale-lookback 60 --parity-summary reports/s43_node_eval_${f}_20260717/${f}/summary.json \
    --device cuda --max-steps 194 --seed 17 > logs/s43_plan_${f}.log 2>&1
  echo "  plan exit=$?"
done
touch reports/S43_EVAL_DONE
echo "=== seed-43 evaluation done $(date -u) ==="
