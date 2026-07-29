set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v11_fixeddata_5fold_seed17_20260718

# v9's configuration on the corrected panel, all five rolling folds. This is the
# operational-candidate critical path: the same intent gate, the same per-fold
# frontiers REBUILT on the corrected panel, no threshold moves.
#
# Refuse GPU unless both data fixes are present. A five-fold run on the old panel
# would silently re-inherit the two defects the audit found.
grep -q 'carried flat' stock_v2/market_data.py || { echo 'ABORT: suspension fix absent'; exit 1; }
grep -q '_discrete_quarters' stock_v2/fundamental_features.py || { echo 'ABORT: accounting-basis fix absent'; exit 1; }
echo "both data fixes present $(date -u)"

check_quota() {
  dd if=/dev/zero of=/workspace/_q.bin bs=1M count=400 2>/dev/null
  local sz; sz=$(stat -c%s /workspace/_q.bin 2>/dev/null || echo 0)
  rm -f /workspace/_q.bin
  if [ "$sz" -lt 419430400 ]; then echo "ABORT: /workspace truncates writes ($sz)"; df -h /workspace | tail -1; exit 1; fi
}

sed -n '/^fold_args() {/,/^}/p' scripts/launch_v7_5fold.sh \
  | sed "s|reports/\${RUN}|reports/${RUN}|g; s|models/\${RUN}|models/${RUN}|g" > /tmp/v11_5f.sh
source /tmp/v11_5f.sh

# v9's flags exactly; mask stays `mixed` (v12 proved graph-on hurts prediction).
v11_args() {
  fold_args "$1" "$2" "$3" \
    | sed 's|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |' \
    | sed 's|$| --downstream-continuation-weight 1.0 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01|'
}

# Verify substitutions on r3 before spending any GPU.
P=$(v11_args 2024-11-05 2024-01-03 r3)
echo "$P" | grep -q -- '--mask-strategy mixed ' || { echo 'ABORT: v9 mask not preserved'; exit 1; }
echo "$P" | grep -q -- '--mask-strategy operational_mixed' && { echo 'ABORT: v10 mask leaked in'; exit 1; }
echo "$P" | grep -q -- '--temporal-graph-neighbor-scale 0.0' || { echo 'ABORT: temporal graph must stay off (v12 verdict)'; exit 1; }
echo "$P" | grep -q -- '--latent-loss-weight 0.25' || { echo 'ABORT: latent weight must stay at v9 value'; exit 1; }
echo "$P" | grep -q -- '--downstream-continuation-weight 1.0' || { echo 'ABORT: continuation flag lost'; exit 1; }
echo "$P" | grep -q -- '--max-tickers 500' || { echo 'ABORT: fold_args broken'; exit 1; }
echo "v11 5fold args verified: $(echo "$P" | wc -w) flags"

FOLDS=("2023-03-06:2022-05-09:r1" "2024-01-03:2023-03-06:r2" "2024-11-05:2024-01-03:r3" \
       "2025-09-05:2024-11-05:r4" "2026-07-10:2025-09-05:r5")
mkdir -p logs models/${RUN}
for f in "${FOLDS[@]}"; do
  IFS=: read END TE TAG <<< "$f"
  if [ -f models/${RUN}/${TAG}/graph_jepa_real.pt ]; then echo "  ${TAG} already done, skip"; continue; fi
  check_quota
  echo "--- TRAIN ${TAG} (end=$END train_end=$TE) $(date -u) ---"
  $PY scripts/run_real_backtest.py $(v11_args "$END" "$TE" "$TAG") > logs/${RUN}_${TAG}.log 2>&1
  echo "  ${TAG} exit=$? $(date -u)"
  grep -E '^epoch' logs/${RUN}_${TAG}.log | tail -1 | cut -c1-100
  SZ=$(stat -c%s models/${RUN}/${TAG}/graph_jepa_real.pt 2>/dev/null || echo 0)
  [ "$SZ" -gt 1000000 ] || { echo "  ABORT: ${TAG} checkpoint truncated ($SZ) -- disk quota"; exit 1; }
done
touch reports/V11_5FOLD_DONE
echo "=== v11 5fold complete $(date -u) ==="
