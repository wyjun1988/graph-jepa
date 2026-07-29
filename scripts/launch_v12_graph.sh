set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v12_temporalgraph_seed17_20260717
BASE=v11_fixeddata_seed17_20260717

# Prerequisite: v11 must have trained cleanly. A comparison against a broken base
# is not a comparison. Verified by outcome -- the checkpoint's bytes and the
# epoch count in its own log -- not by a process name.
[ -f models/${BASE}/r3/graph_jepa_real.pt ] || { echo 'ABORT: v11 checkpoint absent'; exit 1; }
SZ=$(stat -c%s models/${BASE}/r3/graph_jepa_real.pt)
[ "$SZ" -gt 1000000 ] || { echo "ABORT: v11 checkpoint truncated ($SZ bytes -- disk quota?)"; exit 1; }
EP=$(grep -c '^epoch' logs/${BASE}_r3.log 2>/dev/null || echo 0)
[ "$EP" -ge 24 ] || { echo "ABORT: v11 only reached ${EP}/24 epochs"; exit 1; }
grep -qi 'nan' logs/${BASE}_r3.log && { echo 'ABORT: v11 log contains nan'; exit 1; }
echo "v11 base verified: ${EP} epochs, ${SZ} bytes, no nan"

# The corrected panel must still be here. v12 changes the graph flag, nothing else.
grep -q 'carried flat' stock_v2/market_data.py || { echo 'ABORT: suspension fix absent'; exit 1; }
grep -q '_discrete_quarters' stock_v2/fundamental_features.py || { echo 'ABORT: accounting-basis fix absent'; exit 1; }

check_quota() {
  dd if=/dev/zero of=/workspace/_q.bin bs=1M count=400 2>/dev/null
  local sz; sz=$(stat -c%s /workspace/_q.bin 2>/dev/null || echo 0)
  rm -f /workspace/_q.bin
  if [ "$sz" -lt 419430400 ]; then echo "ABORT: /workspace truncates writes ($sz)"; df -h /workspace | tail -1; exit 1; fi
}
check_quota; echo "write check OK"

sed -n '/^fold_args() {/,/^}/p' scripts/launch_v7_5fold.sh \
  | sed "s|reports/\${RUN}|reports/${RUN}|g; s|models/\${RUN}|models/${RUN}|g" > /tmp/v12.sh
source /tmp/v12.sh

# v11's flags, with the neighbour term switched back on in the temporal path.
v12_args() {
  fold_args "$1" "$2" "$3" \
    | sed 's|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |' \
    | sed 's|--temporal-graph-neighbor-scale 0.0|--temporal-graph-neighbor-scale 1.0|' \
    | sed 's|$| --downstream-continuation-weight 1.0 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01|'
}
P=$(v12_args 2024-11-05 2024-01-03 r3)
echo "$P" | grep -q -- '--temporal-graph-neighbor-scale 1.0' || { echo 'ABORT: the one change did not apply'; exit 1; }
echo "$P" | grep -q -- '--temporal-graph-neighbor-scale 0.0' && { echo 'ABORT: the old scale survived'; exit 1; }
echo "$P" | grep -q -- '--mask-strategy mixed ' || { echo 'ABORT: mask changed'; exit 1; }
echo "$P" | grep -q -- '--downstream-continuation-weight 1.0' || { echo 'ABORT: continuation flag lost'; exit 1; }
echo "$P" | grep -q -- '--max-tickers 500' || { echo 'ABORT: fold_args broken'; exit 1; }
echo "$P" | grep -q "models/${RUN}/r3" || { echo 'ABORT: path substitution wrong'; exit 1; }
echo "v12 args verified: $(echo "$P" | wc -w) flags -- one change from v11"

mkdir -p logs models/${RUN}
echo "--- TRAIN v12 r3 (temporal graph ON) $(date -u) ---"
$PY scripts/run_real_backtest.py $P > logs/${RUN}_r3.log 2>&1
echo "train exit=$? $(date -u)"
grep -E '^epoch' logs/${RUN}_r3.log | tail -1
