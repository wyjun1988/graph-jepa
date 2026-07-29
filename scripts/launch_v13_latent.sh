set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v13_latentablation_seed17_20260717
BASE=v11_fixeddata_seed17_20260717

# Prerequisite: v11 is the comparison base and must have trained cleanly.
# Verified by outcome -- checkpoint bytes and the epoch count in its own log --
# not by a process name.
[ -f models/${BASE}/r3/graph_jepa_real.pt ] || { echo 'ABORT: v11 checkpoint absent'; exit 1; }
SZ=$(stat -c%s models/${BASE}/r3/graph_jepa_real.pt)
[ "$SZ" -gt 1000000 ] || { echo "ABORT: v11 checkpoint truncated ($SZ bytes -- disk quota?)"; exit 1; }
EP=$(grep -c '^epoch' logs/${BASE}_r3.log 2>/dev/null || echo 0)
[ "$EP" -ge 24 ] || { echo "ABORT: v11 only reached ${EP}/24 epochs"; exit 1; }
grep -qi 'nan' logs/${BASE}_r3.log && { echo 'ABORT: v11 log contains nan'; exit 1; }
echo "v11 base verified: ${EP} epochs, ${SZ} bytes, no nan"

# The corrected panel must still be here. v13 changes the latent weight, nothing else.
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
  | sed "s|reports/\${RUN}|reports/${RUN}|g; s|models/\${RUN}|models/${RUN}|g" > /tmp/v13.sh
source /tmp/v13.sh

# v11's flags with the JEPA latent term switched off. The graph flag STAYS at
# v11's 0.0 -- that is v12's question, and moving both would answer neither.
v13_args() {
  fold_args "$1" "$2" "$3" \
    | sed 's|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |' \
    | sed 's|--latent-loss-weight 0.25|--latent-loss-weight 0.0|' \
    | sed 's|$| --downstream-continuation-weight 1.0 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01|'
}
P=$(v13_args 2024-11-05 2024-01-03 r3)
echo "$P" | grep -q -- '--latent-loss-weight 0.0' || { echo 'ABORT: the latent ablation did not apply'; exit 1; }
echo "$P" | grep -q -- '--latent-loss-weight 0.25' && { echo 'ABORT: the old latent weight survived'; exit 1; }
echo "$P" | grep -q -- '--temporal-graph-neighbor-scale 0.0' || { echo 'ABORT: v13 must keep v11 graph flag -- the graph is v12 question'; exit 1; }
echo "$P" | grep -q -- '--mask-strategy mixed ' || { echo 'ABORT: mask changed'; exit 1; }
echo "$P" | grep -q -- '--downstream-continuation-weight 1.0' || { echo 'ABORT: continuation flag lost'; exit 1; }
echo "$P" | grep -q -- '--max-tickers 500' || { echo 'ABORT: fold_args broken'; exit 1; }
echo "$P" | grep -q "models/${RUN}/r3" || { echo 'ABORT: path substitution wrong'; exit 1; }
echo "v13 args verified: $(echo "$P" | wc -w) flags -- one change from v11"

mkdir -p logs models/${RUN}
echo "--- TRAIN v13 r3 (latent term ablated) $(date -u) ---"
$PY scripts/run_real_backtest.py $P > logs/${RUN}_r3.log 2>&1
echo "train exit=$? $(date -u)"
grep -E '^epoch' logs/${RUN}_r3.log | tail -1
