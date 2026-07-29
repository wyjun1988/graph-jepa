set -u
cd /workspace/stock-v2-pilot-v7
PY=/root/venvs/stock-v2-cu128/bin/python
RUN=v14_hidden_completion_seed17_20260718

# v11's configuration on the corrected panel, plus the hidden-completion head that
# estimates t's unobserved investor flow from the graph-on completion context.
# One flag against v11: --hidden-completion-weight 0.0 -> 0.25.

grep -q 'carried flat' stock_v2/market_data.py || { echo 'ABORT: suspension fix absent'; exit 1; }
grep -q '_discrete_quarters' stock_v2/fundamental_features.py || { echo 'ABORT: accounting-basis fix absent'; exit 1; }
grep -q 'hidden_completion' stock_v2/graph_jepa.py || { echo 'ABORT: hidden-completion patch absent from model'; exit 1; }
grep -q 'attach_hidden_completion_targets' scripts/run_real_backtest.py || { echo 'ABORT: hidden-completion patch absent from trainer'; exit 1; }
echo "data fixes + hidden-completion patch present $(date -u)"

check_quota() {
  dd if=/dev/zero of=/workspace/_q.bin bs=1M count=400 2>/dev/null
  local sz; sz=$(stat -c%s /workspace/_q.bin 2>/dev/null || echo 0)
  rm -f /workspace/_q.bin
  if [ "$sz" -lt 419430400 ]; then echo "ABORT: /workspace truncates writes ($sz)"; df -h /workspace | tail -1; exit 1; fi
}
check_quota; echo "write check OK"

sed -n '/^fold_args() {/,/^}/p' scripts/launch_v7_5fold.sh \
  | sed "s|reports/\${RUN}|reports/${RUN}|g; s|models/\${RUN}|models/${RUN}|g" > /tmp/v14.sh
source /tmp/v14.sh

# v11's flags plus the hidden-completion weight. The head reads the graph-on
# completion context; the target is t's flow, disclosed at t+1.
v14_args() {
  fold_args "$1" "$2" "$3" \
    | sed 's|--downstream-auxiliary-loss-weight 0.0 |--downstream-auxiliary-loss-weight 0.25 |' \
    | sed 's|$| --downstream-continuation-weight 1.0 --downstream-plan-loss-weight 0.25 --plan-temperature 0.01 --hidden-completion-weight 0.25|'
}
P=$(v14_args 2024-11-05 2024-01-03 r3)
echo "$P" | grep -q -- '--hidden-completion-weight 0.25' || { echo 'ABORT: hidden weight not applied'; exit 1; }
echo "$P" | grep -q -- '--mask-strategy mixed ' || { echo 'ABORT: v9 mask not preserved'; exit 1; }
echo "$P" | grep -q -- '--temporal-graph-neighbor-scale 0.0' || { echo 'ABORT: temporal graph must stay off'; exit 1; }
echo "$P" | grep -q -- '--latent-loss-weight 0.25' || { echo 'ABORT: latent weight changed'; exit 1; }
echo "$P" | grep -q -- '--downstream-continuation-weight 1.0' || { echo 'ABORT: continuation flag lost'; exit 1; }
echo "$P" | grep -q -- '--max-tickers 500' || { echo 'ABORT: fold_args broken'; exit 1; }
echo "$P" | grep -q "models/${RUN}/r3" || { echo 'ABORT: path substitution wrong'; exit 1; }
echo "v14 args verified: $(echo "$P" | wc -w) flags -- one change from v11"

mkdir -p logs models/${RUN}
echo "--- TRAIN v14 r3 (hidden-completion head, corrected panel) $(date -u) ---"
$PY scripts/run_real_backtest.py $P > logs/${RUN}_r3.log 2>&1
echo "train exit=$? $(date -u)"
grep -E '^epoch' logs/${RUN}_r3.log | tail -1 | cut -c1-130
echo "--- hidden_completion_loss 궤적 ---"
grep -oE 'hidden_completion_loss=[0-9.]+' logs/${RUN}_r3.log | head -3
grep -oE 'hidden_completion_loss=[0-9.]+' logs/${RUN}_r3.log | tail -1
