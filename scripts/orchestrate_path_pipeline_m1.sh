#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="/Users/wooyeol/work/stock/venv/bin/python"
REMOTE="root@194.68.245.170"
SSH=(ssh -p 22045 -i /Users/wooyeol/.ssh/id_ed25519 -o BatchMode=yes)
RSYNC_RSH="ssh -p 22045 -i /Users/wooyeol/.ssh/id_ed25519 -o BatchMode=yes"
EDGE_ROOT="reports/causal453_edge_candidates_m1_20260713"
SELECTION_DIR="$EDGE_ROOT/temporal_architecture_selection"
REMOTE_CHECKPOINT_MARKER="/workspace/stock-v2/reports/checkpoint_selection_causal453_hres_v2_20260713/CHECKPOINT_EVAL_COMPLETE"

cd "$ROOT"
while [[ ! -f "$EDGE_ROOT/EXTERNAL_ONLY_QUEUE_COMPLETE" ]]; do
  sleep 60
done
until "${SSH[@]}" "$REMOTE" "test -f '$REMOTE_CHECKPOINT_MARKER'"; do
  sleep 60
done

summary_path() {
  local name="$1"
  printf '%s' "$EDGE_ROOT/$name/walk_forward/node_eval/${name}_fold1_20231229_to_20241230/summary.json"
}

"$PYTHON" scripts/select_temporal_architecture.py \
  --candidate "global_t0_exog,0.0,1.0=$(summary_path signed_k6_t0_exog)" \
  --candidate "global_t025_exog,0.25,1.0=$(summary_path signed_k6_t025_exog)" \
  --candidate "global_t05_exog,0.5,1.0=$(summary_path signed_k6_t05_exog)" \
  --candidate "global_t1_exog,1.0,1.0=$(summary_path signed_k6_t1_exog)" \
  --candidate "external_only_exog,1.0,0.0=$(summary_path signed_k6_external_only_exog)" \
  --output-dir "$SELECTION_DIR"

read -r TEMPORAL_SCALE STOCK_EDGE_SCALE < <(
  "$PYTHON" - <<'PY'
import json
data = json.load(open('reports/causal453_edge_candidates_m1_20260713/temporal_architecture_selection/selection.json'))
print(data['selected_temporal_graph_neighbor_scale'], data['selected_temporal_stock_edge_scale'])
PY
)

rsync -az --no-owner --no-group -e "$RSYNC_RSH" \
  stock_v2 scripts tests "$REMOTE:/workspace/stock-v2/"
"${SSH[@]}" "$REMOTE" "mkdir -p '/workspace/stock-v2/$SELECTION_DIR'"
rsync -az --no-owner --no-group -e "$RSYNC_RSH" "$SELECTION_DIR/" \
  "$REMOTE:/workspace/stock-v2/$SELECTION_DIR/"

"${SSH[@]}" "$REMOTE" 'bash -s' <<REMOTE_SCRIPT
set -e
cd /workspace/stock-v2
/root/venvs/news-vllm-cu128/bin/python -m pytest -q
PID_FILE=reports/causal453_path_v1_pipeline_20260713.pid
if [[ -f "\$PID_FILE" ]] && kill -0 "\$(cat "\$PID_FILE")" 2>/dev/null; then
  echo "path pipeline already active pid=\$(cat "\$PID_FILE")"
  exit 0
fi
nohup nice -n 5 bash scripts/run_path_objective_pipeline_gpu.sh \
  "$TEMPORAL_SCALE" "$STOCK_EDGE_SCALE" \
  > reports/causal453_path_v1_pipeline_20260713.nohup.log 2>&1 &
pid=\$!
echo "\$pid" > "\$PID_FILE"
echo "path_pipeline_pid=\$pid temporal_scale=$TEMPORAL_SCALE stock_edge_scale=$STOCK_EDGE_SCALE"
REMOTE_SCRIPT

if [[ ! -f reports/causal453_path_v1_pull_20260713.pid ]] || \
   ! kill -0 "$(cat reports/causal453_path_v1_pull_20260713.pid 2>/dev/null || true)" 2>/dev/null; then
  nohup caffeinate -dimsu nice -n 15 bash scripts/pull_path_objective_results_m1.sh \
    > logs/causal453_path_v1_pull_20260713.nohup.log 2>&1 &
  echo "$!" > reports/causal453_path_v1_pull_20260713.pid
fi
touch reports/causal453_path_v1_orchestration_started_20260713
