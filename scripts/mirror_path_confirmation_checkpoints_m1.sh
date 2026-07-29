#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
PYTHON="/Users/wooyeol/work/stock/venv/bin/python"
REMOTE="root@194.68.245.170"
RSH=(ssh -p 22045 -i /Users/wooyeol/.ssh/id_ed25519 -o BatchMode=yes)
REMOTE_ROOT="/workspace/stock-v2"
RUN_ROOT="walk_forward_causal453_path_v2_20260713"
MODEL_NAME="strict_causal453_path_v2_path_w12_p005_l025_skip_seed17_fold2_20241230_to_20260710"
LOCAL_MODEL="$ROOT/models/$RUN_ROOT/$MODEL_NAME"
REMOTE_MODEL="$REMOTE_ROOT/models/$RUN_ROOT/$MODEL_NAME"
REPORT_ROOT="$ROOT/reports/path_v2_crosshost_m1_20260713/fold2"

cd "$ROOT"
mkdir -p "$LOCAL_MODEL" "$REPORT_ROOT"
export PYTHONUNBUFFERED=1

wait_for_remote_file() {
  local path="$1"
  while ! "${RSH[@]}" "$REMOTE" "test -f '$path'"; do
    sleep 30
  done
}

evaluate_checkpoint() {
  local label="$1"
  local remote_dir="$2"
  local local_dir="$3"
  local output_parent="$4"

  if [[ -f "$output_parent/$label/summary.json" ]]; then
    echo "M1 evaluation already complete: $label"
    return
  fi
  wait_for_remote_file "$remote_dir/graph_jepa_real.pt"
  mkdir -p "$local_dir" "$output_parent"
  rsync -az -e "${RSH[*]}" "$REMOTE:$remote_dir/" "$local_dir/"
  "$PYTHON" scripts/evaluate_node_prediction.py \
    --model-dir "$local_dir" \
    --output-dir "$output_parent" \
    --horizons 1,2,3,5,10 \
    --mask-strategy mixed \
    --max-steps 0 \
    --edge-cache-workers 8 \
    --device mps \
    --seed 17 \
    --cache-dir data/staging/ohlcv_causal_return_index_krx500_pit_20260710_v2/ohlcv \
    --external-cache-dir data/external_cache
}

evaluate_checkpoint \
  epoch_016 \
  "$REMOTE_MODEL/epoch_016" \
  "$LOCAL_MODEL/epoch_016" \
  "$REPORT_ROOT/checkpoints"

evaluate_checkpoint \
  "$MODEL_NAME" \
  "$REMOTE_MODEL" \
  "$LOCAL_MODEL" \
  "$REPORT_ROOT/final"

touch "$ROOT/reports/path_v2_crosshost_m1_20260713/FOLD2_MIRROR_COMPLETE"
echo "Fold 2 epoch-16 and epoch-24 M1 evaluations complete."
