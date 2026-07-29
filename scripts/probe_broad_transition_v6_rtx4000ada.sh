#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON_BIN="${PYTHON:-/root/venvs/stock-v2-cu128/bin/python}"
BATCH_CANDIDATES="${BATCH_CANDIDATES:-4 6 8 10 12}"
PROBE_STEPS="${PROBE_STEPS:-3}"
AMP_DTYPE="${AMP_DTYPE:-bfloat16}"
DEVICE_TAG="${DEVICE_TAG:-rtx4000ada}"
RESULT_ROOT="${RESULT_ROOT:-reports/${DEVICE_TAG}_capacity_probe_20260714}"

cd "$ROOT"
mkdir -p "$RESULT_ROOT"
printf \
  '{"scope":"capacity_probe_only","target_version":"market_transition_v6_systemic_impact_20260714","amp_dtype":"%s","probe_steps":%s,"live_orders_allowed":false}\n' \
  "$AMP_DTYPE" "$PROBE_STEPS" \
  > "$RESULT_ROOT/experiment_contract.json"

for batch_size in $BATCH_CANDIDATES; do
  run_name="${DEVICE_TAG}_v6_capacity_b${batch_size}_20260714"
  status_file="$RESULT_ROOT/batch_${batch_size}.status"
  if [[ -f "$status_file" ]]; then
    continue
  fi
  set +e
  ROOT="$ROOT" \
  PYTHON="$PYTHON_BIN" \
  RUN_NAME="$run_name" \
  TARGET_VERSION="market_transition_v6_systemic_impact_20260714" \
  TRAIN_BATCH_SIZE="$batch_size" \
  SNAPSHOT_WORKERS="$batch_size" \
  AMP_DTYPE="$AMP_DTYPE" \
  MAX_TRAIN_STEPS="$PROBE_STEPS" \
  TRAINING_ONLY=1 \
  bash scripts/run_broad_transition_jepa_v4_a5000.sh
  exit_status=$?
  set -e
  printf '%s\n' "$exit_status" > "$status_file"
  if [[ "$exit_status" -ne 0 ]]; then
    break
  fi
done

touch "$RESULT_ROOT/PROBE_COMPLETE"
