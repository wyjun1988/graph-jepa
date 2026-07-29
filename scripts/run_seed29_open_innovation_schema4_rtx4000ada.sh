#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/stock-v2-cu128/bin/python}"
PERSISTENT_ROOT="${PERSISTENT_ROOT:-/workspace/stock-v2-seed-stability}"
TRAIN_RUN="${TRAIN_RUN:-broad_transition_jepa_v6_lifecycle500_v4_seed29_diagnostic_rtx4000ada_20260715}"
STABILITY_NAME="${STABILITY_NAME:-seed29_stability_v1_20260715}"
RUN_NAME="${RUN_NAME:-seed29_open_innovation_schema4_v1_20260715}"
CONTRACT_PATH="${CONTRACT_PATH:-$PERSISTENT_ROOT/contracts/seed29-open-innovation-schema4-v1-20260715.json}"
CONTRACT_SHA256="a2e2804b879f09eb0b449024582063205be93c18414e20ebe71c8a41c7c0de09"
TRAIN_ROOT="$PERSISTENT_ROOT/reports/$TRAIN_RUN"
STABILITY_ROOT="$PERSISTENT_ROOT/reports/$STABILITY_NAME"
OUTPUT_ROOT="$PERSISTENT_ROOT/reports/$RUN_NAME"

sha256_file() {
  sha256sum "$1" | awk '{print $1}'
}

cd "$ROOT"
test -f "$TRAIN_ROOT/PIPELINE_COMPLETE"
test -f "$TRAIN_ROOT/summary.json"
test -f "$STABILITY_ROOT/EVALUATION_COMPLETE"
test -f "$STABILITY_ROOT/final_gate/summary.json"
test -f "$CONTRACT_PATH"
if [[ "$(sha256_file "$CONTRACT_PATH")" != "$CONTRACT_SHA256" ]]; then
  printf '%s\n' "open-innovation contract SHA-256 mismatch" >&2
  exit 5
fi

"$PYTHON_BIN" - "$CONTRACT_PATH" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

contract = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
for raw_path, expected in contract["source_sha256"].items():
    path = Path(raw_path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    actual = digest.hexdigest()
    if actual != expected:
        raise SystemExit(
            f"source SHA-256 mismatch for {path}: expected {expected} got {actual}"
        )
PY

mapfile -t SETTINGS < <(
  "$PYTHON_BIN" - "$CONTRACT_PATH" <<'PY'
import json
import sys

benchmark = json.load(open(sys.argv[1], encoding="utf-8"))["benchmark"]
print(benchmark["validation_days"])
print(",".join(str(value) for value in benchmark["split_horizons"]))
print(",".join(benchmark["model_configs"]))
print(",".join(str(value) for value in benchmark["placebo_seeds"]))
print(benchmark["jepa_feature_mode"])
print(benchmark["downstream_seed"])
print(benchmark["num_threads"])
print(benchmark["cache_batch_size"])
print(benchmark["cache_edge_workers"])
print(benchmark["cache_device"])
print(benchmark["ohlcv_cache_dir"])
print(benchmark["external_cache_dir"])
PY
)
if [[ "${#SETTINGS[@]}" -ne 12 ]]; then
  printf '%s\n' "open-innovation contract settings are incomplete" >&2
  exit 4
fi
VALIDATION_DAYS="${SETTINGS[0]}"
SPLIT_HORIZONS="${SETTINGS[1]}"
MODEL_CONFIGS="${SETTINGS[2]}"
PLACEBO_SEEDS="${SETTINGS[3]}"
JEPA_FEATURE_MODE="${SETTINGS[4]}"
DOWNSTREAM_SEED="${SETTINGS[5]}"
NUM_THREADS="${SETTINGS[6]}"
CACHE_BATCH_SIZE="${SETTINGS[7]}"
CACHE_EDGE_WORKERS="${SETTINGS[8]}"
CACHE_DEVICE="${SETTINGS[9]}"
OHLCV_CACHE_DIR="${SETTINGS[10]}"
EXTERNAL_CACHE_DIR="${SETTINGS[11]}"

mapfile -t MODEL_DIRS < <(
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); [print(row["model_dir"]) for row in p["folds"]]' \
    "$TRAIN_ROOT/summary.json"
)
if [[ "${#MODEL_DIRS[@]}" -ne 5 ]]; then
  printf 'expected five trained folds, found %s\n' "${#MODEL_DIRS[@]}" >&2
  exit 4
fi

mkdir -p "$OUTPUT_ROOT/folds" "$OUTPUT_ROOT/caches" "$OUTPUT_ROOT/aggregate"
touch "$OUTPUT_ROOT/DIAGNOSTIC_ONLY"
exec 9>"$OUTPUT_ROOT/.replication.lock"
if ! flock -n 9; then
  printf '%s\n' "seed29 open-innovation replication is already running"
  exit 0
fi

FOLD_ARGS=()
for index in "${!MODEL_DIRS[@]}"; do
  fold_number=$((index + 1))
  fold="fold${fold_number}"
  model_dir="${MODEL_DIRS[$index]}"
  cache_dir="$OUTPUT_ROOT/caches/$fold"
  report_dir="$OUTPUT_ROOT/folds/$fold"
  test -f "$model_dir/graph_jepa_real.pt"

  if [[ ! -f "$cache_dir/CACHE_COMPLETE" ]]; then
    "$PYTHON_BIN" scripts/build_h1_state_forecast_cache.py \
      --model-dir "$model_dir" \
      --output-dir "$cache_dir" \
      --split-horizons "$SPLIT_HORIZONS" \
      --validation-days "$VALIDATION_DAYS" \
      --batch-size "$CACHE_BATCH_SIZE" \
      --edge-cache-workers "$CACHE_EDGE_WORKERS" \
      --device "$CACHE_DEVICE" \
      --cache-dir "$OHLCV_CACHE_DIR" \
      --external-cache-dir "$EXTERNAL_CACHE_DIR" \
      >"$OUTPUT_ROOT/folds/${fold}_cache.log" 2>&1
  fi
  test -f "$cache_dir/CACHE_COMPLETE"

  if [[ ! -f "$report_dir/EXPERIMENT_COMPLETE" ]]; then
    "$PYTHON_BIN" scripts/benchmark_open_innovation_nowcast.py \
      --model-dir "$model_dir" \
      --forecast-cache-dir "$cache_dir" \
      --output-dir "$report_dir" \
      --validation-days "$VALIDATION_DAYS" \
      --split-horizons "$SPLIT_HORIZONS" \
      --configs "$MODEL_CONFIGS" \
      --placebo-seeds "$PLACEBO_SEEDS" \
      --jepa-feature-mode "$JEPA_FEATURE_MODE" \
      --seed "$DOWNSTREAM_SEED" \
      --num-threads "$NUM_THREADS" \
      --cache-dir "$OHLCV_CACHE_DIR" \
      --external-cache-dir "$EXTERNAL_CACHE_DIR" \
      >"$OUTPUT_ROOT/folds/${fold}_benchmark.log" 2>&1
  fi
  test -f "$report_dir/EXPERIMENT_COMPLETE"
  FOLD_ARGS+=(--fold "$fold=$report_dir")
done

if [[ ! -f "$OUTPUT_ROOT/aggregate/AGGREGATION_COMPLETE" ]]; then
  "$PYTHON_BIN" scripts/aggregate_open_innovation_multifold.py \
    "${FOLD_ARGS[@]}" \
    --output-dir "$OUTPUT_ROOT/aggregate" \
    >"$OUTPUT_ROOT/aggregate.log" 2>&1
fi
test -f "$OUTPUT_ROOT/aggregate/AGGREGATION_COMPLETE"

"$PYTHON_BIN" - "$TRAIN_ROOT/summary.json" "$CONTRACT_PATH" "$OUTPUT_ROOT/run_manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

training_summary = Path(sys.argv[1])
contract = Path(sys.argv[2])
output = Path(sys.argv[3])
training = json.loads(training_summary.read_text(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

payload = {
    "schema_version": 1,
    "role": "seed29_open_innovation_replication_run_manifest",
    "training_summary": {"path": str(training_summary), "sha256": sha256(training_summary)},
    "contract": {"path": str(contract), "sha256": sha256(contract)},
    "runner": {
        "path": "scripts/run_seed29_open_innovation_schema4_rtx4000ada.sh",
        "sha256": sha256(Path("scripts/run_seed29_open_innovation_schema4_rtx4000ada.sh")),
    },
    "checkpoints": [
        {
            "fold": f"fold{index + 1}",
            "path": str(Path(row["model_dir"]) / "graph_jepa_real.pt"),
            "sha256": sha256(Path(row["model_dir"]) / "graph_jepa_real.pt"),
        }
        for index, row in enumerate(training["folds"])
    ],
    "promotion_eligible": False,
    "live_orders_allowed": False,
}
output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
PY

VERIFY_ARGS=()
for index in "${!MODEL_DIRS[@]}"; do
  fold="fold$((index + 1))"
  VERIFY_ARGS+=(--fold "$fold=$OUTPUT_ROOT/folds/$fold/summary.json")
done
set +e
"$PYTHON_BIN" scripts/verify_seed29_open_innovation_replication.py \
  --contract "$CONTRACT_PATH" \
  --stability-summary "$STABILITY_ROOT/final_gate/summary.json" \
  "${VERIFY_ARGS[@]}" \
  --aggregate-summary "$OUTPUT_ROOT/aggregate/summary.json" \
  --output-dir "$OUTPUT_ROOT/final_gate" \
  >"$OUTPUT_ROOT/final_gate.log" 2>&1
gate_status=$?
set -e
if [[ "$gate_status" -ne 0 && "$gate_status" -ne 2 ]]; then
  printf 'open-innovation replication verifier failed with status %s\n' \
    "$gate_status" >&2
  exit "$gate_status"
fi

touch "$OUTPUT_ROOT/REPLICATION_COMPLETE"
printf 'seed29 open-innovation replication complete gate_exit=%s\n' "$gate_status"
