#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-runtime}"
PYTHON_BIN="${PYTHON_BIN:-/root/venvs/stock-v2-cu128/bin/python}"
TRAIN_RUN="${TRAIN_RUN:-broad_transition_jepa_v6_lifecycle500_v4_seed29_diagnostic_rtx4000ada_20260715}"
REPORT_BASE="${REPORT_BASE:-/workspace/stock-v2-seed-stability/reports}"
REFERENCE_ROOT="${REFERENCE_ROOT:-/workspace/stock-v2-seed-stability/reference_seed17}"
EVAL_NAME="${EVAL_NAME:-seed29_stability_v1_20260715}"
TRAIN_ROOT="$REPORT_BASE/$TRAIN_RUN"
EVAL_ROOT="$REPORT_BASE/$EVAL_NAME"
CONTRACT_SHA256="7dca8518267e5b5f9ca5985ea12076d3086e742825085079ae5340858c07e785"
REFERENCE_SHA256=(
  3e2a35c6934b6e487ce821ac849efd5becb0c708b2ced47b56ba2bb25fb536e9
  c39c7bac23bde4491cdd657ef77dd3c58580e6888d5961cb7f2d4d0dfff7ec8a
  dfb31209e1ef81207de955204a70d4cefefc8d92975a44056dbc745b420a6a86
  f3dd1558eb2f3e4862887b797834d6dcf0e9dddc4f508e552e907af7c76c0020
  e6654d8fd8182a1c60fb7e2796e894a8dd7c754ff41af7e919a082d53436929d
)
DIRECT_SHA256=(
  a8737083eeac9a59e54abe5eba2b3af461393d6ae99059c3d76613b0ad3c462f
  44f463b97c97ba087c1448844fc5a6ea44350e8e9b90375c2bde5823804f3619
  f3a32e1c6dfa976f0f1fb4847fbd4dc0a3010e45cef00cd34ab7d33411fd29c0
  8d6e69f060b484056b4596303000b3675c39ce70c48339d9f7cbf05a166438b0
  39cbfd7f5b1b79c4a0f979199d67a37860f1bf22364ab504f3a39fbe253b00ca
)

verify_sha256() {
  local path="$1"
  local expected="$2"
  local actual
  actual="$(sha256sum "$path" | awk '{print $1}')"
  if [[ "$actual" != "$expected" ]]; then
    printf 'sha256 mismatch for %s: expected %s got %s\n' \
      "$path" "$expected" "$actual" >&2
    exit 5
  fi
}

cd "$ROOT"
verify_sha256 scripts/evaluate_node_prediction.py \
  fc20b16eba0a5017c01d828b0219520c48bde3da9209e24b19ae2fc1fd963523
verify_sha256 scripts/compare_direct_state_mlp.py \
  a3d6df4c19674c6ef84e175aec619737bc1a1916ea5317ddf837d9c8edfda7ea
verify_sha256 stock_v2/downstream_probes.py \
  12b4041edbff50e848565a06a5f247907291831c47813df0a09cdc05b48370e8
verify_sha256 stock_v2/seed_stability.py \
  bf7aa63a9c617a04439e3d2d0b6bebe9862b2131c335d3b3a7526b05d9600b56
verify_sha256 scripts/evaluate_seed_stability.py \
  9177adb3b7efc3fb71f9da4bd16b84a35bbeb5ee4160a5184d2239111ac66f55
test -f "$TRAIN_ROOT/PIPELINE_COMPLETE"
test -f "$TRAIN_ROOT/summary.json"
test -f "$REFERENCE_ROOT/contract.json"
verify_sha256 "$REFERENCE_ROOT/contract.json" "$CONTRACT_SHA256"
mkdir -p "$EVAL_ROOT"
touch "$EVAL_ROOT/DIAGNOSTIC_ONLY"

exec 9>"$EVAL_ROOT/.evaluation.lock"
if ! flock -n 9; then
  printf '%s\n' "seed stability evaluation is already running"
  exit 0
fi

mapfile -t MODEL_DIRS < <(
  "$PYTHON_BIN" -c \
    'import json,sys; p=json.load(open(sys.argv[1])); [print(row["model_dir"]) for row in p["folds"]]' \
    "$TRAIN_ROOT/summary.json"
)
if [[ "${#MODEL_DIRS[@]}" -ne 5 ]]; then
  printf 'expected five trained folds, found %s\n' "${#MODEL_DIRS[@]}" >&2
  exit 4
fi

REFERENCE_ARGS=()
CANDIDATE_ARGS=()
DIRECT_ARGS=()
for index in "${!MODEL_DIRS[@]}"; do
  fold_number=$((index + 1))
  fold="fold${fold_number}"
  model_dir="${MODEL_DIRS[$index]}"
  model_name="$(basename "$model_dir")"
  fold_root="$EVAL_ROOT/$fold"
  node_root="$fold_root/node_eval"
  candidate_daily="$node_root/$model_name/future_rollout.csv"
  direct_daily="$REFERENCE_ROOT/direct_${fold}.csv"
  direct_comparison="$fold_root/direct_comparison/comparison.json"
  reference_daily="$REFERENCE_ROOT/${fold}.csv"
  test -f "$model_dir/graph_jepa_real.pt"
  test -f "$reference_daily"
  test -f "$direct_daily"
  verify_sha256 "$reference_daily" "${REFERENCE_SHA256[$index]}"
  verify_sha256 "$direct_daily" "${DIRECT_SHA256[$index]}"
  mkdir -p "$fold_root"

  if [[ ! -f "$fold_root/NODE_EVAL_COMPLETE" ]]; then
    "$PYTHON_BIN" scripts/evaluate_node_prediction.py \
      --model-dir "$model_dir" \
      --output-dir "$node_root" \
      --horizons 1,2,3,5,10 \
      --state-target-scope checkpoint_temporal \
      --max-steps 0 \
      --save-return-forecasts \
      --edge-cache-workers 16 \
      --device cuda \
      --seed 29 \
      >"$fold_root/node_eval.log" 2>&1
    test -f "$candidate_daily"
    touch "$fold_root/NODE_EVAL_COMPLETE"
  fi

  if [[ ! -f "$direct_comparison" ]]; then
    "$PYTHON_BIN" scripts/compare_direct_state_mlp.py \
      --direct-daily "$direct_daily" \
      --jepa-daily "$candidate_daily" \
      --output-dir "$fold_root/direct_comparison" \
      --required-state-target-scope checkpoint_temporal \
      >"$fold_root/direct_comparison.log" 2>&1
  fi

  REFERENCE_ARGS+=(--reference-fold "$fold=$reference_daily")
  CANDIDATE_ARGS+=(--candidate-fold "$fold=$candidate_daily")
  DIRECT_ARGS+=(--direct-comparison "$fold=$direct_comparison")
done

"$PYTHON_BIN" - "$TRAIN_ROOT/summary.json" "$EVAL_ROOT/checkpoint_manifest.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

training_summary = Path(sys.argv[1])
output = Path(sys.argv[2])
payload = json.loads(training_summary.read_text(encoding="utf-8"))

def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()

checkpoints = []
for row in payload["folds"]:
    checkpoint = Path(row["model_dir"]) / "graph_jepa_real.pt"
    checksums = sha256(checkpoint)
    checkpoints.append({"fold": row["fold"], "path": str(checkpoint), "sha256": checksums})
source_paths = [
    Path("scripts/evaluate_node_prediction.py"),
    Path("scripts/compare_direct_state_mlp.py"),
    Path("stock_v2/downstream_probes.py"),
    Path("stock_v2/seed_stability.py"),
    Path("scripts/evaluate_seed_stability.py"),
    Path("scripts/run_seed29_stability_eval_rtx4000ada.sh"),
]
manifest = {
    "schema_version": 1,
    "role": "diagnostic_seed_checkpoint_manifest",
    "training_summary": str(training_summary),
    "training_summary_sha256": sha256(training_summary),
    "checkpoints": checkpoints,
    "evaluation_sources": [
        {"path": str(path), "sha256": sha256(path)} for path in source_paths
    ],
    "promotion_eligible": False,
    "live_orders_allowed": False,
}
output.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
PY

set +e
"$PYTHON_BIN" scripts/evaluate_seed_stability.py \
  --contract "$REFERENCE_ROOT/contract.json" \
  "${REFERENCE_ARGS[@]}" \
  "${CANDIDATE_ARGS[@]}" \
  "${DIRECT_ARGS[@]}" \
  --output-dir "$EVAL_ROOT/final_gate" \
  >"$EVAL_ROOT/final_gate.log" 2>&1
gate_status=$?
set -e
if [[ "$gate_status" -ne 0 && "$gate_status" -ne 2 ]]; then
  printf 'seed stability evaluator failed with status %s\n' "$gate_status" >&2
  exit "$gate_status"
fi

touch "$EVAL_ROOT/EVALUATION_COMPLETE"
printf 'seed29 stability diagnostic complete gate_exit=%s\n' "$gate_status"
