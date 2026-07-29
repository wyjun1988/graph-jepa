#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 5 ]]; then
  echo "usage: $0 CONTRACT FOLD MODEL_DIR OUTPUT_DIR TARGET_AUDIT_DIR" >&2
  exit 2
fi

CONTRACT=$1
FOLD_NAME=$2
MODEL_DIR=$3
OUTPUT_DIR=$4
TARGET_AUDIT_DIR=$5
PYTHON_BIN=${PYTHON_BIN:-python}

if [[ -e "$OUTPUT_DIR" ]]; then
  echo "refusing to overwrite a direct risk comparator: $OUTPUT_DIR" >&2
  exit 3
fi
if [[ ! -f "$TARGET_AUDIT_DIR/summary.json" ]]; then
  echo "target audit is missing or incomplete: $TARGET_AUDIT_DIR" >&2
  exit 4
fi

"$PYTHON_BIN" - "$CONTRACT" "$FOLD_NAME" "$MODEL_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


contract_path = Path(sys.argv[1])
fold_name = sys.argv[2]
model_dir = Path(sys.argv[3])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if contract.get("role") != "direct_current_state_magnitude_risk_comparator":
    raise SystemExit("invalid direct comparator contract role")
parent_spec = contract["parent_risk_contract"]
parent_path = Path(parent_spec["path"])
if sha256(parent_path) != parent_spec["sha256"]:
    raise SystemExit("parent risk contract hash mismatch")
parent = json.loads(parent_path.read_text(encoding="utf-8"))
if parent.get("role") != "retrospective_family_query_magnitude_risk_gate":
    raise SystemExit("invalid parent risk contract role")
if fold_name not in contract["confirmation_folds"]:
    raise SystemExit(f"fold is not preregistered: {fold_name}")
expected = parent["folds"][fold_name]
if model_dir.name != expected["model_dir_name"]:
    raise SystemExit("model directory differs from preregistration")
if sha256(model_dir / "graph_jepa_real.pt") != expected["checkpoint_sha256"]:
    raise SystemExit("checkpoint hash differs from preregistration")
for relative, expected_sha in {
    **parent["source_pins"],
    **contract["direct_source_pins"],
}.items():
    path = Path(relative)
    if not path.is_file() or sha256(path) != expected_sha:
        raise SystemExit(f"source pin mismatch: {relative}")
PY

mkdir -p "$OUTPUT_DIR"

"$PYTHON_BIN" scripts/benchmark_direct_market_transition_head.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --horizons 1,2,3,5,10 \
  --validation-days 126 \
  --epochs 80 \
  --patience 10 \
  --hidden-dim 256 \
  --layers 2 \
  --heads 8 \
  --dropout 0.10 \
  --learning-rate 0.0003 \
  --weight-decay 0.001 \
  --batch-size 128 \
  --eval-batch-size 512 \
  --device cuda \
  --seed 2701 \
  --cache-dir data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv \
  --external-cache-dir data/external_cache

"$PYTHON_BIN" scripts/evaluate_major_market_trajectory.py \
  --target-audit-root "$TARGET_AUDIT_DIR" \
  --prediction-root "$OUTPUT_DIR" \
  --output-dir "$OUTPUT_DIR/major_trajectory" \
  --major-event-quantile 0.90

"$PYTHON_BIN" - "$CONTRACT" "$FOLD_NAME" "$MODEL_DIR" "$OUTPUT_DIR" "$TARGET_AUDIT_DIR" <<'PY'
import hashlib
import json
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


contract_path = Path(sys.argv[1])
fold_name = sys.argv[2]
model_dir = Path(sys.argv[3])
output_dir = Path(sys.argv[4])
target_dir = Path(sys.argv[5])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
parent_path = Path(contract["parent_risk_contract"]["path"])
parent = json.loads(parent_path.read_text(encoding="utf-8"))
artifact_paths = {
    "summary": output_dir / "summary.json",
    "head": output_dir / "direct_market_transition_head.pt",
    "daily_validation": output_dir / "daily_validation.csv",
    "daily_test": output_dir / "daily_test.csv",
    "major_summary": output_dir / "major_trajectory" / "summary.json",
    "major_daily": output_dir / "major_trajectory" / "daily_major_trajectory.csv",
    "target_summary": target_dir / "summary.json",
    "target_daily": target_dir / "daily_market_transition_targets.csv",
}
manifest = {
    "schema_version": 1,
    "role": "direct_current_state_magnitude_risk_comparator_fold_manifest",
    "fold": fold_name,
    "contract_sha256": sha256(contract_path),
    "parent_risk_contract_sha256": sha256(parent_path),
    "model_dir_name": model_dir.name,
    "checkpoint_sha256": sha256(model_dir / "graph_jepa_real.pt"),
    "training_recipe": contract["training_recipe"],
    "source_pins": {
        relative: sha256(Path(relative))
        for relative in {**parent["source_pins"], **contract["direct_source_pins"]}
    },
    "artifacts": {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in artifact_paths.items()
    },
    "test_used_for_selection": False,
    "promotion_eligible": False,
    "live_orders_allowed": False,
}
(output_dir / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

touch "$OUTPUT_DIR/COMPARATOR_COMPLETE"
