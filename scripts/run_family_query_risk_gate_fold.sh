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
  echo "refusing to overwrite a risk-gate fold: $OUTPUT_DIR" >&2
  exit 3
fi
if [[ -e "$TARGET_AUDIT_DIR" && ! -f "$TARGET_AUDIT_DIR/summary.json" ]]; then
  echo "refusing to reuse an incomplete target audit: $TARGET_AUDIT_DIR" >&2
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


contract_path, fold_name, model_dir = Path(sys.argv[1]), sys.argv[2], Path(sys.argv[3])
contract = json.loads(contract_path.read_text(encoding="utf-8"))
if fold_name not in contract["confirmation_folds"]:
    raise SystemExit(f"fold is not preregistered for confirmation: {fold_name}")
expected = contract["folds"][fold_name]
if model_dir.name != expected["model_dir_name"]:
    raise SystemExit("model directory differs from preregistration")
checkpoint = model_dir / "graph_jepa_real.pt"
if sha256(checkpoint) != expected["checkpoint_sha256"]:
    raise SystemExit("checkpoint SHA differs from preregistration")
for relative, expected_sha in contract["source_pins"].items():
    path = Path(relative)
    if not path.is_file() or sha256(path) != expected_sha:
        raise SystemExit(f"source pin mismatch: {relative}")
PY

mkdir -p "$OUTPUT_DIR"

if [[ ! -f "$TARGET_AUDIT_DIR/summary.json" ]]; then
  mkdir -p "$TARGET_AUDIT_DIR"
  "$PYTHON_BIN" scripts/audit_market_transition_targets.py \
    --model-dir "$MODEL_DIR" \
    --output-dir "$TARGET_AUDIT_DIR" \
    --horizons 1,2,3,5,10 \
    --validation-days 126 \
    --component-scale-quantile 0.90 \
    --family-event-quantile 0.95 \
    --top-events 30 \
    --cache-dir data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv \
    --external-cache-dir data/external_cache
fi

"$PYTHON_BIN" scripts/benchmark_market_transition_head.py \
  --model-dir "$MODEL_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --horizons 1,2,3,5,10 \
  --validation-days 126 \
  --epochs 80 \
  --patience 10 \
  --projection-dim 128 \
  --family-query-pooling \
  --stock-quantile-pooling \
  --hidden-dim 256 \
  --layers 2 \
  --heads 8 \
  --dropout 0.10 \
  --learning-rate 0.0003 \
  --weight-decay 0.001 \
  --batch-size 16 \
  --eval-batch-size 32 \
  --edge-cache-workers 16 \
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
artifact_paths = {
    "summary": output_dir / "summary.json",
    "head": output_dir / "market_transition_head.pt",
    "daily_test": output_dir / "daily_test.csv",
    "major_summary": output_dir / "major_trajectory" / "summary.json",
    "major_daily": output_dir / "major_trajectory" / "daily_major_trajectory.csv",
    "target_summary": target_dir / "summary.json",
    "target_daily": target_dir / "daily_market_transition_targets.csv",
}
manifest = {
    "schema_version": 1,
    "role": "family_query_magnitude_risk_gate_fold_manifest",
    "fold": fold_name,
    "contract_sha256": sha256(contract_path),
    "model_dir_name": model_dir.name,
    "checkpoint_sha256": sha256(model_dir / "graph_jepa_real.pt"),
    "training_recipe": contract["training_recipe"],
    "source_pins": {
        relative: sha256(Path(relative)) for relative in contract["source_pins"]
    },
    "artifacts": {
        name: {"path": str(path), "sha256": sha256(path)}
        for name, path in artifact_paths.items()
    },
    "promotion_eligible": False,
    "live_orders_allowed": False,
}
(output_dir / "run_manifest.json").write_text(
    json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
)
PY

touch "$OUTPUT_DIR/FOLD_COMPLETE"
