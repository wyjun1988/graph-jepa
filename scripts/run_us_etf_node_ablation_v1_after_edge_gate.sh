#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-/root/stock-v2-etf-ablation-v1}"
PYTHON_BIN="${PYTHON:-/root/venvs/stock-v2-cu128/bin/python}"
PREFLIGHT_ROOT="reports/us_etf_node_ablation_v1_baseline_exact_preflight_20260716"
FROZEN="$PREFLIGHT_ROOT/frozen_edge_manifests.json"
EDGE_ROOT="reports/us_etf_node_ablation_v1_edge_connectivity_20260716"

cd "$ROOT"
if [[ ! -f "$PREFLIGHT_ROOT/PREFLIGHT_COMPLETE" || ! -f "$FROZEN" ]]; then
  printf '%s\n' "frozen ETF preflight is incomplete" >&2
  exit 3
fi
if [[ ! -f "$EDGE_ROOT/COMPLETE" ]]; then
  printf '%s\n' "ETF edge connectivity audit is incomplete" >&2
  exit 4
fi

"$PYTHON_BIN" - "$FROZEN" "$EDGE_ROOT/fold3.json" "$EDGE_ROOT/fold4.json" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys


frozen = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
audit_paths = [Path(value) for value in sys.argv[2:]]
if not audit_paths or len(frozen.get("fold_manifests", [])) != len(audit_paths):
    raise ValueError("edge gate fold count mismatch")
for fold, path in zip(frozen["fold_manifests"], audit_paths):
    audit = json.loads(path.read_text(encoding="utf-8"))
    if audit.get("role") != "us_etf_training_edge_connectivity_audit":
        raise ValueError(f"invalid edge audit role: {path}")
    if audit.get("node_counts") != {"stock": 500, "external": 13, "etf": 34}:
        raise ValueError(f"edge audit node geometry changed: {path}")
    if int(audit.get("steps", -1)) != int(fold["training_edge_steps"]):
        raise ValueError(f"edge audit step count changed: {path}")
    for direction in ("etf_to_stock", "stock_to_etf"):
        record = audit.get(direction) or {}
        if int(record.get("edges", 0)) <= 0:
            raise ValueError(f"no {direction} edges: {path}")
        if int(record.get("steps_with_edges", -1)) != int(audit["steps"]):
            raise ValueError(f"{direction} is absent in some training steps: {path}")
    provenance = audit.get("source_provenance") or {}
    if provenance.get("run_real_backtest_sha256") != frozen["source_pins"][
        "scripts/run_real_backtest.py"
    ]:
        raise ValueError(f"edge audit used a different training source: {path}")
    if provenance.get("real_features_sha256") != frozen["source_pins"][
        "stock_v2/real_features.py"
    ]:
        raise ValueError(f"edge audit used a different edge builder: {path}")
    if audit.get("live_orders_allowed") is not False:
        raise ValueError(f"unsafe edge audit: {path}")
    if int(audit.get("broker_order_calls_executed", -1)) != 0:
        raise ValueError(f"edge audit executed a broker order call: {path}")
print(
    json.dumps(
        {
            "status": "pass",
            "folds": len(audit_paths),
            "live_orders_allowed": False,
            "broker_order_calls_executed": 0,
        },
        sort_keys=True,
    )
)
PY

exec env ROOT="$ROOT" PYTHON="$PYTHON_BIN" MODE=train \
  bash scripts/run_us_etf_node_ablation_v1_rtx4000ada.sh
