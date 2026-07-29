from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


SOURCE_PATHS = (
    "scripts/run_real_backtest.py",
    "scripts/run_walk_forward_node_eval.py",
    "scripts/evaluate_node_prediction.py",
    "scripts/compare_us_etf_node_ablation.py",
    "scripts/freeze_us_etf_node_ablation.py",
    "scripts/run_us_etf_node_ablation_v1_rtx4000ada.sh",
    "stock_v2/__init__.py",
    "stock_v2/backtest.py",
    "stock_v2/data_contract.py",
    "stock_v2/event_features.py",
    "stock_v2/external_etf_nodes.py",
    "stock_v2/external_factors.py",
    "stock_v2/fundamental_features.py",
    "stock_v2/graph_jepa.py",
    "stock_v2/kiwoom_investor.py",
    "stock_v2/lifecycle_ohlcv.py",
    "stock_v2/market_data.py",
    "stock_v2/market_transition.py",
    "stock_v2/market_transition_auxiliary.py",
    "stock_v2/market_transition_head.py",
    "stock_v2/ops/__init__.py",
    "stock_v2/ops/brokers.py",
    "stock_v2/ops/config.py",
    "stock_v2/ops/store.py",
    "stock_v2/ops/types.py",
    "stock_v2/real_features.py",
    "stock_v2/static_edges.py",
    "stock_v2/systemic_head.py",
    "stock_v2/systemic_transition.py",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _validate_contract(path: Path) -> dict[str, Any]:
    contract = load_json(path)
    if contract.get("role") != "predeclared_us_etf_external_node_ablation":
        raise ValueError("invalid US ETF ablation contract role")
    if len(contract.get("folds", [])) != 2:
        raise ValueError("US ETF screening contract must contain exactly two folds")
    for key in (
        "promotion_eligible",
        "deployment_eligible",
        "counts_as_primary_forward_evidence",
        "live_orders_allowed",
        "broker_order_calls_allowed",
    ):
        if contract.get(key) is not False:
            raise ValueError(f"unsafe contract field: {key}")
    source = contract["source_release"]
    for path_key, hash_key in (
        ("sensor_contract", "sensor_contract_sha256"),
        ("quality_contract", "quality_contract_sha256"),
    ):
        source_path = ROOT / source[path_key]
        if file_sha256(source_path) != source[hash_key]:
            raise ValueError(f"source contract hash changed: {source_path}")
    panel_root = ROOT / source["panel_root"]
    for name, hash_key in (
        ("contract.json", "panel_contract_sha256"),
        ("panel.parquet", "panel_sha256"),
        ("summary.json", "panel_summary_sha256"),
    ):
        if file_sha256(panel_root / name) != source[hash_key]:
            raise ValueError(f"ETF panel artifact hash changed: {name}")
    parity = contract["source_parity"]
    patched = parity["etf_patched_sha256"]
    for relative, expected in parity["baseline_original_sha256"].items():
        if relative in patched:
            continue
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"baseline source parity changed: {relative}")
    for relative, expected in patched.items():
        if file_sha256(ROOT / relative) != expected:
            raise ValueError(f"ETF-only source patch changed: {relative}")
    return contract


def _fold_root(reports_root: Path, run_name: str, ordinal: int, fold: dict[str, Any]) -> Path:
    train_token = str(fold["train_end"]).replace("-", "")
    eval_token = str(fold["eval_end"]).replace("-", "")
    return reports_root / f"{run_name}_fold{ordinal}_{train_token}_to_{eval_token}"


def collect_preflight(
    contract_path: Path,
    reports_root: Path,
    run_name: str,
) -> dict[str, Any]:
    contract = _validate_contract(contract_path)
    source_pins = {path: file_sha256(ROOT / path) for path in SOURCE_PATHS}
    rows: list[dict[str, Any]] = []
    for ordinal, fold in enumerate(contract["folds"], start=1):
        root = _fold_root(reports_root, run_name, ordinal, fold)
        data_path = root / "training_data_manifest.json"
        edge_path = root / "training_edge_manifest.json"
        diagnostics_path = root / "training_data_diagnostics.json"
        audit_path = root / "external_etf_node_audit.json"
        data = load_json(data_path)
        edge = load_json(edge_path)
        diagnostics = load_json(diagnostics_path)
        audit = load_json(audit_path)

        if data.get("sha256") != fold["candidate_training_data_sha256"]:
            raise ValueError(f"{fold['label']} candidate data manifest changed")
        if file_sha256(data_path) != fold["candidate_training_manifest_file_sha256"]:
            raise ValueError(f"{fold['label']} candidate data manifest file changed")
        if int(data.get("stock_node_count", -1)) != 500:
            raise ValueError(f"{fold['label']} stock target count changed")
        if len(data.get("node_tickers", [])) != 547:
            raise ValueError(f"{fold['label']} candidate node count changed")
        if len(data.get("feature_names", [])) != 153:
            raise ValueError(f"{fold['label']} candidate feature count changed")
        if int(diagnostics.get("nodes", -1)) != 547 or int(
            diagnostics.get("features", -1)
        ) != 153:
            raise ValueError(f"{fold['label']} diagnostics geometry changed")
        if audit.get("live_orders_allowed") is not False:
            raise ValueError(f"{fold['label']} ETF audit does not prohibit orders")
        if int(audit.get("nodes", -1)) != 34:
            raise ValueError(f"{fold['label']} ETF node count changed")
        if len(str(edge.get("sha256") or "")) != 64:
            raise ValueError(f"{fold['label']} edge manifest has no SHA-256")

        rows.append(
            {
                "label": fold["label"],
                "ordinal": ordinal,
                "fold_root": str(root),
                "train_end": fold["train_end"],
                "eval_end": fold["eval_end"],
                "training_data_manifest_sha256": data["sha256"],
                "training_data_manifest_file_sha256": file_sha256(data_path),
                "training_edge_manifest_sha256": edge["sha256"],
                "training_edge_manifest_file_sha256": file_sha256(edge_path),
                "training_edge_steps": int(edge["step_count"]),
                "training_edge_count": int(edge["total_edges"]),
                "external_etf_audit_file_sha256": file_sha256(audit_path),
                "external_etf_visible_events": int(audit["source_events_visible"]),
                "external_etf_holiday_bundles": int(audit["bundled_holiday_events"]),
            }
        )

    return {
        "schema_version": 1,
        "role": "frozen_us_etf_node_ablation_preflight",
        "contract": str(contract_path),
        "contract_file_sha256": file_sha256(contract_path),
        "run_name": run_name,
        "fold_manifests": rows,
        "source_pins": source_pins,
        "preflight_generated_model_predictions": False,
        "test_used_for_selection": False,
        "candidate_results_observed": False,
        "promotion_eligible": False,
        "deployment_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the two-fold US ETF node ablation preflight."
    )
    parser.add_argument("--contract", required=True)
    parser.add_argument("--reports-root", required=True)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()

    contract_path = Path(args.contract)
    reports_root = Path(args.reports_root)
    output = Path(args.output)
    current = collect_preflight(contract_path, reports_root, args.run_name)
    if args.verify:
        frozen = load_json(output)
        if current != frozen:
            raise ValueError("US ETF preflight differs from the frozen contract")
        print(
            json.dumps(
                {
                    "status": "pass",
                    "folds": len(current["fold_manifests"]),
                    "contract_file_sha256": current["contract_file_sha256"],
                    "live_orders_allowed": False,
                },
                sort_keys=True,
            )
        )
        return

    if output.exists():
        raise FileExistsError(f"refusing to replace frozen preflight: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(current, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "frozen",
                "folds": len(current["fold_manifests"]),
                "contract_file_sha256": current["contract_file_sha256"],
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
