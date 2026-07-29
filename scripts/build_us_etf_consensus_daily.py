from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.collect_kiwoom_us_daily_screen import (
    _write_immutable_json,
    _write_immutable_parquet,
    file_sha256,
    load_sensor_config,
    symbol_key,
)
from stock_v2.us_etf_consensus import build_us_etf_daily_consensus


ROLE = "us_etf_cross_source_consensus_daily_panel"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a causally timestamped, source-masked ETF diagnostic panel. "
            "It cannot authorize deployment or live orders."
        )
    )
    parser.add_argument("--sensor-config", required=True)
    parser.add_argument("--quality-config", required=True)
    parser.add_argument("--kiwoom-release", required=True)
    parser.add_argument("--yahoo-release", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON object required: {path}")
    return payload


def _source_pins() -> dict[str, str]:
    paths = (
        "scripts/build_us_etf_consensus_daily.py",
        "stock_v2/us_etf_consensus.py",
        "scripts/collect_kiwoom_us_daily_screen.py",
    )
    return {path: file_sha256(ROOT / path) for path in paths}


def _records(root: Path, expected_role: str) -> dict[tuple[str, str], dict[str, Any]]:
    manifest = _load(root / "manifest.json")
    if (
        manifest.get("status") != "complete"
        or manifest.get("role") != expected_role
        or manifest.get("live_orders_allowed") is not False
        or manifest.get("broker_order_calls_executed") != 0
    ):
        raise ValueError(f"invalid source release manifest: {root}")
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for record_path in root.glob("symbols/*/*/record.json"):
        record = _load(record_path)
        key = (str(record["exchange"]), str(record["ticker"]))
        if key in records:
            raise ValueError(f"duplicate source record: {key}")
        for relative, expected in record.get("artifact_sha256", {}).items():
            artifact = root / relative
            if not artifact.is_file() or file_sha256(artifact) != expected:
                raise ValueError(f"source artifact hash changed: {artifact}")
        records[key] = record
    if len(records) != int(manifest.get("completed_nodes", -1)):
        raise ValueError(f"source record count mismatch: {root}")
    return records


def _parquet(root: Path, record: Mapping[str, Any]) -> Path:
    candidates = [
        relative
        for relative in record.get("artifact_sha256", {})
        if str(relative).endswith("daily.parquet")
    ]
    if len(candidates) != 1:
        raise ValueError("source record must reference one daily parquet")
    return root / candidates[0]


def main() -> None:
    args = parse_args()
    sensor_path = Path(args.sensor_config).expanduser()
    quality_path = Path(args.quality_config).expanduser()
    if not sensor_path.is_absolute():
        sensor_path = ROOT / sensor_path
    if not quality_path.is_absolute():
        quality_path = ROOT / quality_path
    sensor, nodes = load_sensor_config(sensor_path)
    quality = _load(quality_path)
    if sensor is None or sensor.get("instrument_policy", {}).get("us_etfs_only") is not True:
        raise ValueError("consensus panel requires the frozen ETF-only contract")
    if (
        quality.get("role") != "us_etf_cross_source_quality_contract"
        or quality.get("live_orders_allowed") is not False
        or quality.get("promotion_eligible") is not False
        or quality.get("external_nodes_masked_for_prediction_loss") is not False
    ):
        raise ValueError("invalid US ETF source-quality contract")
    exclusions = {
        (str(row["exchange"]), str(row["ticker"])): str(row["reason"])
        for row in quality.get("excluded_nodes", [])
    }
    node_map = {(row["exchange"], row["ticker"]): row for row in nodes}
    if not exclusions or not set(exclusions).issubset(node_map):
        raise ValueError("quality exclusions must be a non-empty sensor subset")

    kiwoom_root = Path(args.kiwoom_release).expanduser()
    yahoo_root = Path(args.yahoo_release).expanduser()
    kiwoom_records = _records(kiwoom_root, "kiwoom_us_etf_daily_backfill")
    yahoo_records = _records(
        yahoo_root, "yahoo_us_etf_daily_cross_source_diagnostic"
    )
    if set(kiwoom_records) != set(node_map) or set(yahoo_records) != set(node_map):
        raise ValueError("source releases do not exactly match the sensor contract")

    panels: list[pd.DataFrame] = []
    per_node: dict[str, dict[str, object]] = {}
    for key, node in node_map.items():
        if key in exclusions:
            continue
        kiwoom = pd.read_parquet(_parquet(kiwoom_root, kiwoom_records[key]))
        yahoo = pd.read_parquet(_parquet(yahoo_root, yahoo_records[key]))
        panel, summary = build_us_etf_daily_consensus(
            kiwoom,
            yahoo,
            close_relative_tolerance=float(quality["close_relative_tolerance"]),
            volume_relative_tolerance=float(quality["volume_relative_tolerance"]),
            volume_lookback_sessions=int(quality["volume_lookback_sessions"]),
            volume_minimum_history=int(quality["volume_minimum_history"]),
        )
        panel = panel.reset_index()
        panel.insert(0, "NodeRole", node["node_role"])
        panel.insert(0, "Channel", node["channel"])
        panel.insert(0, "Ticker", key[1])
        panel.insert(0, "Exchange", key[0])
        panels.append(panel)
        per_node[f"{key[0]}:{key[1]}"] = summary

    combined = pd.concat(panels, ignore_index=True)
    combined = combined.sort_values(["Date", "Exchange", "Ticker"], kind="stable")
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    contract = {
        "schema_version": 1,
        "role": ROLE,
        "sensor_config": str(sensor_path),
        "sensor_config_sha256": file_sha256(sensor_path),
        "quality_config": str(quality_path),
        "quality_config_sha256": file_sha256(quality_path),
        "kiwoom_release": str(kiwoom_root),
        "kiwoom_manifest_sha256": file_sha256(kiwoom_root / "manifest.json"),
        "yahoo_release": str(yahoo_root),
        "yahoo_manifest_sha256": file_sha256(yahoo_root / "manifest.json"),
        "source_pins": _source_pins(),
        "feature_allowlist": quality["feature_allowlist"],
        "external_nodes_masked_for_prediction_loss": False,
        "external_node_role": "input_only_observed_sensor",
        "thresholds_selected_after_source_distribution_audit": True,
        "point_in_time_universe": False,
        "survivorship_bias": True,
        "diagnostic_model_experiment_eligible": True,
        "counts_as_primary_forward_evidence": False,
        "counts_as_training_release": False,
        "deployment_eligible": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_allowed": False,
    }
    contract_path = output / "contract.json"
    _write_immutable_json(contract_path, contract)
    panel_path = output / "panel.parquet"
    panel_sha = _write_immutable_parquet(panel_path, combined)
    summary = {
        "schema_version": 1,
        "role": ROLE,
        "status": "complete",
        "nodes": len(per_node),
        "rows": int(len(combined)),
        "first_date": combined["Date"].min().date().isoformat(),
        "last_date": combined["Date"].max().date().isoformat(),
        "excluded_nodes": {
            f"{key[0]}:{key[1]}": reason for key, reason in exclusions.items()
        },
        "close_consensus_invalid_rows": int(
            sum(value["close_consensus_invalid_rows"] for value in per_node.values())
        ),
        "volume_consensus_invalid_rows": int(
            sum(value["volume_consensus_invalid_rows"] for value in per_node.values())
        ),
        "ohlc_envelope_invalid_rows": int(
            sum(value["ohlc_envelope_invalid_rows"] for value in per_node.values())
        ),
        "total_return_valid_rows": int(
            sum(value["total_return_valid_rows"] for value in per_node.values())
        ),
        "volume_feature_valid_rows": int(
            sum(value["volume_feature_valid_rows"] for value in per_node.values())
        ),
        "per_node": per_node,
        "contract_sha256": file_sha256(contract_path),
        "panel_sha256": panel_sha,
        "feature_allowlist": quality["feature_allowlist"],
        "external_nodes_masked_for_prediction_loss": False,
        "diagnostic_model_experiment_eligible": True,
        "counts_as_primary_forward_evidence": False,
        "counts_as_training_release": False,
        "deployment_eligible": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    _write_immutable_json(output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
