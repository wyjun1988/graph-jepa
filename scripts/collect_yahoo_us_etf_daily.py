from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.backfill_kiwoom_us_daily import audit_session_coverage
from scripts.collect_kiwoom_us_daily_screen import (
    _write_immutable_json,
    _write_immutable_parquet,
    file_sha256,
    load_sensor_config,
    symbol_key,
)
from stock_v2.kiwoom_ohlcv import (
    canonical_json_sha256,
    write_immutable_raw_page,
)
from stock_v2.yahoo_us import (
    YAHOO_CHART_URL,
    audit_yahoo_us_daily_frame,
    fetch_yahoo_us_daily_history,
    yahoo_chart_request,
)


ROLE = "yahoo_us_etf_daily_cross_source_diagnostic"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect immutable Yahoo Chart ETF histories only as an independent "
            "diagnostic source. These files are never a deployment release."
        )
    )
    parser.add_argument("--sensor-config", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--sleep-sec", type=float, default=0.25)
    parser.add_argument("--timeout-sec", type=float, default=30.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _source_pins() -> dict[str, str]:
    paths = (
        "scripts/collect_yahoo_us_etf_daily.py",
        "scripts/backfill_kiwoom_us_daily.py",
        "scripts/collect_kiwoom_us_daily_screen.py",
        "stock_v2/yahoo_us.py",
        "stock_v2/kiwoom_ohlcv.py",
        "requirements.txt",
    )
    return {path: file_sha256(ROOT / path) for path in paths}


def _load_or_create_contract(path: Path, static: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in static.items():
            if existing.get(key) != value:
                raise RuntimeError(f"Yahoo ETF diagnostic contract changed: {key}")
        return existing
    contract = {
        **dict(static),
        "collection_started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_immutable_json(path, contract)
    return contract


def _verify_record(output: Path, path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    record = json.loads(path.read_text(encoding="utf-8"))
    for relative, expected in record.get("artifact_sha256", {}).items():
        artifact = output / relative
        if not artifact.is_file() or file_sha256(artifact) != expected:
            raise RuntimeError(f"Yahoo ETF diagnostic artifact changed: {artifact}")
    return record


def main() -> int:
    args = parse_args()
    if args.sleep_sec < 0 or args.timeout_sec <= 0 or args.limit < 0:
        raise ValueError("invalid Yahoo diagnostic runtime limits")
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise ValueError("end must not precede start")
    sensor_path = Path(args.sensor_config).expanduser()
    if not sensor_path.is_absolute():
        sensor_path = ROOT / sensor_path
    sensor, nodes = load_sensor_config(sensor_path)
    if (
        sensor is None
        or sensor.get("instrument_policy", {}).get("us_etfs_only") is not True
    ):
        raise ValueError("Yahoo diagnostic requires the frozen ETF-only contract")
    if args.limit:
        nodes = nodes[: int(args.limit)]

    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    static_contract = {
        "schema_version": 1,
        "role": ROLE,
        "run_id": output.name,
        "sensor_config": str(sensor_path),
        "sensor_config_sha256": file_sha256(sensor_path),
        "start": start.date().isoformat(),
        "end": end.date().isoformat(),
        "selected_nodes": len(nodes),
        "endpoint": YAHOO_CHART_URL,
        "corporate_action_events_requested": ["div", "splits"],
        "adjusted_close_requested": True,
        "source_pins": _source_pins(),
        "official_exchange_data_contract": False,
        "independent_diagnostic_only": True,
        "point_in_time_universe": False,
        "survivorship_bias": True,
        "counts_as_training_release": False,
        "deployment_eligible": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_allowed": False,
    }
    contract_path = output / "contract.json"
    _load_or_create_contract(contract_path, static_contract)
    contract_sha256 = file_sha256(contract_path)

    records: list[dict[str, Any]] = []
    errors = 0
    for position, node in enumerate(nodes, 1):
        exchange = node["exchange"]
        ticker = node["ticker"]
        safe = symbol_key(exchange, ticker)
        symbol_root = output / "symbols" / exchange / safe
        record_path = symbol_root / "record.json"
        prior = _verify_record(output, record_path)
        if prior is not None:
            records.append(prior)
            continue
        attempt_path = symbol_root / "attempt.json"
        url, parameters = yahoo_chart_request(ticker, start, end)
        attempt_static = {
            "schema_version": 1,
            "role": "yahoo_us_etf_daily_symbol_attempt",
            "contract_sha256": contract_sha256,
            "exchange": exchange,
            "ticker": ticker,
            "url": url,
            "parameters": parameters,
            "live_orders_allowed": False,
        }
        if attempt_path.exists():
            attempt = json.loads(attempt_path.read_text(encoding="utf-8"))
            for key, value in attempt_static.items():
                if attempt.get(key) != value:
                    raise RuntimeError(f"Yahoo ETF attempt changed: {attempt_path}")
        else:
            attempt = {
                **attempt_static,
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            _write_immutable_json(attempt_path, attempt)
        raw_relative = Path("raw") / exchange / safe / "chart.json"
        raw_path = output / raw_relative
        raw_file_sha: str | None = None
        raw_envelope_sha: str | None = None

        def raw_sink(
            request_parameters: Mapping[str, Any], response: Mapping[str, Any]
        ) -> None:
            nonlocal raw_file_sha, raw_envelope_sha
            envelope = {
                "schema_version": 1,
                "source": "yahoo_chart_unofficial_diagnostic",
                "endpoint": url,
                "run_id": output.name,
                "ticker": ticker,
                "basis": "raw_ohlcv_with_dividend_split_adjusted_close",
                "request": dict(request_parameters),
                "page_index": 1,
                "retrieved_at_utc": attempt["retrieved_at_utc"],
                "response_sha256": canonical_json_sha256(response),
                "response": dict(response),
            }
            raw_envelope_sha = write_immutable_raw_page(raw_path, envelope)
            raw_file_sha = file_sha256(raw_path)

        try:
            frame = fetch_yahoo_us_daily_history(
                ticker,
                start,
                end,
                timeout_sec=float(args.timeout_sec),
                raw_sink=raw_sink,
            )
            audit = audit_yahoo_us_daily_frame(frame)
            if frame.empty or pd.Timestamp(frame.index.max()).normalize() != end:
                raise ValueError(f"Yahoo ETF history does not cover end: {ticker}")
            coverage = audit_session_coverage(frame)
            status = "ok" if coverage["missing_session_count"] == 0 else "quality_warning"
            parquet_relative = Path("symbols") / exchange / safe / "daily.parquet"
            parquet_sha = _write_immutable_parquet(output / parquet_relative, frame)
            if raw_file_sha is None or raw_envelope_sha is None:
                raise RuntimeError("Yahoo raw response was not persisted")
            artifacts = {
                str(raw_relative): raw_file_sha,
                str(parquet_relative): parquet_sha,
                str(Path("symbols") / exchange / safe / "attempt.json"): file_sha256(
                    attempt_path
                ),
            }
            record = {
                "schema_version": 1,
                "role": "yahoo_us_etf_daily_symbol_record",
                "status": status,
                "position": position,
                "exchange": exchange,
                "ticker": ticker,
                "channel": node["channel"],
                "node_role": node["node_role"],
                "contract_sha256": contract_sha256,
                "audit": audit,
                "session_coverage": coverage,
                "observed_lifecycle_start": audit["first_date"],
                "observed_lifecycle_end": audit["last_date"],
                "raw_envelope_sha256": raw_envelope_sha,
                "artifact_sha256": artifacts,
                "independent_diagnostic_only": True,
                "counts_as_training_release": False,
                "deployment_eligible": False,
                "live_orders_allowed": False,
                "broker_order_calls_executed": 0,
            }
            _write_immutable_json(record_path, record)
            records.append(record)
            print(
                json.dumps(
                    {
                        "position": position,
                        "total": len(nodes),
                        "ticker": ticker,
                        "rows": len(frame),
                        "status": status,
                        "orders": 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
        except Exception as exc:
            errors += 1
            print(
                json.dumps(
                    {
                        "position": position,
                        "ticker": ticker,
                        "status": "error",
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:500],
                        "orders": 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
            if args.fail_fast:
                raise
        time.sleep(float(args.sleep_sec))

    status_counts: dict[str, int] = {}
    for record in records:
        value = str(record["status"])
        status_counts[value] = status_counts.get(value, 0) + 1
    complete = errors == 0 and len(records) == len(nodes)
    manifest = {
        "schema_version": 1,
        "role": ROLE,
        "status": "complete" if complete else "incomplete",
        "quality_gate_passed": complete and status_counts == {"ok": len(nodes)},
        "run_id": output.name,
        "contract_sha256": contract_sha256,
        "selected_nodes": len(nodes),
        "completed_nodes": len(records),
        "errors": errors,
        "status_counts": status_counts,
        "record_sha256": {
            f"{record['exchange']}:{record['ticker']}": file_sha256(
                output
                / "symbols"
                / record["exchange"]
                / symbol_key(record["exchange"], record["ticker"])
                / "record.json"
            )
            for record in records
        },
        "official_exchange_data_contract": False,
        "independent_diagnostic_only": True,
        "counts_as_training_release": False,
        "deployment_eligible": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    if complete:
        _write_immutable_json(output / "manifest.json", manifest)
    print(json.dumps(manifest, sort_keys=True), flush=True)
    return 0 if complete else 2


if __name__ == "__main__":
    sys.exit(main())
