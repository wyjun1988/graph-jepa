from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Mapping

import exchange_calendars as xcals
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
from stock_v2.cross_market_clock import (
    EXCHANGE_CALENDARS_VERSION,
    annotate_us_daily_availability,
    us_daily_session_available_at,
)
from stock_v2.kiwoom_ohlcv import (
    canonical_json_sha256,
    write_immutable_raw_page,
)
from stock_v2.kiwoom_us import (
    audit_kiwoom_us_daily_frame,
    fetch_kiwoom_us_daily_history,
    repair_kiwoom_us_daily_ohlc_envelope,
)
from stock_v2.ops.brokers import KiwoomRestBroker
from stock_v2.ops.config import KiwoomConfig


ROLE = "kiwoom_us_etf_daily_backfill"
SOURCE_URL = (
    "https://openapi.kiwoom.com/guide/apiguide?jobTpCode=36&apiId=usa06012"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect immutable adjusted usa06012 histories for a frozen US ETF "
            "sensor contract. The newest mutable vendor row is forbidden."
        )
    )
    parser.add_argument("--sensor-config", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--server", choices=["real", "mock"], default="real")
    parser.add_argument("--sleep-sec", type=float, default=0.20)
    parser.add_argument("--max-pages", type=int, default=40)
    parser.add_argument("--maximum-relative-ohlc-repair", type=float, default=0.01)
    parser.add_argument("--maximum-repaired-fraction", type=float, default=0.01)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def _source_pins() -> dict[str, str]:
    paths = (
        "scripts/backfill_kiwoom_us_daily.py",
        "scripts/collect_kiwoom_us_daily_screen.py",
        "stock_v2/cross_market_clock.py",
        "stock_v2/kiwoom_us.py",
        "stock_v2/kiwoom_ohlcv.py",
        "stock_v2/ops/brokers.py",
        "stock_v2/ops/config.py",
        "requirements.txt",
    )
    return {path: file_sha256(ROOT / path) for path in paths}


def _load_or_create_contract(path: Path, static: Mapping[str, Any]) -> dict[str, Any]:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in static.items():
            if existing.get(key) != value:
                raise RuntimeError(f"US daily backfill contract changed: {key}")
        return existing
    contract = {
        **dict(static),
        "collection_started_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_immutable_json(path, contract)
    return contract


def audit_session_coverage(frame: pd.DataFrame) -> dict[str, Any]:
    if len(frame.index) == 0:
        return {
            "expected_sessions": 0,
            "observed_sessions": 0,
            "missing_sessions": [],
            "missing_session_count": 0,
        }
    calendar = xcals.get_calendar("XNYS")
    first = pd.Timestamp(frame.index.min()).normalize()
    last = pd.Timestamp(frame.index.max()).normalize()
    expected = pd.DatetimeIndex(calendar.sessions_in_range(first, last)).tz_localize(None)
    observed = pd.DatetimeIndex(frame.index).tz_localize(None).normalize()
    missing = expected.difference(observed)
    extra = observed.difference(expected)
    if len(extra):
        raise ValueError(
            "US daily history contains non-XNYS sessions: "
            + ",".join(value.date().isoformat() for value in extra[:10])
        )
    return {
        "expected_sessions": int(len(expected)),
        "observed_sessions": int(len(observed)),
        "missing_sessions": [value.date().isoformat() for value in missing],
        "missing_session_count": int(len(missing)),
    }


def _verify_completed_record(output: Path, record_path: Path) -> dict[str, Any] | None:
    if not record_path.is_file():
        return None
    record = json.loads(record_path.read_text(encoding="utf-8"))
    for relative, expected in record.get("artifact_sha256", {}).items():
        path = output / relative
        if not path.is_file() or file_sha256(path) != expected:
            raise RuntimeError(f"US daily backfill artifact changed: {path}")
    return record


def _symbol_attempt(
    path: Path,
    *,
    contract_sha256: str,
    exchange: str,
    ticker: str,
    request: Mapping[str, str],
) -> dict[str, Any]:
    static = {
        "schema_version": 1,
        "role": "kiwoom_us_etf_daily_symbol_attempt",
        "contract_sha256": contract_sha256,
        "exchange": exchange,
        "ticker": ticker,
        "request": dict(request),
        "live_orders_allowed": False,
        "broker_order_calls_allowed": False,
    }
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        for key, value in static.items():
            if existing.get(key) != value:
                raise RuntimeError(f"US daily symbol attempt changed: {path}")
        return existing
    attempt = {
        **static,
        "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    _write_immutable_json(path, attempt)
    return attempt


def _request(
    exchange: str, ticker: str, end: pd.Timestamp
) -> dict[str, str]:
    return {
        "stex_tp": exchange,
        "stk_cd": ticker,
        "strt_dt": end.strftime("%Y%m%d"),
        "upd_stkpc_tp": "1",
        "exrt_appl_tp": "0",
    }


def main() -> int:
    args = parse_args()
    if args.sleep_sec < 0.20 and args.server == "real":
        raise ValueError("real Kiwoom collection requires sleep-sec >= 0.20")
    if args.max_pages <= 0 or args.limit < 0:
        raise ValueError("max-pages must be positive and limit non-negative")
    if (
        args.maximum_relative_ohlc_repair < 0
        or not 0 <= args.maximum_repaired_fraction <= 1
    ):
        raise ValueError("invalid OHLC envelope repair limits")
    start = pd.Timestamp(args.start).normalize()
    end = pd.Timestamp(args.end).normalize()
    if end < start:
        raise ValueError("end must not precede start")
    end_available = us_daily_session_available_at(
        end, vendor_lag="15min", finalization_sessions=1
    )
    if end_available > pd.Timestamp.now(tz="UTC"):
        raise ValueError(
            "US daily backfill end is not vendor-finalized: "
            f"available_at={end_available.isoformat()}"
        )

    sensor_path = Path(args.sensor_config).expanduser()
    if not sensor_path.is_absolute():
        sensor_path = ROOT / sensor_path
    sensor, nodes = load_sensor_config(sensor_path)
    if (
        sensor is None
        or sensor.get("instrument_policy", {}).get("us_etfs_only") is not True
        or sensor.get("instrument_policy", {}).get(
            "us_individual_equities_allowed"
        )
        is not False
    ):
        raise ValueError("US daily backfill requires the frozen ETF-only contract")
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
        "end_available_at_utc": end_available.isoformat(),
        "selected_nodes": len(nodes),
        "price_basis": "back_adjusted_usd",
        "exchange_rate_applied": False,
        "calendar": "XNYS",
        "calendar_library": "exchange_calendars",
        "calendar_library_version": EXCHANGE_CALENDARS_VERSION,
        "daily_finalization_sessions": 1,
        "daily_vendor_lag": "15min",
        "maximum_relative_ohlc_repair": float(
            args.maximum_relative_ohlc_repair
        ),
        "maximum_repaired_fraction": float(args.maximum_repaired_fraction),
        "repaired_ohlc_range_features_allowed": False,
        "source_pins": _source_pins(),
        "current_universe_only": True,
        "point_in_time_universe": False,
        "survivorship_bias": True,
        "counts_as_training_release": False,
        "deployment_eligible": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_allowed": False,
    }
    contract_path = output / "contract.json"
    contract = _load_or_create_contract(contract_path, static_contract)
    contract_sha256 = file_sha256(contract_path)

    broker = KiwoomRestBroker(
        KiwoomConfig(
            env_file=args.env_file,
            server=args.server,
            timeout_sec=30.0,
        ),
        dry_run=True,
    )
    if not broker.authenticate():
        raise RuntimeError("Kiwoom authentication failed for US daily backfill")

    records: list[dict[str, Any]] = []
    errors = 0
    for position, node in enumerate(nodes, 1):
        exchange = node["exchange"]
        ticker = node["ticker"]
        safe = symbol_key(exchange, ticker)
        symbol_root = output / "symbols" / exchange / safe
        record_path = symbol_root / "record.json"
        completed = _verify_completed_record(output, record_path)
        if completed is not None:
            records.append(completed)
            continue

        request = _request(exchange, ticker, end)
        attempt = _symbol_attempt(
            symbol_root / "attempt.json",
            contract_sha256=contract_sha256,
            exchange=exchange,
            ticker=ticker,
            request=request,
        )
        page_hashes: dict[str, str] = {}
        page_envelope_hashes: dict[str, str] = {}
        last_has_more: bool | None = None

        def raw_sink(
            page_index: int,
            response: Mapping[str, Any],
            has_more: bool,
        ) -> None:
            nonlocal last_has_more
            envelope = {
                "schema_version": 1,
                "source": "kiwoom_rest_us",
                "source_url": SOURCE_URL,
                "endpoint": "/api/us/chart",
                "api_id": "usa06012",
                "run_id": output.name,
                "exchange": exchange,
                "ticker": ticker,
                "basis": "back_adjusted_usd",
                "request": request,
                "page_index": int(page_index),
                "has_more": bool(has_more),
                "retrieved_at_utc": attempt["retrieved_at_utc"],
                "response_sha256": canonical_json_sha256(response),
                "response": dict(response),
            }
            relative = (
                Path("raw") / exchange / safe / f"page_{page_index:04d}.json"
            )
            page_envelope_hashes[str(relative)] = write_immutable_raw_page(
                output / relative, envelope
            )
            page_hashes[str(relative)] = file_sha256(output / relative)
            last_has_more = bool(has_more)

        try:
            frame = fetch_kiwoom_us_daily_history(
                broker,
                exchange,
                ticker,
                start,
                end,
                adjusted=True,
                apply_exchange_rate=False,
                sleep_sec=float(args.sleep_sec),
                max_pages=int(args.max_pages),
                raw_page_sink=raw_sink,
            )
            if frame.empty:
                raise ValueError(f"US ETF has no daily history: {exchange}:{ticker}")
            frame, repair_audit = repair_kiwoom_us_daily_ohlc_envelope(
                frame,
                maximum_relative_repair=float(
                    args.maximum_relative_ohlc_repair
                ),
                maximum_repaired_fraction=float(args.maximum_repaired_fraction),
            )
            audit = audit_kiwoom_us_daily_frame(frame)
            if pd.Timestamp(frame.index.max()).normalize() != end:
                raise ValueError(
                    f"US ETF does not cover finalized end date: {exchange}:{ticker}"
                )
            aligned = annotate_us_daily_availability(
                frame, vendor_lag="15min", finalization_sessions=1
            )
            collection_started = pd.Timestamp(contract["collection_started_at_utc"])
            if (aligned["AvailableAtUTC"] > collection_started).any():
                raise ValueError(f"US ETF contains data unavailable at collection start: {ticker}")
            coverage = audit_session_coverage(frame)
            status = (
                "ok"
                if coverage["missing_session_count"] == 0
                and repair_audit["repaired_rows"] == 0
                else "quality_warning"
            )
            relative_parquet = Path("symbols") / exchange / safe / "daily.parquet"
            parquet_sha = _write_immutable_parquet(output / relative_parquet, aligned)
            artifacts = {
                **page_hashes,
                str(relative_parquet): parquet_sha,
                str((Path("symbols") / exchange / safe / "attempt.json")): file_sha256(
                    symbol_root / "attempt.json"
                ),
            }
            record = {
                "schema_version": 1,
                "role": "kiwoom_us_etf_daily_symbol_record",
                "status": status,
                "position": position,
                "exchange": exchange,
                "ticker": ticker,
                "channel": node["channel"],
                "node_role": node["node_role"],
                "contract_sha256": contract_sha256,
                "audit": audit,
                "ohlc_envelope_repair": repair_audit,
                "repaired_ohlc_range_features_allowed": False,
                "usable_unrepaired_fields": [
                    "Close",
                    "Volume",
                    "TradingValue",
                    "PreviousChange",
                    "ChangePct",
                ],
                "session_coverage": coverage,
                "observed_lifecycle_start": audit["first_date"],
                "observed_lifecycle_end": audit["last_date"],
                "history_stopped_without_more_pages": last_has_more is False,
                "raw_page_count": len(page_hashes),
                "raw_page_envelope_sha256": page_envelope_hashes,
                "artifact_sha256": artifacts,
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
                        "pages": len(page_hashes),
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
    quality_gate = complete and status_counts == {"ok": len(nodes)}
    manifest = {
        "schema_version": 1,
        "role": ROLE,
        "status": "complete" if complete else "incomplete",
        "quality_gate_passed": quality_gate,
        "run_id": output.name,
        "contract_sha256": contract_sha256,
        "sensor_config_sha256": file_sha256(sensor_path),
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
        "current_universe_only": True,
        "point_in_time_universe": False,
        "survivorship_bias": True,
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
