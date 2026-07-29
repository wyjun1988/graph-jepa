from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Mapping

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.kiwoom_ohlcv import (
    canonical_json_sha256,
    write_immutable_raw_page,
)
from stock_v2.kiwoom_us import (
    KIWOOM_US_EXCHANGES,
    audit_kiwoom_us_daily_frame,
    fetch_kiwoom_us_universe,
    normalize_us_exchange,
    normalize_us_ticker,
    parse_kiwoom_us_daily_rows,
)
from stock_v2.cross_market_clock import (
    EXCHANGE_CALENDARS_VERSION,
    us_daily_session_available_at,
)
from stock_v2.ops.brokers import KiwoomRestBroker
from stock_v2.ops.config import KiwoomConfig


ROLE = "kiwoom_us_current_universe_daily_liquidity_screen"
LEDGER_ROLE = "kiwoom_us_daily_screen_record"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect one immutable usa06012 page for the current Kiwoom US "
            "universe. This is a survivorship-biased liquidity screen, not a "
            "point-in-time research release."
        )
    )
    parser.add_argument("--as-of", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--server", choices=["real", "mock"], default="real")
    parser.add_argument("--exchange", action="append", default=[])
    parser.add_argument("--sensor-config")
    parser.add_argument("--sleep-sec", type=float, default=0.25)
    parser.add_argument("--timeout-sec", type=float, default=20.0)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def symbol_key(exchange: object, ticker: object) -> str:
    exchange = normalize_us_exchange(exchange)
    ticker = normalize_us_ticker(ticker)
    readable = re.sub(r"[^A-Z0-9.-]+", "_", ticker).strip("_") or "symbol"
    suffix = hashlib.sha256(f"{exchange}|{ticker}".encode("ascii")).hexdigest()[:10]
    return f"{exchange}_{readable}_{suffix}"


def daily_request(exchange: object, ticker: object, as_of: object) -> dict[str, str]:
    return {
        "stex_tp": normalize_us_exchange(exchange),
        "stk_cd": normalize_us_ticker(ticker),
        "strt_dt": pd.Timestamp(as_of).strftime("%Y%m%d"),
        "upd_stkpc_tp": "1",
        "exrt_appl_tp": "0",
    }


def _write_immutable_json(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, indent=2
    ) + "\n"
    if path.exists():
        if path.read_text(encoding="utf-8") != encoded:
            raise RuntimeError(f"immutable JSON changed: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(encoded, encoding="utf-8")
    temporary.replace(path)


def _write_immutable_parquet(path: Path, frame: pd.DataFrame) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_parquet(temporary)
    digest = file_sha256(temporary)
    if path.exists():
        existing = file_sha256(path)
        temporary.unlink()
        if existing != digest:
            raise RuntimeError(f"immutable parquet changed: {path}")
        return existing
    temporary.replace(path)
    return digest


def read_ledger(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    previous: str | None = None
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        claimed = str(record.pop("record_sha256"))
        if record.get("previous_record_sha256") != previous:
            raise ValueError(f"US screen ledger chain broke at line {line_number}")
        actual = canonical_json_sha256(record)
        if claimed != actual:
            raise ValueError(f"US screen ledger hash broke at line {line_number}")
        record["record_sha256"] = claimed
        records.append(record)
        previous = claimed
    return records


def append_ledger(path: Path, payload: Mapping[str, Any]) -> dict[str, Any]:
    records = read_ledger(path)
    record = {
        **dict(payload),
        "previous_record_sha256": (
            records[-1]["record_sha256"] if records else None
        ),
    }
    record["record_sha256"] = canonical_json_sha256(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(
            json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
        )
        handle.flush()
    return record


def _source_pins() -> dict[str, str]:
    relatives = (
        "scripts/collect_kiwoom_us_daily_screen.py",
        "stock_v2/kiwoom_us.py",
        "stock_v2/cross_market_clock.py",
        "stock_v2/kiwoom_ohlcv.py",
        "stock_v2/ops/brokers.py",
        "stock_v2/ops/config.py",
    )
    return {relative: file_sha256(ROOT / relative) for relative in relatives}


def load_sensor_config(path: Path | None) -> tuple[dict[str, Any] | None, list[dict[str, str]]]:
    if path is None:
        return None, []
    payload = json.loads(path.read_text(encoding="utf-8"))
    if (
        payload.get("role") != "kiwoom_us_korea_impact_sensor_contract"
        or payload.get("live_orders_allowed") is not False
        or payload.get("promotion_eligible") is not False
    ):
        raise ValueError("invalid or unsafe US impact sensor contract")
    raw_nodes = payload.get("nodes")
    if not isinstance(raw_nodes, list) or not raw_nodes:
        raise ValueError("US impact sensor contract has no nodes")
    nodes: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for raw in raw_nodes:
        if not isinstance(raw, Mapping):
            raise ValueError("US impact sensor node is invalid")
        node = {
            "exchange": normalize_us_exchange(raw.get("exchange")),
            "ticker": normalize_us_ticker(raw.get("ticker")),
            "channel": str(raw.get("channel") or "").strip(),
            "node_role": str(raw.get("node_role") or "").strip(),
        }
        key = (node["exchange"], node["ticker"])
        if key in seen or not node["channel"] or not node["node_role"]:
            raise ValueError(f"invalid duplicate US impact sensor: {key}")
        seen.add(key)
        nodes.append(node)
    return payload, nodes


def _load_or_fetch_universe(
    broker: KiwoomRestBroker,
    *,
    output: Path,
    exchanges: tuple[str, ...],
    sleep_sec: float,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    parquet = output / "universe" / "current_universe.parquet"
    metadata_path = output / "universe" / "snapshot.json"
    if parquet.exists() or metadata_path.exists():
        if not parquet.is_file() or not metadata_path.is_file():
            raise RuntimeError("US universe snapshot is incomplete")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if file_sha256(parquet) != metadata["parquet_sha256"]:
            raise RuntimeError("US universe parquet hash changed")
        frame = pd.read_parquet(parquet)
        if tuple(sorted(set(frame.index.get_level_values("Exchange")))) != exchanges:
            raise RuntimeError("US universe exchange scope changed")
        return frame, metadata

    raw_hashes: list[str] = []

    def sink(
        exchange: str,
        page_index: int,
        response: Mapping[str, Any],
        has_more: bool,
    ) -> None:
        request = {"stex_tp": exchange}
        envelope = {
            "source": "kiwoom_rest_us",
            "endpoint": "/api/us/stkinfo",
            "api_id": "usa10099",
            "run_id": output.name,
            "ticker": exchange,
            "basis": "current_universe",
            "request": request,
            "page_index": page_index,
            "has_more": bool(has_more),
            "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
            "response_sha256": canonical_json_sha256(response),
            "response": dict(response),
        }
        path = output / "raw" / "universe" / exchange / f"page_{page_index:04d}.json"
        raw_hashes.append(write_immutable_raw_page(path, envelope))

    frame = fetch_kiwoom_us_universe(
        broker,
        exchanges=exchanges,
        sleep_sec=sleep_sec,
        raw_page_sink=sink,
    )
    parquet_sha = _write_immutable_parquet(parquet, frame)
    metadata = {
        "schema_version": 1,
        "role": "kiwoom_us_current_universe_snapshot",
        "rows": int(len(frame)),
        "exchanges": list(exchanges),
        "parquet": str(parquet.relative_to(output)),
        "parquet_sha256": parquet_sha,
        "raw_page_sha256": raw_hashes,
        "point_in_time_universe": False,
        "survivorship_bias": True,
        "deployment_eligible": False,
        "live_orders_allowed": False,
    }
    _write_immutable_json(metadata_path, metadata)
    return frame, metadata


def _record_key(record: Mapping[str, Any]) -> tuple[str, str]:
    return str(record["exchange"]), str(record["ticker"])


def main() -> None:
    args = parse_args()
    if args.sleep_sec < 0.2:
        raise ValueError("US daily screen requires sleep-sec >= 0.2")
    if args.limit < 0:
        raise ValueError("limit must be non-negative")
    as_of = pd.Timestamp(args.as_of).normalize()
    collection_cutoff_utc = pd.Timestamp.now(tz="UTC")
    as_of_available_utc = us_daily_session_available_at(
        as_of, vendor_lag="15min", finalization_sessions=1
    )
    if as_of_available_utc > collection_cutoff_utc:
        raise ValueError(
            "requested US daily as-of is not finalized: "
            f"available_at={as_of_available_utc.isoformat()} "
            f"cutoff={collection_cutoff_utc.isoformat()}"
        )
    output = Path(args.output_dir).expanduser()
    output.mkdir(parents=True, exist_ok=True)
    sensor_path = None
    if args.sensor_config:
        candidate = Path(args.sensor_config).expanduser()
        sensor_path = candidate if candidate.is_absolute() else ROOT / candidate
    sensor_contract, sensor_nodes = load_sensor_config(sensor_path)
    exchanges = tuple(
        sorted(
            normalize_us_exchange(value)
            for value in (args.exchange or list(KIWOOM_US_EXCHANGES))
        )
    )
    if len(exchanges) != len(set(exchanges)):
        raise ValueError("exchange scope contains duplicates")

    contract = {
        "schema_version": 1,
        "role": ROLE,
        "run_id": output.name,
        "as_of": as_of.date().isoformat(),
        "exchanges": list(exchanges),
        "requested_limit": int(args.limit),
        "sensor_contract": (
            None
            if sensor_path is None
            else {
                "path": str(sensor_path),
                "sha256": file_sha256(sensor_path),
                "nodes": len(sensor_nodes),
            }
        ),
        "api_contract": {
            "universe": ["/api/us/stkinfo", "usa10099"],
            "daily": ["/api/us/chart", "usa06012"],
            "daily_pages_per_symbol": 1,
            "adjusted": True,
            "exchange_rate_applied": False,
            "daily_finalization_sessions": 1,
            "daily_vendor_lag": "15min",
            "latest_mutable_daily_row_allowed": False,
            "calendar": "XNYS",
            "calendar_library": "exchange_calendars",
            "calendar_library_version": EXCHANGE_CALENDARS_VERSION,
        },
        "source_pins": _source_pins(),
        "point_in_time_universe": False,
        "survivorship_bias": True,
        "research_use": "current-universe liquidity screen only",
        "counts_as_training_release": False,
        "deployment_eligible": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_allowed": False,
        "credential_material_recorded": False,
    }
    contract_path = output / "contract.json"
    _write_immutable_json(contract_path, contract)

    broker = KiwoomRestBroker(
        KiwoomConfig(
            env_file=args.env_file,
            server=args.server,
            timeout_sec=float(args.timeout_sec),
        ),
        dry_run=True,
    )
    universe, universe_metadata = _load_or_fetch_universe(
        broker,
        output=output,
        exchanges=exchanges,
        sleep_sec=float(args.sleep_sec),
    )
    sensor_metadata: dict[tuple[str, str], dict[str, str]] = {}
    if sensor_nodes:
        selected = []
        for node in sensor_nodes:
            key = (node["exchange"], node["ticker"])
            if key not in universe.index:
                raise ValueError(f"US impact sensor is absent from current universe: {key}")
            selected.append(key)
            sensor_metadata[key] = node
    else:
        selected = list(universe.sort_index().index)
    if args.limit:
        selected = selected[: int(args.limit)]

    ledger_path = output / "records.jsonl"
    records = read_ledger(ledger_path)
    latest = {_record_key(record): record for record in records}
    completed = {
        key for key, record in latest.items() if record["status"] in {"ok", "empty"}
    }
    failures = 0
    for position, (exchange, ticker) in enumerate(selected, 1):
        key = (str(exchange), str(ticker))
        if key in completed:
            continue
        request = daily_request(exchange, ticker, as_of)
        safe_key = symbol_key(exchange, ticker)
        metadata = sensor_metadata.get(key, {})
        try:
            started = time.perf_counter()
            data, has_more, _cursor = broker.post_readonly_with_continuation(
                "/api/us/chart",
                "usa06012",
                request,
                continuation=False,
                next_key=None,
            )
            envelope = {
                "source": "kiwoom_rest_us",
                "endpoint": "/api/us/chart",
                "api_id": "usa06012",
                "run_id": output.name,
                "ticker": f"{exchange}:{ticker}",
                "basis": "adjusted_usd_current_universe_screen",
                "request": request,
                "page_index": 1,
                "has_more": bool(has_more),
                "retrieved_at_utc": datetime.now(timezone.utc).isoformat(),
                "response_sha256": canonical_json_sha256(data),
                "response": data,
            }
            raw_path = output / "raw" / "daily" / exchange / f"{safe_key}.json"
            raw_sha = write_immutable_raw_page(raw_path, envelope)
            frame = parse_kiwoom_us_daily_rows(data.get("result_list"))
            frame = frame.loc[frame.index <= as_of]
            if frame.empty:
                status = "empty"
                audit = {"rows": 0, "first_date": None, "last_date": None}
                parquet_relative = None
                parquet_sha = None
                median_trading_value = None
                median_volume = None
            else:
                audit = audit_kiwoom_us_daily_frame(frame)
                parquet_path = output / "daily" / exchange / f"{safe_key}.parquet"
                parquet_sha = _write_immutable_parquet(parquet_path, frame)
                parquet_relative = str(parquet_path.relative_to(output))
                trading_values = frame["TradingValue"].to_numpy(float)
                volumes = frame["Volume"].to_numpy(float)
                median_trading_value = float(
                    np.nanmedian(trading_values[np.isfinite(trading_values)])
                )
                median_volume = float(np.nanmedian(volumes[np.isfinite(volumes)]))
                status = "ok"
            append_ledger(
                ledger_path,
                {
                    "schema_version": 1,
                    "role": LEDGER_ROLE,
                    "status": status,
                    "exchange": exchange,
                    "ticker": ticker,
                    "position": position,
                    "sensor_channel": metadata.get("channel"),
                    "sensor_node_role": metadata.get("node_role"),
                    "as_of": as_of.date().isoformat(),
                    "as_of_available_at_utc": as_of_available_utc.isoformat(),
                    "request": request,
                    "raw_path": str(raw_path.relative_to(output)),
                    "raw_sha256": raw_sha,
                    "parquet_path": parquet_relative,
                    "parquet_sha256": parquet_sha,
                    "audit": audit,
                    "median_trading_value": median_trading_value,
                    "median_volume": median_volume,
                    "vendor_has_more": bool(has_more),
                    "counts_as_training_release": False,
                    "deployment_eligible": False,
                    "live_orders_allowed": False,
                    "broker_order_calls_executed": 0,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            elapsed = time.perf_counter() - started
            time.sleep(max(0.0, float(args.sleep_sec) - elapsed))
        except Exception as exc:
            failures += 1
            append_ledger(
                ledger_path,
                {
                    "schema_version": 1,
                    "role": LEDGER_ROLE,
                    "status": "error",
                    "exchange": exchange,
                    "ticker": ticker,
                    "position": position,
                    "sensor_channel": metadata.get("channel"),
                    "sensor_node_role": metadata.get("node_role"),
                    "as_of": as_of.date().isoformat(),
                    "error_type": type(exc).__name__,
                    "counts_as_training_release": False,
                    "deployment_eligible": False,
                    "live_orders_allowed": False,
                    "broker_order_calls_executed": 0,
                    "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
                },
            )
            if args.fail_fast:
                raise
        if position % 100 == 0:
            print(
                json.dumps(
                    {
                        "processed": position,
                        "total": len(selected),
                        "failures_this_run": failures,
                        "orders": 0,
                    },
                    sort_keys=True,
                ),
                flush=True,
            )

    final_records = read_ledger(ledger_path)
    final_latest = {_record_key(record): record for record in final_records}
    status_counts: dict[str, int] = {}
    for key in selected:
        status = str(final_latest.get(tuple(key), {}).get("status", "missing"))
        status_counts[status] = status_counts.get(status, 0) + 1
    complete = status_counts.get("missing", 0) == 0 and status_counts.get("error", 0) == 0
    manifest = {
        "schema_version": 1,
        "role": ROLE,
        "status": "complete" if complete else "incomplete",
        "run_id": output.name,
        "as_of": as_of.date().isoformat(),
        "as_of_available_at_utc": as_of_available_utc.isoformat(),
        "exchanges": list(exchanges),
        "universe_rows": int(len(universe)),
        "selected_rows": int(len(selected)),
        "sensor_contract_sha256": (
            None if sensor_path is None else file_sha256(sensor_path)
        ),
        "status_counts": status_counts,
        "contract_sha256": file_sha256(contract_path),
        "universe_snapshot_sha256": canonical_json_sha256(universe_metadata),
        "records_sha256": file_sha256(ledger_path),
        "records_head_sha256": final_records[-1]["record_sha256"],
        "point_in_time_universe": False,
        "survivorship_bias": True,
        "counts_as_training_release": False,
        "deployment_eligible": False,
        "promotion_eligible": False,
        "live_orders_allowed": False,
        "broker_order_calls_executed": 0,
    }
    manifest_name = "manifest.json" if complete else "manifest.partial.json"
    _write_immutable_json(output / manifest_name, manifest)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
