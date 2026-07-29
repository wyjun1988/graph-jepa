from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
from typing import Any, Mapping
from uuid import uuid4

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.kiwoom_minute import (
    KST,
    audit_kiwoom_minute_frame,
    fetch_kiwoom_minute_history,
    normalize_kiwoom_ticker,
    write_immutable_gzip_json,
)
from stock_v2.kiwoom_ohlcv import canonical_json_sha256
from stock_v2.ops.brokers import KiwoomRestBroker
from stock_v2.ops.config import KiwoomConfig


SOURCE_URL = "https://openapi.kiwoom.com/guide/apiguide?jobTpCode=07&apiId=ka10080"
COLLECTOR_SOURCE_PATHS = (
    "scripts/capture_kiwoom_live_minute_snapshot.py",
    "stock_v2/kiwoom_minute.py",
    "stock_v2/ops/brokers.py",
    "stock_v2/ops/config.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Capture a read-only, common-cutoff Kiwoom minute snapshot for live "
            "post-impact shadow inference."
        )
    )
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--session", required=True)
    parser.add_argument("--interval-minutes", type=int, default=5)
    parser.add_argument("--timestamp-semantics", choices=("start", "end"), required=True)
    parser.add_argument("--cutoff-hhmm")
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--server", choices=("real", "mock"), default="real")
    parser.add_argument("--sleep-sec", type=float, default=0.20)
    parser.add_argument("--max-pages", type=int, default=2)
    parser.add_argument("--minimum-populated-tickers", type=int, default=400)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collector_code_provenance() -> dict[str, Any]:
    sources = {
        relative: file_sha256(ROOT / relative) for relative in COLLECTOR_SOURCE_PATHS
    }
    return {
        "sources": sources,
        "source_tree_sha256": canonical_json_sha256(sources),
    }


def load_universe(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("universe") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("live minute universe must be a non-empty object list")
    result = [
        dict(row, ticker=normalize_kiwoom_ticker(row.get("ticker", "")))
        for row in rows
    ]
    tickers = [row["ticker"] for row in result]
    if len(tickers) != len(set(tickers)):
        raise ValueError("live minute universe contains duplicate tickers")
    return result


def active_on_session(row: Mapping[str, Any], session: pd.Timestamp) -> bool:
    listing = pd.to_datetime(row.get("listing_date"), errors="coerce")
    delisting = pd.to_datetime(row.get("delisting_date"), errors="coerce")
    return bool(
        (pd.isna(listing) or pd.Timestamp(listing).normalize() <= session)
        and (pd.isna(delisting) or session <= pd.Timestamp(delisting).normalize())
    )


def _parse_hhmm(session: pd.Timestamp, value: str) -> pd.Timestamp:
    if not re.fullmatch(r"[0-2][0-9]:[0-5][0-9]", str(value)):
        raise ValueError("cutoff must use HH:MM")
    hour, minute = (int(part) for part in str(value).split(":"))
    if hour >= 24:
        raise ValueError("cutoff hour is outside one day")
    return session.tz_localize(KST) + pd.Timedelta(hour * 60 + minute, unit="minute")


def resolve_common_cutoff(
    *,
    session: pd.Timestamp,
    now: pd.Timestamp,
    interval_minutes: int,
    explicit_hhmm: str | None,
) -> pd.Timestamp:
    interval = int(interval_minutes)
    if interval <= 0 or 60 % interval:
        raise ValueError("live snapshot interval must be a positive divisor of 60")
    current = pd.Timestamp(now)
    if current.tzinfo is None:
        raise ValueError("live snapshot current time must be timezone aware")
    current = current.tz_convert(KST)
    if current.tz_localize(None).normalize() != session:
        raise ValueError("live snapshot session must equal today in Asia/Seoul")
    if explicit_hhmm:
        cutoff = _parse_hhmm(session, explicit_hhmm)
    else:
        minute = (current.minute // interval) * interval
        cutoff = current.normalize() + pd.Timedelta(
            current.hour * 60 + minute, unit="minute"
        )
    latest_completed = current.floor(f"{interval}min")
    if cutoff > latest_completed:
        raise ValueError("live snapshot cutoff exceeds the latest completed interval")
    minute_of_day = cutoff.hour * 60 + cutoff.minute
    if minute_of_day < 9 * 60 + 15 or minute_of_day > 15 * 60 + 15:
        raise ValueError("live snapshot cutoff is outside model decision clocks")
    if (minute_of_day - 9 * 60) % interval:
        raise ValueError("live snapshot cutoff is not interval aligned")
    return cutoff


def completed_bar_mask(
    index: pd.DatetimeIndex,
    *,
    cutoff: pd.Timestamp,
    interval_minutes: int,
    timestamp_semantics: str,
) -> np.ndarray:
    timestamps = pd.DatetimeIndex(index)
    if timestamps.tz is None or cutoff.tzinfo is None:
        raise ValueError("completed-bar timestamps must be timezone aware")
    local = timestamps.tz_convert(KST)
    resolved_cutoff = pd.Timestamp(cutoff).tz_convert(KST)
    if timestamp_semantics == "start":
        completed = local + pd.Timedelta(int(interval_minutes), unit="minute")
    elif timestamp_semantics == "end":
        completed = local
    else:
        raise ValueError("timestamp semantics must be start or end")
    return np.asarray(completed <= resolved_cutoff, dtype=bool)


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    frame.to_parquet(temporary, compression="zstd", index=True)
    temporary.replace(path)


def main() -> int:
    args = parse_args()
    if args.server == "real" and float(args.sleep_sec) < 0.20:
        raise ValueError("real Kiwoom snapshot collection requires sleep-sec >= 0.20")
    if int(args.max_pages) <= 0 or int(args.minimum_populated_tickers) <= 0:
        raise ValueError("live snapshot page and population limits must be positive")
    session = pd.Timestamp(args.session).normalize()
    started_at = pd.Timestamp.now(tz=KST)
    cutoff = resolve_common_cutoff(
        session=session,
        now=started_at,
        interval_minutes=args.interval_minutes,
        explicit_hhmm=args.cutoff_hhmm,
    )
    universe_path = Path(args.universe_manifest)
    universe = load_universe(universe_path)
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise FileExistsError(f"immutable live snapshot already exists: {output_dir}")
    broker = KiwoomRestBroker(
        KiwoomConfig(server=args.server, env_file=args.env_file, timeout_sec=30.0),
        dry_run=True,
    )
    if not broker.authenticate():
        detail = f": {broker.last_auth_error}" if broker.last_auth_error else ""
        raise RuntimeError(f"Kiwoom read-only authentication failed{detail}")
    temporary = output_dir.with_name(f".{output_dir.name}.tmp-{uuid4().hex}")
    temporary.mkdir(parents=True, exist_ok=False)

    records: list[dict[str, Any]] = []
    output_records: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    active_count = 0
    try:
        for position, security in enumerate(universe, start=1):
            ticker = security["ticker"]
            if not active_on_session(security, session):
                records.append({"ticker": ticker, "status": "outside_lifecycle"})
                continue
            active_count += 1
            retrieved_at = datetime.now(tz=timezone.utc).isoformat()
            page_hashes: list[str] = []

            def save_page(
                page_index: int, response: Mapping[str, Any], has_more: bool
            ) -> None:
                envelope = {
                    "schema_version": 1,
                    "source": "kiwoom_rest",
                    "source_url": SOURCE_URL,
                    "endpoint": "/api/dostk/chart",
                    "api_id": "ka10080",
                    "ticker": ticker,
                    "session": str(session.date()),
                    "interval_minutes": int(args.interval_minutes),
                    "basis": "raw",
                    "common_cutoff_kst": cutoff.isoformat(),
                    "page_index": int(page_index),
                    "has_more": bool(has_more),
                    "retrieved_at_utc": retrieved_at,
                    "response_sha256": canonical_json_sha256(response),
                    "response": response,
                }
                raw_path = temporary / "raw" / ticker / f"page_{page_index:03d}.json.gz"
                page_hashes.append(write_immutable_gzip_json(raw_path, envelope))

            try:
                frame = fetch_kiwoom_minute_history(
                    broker,
                    ticker,
                    session,
                    session,
                    interval_minutes=int(args.interval_minutes),
                    adjusted=False,
                    sleep_sec=float(args.sleep_sec),
                    max_pages=int(args.max_pages),
                    raw_page_sink=save_page,
                    regular_session_only=True,
                )
                if not frame.empty:
                    local_dates = frame.index.tz_convert(KST).tz_localize(None).normalize()
                    selected = np.asarray(local_dates == session, dtype=bool)
                    selected &= completed_bar_mask(
                        frame.index,
                        cutoff=cutoff,
                        interval_minutes=int(args.interval_minutes),
                        timestamp_semantics=args.timestamp_semantics,
                    )
                    frame = frame.loc[selected].copy()
                audit = audit_kiwoom_minute_frame(frame, regular_session_only=True)
                if frame.empty:
                    record = {
                        "ticker": ticker,
                        "status": "empty",
                        "retrieved_at_utc": retrieved_at,
                        "raw_page_sha256": page_hashes,
                        "audit": audit,
                    }
                else:
                    path = temporary / "outputs" / f"{ticker}.parquet"
                    _write_frame(frame, path)
                    relative = path.relative_to(temporary).as_posix()
                    digest = file_sha256(path)
                    record = {
                        "ticker": ticker,
                        "status": "ok",
                        "retrieved_at_utc": retrieved_at,
                        "raw_page_sha256": page_hashes,
                        "audit": audit,
                        "path": relative,
                        "sha256": digest,
                        "bytes": path.stat().st_size,
                        "first_timestamp": frame.index[0].isoformat(),
                        "last_timestamp": frame.index[-1].isoformat(),
                    }
                    output_records.append(record)
                records.append(record)
            except Exception as exc:
                error = {
                    "ticker": ticker,
                    "error": f"{type(exc).__name__}: {str(exc)[:500]}",
                }
                records.append({"ticker": ticker, "status": "error", **error})
                errors.append(error)
            if position % 25 == 0 or position == len(universe):
                print(
                    f"captured={position}/{len(universe)} populated={len(output_records)} "
                    f"errors={len(errors)} cutoff={cutoff.strftime('%H:%M')}",
                    flush=True,
                )
            if float(args.sleep_sec) > 0:
                time.sleep(float(args.sleep_sec))

        if errors:
            raise RuntimeError(f"live minute snapshot has {len(errors)} ticker errors")
        if len(output_records) < int(args.minimum_populated_tickers):
            raise RuntimeError(
                f"only {len(output_records)} live tickers; require "
                f"{int(args.minimum_populated_tickers)}"
            )
        manifest = {
            "schema_version": 1,
            "role": "kiwoom_live_completed_minute_snapshot",
            "source": "kiwoom_rest_ka10080",
            "session": str(session.date()),
            "interval_minutes": int(args.interval_minutes),
            "timestamp_semantics": args.timestamp_semantics,
            "common_cutoff_kst": cutoff.isoformat(),
            "capture_started_at_kst": started_at.isoformat(),
            "capture_finished_at_kst": pd.Timestamp.now(tz=KST).isoformat(),
            "universe_tickers": len(universe),
            "active_lifecycle_tickers": active_count,
            "populated_tickers": len(output_records),
            "empty_tickers": sum(record["status"] == "empty" for record in records),
            "outside_lifecycle_tickers": sum(
                record["status"] == "outside_lifecycle" for record in records
            ),
            "errors": errors,
            "records": records,
            "outputs_sha256": canonical_json_sha256(output_records),
            "inputs": {
                "universe_manifest": str(universe_path),
                "universe_manifest_sha256": file_sha256(universe_path),
            },
            "code_provenance": collector_code_provenance(),
            "causality": {
                "common_cutoff_fixed_before_first_ticker_request": True,
                "only_completed_bars_retained": True,
                "in_progress_bar_excluded": True,
                "future_bars_absent": True,
                "labels_absent": True,
            },
            "promotion_eligible": False,
            "live_orders_allowed": False,
            "broker_order_calls_executed": 0,
        }
        manifest_path = temporary / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        temporary.replace(output_dir)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "pass",
                "session": str(session.date()),
                "cutoff_kst": cutoff.isoformat(),
                "populated_tickers": len(output_records),
                "live_orders_allowed": False,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
