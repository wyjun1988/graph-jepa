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
from zoneinfo import ZoneInfo

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.kiwoom_minute import (
    KST,
    KIWOOM_MINUTE_INTERVALS,
    audit_kiwoom_minute_frame,
    fetch_kiwoom_minute_history,
    normalize_kiwoom_ticker,
    write_immutable_gzip_json,
)
from stock_v2.kiwoom_ohlcv import canonical_json_sha256
from stock_v2.ops.brokers import KiwoomRestBroker
from stock_v2.ops.config import KiwoomConfig


SOURCE_URL = (
    "https://openapi.kiwoom.com/guide/apiguide?jobTpCode=07&apiId=ka10080"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Collect immutable read-only Kiwoom ka10080 histories for PIT intraday sensing."
        )
    )
    parser.add_argument("--universe-manifest", required=True)
    parser.add_argument("--start", required=True)
    parser.add_argument("--end", required=True)
    parser.add_argument(
        "--interval-minutes",
        type=int,
        choices=KIWOOM_MINUTE_INTERVALS,
        default=5,
    )
    parser.add_argument("--basis", choices=["raw", "adjusted"], default="raw")
    parser.add_argument("--output-format", choices=["parquet", "csv.gz"], default="parquet")
    parser.add_argument("--cache-dir", default="data/kiwoom_minute_cache")
    parser.add_argument("--raw-cache-dir", default="data/raw/kiwoom_minute")
    parser.add_argument(
        "--coverage-output",
        default="data/kiwoom_minute_cache/coverage.jsonl",
    )
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--env-file", default="../stock/.env")
    parser.add_argument("--server", choices=["real", "mock"], default="real")
    parser.add_argument("--sleep-sec", type=float, default=0.25)
    parser.add_argument("--max-pages", type=int, default=10_000)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--reuse-verified-existing",
        action="store_true",
        help=(
            "Reuse an existing successful file for the same ticker/range/basis "
            "after checksum and lifecycle-bound verification, even across run IDs."
        ),
    )
    parser.add_argument("--max-tickers", type=int, default=0)
    parser.add_argument("--include-outside-session", action="store_true")
    return parser.parse_args()


def load_universe(path: Path, max_tickers: int) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("universe") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise ValueError("universe manifest must contain a non-empty universe list")
    selected = rows[: max_tickers or None]
    normalized: list[dict[str, Any]] = []
    for row in selected:
        if not isinstance(row, dict):
            raise ValueError("universe row must be an object")
        normalized.append(dict(row, ticker=normalize_kiwoom_ticker(row.get("ticker", ""))))
    tickers = [row["ticker"] for row in normalized]
    if len(tickers) != len(set(tickers)):
        raise ValueError("universe manifest contains duplicate tickers")
    return normalized


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_verified_reuse_record(
    prior: dict[str, Any],
    *,
    output: Path,
    run_id: str,
    effective_start: pd.Timestamp,
    effective_end: pd.Timestamp,
    reused_at_utc: str,
) -> dict[str, Any] | None:
    if prior.get("status") != "ok" or not prior.get("output_sha256"):
        return None
    prior_path = Path(str(prior.get("output") or ""))
    if not prior_path.is_absolute():
        prior_path = ROOT / prior_path
    expected_output = output if output.is_absolute() else ROOT / output
    if prior_path.resolve() != expected_output.resolve() or not expected_output.is_file():
        return None
    if str(prior.get("effective_start")) != str(effective_start.date()):
        return None
    if str(prior.get("effective_end")) != str(effective_end.date()):
        return None
    actual_sha256 = file_sha256(expected_output)
    if actual_sha256 != prior.get("output_sha256"):
        return None
    reused = dict(prior)
    reused["run_id"] = str(run_id)
    reused["reused_from_run_id"] = str(prior.get("run_id"))
    reused["reused_at_utc"] = str(reused_at_utc)
    reused["reuse_verified_sha256"] = actual_sha256
    return reused


def coverage_key(record: dict[str, Any]) -> tuple[str, int, str, str, str]:
    return (
        str(record.get("ticker")),
        int(record.get("interval_minutes")),
        str(record.get("basis")),
        str(record.get("requested_start")),
        str(record.get("requested_end")),
    )


def load_coverage(path: Path) -> dict[tuple[str, int, str, str, str], dict[str, Any]]:
    records: dict[tuple[str, int, str, str, str], dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            records[coverage_key(record)] = record
    return records


def write_coverage(
    path: Path,
    records: dict[tuple[str, int, str, str, str], dict[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for key in sorted(records):
            handle.write(json.dumps(records[key], ensure_ascii=False, sort_keys=True) + "\n")
    temporary.replace(path)


def _bounded_lifecycle(
    security: dict[str, Any],
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
) -> tuple[pd.Timestamp, pd.Timestamp] | None:
    start = requested_start
    end = requested_end
    listing = pd.to_datetime(security.get("listing_date"), errors="coerce")
    delisting = pd.to_datetime(security.get("delisting_date"), errors="coerce")
    if not pd.isna(listing):
        start = max(start, pd.Timestamp(listing).normalize())
    if not pd.isna(delisting):
        end = min(end, pd.Timestamp(delisting).normalize())
    if end < start:
        return None
    return start, end


def _write_frame(frame: pd.DataFrame, path: Path, output_format: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(str(path) + ".tmp")
    if output_format == "parquet":
        frame.to_parquet(temporary, compression="zstd", index=True)
    else:
        frame.to_csv(temporary, compression="gzip", index=True)
    temporary.replace(path)


def _coverage_boundary(
    frame: pd.DataFrame,
    effective_start: pd.Timestamp,
    effective_end: pd.Timestamp,
) -> dict[str, object]:
    if frame.empty:
        return {
            "first_date": None,
            "last_date": None,
            "start_gap_calendar_days": None,
            "end_gap_calendar_days": None,
            "history_start_covered": False,
            "history_end_covered": False,
        }
    local_dates = frame.index.tz_convert(KST).tz_localize(None).normalize()
    first = pd.Timestamp(local_dates.min())
    last = pd.Timestamp(local_dates.max())
    start_gap = int((first - effective_start).days)
    end_gap = int((effective_end - last).days)
    return {
        "first_date": str(first.date()),
        "last_date": str(last.date()),
        "start_gap_calendar_days": start_gap,
        "end_gap_calendar_days": end_gap,
        # Calendar-day tolerance covers long weekends while still detecting retention truncation.
        "history_start_covered": 0 <= start_gap <= 10,
        "history_end_covered": 0 <= end_gap <= 10,
    }


def main() -> int:
    args = parse_args()
    if not re.fullmatch(r"[A-Za-z0-9._-]+", args.run_id):
        raise ValueError("run-id may contain only letters, digits, dot, underscore, and dash")
    if args.sleep_sec < 0:
        raise ValueError("sleep-sec must be non-negative")
    if args.server == "real" and args.sleep_sec < 0.20:
        raise ValueError(
            "real Kiwoom read-only collection requires sleep-sec >= 0.20 (5 TR/sec limit)"
        )
    if args.max_pages <= 0:
        raise ValueError("max-pages must be positive")

    requested_start = pd.Timestamp(args.start).normalize()
    requested_end = pd.Timestamp(args.end).normalize()
    if requested_end < requested_start:
        raise ValueError("end must not precede start")
    today_kst = pd.Timestamp(datetime.now(ZoneInfo(KST)).date())
    if requested_end >= today_kst:
        raise ValueError(
            "historical training collection requires end before today in Asia/Seoul"
        )

    universe_path = Path(args.universe_manifest)
    universe = load_universe(universe_path, max(0, args.max_tickers))
    cache_root = Path(args.cache_dir)
    raw_root = Path(args.raw_cache_dir) / args.run_id
    coverage_path = Path(args.coverage_output)
    coverage = load_coverage(coverage_path)
    adjusted = args.basis == "adjusted"
    extension = ".parquet" if args.output_format == "parquet" else ".csv.gz"

    # This collector is structurally read-only: dry_run is fixed and no order API is accepted.
    broker = KiwoomRestBroker(
        KiwoomConfig(server=args.server, env_file=args.env_file, timeout_sec=30.0),
        dry_run=True,
    )
    if not broker.authenticate():
        detail = f": {broker.last_auth_error}" if broker.last_auth_error else ""
        raise RuntimeError(f"Kiwoom authentication failed for read-only minute collection{detail}")

    counts = {
        "ok": 0,
        "partial": 0,
        "empty": 0,
        "outside_lifecycle": 0,
        "errors": 0,
        "skipped": 0,
        "reused": 0,
    }
    for ticker_index, security in enumerate(universe, start=1):
        ticker = security["ticker"]
        bounded = _bounded_lifecycle(security, requested_start, requested_end)
        record_key = (ticker, args.interval_minutes, args.basis, args.start, args.end)
        if bounded is None:
            record = {
                "schema_version": 1,
                "ticker": ticker,
                "interval_minutes": args.interval_minutes,
                "basis": args.basis,
                "requested_start": args.start,
                "requested_end": args.end,
                "run_id": args.run_id,
                "status": "outside_lifecycle",
            }
            coverage[record_key] = record
            write_coverage(coverage_path, coverage)
            counts["outside_lifecycle"] += 1
            continue
        effective_start, effective_end = bounded
        output = (
            cache_root
            / f"{args.interval_minutes}min"
            / args.basis
            / f"{ticker}_{args.start.replace('-', '')}_{args.end.replace('-', '')}{extension}"
        )
        prior = coverage.get(record_key, {})
        if (
            args.resume
            and prior.get("status") == "ok"
            and prior.get("run_id") == args.run_id
            and output.exists()
            and prior.get("output_sha256") == file_sha256(output)
        ):
            counts["skipped"] += 1
            continue
        if args.reuse_verified_existing:
            reused = build_verified_reuse_record(
                prior,
                output=output,
                run_id=args.run_id,
                effective_start=effective_start,
                effective_end=effective_end,
                reused_at_utc=datetime.now(tz=timezone.utc).isoformat(),
            )
            if reused is not None:
                coverage[record_key] = reused
                write_coverage(coverage_path, coverage)
                counts["reused"] += 1
                print(
                    f"ticker={ticker} status=reused "
                    f"from_run={reused['reused_from_run_id']} "
                    f"complete={ticker_index}/{len(universe)}",
                    flush=True,
                )
                continue

        page_digests: list[str] = []
        collected_at = datetime.now(tz=timezone.utc).isoformat()
        request_payload = {
            "stk_cd": ticker,
            "tic_scope": str(args.interval_minutes),
            "upd_stkpc_tp": "1" if adjusted else "0",
            "base_dt": effective_end.strftime("%Y%m%d"),
        }

        def save_page(page_index: int, response: Mapping[str, Any], has_more: bool) -> None:
            response_sha256 = canonical_json_sha256(response)
            envelope = {
                "schema_version": 1,
                "source": "kiwoom_rest",
                "source_url": SOURCE_URL,
                "endpoint": "/api/dostk/chart",
                "api_id": "ka10080",
                "run_id": args.run_id,
                "ticker": ticker,
                "interval_minutes": args.interval_minutes,
                "basis": args.basis,
                "request": request_payload,
                "page_index": page_index,
                "has_more": bool(has_more),
                "retrieved_at_utc": collected_at,
                "response_sha256": response_sha256,
                "response": response,
            }
            raw_path = (
                raw_root
                / f"{args.interval_minutes}min"
                / args.basis
                / ticker
                / f"page_{page_index:06d}.json.gz"
            )
            page_digests.append(write_immutable_gzip_json(raw_path, envelope))

        try:
            frame = fetch_kiwoom_minute_history(
                broker,
                ticker,
                effective_start,
                effective_end,
                interval_minutes=args.interval_minutes,
                adjusted=adjusted,
                sleep_sec=args.sleep_sec,
                max_pages=args.max_pages,
                raw_page_sink=save_page,
                regular_session_only=not args.include_outside_session,
            )
            audit = audit_kiwoom_minute_frame(
                frame,
                regular_session_only=not args.include_outside_session,
            )
            boundary = _coverage_boundary(frame, effective_start, effective_end)
            if frame.empty:
                status = "empty"
                output_sha256 = None
            else:
                _write_frame(frame, output, args.output_format)
                output_sha256 = file_sha256(output)
                status = (
                    "ok"
                    if boundary["history_start_covered"]
                    and boundary["history_end_covered"]
                    else "partial"
                )
            counts[status] += 1
            record = {
                "schema_version": 1,
                "ticker": ticker,
                "name": security.get("name"),
                "market": security.get("market"),
                "listing_date": security.get("listing_date"),
                "delisting_date": security.get("delisting_date"),
                "interval_minutes": args.interval_minutes,
                "basis": args.basis,
                "run_id": args.run_id,
                "status": status,
                "requested_start": args.start,
                "requested_end": args.end,
                "effective_start": str(effective_start.date()),
                "effective_end": str(effective_end.date()),
                **boundary,
                "audit": audit,
                "raw_page_count": len(page_digests),
                "raw_page_envelope_sha256": page_digests,
                "output": str(output) if len(frame) else None,
                "output_format": args.output_format,
                "output_sha256": output_sha256,
                "collected_at_utc": collected_at,
                "error": None,
            }
            print(
                f"ticker={ticker} status={status} rows={len(frame)} "
                f"pages={len(page_digests)} complete={ticker_index}/{len(universe)}",
                flush=True,
            )
        except Exception as exc:
            counts["errors"] += 1
            record = {
                "schema_version": 1,
                "ticker": ticker,
                "interval_minutes": args.interval_minutes,
                "basis": args.basis,
                "run_id": args.run_id,
                "status": "error",
                "requested_start": args.start,
                "requested_end": args.end,
                "effective_start": str(effective_start.date()),
                "effective_end": str(effective_end.date()),
                "raw_page_count": len(page_digests),
                "raw_page_envelope_sha256": page_digests,
                "collected_at_utc": collected_at,
                "error": f"{type(exc).__name__}: {str(exc)[:500]}",
            }
            print(f"ticker={ticker} error={record['error']}", flush=True)
        coverage[record_key] = record
        write_coverage(coverage_path, coverage)
        if args.sleep_sec > 0:
            time.sleep(float(args.sleep_sec))

    summary = {
        "schema_version": 1,
        "source": "kiwoom_rest",
        "api_id": "ka10080",
        "universe_manifest": str(universe_path),
        "universe_sha256": file_sha256(universe_path),
        "requested_start": args.start,
        "requested_end": args.end,
        "interval_minutes": args.interval_minutes,
        "basis": args.basis,
        "run_id": args.run_id,
        "securities": len(universe),
        **counts,
    }
    summary_path = coverage_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), flush=True)
    return 0 if counts["errors"] == 0 and counts["partial"] == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
