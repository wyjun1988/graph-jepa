from __future__ import annotations

from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd

from stock_v2.kiwoom_minute import (
    KST,
    audit_kiwoom_minute_frame,
    normalize_kiwoom_ticker,
)
from stock_v2.kiwoom_ohlcv import canonical_json_sha256


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_minute_frame(path: Path) -> pd.DataFrame:
    if path.suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif path.name.endswith(".csv.gz"):
        frame = pd.read_csv(path, index_col=0, parse_dates=True)
    else:
        raise ValueError(f"unsupported minute output format: {path}")
    frame.index = pd.DatetimeIndex(pd.to_datetime(frame.index))
    frame.index.name = "Timestamp"
    return frame


def _effective_lifecycle(
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
    return None if end < start else (start, end)


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
        "history_start_covered": 0 <= start_gap <= 10,
        "history_end_covered": 0 <= end_gap <= 10,
    }


def audit_kiwoom_minute_collection(
    *,
    coverage_path: Path,
    universe_manifest_path: Path,
    repository_root: Path,
    raw_cache_dir: Path,
    run_id: str,
    requested_start: str,
    requested_end: str,
    interval_minutes: int,
    basis: str,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []

    def fail(ticker: str, check: str, detail: object) -> None:
        failures.append(
            {"ticker": ticker, "check": check, "detail": str(detail)[:1000]}
        )

    universe_payload = json.loads(universe_manifest_path.read_text(encoding="utf-8"))
    universe_rows = universe_payload.get("universe")
    if not isinstance(universe_rows, list) or not universe_rows:
        raise ValueError("universe manifest must contain a non-empty universe list")
    universe: dict[str, dict[str, Any]] = {}
    for row in universe_rows:
        ticker = normalize_kiwoom_ticker(row.get("ticker", ""))
        if ticker in universe:
            raise ValueError(f"duplicate universe ticker: {ticker}")
        universe[ticker] = dict(row, ticker=ticker)

    coverage_rows = [
        json.loads(line)
        for line in coverage_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    selected: dict[str, dict[str, Any]] = {}
    for row in coverage_rows:
        if (
            str(row.get("run_id")) == str(run_id)
            and str(row.get("requested_start")) == str(requested_start)
            and str(row.get("requested_end")) == str(requested_end)
            and int(row.get("interval_minutes", -1)) == int(interval_minutes)
            and str(row.get("basis")) == str(basis)
        ):
            ticker = normalize_kiwoom_ticker(row.get("ticker", ""))
            if ticker in selected:
                fail(ticker, "coverage_unique", "duplicate matching coverage record")
            selected[ticker] = row

    requested_start_ts = pd.Timestamp(requested_start).normalize()
    requested_end_ts = pd.Timestamp(requested_end).normalize()
    status_counts: Counter[str] = Counter()
    output_rows = 0
    output_bytes = 0
    output_files_verified = 0
    raw_pages_verified = 0
    reused_records = 0
    partial_tickers: list[str] = []
    empty_tickers: list[str] = []
    outside_lifecycle_tickers: list[str] = []

    for ticker, security in universe.items():
        record = selected.get(ticker)
        if record is None:
            fail(ticker, "coverage_present", "missing exact run/range coverage record")
            continue
        status = str(record.get("status"))
        status_counts[status] += 1
        bounded = _effective_lifecycle(security, requested_start_ts, requested_end_ts)
        if bounded is None:
            outside_lifecycle_tickers.append(ticker)
            if status != "outside_lifecycle":
                fail(ticker, "outside_lifecycle_status", status)
            continue
        effective_start, effective_end = bounded
        if str(record.get("effective_start")) != str(effective_start.date()):
            fail(ticker, "effective_start", record.get("effective_start"))
        if str(record.get("effective_end")) != str(effective_end.date()):
            fail(ticker, "effective_end", record.get("effective_end"))
        if status == "error":
            fail(ticker, "collection_status", record.get("error"))
            continue
        if status not in {"ok", "partial", "empty"}:
            fail(ticker, "collection_status", status)
            continue
        if status == "partial":
            partial_tickers.append(ticker)
        if status == "empty":
            empty_tickers.append(ticker)

        frame = pd.DataFrame()
        output_value = record.get("output")
        if status in {"ok", "partial"}:
            if not output_value or not record.get("output_sha256"):
                fail(ticker, "output_reference", "missing path or checksum")
                continue
            output_path = Path(str(output_value))
            if not output_path.is_absolute():
                output_path = repository_root / output_path
            if not output_path.is_file():
                fail(ticker, "output_exists", output_path)
                continue
            actual_output_sha = file_sha256(output_path)
            if actual_output_sha != record.get("output_sha256"):
                fail(ticker, "output_sha256", actual_output_sha)
                continue
            if record.get("reused_from_run_id"):
                reused_records += 1
                if record.get("reuse_verified_sha256") != actual_output_sha:
                    fail(ticker, "reuse_verified_sha256", record.get("reuse_verified_sha256"))
            try:
                frame = _read_minute_frame(output_path)
                recomputed_audit = audit_kiwoom_minute_frame(
                    frame, regular_session_only=True
                )
            except Exception as exc:
                fail(ticker, "output_frame_audit", f"{type(exc).__name__}: {exc}")
                continue
            if recomputed_audit != record.get("audit"):
                fail(ticker, "recorded_frame_audit", "recomputed audit differs")
            boundary = _coverage_boundary(frame, effective_start, effective_end)
            for name, expected in boundary.items():
                if record.get(name) != expected:
                    fail(ticker, f"boundary_{name}", record.get(name))
            output_rows += len(frame)
            output_bytes += output_path.stat().st_size
            output_files_verified += 1
        else:
            if output_value is not None or record.get("output_sha256") is not None:
                fail(ticker, "empty_output", "empty record must not reference output")
            expected_empty_audit = {
                "rows": 0,
                "sessions": 0,
                "first_timestamp": None,
                "last_timestamp": None,
                "min_bars_per_session": 0,
                "median_bars_per_session": 0.0,
                "max_bars_per_session": 0,
                "cumulative_volume_coverage": 0.0,
            }
            if record.get("audit") != expected_empty_audit:
                fail(ticker, "recorded_frame_audit", "empty audit differs")
            boundary = _coverage_boundary(frame, effective_start, effective_end)
            for name, expected in boundary.items():
                if record.get(name) != expected:
                    fail(ticker, f"boundary_{name}", record.get(name))

        page_digests = record.get("raw_page_envelope_sha256")
        page_count = int(record.get("raw_page_count", -1))
        if not isinstance(page_digests, list) or len(page_digests) != page_count:
            fail(ticker, "raw_page_digest_count", page_count)
            continue
        source_run_id = str(record.get("reused_from_run_id") or run_id)
        page_dir = (
            raw_cache_dir
            / source_run_id
            / f"{int(interval_minutes)}min"
            / str(basis)
            / ticker
        )
        page_paths = sorted(page_dir.glob("page_*.json.gz"))
        if len(page_paths) != page_count:
            fail(
                ticker,
                "raw_page_file_count",
                f"expected={page_count} actual={len(page_paths)} source_run={source_run_id}",
            )
            continue
        for page_index, (page_path, expected_digest) in enumerate(
            zip(page_paths, page_digests), start=1
        ):
            try:
                with gzip.open(page_path, "rt", encoding="utf-8") as handle:
                    envelope = json.load(handle)
                actual_digest = canonical_json_sha256(envelope)
                if actual_digest != expected_digest:
                    fail(ticker, "raw_page_envelope_sha256", page_path.name)
                expected_fields = {
                    "source": "kiwoom_rest",
                    "endpoint": "/api/dostk/chart",
                    "api_id": "ka10080",
                    "run_id": source_run_id,
                    "ticker": ticker,
                    "interval_minutes": int(interval_minutes),
                    "basis": str(basis),
                    "page_index": page_index,
                }
                for name, expected in expected_fields.items():
                    if envelope.get(name) != expected:
                        fail(ticker, f"raw_page_{name}", page_path.name)
                request = envelope.get("request")
                expected_request = {
                    "stk_cd": ticker,
                    "tic_scope": str(interval_minutes),
                    "upd_stkpc_tp": "1" if basis == "adjusted" else "0",
                    "base_dt": effective_end.strftime("%Y%m%d"),
                }
                if request != expected_request:
                    fail(ticker, "raw_page_request", page_path.name)
                response = envelope.get("response")
                if not isinstance(response, dict) or canonical_json_sha256(response) != envelope.get(
                    "response_sha256"
                ):
                    fail(ticker, "raw_page_response_sha256", page_path.name)
                raw_pages_verified += 1
            except Exception as exc:
                fail(ticker, "raw_page_read", f"{page_path.name}: {type(exc).__name__}: {exc}")

    unexpected = sorted(set(selected) - set(universe))
    for ticker in unexpected:
        fail(ticker, "coverage_universe_membership", "ticker absent from universe")
    missing = sorted(set(universe) - set(selected))
    integrity_gate_passed = not failures and not missing and not unexpected
    complete_history_gate_passed = integrity_gate_passed and not (
        partial_tickers or empty_tickers or status_counts.get("error", 0)
    )
    return {
        "schema_version": 1,
        "audit_contract": "kiwoom_minute_collection_integrity_v1",
        "run_id": run_id,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "interval_minutes": int(interval_minutes),
        "basis": basis,
        "universe_tickers": len(universe),
        "coverage_records": len(selected),
        "status_counts": dict(sorted(status_counts.items())),
        "reused_records": reused_records,
        "output_files_verified": output_files_verified,
        "output_rows": output_rows,
        "output_bytes": output_bytes,
        "raw_pages_verified": raw_pages_verified,
        "partial_tickers": partial_tickers,
        "empty_tickers": empty_tickers,
        "outside_lifecycle_tickers": outside_lifecycle_tickers,
        "missing_tickers": missing,
        "unexpected_tickers": unexpected,
        "failures": failures,
        "inputs": {
            "coverage": str(coverage_path),
            "coverage_sha256": file_sha256(coverage_path),
            "universe_manifest": str(universe_manifest_path),
            "universe_manifest_sha256": file_sha256(universe_manifest_path),
            "raw_cache_dir": str(raw_cache_dir),
        },
        "integrity_gate_passed": integrity_gate_passed,
        "complete_history_gate_passed": complete_history_gate_passed,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
