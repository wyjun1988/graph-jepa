from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_v2.kiwoom_minute import normalize_kiwoom_ticker
from stock_v2.kiwoom_ohlcv import (
    KIWOOM_OHLCV_COLUMNS,
    canonical_json_sha256,
    parse_kiwoom_ohlcv_rows,
    trim_to_security_lifecycle,
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _deduplicate_pages(frames: list[pd.DataFrame], ticker: str) -> pd.DataFrame:
    if not frames:
        return pd.DataFrame(
            columns=KIWOOM_OHLCV_COLUMNS,
            index=pd.DatetimeIndex([], name="Date"),
            dtype=float,
        )
    frame = pd.concat(frames).sort_index()
    for date in frame.index[frame.index.duplicated(keep=False)].unique():
        rows = frame.loc[[date]]
        first = rows.iloc[0]
        for _index, candidate in rows.iloc[1:].iterrows():
            equal = (first.eq(candidate) | (first.isna() & candidate.isna())).all()
            if not bool(equal):
                raise ValueError(f"conflicting raw-page duplicate {ticker} {date.date()}")
    return frame.loc[~frame.index.duplicated(keep="first")].sort_index()


def _audit_ohlcv_frame(frame: pd.DataFrame) -> None:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("daily frame must use a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("daily dates must be sorted and unique")
    missing = [name for name in KIWOOM_OHLCV_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"daily frame missing columns: {missing}")
    if frame.empty:
        return
    core = frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(float)
    if not np.isfinite(core).all():
        raise ValueError("daily frame has non-finite core OHLCV")
    if (frame[["Open", "High", "Low", "Close"]].to_numpy(float) <= 0).any():
        raise ValueError("daily frame has non-positive prices")
    if (frame["Volume"].to_numpy(float) < 0).any():
        raise ValueError("daily frame has negative volume")
    high = frame["High"].to_numpy(float)
    low = frame["Low"].to_numpy(float)
    open_ = frame["Open"].to_numpy(float)
    close = frame["Close"].to_numpy(float)
    if (high < np.maximum(open_, close)).any() or (low > np.minimum(open_, close)).any():
        raise ValueError("daily frame violates OHLC bounds")


def audit_kiwoom_ohlcv_collection(
    *,
    coverage_path: Path,
    universe_manifest_path: Path,
    repository_root: Path,
    raw_cache_dir: Path,
    run_id: str,
    requested_start: str,
    requested_end: str,
    basis: str,
) -> dict[str, Any]:
    failures: list[dict[str, str]] = []

    def fail(ticker: str, check: str, detail: object) -> None:
        failures.append(
            {"ticker": ticker, "check": check, "detail": str(detail)[:1000]}
        )

    universe_payload = json.loads(universe_manifest_path.read_text(encoding="utf-8"))
    rows = universe_payload.get("universe")
    if not isinstance(rows, list) or not rows:
        raise ValueError("universe manifest must contain a non-empty universe list")
    universe: dict[str, dict[str, Any]] = {}
    for row in rows:
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
    for record in coverage_rows:
        if str(record.get("run_id")) != str(run_id) or str(record.get("basis")) != str(
            basis
        ):
            continue
        ticker = normalize_kiwoom_ticker(record.get("ticker", ""))
        if ticker in selected:
            fail(ticker, "coverage_unique", "duplicate matching coverage record")
        selected[ticker] = record

    start = pd.Timestamp(requested_start).normalize()
    end = pd.Timestamp(requested_end).normalize()
    status_counts: Counter[str] = Counter()
    output_files_verified = 0
    output_rows = 0
    output_bytes = 0
    raw_pages_verified = 0
    raw_rows_reparsed = 0
    incomplete_history_tickers: list[str] = []

    for ticker, security in universe.items():
        record = selected.get(ticker)
        if record is None:
            fail(ticker, "coverage_present", "missing run/basis coverage record")
            continue
        status = str(record.get("status"))
        status_counts[status] += 1
        if status == "error":
            fail(ticker, "collection_status", record.get("error"))
            continue
        if status not in {"ok", "empty"}:
            fail(ticker, "collection_status", status)
            continue
        if record.get("price_basis") != (
            "back_adjusted" if basis == "adjusted" else "raw"
        ):
            fail(ticker, "price_basis", record.get("price_basis"))

        page_count = int(record.get("raw_page_count", -1))
        page_digests = record.get("raw_page_envelope_sha256")
        if not isinstance(page_digests, list) or len(page_digests) != page_count:
            fail(ticker, "raw_page_digest_count", page_count)
            continue
        page_dir = raw_cache_dir / run_id / basis / ticker
        page_paths = sorted(page_dir.glob("page_*.json"))
        if len(page_paths) != page_count:
            fail(
                ticker,
                "raw_page_file_count",
                f"expected={page_count} actual={len(page_paths)}",
            )
            continue
        page_frames: list[pd.DataFrame] = []
        for page_index, (page_path, expected_digest) in enumerate(
            zip(page_paths, page_digests), start=1
        ):
            try:
                envelope = json.loads(page_path.read_text(encoding="utf-8"))
                if canonical_json_sha256(envelope) != expected_digest:
                    fail(ticker, "raw_page_envelope_sha256", page_path.name)
                expected_fields = {
                    "source": "kiwoom_rest",
                    "endpoint": "/api/dostk/chart",
                    "api_id": "ka10081",
                    "run_id": run_id,
                    "ticker": ticker,
                    "basis": basis,
                    "page_index": page_index,
                }
                for name, expected in expected_fields.items():
                    if envelope.get(name) != expected:
                        fail(ticker, f"raw_page_{name}", page_path.name)
                expected_request = {
                    "stk_cd": ticker,
                    "base_dt": end.strftime("%Y%m%d"),
                    "upd_stkpc_tp": "1" if basis == "adjusted" else "0",
                }
                if envelope.get("request") != expected_request:
                    fail(ticker, "raw_page_request", page_path.name)
                response = envelope.get("response")
                if not isinstance(response, dict) or canonical_json_sha256(response) != envelope.get(
                    "response_sha256"
                ):
                    fail(ticker, "raw_page_response_sha256", page_path.name)
                else:
                    page_frames.append(
                        parse_kiwoom_ohlcv_rows(response.get("stk_dt_pole_chart_qry"))
                    )
                raw_pages_verified += 1
            except Exception as exc:
                fail(ticker, "raw_page_read", f"{page_path.name}: {type(exc).__name__}: {exc}")

        try:
            source = _deduplicate_pages(page_frames, ticker)
            source = source.loc[(source.index >= start) & (source.index <= end)].copy()
            raw_rows_reparsed += len(source)
            rebuilt, removed = trim_to_security_lifecycle(
                source,
                listing_date=security.get("listing_date"),
                delisting_date=security.get("delisting_date"),
                release_start=start,
                release_end=end,
            )
            _audit_ohlcv_frame(rebuilt)
        except Exception as exc:
            fail(ticker, "raw_page_rebuild", f"{type(exc).__name__}: {exc}")
            continue
        if int(record.get("source_rows", -1)) != len(source):
            fail(ticker, "source_rows", record.get("source_rows"))
        if int(record.get("outside_lifecycle_rows", -1)) != removed:
            fail(ticker, "outside_lifecycle_rows", record.get("outside_lifecycle_rows"))
        if int(record.get("rows", -1)) != len(rebuilt):
            fail(ticker, "rows", record.get("rows"))
        expected_first = str(rebuilt.index.min().date()) if len(rebuilt) else None
        expected_last = str(rebuilt.index.max().date()) if len(rebuilt) else None
        if record.get("first_date") != expected_first:
            fail(ticker, "first_date", record.get("first_date"))
        if record.get("last_date") != expected_last:
            fail(ticker, "last_date", record.get("last_date"))

        output_value = record.get("output")
        if status == "empty":
            if len(rebuilt) or record.get("output_sha256") is not None:
                fail(ticker, "empty_output", "empty record disagrees with rebuilt raw pages")
            if output_value:
                empty_path = Path(str(output_value))
                if not empty_path.is_absolute():
                    empty_path = repository_root / empty_path
                if empty_path.exists():
                    fail(ticker, "empty_output_exists", empty_path)
        else:
            if not output_value or not record.get("output_sha256"):
                fail(ticker, "output_reference", "missing path or checksum")
                continue
            output_path = Path(str(output_value))
            if not output_path.is_absolute():
                output_path = repository_root / output_path
            if not output_path.is_file():
                fail(ticker, "output_exists", output_path)
                continue
            actual_sha = file_sha256(output_path)
            if actual_sha != record.get("output_sha256"):
                fail(ticker, "output_sha256", actual_sha)
                continue
            try:
                output_frame = pd.read_csv(output_path, index_col="Date", parse_dates=True)
                output_frame.index = pd.DatetimeIndex(output_frame.index).normalize()
                output_frame = output_frame.reindex(columns=KIWOOM_OHLCV_COLUMNS).astype(float)
                _audit_ohlcv_frame(output_frame)
                pd.testing.assert_frame_equal(
                    output_frame,
                    rebuilt,
                    check_exact=False,
                    rtol=1e-12,
                    atol=1e-12,
                    check_freq=False,
                )
            except Exception as exc:
                fail(ticker, "output_rebuild_equality", f"{type(exc).__name__}: {exc}")
                continue
            output_files_verified += 1
            output_rows += len(output_frame)
            output_bytes += output_path.stat().st_size

        listed = pd.to_datetime(security.get("listing_date"), errors="coerce")
        delisted = pd.to_datetime(security.get("delisting_date"), errors="coerce")
        effective_start = max(start, pd.Timestamp(listed).normalize()) if not pd.isna(listed) else start
        effective_end = min(end, pd.Timestamp(delisted).normalize()) if not pd.isna(delisted) else end
        if effective_end >= effective_start:
            if rebuilt.empty:
                incomplete_history_tickers.append(ticker)
            else:
                start_gap = int((rebuilt.index.min() - effective_start).days)
                end_gap = int((effective_end - rebuilt.index.max()).days)
                if not (0 <= start_gap <= 10 and 0 <= end_gap <= 10):
                    incomplete_history_tickers.append(ticker)

    unexpected = sorted(set(selected) - set(universe))
    for ticker in unexpected:
        fail(ticker, "coverage_universe_membership", "ticker absent from universe")
    missing = sorted(set(universe) - set(selected))
    integrity_gate_passed = not failures and not missing and not unexpected
    return {
        "schema_version": 1,
        "audit_contract": "kiwoom_ohlcv_collection_raw_rebuild_v1",
        "run_id": run_id,
        "requested_start": requested_start,
        "requested_end": requested_end,
        "basis": basis,
        "universe_tickers": len(universe),
        "coverage_records": len(selected),
        "status_counts": dict(sorted(status_counts.items())),
        "output_files_verified": output_files_verified,
        "output_rows": output_rows,
        "output_bytes": output_bytes,
        "raw_pages_verified": raw_pages_verified,
        "raw_rows_reparsed": raw_rows_reparsed,
        "incomplete_history_tickers": sorted(set(incomplete_history_tickers)),
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
        "complete_history_gate_passed": integrity_gate_passed
        and not incomplete_history_tickers,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
