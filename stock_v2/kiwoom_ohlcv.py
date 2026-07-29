from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping

import pandas as pd

from stock_v2.kiwoom_investor import parse_kiwoom_number
from stock_v2.ops.brokers import KiwoomRestBroker


KIWOOM_OHLCV_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "TradingValueM",
    "TurnoverPct",
    "PreviousChange",
    "PreviousChangeSign",
)

_FIELD_MAP = {
    "open_pric": "Open",
    "high_pric": "High",
    "low_pric": "Low",
    "cur_prc": "Close",
    "trde_qty": "Volume",
    "trde_prica": "TradingValueM",
    "trde_tern_rt": "TurnoverPct",
    "pred_pre": "PreviousChange",
    "pred_pre_sig": "PreviousChangeSign",
}

RawPageSink = Callable[[int, Mapping[str, Any], bool], None]


def _empty_ohlcv_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=KIWOOM_OHLCV_COLUMNS, dtype=float)
    frame.index = pd.DatetimeIndex([], name="Date")
    return frame


def parse_kiwoom_ohlcv_rows(rows: object) -> pd.DataFrame:
    """Normalize Kiwoom ka10081 rows without filling unavailable bars."""

    records: list[dict[str, object]] = []
    if not isinstance(rows, list):
        return _empty_ohlcv_frame()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        date = pd.to_datetime(str(row.get("dt", "")), format="%Y%m%d", errors="coerce")
        if pd.isna(date):
            continue
        record: dict[str, object] = {"Date": pd.Timestamp(date).normalize()}
        for source_name, target_name in _FIELD_MAP.items():
            value = parse_kiwoom_number(row.get(source_name))
            if target_name in {"Open", "High", "Low", "Close", "Volume", "TradingValueM"}:
                value = abs(value) if math.isfinite(value) else value
            record[target_name] = value
        records.append(record)
    if not records:
        return _empty_ohlcv_frame()
    frame = pd.DataFrame.from_records(records).set_index("Date").sort_index()
    return frame.reindex(columns=KIWOOM_OHLCV_COLUMNS).astype(float)


def _reject_conflicting_duplicates(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    duplicate_dates = frame.index[frame.index.duplicated(keep=False)].unique()
    for date in duplicate_dates:
        rows = frame.loc[[date]]
        first = rows.iloc[0]
        for _index, candidate in rows.iloc[1:].iterrows():
            equal = (first.eq(candidate) | (first.isna() & candidate.isna())).all()
            if not bool(equal):
                raise ValueError(f"Kiwoom ka10081 conflicting duplicate {ticker} {date.date()}")
    return frame.loc[~frame.index.duplicated(keep="first")].sort_index()


def fetch_kiwoom_ohlcv_history(
    broker: KiwoomRestBroker,
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    adjusted: bool,
    sleep_sec: float = 0.0,
    max_pages: int = 20,
    raw_page_sink: RawPageSink | None = None,
) -> pd.DataFrame:
    """Fetch bounded raw or back-adjusted daily bars through ka10081."""

    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if end_date < start_date:
        raise ValueError("end must not precede start")
    normalized_ticker = str(ticker).replace("A", "").strip().zfill(6)
    payload = {
        "stk_cd": normalized_ticker,
        "base_dt": end_date.strftime("%Y%m%d"),
        "upd_stkpc_tp": "1" if adjusted else "0",
    }
    pages: list[pd.DataFrame] = []
    continuation = False
    next_key: str | None = None
    seen_cursors: set[str] = set()
    previous_oldest: pd.Timestamp | None = None
    for page_index in range(1, max(1, int(max_pages)) + 1):
        data, has_more, returned_key = broker.post_readonly_with_continuation(
            "/api/dostk/chart",
            "ka10081",
            payload,
            continuation=continuation,
            next_key=next_key,
        )
        if raw_page_sink is not None:
            raw_page_sink(page_index, data, has_more)
        page = parse_kiwoom_ohlcv_rows(data.get("stk_dt_pole_chart_qry"))
        if page.empty:
            break
        oldest = pd.Timestamp(page.index.min())
        if previous_oldest is not None and oldest >= previous_oldest:
            raise RuntimeError(f"Kiwoom ka10081 pagination did not move backward for {normalized_ticker}")
        previous_oldest = oldest
        pages.append(page)
        if oldest <= start_date or not has_more:
            break
        if not returned_key:
            raise RuntimeError("Kiwoom ka10081 indicated continuation without a next-key")
        if returned_key in seen_cursors:
            raise RuntimeError(f"Kiwoom ka10081 repeated a cursor for {normalized_ticker}")
        seen_cursors.add(returned_key)
        continuation = True
        next_key = returned_key
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))
    else:
        raise RuntimeError(f"Kiwoom ka10081 exceeded max_pages={max_pages} for {normalized_ticker}")

    if not pages:
        return _empty_ohlcv_frame()
    frame = _reject_conflicting_duplicates(pd.concat(pages).sort_index(), normalized_ticker)
    return frame.loc[(frame.index >= start_date) & (frame.index <= end_date)].copy()


def trim_to_security_lifecycle(
    frame: pd.DataFrame,
    *,
    listing_date: object,
    delisting_date: object,
    release_start: str | pd.Timestamp,
    release_end: str | pd.Timestamp,
) -> tuple[pd.DataFrame, int]:
    """Remove rows that belong to a different lifecycle of a reused code."""

    if frame.empty:
        return frame.copy(), 0
    start = pd.Timestamp(release_start).normalize()
    end = pd.Timestamp(release_end).normalize()
    listed = pd.to_datetime(listing_date, errors="coerce")
    delisted = pd.to_datetime(delisting_date, errors="coerce")
    if not pd.isna(listed):
        start = max(start, pd.Timestamp(listed).normalize())
    if not pd.isna(delisted):
        end = min(end, pd.Timestamp(delisted).normalize())
    mask = (frame.index >= start) & (frame.index <= end)
    removed = int((~mask).sum())
    return frame.loc[mask].copy(), removed


def canonical_json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_immutable_raw_page(path: Path, envelope: Mapping[str, Any]) -> str:
    """Write one source response once, rejecting silent vendor revisions."""

    digest = canonical_json_sha256(envelope)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        identity_fields = (
            "source",
            "endpoint",
            "api_id",
            "run_id",
            "ticker",
            "basis",
            "request",
            "page_index",
            "response_sha256",
            "response",
        )
        if any(existing.get(field) != envelope.get(field) for field in identity_fields):
            raise RuntimeError(f"immutable raw page changed: {path}")
        return canonical_json_sha256(existing)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)
    return digest
