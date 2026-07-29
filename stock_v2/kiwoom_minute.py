from __future__ import annotations

import gzip
import json
import math
import time
from datetime import time as wall_time
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd

from stock_v2.kiwoom_investor import parse_kiwoom_number
from stock_v2.kiwoom_ohlcv import canonical_json_sha256
from stock_v2.ops.brokers import KiwoomRestBroker


KST = "Asia/Seoul"
KIWOOM_MINUTE_INTERVALS = (1, 3, 5, 10, 15, 30, 45, 60)
KIWOOM_MINUTE_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "CumulativeVolume",
    "PreviousChange",
    "PreviousChangeSign",
)

_FIELD_MAP = {
    "open_pric": "Open",
    "high_pric": "High",
    "low_pric": "Low",
    "cur_prc": "Close",
    "trde_qty": "Volume",
    "acc_trde_qty": "CumulativeVolume",
    "pred_pre": "PreviousChange",
    "pred_pre_sig": "PreviousChangeSign",
}

_ABSOLUTE_FIELDS = {
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "CumulativeVolume",
}

RawPageSink = Callable[[int, Mapping[str, Any], bool], None]


def _regular_session_mask(index: pd.DatetimeIndex) -> np.ndarray:
    local = index.tz_convert(KST)
    minutes = local.hour * 60 + local.minute
    return np.asarray((minutes >= 9 * 60) & (minutes <= 15 * 60 + 30), dtype=bool)


def normalize_kiwoom_ticker(ticker: object) -> str:
    normalized = str(ticker).replace("A", "").strip()
    if not normalized.isdigit() or len(normalized) > 6:
        raise ValueError(f"invalid KRX ticker: {ticker!r}")
    return normalized.zfill(6)


def _empty_minute_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=KIWOOM_MINUTE_COLUMNS, dtype=float)
    frame.index = pd.DatetimeIndex([], name="Timestamp", tz=KST)
    return frame


def parse_kiwoom_minute_rows(rows: object) -> pd.DataFrame:
    """Normalize Kiwoom ka10080 bars without filling missing observations."""

    records: list[dict[str, object]] = []
    if not isinstance(rows, list):
        return _empty_minute_frame()
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        timestamp = pd.to_datetime(
            str(row.get("cntr_tm", "")),
            format="%Y%m%d%H%M%S",
            errors="coerce",
        )
        if pd.isna(timestamp):
            continue
        timestamp = pd.Timestamp(timestamp).tz_localize(KST)
        record: dict[str, object] = {"Timestamp": timestamp}
        for source_name, target_name in _FIELD_MAP.items():
            value = parse_kiwoom_number(row.get(source_name))
            if target_name in _ABSOLUTE_FIELDS and math.isfinite(value):
                value = abs(value)
            record[target_name] = value
        records.append(record)
    if not records:
        return _empty_minute_frame()
    frame = pd.DataFrame.from_records(records).set_index("Timestamp").sort_index()
    return frame.reindex(columns=KIWOOM_MINUTE_COLUMNS).astype(float)


def _reject_conflicting_duplicates(frame: pd.DataFrame, ticker: str) -> pd.DataFrame:
    duplicate_times = frame.index[frame.index.duplicated(keep=False)].unique()
    for timestamp in duplicate_times:
        rows = frame.loc[[timestamp]]
        first = rows.iloc[0]
        for _index, candidate in rows.iloc[1:].iterrows():
            equal = (first.eq(candidate) | (first.isna() & candidate.isna())).all()
            if not bool(equal):
                raise ValueError(
                    f"Kiwoom ka10080 conflicting duplicate {ticker} {timestamp.isoformat()}"
                )
    return frame.loc[~frame.index.duplicated(keep="first")].sort_index()


def audit_kiwoom_minute_frame(
    frame: pd.DataFrame,
    *,
    regular_session_only: bool = True,
) -> dict[str, object]:
    """Return structural diagnostics and reject bars unsafe for model training."""

    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("minute frame must use a DatetimeIndex")
    if frame.index.tz is None:
        raise ValueError("minute timestamps must be timezone-aware")
    if not frame.index.is_monotonic_increasing:
        raise ValueError("minute timestamps must be sorted")
    if frame.index.has_duplicates:
        raise ValueError("minute timestamps must be unique")
    missing_columns = [column for column in KIWOOM_MINUTE_COLUMNS if column not in frame]
    if missing_columns:
        raise ValueError(f"minute frame missing columns: {missing_columns}")
    if frame.empty:
        return {
            "rows": 0,
            "sessions": 0,
            "first_timestamp": None,
            "last_timestamp": None,
            "min_bars_per_session": 0,
            "median_bars_per_session": 0.0,
            "max_bars_per_session": 0,
            "cumulative_volume_coverage": 0.0,
        }

    local_index = frame.index.tz_convert(KST)
    if regular_session_only:
        outside_session = [
            timestamp
            for timestamp in local_index
            if not wall_time(9, 0) <= timestamp.time() <= wall_time(15, 30)
        ]
        if outside_session:
            raise ValueError(
                "minute frame contains bars outside the KRX regular session: "
                f"{outside_session[0].isoformat()}"
            )

    core_columns = ["Open", "High", "Low", "Close", "Volume"]
    core = frame[core_columns].to_numpy(dtype=float)
    if not np.isfinite(core).all():
        raise ValueError("minute frame contains non-finite core OHLCV values")
    if (frame[["Open", "High", "Low", "Close"]].to_numpy(dtype=float) <= 0).any():
        raise ValueError("minute frame contains non-positive prices")
    if (frame["Volume"].to_numpy(dtype=float) < 0).any():
        raise ValueError("minute frame contains negative volume")

    open_values = frame["Open"].to_numpy(dtype=float)
    high_values = frame["High"].to_numpy(dtype=float)
    low_values = frame["Low"].to_numpy(dtype=float)
    close_values = frame["Close"].to_numpy(dtype=float)
    if (high_values < np.maximum(open_values, close_values)).any():
        raise ValueError("minute frame violates high >= max(open, close)")
    if (low_values > np.minimum(open_values, close_values)).any():
        raise ValueError("minute frame violates low <= min(open, close)")
    if (high_values < low_values).any():
        raise ValueError("minute frame violates high >= low")

    cumulative = frame["CumulativeVolume"]
    finite_cumulative = cumulative[np.isfinite(cumulative.to_numpy(dtype=float))]
    if (finite_cumulative < 0).any():
        raise ValueError("minute frame contains negative cumulative volume")
    session_keys = pd.Index(local_index.date)
    if len(finite_cumulative):
        finite_frame = frame.loc[finite_cumulative.index]
        finite_keys = pd.Index(finite_frame.index.tz_convert(KST).date)
        for _session, group in finite_frame.groupby(finite_keys, sort=True):
            if (group["CumulativeVolume"].diff().dropna() < 0).any():
                raise ValueError("cumulative volume decreases within a session")

    bars_per_session = frame.groupby(session_keys, sort=True).size()
    return {
        "rows": int(len(frame)),
        "sessions": int(len(bars_per_session)),
        "first_timestamp": local_index.min().isoformat(),
        "last_timestamp": local_index.max().isoformat(),
        "min_bars_per_session": int(bars_per_session.min()),
        "median_bars_per_session": float(bars_per_session.median()),
        "max_bars_per_session": int(bars_per_session.max()),
        "cumulative_volume_coverage": float(np.isfinite(cumulative).mean()),
    }


def fetch_kiwoom_minute_history(
    broker: KiwoomRestBroker,
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    interval_minutes: int,
    adjusted: bool,
    sleep_sec: float = 0.0,
    max_pages: int = 10_000,
    raw_page_sink: RawPageSink | None = None,
    regular_session_only: bool = True,
) -> pd.DataFrame:
    """Fetch a bounded ka10080 history using backward cursor pagination."""

    interval = int(interval_minutes)
    if interval not in KIWOOM_MINUTE_INTERVALS:
        raise ValueError(
            f"interval_minutes must be one of {KIWOOM_MINUTE_INTERVALS}, got {interval}"
        )
    if int(max_pages) <= 0:
        raise ValueError("max_pages must be positive")
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if end_date < start_date:
        raise ValueError("end must not precede start")
    ticker = normalize_kiwoom_ticker(ticker)
    payload = {
        "stk_cd": ticker,
        "tic_scope": str(interval),
        "upd_stkpc_tp": "1" if adjusted else "0",
        "base_dt": end_date.strftime("%Y%m%d"),
    }

    pages: list[pd.DataFrame] = []
    continuation = False
    next_key: str | None = None
    seen_cursors: set[str] = set()
    previous_oldest: pd.Timestamp | None = None
    for page_index in range(1, int(max_pages) + 1):
        data, has_more, returned_key = broker.post_readonly_with_continuation(
            "/api/dostk/chart",
            "ka10080",
            payload,
            continuation=continuation,
            next_key=next_key,
        )
        if raw_page_sink is not None:
            raw_page_sink(page_index, data, has_more)
        page = parse_kiwoom_minute_rows(data.get("stk_min_pole_chart_qry"))
        if page.empty:
            break
        oldest = pd.Timestamp(page.index.min())
        if previous_oldest is not None and oldest >= previous_oldest:
            raise RuntimeError(
                f"Kiwoom ka10080 pagination did not move backward for {ticker}"
            )
        previous_oldest = oldest
        pages.append(page)
        if oldest.tz_convert(KST).tz_localize(None).normalize() <= start_date:
            break
        if not has_more:
            break
        if not returned_key:
            raise RuntimeError("Kiwoom ka10080 indicated continuation without a next-key")
        if returned_key in seen_cursors:
            raise RuntimeError(f"Kiwoom ka10080 repeated a cursor for {ticker}")
        seen_cursors.add(returned_key)
        continuation = True
        next_key = returned_key
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))
    else:
        raise RuntimeError(f"Kiwoom ka10080 exceeded max_pages={max_pages} for {ticker}")

    if not pages:
        return _empty_minute_frame()
    frame = _reject_conflicting_duplicates(pd.concat(pages).sort_index(), ticker)
    local_dates = frame.index.tz_convert(KST).tz_localize(None).normalize()
    frame = frame.loc[(local_dates >= start_date) & (local_dates <= end_date)].copy()
    if regular_session_only and not frame.empty:
        # ka10080 can return after-hours bars (for example 15:35 for a 5-minute
        # request) in the same immutable response page. Preserve those rows in
        # the raw envelope, but never admit them to the regular-session cache.
        frame = frame.loc[_regular_session_mask(frame.index)].copy()
    audit_kiwoom_minute_frame(frame, regular_session_only=regular_session_only)
    return frame


def write_immutable_gzip_json(path: Path, envelope: Mapping[str, Any]) -> str:
    """Persist one raw API page once, with deterministic gzip encoding."""

    digest = canonical_json_sha256(envelope)
    if path.exists():
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            existing = json.load(handle)
        identity_fields = (
            "source",
            "endpoint",
            "api_id",
            "run_id",
            "ticker",
            "interval_minutes",
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
    temporary = Path(str(path) + ".tmp")
    encoded = (
        json.dumps(envelope, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode("utf-8")
    with temporary.open("wb") as raw_handle:
        with gzip.GzipFile(fileobj=raw_handle, mode="wb", mtime=0) as gzip_handle:
            gzip_handle.write(encoded)
    temporary.replace(path)
    return digest
