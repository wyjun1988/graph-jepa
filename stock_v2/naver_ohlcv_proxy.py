from __future__ import annotations

import hashlib
from io import StringIO
from pathlib import Path
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd


NAVER_DAILY_URL = (
    "https://fchart.stock.naver.com/sise.nhn?"
    "timeframe=day&count=6000&requestType=0&symbol="
)
NAVER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 Chrome/120 Safari/537.36"
    )
}
PROXY_COLUMNS = ("Open", "High", "Low", "Close", "Volume", "Change")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_naver_daily_xml(payload: bytes) -> tuple[dict[str, Any], pd.DataFrame]:
    """Parse one Naver chart response with an XML parser and strict row widths."""

    text = payload.decode("euc-kr", errors="strict")
    stripped = text.lstrip()
    if stripped.startswith("<?xml"):
        declaration_end = stripped.find("?>")
        if declaration_end < 0:
            raise ValueError("Naver chart XML declaration is unterminated")
        stripped = stripped[declaration_end + 2 :]
    root = ET.fromstring(stripped)
    chart = root.find(".//chartdata")
    if chart is None:
        raise ValueError("Naver chart response is missing chartdata")
    rows: list[list[str]] = []
    for item in chart.findall("item"):
        fields = str(item.attrib.get("data", "")).split("|")
        if len(fields) != 6:
            raise ValueError("Naver chart item must contain six pipe-delimited fields")
        rows.append(fields)
    if not rows:
        raise ValueError("Naver chart response contains no daily rows")
    frame = pd.DataFrame(
        rows,
        columns=("Date", "Open", "High", "Low", "Close", "Volume"),
    )
    frame["Date"] = pd.to_datetime(frame["Date"], format="%Y%m%d", errors="raise")
    if frame["Date"].duplicated().any():
        raise ValueError("Naver chart response contains duplicate dates")
    for column in ("Open", "High", "Low", "Close", "Volume"):
        frame[column] = pd.to_numeric(frame[column], errors="raise").astype(np.int64)
    frame = frame.set_index("Date").sort_index()
    frame.index = pd.DatetimeIndex(frame.index).normalize()
    frame.index.name = "Date"
    frame["Change"] = frame["Close"].pct_change(fill_method=None)
    metadata = {
        "symbol": str(chart.attrib.get("symbol", "")).zfill(6),
        "name": str(chart.attrib.get("name", "")),
        "declared_count": int(chart.attrib.get("count", len(frame))),
        "timeframe": str(chart.attrib.get("timeframe", "")),
        "origin_time": str(chart.attrib.get("origintime", "")),
    }
    if metadata["declared_count"] != len(frame):
        raise ValueError("Naver chart declared count does not match parsed rows")
    if metadata["timeframe"] != "day":
        raise ValueError("Naver chart timeframe is not daily")
    return metadata, frame[list(PROXY_COLUMNS)]


def trim_proxy_frame(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    listing_date: str | None,
    delisting_date: str | None,
) -> pd.DataFrame:
    lower = max(
        pd.Timestamp(start).normalize(),
        pd.Timestamp(listing_date).normalize() if listing_date else pd.Timestamp.min,
    )
    upper = min(
        pd.Timestamp(end).normalize(),
        pd.Timestamp(delisting_date).normalize() if delisting_date else pd.Timestamp.max,
    )
    if lower > upper:
        raise ValueError("security lifecycle does not overlap the requested proxy range")
    return frame.loc[(frame.index >= lower) & (frame.index <= upper)].copy()


def validate_proxy_frame(
    frame: pd.DataFrame,
    *,
    start: str,
    end: str,
    listing_date: str | None,
    delisting_date: str | None,
) -> dict[str, Any]:
    missing = [name for name in PROXY_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"proxy frame is missing columns: {missing}")
    if frame.empty:
        raise ValueError("proxy frame contains no lifecycle rows")
    if frame.index.has_duplicates or not frame.index.is_monotonic_increasing:
        raise ValueError("proxy frame dates must be unique and increasing")
    lower = max(
        pd.Timestamp(start).normalize(),
        pd.Timestamp(listing_date).normalize() if listing_date else pd.Timestamp.min,
    )
    upper = min(
        pd.Timestamp(end).normalize(),
        pd.Timestamp(delisting_date).normalize() if delisting_date else pd.Timestamp.max,
    )
    if frame.index.min() < lower or frame.index.max() > upper:
        raise ValueError("proxy frame exceeds the requested security lifecycle")
    prices = frame[["Open", "High", "Low", "Close"]].to_numpy(dtype=np.float64)
    volume = frame["Volume"].to_numpy(dtype=np.float64)
    if not np.isfinite(prices).all() or not np.isfinite(volume).all():
        raise ValueError("proxy frame contains non-finite OHLCV")
    if (prices < 0.0).any() or (volume < 0.0).any():
        raise ValueError("proxy frame contains negative OHLCV")
    observed = (prices > 0.0).all(axis=1)
    suspended = ~observed
    bound_violations = np.zeros(len(frame), dtype=bool)
    maximum_bound_violation_fraction = 0.0
    if observed.any():
        open_, high, low, close = prices[observed].T
        bound_gap = np.maximum.reduce(
            (
                np.maximum(open_, close) - high,
                low - np.minimum(open_, close),
                np.zeros_like(close),
            )
        )
        observed_violations = bound_gap > 0.0
        bound_violations[np.flatnonzero(observed)] = observed_violations
        maximum_bound_violation_fraction = float(
            np.max(bound_gap / np.maximum(close, 1.0))
        )
        if maximum_bound_violation_fraction > 0.01:
            raise ValueError("proxy frame has an OHLC bound violation above one percent")
    if suspended.any():
        suspended_prices = prices[suspended]
        if (
            (suspended_prices[:, :3] != 0.0).any()
            or (suspended_prices[:, 3] <= 0.0).any()
        ):
            raise ValueError("proxy frame has an unsupported partial zero-price bar")
    if not observed.any():
        raise ValueError("proxy frame contains no observed price bars")
    return {
        "rows": int(len(frame)),
        "observed_price_rows": int(observed.sum()),
        "suspended_rows": int(suspended.sum()),
        "suspended_nonzero_volume_rows": int(
            np.count_nonzero(suspended & (volume > 0.0))
        ),
        "ohlc_bound_violation_rows": int(bound_violations.sum()),
        "maximum_ohlc_bound_violation_fraction": maximum_bound_violation_fraction,
        "first_date": str(frame.index.min().date()),
        "last_date": str(frame.index.max().date()),
    }


def proxy_csv_bytes(frame: pd.DataFrame) -> bytes:
    buffer = StringIO()
    frame.to_csv(buffer, date_format="%Y-%m-%d", lineterminator="\n")
    return buffer.getvalue().encode("utf-8")


def write_bytes_atomic(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)
