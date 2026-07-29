from __future__ import annotations

from typing import Any, Callable, Mapping

import numpy as np
import pandas as pd
import requests

from stock_v2.kiwoom_us import normalize_us_ticker


YAHOO_CHART_URL = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
YAHOO_US_TIMEZONE = "America/New_York"
YAHOO_DAILY_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "AdjustedClose",
    "Volume",
    "AdjustmentFactor",
)

YahooRawSink = Callable[[Mapping[str, Any], Mapping[str, Any]], None]


def _required_array(
    payload: Mapping[str, Any], key: str, *, length: int
) -> list[object]:
    values = payload.get(key)
    if not isinstance(values, list) or len(values) != length:
        raise ValueError(f"Yahoo chart has invalid {key} array")
    return values


def parse_yahoo_us_daily_chart(payload: object) -> pd.DataFrame:
    if not isinstance(payload, Mapping):
        raise ValueError("Yahoo chart response must be an object")
    chart = payload.get("chart")
    if not isinstance(chart, Mapping) or chart.get("error") is not None:
        raise ValueError("Yahoo chart returned an error")
    results = chart.get("result")
    if not isinstance(results, list) or len(results) != 1:
        raise ValueError("Yahoo chart must contain exactly one result")
    result = results[0]
    if not isinstance(result, Mapping):
        raise ValueError("Yahoo chart result is invalid")
    metadata = result.get("meta")
    timezone = metadata.get("exchangeTimezoneName") if isinstance(metadata, Mapping) else None
    if timezone != YAHOO_US_TIMEZONE:
        raise ValueError(f"Yahoo chart has unexpected exchange timezone: {timezone}")

    raw_timestamps = result.get("timestamp")
    indicators = result.get("indicators")
    if not isinstance(raw_timestamps, list) or not isinstance(indicators, Mapping):
        raise ValueError("Yahoo chart is missing timestamps or indicators")
    quotes = indicators.get("quote")
    adjusted = indicators.get("adjclose")
    if (
        not isinstance(quotes, list)
        or len(quotes) != 1
        or not isinstance(quotes[0], Mapping)
        or not isinstance(adjusted, list)
        or len(adjusted) != 1
        or not isinstance(adjusted[0], Mapping)
    ):
        raise ValueError("Yahoo chart indicator structure is invalid")
    count = len(raw_timestamps)
    if count == 0:
        empty = pd.DataFrame(columns=YAHOO_DAILY_COLUMNS, dtype=float)
        empty.index = pd.DatetimeIndex([], name="Date")
        return empty
    timestamp_values = pd.to_numeric(pd.Series(raw_timestamps), errors="coerce")
    if timestamp_values.isna().any():
        raise ValueError("Yahoo chart contains an invalid timestamp")
    timestamps_utc = pd.to_datetime(timestamp_values.astype("int64"), unit="s", utc=True)
    dates = pd.DatetimeIndex(
        timestamps_utc.dt.tz_convert(YAHOO_US_TIMEZONE).dt.tz_localize(None).dt.normalize(),
        name="Date",
    )
    quote = quotes[0]
    frame = pd.DataFrame(
        {
            "Open": _required_array(quote, "open", length=count),
            "High": _required_array(quote, "high", length=count),
            "Low": _required_array(quote, "low", length=count),
            "Close": _required_array(quote, "close", length=count),
            "AdjustedClose": _required_array(
                adjusted[0], "adjclose", length=count
            ),
            "Volume": _required_array(quote, "volume", length=count),
        },
        index=dates,
    )
    frame = frame.apply(pd.to_numeric, errors="coerce")
    if frame.isna().any().any():
        missing_dates = frame.index[frame.isna().any(axis=1)]
        raise ValueError(
            "Yahoo chart contains missing daily values: "
            + ",".join(value.date().isoformat() for value in missing_dates[:10])
        )
    if frame.index.has_duplicates:
        raise ValueError("Yahoo chart contains duplicate US dates")
    frame = frame.sort_index()
    frame["AdjustmentFactor"] = frame["AdjustedClose"] / frame["Close"]
    return frame.reindex(columns=YAHOO_DAILY_COLUMNS)


def audit_yahoo_us_daily_frame(frame: pd.DataFrame) -> dict[str, object]:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("Yahoo US daily frame must use DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("Yahoo US daily dates must be sorted and unique")
    missing = [column for column in YAHOO_DAILY_COLUMNS if column not in frame]
    if missing:
        raise ValueError(f"Yahoo US daily frame missing columns: {missing}")
    if frame.empty:
        return {"rows": 0, "first_date": None, "last_date": None}
    values = frame[list(YAHOO_DAILY_COLUMNS)].to_numpy(float)
    if not np.isfinite(values).all():
        raise ValueError("Yahoo US daily frame contains non-finite values")
    if (frame[["Open", "High", "Low", "Close", "AdjustedClose"]] <= 0).any().any():
        raise ValueError("Yahoo US daily frame contains non-positive prices")
    if (frame["Volume"] < 0).any() or (frame["AdjustmentFactor"] <= 0).any():
        raise ValueError("Yahoo US daily frame contains invalid volume or adjustment")
    if (frame["High"] < frame[["Open", "Close"]].max(axis=1)).any():
        raise ValueError("Yahoo US daily frame violates high")
    if (frame["Low"] > frame[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("Yahoo US daily frame violates low")
    return {
        "rows": int(len(frame)),
        "first_date": frame.index.min().date().isoformat(),
        "last_date": frame.index.max().date().isoformat(),
    }


def yahoo_chart_request(
    ticker: object, start: object, end: object
) -> tuple[str, dict[str, object]]:
    ticker = normalize_us_ticker(ticker)
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if end_date < start_date:
        raise ValueError("end must not precede start")
    start_et = start_date.tz_localize(YAHOO_US_TIMEZONE)
    end_exclusive_et = (end_date + pd.Timedelta(days=1)).tz_localize(
        YAHOO_US_TIMEZONE
    )
    return YAHOO_CHART_URL.format(ticker=ticker), {
        "period1": int(start_et.timestamp()),
        "period2": int(end_exclusive_et.timestamp()),
        "interval": "1d",
        "events": "div,splits",
        "includeAdjustedClose": "true",
        "includePrePost": "false",
    }


def fetch_yahoo_us_daily_history(
    ticker: object,
    start: object,
    end: object,
    *,
    timeout_sec: float = 30.0,
    session: requests.Session | None = None,
    raw_sink: YahooRawSink | None = None,
) -> pd.DataFrame:
    url, parameters = yahoo_chart_request(ticker, start, end)
    client = session or requests.Session()
    response = client.get(
        url,
        params=parameters,
        headers={"User-Agent": "Mozilla/5.0 stock-v2-data-audit/1.0"},
        timeout=float(timeout_sec),
    )
    response.raise_for_status()
    payload = response.json()
    if raw_sink is not None:
        raw_sink(parameters, payload)
    frame = parse_yahoo_us_daily_chart(payload)
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    return frame.loc[(frame.index >= start_date) & (frame.index <= end_date)].copy()
