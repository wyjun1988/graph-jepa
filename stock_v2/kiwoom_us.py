from __future__ import annotations

import math
import re
import time
from datetime import time as wall_time
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from stock_v2.kiwoom_investor import parse_kiwoom_number
from stock_v2.ops.brokers import KiwoomRestBroker


US_MARKET_TIMEZONE = "America/New_York"
KIWOOM_US_EXCHANGES = ("NA", "ND", "NY")
KIWOOM_US_MINUTE_INTERVALS = (1, 3, 5, 10, 15, 30, 60)
KIWOOM_US_DAILY_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "TradingValue",
    "PreviousChange",
    "ChangePct",
    "AdjustmentRatio",
)
KIWOOM_US_MINUTE_COLUMNS = (
    "RawTimestamp",
    "RawBusinessDate",
    "BusinessDate",
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "AdjustmentRatio",
)
KIWOOM_US_MINUTE_FLOAT_COLUMNS = (
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "AdjustmentRatio",
)

UniverseRawPageSink = Callable[[str, int, Mapping[str, Any], bool], None]
ChartRawPageSink = Callable[[int, Mapping[str, Any], bool], None]


def normalize_us_exchange(value: object) -> str:
    exchange = str(value).strip().upper()
    if exchange not in KIWOOM_US_EXCHANGES:
        raise ValueError(f"unsupported Kiwoom US exchange: {value!r}")
    return exchange


def normalize_us_ticker(value: object) -> str:
    ticker = str(value).strip().upper()
    if not ticker or len(ticker) > 12:
        raise ValueError(f"invalid Kiwoom US ticker: {value!r}")
    if not re.fullmatch(r"[A-Z0-9][A-Z0-9.\-/^+]*", ticker):
        raise ValueError(f"invalid Kiwoom US ticker: {value!r}")
    return ticker


def _number(value: object, *, absolute: bool = False) -> float:
    parsed = parse_kiwoom_number(value)
    if absolute and math.isfinite(parsed):
        return abs(parsed)
    return parsed


def _is_etf(value: object) -> bool:
    return str(value).strip().upper() in {"1", "TRUE", "Y", "YES"}


def _reject_conflicting_duplicates(
    frame: pd.DataFrame, *, index_names: Sequence[str], label: str
) -> pd.DataFrame:
    duplicate_keys = frame.index[frame.index.duplicated(keep=False)].unique()
    for key in duplicate_keys:
        rows = frame.loc[[key]]
        first = rows.iloc[0]
        for _index, candidate in rows.iloc[1:].iterrows():
            equal = (first.eq(candidate) | (first.isna() & candidate.isna())).all()
            if not bool(equal):
                raise ValueError(f"{label} conflicting duplicate {key}")
    result = frame.loc[~frame.index.duplicated(keep="first")].sort_index()
    result.index.names = list(index_names)
    return result


def parse_kiwoom_us_universe_rows(rows: object) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            try:
                exchange = normalize_us_exchange(row.get("stex_tp"))
                ticker = normalize_us_ticker(row.get("stk_cd"))
            except ValueError:
                continue
            records.append(
                {
                    "Exchange": exchange,
                    "Ticker": ticker,
                    "Name": str(row.get("stk_nm") or "").strip(),
                    "EnglishName": str(row.get("stk_enm") or "").strip(),
                    "ExchangeName": str(row.get("mkgb") or "").strip(),
                    "Industry": str(row.get("upgb") or "").strip(),
                    "IsETF": _is_etf(row.get("isEtf")),
                }
            )
    columns = [
        "Name",
        "EnglishName",
        "ExchangeName",
        "Industry",
        "IsETF",
    ]
    if not records:
        index = pd.MultiIndex.from_arrays([[], []], names=["Exchange", "Ticker"])
        return pd.DataFrame(columns=columns, index=index)
    frame = pd.DataFrame.from_records(records).set_index(["Exchange", "Ticker"])
    return _reject_conflicting_duplicates(
        frame.reindex(columns=columns),
        index_names=("Exchange", "Ticker"),
        label="Kiwoom usa10099",
    )


def fetch_kiwoom_us_universe(
    broker: KiwoomRestBroker,
    *,
    exchanges: Sequence[str] = KIWOOM_US_EXCHANGES,
    sleep_sec: float = 0.25,
    max_pages_per_exchange: int = 20,
    raw_page_sink: UniverseRawPageSink | None = None,
) -> pd.DataFrame:
    pages: list[pd.DataFrame] = []
    requested = tuple(normalize_us_exchange(value) for value in exchanges)
    if len(requested) != len(set(requested)):
        raise ValueError("Kiwoom US exchanges must be unique")
    for exchange in requested:
        continuation = False
        next_key: str | None = None
        seen_cursors: set[str] = set()
        for page_index in range(1, int(max_pages_per_exchange) + 1):
            data, has_more, returned_key = broker.post_readonly_with_continuation(
                "/api/us/stkinfo",
                "usa10099",
                {"stex_tp": exchange},
                continuation=continuation,
                next_key=next_key,
            )
            if raw_page_sink is not None:
                raw_page_sink(exchange, page_index, data, has_more)
            page = parse_kiwoom_us_universe_rows(data.get("list"))
            if not page.empty:
                realized = set(page.index.get_level_values("Exchange"))
                if realized != {exchange}:
                    raise ValueError(
                        f"Kiwoom usa10099 exchange mismatch: {exchange}/{realized}"
                    )
                pages.append(page)
            if not has_more:
                break
            if not returned_key or returned_key in seen_cursors:
                raise RuntimeError(
                    f"Kiwoom usa10099 invalid continuation for {exchange}"
                )
            seen_cursors.add(returned_key)
            continuation = True
            next_key = returned_key
            if sleep_sec > 0:
                time.sleep(float(sleep_sec))
        else:
            raise RuntimeError(
                f"Kiwoom usa10099 exceeded max pages for {exchange}"
            )
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))
    if not pages:
        return parse_kiwoom_us_universe_rows([])
    frame = pd.concat(pages)
    return _reject_conflicting_duplicates(
        frame,
        index_names=("Exchange", "Ticker"),
        label="Kiwoom usa10099",
    )


def _empty_daily_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=KIWOOM_US_DAILY_COLUMNS, dtype=float)
    frame.index = pd.DatetimeIndex([], name="Date")
    return frame


def parse_kiwoom_us_daily_rows(rows: object) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            date = pd.to_datetime(
                str(row.get("dt") or ""), format="%Y%m%d", errors="coerce"
            )
            if pd.isna(date):
                continue
            records.append(
                {
                    "Date": pd.Timestamp(date).normalize(),
                    "Open": _number(row.get("open_pric"), absolute=True),
                    "High": _number(row.get("high_pric"), absolute=True),
                    "Low": _number(row.get("low_pric"), absolute=True),
                    "Close": _number(row.get("cur_prc"), absolute=True),
                    "Volume": _number(row.get("acc_trde_qty"), absolute=True),
                    "TradingValue": _number(
                        row.get("acc_trde_prica"), absolute=True
                    ),
                    "PreviousChange": _number(row.get("pred_pre")),
                    "ChangePct": _number(row.get("flu_rt")),
                    "AdjustmentRatio": _number(row.get("upd_rt")),
                }
            )
    if not records:
        return _empty_daily_frame()
    frame = pd.DataFrame.from_records(records).set_index("Date")
    return _reject_conflicting_duplicates(
        frame.reindex(columns=KIWOOM_US_DAILY_COLUMNS).astype(float),
        index_names=("Date",),
        label="Kiwoom usa06012",
    )


def audit_kiwoom_us_daily_frame(frame: pd.DataFrame) -> dict[str, object]:
    if not isinstance(frame.index, pd.DatetimeIndex):
        raise ValueError("US daily frame must use a DatetimeIndex")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("US daily dates must be sorted and unique")
    missing = [name for name in KIWOOM_US_DAILY_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"US daily frame missing columns: {missing}")
    if frame.empty:
        return {"rows": 0, "first_date": None, "last_date": None}
    core = frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(float)
    if not np.isfinite(core).all():
        raise ValueError("US daily frame contains non-finite core OHLCV values")
    prices = frame[["Open", "High", "Low", "Close"]].to_numpy(float)
    if (prices <= 0).any() or (frame["Volume"].to_numpy(float) < 0).any():
        raise ValueError("US daily frame contains invalid price or volume")
    if (frame["High"] < frame[["Open", "Close"]].max(axis=1)).any():
        raise ValueError("US daily frame violates high")
    if (frame["Low"] > frame[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("US daily frame violates low")
    return {
        "rows": int(len(frame)),
        "first_date": frame.index.min().date().isoformat(),
        "last_date": frame.index.max().date().isoformat(),
    }


def repair_kiwoom_us_daily_ohlc_envelope(
    frame: pd.DataFrame,
    *,
    maximum_relative_repair: float = 0.01,
    maximum_repaired_fraction: float = 0.01,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Repair small vendor OHLC envelope defects while preserving source values."""

    required = ["Open", "High", "Low", "Close"]
    missing = [column for column in required if column not in frame]
    if missing:
        raise ValueError(f"US daily frame missing OHLC columns: {missing}")
    if maximum_relative_repair < 0 or not 0 <= maximum_repaired_fraction <= 1:
        raise ValueError("invalid US daily OHLC repair limits")
    result = frame.copy()
    if result.empty:
        result["VendorHigh"] = pd.Series(index=result.index, dtype=float)
        result["VendorLow"] = pd.Series(index=result.index, dtype=float)
        result["OHLCEnvelopeRepaired"] = pd.Series(index=result.index, dtype=bool)
        result["OHLCEnvelopeRepairMagnitude"] = pd.Series(
            index=result.index, dtype=float
        )
        return result, {
            "repaired_rows": 0,
            "repaired_fraction": 0.0,
            "maximum_relative_repair": 0.0,
            "repaired_dates": [],
        }

    ohlc = result[required].to_numpy(float)
    if not np.isfinite(ohlc).all() or (ohlc <= 0).any():
        raise ValueError("US daily OHLC repair received invalid prices")
    repaired_high = result[required].max(axis=1)
    repaired_low = result[required].min(axis=1)
    high_delta = (repaired_high - result["High"]).clip(lower=0)
    low_delta = (result["Low"] - repaired_low).clip(lower=0)
    repaired = (high_delta > 0) | (low_delta > 0)
    denominator = result["Close"].abs().clip(lower=np.finfo(float).eps)
    magnitude = pd.concat([high_delta, low_delta], axis=1).max(axis=1) / denominator
    count = int(repaired.sum())
    fraction = float(count / len(result))
    maximum = float(magnitude.max())
    if maximum > float(maximum_relative_repair):
        raise ValueError(
            "US daily OHLC envelope repair exceeds relative limit: "
            f"{maximum:.8f}"
        )
    if fraction > float(maximum_repaired_fraction):
        raise ValueError(
            "US daily OHLC envelope repair exceeds row-fraction limit: "
            f"{fraction:.8f}"
        )

    result["VendorHigh"] = result["High"]
    result["VendorLow"] = result["Low"]
    result["High"] = repaired_high
    result["Low"] = repaired_low
    result["OHLCEnvelopeRepaired"] = repaired.astype(bool)
    result["OHLCEnvelopeRepairMagnitude"] = magnitude.astype(float)
    dates = [
        pd.Timestamp(value).date().isoformat()
        for value in result.index[repaired]
    ]
    return result, {
        "repaired_rows": count,
        "repaired_fraction": fraction,
        "maximum_relative_repair": maximum,
        "repaired_dates": dates,
    }


def fetch_kiwoom_us_daily_history(
    broker: KiwoomRestBroker,
    exchange: str,
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    adjusted: bool,
    apply_exchange_rate: bool = False,
    sleep_sec: float = 0.0,
    max_pages: int = 100,
    raw_page_sink: ChartRawPageSink | None = None,
) -> pd.DataFrame:
    exchange = normalize_us_exchange(exchange)
    ticker = normalize_us_ticker(ticker)
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if end_date < start_date:
        raise ValueError("end must not precede start")
    payload = {
        "stex_tp": exchange,
        "stk_cd": ticker,
        "strt_dt": end_date.strftime("%Y%m%d"),
        "upd_stkpc_tp": "1" if adjusted else "0",
        "exrt_appl_tp": "1" if apply_exchange_rate else "0",
    }
    pages: list[pd.DataFrame] = []
    continuation = False
    next_key: str | None = None
    seen_cursors: set[str] = set()
    previous_oldest: pd.Timestamp | None = None
    for page_index in range(1, int(max_pages) + 1):
        data, has_more, returned_key = broker.post_readonly_with_continuation(
            "/api/us/chart",
            "usa06012",
            payload,
            continuation=continuation,
            next_key=next_key,
        )
        if raw_page_sink is not None:
            raw_page_sink(page_index, data, has_more)
        page = parse_kiwoom_us_daily_rows(data.get("result_list"))
        if page.empty:
            break
        oldest = pd.Timestamp(page.index.min())
        if previous_oldest is not None and oldest >= previous_oldest:
            raise RuntimeError(
                f"Kiwoom usa06012 pagination did not move backward for {ticker}"
            )
        previous_oldest = oldest
        pages.append(page)
        if oldest <= start_date or not has_more:
            break
        if not returned_key or returned_key in seen_cursors:
            raise RuntimeError(f"Kiwoom usa06012 invalid cursor for {ticker}")
        seen_cursors.add(returned_key)
        continuation = True
        next_key = returned_key
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))
    else:
        raise RuntimeError(f"Kiwoom usa06012 exceeded max_pages={max_pages}")
    if not pages:
        return _empty_daily_frame()
    frame = _reject_conflicting_duplicates(
        pd.concat(pages),
        index_names=("Date",),
        label="Kiwoom usa06012",
    )
    return frame.loc[(frame.index >= start_date) & (frame.index <= end_date)].copy()


def _empty_minute_frame() -> pd.DataFrame:
    frame = pd.DataFrame(columns=KIWOOM_US_MINUTE_COLUMNS)
    frame.index = pd.DatetimeIndex([], name="Timestamp", tz=US_MARKET_TIMEZONE)
    return frame


def parse_kiwoom_us_minute_rows(rows: object) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    if isinstance(rows, list):
        for row in rows:
            if not isinstance(row, Mapping):
                continue
            raw_timestamp = str(row.get("cntr_tm") or "").strip()
            raw_business_date = str(row.get("bus_dt") or "").strip()
            timestamp = pd.to_datetime(
                raw_timestamp,
                format="%Y%m%d%H%M%S",
                errors="coerce",
            )
            business_date = pd.to_datetime(
                raw_business_date,
                format="%Y%m%d",
                errors="coerce",
            )
            if pd.isna(timestamp) or pd.isna(business_date):
                raise ValueError(
                    "Kiwoom usa06011 returned an invalid cntr_tm or bus_dt"
                )
            records.append(
                {
                    "Timestamp": pd.Timestamp(timestamp),
                    "RawTimestamp": raw_timestamp,
                    "RawBusinessDate": raw_business_date,
                    "BusinessDate": pd.Timestamp(business_date).normalize(),
                    "Open": _number(row.get("open_pric"), absolute=True),
                    "High": _number(row.get("high_pric"), absolute=True),
                    "Low": _number(row.get("low_pric"), absolute=True),
                    "Close": _number(row.get("cur_prc"), absolute=True),
                    "Volume": _number(row.get("trde_qty"), absolute=True),
                    "AdjustmentRatio": _number(row.get("upd_rt")),
                }
            )
    if not records:
        return _empty_minute_frame()
    frame = pd.DataFrame.from_records(records)
    naive = pd.DatetimeIndex(frame.pop("Timestamp"), name="Timestamp")
    localized = naive.tz_localize(
        US_MARKET_TIMEZONE, ambiguous="NaT", nonexistent="NaT"
    )
    valid = ~localized.isna()
    if not valid.all():
        raise ValueError(
            "Kiwoom usa06011 returned an ambiguous or nonexistent US local time"
        )
    frame = frame.loc[valid].copy()
    frame.index = localized[valid]
    frame = frame.reindex(columns=KIWOOM_US_MINUTE_COLUMNS)
    for column in KIWOOM_US_MINUTE_FLOAT_COLUMNS:
        frame[column] = frame[column].astype(float)
    return _reject_conflicting_duplicates(
        frame,
        index_names=("Timestamp",),
        label="Kiwoom usa06011",
    )


def _regular_session_mask(index: pd.DatetimeIndex) -> np.ndarray:
    local = index.tz_convert(US_MARKET_TIMEZONE)
    values = local.time
    return np.asarray(
        [(wall_time(9, 30) <= value <= wall_time(16, 0)) for value in values],
        dtype=bool,
    )


def audit_kiwoom_us_minute_frame(
    frame: pd.DataFrame, *, regular_session_only: bool = True
) -> dict[str, object]:
    if not isinstance(frame.index, pd.DatetimeIndex) or frame.index.tz is None:
        raise ValueError("US minute timestamps must be timezone-aware")
    if str(frame.index.tz) != US_MARKET_TIMEZONE:
        raise ValueError("US minute timestamps must use America/New_York")
    if not frame.index.is_monotonic_increasing or frame.index.has_duplicates:
        raise ValueError("US minute timestamps must be sorted and unique")
    missing = [name for name in KIWOOM_US_MINUTE_COLUMNS if name not in frame]
    if missing:
        raise ValueError(f"US minute frame missing columns: {missing}")
    if frame.empty:
        return {
            "rows": 0,
            "sessions": 0,
            "first_timestamp": None,
            "last_timestamp": None,
        }
    expected_raw_timestamp = frame.index.strftime("%Y%m%d%H%M%S")
    if not np.array_equal(
        frame["RawTimestamp"].astype(str).to_numpy(),
        np.asarray(expected_raw_timestamp),
    ):
        raise ValueError("US minute raw timestamp changed during parsing")
    expected_raw_business_date = pd.to_datetime(
        frame["BusinessDate"]
    ).dt.strftime("%Y%m%d")
    if not np.array_equal(
        frame["RawBusinessDate"].astype(str).to_numpy(),
        expected_raw_business_date.to_numpy(),
    ):
        raise ValueError("US minute raw business date changed during parsing")
    if regular_session_only and not _regular_session_mask(frame.index).all():
        raise ValueError("US minute frame contains bars outside the regular session")
    core = frame[["Open", "High", "Low", "Close", "Volume"]].to_numpy(float)
    if not np.isfinite(core).all():
        raise ValueError("US minute frame contains non-finite core OHLCV values")
    if (frame[["Open", "High", "Low", "Close"]].to_numpy(float) <= 0).any():
        raise ValueError("US minute frame contains non-positive prices")
    if (frame["Volume"].to_numpy(float) < 0).any():
        raise ValueError("US minute frame contains negative volume")
    if (frame["High"] < frame[["Open", "Close"]].max(axis=1)).any():
        raise ValueError("US minute frame violates high")
    if (frame["Low"] > frame[["Open", "Close"]].min(axis=1)).any():
        raise ValueError("US minute frame violates low")
    regular = _regular_session_mask(frame.index)
    if regular.any():
        business_dates = pd.to_datetime(frame.loc[regular, "BusinessDate"]).dt.date
        local_dates = pd.Index(frame.index[regular].date)
        if not np.array_equal(np.asarray(business_dates), np.asarray(local_dates)):
            raise ValueError("US minute regular bars disagree with business date")
    return {
        "rows": int(len(frame)),
        "sessions": int(pd.Index(frame["BusinessDate"]).nunique()),
        "first_timestamp": frame.index.min().isoformat(),
        "last_timestamp": frame.index.max().isoformat(),
    }


def fetch_kiwoom_us_minute_history(
    broker: KiwoomRestBroker,
    exchange: str,
    ticker: str,
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    *,
    interval_minutes: int,
    adjusted: bool,
    apply_exchange_rate: bool = False,
    regular_session_only: bool = True,
    sleep_sec: float = 0.0,
    max_pages: int = 10_000,
    raw_page_sink: ChartRawPageSink | None = None,
) -> pd.DataFrame:
    exchange = normalize_us_exchange(exchange)
    ticker = normalize_us_ticker(ticker)
    interval = int(interval_minutes)
    if interval not in KIWOOM_US_MINUTE_INTERVALS:
        raise ValueError(
            f"interval_minutes must be one of {KIWOOM_US_MINUTE_INTERVALS}"
        )
    start_date = pd.Timestamp(start).normalize()
    end_date = pd.Timestamp(end).normalize()
    if end_date < start_date:
        raise ValueError("end must not precede start")
    payload = {
        "stex_tp": exchange,
        "stk_cd": ticker,
        "strt_dt": end_date.strftime("%Y%m%d"),
        "tic_scope": str(interval),
        "upd_stkpc_tp": "1" if adjusted else "0",
        "exrt_appl_tp": "1" if apply_exchange_rate else "0",
    }
    pages: list[pd.DataFrame] = []
    continuation = False
    next_key: str | None = None
    seen_cursors: set[str] = set()
    previous_oldest: pd.Timestamp | None = None
    for page_index in range(1, int(max_pages) + 1):
        data, has_more, returned_key = broker.post_readonly_with_continuation(
            "/api/us/chart",
            "usa06011",
            payload,
            continuation=continuation,
            next_key=next_key,
        )
        if raw_page_sink is not None:
            raw_page_sink(page_index, data, has_more)
        page = parse_kiwoom_us_minute_rows(data.get("result_list"))
        if page.empty:
            break
        oldest = pd.Timestamp(page.index.min())
        if previous_oldest is not None and oldest >= previous_oldest:
            raise RuntimeError(
                f"Kiwoom usa06011 pagination did not move backward for {ticker}"
            )
        previous_oldest = oldest
        pages.append(page)
        oldest_business_date = pd.Timestamp(page["BusinessDate"].min()).normalize()
        # One US session spans several 100-row pages, so equality does not
        # prove that the opening bars for the requested first day were read.
        if oldest_business_date < start_date or not has_more:
            break
        if not returned_key or returned_key in seen_cursors:
            raise RuntimeError(f"Kiwoom usa06011 invalid cursor for {ticker}")
        seen_cursors.add(returned_key)
        continuation = True
        next_key = returned_key
        if sleep_sec > 0:
            time.sleep(float(sleep_sec))
    else:
        raise RuntimeError(f"Kiwoom usa06011 exceeded max_pages={max_pages}")
    if not pages:
        return _empty_minute_frame()
    frame = _reject_conflicting_duplicates(
        pd.concat(pages),
        index_names=("Timestamp",),
        label="Kiwoom usa06011",
    )
    business = pd.to_datetime(frame["BusinessDate"])
    selected = (business >= start_date) & (business <= end_date)
    if regular_session_only:
        selected &= _regular_session_mask(frame.index)
    return frame.loc[selected].copy()
