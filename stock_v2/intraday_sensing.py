from __future__ import annotations

from dataclasses import dataclass
from datetime import time as wall_time
from typing import Mapping, Sequence
import warnings

import numpy as np
import pandas as pd

from stock_v2.kiwoom_minute import KST, audit_kiwoom_minute_frame


WINDOW_COLUMNS = (
    "SessionOpen",
    "DecisionPrice",
    "EarlyReturn",
    "EarlyRange",
    "RealizedAbsoluteReturn",
    "EarlyVolume",
    "EarlyTradedValue",
    "BarCount",
    "ExpectedBarCount",
    "BarCoverage",
    "WindowComplete",
)

SESSION_TARGET_COLUMNS = (
    "SessionClose",
    "SessionHigh",
    "SessionLow",
    "SessionVolume",
    "SessionBarCount",
    "LastBarTimestamp",
    "SessionComplete",
)

PANEL_FEATURE_NAMES = (
    "early_return",
    "early_range",
    "realized_absolute_return",
    "log_early_volume",
    "log_early_traded_value",
    "early_volume_shock_20",
    "early_value_shock_20",
    "bar_coverage",
)

MARKET_STATISTICS = (
    "mean",
    "std",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "coverage",
)


@dataclass(frozen=True)
class IntradayWindowPanel:
    dates: pd.DatetimeIndex
    tickers: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray
    available: np.ndarray
    decision_price: np.ndarray
    session_open: np.ndarray
    window_complete: np.ndarray


@dataclass(frozen=True)
class IntradayMarketDesign:
    dates: pd.DatetimeIndex
    values: np.ndarray
    feature_names: tuple[str, ...]


def _parse_decision_time(value: str | wall_time) -> wall_time:
    if isinstance(value, wall_time):
        result = value
    else:
        try:
            result = wall_time.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError("decision_time must use HH:MM or HH:MM:SS") from exc
    if not wall_time(9, 0) < result <= wall_time(15, 30):
        raise ValueError("decision_time must be inside the KRX regular session")
    return result


def _minutes_since_midnight(value: wall_time) -> int:
    if value.second or value.microsecond:
        raise ValueError("decision_time must align to a whole minute")
    return value.hour * 60 + value.minute


def summarize_early_window(
    frame: pd.DataFrame,
    *,
    interval_minutes: int,
    decision_time: str | wall_time = "09:15",
    timestamp_semantics: str,
) -> pd.DataFrame:
    """Summarize only bars fully completed by the stated decision time.

    ``timestamp_semantics`` is mandatory because including a start-labelled bar
    stamped at the decision minute would leak the following interval.
    """

    interval = int(interval_minutes)
    if interval <= 0:
        raise ValueError("interval_minutes must be positive")
    if timestamp_semantics not in {"start", "end"}:
        raise ValueError("timestamp_semantics must be start or end")
    decision = _parse_decision_time(decision_time)
    open_minutes = 9 * 60
    decision_minutes = _minutes_since_midnight(decision)
    window_minutes = decision_minutes - open_minutes
    if window_minutes % interval:
        raise ValueError("decision window must be divisible by interval_minutes")
    expected_bars = window_minutes // interval
    if expected_bars <= 0:
        raise ValueError("decision window must contain at least one completed bar")

    audit_kiwoom_minute_frame(frame, regular_session_only=True)
    if frame.empty:
        empty = pd.DataFrame(columns=WINDOW_COLUMNS, dtype=float)
        empty.index = pd.DatetimeIndex([], name="Date")
        return empty

    local = frame.copy()
    local.index = local.index.tz_convert(KST)
    minute = local.index.hour * 60 + local.index.minute
    if timestamp_semantics == "start":
        in_window = (minute >= open_minutes) & (minute + interval <= decision_minutes)
    else:
        in_window = (minute > open_minutes) & (minute <= decision_minutes)
    selected = local.loc[in_window].copy()
    if selected.empty:
        empty = pd.DataFrame(columns=WINDOW_COLUMNS, dtype=float)
        empty.index = pd.DatetimeIndex([], name="Date")
        return empty

    selected["_session_date"] = selected.index.tz_localize(None).normalize()
    records: list[dict[str, object]] = []
    for session_date, bars in selected.groupby("_session_date", sort=True):
        bars = bars.drop(columns="_session_date").sort_index()
        bar_count = int(len(bars))
        session_day = bars.index[0].tz_localize(None).normalize()
        session_start = session_day.tz_localize(KST) + pd.Timedelta(9, unit="h")
        first_offset = 0 if timestamp_semantics == "start" else interval
        expected_index = session_start + pd.to_timedelta(
            first_offset + np.arange(expected_bars) * interval,
            unit="minute",
        )
        matched_bars = int(
            np.intersect1d(bars.index.asi8, expected_index.asi8).size
        )
        window_complete = bool(
            bar_count == expected_bars
            and np.array_equal(bars.index.asi8, expected_index.asi8)
        )
        session_open = float(bars["Open"].iloc[0])
        decision_price = float(bars["Close"].iloc[-1])
        high = float(bars["High"].max())
        low = float(bars["Low"].min())
        close_path = np.concatenate(
            ([session_open], bars["Close"].to_numpy(dtype=np.float64))
        )
        step_returns = close_path[1:] / close_path[:-1] - 1.0
        typical_price = bars[["Open", "High", "Low", "Close"]].mean(axis=1)
        volume = bars["Volume"].to_numpy(dtype=np.float64)
        record = {
            "Date": session_day,
            "SessionOpen": session_open,
            "DecisionPrice": decision_price,
            "EarlyReturn": decision_price / session_open - 1.0,
            "EarlyRange": high / low - 1.0,
            "RealizedAbsoluteReturn": float(np.abs(step_returns).sum()),
            "EarlyVolume": float(volume.sum()),
            "EarlyTradedValue": float(
                np.sum(typical_price.to_numpy(dtype=np.float64) * volume)
            ),
            "BarCount": bar_count,
            "ExpectedBarCount": expected_bars,
            "BarCoverage": float(matched_bars / expected_bars),
            "WindowComplete": window_complete,
        }
        records.append(record)
    result = pd.DataFrame.from_records(records).set_index("Date").sort_index()
    return result.reindex(columns=WINDOW_COLUMNS)


def summarize_session_targets(
    frame: pd.DataFrame,
    *,
    interval_minutes: int,
    timestamp_semantics: str,
) -> pd.DataFrame:
    """Extract full-session labels separately from decision-time inputs."""

    interval = int(interval_minutes)
    if interval <= 0:
        raise ValueError("interval_minutes must be positive")
    if timestamp_semantics not in {"start", "end"}:
        raise ValueError("timestamp_semantics must be start or end")

    audit_kiwoom_minute_frame(frame, regular_session_only=True)
    if frame.empty:
        empty = pd.DataFrame(columns=SESSION_TARGET_COLUMNS)
        empty.index = pd.DatetimeIndex([], name="Date")
        return empty
    local = frame.copy()
    local.index = local.index.tz_convert(KST)
    local["_session_date"] = local.index.tz_localize(None).normalize()
    records: list[dict[str, object]] = []
    for session_date, bars in local.groupby("_session_date", sort=True):
        bars = bars.drop(columns="_session_date").sort_index()
        session_day = bars.index[0].tz_localize(None).normalize()
        session_start = session_day.tz_localize(KST) + pd.Timedelta(9, unit="h")
        auction_start_minutes = 15 * 60 + 20
        if timestamp_semantics == "start":
            continuous_offsets = np.arange(
                0,
                auction_start_minutes - 9 * 60,
                interval,
                dtype=np.int64,
            )
        else:
            continuous_offsets = np.arange(
                interval,
                auction_start_minutes - 9 * 60 + 1,
                interval,
                dtype=np.int64,
            )
        expected_continuous = session_start + pd.to_timedelta(
            continuous_offsets, unit="minute"
        )
        expected_close = session_day.tz_localize(KST) + pd.Timedelta(
            15 * 60 + 30, unit="minute"
        )
        expected_index = expected_continuous.append(pd.DatetimeIndex([expected_close]))
        session_complete = bool(
            bars.index[-1] == expected_close
            and np.intersect1d(bars.index.asi8, expected_index.asi8).size
            == len(expected_index)
        )
        records.append(
            {
                "Date": session_day,
                "SessionClose": float(bars["Close"].iloc[-1]),
                "SessionHigh": float(bars["High"].max()),
                "SessionLow": float(bars["Low"].min()),
                "SessionVolume": float(bars["Volume"].sum()),
                "SessionBarCount": int(len(bars)),
                "LastBarTimestamp": bars.index[-1].isoformat(),
                "SessionComplete": session_complete,
            }
        )
    return (
        pd.DataFrame.from_records(records)
        .set_index("Date")
        .sort_index()
        .reindex(columns=SESSION_TARGET_COLUMNS)
    )


def _causal_log_shock(
    values: pd.Series,
    *,
    rolling_window: int,
    min_history: int,
) -> pd.Series:
    logged = np.log1p(values.astype(float).where(values >= 0.0))
    baseline = logged.shift(1).rolling(
        window=int(rolling_window),
        min_periods=int(min_history),
    ).median()
    return logged - baseline


def build_intraday_window_panel(
    summaries: Mapping[str, pd.DataFrame],
    *,
    dates: Sequence[object],
    tickers: Sequence[str],
    eligible: np.ndarray | None = None,
    rolling_window: int = 20,
    min_history: int = 10,
) -> IntradayWindowPanel:
    """Build a node panel with shifted rolling baselines and explicit masks."""

    if int(rolling_window) <= 0 or not 1 <= int(min_history) <= int(rolling_window):
        raise ValueError("rolling baseline requires 1 <= min_history <= rolling_window")
    date_index = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize()
    if date_index.has_duplicates or not date_index.is_monotonic_increasing:
        raise ValueError("intraday panel dates must be unique and sorted")
    ticker_tuple = tuple(str(ticker) for ticker in tickers)
    if not ticker_tuple or len(set(ticker_tuple)) != len(ticker_tuple):
        raise ValueError("intraday panel tickers must be non-empty and unique")
    shape = (len(date_index), len(ticker_tuple))
    if eligible is None:
        eligible_mask = np.ones(shape, dtype=bool)
    else:
        eligible_mask = np.asarray(eligible, dtype=bool)
        if eligible_mask.shape != shape:
            raise ValueError(f"eligible mask shape {eligible_mask.shape} != {shape}")

    values = np.full((*shape, len(PANEL_FEATURE_NAMES)), np.nan, dtype=np.float32)
    available = np.zeros_like(values, dtype=bool)
    decision_price = np.full(shape, np.nan, dtype=np.float32)
    session_open = np.full(shape, np.nan, dtype=np.float32)
    complete = np.zeros(shape, dtype=bool)

    for ticker_index, ticker in enumerate(ticker_tuple):
        source = summaries.get(ticker)
        if source is None or source.empty:
            continue
        missing = [column for column in WINDOW_COLUMNS if column not in source]
        if missing:
            raise ValueError(f"summary for {ticker} is missing columns: {missing}")
        aligned = source.copy()
        aligned.index = pd.DatetimeIndex(pd.to_datetime(aligned.index)).normalize()
        if aligned.index.has_duplicates or not aligned.index.is_monotonic_increasing:
            raise ValueError(f"summary dates for {ticker} must be unique and sorted")
        aligned = aligned.reindex(date_index)
        volume_shock = _causal_log_shock(
            aligned["EarlyVolume"],
            rolling_window=rolling_window,
            min_history=min_history,
        )
        value_shock = _causal_log_shock(
            aligned["EarlyTradedValue"],
            rolling_window=rolling_window,
            min_history=min_history,
        )
        columns = (
            aligned["EarlyReturn"],
            aligned["EarlyRange"],
            aligned["RealizedAbsoluteReturn"],
            np.log1p(aligned["EarlyVolume"].where(aligned["EarlyVolume"] >= 0.0)),
            np.log1p(
                aligned["EarlyTradedValue"].where(aligned["EarlyTradedValue"] >= 0.0)
            ),
            volume_shock,
            value_shock,
            aligned["BarCoverage"],
        )
        matrix = np.column_stack(
            [np.asarray(column, dtype=np.float64) for column in columns]
        )
        ticker_complete = aligned["WindowComplete"].fillna(False).to_numpy(dtype=bool)
        valid = (
            np.isfinite(matrix)
            & ticker_complete[:, None]
            & eligible_mask[:, ticker_index, None]
        )
        values[:, ticker_index] = np.where(valid, matrix, np.nan).astype(np.float32)
        available[:, ticker_index] = valid
        raw_decision = aligned["DecisionPrice"].to_numpy(dtype=np.float64)
        raw_open = aligned["SessionOpen"].to_numpy(dtype=np.float64)
        price_valid = (
            ticker_complete
            & eligible_mask[:, ticker_index]
            & np.isfinite(raw_decision)
            & np.isfinite(raw_open)
            & (raw_decision > 0.0)
            & (raw_open > 0.0)
        )
        decision_price[:, ticker_index] = np.where(
            price_valid, raw_decision, np.nan
        ).astype(np.float32)
        session_open[:, ticker_index] = np.where(
            price_valid, raw_open, np.nan
        ).astype(np.float32)
        complete[:, ticker_index] = price_valid

    return IntradayWindowPanel(
        dates=date_index,
        tickers=ticker_tuple,
        feature_names=PANEL_FEATURE_NAMES,
        values=values,
        available=available,
        decision_price=decision_price,
        session_open=session_open,
        window_complete=complete,
    )


def _masked_statistics(
    values: np.ndarray,
    available: np.ndarray,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = np.asarray(values, dtype=np.float64)
    available = np.asarray(available, dtype=bool) & np.isfinite(values)
    if values.shape != available.shape or values.ndim != 2:
        raise ValueError("market statistics require aligned [date, stock] arrays")
    count = available.sum(axis=1).astype(np.float64)
    total = np.where(available, values, 0.0).sum(axis=1)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0.0)
    centered = np.where(available, values - mean[:, None], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=1),
        count,
        out=np.zeros_like(total),
        where=count > 0.0,
    )
    masked = np.where(available, values, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        quantiles = np.nanquantile(
            masked, [0.10, 0.25, 0.50, 0.75, 0.90], axis=1
        )
    output = np.column_stack(
        (
            mean,
            np.sqrt(np.maximum(variance, 0.0)),
            quantiles.T,
            count / max(values.shape[1], 1),
        )
    )
    return np.nan_to_num(output, nan=0.0).astype(np.float32), MARKET_STATISTICS


def build_intraday_market_design(
    panel: IntradayWindowPanel,
    *,
    market_by_ticker: Mapping[str, str] | None = None,
) -> IntradayMarketDesign:
    """Aggregate node observations into all-market and KOSPI/KOSDAQ sensors."""

    if panel.values.shape != panel.available.shape or panel.values.ndim != 3:
        raise ValueError("intraday panel values and masks must be aligned")
    if panel.values.shape[:2] != (len(panel.dates), len(panel.tickers)):
        raise ValueError("intraday panel axes do not match dates and tickers")
    if panel.values.shape[2] != len(panel.feature_names):
        raise ValueError("intraday panel feature axis does not match names")

    groups: list[tuple[str, np.ndarray]] = [
        ("all", np.ones(len(panel.tickers), dtype=bool))
    ]
    if market_by_ticker:
        labels = np.asarray(
            [str(market_by_ticker.get(ticker, "")) for ticker in panel.tickers]
        )
        for market in ("KOSPI", "KOSDAQ"):
            selected = labels == market
            if selected.any():
                groups.append((market.lower(), selected))

    blocks: list[np.ndarray] = []
    names: list[str] = []
    for group_name, group_mask in groups:
        for feature_index, feature_name in enumerate(panel.feature_names):
            block, statistics = _masked_statistics(
                panel.values[:, group_mask, feature_index],
                panel.available[:, group_mask, feature_index],
            )
            blocks.append(block)
            names.extend(
                f"intraday_{group_name}_{statistic}:{feature_name}"
                for statistic in statistics
            )

        return_index = panel.feature_names.index("early_return")
        returns = panel.values[:, group_mask, return_index].astype(np.float64)
        valid = panel.available[:, group_mask, return_index] & np.isfinite(returns)
        count = valid.sum(axis=1).astype(np.float64)

        def fraction(condition: np.ndarray) -> np.ndarray:
            return np.divide(
                (valid & condition).sum(axis=1),
                count,
                out=np.zeros_like(count),
                where=count > 0.0,
            )

        mean_absolute = np.divide(
            np.where(valid, np.abs(returns), 0.0).sum(axis=1),
            count,
            out=np.zeros_like(count),
            where=count > 0.0,
        )
        mean_return = np.divide(
            np.where(valid, returns, 0.0).sum(axis=1),
            count,
            out=np.zeros_like(count),
            where=count > 0.0,
        )
        coherence = np.divide(
            mean_return,
            mean_absolute,
            out=np.zeros_like(mean_return),
            where=mean_absolute > 1e-12,
        )
        breadth = np.column_stack(
            (
                fraction(returns > 0.0),
                fraction(returns < 0.0),
                fraction(returns >= 0.005),
                fraction(returns <= -0.005),
                fraction(returns >= 0.01),
                fraction(returns <= -0.01),
                mean_absolute,
                coherence,
            )
        ).astype(np.float32)
        blocks.append(breadth)
        names.extend(
            f"intraday_{group_name}_{name}"
            for name in (
                "positive_fraction",
                "negative_fraction",
                "up_50bp_fraction",
                "down_50bp_fraction",
                "up_100bp_fraction",
                "down_100bp_fraction",
                "mean_absolute_return",
                "return_coherence",
            )
        )

    values = np.concatenate(blocks, axis=1)
    if not np.isfinite(values).all():
        raise ValueError("intraday market design contains non-finite values")
    return IntradayMarketDesign(panel.dates, values, tuple(names))


def remaining_session_returns(
    panel: IntradayWindowPanel,
    daily_close: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """Return decision-price-to-close labels, never open-to-close labels."""

    close = daily_close.reindex(index=panel.dates, columns=panel.tickers).to_numpy(
        dtype=np.float64
    )
    decision = panel.decision_price.astype(np.float64)
    valid = (
        panel.window_complete
        & np.isfinite(decision)
        & (decision > 0.0)
        & np.isfinite(close)
        & (close > 0.0)
    )
    result = np.full(decision.shape, np.nan, dtype=np.float32)
    result[valid] = (close[valid] / decision[valid] - 1.0).astype(np.float32)
    return result, valid
