from __future__ import annotations

from dataclasses import replace
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from stock_v2.intraday_trajectory import IntradayTrajectoryPanel
from stock_v2.kiwoom_minute import KIWOOM_MINUTE_COLUMNS, KST


SHOCK_SOURCE_FEATURES = {
    "cumulative_volume_shock_20": "log_cumulative_volume",
    "cumulative_value_shock_20": "log_cumulative_traded_value",
    "recent_volume_5m_shock_20": "log_recent_volume_5m",
    "recent_volume_15m_shock_20": "log_recent_volume_15m",
    "realized_absolute_return_15m_shock_20": "realized_absolute_return_15m",
}


def synthetic_prior_close_frame(session: str, close: float) -> pd.DataFrame:
    prior_session = pd.Timestamp(session).normalize()
    value = float(close)
    if not np.isfinite(value) or value <= 0.0:
        raise ValueError("prior raw close must be finite and positive")
    timestamp = prior_session.tz_localize(KST) + pd.Timedelta(
        15 * 60 + 30, unit="minute"
    )
    row = {
        "Open": value,
        "High": value,
        "Low": value,
        "Close": value,
        "Volume": 0.0,
        "CumulativeVolume": np.nan,
        "PreviousChange": np.nan,
        "PreviousChangeSign": np.nan,
    }
    frame = pd.DataFrame([row], index=pd.DatetimeIndex([timestamp], name="Timestamp"))
    return frame.reindex(columns=KIWOOM_MINUTE_COLUMNS).astype(float)


def _clock_rows(timestamps_utc_ns: np.ndarray) -> dict[int, int]:
    timestamps = pd.to_datetime(
        np.asarray(timestamps_utc_ns, dtype=np.int64), unit="ns", utc=True
    ).tz_convert(KST)
    clocks = np.asarray(timestamps.hour * 60 + timestamps.minute, dtype=np.int16)
    if len(clocks) != len(set(int(value) for value in clocks)):
        raise ValueError("historical intraday day has duplicate decision clocks")
    return {int(clock): int(row) for row, clock in enumerate(clocks)}


def apply_historical_same_clock_shocks(
    panel: IntradayTrajectoryPanel,
    historical_release: Any,
    history_dates: Sequence[str],
    *,
    rolling_window: int = 20,
    min_history: int = 10,
) -> tuple[IntradayTrajectoryPanel, dict[str, Any]]:
    if panel.feature_names != historical_release.feature_names:
        raise ValueError("live and historical trajectory feature contracts differ")
    if panel.tickers != historical_release.tickers:
        raise ValueError("live and historical trajectory ticker orders differ")
    if int(rolling_window) <= 0 or not 1 <= int(min_history) <= int(rolling_window):
        raise ValueError("live shock history requires 1 <= min_history <= rolling_window")
    dates = tuple(str(value) for value in history_dates)
    if not dates or dates != tuple(sorted(set(dates))):
        raise ValueError("live shock history dates must be non-empty, unique, and sorted")
    missing = [date for date in dates if date not in historical_release.records]
    if missing:
        raise ValueError(f"live shock history date is absent: {missing[0]}")

    clocks = np.asarray(panel.decision_clock_minutes, dtype=np.int16)
    clock_values = tuple(int(value) for value in clocks)
    if len(clock_values) != len(set(clock_values)):
        raise ValueError("live trajectory contains duplicate decision clocks")
    source_names = tuple(SHOCK_SOURCE_FEATURES.values())
    source_indices = np.asarray(
        [panel.feature_names.index(name) for name in source_names], dtype=np.int64
    )
    history = np.full(
        (len(dates), len(clocks), len(panel.tickers), len(source_names)),
        np.nan,
        dtype=np.float32,
    )
    record_present = np.zeros(
        (len(dates), len(clocks), len(panel.tickers)), dtype=bool
    )
    for date_index, date in enumerate(dates):
        day = historical_release.load(date)
        row_by_clock = _clock_rows(day["timestamps_utc_ns"])
        for clock_index, clock in enumerate(clock_values):
            row = row_by_clock.get(clock)
            if row is None:
                continue
            day_values = np.asarray(day["node_values"][row], dtype=np.float32)
            day_available = np.asarray(day["node_available"][row], dtype=bool)
            decision_price = np.asarray(day["decision_price"][row], dtype=np.float32)
            present = np.isfinite(decision_price) & (decision_price > 0.0)
            selected = day_values[:, source_indices]
            selected_available = day_available[:, source_indices] & np.isfinite(selected)
            history[date_index, clock_index] = np.where(
                selected_available,
                selected,
                np.nan,
            )
            record_present[date_index, clock_index] = present

    realized_source = source_names.index("realized_absolute_return_15m")
    history[..., realized_source] = np.log1p(
        10_000.0 * history[..., realized_source]
    )
    values = np.asarray(panel.values, dtype=np.float32).copy()
    available = np.asarray(panel.available, dtype=bool).copy()
    baseline_counts: list[int] = []
    assigned = 0
    for clock_index, clock in enumerate(clock_values):
        current_source = values[clock_index][:, source_indices].astype(
            np.float64, copy=True
        )
        current_source[:, realized_source] = np.log1p(
            10_000.0 * current_source[:, realized_source]
        )
        for node in range(len(panel.tickers)):
            observed_rows = np.flatnonzero(record_present[:, clock_index, node])
            selected_rows = observed_rows[-int(rolling_window) :]
            prior = history[selected_rows, clock_index, node]
            for source_position, shock_name in enumerate(SHOCK_SOURCE_FEATURES):
                finite = np.isfinite(prior[:, source_position])
                count = int(finite.sum())
                baseline_counts.append(count)
                shock_index = panel.feature_names.index(shock_name)
                current = current_source[node, source_position]
                valid = count >= int(min_history) and np.isfinite(current)
                if valid:
                    baseline = float(np.median(prior[finite, source_position]))
                    values[clock_index, node, shock_index] = current - baseline
                    available[clock_index, node, shock_index] = True
                    assigned += 1
                else:
                    values[clock_index, node, shock_index] = np.nan
                    available[clock_index, node, shock_index] = False
    result = replace(panel, values=values, available=available)
    counts = np.asarray(baseline_counts, dtype=np.int16)
    diagnostics = {
        "history_dates": len(dates),
        "rolling_window": int(rolling_window),
        "minimum_history": int(min_history),
        "current_clocks": list(clock_values),
        "assigned_shock_cells": int(assigned),
        "baseline_finite_count_minimum": int(counts.min()),
        "baseline_finite_count_median": float(np.median(counts)),
        "baseline_finite_count_maximum": int(counts.max()),
        "contract": "last_20_observed_decision_rows_same_clock_v1",
    }
    return result, diagnostics
