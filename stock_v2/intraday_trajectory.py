from __future__ import annotations

from dataclasses import dataclass
from datetime import time as wall_time
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from stock_v2.kiwoom_minute import KST, audit_kiwoom_minute_frame


INTRADAY_TRAJECTORY_FEATURE_NAMES_V1 = (
    "clock_fraction",
    "gap_open",
    "return_from_prev_close",
    "return_from_open",
    "return_5m",
    "return_15m",
    "return_30m",
    "cumulative_range",
    "realized_absolute_return_15m",
    "realized_absolute_return_30m",
    "log_cumulative_volume",
    "log_cumulative_traded_value",
    "log_recent_volume_5m",
    "log_recent_volume_15m",
    "price_to_vwap",
    "cumulative_volume_shock_20",
    "cumulative_value_shock_20",
    "recent_volume_5m_shock_20",
    "recent_volume_15m_shock_20",
    "realized_absolute_return_15m_shock_20",
)

INTRADAY_TRAJECTORY_FEATURE_NAMES = INTRADAY_TRAJECTORY_FEATURE_NAMES_V1 + (
    "missing_intervals_since_previous_observation",
    "cumulative_missing_interval_fraction",
)

INTRADAY_TRAJECTORY_TARGET_NAMES = (
    "endpoint_return",
    "mfe",
    "mae",
    "realized_absolute_return",
    "future_range",
    "time_to_peak_fraction",
    "time_to_trough_fraction",
    "future_volume_shock_20",
)

SYSTEMIC_TRAJECTORY_TARGET_NAMES = (
    "median_return",
    "mean_return",
    "return_energy",
    "mean_absolute_return",
    "return_dispersion",
    "positive_breadth",
    "negative_breadth",
    "move_50bp_breadth",
    "direction_coherence",
    "median_mfe",
    "median_mae",
    "median_future_range",
    "state_change_energy",
    "median_future_volume_shock",
    "volume_expansion_breadth",
)


@dataclass(frozen=True)
class TickerIntradayTrajectory:
    timestamps: pd.DatetimeIndex
    feature_names: tuple[str, ...]
    values: np.ndarray
    available: np.ndarray
    decision_price: np.ndarray
    horizons_minutes: tuple[int, ...]
    target_names: tuple[str, ...]
    horizon_targets: np.ndarray
    horizon_available: np.ndarray
    close_targets: np.ndarray
    close_available: np.ndarray


@dataclass(frozen=True)
class IntradayTrajectoryPanel:
    timestamps: pd.DatetimeIndex
    session_dates: pd.DatetimeIndex
    decision_clock_minutes: np.ndarray
    tickers: tuple[str, ...]
    feature_names: tuple[str, ...]
    values: np.ndarray
    available: np.ndarray
    decision_price: np.ndarray
    horizons_minutes: tuple[int, ...]
    target_names: tuple[str, ...]
    horizon_targets: np.ndarray
    horizon_available: np.ndarray
    close_targets: np.ndarray
    close_available: np.ndarray


@dataclass(frozen=True)
class SystemicIntradayTrajectory:
    horizon_labels: tuple[str, ...]
    target_names: tuple[str, ...]
    values: np.ndarray
    available: np.ndarray


def _parse_clock(value: str | wall_time, label: str) -> wall_time:
    if isinstance(value, wall_time):
        result = value
    else:
        try:
            result = wall_time.fromisoformat(str(value))
        except ValueError as exc:
            raise ValueError(f"{label} must use HH:MM or HH:MM:SS") from exc
    if result.second or result.microsecond:
        raise ValueError(f"{label} must align to a whole minute")
    return result


def _clock_minutes(value: wall_time) -> int:
    return value.hour * 60 + value.minute


def _causal_same_clock_shock(
    values: np.ndarray,
    timestamps: pd.DatetimeIndex,
    *,
    rolling_window: int,
    min_history: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(timestamps),):
        raise ValueError("same-clock shock values must match timestamps")
    output = np.full(values.shape, np.nan, dtype=np.float64)
    clocks = timestamps.hour * 60 + timestamps.minute
    for clock in np.unique(clocks):
        rows = np.flatnonzero(clocks == clock)
        series = pd.Series(values[rows], index=np.arange(len(rows)))
        baseline = series.shift(1).rolling(
            window=int(rolling_window), min_periods=int(min_history)
        ).median()
        output[rows] = series.to_numpy(dtype=np.float64) - baseline.to_numpy(
            dtype=np.float64
        )
    return output


def _path_targets(
    decision_price: float,
    path: pd.DataFrame,
    elapsed_minutes: np.ndarray,
) -> tuple[np.ndarray, float]:
    if path.empty or len(path) != len(elapsed_minutes):
        raise ValueError("trajectory target path and elapsed times must be aligned")
    return _path_targets_arrays(
        decision_price,
        path["High"].to_numpy(dtype=np.float64),
        path["Low"].to_numpy(dtype=np.float64),
        path["Close"].to_numpy(dtype=np.float64),
        path["Volume"].to_numpy(dtype=np.float64),
        elapsed_minutes,
    )


def _path_targets_arrays(
    decision_price: float,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    volume: np.ndarray,
    elapsed_minutes: np.ndarray,
) -> tuple[np.ndarray, float]:
    high = np.asarray(high, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    close = np.asarray(close, dtype=np.float64)
    volume = np.asarray(volume, dtype=np.float64)
    elapsed_minutes = np.asarray(elapsed_minutes, dtype=np.float64)
    if not (
        len(high)
        and high.shape == low.shape == close.shape == volume.shape == elapsed_minutes.shape
    ):
        raise ValueError("trajectory target arrays must be non-empty and aligned")
    if (elapsed_minutes <= 0).any() or not np.all(np.diff(elapsed_minutes) > 0):
        raise ValueError("trajectory target elapsed times must be positive and sorted")
    close_path = np.concatenate(([float(decision_price)], close))
    step_returns = close_path[1:] / close_path[:-1] - 1.0
    peak_index = int(np.argmax(high))
    trough_index = int(np.argmin(low))
    total_minutes = float(elapsed_minutes[-1])
    target = np.asarray(
        (
            close[-1] / decision_price - 1.0,
            high[peak_index] / decision_price - 1.0,
            low[trough_index] / decision_price - 1.0,
            np.abs(step_returns).sum(),
            max(float(high.max()), decision_price)
            / min(float(low.min()), decision_price)
            - 1.0,
            float(elapsed_minutes[peak_index]) / total_minutes,
            float(elapsed_minutes[trough_index]) / total_minutes,
        ),
        dtype=np.float64,
    )
    return target, float(volume.sum())


def _empty_ticker_trajectory(
    horizons_minutes: tuple[int, ...],
) -> TickerIntradayTrajectory:
    feature_count = len(INTRADAY_TRAJECTORY_FEATURE_NAMES)
    target_count = len(INTRADAY_TRAJECTORY_TARGET_NAMES)
    horizon_count = len(horizons_minutes)
    return TickerIntradayTrajectory(
        timestamps=pd.DatetimeIndex([], tz=KST, name="DecisionTimestamp"),
        feature_names=INTRADAY_TRAJECTORY_FEATURE_NAMES,
        values=np.empty((0, feature_count), dtype=np.float32),
        available=np.empty((0, feature_count), dtype=bool),
        decision_price=np.empty(0, dtype=np.float32),
        horizons_minutes=horizons_minutes,
        target_names=INTRADAY_TRAJECTORY_TARGET_NAMES,
        horizon_targets=np.empty(
            (0, horizon_count, target_count), dtype=np.float32
        ),
        horizon_available=np.empty(
            (0, horizon_count, target_count), dtype=bool
        ),
        close_targets=np.empty((0, target_count), dtype=np.float32),
        close_available=np.empty((0, target_count), dtype=bool),
    )


def build_ticker_intraday_trajectory(
    frame: pd.DataFrame,
    *,
    interval_minutes: int,
    timestamp_semantics: str,
    horizons_minutes: Sequence[int] = (5, 15, 30, 60),
    decision_start: str | wall_time = "09:15",
    decision_end: str | wall_time = "15:15",
    rolling_window: int = 20,
    min_history: int = 10,
) -> TickerIntradayTrajectory:
    """Build causal rolling inputs and strictly post-decision path labels.

    A start-labelled bar stamped at the decision time is always a target bar,
    never an input bar. Missing bars invalidate the affected sample; this
    function never forward-fills or fabricates an intraday path.
    """

    interval = int(interval_minutes)
    if interval <= 0:
        raise ValueError("interval_minutes must be positive")
    if timestamp_semantics not in {"start", "end"}:
        raise ValueError("timestamp_semantics must be start or end")
    horizons = tuple(int(value) for value in horizons_minutes)
    if (
        not horizons
        or tuple(sorted(horizons)) != horizons
        or len(set(horizons)) != len(horizons)
        or any(value <= 0 or value % interval for value in horizons)
    ):
        raise ValueError("horizons must be positive, unique, sorted interval multiples")
    if int(rolling_window) <= 0 or not 1 <= int(min_history) <= int(rolling_window):
        raise ValueError("rolling baseline requires 1 <= min_history <= rolling_window")

    start_clock = _parse_clock(decision_start, "decision_start")
    end_clock = _parse_clock(decision_end, "decision_end")
    open_minute = 9 * 60
    continuous_close_minute = 15 * 60 + 20
    close_auction_minute = 15 * 60 + 30
    start_minute = _clock_minutes(start_clock)
    end_minute = _clock_minutes(end_clock)
    if (
        start_minute < open_minute + interval
        or end_minute > continuous_close_minute - interval
        or start_minute > end_minute
        or (start_minute - open_minute) % interval
        or (end_minute - open_minute) % interval
    ):
        raise ValueError("decision clocks must align inside the continuous KRX session")

    audit_kiwoom_minute_frame(frame, regular_session_only=True)
    if frame.empty:
        return _empty_ticker_trajectory(horizons)
    local = frame.copy()
    local.index = local.index.tz_convert(KST)
    session_keys = local.index.normalize()
    grouped = {
        pd.Timestamp(day): bars.sort_index()
        for day, bars in local.groupby(session_keys, sort=True)
    }

    records: list[dict[str, object]] = []
    previous_close = np.nan
    for session_day, bars in grouped.items():
        session_start = session_day + pd.Timedelta(open_minute, unit="minute")
        if timestamp_semantics == "start":
            raw_offsets = np.arange(
                0,
                continuous_close_minute - open_minute,
                interval,
                dtype=np.int64,
            )
            completion_offsets = raw_offsets + interval
        else:
            raw_offsets = np.arange(
                interval,
                continuous_close_minute - open_minute + 1,
                interval,
                dtype=np.int64,
            )
            completion_offsets = raw_offsets.copy()
        expected_continuous = session_start + pd.to_timedelta(
            raw_offsets, unit="minute"
        )
        completion_times = session_start + pd.to_timedelta(
            completion_offsets, unit="minute"
        )
        close_auction = session_day + pd.Timedelta(
            close_auction_minute, unit="minute"
        )
        continuous = bars.reindex(expected_continuous)
        present = np.asarray(expected_continuous.isin(bars.index), dtype=bool)
        history_complete = np.logical_and.accumulate(present)
        open_values = continuous["Open"].to_numpy(dtype=np.float64)
        high_values = continuous["High"].to_numpy(dtype=np.float64)
        low_values = continuous["Low"].to_numpy(dtype=np.float64)
        close_values = continuous["Close"].to_numpy(dtype=np.float64)
        volume_values = continuous["Volume"].to_numpy(dtype=np.float64)
        session_open = float(open_values[0]) if present[0] else np.nan
        previous_values = np.concatenate(([session_open], close_values[:-1]))
        step_returns = close_values / previous_values - 1.0
        typical_values = 0.25 * (
            open_values + high_values + low_values + close_values
        )
        traded_values = typical_values * volume_values
        cumulative_volume = np.cumsum(np.where(present, volume_values, 0.0))
        cumulative_value = np.cumsum(np.where(present, traded_values, 0.0))
        cumulative_absolute = np.cumsum(
            np.where(present & np.isfinite(step_returns), np.abs(step_returns), 0.0)
        )
        cumulative_high = np.maximum.accumulate(
            np.where(present, high_values, -np.inf)
        )
        cumulative_low = np.minimum.accumulate(
            np.where(present, low_values, np.inf)
        )
        auction_present = close_auction in bars.index
        if auction_present:
            auction = bars.loc[close_auction]

        for decision_minute in range(start_minute, end_minute + 1, interval):
            decision_timestamp = session_day + pd.Timedelta(
                decision_minute, unit="minute"
            )
            completed_count = (decision_minute - open_minute) // interval
            if not np.isfinite(session_open) or session_open <= 0.0:
                continue
            decision_index = completed_count - 1
            if not present[decision_index]:
                continue
            decision_price = float(close_values[decision_index])
            if not np.isfinite(decision_price) or decision_price <= 0.0:
                continue
            observed_volume = float(cumulative_volume[decision_index])
            observed_value = float(cumulative_value[decision_index])
            vwap = (
                observed_value / observed_volume
                if observed_volume > 0.0
                else np.nan
            )

            def trailing_return(bars_back: int) -> float:
                if completed_count < bars_back:
                    return np.nan
                start_index = completed_count - bars_back
                if not present[start_index:completed_count].all():
                    return np.nan
                if start_index > 0 and not present[start_index - 1]:
                    return np.nan
                boundary = (
                    session_open
                    if start_index == 0
                    else close_values[start_index - 1]
                )
                return float(decision_price / boundary - 1.0)

            def trailing_absolute(bars_back: int) -> float:
                if completed_count < bars_back:
                    return np.nan
                start_index = completed_count - bars_back
                if not present[start_index:completed_count].all():
                    return np.nan
                if start_index > 0 and not present[start_index - 1]:
                    return np.nan
                selected_returns = step_returns[start_index:completed_count]
                if not np.isfinite(selected_returns).all():
                    return np.nan
                return float(np.abs(selected_returns).sum())

            prior_present = np.flatnonzero(present[:decision_index])
            missing_since_previous = (
                decision_index - int(prior_present[-1]) - 1
                if len(prior_present)
                else decision_index
            )
            cumulative_missing_fraction = float(
                1.0 - present[:completed_count].mean()
            )
            cumulative_complete = bool(history_complete[decision_index])

            raw_features = np.full(
                len(INTRADAY_TRAJECTORY_FEATURE_NAMES), np.nan, dtype=np.float64
            )
            feature = {
                "clock_fraction": (decision_minute - open_minute)
                / (continuous_close_minute - open_minute),
                "gap_open": (
                    session_open / previous_close - 1.0
                    if np.isfinite(previous_close) and previous_close > 0.0
                    else np.nan
                ),
                "return_from_prev_close": (
                    decision_price / previous_close - 1.0
                    if np.isfinite(previous_close) and previous_close > 0.0
                    else np.nan
                ),
                "return_from_open": decision_price / session_open - 1.0,
                "return_5m": trailing_return(1),
                "return_15m": trailing_return(max(1, 15 // interval)),
                "return_30m": trailing_return(max(1, 30 // interval)),
                "cumulative_range": (
                    float(cumulative_high[decision_index])
                    / float(cumulative_low[decision_index])
                    - 1.0
                    if cumulative_complete
                    else np.nan
                ),
                "realized_absolute_return_15m": trailing_absolute(
                    max(1, 15 // interval)
                ),
                "realized_absolute_return_30m": trailing_absolute(
                    max(1, 30 // interval)
                ),
                "log_cumulative_volume": (
                    np.log1p(observed_volume) if cumulative_complete else np.nan
                ),
                "log_cumulative_traded_value": (
                    np.log1p(observed_value) if cumulative_complete else np.nan
                ),
                "log_recent_volume_5m": np.log1p(float(volume_values[decision_index])),
                "log_recent_volume_15m": (
                    np.log1p(
                        float(
                            volume_values[
                                completed_count - max(1, 15 // interval) : completed_count
                            ].sum()
                        )
                    )
                    if completed_count >= max(1, 15 // interval)
                    and present[
                        completed_count
                        - max(1, 15 // interval) : completed_count
                    ].all()
                    else np.nan
                ),
                "price_to_vwap": (
                    decision_price / vwap - 1.0
                    if cumulative_complete and np.isfinite(vwap) and vwap > 0.0
                    else np.nan
                ),
                "missing_intervals_since_previous_observation": float(
                    missing_since_previous
                ),
                "cumulative_missing_interval_fraction": cumulative_missing_fraction,
            }
            for name, value in feature.items():
                raw_features[INTRADAY_TRAJECTORY_FEATURE_NAMES.index(name)] = value

            horizon_core = np.full(
                (len(horizons), len(INTRADAY_TRAJECTORY_TARGET_NAMES) - 1),
                np.nan,
                dtype=np.float64,
            )
            horizon_volume = np.full(len(horizons), np.nan, dtype=np.float64)
            for horizon_index, horizon in enumerate(horizons):
                step_count = horizon // interval
                end_index = completed_count + step_count
                if end_index > len(expected_continuous):
                    continue
                endpoint_index = end_index - 1
                if present[endpoint_index]:
                    horizon_core[horizon_index, 0] = (
                        float(close_values[endpoint_index]) / decision_price - 1.0
                    )
                if not present[completed_count:end_index].all():
                    continue
                selected = slice(completed_count, end_index)
                elapsed = np.arange(1, step_count + 1, dtype=np.float64) * interval
                core, future_volume = _path_targets_arrays(
                    decision_price,
                    high_values[selected],
                    low_values[selected],
                    close_values[selected],
                    volume_values[selected],
                    elapsed,
                )
                horizon_core[horizon_index] = core
                horizon_volume[horizon_index] = future_volume

            close_core = np.full(
                len(INTRADAY_TRAJECTORY_TARGET_NAMES) - 1,
                np.nan,
                dtype=np.float64,
            )
            close_volume = np.nan
            if auction_present:
                close_core[0] = float(auction["Close"]) / decision_price - 1.0
            if present[completed_count:].all() and auction_present:
                remaining_high = np.concatenate(
                    (high_values[completed_count:], [float(auction["High"])])
                )
                remaining_low = np.concatenate(
                    (low_values[completed_count:], [float(auction["Low"])])
                )
                remaining_close = np.concatenate(
                    (close_values[completed_count:], [float(auction["Close"])])
                )
                remaining_volume = np.concatenate(
                    (volume_values[completed_count:], [float(auction["Volume"])])
                )
                continuous_elapsed = (
                    completion_offsets[completed_count:]
                    - (decision_minute - open_minute)
                ).astype(np.float64)
                close_elapsed = np.concatenate(
                    (
                        continuous_elapsed,
                        np.asarray(
                            [
                                (close_auction - decision_timestamp).total_seconds()
                                / 60.0
                            ]
                        ),
                    )
                )
                close_core, close_volume = _path_targets_arrays(
                    decision_price,
                    remaining_high,
                    remaining_low,
                    remaining_close,
                    remaining_volume,
                    close_elapsed,
                )

            records.append(
                {
                    "timestamp": decision_timestamp,
                    "features": raw_features,
                    "decision_price": decision_price,
                    "horizon_core": horizon_core,
                    "horizon_volume": horizon_volume,
                    "close_core": close_core,
                    "close_volume": close_volume,
                }
            )

        if auction_present:
            candidate = float(auction["Close"])
            if np.isfinite(candidate) and candidate > 0.0:
                previous_close = candidate

    if not records:
        return _empty_ticker_trajectory(horizons)
    records.sort(key=lambda row: row["timestamp"])
    timestamps = pd.DatetimeIndex(
        [row["timestamp"] for row in records], name="DecisionTimestamp"
    )
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("intraday trajectory timestamps must be unique and sorted")
    values = np.stack([row["features"] for row in records]).astype(np.float64)

    shock_sources = {
        "cumulative_volume_shock_20": values[
            :, INTRADAY_TRAJECTORY_FEATURE_NAMES.index("log_cumulative_volume")
        ],
        "cumulative_value_shock_20": values[
            :, INTRADAY_TRAJECTORY_FEATURE_NAMES.index(
                "log_cumulative_traded_value"
            )
        ],
        "recent_volume_5m_shock_20": values[
            :, INTRADAY_TRAJECTORY_FEATURE_NAMES.index("log_recent_volume_5m")
        ],
        "recent_volume_15m_shock_20": values[
            :, INTRADAY_TRAJECTORY_FEATURE_NAMES.index("log_recent_volume_15m")
        ],
        "realized_absolute_return_15m_shock_20": np.log1p(
            10_000.0
            * values[
                :,
                INTRADAY_TRAJECTORY_FEATURE_NAMES.index(
                    "realized_absolute_return_15m"
                ),
            ]
        ),
    }
    for name, source in shock_sources.items():
        values[:, INTRADAY_TRAJECTORY_FEATURE_NAMES.index(name)] = (
            _causal_same_clock_shock(
                source,
                timestamps,
                rolling_window=rolling_window,
                min_history=min_history,
            )
        )

    target_count = len(INTRADAY_TRAJECTORY_TARGET_NAMES)
    horizon_targets = np.full(
        (len(records), len(horizons), target_count), np.nan, dtype=np.float64
    )
    close_targets = np.full((len(records), target_count), np.nan, dtype=np.float64)
    for row_index, record in enumerate(records):
        horizon_targets[row_index, :, :-1] = record["horizon_core"]
        close_targets[row_index, :-1] = record["close_core"]
    volume_target = target_count - 1
    horizon_raw_volume = np.stack(
        [row["horizon_volume"] for row in records]
    ).astype(np.float64)
    for horizon_index in range(len(horizons)):
        horizon_targets[:, horizon_index, volume_target] = (
            _causal_same_clock_shock(
                np.log1p(horizon_raw_volume[:, horizon_index]),
                timestamps,
                rolling_window=rolling_window,
                min_history=min_history,
            )
        )
    close_targets[:, volume_target] = _causal_same_clock_shock(
        np.log1p(np.asarray([row["close_volume"] for row in records], dtype=float)),
        timestamps,
        rolling_window=rolling_window,
        min_history=min_history,
    )

    return TickerIntradayTrajectory(
        timestamps=timestamps,
        feature_names=INTRADAY_TRAJECTORY_FEATURE_NAMES,
        values=values.astype(np.float32),
        available=np.isfinite(values),
        decision_price=np.asarray(
            [row["decision_price"] for row in records], dtype=np.float32
        ),
        horizons_minutes=horizons,
        target_names=INTRADAY_TRAJECTORY_TARGET_NAMES,
        horizon_targets=horizon_targets.astype(np.float32),
        horizon_available=np.isfinite(horizon_targets),
        close_targets=close_targets.astype(np.float32),
        close_available=np.isfinite(close_targets),
    )


def build_intraday_trajectory_panel(
    trajectories: Mapping[str, TickerIntradayTrajectory],
    *,
    tickers: Sequence[str],
) -> IntradayTrajectoryPanel:
    ticker_tuple = tuple(str(ticker) for ticker in tickers)
    if not ticker_tuple or len(set(ticker_tuple)) != len(ticker_tuple):
        raise ValueError("trajectory panel tickers must be non-empty and unique")
    populated = [
        trajectories[ticker]
        for ticker in ticker_tuple
        if ticker in trajectories and len(trajectories[ticker].timestamps)
    ]
    if not populated:
        raise ValueError("trajectory panel has no populated ticker inputs")
    template = populated[0]
    for trajectory in populated[1:]:
        if (
            trajectory.feature_names != template.feature_names
            or trajectory.horizons_minutes != template.horizons_minutes
            or trajectory.target_names != template.target_names
        ):
            raise ValueError("trajectory ticker contracts are inconsistent")
    timestamps = populated[0].timestamps
    for trajectory in populated[1:]:
        timestamps = timestamps.union(trajectory.timestamps)
    timestamps = timestamps.sort_values()
    time_count = len(timestamps)
    node_count = len(ticker_tuple)
    feature_count = len(template.feature_names)
    horizon_count = len(template.horizons_minutes)
    target_count = len(template.target_names)
    values = np.full((time_count, node_count, feature_count), np.nan, dtype=np.float32)
    available = np.zeros(values.shape, dtype=bool)
    decision_price = np.full((time_count, node_count), np.nan, dtype=np.float32)
    horizon_targets = np.full(
        (time_count, node_count, horizon_count, target_count),
        np.nan,
        dtype=np.float32,
    )
    horizon_available = np.zeros(horizon_targets.shape, dtype=bool)
    close_targets = np.full(
        (time_count, node_count, target_count), np.nan, dtype=np.float32
    )
    close_available = np.zeros(close_targets.shape, dtype=bool)
    for node_index, ticker in enumerate(ticker_tuple):
        trajectory = trajectories.get(ticker)
        if trajectory is None or not len(trajectory.timestamps):
            continue
        rows = timestamps.get_indexer(trajectory.timestamps)
        if (rows < 0).any():
            raise RuntimeError("trajectory timestamps failed to align")
        values[rows, node_index] = trajectory.values
        available[rows, node_index] = trajectory.available
        decision_price[rows, node_index] = trajectory.decision_price
        horizon_targets[rows, node_index] = trajectory.horizon_targets
        horizon_available[rows, node_index] = trajectory.horizon_available
        close_targets[rows, node_index] = trajectory.close_targets
        close_available[rows, node_index] = trajectory.close_available
    local = timestamps.tz_convert(KST)
    return IntradayTrajectoryPanel(
        timestamps=timestamps,
        session_dates=local.tz_localize(None).normalize(),
        decision_clock_minutes=np.asarray(
            local.hour * 60 + local.minute, dtype=np.int16
        ),
        tickers=ticker_tuple,
        feature_names=template.feature_names,
        values=values,
        available=available,
        decision_price=decision_price,
        horizons_minutes=template.horizons_minutes,
        target_names=template.target_names,
        horizon_targets=horizon_targets,
        horizon_available=horizon_available,
        close_targets=close_targets,
        close_available=close_available,
    )


def summarize_systemic_intraday_trajectory(
    panel: IntradayTrajectoryPanel,
    *,
    min_nodes: int = 20,
) -> SystemicIntradayTrajectory:
    if not 2 <= int(min_nodes) <= len(panel.tickers):
        raise ValueError("systemic trajectory min_nodes must fit the stock axis")
    endpoint_index = panel.target_names.index("endpoint_return")
    mfe_index = panel.target_names.index("mfe")
    mae_index = panel.target_names.index("mae")
    range_index = panel.target_names.index("future_range")
    volume_index = panel.target_names.index("future_volume_shock_20")
    targets = np.concatenate(
        (panel.horizon_targets, panel.close_targets[:, :, None, :]), axis=2
    ).astype(np.float64)
    target_available = np.concatenate(
        (panel.horizon_available, panel.close_available[:, :, None, :]), axis=2
    )
    labels = tuple(f"{value}m" for value in panel.horizons_minutes) + ("close",)
    values = np.full(
        (len(panel.timestamps), len(labels), len(SYSTEMIC_TRAJECTORY_TARGET_NAMES)),
        np.nan,
        dtype=np.float64,
    )
    available = np.zeros(values.shape, dtype=bool)
    for time_index in range(len(panel.timestamps)):
        for horizon_index in range(len(labels)):
            endpoint_valid = (
                target_available[time_index, :, horizon_index, endpoint_index]
                & np.isfinite(targets[time_index, :, horizon_index, endpoint_index])
            )
            if int(endpoint_valid.sum()) >= int(min_nodes):
                node_return = targets[
                    time_index, endpoint_valid, horizon_index, endpoint_index
                ]
                mean_absolute = float(np.mean(np.abs(node_return)))
                mean_return = float(np.mean(node_return))
                return_core = (
                    float(np.median(node_return)),
                    mean_return,
                    float(np.sqrt(np.mean(np.square(node_return)))),
                    mean_absolute,
                    float(np.std(node_return)),
                    float(np.mean(node_return > 0.0)),
                    float(np.mean(node_return < 0.0)),
                    float(np.mean(np.abs(node_return) >= 0.005)),
                    abs(mean_return) / mean_absolute if mean_absolute > 1e-12 else 0.0,
                )
                values[time_index, horizon_index, : len(return_core)] = return_core
                available[time_index, horizon_index, : len(return_core)] = True
            path_valid = (
                endpoint_valid
                & target_available[time_index, :, horizon_index, mfe_index]
                & target_available[time_index, :, horizon_index, mae_index]
                & target_available[time_index, :, horizon_index, range_index]
            )
            if int(path_valid.sum()) >= int(min_nodes):
                node_return = targets[
                    time_index, path_valid, horizon_index, endpoint_index
                ]
                node_mfe = targets[time_index, path_valid, horizon_index, mfe_index]
                node_mae = targets[time_index, path_valid, horizon_index, mae_index]
                node_range = targets[
                    time_index, path_valid, horizon_index, range_index
                ]
                path_core = (
                    float(np.median(node_mfe)),
                    float(np.median(node_mae)),
                    float(np.median(node_range)),
                    float(
                        np.sqrt(
                            np.mean(
                                np.square(node_return)
                                + 0.25 * np.square(node_range)
                            )
                        )
                    ),
                )
                values[time_index, horizon_index, 9:13] = path_core
                available[time_index, horizon_index, 9:13] = True
            volume_valid = (
                target_available[time_index, :, horizon_index, volume_index]
                & np.isfinite(targets[time_index, :, horizon_index, volume_index])
            )
            if int(volume_valid.sum()) >= int(min_nodes):
                volume = targets[time_index, volume_valid, horizon_index, volume_index]
                values[time_index, horizon_index, -2:] = (
                    float(np.median(volume)),
                    float(np.mean(volume >= np.log(1.5))),
                )
                available[time_index, horizon_index, -2:] = True
    return SystemicIntradayTrajectory(
        horizon_labels=labels,
        target_names=SYSTEMIC_TRAJECTORY_TARGET_NAMES,
        values=values.astype(np.float32),
        available=available,
    )
