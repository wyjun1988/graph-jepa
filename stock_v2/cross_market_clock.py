from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from stock_v2.kiwoom_us import (
    US_MARKET_TIMEZONE,
    audit_kiwoom_us_daily_frame,
    audit_kiwoom_us_minute_frame,
)


KOREA_TIMEZONE = "Asia/Seoul"
UTC_TIMEZONE = "UTC"
US_EQUITY_CALENDAR = "XNYS"
US_SESSION_PHASES = ("regular", "premarket", "after_hours", "overnight")
EXCHANGE_CALENDARS_VERSION = xcals.__version__


@lru_cache(maxsize=4)
def _calendar(name: str = US_EQUITY_CALENDAR):
    return xcals.get_calendar(name)


def _aware_timestamp(value: object, *, label: str) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError(f"{label} must be timezone-aware")
    return timestamp


def _nonnegative_timedelta(value: object, *, label: str) -> pd.Timedelta:
    if isinstance(value, (int, float, np.integer, np.floating)):
        duration = pd.to_timedelta(value, unit="s")
    else:
        duration = pd.to_timedelta(value)
    if pd.isna(duration) or duration.value < 0:
        raise ValueError(f"{label} must be a non-negative duration")
    return duration


def _session_label(value: object) -> pd.Timestamp:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is not None:
        timestamp = timestamp.tz_localize(None)
    return timestamp.normalize()


def _session_bounds_utc(
    business_date: object, *, calendar_name: str
) -> tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]:
    calendar = _calendar(calendar_name)
    label = _session_label(business_date)
    if not calendar.is_session(label):
        raise ValueError(
            f"US business date is not a {calendar_name} session: "
            f"{label.date().isoformat()}"
        )
    opened = pd.Timestamp(calendar.session_open(label)).tz_convert(UTC_TIMEZONE)
    closed = pd.Timestamp(calendar.session_close(label)).tz_convert(UTC_TIMEZONE)
    return label, opened, closed


def _minute_of_day(timestamp: pd.Timestamp) -> int:
    return int(timestamp.hour * 60 + timestamp.minute)


def _classify_phase(
    source_et: pd.Timestamp,
    *,
    business_label: pd.Timestamp,
    session_open_utc: pd.Timestamp,
    session_close_utc: pd.Timestamp,
    calendar_name: str,
) -> str:
    source_utc = source_et.tz_convert(UTC_TIMEZONE)
    source_label = pd.Timestamp(source_et.date())
    calendar = _calendar(calendar_name)

    if calendar.is_session(source_label):
        local_open = pd.Timestamp(calendar.session_open(source_label)).tz_convert(
            UTC_TIMEZONE
        )
        local_close = pd.Timestamp(calendar.session_close(source_label)).tz_convert(
            UTC_TIMEZONE
        )
        if local_open <= source_utc <= local_close and source_label != business_label:
            raise ValueError(
                "US regular-session timestamp disagrees with Kiwoom business date"
            )

    if source_label == business_label:
        if session_open_utc <= source_utc <= session_close_utc:
            return "regular"
        minute = _minute_of_day(source_et)
        open_et = session_open_utc.tz_convert(US_MARKET_TIMEZONE)
        close_et = session_close_utc.tz_convert(US_MARKET_TIMEZONE)
        if 4 * 60 <= minute < _minute_of_day(open_et):
            return "premarket"
        if _minute_of_day(close_et) < minute < 20 * 60:
            return "after_hours"
    return "overnight"


def _phase_bucket(
    source_et: pd.Timestamp,
    *,
    phase: str,
    interval_minutes: int,
    session_open_utc: pd.Timestamp,
    session_close_utc: pd.Timestamp,
) -> str:
    source_utc = source_et.tz_convert(UTC_TIMEZONE)
    if phase == "regular":
        offset = int((source_utc - session_open_utc).total_seconds() // 60)
    elif phase == "premarket":
        offset = _minute_of_day(source_et) - 4 * 60
    elif phase == "after_hours":
        offset = int((source_utc - session_close_utc).total_seconds() // 60)
    else:
        offset = _minute_of_day(source_et)
    bucket = max(0, offset // interval_minutes)
    return f"{phase}:{bucket:04d}"


def annotate_us_minute_availability(
    frame: pd.DataFrame,
    *,
    interval_minutes: int,
    vendor_lag: object = "15s",
    calendar_name: str = US_EQUITY_CALENDAR,
) -> pd.DataFrame:
    """Attach causal availability metadata to raw Kiwoom US minute bars.

    Kiwoom does not document whether ``cntr_tm`` labels the start or end of a
    bucket. Treating it as the start and adding a full interval is conservative:
    a bar cannot enter a Korean decision snapshot before it is certainly closed.
    """

    interval = int(interval_minutes)
    if interval <= 0:
        raise ValueError("interval_minutes must be positive")
    lag = _nonnegative_timedelta(vendor_lag, label="vendor_lag")
    audit_kiwoom_us_minute_frame(frame, regular_session_only=False)
    result = frame.copy()
    if result.empty:
        for column in (
            "SourceTimestampET",
            "SourceTimestampUTC",
            "SourceTimestampKST",
            "BarEndUTC",
            "AvailableAtUTC",
            "AvailableAtKST",
            "SessionOpenUTC",
            "SessionCloseUTC",
            "SessionPhase",
            "PhaseBucket",
            "TimestampSemantics",
            "IntervalMinutes",
            "VendorLagMilliseconds",
        ):
            result[column] = pd.Series(index=result.index, dtype=object)
        return result

    source_et = result.index.tz_convert(US_MARKET_TIMEZONE)
    source_utc = source_et.tz_convert(UTC_TIMEZONE)
    bar_end_utc = source_utc + pd.to_timedelta(interval, unit="min")
    available_utc = bar_end_utc + lag

    opens: list[pd.Timestamp] = []
    closes: list[pd.Timestamp] = []
    phases: list[str] = []
    buckets: list[str] = []
    cache: dict[pd.Timestamp, tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]] = {}
    business_dates = pd.to_datetime(result["BusinessDate"])
    for source, business_date in zip(source_et, business_dates, strict=True):
        label = _session_label(business_date)
        bounds = cache.get(label)
        if bounds is None:
            bounds = _session_bounds_utc(label, calendar_name=calendar_name)
            cache[label] = bounds
        business_label, opened, closed = bounds
        phase = _classify_phase(
            pd.Timestamp(source),
            business_label=business_label,
            session_open_utc=opened,
            session_close_utc=closed,
            calendar_name=calendar_name,
        )
        opens.append(opened)
        closes.append(closed)
        phases.append(phase)
        buckets.append(
            _phase_bucket(
                pd.Timestamp(source),
                phase=phase,
                interval_minutes=interval,
                session_open_utc=opened,
                session_close_utc=closed,
            )
        )

    result["SourceTimestampET"] = source_et
    result["SourceTimestampUTC"] = source_utc
    result["SourceTimestampKST"] = source_utc.tz_convert(KOREA_TIMEZONE)
    result["BarEndUTC"] = bar_end_utc
    result["AvailableAtUTC"] = available_utc
    result["AvailableAtKST"] = available_utc.tz_convert(KOREA_TIMEZONE)
    result["SessionOpenUTC"] = pd.DatetimeIndex(opens)
    result["SessionCloseUTC"] = pd.DatetimeIndex(closes)
    result["SessionPhase"] = phases
    result["PhaseBucket"] = buckets
    result["TimestampSemantics"] = "conservative_bar_start"
    result["IntervalMinutes"] = interval
    result["VendorLagMilliseconds"] = int(lag.total_seconds() * 1000)
    return result


def available_us_minute_bars(
    frame: pd.DataFrame,
    decision_cutoff: object,
    *,
    interval_minutes: int,
    vendor_lag: object = "15s",
    calendar_name: str = US_EQUITY_CALENDAR,
) -> pd.DataFrame:
    cutoff_utc = _aware_timestamp(
        decision_cutoff, label="decision_cutoff"
    ).tz_convert(UTC_TIMEZONE)
    annotated = annotate_us_minute_availability(
        frame,
        interval_minutes=interval_minutes,
        vendor_lag=vendor_lag,
        calendar_name=calendar_name,
    )
    return annotated.loc[annotated["AvailableAtUTC"] <= cutoff_utc].copy()


def annotate_us_daily_availability(
    frame: pd.DataFrame,
    *,
    vendor_lag: object = "15min",
    finalization_sessions: int = 1,
    calendar_name: str = US_EQUITY_CALENDAR,
) -> pd.DataFrame:
    """Attach conservative availability times to Kiwoom ``usa06012`` rows.

    The endpoint's newest row can combine regular-session OHLCV with a mutable
    overnight ``cur_prc``. A one-session quarantine keeps that row out of a
    research release. Live previous-close features must instead be reconstructed
    from completed minute bars.
    """

    lag = _nonnegative_timedelta(vendor_lag, label="vendor_lag")
    quarantine = int(finalization_sessions)
    if quarantine < 0:
        raise ValueError("finalization_sessions must be non-negative")
    audit_kiwoom_us_daily_frame(frame)
    result = frame.copy()
    closes: list[pd.Timestamp] = []
    finalization_labels: list[pd.Timestamp] = []
    finalization_closes: list[pd.Timestamp] = []
    calendar = _calendar(calendar_name)
    for date in result.index:
        label, _opened, closed = _session_bounds_utc(
            date, calendar_name=calendar_name
        )
        closes.append(closed)
        finalized_label = label
        for _ in range(quarantine):
            finalized_label = pd.Timestamp(calendar.next_session(finalized_label))
        _label, _opened, finalized_close = _session_bounds_utc(
            finalized_label, calendar_name=calendar_name
        )
        finalization_labels.append(finalized_label)
        finalization_closes.append(finalized_close)
    close_utc = pd.DatetimeIndex(closes)
    if close_utc.tz is None:
        close_utc = close_utc.tz_localize(UTC_TIMEZONE)
    else:
        close_utc = close_utc.tz_convert(UTC_TIMEZONE)
    finalization_close_utc = pd.DatetimeIndex(finalization_closes)
    if finalization_close_utc.tz is None:
        finalization_close_utc = finalization_close_utc.tz_localize(UTC_TIMEZONE)
    else:
        finalization_close_utc = finalization_close_utc.tz_convert(UTC_TIMEZONE)
    available_utc = finalization_close_utc + lag
    result["BusinessDate"] = pd.DatetimeIndex(result.index).normalize()
    result["SourceSessionCloseET"] = close_utc.tz_convert(US_MARKET_TIMEZONE)
    result["SourceSessionCloseUTC"] = close_utc
    result["SourceSessionCloseKST"] = close_utc.tz_convert(KOREA_TIMEZONE)
    result["FinalizationSession"] = pd.DatetimeIndex(finalization_labels).normalize()
    result["FinalizationSessionCloseUTC"] = finalization_close_utc
    result["AvailableAtUTC"] = available_utc
    result["AvailableAtKST"] = available_utc.tz_convert(KOREA_TIMEZONE)
    result["TimestampSemantics"] = "completed_regular_session"
    result["FinalizationSessions"] = quarantine
    result["VendorLagMilliseconds"] = int(lag.total_seconds() * 1000)
    return result


def latest_available_us_daily_bar(
    frame: pd.DataFrame,
    decision_cutoff: object,
    *,
    vendor_lag: object = "15min",
    finalization_sessions: int = 1,
    calendar_name: str = US_EQUITY_CALENDAR,
) -> pd.DataFrame:
    cutoff_utc = _aware_timestamp(
        decision_cutoff, label="decision_cutoff"
    ).tz_convert(UTC_TIMEZONE)
    annotated = annotate_us_daily_availability(
        frame,
        vendor_lag=vendor_lag,
        finalization_sessions=finalization_sessions,
        calendar_name=calendar_name,
    )
    eligible = annotated.loc[annotated["AvailableAtUTC"] <= cutoff_utc]
    if eligible.empty:
        return eligible.copy()
    return eligible.iloc[[-1]].copy()


def us_daily_session_available_at(
    business_date: object,
    *,
    vendor_lag: object = "15min",
    finalization_sessions: int = 1,
    calendar_name: str = US_EQUITY_CALENDAR,
) -> pd.Timestamp:
    lag = _nonnegative_timedelta(vendor_lag, label="vendor_lag")
    quarantine = int(finalization_sessions)
    if quarantine < 0:
        raise ValueError("finalization_sessions must be non-negative")
    label, _opened, closed = _session_bounds_utc(
        business_date, calendar_name=calendar_name
    )
    calendar = _calendar(calendar_name)
    for _ in range(quarantine):
        label = pd.Timestamp(calendar.next_session(label))
        _label, _opened, closed = _session_bounds_utc(
            label, calendar_name=calendar_name
        )
    return closed + lag


def add_causal_phase_volume_baseline(
    frame: pd.DataFrame,
    *,
    lookback_observations: int = 20,
    minimum_history: int = 5,
    entity_columns: Sequence[str] = (),
) -> pd.DataFrame:
    """Add a shifted volume baseline without mixing market session phases."""

    lookback = int(lookback_observations)
    minimum = int(minimum_history)
    if lookback <= 0 or minimum <= 0 or minimum > lookback:
        raise ValueError("invalid causal volume baseline window")
    required = {
        "BusinessDate",
        "Volume",
        "SessionPhase",
        "PhaseBucket",
        "SourceTimestampUTC",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"minute availability frame missing columns: {missing}")
    entities = tuple(str(value) for value in entity_columns)
    missing_entities = [name for name in entities if name not in frame]
    if missing_entities:
        raise ValueError(f"volume baseline missing entity columns: {missing_entities}")

    result = frame.sort_values("SourceTimestampUTC", kind="stable").copy()
    group_columns = [*entities, "SessionPhase", "PhaseBucket"]
    uniqueness = [*group_columns, "BusinessDate"]
    if result.duplicated(uniqueness).any():
        raise ValueError(
            "phase volume baseline has multiple observations for one session slot"
        )

    grouped = result.groupby(group_columns, sort=False, dropna=False)["Volume"]

    def baseline(values: pd.Series) -> pd.Series:
        return values.shift(1).rolling(lookback, min_periods=minimum).median()

    def observations(values: pd.Series) -> pd.Series:
        return values.shift(1).rolling(lookback, min_periods=1).count()

    result["PhaseVolumeBaseline"] = grouped.transform(baseline)
    result["PhaseBaselineObservations"] = grouped.transform(observations).astype(int)
    valid = result["PhaseVolumeBaseline"] >= 0
    result["PhaseLogVolumeShock"] = np.nan
    result.loc[valid, "PhaseLogVolumeShock"] = (
        np.log1p(result.loc[valid, "Volume"].astype(float))
        - np.log1p(result.loc[valid, "PhaseVolumeBaseline"].astype(float))
    )
    return result.sort_index(kind="stable")
