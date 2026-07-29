from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExternalFactor:
    symbol: str
    name: str


DEFAULT_KR_GLOBAL_FACTORS: tuple[ExternalFactor, ...] = (
    ExternalFactor("KS11", "kospi"),
    ExternalFactor("KQ11", "kosdaq"),
    ExternalFactor("US500", "sp500"),
    ExternalFactor("IXIC", "nasdaq"),
    ExternalFactor("DJI", "dow"),
    ExternalFactor("VIX", "vix"),
    ExternalFactor("USD/KRW", "usdkrw"),
    ExternalFactor("JPY/KRW", "jpykrw"),
    ExternalFactor("US10YT", "us10y"),
    ExternalFactor("GC=F", "gold"),
    ExternalFactor("CL=F", "wti"),
)

BOK_BASE_RATE_URL = "https://www.bok.or.kr/portal/singl/baseRate/list.do?menuNo=200656"
POLICY_RATE_FACTORS: tuple[ExternalFactor, ...] = (
    ExternalFactor("BOK:BASE_RATE", "bok_base_rate"),
    ExternalFactor("FRED:DFEDTARU", "fed_target_upper"),
)
POLICY_RATE_FACTOR_NAMES = frozenset(factor.name for factor in POLICY_RATE_FACTORS)


def _safe_name(value: str) -> str:
    name = re.sub(r"[^0-9A-Za-z가-힣]+", "_", value.strip().lower())
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "factor"


def parse_external_factor(value: str) -> ExternalFactor:
    if ":" in value:
        symbol, name = value.split(":", 1)
        return ExternalFactor(symbol=symbol.strip(), name=_safe_name(name))
    symbol = value.strip()
    return ExternalFactor(symbol=symbol, name=_safe_name(symbol))


def resolve_external_factors(
    preset: str = "none",
    symbols: Sequence[str] | None = None,
) -> list[ExternalFactor]:
    factors: list[ExternalFactor] = []
    if preset == "kr_global":
        factors.extend(DEFAULT_KR_GLOBAL_FACTORS)
    elif preset == "kr_global_rates":
        factors.extend(DEFAULT_KR_GLOBAL_FACTORS)
        factors.extend(POLICY_RATE_FACTORS)
    elif preset not in ("none", "", None):
        raise ValueError(f"unknown external factor preset: {preset}")
    for item in symbols or []:
        if str(item).strip():
            factors.append(parse_external_factor(str(item)))

    deduped: list[ExternalFactor] = []
    seen: set[str] = set()
    for factor in factors:
        key = factor.name
        if key in seen:
            continue
        seen.add(key)
        deduped.append(factor)
    return deduped


def _cache_path(cache_dir: Path, factor: ExternalFactor, start: str, end: str) -> Path:
    safe_symbol = _safe_name(factor.symbol)
    safe_start = start.replace("-", "")
    safe_end = end.replace("-", "")
    return cache_dir / f"{safe_symbol}_{factor.name}_{safe_start}_{safe_end}.csv"


def _covering_cache_path(
    cache_dir: Path,
    factor: ExternalFactor,
    start: str,
    end: str,
) -> Path | None:
    """Return the freshest cached factor range that covers the request."""

    requested_start = pd.Timestamp(start)
    requested_end = pd.Timestamp(end)
    prefix = f"{_safe_name(factor.symbol)}_{factor.name}"
    pattern = re.compile(rf"^{re.escape(prefix)}_(\d{{8}})_(\d{{8}})\.csv$")
    candidates: list[tuple[int, int, str, Path]] = []
    for candidate in cache_dir.glob(f"{prefix}_*.csv"):
        match = pattern.fullmatch(candidate.name)
        if match is None:
            continue
        try:
            cached_start = pd.to_datetime(match.group(1), format="%Y%m%d", errors="raise")
            cached_end = pd.to_datetime(match.group(2), format="%Y%m%d", errors="raise")
        except ValueError:
            continue
        if cached_start <= requested_start and cached_end >= requested_end:
            start_gap_days = int((requested_start - cached_start).days)
            candidates.append((-int(cached_end.value), start_gap_days, candidate.name, candidate))
    return min(candidates)[3] if candidates else None


def _close_column(frame: pd.DataFrame) -> pd.Series:
    for column in ("Close", "close", "Adj Close", "AdjClose"):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
    numeric = frame.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("no numeric close-like column found")
    return pd.to_numeric(numeric.iloc[:, 0], errors="coerce")


def _fetch_bok_base_rate_frame(start: str, end: str) -> pd.DataFrame:
    """Load the official BOK change-date table as an effective daily step series."""

    tables = pd.read_html(BOK_BASE_RATE_URL)
    table = next((item for item in tables if item.shape[1] >= 3), None)
    if table is None or table.empty:
        raise ValueError("official BOK base-rate table was empty")

    years = pd.to_numeric(table.iloc[:, 0], errors="coerce")
    month_day = table.iloc[:, 1].astype(str).str.extract(r"(\d+)\D+(\d+)")
    rates = pd.to_numeric(table.iloc[:, 2], errors="coerce")
    valid = years.notna() & month_day.notna().all(axis=1) & rates.notna()
    if not valid.any():
        raise ValueError("official BOK base-rate table had no parseable rows")

    effective_dates = pd.to_datetime(
        {
            "year": years.loc[valid].astype(int),
            "month": month_day.loc[valid, 0].astype(int),
            "day": month_day.loc[valid, 1].astype(int),
        },
        errors="raise",
    )
    changes = pd.Series(rates.loc[valid].to_numpy(dtype=float), index=effective_dates)
    changes = changes[~changes.index.duplicated(keep="last")].sort_index()

    requested_start = pd.Timestamp(start).normalize()
    requested_end = pd.Timestamp(end).normalize()
    calendar = pd.date_range(requested_start, requested_end, freq="D")
    expanded = changes.reindex(changes.index.union(calendar)).sort_index().ffill().reindex(calendar)
    expanded = expanded.dropna()
    if expanded.empty:
        raise ValueError("BOK base-rate history does not cover the requested range")
    frame = expanded.rename("Close").to_frame()
    frame.index.name = "Date"
    return frame


def fetch_external_factor_closes(
    factors: Sequence[ExternalFactor],
    start: str,
    end: str | None,
    cache_dir: str | Path = "data/external_cache",
    refresh: bool = False,
) -> dict[str, pd.Series]:
    import FinanceDataReader as fdr

    cache_root = Path(cache_dir)
    cache_root.mkdir(parents=True, exist_ok=True)
    end_value = end or pd.Timestamp.today().strftime("%Y-%m-%d")
    closes: dict[str, pd.Series] = {}

    for factor in factors:
        path = _cache_path(cache_root, factor, start, end_value)
        cache_source = _covering_cache_path(cache_root, factor, start, end_value)
        try:
            if cache_source is not None and not refresh:
                frame = pd.read_csv(
                    cache_source,
                    parse_dates=["Date"],
                    index_col="Date",
                    float_precision="round_trip",
                )
            else:
                if factor.symbol == "BOK:BASE_RATE":
                    frame = _fetch_bok_base_rate_frame(start, end_value)
                else:
                    frame = fdr.DataReader(factor.symbol, start, end_value)
                if frame is None or frame.empty:
                    continue
                frame = frame.copy()
                frame.index.name = "Date"
                frame.to_csv(path)
            requested_start = pd.Timestamp(start)
            requested_end = pd.Timestamp(end_value)
            frame = frame.loc[(frame.index >= requested_start) & (frame.index <= requested_end)].copy()
            close = _close_column(frame).sort_index()
            close = close.replace([np.inf, -np.inf], np.nan).dropna()
            if len(close) >= 80:
                closes[factor.name] = close
        except Exception as exc:  # FinanceDataReader symbol support varies by source.
            print(f"external factor skipped: {factor.symbol} ({factor.name}): {exc}", flush=True)
    return closes


def _rolling_z(series: pd.Series, window: int) -> pd.Series:
    mean = series.rolling(window).mean()
    std = series.rolling(window).std()
    return (series - mean) / std.replace(0.0, np.nan)


def _days_since_change(series: pd.Series) -> pd.Series:
    changed = series.diff().fillna(0.0).ne(0.0)
    groups = changed.cumsum()
    return series.groupby(groups).cumcount().astype(float)


def _factor_feature_series(
    name: str,
    aligned: pd.Series,
) -> tuple[dict[str, pd.Series], pd.Series]:
    if name in POLICY_RATE_FACTOR_NAMES:
        change1 = aligned.diff(1)
        features = {
            f"ext_{name}_level": aligned,
            f"ext_{name}_change_1d": change1,
            f"ext_{name}_change_20d": aligned.diff(20),
            f"ext_{name}_change_252d": aligned.diff(252),
            f"ext_{name}_days_since_change": np.log1p(_days_since_change(aligned)),
            f"ext_{name}_level_z252": _rolling_z(aligned, 252),
        }
        return features, change1

    positive = aligned.where(aligned > 0.0)
    log_level = np.log(positive)
    ret1 = aligned.pct_change(1)
    features = {
        f"ext_{name}_ret1": ret1,
        f"ext_{name}_ret5": aligned.pct_change(5),
        f"ext_{name}_ret20": aligned.pct_change(20),
        f"ext_{name}_vol20": ret1.rolling(20).std(),
        f"ext_{name}_level_z60": _rolling_z(log_level, 60),
        f"ext_{name}_ma20_gap": aligned / aligned.rolling(20).mean() - 1.0,
    }
    return features, ret1


def _align_factor(
    name: str,
    close: pd.Series,
    dates: pd.DatetimeIndex,
    lag_days: int,
) -> pd.Series:
    aligned = close.reindex(dates).ffill().astype(float)
    # Policy decisions are known by the close on their official effective date.
    # Market closes retain the configured lag to avoid same-close lookahead.
    lag = 0 if name in POLICY_RATE_FACTOR_NAMES else max(0, int(lag_days))
    return aligned.shift(lag) if lag else aligned


def build_risk_free_period_returns(
    dates: pd.DatetimeIndex,
    annual_rate_percent: pd.Series,
    horizons: Sequence[int],
) -> dict[int, np.ndarray]:
    """Compound an effective annual policy rate over each close-to-close period."""

    trading_dates = pd.DatetimeIndex(dates).normalize()
    if len(trading_dates) == 0:
        return {int(horizon): np.empty((0,), dtype=np.float64) for horizon in horizons}
    if not trading_dates.is_monotonic_increasing or trading_dates.has_duplicates:
        raise ValueError("dates must be unique and increasing")

    rate = pd.to_numeric(annual_rate_percent, errors="coerce").sort_index()
    rate.index = pd.DatetimeIndex(rate.index).normalize()
    rate = rate[~rate.index.duplicated(keep="last")]
    calendar = pd.date_range(trading_dates[0], trading_dates[-1], freq="D")
    effective = rate.reindex(rate.index.union(calendar)).sort_index().ffill().reindex(calendar)
    if (effective.dropna() <= -100.0).any():
        raise ValueError("annual risk-free rate must be greater than -100 percent")
    daily_log_growth = np.log1p(effective / 100.0) / 365.0
    cumulative = daily_log_growth.fillna(0.0).cumsum()
    observed = effective.notna().astype(np.int64).cumsum()

    result: dict[int, np.ndarray] = {}
    for raw_horizon in sorted(set(int(item) for item in horizons)):
        if raw_horizon < 1:
            raise ValueError("risk-free horizons must be positive")
        values = np.full(len(trading_dates), np.nan, dtype=np.float64)
        for index in range(len(trading_dates) - raw_horizon):
            start_date = trading_dates[index]
            end_date = trading_dates[index + raw_horizon]
            start_log = float(cumulative.get(start_date, 0.0))
            end_log = float(cumulative.loc[end_date])
            start_observed = int(observed.get(start_date, 0))
            end_observed = int(observed.loc[end_date])
            if end_observed - start_observed == int((end_date - start_date).days):
                values[index] = float(np.expm1(end_log - start_log))
        result[raw_horizon] = values
    return result


def build_external_feature_frames(
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
    factor_closes: dict[str, pd.Series],
    lag_days: int = 1,
) -> dict[str, pd.DataFrame]:
    feature_frames: dict[str, pd.DataFrame] = {}
    columns = list(tickers)

    for name, close in factor_closes.items():
        aligned = _align_factor(name, close, dates, lag_days)
        if aligned.notna().sum() < 80:
            continue

        series_by_feature, _edge_return = _factor_feature_series(name, aligned)
        for feature_name, series in series_by_feature.items():
            values = series.to_numpy(dtype=np.float32)
            frame = pd.DataFrame(
                np.repeat(values[:, None], len(columns), axis=1),
                index=dates,
                columns=columns,
            )
            feature_frames[feature_name] = frame

    return feature_frames


def build_external_node_feature_frames(
    dates: pd.DatetimeIndex,
    factor_closes: dict[str, pd.Series],
    lag_days: int = 1,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    """Build feature frames for external factors as their own graph nodes.

    The returned feature frames use the same feature axis as stock nodes, but
    each factor node only observes the features derived from its own series.
    Missing modality cells are left as NaN so panel normalization ignores them
    and later converts the unavailable cells to neutral zeros.
    """

    node_ids = [f"EXT:{name}" for name in factor_closes]
    node_names = {f"EXT:{name}": name for name in factor_closes}
    feature_frames: dict[str, pd.DataFrame] = {}
    returns = pd.DataFrame(np.nan, index=dates, columns=node_ids, dtype=np.float32)

    def ensure_frame(feature_name: str) -> pd.DataFrame:
        frame = feature_frames.get(feature_name)
        if frame is None:
            frame = pd.DataFrame(np.nan, index=dates, columns=node_ids, dtype=np.float32)
            feature_frames[feature_name] = frame
        return frame

    for name, close in factor_closes.items():
        node_id = f"EXT:{name}"
        aligned = _align_factor(name, close, dates, lag_days)
        if aligned.notna().sum() < 80:
            continue

        series_by_feature, edge_return = _factor_feature_series(name, aligned)
        returns[node_id] = edge_return.to_numpy(dtype=np.float32)
        for feature_name, series in series_by_feature.items():
            ensure_frame(feature_name)[node_id] = series.to_numpy(dtype=np.float32)

    active_nodes = [node_id for node_id in node_ids if returns[node_id].notna().sum() >= 80]
    if len(active_nodes) != len(node_ids):
        returns = returns.reindex(columns=active_nodes)
        node_names = {node_id: node_names[node_id] for node_id in active_nodes}
        feature_frames = {
            name: frame.reindex(columns=active_nodes)
            for name, frame in feature_frames.items()
        }
    return feature_frames, returns, node_names
