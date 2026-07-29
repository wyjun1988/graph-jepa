from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from stock_v2.corporate_actions import build_causal_ohlcv


PRICE_COLUMNS = ("Open", "High", "Low", "Close")
RAW_COLUMNS = ("RawOpen", "RawHigh", "RawLow", "RawClose", "RawVolume")


def read_ohlcv_csv(path: str | Path) -> pd.DataFrame:
    """Read an immutable OHLCV CSV without changing serialized float values."""

    return pd.read_csv(
        path,
        parse_dates=["Date"],
        index_col="Date",
        float_precision="round_trip",
    )


def _normalized(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="coerce")).normalize()
    if result.index.isna().any():
        raise ValueError("OHLCV contains an invalid date")
    result = result[~result.index.duplicated(keep="last")].sort_index()
    return result


def _validate_bridge(base: pd.DataFrame, raw: pd.DataFrame, bridge: pd.Timestamp) -> None:
    required = set(RAW_COLUMNS).difference(base.columns)
    if required:
        raise ValueError(f"base causal OHLCV is missing columns: {sorted(required)}")
    if bridge not in raw.index:
        raise ValueError(f"incremental raw OHLCV does not contain bridge date {bridge.date()}")
    base_values = base.loc[
        bridge,
        ["RawOpen", "RawHigh", "RawLow", "RawClose", "RawVolume"],
    ].to_numpy(dtype=np.float64)
    raw_values = raw.loc[
        bridge,
        ["Open", "High", "Low", "Close", "Volume"],
    ].to_numpy(dtype=np.float64)
    if not np.isfinite(base_values).all() or not np.isfinite(raw_values).all():
        raise ValueError("bridge row contains non-finite raw values")
    if not np.allclose(base_values, raw_values, rtol=1e-9, atol=1e-6):
        raise ValueError("incremental raw OHLCV bridge does not match the immutable base")


def extend_causal_history(
    base_frame: pd.DataFrame,
    raw_increment: pd.DataFrame,
    adjusted_increment: pd.DataFrame,
    *,
    ticker: str,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Append adjacent adjusted returns while preserving the base causal scale."""

    base = _normalized(base_frame)
    raw = _normalized(raw_increment)
    adjusted = _normalized(adjusted_increment)
    if base.empty:
        raise ValueError("base causal OHLCV must not be empty")
    if raw.empty or adjusted.empty:
        raise ValueError("incremental raw and adjusted OHLCV must not be empty")
    if not raw.index.equals(adjusted.index):
        raise ValueError("incremental raw and adjusted indexes must match")

    bridge = pd.Timestamp(base.index[-1]).normalize()
    _validate_bridge(base, raw, bridge)
    recent, events = build_causal_ohlcv(raw, adjusted, ticker=ticker)
    bridge_scale = float(base.loc[bridge, "CausalPriceScale"])
    recent_bridge_scale = float(recent.loc[bridge, "CausalPriceScale"])
    if (
        not np.isfinite(bridge_scale)
        or not np.isfinite(recent_bridge_scale)
        or bridge_scale <= 0.0
        or recent_bridge_scale <= 0.0
    ):
        raise ValueError("causal bridge scale must be finite and positive")
    scale_ratio = bridge_scale / recent_bridge_scale
    for column in PRICE_COLUMNS:
        recent[column] = recent[column] * scale_ratio
    recent["Volume"] = recent["Volume"] / scale_ratio
    recent["CausalPriceScale"] = recent["CausalPriceScale"] * scale_ratio

    appended = recent.loc[recent.index > bridge].copy()
    if appended.empty:
        return base.copy(), []
    missing_columns = set(base.columns).difference(appended.columns)
    if missing_columns:
        raise ValueError(
            f"incremental causal OHLCV is missing base columns: {sorted(missing_columns)}"
        )
    appended = appended.reindex(columns=base.columns)
    merged = pd.concat([base, appended], axis=0)
    if merged.index.has_duplicates or not merged.index.is_monotonic_increasing:
        raise RuntimeError("extended causal OHLCV has an invalid date index")

    raw_notional = (
        merged["RawClose"].to_numpy(dtype=np.float64)
        * merged["RawVolume"].to_numpy(dtype=np.float64)
    )
    canonical_notional = (
        merged["Close"].to_numpy(dtype=np.float64)
        * merged["Volume"].to_numpy(dtype=np.float64)
    )
    if not np.allclose(raw_notional, canonical_notional, rtol=1e-9, atol=1e-5):
        raise RuntimeError("extended causal OHLCV violates the notional invariant")
    bridge_date = str(bridge.date())
    return merged, [
        event for event in events if str(event.get("effective_date")) > bridge_date
    ]
