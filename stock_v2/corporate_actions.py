from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


PRICE_COLUMNS = ("Open", "High", "Low", "Close")
REQUIRED_COLUMNS = (*PRICE_COLUMNS, "Volume")


def _numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(REQUIRED_COLUMNS).difference(frame.columns)
    if missing:
        raise ValueError(f"OHLCV frame is missing columns: {sorted(missing)}")
    result = frame.copy()
    result.index = pd.DatetimeIndex(pd.to_datetime(result.index, errors="coerce")).normalize()
    if result.index.isna().any() or result.index.duplicated().any() or not result.index.is_monotonic_increasing:
        raise ValueError("OHLCV index must be a sorted, unique DatetimeIndex")
    for column in result.columns:
        if column in REQUIRED_COLUMNS or column == "TradingValueM":
            result[column] = pd.to_numeric(result[column], errors="coerce")
    return result


def vendor_adjustment_factor(raw: pd.DataFrame, adjusted: pd.DataFrame) -> pd.Series:
    """Estimate the vendor price factor without letting volume outvote OHLC."""

    if not raw.index.equals(adjusted.index):
        raise ValueError("raw and adjusted OHLCV indexes must match exactly")
    candidates: list[pd.Series] = []
    for column in PRICE_COLUMNS:
        denominator = raw[column].where(raw[column] > 0.0)
        candidates.append((adjusted[column] / denominator).where(adjusted[column] > 0.0))
    factor = pd.concat(candidates, axis=1).median(axis=1, skipna=True)
    factor = factor.where(np.isfinite(factor) & (factor > 0.0)).ffill().bfill()
    if factor.isna().any():
        raise ValueError("could not derive a finite adjustment factor")
    return factor.astype(float)


def vendor_volume_adjustment_factor(raw: pd.DataFrame, adjusted: pd.DataFrame) -> pd.Series:
    denominator = adjusted["Volume"].where(adjusted["Volume"] > 0.0)
    factor = (raw["Volume"] / denominator).where(raw["Volume"] > 0.0)
    return factor.where(np.isfinite(factor) & (factor > 0.0)).astype(float)


def _causal_return_index(raw_close: pd.Series, adjusted_close: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Reconstruct a forward index from adjacent adjusted returns.

    A later corporate action can multiply an earlier adjusted-price segment by a
    constant, but that constant cancels in adjacent returns. Rebuilding the level
    from those returns avoids importing the vendor's release-end price scale.
    """

    if (raw_close <= 0.0).any() or (adjusted_close <= 0.0).any():
        raise ValueError("OHLCV close prices must be positive")
    adjusted_return = adjusted_close.pct_change(fill_method=None).fillna(0.0)
    if (~np.isfinite(adjusted_return)).any() or (adjusted_return <= -1.0).any():
        raise ValueError("vendor-adjusted close produced an invalid daily return")
    index = float(raw_close.iloc[0]) * (1.0 + adjusted_return).cumprod()
    return index.astype(float), adjusted_return.astype(float)


def build_causal_ohlcv(
    raw_frame: pd.DataFrame,
    adjusted_frame: pd.DataFrame,
    *,
    ticker: str = "",
    minimum_jump_ratio: float = 0.02,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Build a causal adjusted-return index while preserving executable raw bars.

    The canonical OHLC columns are a return index, not historical executable
    prices. RawOHLC/RawVolume remain the source of execution and liquidity data.
    """

    raw = _numeric_frame(raw_frame)
    adjusted = _numeric_frame(adjusted_frame)
    if not raw.index.equals(adjusted.index):
        raise ValueError("raw and adjusted OHLCV indexes must match exactly")
    if raw.empty:
        return raw.copy(), []
    if minimum_jump_ratio <= 0.0:
        raise ValueError("minimum_jump_ratio must be positive")
    factor = vendor_adjustment_factor(raw, adjusted)
    volume_factor = vendor_volume_adjustment_factor(raw, adjusted)
    causal_index, adjusted_return = _causal_return_index(raw["Close"], adjusted["Close"])
    raw_return = raw["Close"].pct_change(fill_method=None).fillna(0.0).astype(float)
    scale = causal_index / raw["Close"]
    factor_ratio = factor / factor.shift(1)
    action_mask = (factor_ratio - 1.0).abs() >= float(minimum_jump_ratio)
    action_mask.iloc[0] = False
    events: list[dict[str, Any]] = []
    for boundary in np.flatnonzero(action_mask.to_numpy()):
        previous_factor = float(factor.iloc[boundary - 1])
        current_factor = float(factor.iloc[boundary])
        current_volume_factor = float(volume_factor.iloc[boundary]) if pd.notna(volume_factor.iloc[boundary]) else None
        factor_corroborated = (
            current_volume_factor is not None
            and abs(current_volume_factor / current_factor - 1.0) <= 0.02
        )
        events.append(
            {
                "schema_version": 2,
                "ticker": str(ticker).replace("A", "").zfill(6) if ticker else "",
                "effective_date": str(raw.index[boundary].date()),
                "vendor_factor_before": previous_factor,
                "vendor_factor_after": current_factor,
                "action_ratio": previous_factor / current_factor,
                "vendor_volume_factor_after": current_volume_factor,
                "price_volume_factor_corroborated": factor_corroborated,
                "raw_return": float(raw_return.iloc[boundary]),
                "vendor_adjusted_return": float(adjusted_return.iloc[boundary]),
                "causal_price_scale_after": float(scale.iloc[boundary]),
                "derivation": "kiwoom_ka10081_adjacent_adjusted_return_v2",
            }
        )

    canonical = pd.DataFrame(index=raw.index)
    for column in PRICE_COLUMNS:
        canonical[column] = raw[column] * scale
        canonical[f"Raw{column}"] = raw[column]
        canonical[f"VendorAdjusted{column}"] = adjusted[column]
    canonical["Volume"] = raw["Volume"] / scale
    canonical["RawVolume"] = raw["Volume"]
    canonical["VendorAdjustedVolume"] = adjusted["Volume"]
    if "TradingValueM" in raw:
        canonical["TradingValueM"] = raw["TradingValueM"]
    canonical["CausalPriceScale"] = scale
    canonical["VendorAdjustmentFactor"] = factor
    canonical["VendorVolumeAdjustmentFactor"] = volume_factor
    canonical["RawReturn"] = raw_return
    canonical["CausalAdjustedReturn"] = adjusted_return
    canonical["AdjustmentReturnGap"] = adjusted_return - raw_return
    canonical["CorporateActionFlag"] = action_mask.astype(bool)

    valid_turnover = (raw["Close"] > 0.0) & (raw["Volume"] >= 0.0)
    raw_turnover = raw.loc[valid_turnover, "Close"] * raw.loc[valid_turnover, "Volume"]
    canonical_turnover = (
        canonical.loc[valid_turnover, "Close"] * canonical.loc[valid_turnover, "Volume"]
    )
    if not np.allclose(raw_turnover, canonical_turnover, rtol=1e-10, atol=1e-6):
        raise RuntimeError("causal adjustment failed to preserve traded notional")
    canonical.index.name = "Date"
    return canonical, events
