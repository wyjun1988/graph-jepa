from __future__ import annotations

import numpy as np
import pandas as pd


def build_us_etf_daily_consensus(
    kiwoom: pd.DataFrame,
    yahoo: pd.DataFrame,
    *,
    close_relative_tolerance: float = 0.001,
    volume_relative_tolerance: float = 0.05,
    volume_lookback_sessions: int = 20,
    volume_minimum_history: int = 10,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Build a masked daily feature panel from two independent source views."""

    if close_relative_tolerance < 0 or volume_relative_tolerance < 0:
        raise ValueError("cross-source tolerances must be non-negative")
    lookback = int(volume_lookback_sessions)
    minimum = int(volume_minimum_history)
    if lookback <= 0 or minimum <= 0 or minimum > lookback:
        raise ValueError("invalid volume baseline history")
    required_kiwoom = {
        "Close",
        "Volume",
        "OHLCEnvelopeRepaired",
        "AvailableAtUTC",
    }
    required_yahoo = {"Close", "AdjustedClose", "Volume"}
    missing_kiwoom = sorted(required_kiwoom.difference(kiwoom.columns))
    missing_yahoo = sorted(required_yahoo.difference(yahoo.columns))
    if missing_kiwoom or missing_yahoo:
        raise ValueError(
            "consensus source columns missing: "
            f"kiwoom={missing_kiwoom}, yahoo={missing_yahoo}"
        )
    if not kiwoom.index.equals(yahoo.index):
        raise ValueError("consensus sources must have exactly matching US sessions")
    if not kiwoom.index.is_monotonic_increasing or kiwoom.index.has_duplicates:
        raise ValueError("consensus source dates must be sorted and unique")

    result = pd.DataFrame(index=kiwoom.index.copy())
    result.index.name = "Date"
    result["KiwoomClose"] = kiwoom["Close"].astype(float)
    result["YahooClose"] = yahoo["Close"].astype(float)
    result["YahooAdjustedClose"] = yahoo["AdjustedClose"].astype(float)
    result["KiwoomVolume"] = kiwoom["Volume"].astype(float)
    result["YahooVolume"] = yahoo["Volume"].astype(float)
    result["AvailableAtUTC"] = pd.to_datetime(
        kiwoom["AvailableAtUTC"], utc=True
    )
    result["OHLCEnvelopeValid"] = ~kiwoom[
        "OHLCEnvelopeRepaired"
    ].astype(bool)

    result["CloseRelativeDifference"] = (
        (result["KiwoomClose"] - result["YahooClose"]).abs()
        / result["YahooClose"].abs().clip(lower=np.finfo(float).eps)
    )
    result["CloseConsensusValid"] = (
        result["CloseRelativeDifference"] <= float(close_relative_tolerance)
    )
    result["VolumeRelativeDifference"] = (
        (result["KiwoomVolume"] - result["YahooVolume"]).abs()
        / result["YahooVolume"].abs().clip(lower=1.0)
    )
    result["VolumeConsensusValid"] = (
        result["VolumeRelativeDifference"] <= float(volume_relative_tolerance)
    )

    unmasked_return = np.log(result["YahooAdjustedClose"]).diff()
    return_valid = (
        result["CloseConsensusValid"]
        & result["CloseConsensusValid"].shift(1, fill_value=False)
        & unmasked_return.notna()
    )
    result["TotalReturnValid"] = return_valid
    result["TotalLogReturn"] = unmasked_return.where(return_valid)

    valid_log_volume = np.log1p(result["YahooVolume"]).where(
        result["VolumeConsensusValid"]
    )
    shifted = valid_log_volume.shift(1)
    result["LogVolumeBaseline"] = shifted.rolling(
        lookback, min_periods=minimum
    ).median()
    result["VolumeBaselineObservations"] = shifted.rolling(
        lookback, min_periods=1
    ).count().astype(int)
    volume_valid = (
        result["VolumeConsensusValid"]
        & result["LogVolumeBaseline"].notna()
    )
    result["VolumeFeatureValid"] = volume_valid
    result["LogVolumeShock"] = (
        valid_log_volume - result["LogVolumeBaseline"]
    ).where(volume_valid)

    summary = {
        "rows": int(len(result)),
        "close_consensus_invalid_rows": int(
            (~result["CloseConsensusValid"]).sum()
        ),
        "volume_consensus_invalid_rows": int(
            (~result["VolumeConsensusValid"]).sum()
        ),
        "ohlc_envelope_invalid_rows": int(
            (~result["OHLCEnvelopeValid"]).sum()
        ),
        "total_return_valid_rows": int(result["TotalReturnValid"].sum()),
        "volume_feature_valid_rows": int(result["VolumeFeatureValid"].sum()),
        "maximum_close_relative_difference": float(
            result["CloseRelativeDifference"].max()
        ),
        "maximum_volume_relative_difference": float(
            result["VolumeRelativeDifference"].max()
        ),
    }
    return result, summary
