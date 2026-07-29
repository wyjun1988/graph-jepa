from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List, Mapping

import numpy as np
import pandas as pd
import torch

from stock_v2.data_contract import validate_checkpoint_panel
from stock_v2.event_features import (
    build_event_feature_frames,
    build_event_theme_exposure,
    build_event_ticker_coverage,
)
from stock_v2.fundamental_features import build_fundamental_feature_frames, load_fundamental_observations
from stock_v2.external_factors import (
    build_external_feature_frames,
    build_external_node_feature_frames,
    fetch_external_factor_closes,
    resolve_external_factors,
)
from stock_v2.kiwoom_investor import build_investor_feature_frames, load_investor_flow_frames
from stock_v2.graph_jepa import GraphBatch, StockGraphJEPA
from stock_v2.latent_path_head import (
    LoadedLatentPathHead,
    blend_latent_path_scores,
    load_latent_path_head,
)
from stock_v2.market_data import fetch_krx_ohlcv, make_ohlcv_panel
from stock_v2.ops.types import Quote, Signal
from stock_v2.real_features import FeaturePanel, build_feature_panel, make_real_snapshot
from stock_v2.static_edges import build_industry_edge_arrays, load_industry_codes


WORLD_MODEL_SIGNAL = "world_model"


def rank_tradeable_scores(
    scores: np.ndarray,
    prices: np.ndarray,
    return_1d_available: np.ndarray,
    top_n: int,
    min_price: float = 0.0,
    max_price: float = float("inf"),
) -> np.ndarray:
    """Rank only symbols with a current executable price and observed market state."""

    scores = np.asarray(scores, dtype=np.float64)
    prices = np.asarray(prices, dtype=np.float64)
    return_1d_available = np.asarray(return_1d_available, dtype=bool)
    if scores.ndim != 1 or prices.shape != scores.shape or return_1d_available.shape != scores.shape:
        raise ValueError("scores, prices, and return_1d_available must be aligned one-dimensional arrays")
    eligible = (
        np.isfinite(scores)
        & np.isfinite(prices)
        & (prices > 0.0)
        & (prices >= float(min_price))
        & (prices <= float(max_price))
        & return_1d_available
    )
    indices = np.flatnonzero(eligible)
    if indices.size == 0 or top_n <= 0:
        return np.empty((0,), dtype=np.int64)
    ranked = indices[np.argsort(scores[indices], kind="stable")[::-1]]
    return ranked[: int(top_n)]


def select_latest_complete_step(
    features: FeaturePanel,
    min_coverage_ratio: float = 0.90,
    lookback: int = 20,
) -> dict[str, int | float | str]:
    """Select the newest daily state that is not a partial provider update."""

    if not 0.0 < min_coverage_ratio <= 1.0:
        raise ValueError("min_coverage_ratio must be in (0, 1]")
    if lookback < 1 or len(features.dates) == 0:
        raise ValueError("lookback and feature dates must be non-empty")
    stock_count = features.tradable_count
    return_index = features.feature_index("return_1d")
    coverage = (
        (features.available_mask[:, :stock_count, return_index] > 0.5)
        & np.isfinite(features.close[:, :stock_count])
        & (features.close[:, :stock_count] > 0.0)
    ).sum(axis=1)
    left = max(0, len(coverage) - int(lookback))
    reference = int(coverage[left:].max())
    threshold = max(1, int(np.ceil(reference * float(min_coverage_ratio))))
    candidates = np.flatnonzero(coverage >= threshold)
    if candidates.size == 0:
        raise RuntimeError("no market state satisfies the minimum daily coverage")
    step = int(candidates[-1])
    return {
        "step": step,
        "date": str(features.dates[step].date()),
        "observed_stocks": int(coverage[step]),
        "reference_stocks": reference,
        "threshold_stocks": threshold,
        "coverage_ratio": float(coverage[step] / max(reference, 1)),
        "panel_latest_date": str(features.dates[-1].date()),
        "panel_latest_observed_stocks": int(coverage[-1]),
    }


def materialize_quote_overlay_step(features: FeaturePanel, base_step: int) -> int:
    """Copy a complete base snapshot into the newest partial date for quote overlay."""

    target_step = len(features.dates) - 1
    if base_step < 0 or base_step > target_step:
        raise IndexError("base_step is outside the feature panel")
    if base_step == target_step:
        return target_step
    features.features[target_step] = features.features[base_step]
    features.raw_features[target_step] = features.raw_features[base_step]
    features.available_mask[target_step] = features.available_mask[base_step]
    features.returns_1d[target_step] = features.returns_1d[base_step]
    features.open[target_step] = features.open[base_step]
    features.close[target_step] = features.close[base_step]
    if features.event_theme_exposure is not None:
        features.event_theme_exposure[target_step] = features.event_theme_exposure[base_step]
    if features.execution_close is not None:
        features.execution_close[target_step] = features.execution_close[base_step]
    return target_step


def missing_business_days_between(
    base_date: str | pd.Timestamp,
    session_date: str | pd.Timestamp,
) -> int:
    """Count weekdays strictly between a complete close and an intraday session."""

    base = pd.Timestamp(base_date).normalize()
    session = pd.Timestamp(session_date).normalize()
    if pd.isna(base) or pd.isna(session):
        raise ValueError("base and session dates must be valid")
    if session < base:
        raise ValueError("intraday session precedes the complete daily state")
    start = np.datetime64(base.date(), "D") + np.timedelta64(1, "D")
    end = np.datetime64(session.date(), "D")
    return int(
        np.busday_count(
            start,
            end,
        )
    )


def _append_base_row(array: np.ndarray, base_step: int) -> np.ndarray:
    return np.concatenate([array, array[base_step : base_step + 1].copy()], axis=0)


def materialize_quote_overlay_session(
    features: FeaturePanel,
    base_step: int,
    session_date: str | pd.Timestamp,
) -> int:
    """Create or reset a synthetic current-session row above a complete daily state."""

    if base_step < 0 or base_step >= len(features.dates):
        raise IndexError("base_step is outside the feature panel")
    session = pd.Timestamp(session_date).normalize()
    base_date = pd.Timestamp(features.dates[base_step]).normalize()
    latest_date = pd.Timestamp(features.dates[-1]).normalize()
    if pd.isna(session):
        raise ValueError("intraday session date must be valid")
    if session < base_date:
        raise ValueError("intraday session precedes the complete daily state")
    if session < latest_date:
        raise ValueError("intraday session precedes the latest panel row")
    if session == latest_date and base_step < len(features.dates) - 1:
        return materialize_quote_overlay_step(features, base_step)

    features.features = _append_base_row(features.features, base_step)
    features.raw_features = _append_base_row(features.raw_features, base_step)
    features.available_mask = _append_base_row(features.available_mask, base_step)
    features.returns_1d = _append_base_row(features.returns_1d, base_step)
    features.open = _append_base_row(features.open, base_step)
    features.close = _append_base_row(features.close, base_step)
    if features.execution_close is not None:
        features.execution_close = _append_base_row(
            features.execution_close,
            base_step,
        )
    if features.event_theme_exposure is not None:
        features.event_theme_exposure = _append_base_row(
            features.event_theme_exposure,
            base_step,
        )
    target_shape = features.target_returns[base_step : base_step + 1].shape
    features.target_returns = np.concatenate(
        [
            features.target_returns,
            np.full(target_shape, np.nan, dtype=features.target_returns.dtype),
        ],
        axis=0,
    )
    features.target_return_paths = {
        horizon: np.concatenate(
            [
                values,
                np.full(
                    values[base_step : base_step + 1].shape,
                    np.nan,
                    dtype=values.dtype,
                ),
            ],
            axis=0,
        )
        for horizon, values in features.target_return_paths.items()
    }
    features.dates = features.dates.append(pd.DatetimeIndex([session]))
    return len(features.dates) - 1


def configured_rollout_offsets(value: object) -> list[int]:
    """Normalize checkpoint rollout offsets without importing the training CLI."""

    if isinstance(value, str):
        values = [part.strip() for part in value.split(",") if part.strip()]
    elif isinstance(value, (list, tuple, set)):
        values = list(value)
    elif value is None:
        values = []
    else:
        values = [value]
    offsets = sorted({int(item) for item in values})
    if not offsets or any(offset < 1 for offset in offsets):
        raise ValueError("world-model signals require one or more positive rollout offsets")
    return offsets


def merge_event_paths(*path_groups: object) -> list[str]:
    """Preserve event-source order while appending live sources exactly once."""

    paths: list[str] = []
    for group in path_groups:
        if group is None:
            continue
        values = [group] if isinstance(group, (str, Path)) else group
        for value in values:
            path = str(value).strip()
            if path and path not in paths:
                paths.append(path)
    return paths


def rollout_steps_for_offset(checkpoint_args: Dict, offset: int) -> int:
    """Match the temporal-rollout step mapping used during training."""

    if checkpoint_args.get("pretrain_task") != "temporal":
        raise ValueError("world-model signals require a temporally trained checkpoint")
    temporal_offset = max(1, int(checkpoint_args.get("temporal_offset", offset)))
    latent_rollout_steps = max(1, int(checkpoint_args.get("latent_rollout_steps", 1)))
    return max(1, int(round(int(offset) * latent_rollout_steps / temporal_offset)))


def _weighted_row_mean(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    weighted = np.where(finite, values * weights[None, :], 0.0)
    denominator = np.where(finite, weights[None, :], 0.0).sum(axis=1)
    return np.divide(
        weighted.sum(axis=1),
        denominator,
        out=np.full(values.shape[0], np.nan, dtype=np.float64),
        where=denominator > 0.0,
    )


def _derive_entry_path_return(
    horizon_close_return: np.ndarray,
    next_open_gap: np.ndarray,
) -> np.ndarray:
    denominator = 1.0 + np.asarray(next_open_gap, dtype=np.float64)
    close_return = np.asarray(horizon_close_return, dtype=np.float64)
    valid = np.isfinite(close_return) & np.isfinite(denominator) & (denominator > 1e-6)
    result = np.full(close_return.shape, np.nan, dtype=np.float64)
    result[valid] = (1.0 + close_return[valid]) / denominator[valid] - 1.0
    return result


def world_model_state_scores(
    forecast_by_horizon: Dict[int, np.ndarray],
    feature_names: List[str],
    train_mean: np.ndarray,
    train_std: np.ndarray,
) -> tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Score a multi-horizon state trajectory without a return-regression head.

    The score is deliberately a transparent state score, rather than an
    estimated PnL. For each supervised horizon ``h``, it reads ``return_hd``
    at target state ``t+h``. That is the cumulative close-to-close return from
    ``t`` to ``t+h``; an executable next-open entry path additionally contains
    the unknown overnight gap. Returns are dailyized before aggregation and
    forecast volatility is subtracted. The score is intended for shadow
    ranking and must be validated against realized entry paths before order use.
    """

    if not forecast_by_horizon:
        raise ValueError("world-model signals require at least one forecast horizon")
    mean = np.asarray(train_mean, dtype=np.float64)
    std = np.asarray(train_std, dtype=np.float64)
    if mean.shape != (len(feature_names),) or std.shape != (len(feature_names),):
        raise ValueError("training normalization statistics do not match feature names")

    horizons = sorted(int(horizon) for horizon in forecast_by_horizon)
    normalized = [np.asarray(forecast_by_horizon[horizon], dtype=np.float64) for horizon in horizons]
    node_count = normalized[0].shape[0]
    if any(frame.shape != (node_count, len(feature_names)) for frame in normalized):
        raise ValueError("world-model forecast states must share node and feature shapes")
    raw = np.stack(normalized, axis=1) * std[None, None, :] + mean[None, None, :]
    horizon_weights = np.asarray([min(float(horizon), 5.0) for horizon in horizons], dtype=np.float64)
    horizon_weights /= horizon_weights.sum()

    if (
        1 not in horizons
        or "gap_open" not in feature_names
        or "intraday_return" not in feature_names
    ):
        raise ValueError(
            "world-model entry-path signals require horizon 1, gap_open, and intraday_return"
        )
    next_open_gap = raw[:, horizons.index(1), feature_names.index("gap_open")]
    horizon_close_returns = []
    predicted_entry_paths = []
    dailyized_paths = []
    for position, horizon in enumerate(horizons):
        feature_name = f"return_{horizon}d"
        if feature_name not in feature_names:
            raise ValueError(
                f"world-model signals require matched path feature {feature_name}"
            )
        path_return = raw[:, position, feature_names.index(feature_name)]
        horizon_close_returns.append(path_return)
        entry_path = (
            raw[:, position, feature_names.index("intraday_return")]
            if horizon == 1
            else _derive_entry_path_return(path_return, next_open_gap)
        )
        predicted_entry_paths.append(entry_path)
        clipped = np.maximum(entry_path, -0.999999)
        dailyized_paths.append(np.expm1(np.log1p(clipped) / float(horizon)))
    horizon_close_return_matrix = np.stack(horizon_close_returns, axis=1)
    predicted_entry_path_matrix = np.stack(predicted_entry_paths, axis=1)
    dailyized_path_matrix = np.stack(dailyized_paths, axis=1)
    expected_dailyized_entry_path_return = _weighted_row_mean(
        dailyized_path_matrix,
        horizon_weights,
    )
    score = expected_dailyized_entry_path_return.copy()
    diagnostics: Dict[str, np.ndarray] = {
        "expected_dailyized_entry_path_return": expected_dailyized_entry_path_return,
        "horizon_close_returns": horizon_close_return_matrix,
        "predicted_entry_path_returns": predicted_entry_path_matrix,
        "predicted_next_open_gap": next_open_gap,
        # Compatibility alias for existing read-only metadata consumers.
        "expected_dailyized_path_return": expected_dailyized_entry_path_return,
        "expected_return_1d": expected_dailyized_entry_path_return,
        "matched_path_returns": predicted_entry_path_matrix,
        "risk_penalty": np.zeros(node_count, dtype=np.float64),
    }
    if "cs_rank_return_20d" in feature_names:
        relative_rank = _weighted_row_mean(
            raw[:, :, feature_names.index("cs_rank_return_20d")],
            horizon_weights,
        )
        score += 0.01 * relative_rank
        diagnostics["expected_cs_rank_return_20d"] = relative_rank

    risk_terms = []
    for feature_name in ("volatility_20d", "downside_volatility_20d"):
        if feature_name in feature_names:
            forecast_risk = _weighted_row_mean(
                raw[:, :, feature_names.index(feature_name)],
                horizon_weights,
            )
            risk_terms.append(np.maximum(forecast_risk, 0.0))
            diagnostics[f"expected_{feature_name}"] = forecast_risk
    if risk_terms:
        risk_penalty = 0.35 * np.mean(np.stack(risk_terms, axis=1), axis=1)
        score -= risk_penalty
        diagnostics["risk_penalty"] = risk_penalty

    diagnostics["score_before_risk"] = score + diagnostics["risk_penalty"]
    diagnostics["horizons"] = np.asarray(horizons, dtype=np.int64)
    return score.astype(np.float32), diagnostics


def _set_latest_feature(
    features: FeaturePanel,
    step: int,
    node_index: int,
    feature_index: Mapping[str, int],
    feature_name: str,
    value: float,
) -> bool:
    index = feature_index.get(feature_name)
    if index is None or not np.isfinite(value):
        return False
    std = float(features.train_std[index])
    if not np.isfinite(std) or abs(std) < 1e-6:
        std = 1.0
    features.raw_features[step, node_index, index] = np.float32(value)
    features.features[step, node_index, index] = np.float32(
        (value - float(features.train_mean[index])) / std
    )
    features.available_mask[step, node_index, index] = 1.0
    return True


def apply_intraday_quote_overlay(
    features: FeaturePanel,
    quotes: Mapping[str, Quote] | None,
    step: int | None = None,
) -> dict[str, object]:
    """Inject a bounded quote overlay into the latest graph state.

    Only received quotes are overlaid; all other nodes retain their last
    verified daily observation. This makes partial intraday sensing explicit
    while letting the dynamic edge builder consume the refreshed return state.
    """

    if not quotes or len(features.dates) < 2:
        return {"applied_tickers": [], "updated_feature_cells": 0, "daily_prices": {}}
    step = len(features.dates) - 1 if step is None else int(step)
    if step < 1 or step >= len(features.dates):
        raise IndexError("quote overlay step must have a previous market state")
    stock_count = features.tradable_count
    ticker_index = {ticker: index for index, ticker in enumerate(features.tickers)}
    feature_index = {name: index for index, name in enumerate(features.feature_names)}
    updated_cells = 0
    applied: list[str] = []
    received_at: dict[str, str] = {}
    daily_prices: dict[str, float] = {}

    for raw_ticker, quote in quotes.items():
        ticker = str(raw_ticker).replace("A", "").strip()
        node_index = ticker_index.get(ticker)
        price = quote.usable_price
        if node_index is None or node_index >= stock_count or price is None or price <= 0.0:
            continue
        previous_close = float(features.close[step - 1, node_index])
        if not np.isfinite(previous_close) or previous_close <= 0.0:
            continue

        model_daily_price = float(features.close[step, node_index])
        execution_daily_price = (
            float(features.execution_close[step, node_index])
            if features.execution_close is not None
            else model_daily_price
        )
        if not np.isfinite(execution_daily_price) or execution_daily_price <= 0.0:
            continue
        model_price_scale = model_daily_price / execution_daily_price
        model_price = float(price) * model_price_scale
        daily_prices[ticker] = execution_daily_price
        features.close[step, node_index] = np.float32(model_price)
        if features.execution_close is not None:
            features.execution_close[step, node_index] = np.float32(price)
        if quote.open_price is not None and quote.open_price > 0.0:
            features.open[step, node_index] = np.float32(
                quote.open_price * model_price_scale
            )

        latest_return = float(model_price / previous_close - 1.0)
        features.returns_1d[step, node_index] = np.float32(latest_return)
        for horizon in (1, 2, 3, 5, 10, 20, 60, 120):
            reference_step = step - horizon
            if reference_step < 0:
                continue
            reference_close = float(features.close[reference_step, node_index])
            if reference_close > 0.0 and np.isfinite(reference_close):
                updated_cells += _set_latest_feature(
                    features,
                    step,
                    node_index,
                    feature_index,
                    f"return_{horizon}d",
                    float(model_price / reference_close - 1.0),
                )

        for window in (5, 10, 20, 60, 120):
            left = step - window + 1
            if left < 0:
                continue
            closes = features.close[left : step + 1, node_index]
            if np.isfinite(closes).all() and float(np.mean(closes)) > 0.0:
                updated_cells += _set_latest_feature(
                    features,
                    step,
                    node_index,
                    feature_index,
                    f"ma{window}_gap",
                    float(model_price / np.mean(closes) - 1.0),
                )

        volatility_by_window: dict[int, float] = {}
        for window in (5, 10, 20, 60):
            left = step - window + 1
            if left < 0:
                continue
            returns = features.returns_1d[left : step + 1, node_index]
            if np.isfinite(returns).all() and len(returns) > 1:
                volatility = float(np.std(returns, ddof=1))
                volatility_by_window[window] = volatility
                updated_cells += _set_latest_feature(
                    features,
                    step,
                    node_index,
                    feature_index,
                    f"volatility_{window}d",
                    volatility,
                )
        returns20 = features.returns_1d[max(0, step - 19) : step + 1, node_index]
        if len(returns20) == 20 and np.isfinite(returns20).all():
            downside = np.where(returns20 < 0.0, returns20, 0.0)
            updated_cells += _set_latest_feature(
                features,
                step,
                node_index,
                feature_index,
                "downside_volatility_20d",
                float(np.std(downside, ddof=1)),
            )
        if 20 in volatility_by_window and 60 in volatility_by_window and volatility_by_window[60] > 0.0:
            updated_cells += _set_latest_feature(
                features,
                step,
                node_index,
                feature_index,
                "volatility_ratio_20_60",
                volatility_by_window[20] / volatility_by_window[60],
            )

        if quote.high_price is not None and quote.low_price is not None:
            if quote.high_price >= quote.low_price >= 0.0:
                updated_cells += _set_latest_feature(
                    features,
                    step,
                    node_index,
                    feature_index,
                    "range_pct",
                    float((quote.high_price - quote.low_price) / price),
                )
        if quote.open_price is not None and quote.open_price > 0.0:
            updated_cells += _set_latest_feature(
                features,
                step,
                node_index,
                feature_index,
                "gap_open",
                    float(
                        quote.open_price * model_price_scale / previous_close - 1.0
                    ),
            )
            updated_cells += _set_latest_feature(
                features,
                step,
                node_index,
                feature_index,
                "intraday_return",
                float(price / quote.open_price - 1.0),
            )
        for window in (20, 120):
            left = step - window + 1
            if left < 0:
                continue
            closes = features.close[left : step + 1, node_index]
            if not np.isfinite(closes).all():
                continue
            high = float(np.max(closes))
            low = float(np.min(closes))
            updated_cells += _set_latest_feature(
                features,
                step,
                node_index,
                feature_index,
                f"drawdown_{window}d",
                float(model_price / high - 1.0) if high > 0.0 else float("nan"),
            )
            updated_cells += _set_latest_feature(
                features,
                step,
                node_index,
                feature_index,
                f"breakout_{window}d",
                float(model_price / low - 1.0) if low > 0.0 else float("nan"),
            )
            if window == 120 and high > low:
                updated_cells += _set_latest_feature(
                    features,
                    step,
                    node_index,
                    feature_index,
                    "range_position_120d",
                    float((model_price - low) / (high - low)),
                )

        applied.append(ticker)
        if quote.received_at:
            received_at[ticker] = quote.received_at
    return {
        "applied_tickers": applied,
        "updated_feature_cells": int(updated_cells),
        "received_at": received_at,
        "daily_prices": daily_prices,
    }


def apply_intraday_sensor_mask(
    batch: GraphBatch,
    features: FeaturePanel,
    quotes: Mapping[str, Quote],
    applied_tickers: set[str],
) -> dict[str, int]:
    """Hide unsensed intraday market state while retaining slow modalities."""

    stock_count = features.tradable_count
    ticker_index = {ticker: index for index, ticker in enumerate(features.tickers)}

    def is_dynamic(name: str) -> bool:
        return (
            name.startswith("return_")
            or "relative_return" in name
            or "cs_rank_return" in name
            or name.startswith("ma")
            or "drawdown" in name
            or "breakout" in name
            or "volatility" in name
            or "range" in name
            or "volume" in name
            or "value" in name
            or "cs_rank_value" in name
            or "amihud" in name
            or name.startswith("market_")
            or "gap_open" in name
            or "intraday" in name
        )

    def is_price_observed(name: str, quote: Quote) -> bool:
        if name.startswith("return_"):
            return True
        if name.startswith("ma") and name.endswith("_gap"):
            return True
        if name.startswith("volatility_") or name == "downside_volatility_20d":
            return True
        if name.startswith("drawdown_") or name.startswith("breakout_"):
            return True
        if name == "range_position_120d":
            return True
        if name == "range_pct":
            return quote.high_price is not None and quote.low_price is not None
        if name in {"gap_open", "intraday_return"}:
            return quote.open_price is not None and quote.open_price > 0.0
        return False

    dynamic_indices = [
        index for index, name in enumerate(features.feature_names) if is_dynamic(name)
    ]
    if dynamic_indices:
        batch.feature_mask[:stock_count, dynamic_indices] = 0.0
    directly_sensed = 0
    for ticker in sorted(applied_tickers):
        node_index = ticker_index.get(ticker)
        quote = quotes.get(ticker)
        if node_index is None or node_index >= stock_count or quote is None:
            continue
        observed_indices = [
            index
            for index, name in enumerate(features.feature_names)
            if is_price_observed(name, quote)
        ]
        if observed_indices:
            batch.feature_mask[node_index, observed_indices] = batch.available_mask[
                node_index, observed_indices
            ]
        directly_sensed += 1
    return {
        "intraday_dynamic_feature_count": len(dynamic_indices),
        "intraday_directly_sensed_node_count": directly_sensed,
        "intraday_imputed_dynamic_node_count": stock_count - directly_sensed,
        "intraday_hidden_dynamic_cell_count": int(
            (batch.feature_mask[:stock_count, dynamic_indices] < 0.5).sum().item()
        )
        if dynamic_indices
        else 0,
    }


class SignalEngine:
    def __init__(
        self,
        model_dir: str | Path,
        signal_model: str = "jepa_only_ridge",
        device: str = "cpu",
        live_event_paths: List[str] | None = None,
        latent_path_head_path: str | Path | None = None,
    ):
        self.model_dir = Path(model_dir)
        self.signal_model = signal_model
        self.device = torch.device(device)
        self.live_event_paths = merge_event_paths(live_event_paths)
        self.parent_model_path = self.model_dir / "graph_jepa_real.pt"
        self.checkpoint = torch.load(self.parent_model_path, map_location="cpu", weights_only=False)
        self.args: Dict = dict(self.checkpoint["args"])
        self.tickers: List[str] = list(self.checkpoint["tickers"])
        self.return_bundle: Dict = {}
        if self.signal_model != WORLD_MODEL_SIGNAL:
            with (self.model_dir / "return_models.pkl").open("rb") as file:
                self.return_bundle = pickle.load(file)
        self.names: Dict[str, str] = dict(self.checkpoint.get("names", self.return_bundle.get("names", {})))
        self.feature_names: List[str] = list(self.checkpoint["feature_names"])
        self.model = StockGraphJEPA(
            num_features=len(self.feature_names),
            hidden_dim=int(self.args["hidden_dim"]),
            num_layers=int(self.args["layers"]),
            ema_decay=float(self.args.get("ema_decay", 0.98)),
            latent_loss_weight=float(self.args.get("latent_loss_weight", 1.0)),
            state_loss_weight=float(self.args.get("state_loss_weight", 0.35)),
            current_imputation_loss_weight=float(
                self.args.get("current_imputation_loss_weight", 0.0)
            ),
            temporal_state_mode=str(self.args.get("temporal_state_mode", "direct")),
            feature_names=self.feature_names,
            temporal_residual_short_steps=int(self.args.get("temporal_residual_short_steps", 2)),
            temporal_head_steps=self.args.get("temporal_head_steps"),
            temporal_state_feature_weights=self.checkpoint.get(
                "temporal_state_feature_weights"
            ),
            temporal_state_context_skip=bool(
                self.args.get("temporal_state_context_skip", False)
            ),
            hybrid_fast_direct=bool(self.args.get("hybrid_fast_direct", False)),
            return_correlation_loss_weight=float(
                self.args.get("return_correlation_loss_weight", 0.0)
            ),
            entry_path_correlation_loss_weight=float(
                self.args.get("entry_path_correlation_loss_weight", 0.0)
            ),
            feature_means=self.checkpoint.get("train_mean"),
            feature_stds=self.checkpoint.get("train_std"),
            normalize_predictor_output=bool(self.args.get("normalize_predictor_output", False)),
            graph_neighbor_scale=float(self.args.get("graph_neighbor_scale", 1.0)),
            temporal_graph_neighbor_scale=self.args.get(
                "temporal_graph_neighbor_scale"
            ),
            temporal_stock_edge_scale=float(
                self.args.get("temporal_stock_edge_scale", 1.0)
            ),
            global_stock_context=bool(
                self.args.get("global_stock_context", False)
            ),
            downstream_auxiliary_loss_weight=float(
                self.args.get("downstream_auxiliary_loss_weight", 0.0)
            ),
            downstream_auxiliary_task_weights=self.args.get(
                "downstream_auxiliary_task_weights"
            ),
            downstream_market_loss_weight=float(
                self.args.get("downstream_market_loss_weight", 0.0)
            ),
            downstream_market_cost_bps=float(
                self.args.get("downstream_market_cost_bps", 50.0)
            ),
            downstream_transition_loss_weight=float(
                self.args.get("downstream_transition_loss_weight", 0.0)
            ),
            downstream_transition_pooling=str(
                self.args.get("downstream_transition_pooling", "mean")
            ),
            temporal_impact_loss_mix=float(
                self.args.get("temporal_impact_loss_mix", 0.0)
            ),
        ).to(self.device)
        self.model.load_state_dict(self.checkpoint["model_state"])
        self.model.eval()
        self._panel_cache_key: tuple | None = None
        self._panel_cache_features: FeaturePanel | None = None
        self._panel_cache_coverage: dict[str, int | float | str] | None = None
        self.latent_path_head: LoadedLatentPathHead | None = None
        if latent_path_head_path is not None:
            if self.signal_model != WORLD_MODEL_SIGNAL:
                raise ValueError("latent path head requires signal_model=world_model")
            self.latent_path_head = load_latent_path_head(
                checkpoint_path=latent_path_head_path,
                parent_model_path=self.parent_model_path,
                parent_checkpoint=self.checkpoint,
                device=self.device,
            )
        if self.signal_model != WORLD_MODEL_SIGNAL and self.signal_model not in self.return_bundle:
            raise ValueError(f"unknown signal model: {self.signal_model}")

    def _world_model_scores(
        self,
        batch,
        stock_count: int,
        train_mean: np.ndarray,
        train_std: np.ndarray,
    ) -> tuple[np.ndarray, Dict[str, np.ndarray], Dict[int, int], Dict[int, np.ndarray]]:
        offsets = configured_rollout_offsets(self.args.get("rollout_offsets"))
        forecast_by_horizon: Dict[int, np.ndarray] = {}
        predicted_latents: Dict[int, torch.Tensor] = {}
        rollout_steps: Dict[int, int] = {}
        context = self.model.encode_temporal_context(batch)
        for offset in offsets:
            steps = rollout_steps_for_offset(self.args, offset)
            z_pred = self.model.rollout_latent(context, steps=steps)
            forecast = self.model.predict_temporal_state(
                batch,
                z_pred,
                rollout_steps=steps,
                z_context=context,
            ).detach().cpu().numpy()[:stock_count]
            forecast_by_horizon[offset] = forecast
            predicted_latents[offset] = z_pred
            rollout_steps[offset] = steps
        scores, diagnostics = world_model_state_scores(
            forecast_by_horizon,
            self.feature_names,
            train_mean,
            train_std,
        )
        if self.latent_path_head is not None:
            horizons = tuple(sorted(forecast_by_horizon))
            latent_scores = np.stack(
                [
                    self.latent_path_head.model(
                        context[:stock_count],
                        predicted_latents[horizon][:stock_count],
                        horizon,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    for horizon in horizons
                ],
                axis=1,
            )
            base_scores = scores.copy()
            scores, blend_diagnostics = blend_latent_path_scores(
                diagnostics["predicted_entry_path_returns"],
                latent_scores,
                horizons,
                self.latent_path_head.latent_blend_weight,
            )
            diagnostics.update(blend_diagnostics)
            diagnostics["latent_path_head_raw_scores"] = latent_scores
            diagnostics["base_state_trajectory_scores"] = base_scores
        return scores, diagnostics, rollout_steps, forecast_by_horizon

    def _build_latest_panel(self, start: str, end: str | None, cache_dir: str):
        universe = [(ticker, self.names.get(ticker, ticker)) for ticker in self.tickers]
        raw = fetch_krx_ohlcv(
            universe=universe,
            start=start,
            end=end,
            cache_dir=cache_dir,
            refresh=False,
            min_rows=120,
        )
        missing = sorted(set(self.tickers) - set(raw))
        if missing:
            raise RuntimeError(f"missing OHLCV data for trained tickers: {missing[:10]}")
        panel = make_ohlcv_panel({ticker: raw[ticker] for ticker in self.tickers}, names=self.names)
        historical_event_paths = merge_event_paths(self.args.get("event_path", []))
        event_paths = merge_event_paths(
            historical_event_paths,
            self.live_event_paths,
        )
        event_feature_frames = None
        event_feature_names: list[str] = []
        event_ticker_coverage = None
        event_theme_exposure = None
        event_theme_names = []
        fundamental_feature_frames = None
        investor_feature_frames = None
        external_node_feature_frames = None
        external_node_returns = None
        external_node_names = {}
        static_edge_index = None
        static_edge_weight = None
        if event_paths:
            event_feature_frames = build_event_feature_frames(
                dates=panel.close.index,
                tickers=panel.tickers,
                event_paths=event_paths,
                half_life_days=float(self.args.get("event_half_life_days", 5.0)),
                lag_days=int(self.args.get("event_lag_days", 1)),
                max_decay_days=int(self.args.get("event_max_decay_days", 60)),
            )
            event_feature_names = list(event_feature_frames)
            if str(
                self.args.get("event_coverage_mode", "legacy_all_observed")
            ) == "mask_uncovered":
                event_ticker_coverage = build_event_ticker_coverage(
                    dates=panel.close.index,
                    tickers=panel.tickers,
                    event_paths=historical_event_paths,
                )
            if int(self.args.get("event_edge_top_k", 0) or 0) > 0:
                event_theme_exposure, event_theme_names = build_event_theme_exposure(
                    dates=panel.close.index,
                    tickers=panel.tickers,
                    event_paths=historical_event_paths,
                    half_life_days=float(self.args.get("event_half_life_days", 5.0)),
                    lag_days=int(self.args.get("event_lag_days", 1)),
                    max_decay_days=int(self.args.get("event_max_decay_days", 60)),
                    max_themes=int(self.args.get("event_edge_max_themes", 96)),
                    min_theme_count=int(self.args.get("event_edge_min_theme_count", 2)),
                )
        fundamental_paths = list(self.args.get("fundamental_path", []) or [])
        if fundamental_paths:
            fundamental_feature_frames = build_fundamental_feature_frames(
                dates=panel.close.index,
                tickers=panel.tickers,
                observations=load_fundamental_observations(fundamental_paths),
                lag_days=int(self.args.get("fundamental_lag_days", 1)),
            )
        investor_cache_dir = self.args.get("investor_cache_dir")
        if investor_cache_dir:
            investor_flow_frames = load_investor_flow_frames(
                cache_dir=str(investor_cache_dir),
                dates=panel.close.index,
                tickers=panel.tickers,
            )
            observed_close = panel.close.where(panel.price_observed)
            observed_volume = panel.volume.where(panel.price_observed)
            investor_feature_frames = build_investor_feature_frames(
                investor_flow_frames,
                traded_value=observed_close * observed_volume,
                lag_days=int(self.args.get("investor_flow_lag_days", 1)),
            )
        external_factors = resolve_external_factors(
            str(self.args.get("external_preset", "none") or "none"),
            list(self.args.get("external_symbol", []) or []),
        )
        if external_factors:
            factor_closes = fetch_external_factor_closes(
                external_factors,
                start=start,
                end=end,
                cache_dir=str(self.args.get("external_cache_dir", "data/external_cache")),
                refresh=False,
            )
            external_node_mode = str(self.args.get("external_node_mode", "features") or "features")
            external_lag_days = int(self.args.get("external_lag_days", 1))
            if external_node_mode in {"features", "both"}:
                external_feature_frames = build_external_feature_frames(
                    dates=panel.close.index,
                    tickers=panel.tickers,
                    factor_closes=factor_closes,
                    lag_days=external_lag_days,
                )
                if external_feature_frames:
                    event_feature_frames = dict(event_feature_frames or {})
                    event_feature_frames.update(external_feature_frames)
            if external_node_mode in {"nodes", "both"}:
                external_node_feature_frames, external_node_returns, external_node_names = build_external_node_feature_frames(
                    dates=panel.close.index,
                    factor_closes=factor_closes,
                    lag_days=external_lag_days,
                )
        industry_profile_paths = list(self.args.get("industry_profile_path", []) or [])
        if industry_profile_paths:
            industry_codes = load_industry_codes(industry_profile_paths)
            static_edge_index, static_edge_weight, _industry_stats = build_industry_edge_arrays(
                panel.tickers,
                industry_codes,
                prefix_length=int(self.args.get("industry_prefix_length", 2)),
                scale=float(self.args.get("industry_edge_scale", 0.20)),
            )
        features = build_feature_panel(
            panel,
            horizon=int(self.args.get("horizon", 5)),
            train_end=str(self.args.get("train_end", "2023-12-29")),
            require_targets=False,
            feature_names=self.feature_names,
            event_feature_frames=event_feature_frames,
            event_feature_names=event_feature_names,
            event_ticker_coverage=event_ticker_coverage,
            fundamental_feature_frames=fundamental_feature_frames,
            investor_feature_frames=investor_feature_frames,
            external_node_feature_frames=external_node_feature_frames,
            external_node_returns=external_node_returns,
            external_node_names=external_node_names,
            event_theme_exposure=event_theme_exposure,
            event_theme_names=event_theme_names,
            static_edge_index=static_edge_index,
            static_edge_weight=static_edge_weight,
            path_horizons=configured_rollout_offsets(
                self.args.get("rollout_offsets")
            ),
        )
        aligned = self._align_features(features)
        validate_checkpoint_panel(
            self.checkpoint,
            aligned,
            str(self.args.get("train_end", "2023-12-29")),
        )
        return aligned

    @staticmethod
    def _path_fingerprint(value: str | Path) -> tuple[str, int | None, int | None]:
        path = Path(value).expanduser()
        try:
            resolved = path.resolve(strict=True)
            stat = resolved.stat()
            return str(resolved), int(stat.st_mtime_ns), int(stat.st_size)
        except FileNotFoundError:
            return str(path), None, None

    def _latest_panel_cache_key(
        self,
        start: str,
        end: str | None,
        cache_dir: str,
        min_coverage_ratio: float,
        coverage_lookback: int,
    ) -> tuple:
        cache = Path(cache_dir)
        tracked_paths: list[str | Path] = [
            cache,
            cache / "manifest.json",
            cache.parent / "manifest.json",
            *merge_event_paths(self.args.get("event_path", []), self.live_event_paths),
            *list(self.args.get("fundamental_path", []) or []),
        ]
        for key in ("investor_cache_dir", "external_cache_dir"):
            if self.args.get(key):
                tracked_paths.append(str(self.args[key]))
        return (
            str(start),
            None if end is None else str(end),
            float(min_coverage_ratio),
            int(coverage_lookback),
            tuple(self._path_fingerprint(path) for path in tracked_paths),
        )

    def _prepare_latest_panel(
        self,
        start: str,
        end: str | None,
        cache_dir: str,
        min_coverage_ratio: float,
        coverage_lookback: int,
    ) -> tuple[FeaturePanel, dict[str, int | float | str]]:
        cache_key = self._latest_panel_cache_key(
            start,
            end,
            cache_dir,
            min_coverage_ratio,
            coverage_lookback,
        )
        if (
            self._panel_cache_key != cache_key
            or self._panel_cache_features is None
            or self._panel_cache_coverage is None
        ):
            features = self._build_latest_panel(
                start=start,
                end=end,
                cache_dir=cache_dir,
            )
            coverage = select_latest_complete_step(
                features,
                min_coverage_ratio=min_coverage_ratio,
                lookback=coverage_lookback,
            )
            self._panel_cache_key = cache_key
            self._panel_cache_features = features
            self._panel_cache_coverage = coverage
        return self._panel_cache_features, dict(self._panel_cache_coverage)

    def _align_features(self, features: FeaturePanel) -> FeaturePanel:
        if features.feature_names == self.feature_names:
            return features

        missing = [name for name in self.feature_names if name not in features.feature_names]
        if missing:
            raise RuntimeError(f"current feature builder is missing model features: {missing}")

        indices = [features.feature_index(name) for name in self.feature_names]
        return FeaturePanel(
            dates=features.dates,
            tickers=features.tickers,
            names=features.names,
            feature_names=list(self.feature_names),
            features=features.features[:, :, indices],
            raw_features=features.raw_features[:, :, indices],
            available_mask=features.available_mask[:, :, indices],
            returns_1d=features.returns_1d,
            target_returns=features.target_returns,
            target_return_paths=features.target_return_paths,
            open=features.open,
            close=features.close,
            train_mean=features.train_mean[indices],
            train_std=features.train_std[indices],
            event_theme_exposure=features.event_theme_exposure,
            event_theme_names=features.event_theme_names,
            static_edge_index=features.static_edge_index,
            static_edge_weight=features.static_edge_weight,
            node_tickers=features.node_tickers,
            node_names=features.node_names,
            stock_node_count=features.stock_node_count,
            execution_close=features.execution_close,
        )

    @torch.no_grad()
    def latest_signals(
        self,
        start: str,
        end: str | None,
        cache_dir: str,
        top_n: int = 20,
        intraday_quotes: Mapping[str, Quote] | None = None,
        intraday_session_date: str | pd.Timestamp | None = None,
        max_intraday_missing_business_days: int | None = None,
        min_coverage_ratio: float = 0.90,
        coverage_lookback: int = 20,
        min_price: float = 0.0,
        max_price: float = float("inf"),
    ) -> list[Signal]:
        features, coverage = self._prepare_latest_panel(
            start=start,
            end=end,
            cache_dir=cache_dir,
            min_coverage_ratio=min_coverage_ratio,
            coverage_lookback=coverage_lookback,
        )
        base_step = int(coverage["step"])
        overlay_session = None
        missing_business_days = 0
        if intraday_quotes:
            overlay_session = pd.Timestamp(
                intraday_session_date or features.dates[-1]
            ).normalize()
            missing_business_days = missing_business_days_between(
                features.dates[base_step],
                overlay_session,
            )
            if (
                max_intraday_missing_business_days is not None
                and missing_business_days > int(max_intraday_missing_business_days)
            ):
                raise RuntimeError(
                    "daily state is too stale for intraday overlay: "
                    f"base={features.dates[base_step].date()} "
                    f"session={overlay_session.date()} "
                    f"missing_business_days={missing_business_days}"
                )
            step = materialize_quote_overlay_session(
                features,
                base_step,
                overlay_session,
            )
        else:
            step = base_step
        quote_overlay = apply_intraday_quote_overlay(
            features,
            intraday_quotes,
            step=step,
        )
        quote_overlay_tickers = set(quote_overlay["applied_tickers"])
        batch = make_real_snapshot(
            features,
            step=step,
            full_observation=True,
            edge_window=int(self.args.get("edge_window", 60)),
            top_k=int(self.args.get("edge_top_k", 6)),
            min_abs_corr=float(self.args.get("min_abs_corr", 0.20)),
            correlation_mode=str(self.args.get("edge_correlation_mode", "signed")),
            event_top_k=int(self.args.get("event_edge_top_k", 0) or 0),
            event_min_weight=float(self.args.get("event_edge_min_weight", 0.05)),
            event_scale=float(self.args.get("event_edge_scale", 0.25)),
            partial_corr_top_k=int(self.args.get("partial_corr_top_k", 0) or 0),
            partial_corr_min_abs=float(self.args.get("partial_corr_min_abs", 0.10)),
            partial_corr_mode=str(self.args.get("partial_corr_mode", "signed")),
            partial_corr_scale=float(self.args.get("partial_corr_scale", 0.50)),
            lead_lag_top_k=int(self.args.get("lead_lag_top_k", 0) or 0),
            lead_lag_days=int(self.args.get("lead_lag_days", 1) or 1),
            lead_lag_min_abs_corr=float(self.args.get("lead_lag_min_abs_corr", 0.08)),
            lead_lag_mode=str(self.args.get("lead_lag_mode", "signed")),
            lead_lag_scale=float(self.args.get("lead_lag_scale", 0.50)),
            policy_rate_edge_scale=float(self.args.get("policy_rate_edge_scale", 0.0)),
        )
        sensor_mask_diagnostics: dict[str, int] = {}
        if intraday_quotes:
            sensor_mask_diagnostics = apply_intraday_sensor_mask(
                batch,
                features,
                intraday_quotes,
                quote_overlay_tickers,
            )
        batch = batch.to(self.device)
        stock_count = features.tradable_count
        raw_features = features.features[step, :stock_count]
        diagnostics: Dict[str, np.ndarray] = {}
        rollout_steps: Dict[int, int] = {}
        forecast_by_horizon: Dict[int, np.ndarray] = {}
        if self.signal_model == WORLD_MODEL_SIGNAL:
            scores, diagnostics, rollout_steps, forecast_by_horizon = self._world_model_scores(
                batch,
                stock_count=stock_count,
                train_mean=features.train_mean,
                train_std=features.train_std,
            )
        else:
            encoded = self.model.encode_context(batch).detach().cpu().numpy()[:stock_count]
            if self.signal_model == "jepa_only_ridge":
                design = encoded
            elif self.signal_model == "raw_ridge":
                design = raw_features
            else:
                design = np.concatenate([encoded, raw_features], axis=1)
            scores = self.return_bundle[self.signal_model].predict(design)
        prices = (
            features.execution_close[step, :stock_count]
            if features.execution_close is not None
            else features.close[step, :stock_count]
        )
        return_1d_index = features.feature_index("return_1d")
        order = rank_tradeable_scores(
            scores=scores,
            prices=prices,
            return_1d_available=features.available_mask[step, :stock_count, return_1d_index] > 0.5,
            top_n=top_n,
            min_price=min_price,
            max_price=max_price,
        )
        asof = str(features.dates[step].date())
        result = []
        for rank, idx in enumerate(order, start=1):
            ticker = features.tickers[int(idx)]
            metadata = {
                "return_20d": float(raw_features[int(idx), features.feature_index("return_20d")]),
                "volume_z20": float(raw_features[int(idx), features.feature_index("volume_z20")]),
                "news_score_decay": (
                    float(raw_features[int(idx), features.feature_index("news_score_decay")])
                    if "news_score_decay" in features.feature_names
                    else None
                ),
                "news_count_10d": (
                    float(raw_features[int(idx), features.feature_index("news_count_10d")])
                    if "news_count_10d" in features.feature_names
                    else None
                ),
                "model_dir": str(self.model_dir),
                "model_input_state": (
                    "partial_intraday_quote_overlay"
                    if ticker in quote_overlay_tickers
                    else (
                        "jepa_imputed_intraday_state"
                        if overlay_session is not None
                        else "daily_close"
                    )
                ),
                "model_input_quote_count": len(quote_overlay["applied_tickers"]),
                "model_state_updated_from_quote": ticker in quote_overlay_tickers,
                "intraday_session_date": (
                    str(overlay_session.date()) if overlay_session is not None else None
                ),
                "daily_state_missing_business_days": int(missing_business_days),
                "daily_close_before_quote_overlay": quote_overlay["daily_prices"].get(ticker),
                "daily_state_date": coverage["date"],
                "daily_state_observed_stocks": coverage["observed_stocks"],
                "daily_state_reference_stocks": coverage["reference_stocks"],
                "panel_latest_date": coverage["panel_latest_date"],
                "panel_latest_observed_stocks": coverage["panel_latest_observed_stocks"],
                **sensor_mask_diagnostics,
            }
            if self.signal_model == WORLD_MODEL_SIGNAL:
                latent_head_metadata = {}
                if self.latent_path_head is not None:
                    latent_head_metadata = {
                        "latent_path_head_score": {
                            str(horizon): float(
                                diagnostics["latent_path_head_raw_scores"][int(idx), position]
                            )
                            for position, horizon in enumerate(sorted(forecast_by_horizon))
                        },
                        "latent_path_head_zscore": {
                            str(horizon): float(
                                diagnostics["latent_path_head_zscores"][int(idx), position]
                            )
                            for position, horizon in enumerate(sorted(forecast_by_horizon))
                        },
                        "blended_entry_path_score": {
                            str(horizon): float(
                                diagnostics["blended_entry_path_scores"][int(idx), position]
                            )
                            for position, horizon in enumerate(sorted(forecast_by_horizon))
                        },
                        "latent_path_head_path": str(
                            self.latent_path_head.checkpoint_path
                        ),
                        "latent_path_head_sha256": (
                            self.latent_path_head.checkpoint_sha256
                        ),
                        "parent_model_sha256": (
                            self.latent_path_head.parent_model_sha256
                        ),
                        "latent_blend_weight": (
                            self.latent_path_head.latent_blend_weight
                        ),
                    }
                metadata.update(
                    {
                        "rollout_horizons": sorted(rollout_steps),
                        "rollout_steps": {str(horizon): int(steps) for horizon, steps in rollout_steps.items()},
                        "forecast_return_1d": {
                            str(horizon): float(
                                forecast_by_horizon[horizon][int(idx), self.feature_names.index("return_1d")]
                                * features.train_std[self.feature_names.index("return_1d")]
                                + features.train_mean[self.feature_names.index("return_1d")]
                            )
                            for horizon in sorted(forecast_by_horizon)
                        },
                        "forecast_matched_path_return": {
                            str(horizon): float(
                                diagnostics["predicted_entry_path_returns"][
                                    int(idx), position
                                ]
                            )
                            for position, horizon in enumerate(
                                sorted(forecast_by_horizon)
                            )
                        },
                        "forecast_horizon_close_return": {
                            str(horizon): float(
                                forecast_by_horizon[horizon][
                                    int(idx),
                                    self.feature_names.index(f"return_{horizon}d"),
                                ]
                                * features.train_std[
                                    self.feature_names.index(f"return_{horizon}d")
                                ]
                                + features.train_mean[
                                    self.feature_names.index(f"return_{horizon}d")
                                ]
                            )
                            for horizon in sorted(forecast_by_horizon)
                        },
                        "trajectory_expected_return_1d": float(diagnostics["expected_return_1d"][int(idx)]),
                        "trajectory_expected_dailyized_path_return": float(
                            diagnostics["expected_dailyized_path_return"][int(idx)]
                        ),
                        "trajectory_expected_dailyized_entry_path_return": float(
                            diagnostics["expected_dailyized_entry_path_return"][int(idx)]
                        ),
                        "predicted_next_open_gap": float(
                            diagnostics["predicted_next_open_gap"][int(idx)]
                        ),
                        "trajectory_risk_penalty": float(diagnostics["risk_penalty"][int(idx)]),
                        "signal_contract": (
                            "latent_trajectory_residual_head_shadow_v1"
                            if self.latent_path_head is not None
                            else "multi_horizon_state_trajectory_shadow"
                        ),
                        **latent_head_metadata,
                    }
                )
            result.append(
                Signal(
                    ticker=ticker,
                    name=features.names.get(ticker, ticker),
                    score=float(scores[int(idx)]),
                    rank=rank,
                    price=int(round(float(prices[int(idx)]))),
                    model=self.signal_model,
                    asof=asof,
                    metadata=metadata,
                )
            )
        return result
