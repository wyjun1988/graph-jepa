from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

from stock_v2.backtest import performance_metrics
from stock_v2.downstream_probes import newey_west_mean
from stock_v2.graph_jepa import DOWNSTREAM_AUXILIARY_TASKS


DEFAULT_AUXILIARY_WEIGHTS = {
    "path_return": 1.0,
    "max_favorable_excursion": 0.25,
    "max_adverse_excursion": 0.50,
    "realized_volatility": -0.50,
}


@dataclass(frozen=True)
class AuxiliaryPolicy:
    horizon: int = 5
    top_k: int = 10
    liquidity_top_n: int = 300
    task_weights: Mapping[str, float] = field(
        default_factory=lambda: dict(DEFAULT_AUXILIARY_WEIGHTS)
    )

    def __post_init__(self) -> None:
        if self.horizon < 1 or self.top_k < 1 or self.liquidity_top_n < self.top_k:
            raise ValueError("auxiliary policy dimensions are invalid")
        unknown = sorted(set(self.task_weights) - set(DOWNSTREAM_AUXILIARY_TASKS))
        if unknown:
            raise ValueError(f"unknown auxiliary tasks: {unknown}")
        weights = np.asarray(
            [float(self.task_weights.get(name, 0.0)) for name in DOWNSTREAM_AUXILIARY_TASKS],
            dtype=np.float64,
        )
        if not np.isfinite(weights).all() or not np.any(np.abs(weights) > 0.0):
            raise ValueError("auxiliary task weights must be finite and non-zero")


def liquid_universe_mask(
    liquidity: np.ndarray,
    eligible: np.ndarray,
    top_n: int,
) -> np.ndarray:
    liquidity = np.asarray(liquidity, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if liquidity.shape != eligible.shape or liquidity.ndim != 1:
        raise ValueError("liquidity and eligibility must be aligned vectors")
    selected = eligible & np.isfinite(liquidity)
    indices = np.flatnonzero(selected)
    result = np.zeros_like(selected)
    if not len(indices) or top_n < 1:
        return result
    order = indices[np.argsort(liquidity[indices], kind="stable")[::-1]]
    result[order[: int(top_n)]] = True
    return result


def cross_sectional_zscore(values: np.ndarray, eligible: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if values.shape != eligible.shape or values.ndim != 1:
        raise ValueError("values and eligibility must be aligned vectors")
    valid = eligible & np.isfinite(values)
    result = np.full(values.shape, np.nan, dtype=np.float32)
    if valid.sum() < 3:
        return result
    sample = values[valid]
    scale = float(sample.std())
    if not np.isfinite(scale) or scale < 1e-12:
        return result
    result[valid] = ((sample - sample.mean()) / scale).astype(np.float32)
    return result


def combine_auxiliary_predictions(
    predictions: np.ndarray,
    eligible: np.ndarray,
    task_weights: Mapping[str, float],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    predictions = np.asarray(predictions, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    expected = (len(eligible), len(DOWNSTREAM_AUXILIARY_TASKS))
    if predictions.shape != expected:
        raise ValueError(f"auxiliary predictions must have shape {expected}")
    unknown = sorted(set(task_weights) - set(DOWNSTREAM_AUXILIARY_TASKS))
    if unknown:
        raise ValueError(f"unknown auxiliary tasks: {unknown}")

    score = np.zeros(len(eligible), dtype=np.float64)
    components: dict[str, np.ndarray] = {}
    used = np.zeros(len(eligible), dtype=bool)
    for task_index, task_name in enumerate(DOWNSTREAM_AUXILIARY_TASKS):
        weight = float(task_weights.get(task_name, 0.0))
        if not np.isfinite(weight):
            raise ValueError("auxiliary task weights must be finite")
        component = cross_sectional_zscore(predictions[:, task_index], eligible)
        components[task_name] = component
        if weight == 0.0:
            continue
        valid = np.isfinite(component)
        used |= valid
        score[valid] += weight * component[valid]
    score[~eligible | ~used] = np.nan
    return score.astype(np.float32), components


def evaluate_ranked_strategy(
    scores: np.ndarray,
    target_returns: np.ndarray,
    eligible: np.ndarray,
    dates: Sequence[str],
    tickers: Sequence[str],
    *,
    top_k: int,
    stride: int,
    cost_bps: float,
    risk_free_returns: np.ndarray | None = None,
) -> dict[str, object]:
    scores = np.asarray(scores, dtype=np.float64)
    target_returns = np.asarray(target_returns, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if scores.shape != target_returns.shape or scores.shape != eligible.shape:
        raise ValueError("strategy matrices must share a shape")
    if scores.ndim != 2 or scores.shape[0] != len(dates) or scores.shape[1] != len(tickers):
        raise ValueError("strategy matrices do not align with dates and tickers")
    if top_k < 1 or stride < 1 or cost_bps < 0.0:
        raise ValueError("strategy parameters are invalid")
    if risk_free_returns is not None:
        risk_free_returns = np.asarray(risk_free_returns, dtype=np.float64)
        if risk_free_returns.shape != (len(dates),):
            raise ValueError("risk-free returns must align with dates")

    cost = float(cost_bps) / 10_000.0
    rows: list[dict[str, object]] = []
    previous: set[int] = set()
    for date_index in range(0, len(dates), int(stride)):
        valid = eligible[date_index] & np.isfinite(scores[date_index])
        candidates = np.flatnonzero(valid)
        if len(candidates) < int(top_k):
            continue
        risk_free = (
            float(risk_free_returns[date_index])
            if risk_free_returns is not None
            else float("nan")
        )
        if risk_free_returns is not None and not np.isfinite(risk_free):
            continue
        ranked = candidates[
            np.argsort(scores[date_index, candidates], kind="stable")[::-1]
        ]
        chosen = ranked[: int(top_k)]
        selected = set(int(value) for value in chosen)
        turnover = 1.0 if not previous else 1.0 - len(previous & selected) / float(top_k)
        missing_fallback = risk_free if np.isfinite(risk_free) else 0.0
        candidate_targets = target_returns[date_index, candidates]
        selected_targets = target_returns[date_index, chosen]
        candidate_missing = ~np.isfinite(candidate_targets)
        selected_missing = ~np.isfinite(selected_targets)
        benchmark = float(
            np.where(candidate_missing, missing_fallback, candidate_targets).mean()
            - cost
        )
        period_return = float(
            np.where(selected_missing, missing_fallback, selected_targets).mean()
            - cost
        )
        rows.append(
            {
                "date": str(dates[date_index]),
                "selected": [str(tickers[index]) for index in chosen],
                "period_return": period_return,
                "benchmark_return": benchmark,
                "risk_free_return": risk_free,
                "turnover": float(turnover),
                "missing_target_selected": int(selected_missing.sum()),
                "missing_target_candidates": int(candidate_missing.sum()),
                "missing_target_policy": "risk_free_fallback_full_cost",
            }
        )
        previous = selected

    period_returns = np.asarray([row["period_return"] for row in rows], dtype=np.float64)
    benchmark_returns = np.asarray(
        [row["benchmark_return"] for row in rows], dtype=np.float64
    )
    risk_free = (
        np.asarray([row["risk_free_return"] for row in rows], dtype=np.float64)
        if risk_free_returns is not None
        else None
    )
    metrics = performance_metrics(
        period_returns,
        periods_per_year=252.0 / float(stride),
        risk_free_returns=risk_free,
    )
    alpha = period_returns - benchmark_returns
    metrics.update(
        {
            "mean_turnover": float(np.mean([row["turnover"] for row in rows]))
            if rows
            else float("nan"),
            "worst_period_return": float(period_returns.min())
            if len(period_returns)
            else float("nan"),
            "alpha_vs_equal_weight": newey_west_mean(alpha, lag=1),
        }
    )
    return {
        "cost_bps": float(cost_bps),
        "top_k": int(top_k),
        "stride": int(stride),
        "metrics": metrics,
        "period_returns": period_returns.tolist(),
        "rows": rows,
    }


def paired_strategy_premium(
    candidate: Mapping[str, object],
    baseline: Mapping[str, object],
) -> dict[str, float | int]:
    left = np.asarray(candidate["period_returns"], dtype=np.float64)
    right = np.asarray(baseline["period_returns"], dtype=np.float64)
    if left.shape != right.shape:
        raise ValueError("paired strategies must contain the same rebalance periods")
    return newey_west_mean(left - right, lag=1)
