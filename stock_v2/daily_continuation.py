"""The user's intent 2 at the daily scale: once a shock is observed, does it continue?

configs/post_impact_continuation_gate_v1.json states the question and the label
for intraday cells, and says outright why the definition travels:

    "Dividing both by their durations asks whether the move sustains its
     intensity, which is the intended question and is scale-free."

So the same construction applies to daily bars, and needs nothing but the
lifecycle release the JEPA already trains on -- no intraday release, no waiting
for the 2026-07-20 forward window.

  observed_rate(t)   = |return_1d(t)| / 1 day
  future_rate(t, h)  = |close(t+h)/close(t) - 1| / h days
  label              = future_rate >= observed_rate
  in scope           = the observed move is a SHOCK relative to the node's own
                       recent baseline, and is strictly positive

Two properties carried over deliberately from the frozen contract:

  * the shock condition normalises by the node's OWN baseline, so a habitually
    volatile stock does not qualify for free. The intraday version uses
    realized_absolute_return_15m_shock_20; the daily analogue is
    |return_1d| / volatility_20d.
  * the label is a RATE comparison against the node's own just-observed
    intensity, so a volatile stock gets no free positive either. This is what
    makes the question "does it continue" rather than "is it big", and it is
    why the retrospective frontier study found the absolute definition was
    dominated by cross-sectional volatility sorting (AUC 0.829) while the
    continuation definition collapsed the same features to 0.62.

Everything here is causal: observed_rate and the shock condition read bar t and
earlier only; the label reads t+1..t+h and is never an input.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# The intraday gate takes the 80th percentile of the shock statistic across all
# nodes at the primary clocks in its calibration release. The same rule is
# applied here to the daily statistic, computed on the training window only --
# a threshold fitted on evaluation dates would leak the evaluation regime.
SHOCK_PERCENTILE = 80.0


@dataclass
class ContinuationCells:
    """One (date, horizon) slice of the daily continuation question."""

    observed_rate: np.ndarray  # |return_1d(t)|, per day
    future_rate: np.ndarray  # |cumulative move t+1..t+h| / h, per day
    shock_statistic: np.ndarray  # |return_1d(t)| / volatility_20d(t)
    in_scope: np.ndarray  # bool: shocked, and both rates usable
    label: np.ndarray  # float 1.0/0.0 where in_scope, else nan

    @property
    def base_rate(self) -> float:
        scoped = self.label[self.in_scope]
        scoped = scoped[np.isfinite(scoped)]
        return float(scoped.mean()) if scoped.size else float("nan")


def shock_statistic(
    returns_1d: np.ndarray,
    volatility_20d: np.ndarray,
    step: int,
    stock_count: int,
) -> np.ndarray:
    """|today's move| measured in units of the node's own 20-day volatility.

    The denominator is what stops this from being a volatility sort: a stock
    that moves 5% every day is not shocked when it moves 5% today.
    """

    move = np.abs(np.asarray(returns_1d[int(step), :stock_count], dtype=np.float64))
    baseline = np.asarray(volatility_20d[int(step), :stock_count], dtype=np.float64)
    usable = np.isfinite(move) & np.isfinite(baseline) & (baseline > 1e-8)
    out = np.full(stock_count, np.nan)
    out[usable] = move[usable] / baseline[usable]
    return out


def continuation_threshold(statistics: np.ndarray, percentile: float = SHOCK_PERCENTILE) -> float:
    """The shock cut, fitted on training dates only."""

    values = np.asarray(statistics, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size < 100:
        raise ValueError("not enough finite shock statistics to fit a threshold")
    return float(np.percentile(values, percentile))


def build_cells(
    *,
    close: np.ndarray,
    returns_1d: np.ndarray,
    volatility_20d: np.ndarray,
    step: int,
    horizon: int,
    stock_count: int,
    threshold: float,
) -> ContinuationCells:
    """The daily continuation question for one decision date and one horizon."""

    step = int(step)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if step + horizon >= close.shape[0]:
        raise ValueError("horizon runs past the panel")

    observed = np.abs(np.asarray(returns_1d[step, :stock_count], dtype=np.float64))
    entry = np.asarray(close[step, :stock_count], dtype=np.float64)
    exit_ = np.asarray(close[step + horizon, :stock_count], dtype=np.float64)
    priced = np.isfinite(entry) & np.isfinite(exit_) & (entry > 0.0)
    future = np.full(stock_count, np.nan)
    future[priced] = np.abs(exit_[priced] / entry[priced] - 1.0) / float(horizon)

    statistic = shock_statistic(returns_1d, volatility_20d, step, stock_count)

    # The observed rate is the ratio's denominator, so a zero move would make it
    # undefined -- the frozen contract excludes those explicitly and so does this.
    in_scope = (
        np.isfinite(statistic)
        & (statistic >= threshold)
        & np.isfinite(observed)
        & (observed > 0.0)
        & np.isfinite(future)
    )
    label = np.full(stock_count, np.nan)
    label[in_scope] = (future[in_scope] >= observed[in_scope]).astype(np.float64)
    return ContinuationCells(
        observed_rate=observed,
        future_rate=future,
        shock_statistic=statistic,
        in_scope=in_scope,
        label=label,
    )


def continuation_ratio_score(predicted_future_rate: np.ndarray, observed_rate: np.ndarray) -> np.ndarray:
    """The contract's model score: predicted_future_rate / observed_rate.

    A ratio and not the raw predicted magnitude, for the reason the contract
    gives: "Ranking by the raw predicted magnitude would reintroduce exactly the
    cross-sectional volatility sorting this gate exists to remove, and would let
    a model score well by knowing which stocks are volatile rather than which
    shocks continue."
    """

    predicted = np.asarray(predicted_future_rate, dtype=np.float64)
    observed = np.asarray(observed_rate, dtype=np.float64)
    out = np.full(predicted.shape, np.nan)
    usable = np.isfinite(predicted) & np.isfinite(observed) & (observed > 0.0)
    out[usable] = predicted[usable] / observed[usable]
    return out
