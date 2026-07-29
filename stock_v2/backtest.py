from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd


@dataclass
class TradeRecord:
    date: pd.Timestamp
    strategy: str
    selected: List[str]
    period_return: float
    benchmark_return: float
    risk_free_return: float
    excess_period_return: float
    benchmark_excess_return: float


def performance_metrics(
    returns: Sequence[float],
    periods_per_year: float,
    risk_free_returns: Sequence[float] | None = None,
) -> Dict[str, float]:
    arr = np.asarray(returns, dtype=np.float64)
    if arr.size == 0:
        return {
            "periods": 0,
            "total_return": 0.0,
            "cagr": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "hit_rate": 0.0,
            "avg_period_return": 0.0,
        }

    equity = np.cumprod(1.0 + arr)
    total_return = equity[-1] - 1.0
    years = arr.size / periods_per_year
    cagr = equity[-1] ** (1.0 / max(years, 1e-9)) - 1.0
    vol = arr.std(ddof=1) if arr.size > 1 else 0.0
    sharpe = 0.0 if vol == 0.0 else arr.mean() / vol * np.sqrt(periods_per_year)
    peak = np.maximum.accumulate(equity)
    drawdown = equity / peak - 1.0

    metrics = {
        "periods": int(arr.size),
        "total_return": float(total_return),
        "cagr": float(cagr),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
        "hit_rate": float((arr > 0.0).mean()),
        "avg_period_return": float(arr.mean()),
    }
    if risk_free_returns is not None:
        risk_free = np.asarray(risk_free_returns, dtype=np.float64)
        if risk_free.shape != arr.shape:
            raise ValueError("risk_free_returns must align with returns")
        if not np.isfinite(risk_free).all():
            raise ValueError("risk_free_returns must be finite")
        excess = arr - risk_free
        excess_metrics = performance_metrics(excess, periods_per_year)
        risk_free_metrics = performance_metrics(risk_free, periods_per_year)
        strategy_growth = float(np.prod(1.0 + arr))
        risk_free_growth = float(np.prod(1.0 + risk_free))
        relative_growth = (
            strategy_growth / risk_free_growth if risk_free_growth > 0.0 else float("nan")
        )
        relative_cagr = (
            relative_growth ** (periods_per_year / float(arr.size)) - 1.0
            if relative_growth > 0.0
            else float("nan")
        )
        metrics.update(
            {
                "risk_free_total_return": risk_free_metrics["total_return"],
                "risk_free_cagr": risk_free_metrics["cagr"],
                "excess_total_return": relative_growth - 1.0,
                "excess_cagr": relative_cagr,
                "excess_sharpe": excess_metrics["sharpe"],
                "excess_hit_rate": excess_metrics["hit_rate"],
                "avg_excess_period_return": excess_metrics["avg_period_return"],
            }
        )
    return metrics


def run_rank_backtest(
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
    scores: np.ndarray,
    target_returns: np.ndarray,
    test_indices: Sequence[int],
    top_k: int = 5,
    horizon: int = 5,
    cost_bps: float = 30.0,
    strategy_name: str = "graph_jepa_ridge",
    risk_free_returns: np.ndarray | None = None,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Non-overlapping horizon rebalance ranking backtest."""

    if scores.shape != target_returns.shape:
        raise ValueError("scores and target_returns must have the same shape")
    if risk_free_returns is not None:
        risk_free_returns = np.asarray(risk_free_returns, dtype=np.float64)
        if risk_free_returns.shape != (len(dates),):
            raise ValueError("risk_free_returns must have one value per date")

    roundtrip_cost = cost_bps / 10_000.0
    rows: List[TradeRecord] = []
    selected_indices = list(test_indices)[::horizon]

    for idx in selected_indices:
        risk_free_return = (
            float(risk_free_returns[idx]) if risk_free_returns is not None else float("nan")
        )
        if risk_free_returns is not None and not np.isfinite(risk_free_return):
            continue
        score = scores[idx]
        realized = target_returns[idx]
        valid = np.isfinite(score) & np.isfinite(realized)
        if valid.sum() < top_k:
            continue

        ranked = np.argsort(score[valid])[-top_k:][::-1]
        valid_indices = np.flatnonzero(valid)
        chosen = valid_indices[ranked]
        period_return = float(realized[chosen].mean() - roundtrip_cost)
        benchmark_return = float(realized[valid].mean() - roundtrip_cost)
        excess_period_return = (
            period_return - risk_free_return if risk_free_returns is not None else float("nan")
        )
        benchmark_excess_return = (
            benchmark_return - risk_free_return if risk_free_returns is not None else float("nan")
        )
        rows.append(
            TradeRecord(
                date=dates[idx],
                strategy=strategy_name,
                selected=[tickers[i] for i in chosen.tolist()],
                period_return=period_return,
                benchmark_return=benchmark_return,
                risk_free_return=risk_free_return,
                excess_period_return=excess_period_return,
                benchmark_excess_return=benchmark_excess_return,
            )
        )

    frame = pd.DataFrame(
        [
            {
                "date": row.date,
                "strategy": row.strategy,
                "selected": ",".join(row.selected),
                "period_return": row.period_return,
                "benchmark_return": row.benchmark_return,
                "risk_free_return": row.risk_free_return,
                "excess_period_return": row.excess_period_return,
                "benchmark_excess_return": row.benchmark_excess_return,
            }
            for row in rows
        ]
    )
    periods_per_year = 252.0 / max(horizon, 1)
    metrics = {
        strategy_name: performance_metrics(
            frame["period_return"].to_numpy(),
            periods_per_year,
            frame["risk_free_return"].to_numpy() if risk_free_returns is not None else None,
        )
        if not frame.empty
        else performance_metrics([], periods_per_year),
        "equal_weight_benchmark": performance_metrics(
            frame["benchmark_return"].to_numpy(),
            periods_per_year,
            frame["risk_free_return"].to_numpy() if risk_free_returns is not None else None,
        )
        if not frame.empty
        else performance_metrics([], periods_per_year),
    }
    return frame, metrics


@dataclass
class PathTradeRecord:
    date: pd.Timestamp
    strategy: str
    selected: List[str]
    exit_horizons: List[int]
    period_return: float
    benchmark_return: float
    risk_free_return: float
    excess_period_return: float
    benchmark_excess_return: float


def run_path_rank_backtest(
    dates: pd.DatetimeIndex,
    tickers: Sequence[str],
    scores: np.ndarray,
    exit_horizons: np.ndarray,
    target_returns_by_horizon: Dict[int, np.ndarray],
    test_indices: Sequence[int],
    top_k: int = 5,
    rebalance_stride: int = 5,
    cost_bps: float = 30.0,
    strategy_name: str = "path_aware_ridge",
    risk_free_returns_by_horizon: Dict[int, np.ndarray] | None = None,
) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Rank by path-aware scores and realize returns at predicted exit horizons."""

    if scores.shape != exit_horizons.shape:
        raise ValueError("scores and exit_horizons must have the same shape")
    if not target_returns_by_horizon:
        raise ValueError("target_returns_by_horizon is empty")

    horizons = sorted(int(h) for h in target_returns_by_horizon)
    if risk_free_returns_by_horizon is not None:
        missing_horizons = sorted(set(horizons) - set(risk_free_returns_by_horizon))
        if missing_horizons:
            raise ValueError(f"missing risk-free returns for horizons: {missing_horizons}")
        risk_free_returns_by_horizon = {
            int(horizon): np.asarray(values, dtype=np.float64)
            for horizon, values in risk_free_returns_by_horizon.items()
        }
        if any(values.shape != (len(dates),) for values in risk_free_returns_by_horizon.values()):
            raise ValueError("risk-free return arrays must have one value per date")
    max_horizon = max(horizons)
    roundtrip_cost = cost_bps / 10_000.0
    rows: List[PathTradeRecord] = []
    selected_indices = list(test_indices)[::max(rebalance_stride, 1)]

    default_target = target_returns_by_horizon[max_horizon]
    for idx in selected_indices:
        score = scores[idx]
        chosen_h = exit_horizons[idx].astype(int)
        valid = np.isfinite(score)
        for h, returns in target_returns_by_horizon.items():
            valid &= (chosen_h != h) | np.isfinite(returns[idx])
            if risk_free_returns_by_horizon is not None:
                valid &= (chosen_h != h) | np.isfinite(risk_free_returns_by_horizon[h][idx])
        if valid.sum() < top_k:
            continue

        ranked = np.argsort(score[valid])[-top_k:][::-1]
        valid_indices = np.flatnonzero(valid)
        chosen = valid_indices[ranked]
        realized = []
        realized_risk_free = []
        used_horizons = []
        for node_idx in chosen.tolist():
            horizon = int(chosen_h[node_idx])
            if horizon not in target_returns_by_horizon:
                horizon = min(horizons, key=lambda h: abs(h - horizon))
            realized.append(float(target_returns_by_horizon[horizon][idx, node_idx]))
            if risk_free_returns_by_horizon is not None:
                realized_risk_free.append(float(risk_free_returns_by_horizon[horizon][idx]))
            used_horizons.append(horizon)
        period_return = float(np.mean(realized) - roundtrip_cost)
        risk_free_return = (
            float(np.mean(realized_risk_free))
            if risk_free_returns_by_horizon is not None
            else float("nan")
        )
        excess_period_return = (
            period_return - risk_free_return
            if risk_free_returns_by_horizon is not None
            else float("nan")
        )
        benchmark_valid = np.isfinite(default_target[idx])
        benchmark_return = float(default_target[idx, benchmark_valid].mean() - roundtrip_cost)
        benchmark_risk_free = (
            float(risk_free_returns_by_horizon[max_horizon][idx])
            if risk_free_returns_by_horizon is not None
            else float("nan")
        )
        if risk_free_returns_by_horizon is not None and not np.isfinite(benchmark_risk_free):
            continue
        benchmark_excess_return = (
            benchmark_return - benchmark_risk_free
            if risk_free_returns_by_horizon is not None
            else float("nan")
        )
        rows.append(
            PathTradeRecord(
                date=dates[idx],
                strategy=strategy_name,
                selected=[tickers[i] for i in chosen.tolist()],
                exit_horizons=used_horizons,
                period_return=period_return,
                benchmark_return=benchmark_return,
                risk_free_return=risk_free_return,
                excess_period_return=excess_period_return,
                benchmark_excess_return=benchmark_excess_return,
            )
        )

    frame = pd.DataFrame(
        [
            {
                "date": row.date,
                "strategy": row.strategy,
                "selected": ",".join(row.selected),
                "exit_horizons": ",".join(str(h) for h in row.exit_horizons),
                "avg_exit_horizon": float(np.mean(row.exit_horizons)) if row.exit_horizons else np.nan,
                "period_return": row.period_return,
                "benchmark_return": row.benchmark_return,
                "risk_free_return": row.risk_free_return,
                "excess_period_return": row.excess_period_return,
                "benchmark_excess_return": row.benchmark_excess_return,
            }
            for row in rows
        ]
    )
    avg_horizon = float(frame["avg_exit_horizon"].mean()) if not frame.empty else float(max_horizon)
    periods_per_year = 252.0 / max(avg_horizon, 1.0)
    benchmark_periods = 252.0 / max(max_horizon, 1)
    metrics = {
        strategy_name: performance_metrics(
            frame["period_return"].to_numpy(),
            periods_per_year,
            frame["risk_free_return"].to_numpy()
            if risk_free_returns_by_horizon is not None
            else None,
        )
        if not frame.empty
        else performance_metrics([], periods_per_year),
        "path_equal_weight_benchmark": performance_metrics(
            frame["benchmark_return"].to_numpy(),
            benchmark_periods,
            (
                frame["benchmark_return"].to_numpy()
                - frame["benchmark_excess_return"].to_numpy()
                if risk_free_returns_by_horizon is not None
                else None
            ),
        )
        if not frame.empty
        else performance_metrics([], benchmark_periods),
    }
    if not frame.empty:
        metrics[strategy_name]["avg_exit_horizon"] = avg_horizon
    return frame, metrics


def format_pct(value: float) -> str:
    return f"{value * 100.0:+.2f}%"
