from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stock_v2.ops.signals import world_model_state_scores


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    x = np.asarray(left[valid], dtype=np.float64)
    y = np.asarray(right[valid], dtype=np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def newey_west_mean(values: Iterable[float], max_lag: int) -> dict[str, float | int]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 3:
        return {
            "rows": int(array.size),
            "mean": float("nan"),
            "newey_west_lag": 0,
            "newey_west_standard_error": float("nan"),
            "newey_west_t_stat": float("nan"),
            "positive_day_fraction": float("nan"),
        }
    centered = array - array.mean()
    long_variance = float(centered @ centered / array.size)
    lag_count = min(max(0, int(max_lag)), array.size - 1)
    for lag in range(1, lag_count + 1):
        weight = 1.0 - lag / (lag_count + 1.0)
        covariance = float(centered[lag:] @ centered[:-lag] / array.size)
        long_variance += 2.0 * weight * covariance
    standard_error = float(np.sqrt(max(long_variance, 0.0) / array.size))
    mean = float(array.mean())
    return {
        "rows": int(array.size),
        "mean": mean,
        "newey_west_lag": int(lag_count),
        "newey_west_standard_error": standard_error,
        "newey_west_t_stat": (
            float(mean / standard_error) if standard_error > 1e-12 else float("nan")
        ),
        "positive_day_fraction": float((array > 0.0).mean()),
    }


def cross_sectional_zscore(values: pd.Series) -> pd.Series:
    grouped = values.groupby(level="date")
    mean = grouped.transform("mean")
    std = grouped.transform(lambda item: item.std(ddof=0)).replace(0.0, np.nan)
    return (values - mean) / std


def daily_correlations(
    frame: pd.DataFrame,
    score_column: str,
    target_column: str,
    dates: set[pd.Timestamp],
) -> list[float]:
    selected = frame[frame.index.get_level_values("date").isin(dates)]
    result: list[float] = []
    for _date, group in selected.groupby(level="date", sort=True):
        result.append(
            pearson(
                group[score_column].to_numpy(dtype=np.float64),
                group[target_column].to_numpy(dtype=np.float64),
            )
        )
    return result


def learn_positive_ic_weights(
    frame: pd.DataFrame,
    prediction_columns: list[str],
    target_column: str,
    calibration_dates: set[pd.Timestamp],
) -> tuple[np.ndarray, dict[str, float]]:
    mean_ic = {
        column: float(
            np.nanmean(
                daily_correlations(
                    frame,
                    column,
                    target_column,
                    calibration_dates,
                )
            )
        )
        for column in prediction_columns
    }
    weights = np.asarray([max(mean_ic[column], 0.0) for column in prediction_columns])
    if not np.isfinite(weights).all() or weights.sum() <= 1e-12:
        weights = np.ones(len(prediction_columns), dtype=np.float64)
    weights /= weights.sum()
    return weights, mean_ic


def score_diagnostics(
    frame: pd.DataFrame,
    score_column: str,
    target_column: str,
    evaluation_dates: set[pd.Timestamp],
    horizon: int,
) -> dict[str, float | int]:
    selected = frame[frame.index.get_level_values("date").isin(evaluation_dates)]
    correlations: list[float] = []
    spreads: list[float] = []
    observed = 0
    for _date, group in selected.groupby(level="date", sort=True):
        score = group[score_column].to_numpy(dtype=np.float64)
        target = group[target_column].to_numpy(dtype=np.float64)
        valid = np.isfinite(score) & np.isfinite(target)
        if valid.sum() < 10:
            continue
        score = score[valid]
        target = target[valid]
        correlations.append(pearson(score, target))
        tail = max(1, int(valid.sum() // 10))
        order = np.argsort(score, kind="stable")
        spreads.append(float(target[order[-tail:]].mean() - target[order[:tail]].mean()))
        observed += int(valid.sum())
    significance = newey_west_mean(correlations, max_lag=horizon)
    finite_correlations = np.asarray(correlations, dtype=np.float64)
    finite_correlations = finite_correlations[np.isfinite(finite_correlations)]
    subperiod_split = len(finite_correlations) // 2
    significance.update(
        {
            "observed_stock_dates": int(observed),
            "first_half_mean_ic": (
                float(finite_correlations[:subperiod_split].mean())
                if subperiod_split > 0
                else float("nan")
            ),
            "second_half_mean_ic": (
                float(finite_correlations[subperiod_split:].mean())
                if subperiod_split < len(finite_correlations)
                else float("nan")
            ),
            "mean_top_minus_bottom_decile_return": (
                float(np.nanmean(spreads)) if spreads else float("nan")
            ),
        }
    )
    return significance


def build_wide_forecasts(path: str | Path) -> tuple[pd.DataFrame, list[int]]:
    raw = pd.read_csv(path, parse_dates=["date", "target_date"])
    required = {
        "date",
        "ticker",
        "horizon",
        "prediction_return_1d",
        "realized_path_return",
        "current_value_ma20_log",
    }
    missing = sorted(required - set(raw.columns))
    if missing:
        raise ValueError(f"forecast file is missing columns: {missing}")
    if raw.duplicated(["date", "ticker", "horizon"]).any():
        raise ValueError("forecast rows must be unique by date, ticker, and horizon")

    horizons = sorted(int(value) for value in raw["horizon"].unique())
    index = ["date", "ticker"]
    predictions = raw.pivot(index=index, columns="horizon", values="prediction_return_1d")
    path_returns = raw.pivot(index=index, columns="horizon", values="realized_path_return")
    predictions.columns = [f"prediction_h{int(column)}" for column in predictions.columns]
    path_returns.columns = [f"path_return_h{int(column)}" for column in path_returns.columns]
    liquidity = raw.groupby(index, sort=False)["current_value_ma20_log"].first()
    wide = predictions.join(path_returns, how="inner").join(liquidity.rename("liquidity"))
    optional_predictions = (
        "prediction_return_5d",
        "prediction_cs_rank_return_20d",
        "prediction_volatility_20d",
        "prediction_downside_volatility_20d",
    )
    for source_column in optional_predictions:
        if source_column not in raw.columns:
            continue
        values = raw.pivot(index=index, columns="horizon", values=source_column)
        values.columns = [
            f"{source_column}_h{int(column)}" for column in values.columns
        ]
        wide = wide.join(values, how="left")
    wide.index = wide.index.set_names(["date", "ticker"])
    for horizon in horizons:
        column = f"prediction_h{horizon}"
        wide[f"z_{column}"] = cross_sectional_zscore(wide[column])
    return wide.sort_index(), horizons


def normalize(values: Iterable[float]) -> np.ndarray:
    array = np.asarray(list(values), dtype=np.float64)
    if array.sum() <= 0.0:
        raise ValueError("weights must have a positive sum")
    return array / array.sum()


def production_shadow_scores(
    frame: pd.DataFrame,
    horizons: list[int],
) -> tuple[np.ndarray, np.ndarray] | None:
    feature_sources = {
        "return_1d": "prediction",
        "return_5d": "prediction_return_5d",
        "cs_rank_return_20d": "prediction_cs_rank_return_20d",
        "volatility_20d": "prediction_volatility_20d",
        "downside_volatility_20d": "prediction_downside_volatility_20d",
    }
    required_columns = [
        f"{source}_h{horizon}"
        for source in feature_sources.values()
        for horizon in horizons
    ]
    if any(column not in frame.columns for column in required_columns):
        return None
    feature_names = list(feature_sources)
    forecasts = {
        horizon: np.column_stack(
            [
                frame[f"{source}_h{horizon}"].to_numpy(dtype=np.float64)
                for source in feature_sources.values()
            ]
        )
        for horizon in horizons
    }
    score, diagnostics = world_model_state_scores(
        forecasts,
        feature_names,
        train_mean=np.zeros(len(feature_names), dtype=np.float64),
        train_std=np.ones(len(feature_names), dtype=np.float64),
    )
    return score.astype(np.float64), diagnostics["expected_return_1d"].astype(np.float64)


def calibrate(
    forecast_path: str | Path,
    liquidity_top: int = 300,
) -> dict[str, object]:
    frame, horizons = build_wide_forecasts(forecast_path)
    dates = sorted(pd.Timestamp(value) for value in frame.index.get_level_values("date").unique())
    split = len(dates) // 2
    if split < 20 or len(dates) - split < 20:
        raise ValueError("at least 40 distinct dates are required for calibration")
    calibration_dates = set(dates[:split])
    evaluation_dates = set(dates[split:])
    prediction_columns = [f"z_prediction_h{horizon}" for horizon in horizons]

    liquidity_rank = frame["liquidity"].groupby(level="date").rank(
        method="first",
        ascending=False,
    )
    slices = {"all": frame}
    if liquidity_top > 0:
        slices[f"liquidity_top{liquidity_top}"] = frame[liquidity_rank <= liquidity_top]

    ops_weights = normalize(min(float(horizon), 5.0) for horizon in horizons)
    segment_widths = np.diff([0] + horizons).astype(np.float64)
    integral_weights = normalize(segment_widths)
    raw_prediction_columns = [f"prediction_h{horizon}" for horizon in horizons]
    frame["legacy_ops_return_raw"] = (
        frame[raw_prediction_columns].to_numpy(dtype=np.float64) @ ops_weights
    )
    production_scores = production_shadow_scores(frame, horizons)
    legacy_full_available = production_scores is not None
    if production_scores is not None:
        full_score, return_component = production_scores
        frame["legacy_ops_full_raw"] = full_score
        frame["legacy_ops_return_raw"] = return_component
    result: dict[str, object] = {
        "forecast_path": str(forecast_path),
        "prediction_horizons": horizons,
        "calibration_start": str(dates[0].date()),
        "calibration_end": str(dates[split - 1].date()),
        "evaluation_start": str(dates[split].date()),
        "evaluation_end": str(dates[-1].date()),
        "calibration_dates": split,
        "evaluation_dates": len(dates) - split,
        "targets": {},
    }

    for target_horizon in horizons:
        target_column = f"path_return_h{target_horizon}"
        learned_weights, train_ic = learn_positive_ic_weights(
            frame,
            prediction_columns,
            target_column,
            calibration_dates,
        )
        same_horizon_weights = np.zeros(len(horizons), dtype=np.float64)
        same_horizon_weights[horizons.index(target_horizon)] = 1.0
        methods = {
            "same_horizon": same_horizon_weights,
            "legacy_ops_return_component": ops_weights,
            "trajectory_integral": integral_weights,
            "calibration_ic": learned_weights,
        }
        fixed_scores = {"legacy_ops_return_raw": "legacy_ops_return_raw"}
        if legacy_full_available:
            fixed_scores["legacy_ops_full_raw"] = "legacy_ops_full_raw"
        target_result: dict[str, object] = {
            "calibration_mean_ic_by_prediction_horizon": {
                str(horizon): train_ic[column]
                for horizon, column in zip(horizons, prediction_columns)
            },
            "calibration_ic_weights": {
                str(horizon): float(weight)
                for horizon, weight in zip(horizons, learned_weights)
            },
            "methods": {},
        }
        for method, weights in methods.items():
            score_column = f"score_{target_horizon}_{method}"
            frame[score_column] = frame[prediction_columns].to_numpy(dtype=np.float64) @ weights
            target_result["methods"][method] = {
                "weights": {
                    str(horizon): float(weight)
                    for horizon, weight in zip(horizons, weights)
                },
                "slices": {
                    slice_name: score_diagnostics(
                        slice_frame.join(frame[[score_column]], how="left")
                        if score_column not in slice_frame.columns
                        else slice_frame,
                        score_column,
                        target_column,
                        evaluation_dates,
                        target_horizon,
                    )
                    for slice_name, slice_frame in slices.items()
                },
            }
        for method, score_column in fixed_scores.items():
            target_result["methods"][method] = {
                "weights": None,
                "slices": {
                    slice_name: score_diagnostics(
                        slice_frame.join(frame[[score_column]], how="left")
                        if score_column not in slice_frame.columns
                        else slice_frame,
                        score_column,
                        target_column,
                        evaluation_dates,
                        target_horizon,
                    )
                    for slice_name, slice_frame in slices.items()
                },
            }
        result["targets"][str(target_horizon)] = target_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and evaluate a time-split multi-horizon shadow-score calibration."
    )
    parser.add_argument("--forecasts", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--liquidity-top", type=int, default=300)
    args = parser.parse_args()

    result = calibrate(args.forecasts, liquidity_top=args.liquidity_top)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
