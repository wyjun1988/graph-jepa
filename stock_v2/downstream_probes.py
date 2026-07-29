from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch import nn


CONTINUOUS_TASKS = (
    "path_return",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "realized_volatility",
    "future_liquidity",
)


@dataclass(frozen=True)
class CausalProbeSplits:
    fit_steps: np.ndarray
    validation_steps: np.ndarray
    test_steps: np.ndarray


@dataclass(frozen=True)
class DownstreamTargets:
    continuous: np.ndarray
    continuous_raw: np.ndarray
    continuous_valid: np.ndarray
    direction: np.ndarray
    direction_valid: np.ndarray


def causal_probe_splits(
    dates: pd.DatetimeIndex,
    train_end: str,
    edge_window: int,
    max_horizon: int,
    validation_days: int,
    max_test_steps: int = 0,
    test_end: str | None = None,
) -> CausalProbeSplits:
    cutoff = pd.Timestamp(train_end)
    all_steps = np.arange(len(dates), dtype=np.int64)
    last_target_step = len(dates) - 1
    eligible = all_steps[
        (all_steps >= int(edge_window))
        & (all_steps + int(max_horizon) <= last_target_step)
    ]
    train_date_steps = all_steps[dates <= cutoff]
    if not len(train_date_steps):
        raise ValueError("training cutoff is before the feature panel")
    last_train_step = int(train_date_steps[-1])
    train_steps = eligible[eligible + int(max_horizon) <= last_train_step]
    test_steps = eligible[dates[eligible] > cutoff]
    if test_end is not None:
        test_steps = test_steps[dates[test_steps] <= pd.Timestamp(test_end)]
    if len(train_steps) <= int(validation_days) + int(max_horizon):
        raise ValueError("not enough training dates for a leakage-safe validation split")
    validation_steps = train_steps[-int(validation_days) :]
    fit_steps = train_steps[
        train_steps + int(max_horizon) < int(validation_steps[0])
    ]
    if len(fit_steps) < 260:
        raise ValueError("fit split must contain at least 260 trading dates")
    if max_test_steps > 0 and len(test_steps) > int(max_test_steps):
        positions = np.linspace(0, len(test_steps) - 1, int(max_test_steps))
        test_steps = test_steps[np.rint(positions).astype(np.int64)]
    if not len(test_steps):
        raise ValueError("test split is empty")
    if int(fit_steps[-1]) + int(max_horizon) >= int(validation_steps[0]):
        raise AssertionError("fit targets overlap the validation context period")
    if dates[int(validation_steps[-1]) + int(max_horizon)] > cutoff:
        raise AssertionError("validation targets cross the frozen training cutoff")
    return CausalProbeSplits(fit_steps, validation_steps, test_steps)


def _cross_sectional_zscore(
    values: np.ndarray,
    valid: np.ndarray,
) -> np.ndarray:
    if values.shape != valid.shape:
        raise ValueError("values and validity mask must have the same shape")
    result = np.full(values.shape, np.nan, dtype=np.float32)
    for date_index in range(values.shape[0]):
        for task_index in range(values.shape[2]):
            selected = valid[date_index, :, task_index]
            selected &= np.isfinite(values[date_index, :, task_index])
            if selected.sum() < 3:
                continue
            sample = values[date_index, selected, task_index].astype(np.float64)
            scale = float(sample.std())
            if scale < 1e-12:
                continue
            result[date_index, selected, task_index] = (
                (sample - sample.mean()) / scale
            ).astype(np.float32)
    return result


def build_downstream_targets(
    features,
    steps: Sequence[int],
    horizon: int,
) -> DownstreamTargets:
    selected_steps = np.asarray(steps, dtype=np.int64)
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("horizon must be positive")
    if selected_steps.size and int(selected_steps.max()) + horizon >= len(features.dates):
        raise ValueError("a target horizon extends beyond the feature panel")

    stock_count = int(features.tradable_count)
    date_count = len(selected_steps)
    raw = np.full(
        (date_count, stock_count, len(CONTINUOUS_TASKS)),
        np.nan,
        dtype=np.float32,
    )
    valid = np.zeros(raw.shape, dtype=bool)
    liquidity_index = features.feature_names.index("value_ma20_log")

    for position, step in enumerate(selected_steps):
        step = int(step)
        entry = features.open[step + 1, :stock_count].astype(np.float64)
        close_path = features.close[
            step + 1 : step + horizon + 1, :stock_count
        ].astype(np.float64)
        price_valid = (
            np.isfinite(entry)
            & (entry > 0.0)
            & np.isfinite(close_path).all(axis=0)
        )
        path_returns = np.divide(
            close_path,
            entry[None, :],
            out=np.full_like(close_path, np.nan),
            where=np.isfinite(entry[None, :]) & (entry[None, :] > 0.0),
        ) - 1.0
        raw[position, :, 0] = path_returns[-1].astype(np.float32)
        raw[position, :, 1] = np.nanmax(path_returns, axis=0).astype(np.float32)
        raw[position, :, 2] = np.nanmin(path_returns, axis=0).astype(np.float32)
        future_returns = features.returns_1d[
            step + 1 : step + horizon + 1, :stock_count
        ].astype(np.float64)
        returns_valid = np.isfinite(future_returns).all(axis=0)
        raw[position, :, 3] = np.sqrt(
            np.nanmean(np.square(future_returns), axis=0)
        ).astype(np.float32)
        liquidity = features.raw_features[
            step + horizon, :stock_count, liquidity_index
        ].astype(np.float32)
        raw[position, :, 4] = liquidity
        valid[position, :, :3] = price_valid[:, None]
        valid[position, :, 3] = returns_valid
        valid[position, :, 4] = np.isfinite(liquidity)

    standardized = _cross_sectional_zscore(raw, valid)
    valid &= np.isfinite(standardized)
    direction = (raw[:, :, 0] > 0.0).astype(np.float32)
    direction_valid = valid[:, :, 0].copy()
    return DownstreamTargets(
        continuous=standardized.reshape(-1, len(CONTINUOUS_TASKS)),
        continuous_raw=raw.reshape(-1, len(CONTINUOUS_TASKS)),
        continuous_valid=valid.reshape(-1, len(CONTINUOUS_TASKS)),
        direction=direction.reshape(-1),
        direction_valid=direction_valid.reshape(-1),
    )


class FrozenEncoderProbe(nn.Module):
    def __init__(
        self,
        input_dim: int,
        task_count: int = len(CONTINUOUS_TASKS),
        hidden_dim: int = 256,
        layers: int = 2,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or layers < 1:
            raise ValueError("probe dimensions and layers must be positive")
        blocks: list[nn.Module] = [nn.LayerNorm(int(input_dim))]
        width = int(input_dim)
        for _ in range(int(layers)):
            blocks.extend(
                [
                    nn.Linear(width, int(hidden_dim)),
                    nn.SiLU(),
                    nn.LayerNorm(int(hidden_dim)),
                    nn.Dropout(float(dropout)),
                ]
            )
            width = int(hidden_dim)
        self.trunk = nn.Sequential(*blocks)
        self.continuous_head = nn.Linear(width, int(task_count))
        self.direction_head = nn.Linear(width, 1)

    def forward(self, values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.trunk(values)
        return self.continuous_head(hidden), self.direction_head(hidden).squeeze(-1)


def masked_probe_loss(
    continuous_prediction: torch.Tensor,
    direction_logit: torch.Tensor,
    continuous_target: torch.Tensor,
    continuous_valid: torch.Tensor,
    direction_target: torch.Tensor,
    direction_valid: torch.Tensor,
    task_indices: Sequence[int],
    direction_weight: float = 0.25,
) -> torch.Tensor:
    indices = torch.as_tensor(task_indices, dtype=torch.long, device=continuous_prediction.device)
    prediction = continuous_prediction.index_select(1, indices)
    target = continuous_target.index_select(1, indices)
    valid = continuous_valid.index_select(1, indices)
    denominator = valid.sum().clamp_min(1)
    error = torch.where(valid, prediction - target, torch.zeros_like(prediction))
    regression = error.square().sum() / denominator
    selected_direction = direction_valid > 0.5
    if selected_direction.any() and float(direction_weight) > 0.0:
        classification = nn.functional.binary_cross_entropy_with_logits(
            direction_logit[selected_direction],
            direction_target[selected_direction],
        )
        return regression + float(direction_weight) * classification
    return regression


def pearson(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if valid.sum() < 3:
        return float("nan")
    x = left[valid].astype(np.float64)
    y = right[valid].astype(np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def newey_west_mean(values: Sequence[float], lag: int) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if len(array) < 3:
        return {"rows": int(len(array)), "mean": float("nan"), "newey_west_t": float("nan")}
    centered = array - array.mean()
    long_variance = float(centered @ centered / len(array))
    max_lag = min(max(0, int(lag)), len(array) - 1)
    for offset in range(1, max_lag + 1):
        weight = 1.0 - offset / (max_lag + 1.0)
        covariance = float(centered[offset:] @ centered[:-offset] / len(array))
        long_variance += 2.0 * weight * covariance
    standard_error = float(np.sqrt(max(long_variance, 0.0) / len(array)))
    mean = float(array.mean())
    return {
        "rows": int(len(array)),
        "mean": mean,
        "newey_west_lag": int(max_lag),
        "newey_west_standard_error": standard_error,
        "newey_west_t": float(mean / standard_error) if standard_error > 1e-12 else float("nan"),
        "positive_fraction": float((array > 0.0).mean()),
    }


def evaluate_probe_predictions(
    continuous_prediction: np.ndarray,
    direction_logit: np.ndarray,
    targets: DownstreamTargets,
    date_count: int,
    stock_count: int,
    horizon: int,
) -> dict[str, object]:
    expected_rows = int(date_count) * int(stock_count)
    if continuous_prediction.shape != (expected_rows, len(CONTINUOUS_TASKS)):
        raise ValueError("continuous prediction shape does not match the test panel")
    daily_ic: dict[str, list[float]] = {name: [] for name in CONTINUOUS_TASKS}
    squared_error = np.zeros(len(CONTINUOUS_TASKS), dtype=np.float64)
    target_squares = np.zeros(len(CONTINUOUS_TASKS), dtype=np.float64)
    observed = np.zeros(len(CONTINUOUS_TASKS), dtype=np.int64)
    for date_index in range(int(date_count)):
        start = date_index * int(stock_count)
        end = start + int(stock_count)
        for task_index, name in enumerate(CONTINUOUS_TASKS):
            valid = targets.continuous_valid[start:end, task_index]
            prediction = continuous_prediction[start:end, task_index]
            target = targets.continuous[start:end, task_index]
            valid &= np.isfinite(prediction) & np.isfinite(target)
            daily_ic[name].append(pearson(prediction[valid], target[valid]))
            if valid.any():
                error = prediction[valid] - target[valid]
                squared_error[task_index] += float(error @ error)
                target_squares[task_index] += float(target[valid] @ target[valid])
                observed[task_index] += int(valid.sum())

    probability = 1.0 / (1.0 + np.exp(-np.clip(direction_logit, -30.0, 30.0)))
    direction_valid = targets.direction_valid & np.isfinite(probability)
    direction_target = targets.direction[direction_valid]
    direction_probability = probability[direction_valid]
    return {
        "tasks": {
            name: {
                "daily_ic": newey_west_mean(daily_ic[name], lag=int(horizon)),
                "daily_ic_values": [
                    float(value) if np.isfinite(value) else None
                    for value in daily_ic[name]
                ],
                "observed": int(observed[index]),
                "mse": (
                    float(squared_error[index] / observed[index])
                    if observed[index]
                    else float("nan")
                ),
                "skill_vs_cross_sectional_zero": (
                    float(1.0 - squared_error[index] / target_squares[index])
                    if target_squares[index] > 1e-12
                    else float("nan")
                ),
            }
            for index, name in enumerate(CONTINUOUS_TASKS)
        },
        "direction": {
            "observed": int(direction_valid.sum()),
            "accuracy": (
                float(((direction_probability >= 0.5) == (direction_target >= 0.5)).mean())
                if len(direction_target)
                else float("nan")
            ),
            "brier": (
                float(np.mean(np.square(direction_probability - direction_target)))
                if len(direction_target)
                else float("nan")
            ),
            "base_rate": (
                float(direction_target.mean()) if len(direction_target) else float("nan")
            ),
        },
    }
