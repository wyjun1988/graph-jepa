from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np


DEFAULT_MEMORY_ALPHAS = (0.35, 0.08)


@dataclass(frozen=True)
class ResidualMemoryResult:
    values: np.ndarray
    feature_names: tuple[str, ...]
    feature_scale: np.ndarray
    group_names: tuple[str, ...]
    diagnostics: dict[str, float | int]


def feature_group(name: str) -> str:
    value = str(name).lower()
    if value.startswith("ext_") or ":ext_" in value:
        return "external"
    if value.startswith("investor_") or "flow_ratio" in value:
        return "flow"
    if value.startswith("news_"):
        return "news"
    if value.startswith("fund_"):
        return "fundamental"
    if any(
        token in value
        for token in (
            "volume",
            "turnover",
            "traded_value",
            "value_ma",
            "amihud",
            "liquidity",
        )
    ):
        return "liquidity"
    if any(
        token in value
        for token in (
            "volatility",
            "downside",
            "range",
            "atr",
            "corr",
            "beta",
        )
    ):
        return "risk"
    if any(
        token in value
        for token in (
            "return",
            "price",
            "gap",
            "drawdown",
            "breakout",
            "momentum",
            "ma5",
            "ma20",
            "ma60",
            "ma120",
        )
    ):
        return "price"
    return "other"


def align_matured_residuals(
    forecasts: Mapping[int, np.ndarray],
    actual: np.ndarray,
    available: np.ndarray,
    source_available: np.ndarray,
    steps: Sequence[int],
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Align forecasts to the first row where their targets are observable."""

    actual = np.asarray(actual, dtype=np.float32)
    available = np.asarray(available, dtype=bool)
    source_available = np.asarray(source_available, dtype=bool)
    steps = np.asarray(steps, dtype=np.int64)
    if actual.ndim != 3:
        raise ValueError("actual values must be [time, node, feature]")
    if available.shape != actual.shape or source_available.shape != actual.shape:
        raise ValueError("availability masks must match actual values")
    if len(steps) != len(actual) or len(np.unique(steps)) != len(steps):
        raise ValueError("steps must be unique and aligned to actual values")
    if len(steps) > 1 and np.any(np.diff(steps) <= 0):
        raise ValueError("steps must be strictly increasing")
    horizons = tuple(sorted(int(value) for value in forecasts))
    if not horizons or horizons[0] < 1:
        raise ValueError("forecasts require positive horizons")
    for horizon in horizons:
        if np.asarray(forecasts[horizon]).shape != actual.shape:
            raise ValueError("every forecast tensor must match actual values")

    result = np.full(
        (len(actual), len(horizons), actual.shape[1], actual.shape[2]),
        np.nan,
        dtype=np.float32,
    )
    step_to_row = {int(step): index for index, step in enumerate(steps)}
    for target_row, target_step in enumerate(steps):
        for horizon_index, horizon in enumerate(horizons):
            source_row = step_to_row.get(int(target_step) - int(horizon))
            if source_row is None:
                continue
            prediction = np.asarray(forecasts[horizon][source_row], dtype=np.float32)
            valid = (
                available[target_row]
                & source_available[source_row]
                & np.isfinite(actual[target_row])
                & np.isfinite(prediction)
            )
            residual = actual[target_row] - prediction
            result[target_row, horizon_index][valid] = residual[valid]
    return result, horizons


def fit_feature_scale(
    matured_residuals: np.ndarray,
    fit_rows: Sequence[int],
    *,
    minimum_scale: float = 1e-3,
) -> np.ndarray:
    residuals = np.asarray(matured_residuals, dtype=np.float32)
    rows = np.asarray(fit_rows, dtype=np.int64)
    if residuals.ndim != 4:
        raise ValueError("matured residuals must be [time, horizon, node, feature]")
    if len(rows) == 0 or rows.min() < 0 or rows.max() >= len(residuals):
        raise ValueError("fit rows must index the residual time axis")
    values = residuals[rows].reshape(-1, residuals.shape[-1]).astype(np.float64)
    with np.errstate(invalid="ignore"):
        median = np.nanmedian(values, axis=0)
        mad = np.nanmedian(np.abs(values - median[None, :]), axis=0)
        q75 = np.nanquantile(np.abs(values), 0.75, axis=0)
    scale = np.maximum(mad / 0.6744897501960817, q75 / 1.1503493803760083)
    scale = np.where(np.isfinite(scale) & (scale >= minimum_scale), scale, 1.0)
    return scale.astype(np.float32)


def _summary(
    residual: np.ndarray,
    feature_groups: np.ndarray,
    group_names: Sequence[str],
    stock_count: int,
) -> np.ndarray:
    metric_count = 6
    output = np.full((2 + len(group_names), metric_count), np.nan, dtype=np.float32)
    node_slices = (
        ("stock_all", slice(0, int(stock_count))),
        ("external_all", slice(int(stock_count), residual.shape[0])),
    )

    def metrics(values: np.ndarray, expected: int) -> np.ndarray:
        finite = values[np.isfinite(values)]
        coverage = len(finite) / max(int(expected), 1)
        if len(finite) == 0:
            return np.asarray([0.0, 0.0, 0.0, 0.0, 0.0, coverage], dtype=np.float32)
        absolute = np.abs(finite)
        abs_mean = float(absolute.mean())
        return np.asarray(
            [
                float(finite.mean()),
                float(np.median(finite)),
                abs_mean,
                float(np.quantile(absolute, 0.90)),
                float(np.mean(finite > 0.0)),
                coverage,
            ],
            dtype=np.float32,
        )

    for row_index, (_, node_slice) in enumerate(node_slices):
        selected = residual[node_slice]
        output[row_index] = metrics(selected, selected.size)
    stock = residual[: int(stock_count)]
    for group_index, group_name in enumerate(group_names, start=2):
        selected_features = feature_groups == group_name
        selected = stock[:, selected_features]
        output[group_index] = metrics(selected, selected.size)
    return output.reshape(-1)


def build_causal_residual_memory(
    matured_residuals: np.ndarray,
    fit_rows: Sequence[int],
    feature_names: Sequence[str],
    *,
    stock_count: int,
    alphas: Sequence[float] = DEFAULT_MEMORY_ALPHAS,
    clip: float = 4.0,
    minimum_coverage: float = 0.20,
) -> ResidualMemoryResult:
    residuals = np.asarray(matured_residuals, dtype=np.float32)
    if residuals.ndim != 4:
        raise ValueError("matured residuals must be [time, horizon, node, feature]")
    if residuals.shape[-1] != len(feature_names):
        raise ValueError("feature names do not match residual width")
    if not 0 < int(stock_count) <= residuals.shape[2]:
        raise ValueError("stock_count must fit the residual node axis")
    normalized_alphas = tuple(float(alpha) for alpha in alphas)
    if not normalized_alphas or any(not 0.0 < alpha <= 1.0 for alpha in normalized_alphas):
        raise ValueError("memory alphas must be in (0, 1]")
    if clip <= 0.0 or not 0.0 <= minimum_coverage <= 1.0:
        raise ValueError("invalid residual memory clipping or coverage")

    scale = fit_feature_scale(residuals, fit_rows)
    feature_groups = np.asarray([feature_group(name) for name in feature_names])
    group_names = tuple(sorted(set(feature_groups.tolist())))
    summary_labels = ("stock_all", "external_all", *group_names)
    metric_names = (
        "signed_mean",
        "signed_median",
        "absolute_mean",
        "absolute_q90",
        "positive_fraction",
        "coverage",
    )
    summary_width = len(summary_labels) * len(metric_names)
    current = np.zeros((residuals.shape[1], summary_width), dtype=np.float32)
    memories = [np.zeros_like(current) for _ in normalized_alphas]
    output_width = residuals.shape[1] * summary_width * (1 + len(memories))
    output = np.zeros((len(residuals), output_width), dtype=np.float32)
    updates = 0

    feature_names_out = []
    for horizon_index in range(residuals.shape[1]):
        for state_name in ("current", *[f"ewma_{alpha:g}" for alpha in normalized_alphas]):
            for summary_label in summary_labels:
                for metric_name in metric_names:
                    feature_names_out.append(
                        f"residual_h{horizon_index}_{state_name}_{summary_label}_{metric_name}"
                    )

    coverage_positions = np.arange(len(summary_labels)) * len(metric_names) + (
        len(metric_names) - 1
    )
    for time_index in range(len(residuals)):
        for horizon_index in range(residuals.shape[1]):
            standardized = residuals[time_index, horizon_index] / scale[None, :]
            standardized = np.clip(standardized, -float(clip), float(clip))
            summary = _summary(
                standardized,
                feature_groups,
                group_names,
                int(stock_count),
            )
            coverage = summary[coverage_positions]
            valid_blocks = coverage >= float(minimum_coverage)
            gated = summary.copy()
            for block_index, valid in enumerate(valid_blocks):
                if not valid:
                    start = block_index * len(metric_names)
                    gated[start : start + len(metric_names)] = current[
                        horizon_index, start : start + len(metric_names)
                    ]
            if np.isfinite(gated).all():
                current[horizon_index] = gated
                for alpha, memory in zip(normalized_alphas, memories):
                    memory[horizon_index] += np.float32(alpha) * (
                        gated - memory[horizon_index]
                    )
                updates += 1
        parts = []
        for horizon_index in range(residuals.shape[1]):
            parts.append(current[horizon_index])
            parts.extend(memory[horizon_index] for memory in memories)
        output[time_index] = np.concatenate(parts)

    return ResidualMemoryResult(
        values=output,
        feature_names=tuple(feature_names_out),
        feature_scale=scale,
        group_names=group_names,
        diagnostics={
            "rows": int(len(residuals)),
            "horizons": int(residuals.shape[1]),
            "nodes": int(residuals.shape[2]),
            "features": int(residuals.shape[3]),
            "memory_features": int(output.shape[1]),
            "updates": int(updates),
            "finite_fraction": float(np.isfinite(output).mean()),
        },
    )


def shuffled_within_splits(
    values: np.ndarray,
    split_rows: Mapping[str, Sequence[int]],
    *,
    seed: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = values.copy()
    assigned = np.zeros(len(values), dtype=bool)
    generator = np.random.default_rng(int(seed))
    for rows in split_rows.values():
        index = np.asarray(rows, dtype=np.int64)
        if len(index) == 0 or index.min() < 0 or index.max() >= len(values):
            raise ValueError("shuffle rows must index the memory matrix")
        if assigned[index].any():
            raise ValueError("shuffle splits must not overlap")
        assigned[index] = True
        result[index] = values[generator.permutation(index)]
    return result
