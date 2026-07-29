from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence
import warnings

import numpy as np


OPEN_KNOWN_EXACT = frozenset(("gap_open",))
OPEN_KNOWN_PREFIXES = ("news_", "fund_", "investor_", "ext_")
DEFAULT_STOCK_STATISTICS = ("mean", "std", "coverage")
OPEN_GAP_STATISTICS = (
    "mean",
    "std",
    "q10",
    "q25",
    "median",
    "q75",
    "q90",
    "positive_fraction",
    "negative_fraction",
    "coverage",
)


@dataclass(frozen=True)
class OpenDesign:
    values: np.ndarray
    feature_names: tuple[str, ...]
    current_feature_names: tuple[str, ...]


@dataclass(frozen=True)
class ResidualPCADesign:
    values: np.ndarray
    feature_names: tuple[str, ...]
    contract: Mapping[str, Any]


def open_known_feature_indices(feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, name in enumerate(feature_names)
            if str(name) in OPEN_KNOWN_EXACT
            or str(name).startswith(OPEN_KNOWN_PREFIXES)
        ],
        dtype=np.int64,
    )


def _validate_steps(steps: Sequence[int], total_steps: int) -> np.ndarray:
    result = np.asarray(steps, dtype=np.int64)
    if result.ndim != 1 or not len(result):
        raise ValueError("open design requires at least one origin step")
    if result.min() < 0 or result.max() + 1 >= int(total_steps):
        raise ValueError("every open origin requires a following trading day")
    if len(np.unique(result)) != len(result):
        raise ValueError("open origin steps must be unique")
    return result


def _masked_stock_moments(
    values: np.ndarray,
    available: np.ndarray,
    indices: Sequence[int],
    feature_names: Sequence[str],
    *,
    prefix: str,
    statistics: Sequence[str] = DEFAULT_STOCK_STATISTICS,
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    indices = np.asarray(indices, dtype=np.int64)
    if values.shape != available.shape or values.ndim != 3:
        raise ValueError("stock moments require aligned [time, stock, feature] arrays")
    if values.shape[-1] != len(feature_names):
        raise ValueError("stock moment feature names do not match the value width")
    if indices.ndim != 1 or (
        len(indices) and (indices.min() < 0 or indices.max() >= values.shape[-1])
    ):
        raise ValueError("stock moment indices exceed the feature axis")
    selected = values[:, :, indices]
    mask = available[:, :, indices] & np.isfinite(selected)
    count = mask.sum(axis=1).astype(np.float64)
    total = np.where(mask, selected, 0.0).sum(axis=1)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0.0)
    centered = np.where(mask, selected - mean[:, None, :], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=1),
        count,
        out=np.zeros_like(total),
        where=count > 0.0,
    )
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    coverage = count / max(values.shape[1], 1)
    normalized_statistics = tuple(str(value) for value in statistics)
    available_statistics = {
        "mean": mean,
        "std": standard_deviation,
        "coverage": coverage,
    }
    quantile_levels = {
        "q10": 0.10,
        "q25": 0.25,
        "median": 0.50,
        "q75": 0.75,
        "q90": 0.90,
    }
    requested_quantiles = [
        value for value in normalized_statistics if value in quantile_levels
    ]
    if requested_quantiles:
        masked = np.where(mask, selected, np.nan)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            for value in requested_quantiles:
                available_statistics[value] = np.nanquantile(
                    masked, quantile_levels[value], axis=1
                )
    if "positive_fraction" in normalized_statistics:
        available_statistics["positive_fraction"] = np.divide(
            (mask & (selected > 0.0)).sum(axis=1),
            count,
            out=np.zeros_like(count),
            where=count > 0.0,
        )
    if "negative_fraction" in normalized_statistics:
        available_statistics["negative_fraction"] = np.divide(
            (mask & (selected < 0.0)).sum(axis=1),
            count,
            out=np.zeros_like(count),
            where=count > 0.0,
        )
    supported = {
        "mean",
        "std",
        "coverage",
        *quantile_levels,
        "positive_fraction",
        "negative_fraction",
    }
    unknown = [value for value in normalized_statistics if value not in supported]
    if not normalized_statistics or unknown:
        raise ValueError(f"unknown stock moment statistics: {unknown}")
    output = np.concatenate(
        [
            np.where(np.isfinite(available_statistics[value]), available_statistics[value], 0.0)
            for value in normalized_statistics
        ],
        axis=1,
    )
    selected_names = [str(feature_names[int(index)]) for index in indices]
    names = [
        f"{prefix}_{statistic}:{name}"
        for statistic in normalized_statistics
        for name in selected_names
    ]
    return output.astype(np.float32), names


def _external_owned_values(
    values: np.ndarray,
    available: np.ndarray,
    feature_names: Sequence[str],
    node_tickers: Sequence[str],
    *,
    stock_count: int,
    prefix: str,
) -> tuple[np.ndarray, list[str]]:
    values = np.asarray(values, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    if values.shape != available.shape or values.ndim != 3:
        raise ValueError("external values require aligned [time, node, feature] arrays")
    if values.shape[-1] != len(feature_names):
        raise ValueError("external feature names do not match the value width")
    if not 0 < int(stock_count) <= values.shape[1]:
        raise ValueError("stock count must fit the node axis")
    columns: list[np.ndarray] = []
    names: list[str] = []
    for node_index in range(int(stock_count), values.shape[1]):
        node_id = (
            str(node_tickers[node_index])
            if node_index < len(node_tickers)
            else f"EXT:{node_index - int(stock_count)}"
        )
        factor = node_id.split(":", 1)[-1]
        owned_prefix = f"ext_{factor}_"
        for feature_index, feature_name in enumerate(feature_names):
            if not str(feature_name).startswith(owned_prefix):
                continue
            observed = available[:, node_index, feature_index]
            column = values[:, node_index, feature_index]
            valid = observed & np.isfinite(column)
            columns.extend((np.where(valid, column, 0.0), valid.astype(np.float64)))
            names.extend(
                (
                    f"{prefix}_value:{node_id}:{feature_name}",
                    f"{prefix}_available:{node_id}:{feature_name}",
                )
            )
    if not columns:
        return np.empty((len(values), 0), dtype=np.float32), []
    return np.stack(columns, axis=1).astype(np.float32), names


def build_open_sensor_design(features, origin_steps: Sequence[int]) -> OpenDesign:
    steps = _validate_steps(origin_steps, len(features.dates))
    current_steps = steps + 1
    stock_count = int(features.tradable_count)
    all_indices = np.arange(len(features.feature_names), dtype=np.int64)
    known_indices = open_known_feature_indices(features.feature_names)
    node_tickers = list(features.node_tickers or ())

    previous_stock, previous_stock_names = _masked_stock_moments(
        features.features[steps, :stock_count],
        features.available_mask[steps, :stock_count] > 0.5,
        all_indices,
        features.feature_names,
        prefix="previous_stock",
    )
    current_stock, current_stock_names = _masked_stock_moments(
        features.features[current_steps, :stock_count],
        features.available_mask[current_steps, :stock_count] > 0.5,
        known_indices,
        features.feature_names,
        prefix="open_stock",
    )
    gap_index = features.feature_names.index("gap_open")
    raw_gap, raw_gap_names = _masked_stock_moments(
        features.raw_features[current_steps, :stock_count],
        features.available_mask[current_steps, :stock_count] > 0.5,
        [gap_index],
        features.feature_names,
        prefix="open_gap_raw",
        statistics=OPEN_GAP_STATISTICS,
    )
    previous_external, previous_external_names = _external_owned_values(
        features.features[steps],
        features.available_mask[steps] > 0.5,
        features.feature_names,
        node_tickers,
        stock_count=stock_count,
        prefix="previous_external",
    )
    current_external, current_external_names = _external_owned_values(
        features.features[current_steps],
        features.available_mask[current_steps] > 0.5,
        features.feature_names,
        node_tickers,
        stock_count=stock_count,
        prefix="open_external",
    )
    values = np.concatenate(
        (
            previous_stock,
            previous_external,
            current_stock,
            raw_gap,
            current_external,
        ),
        axis=1,
    )
    names = (
        *previous_stock_names,
        *previous_external_names,
        *current_stock_names,
        *raw_gap_names,
        *current_external_names,
    )
    current_names = tuple(str(features.feature_names[int(index)]) for index in known_indices)
    if any(
        name not in OPEN_KNOWN_EXACT
        and not name.startswith(OPEN_KNOWN_PREFIXES)
        for name in current_names
    ):
        raise AssertionError("non-causal current feature entered the open sensor design")
    return OpenDesign(values, tuple(names), current_names)


def build_jepa_open_innovation_design(
    features,
    origin_steps: Sequence[int],
    predicted_state: np.ndarray,
    eligible_indices: Sequence[int],
) -> OpenDesign:
    steps = _validate_steps(origin_steps, len(features.dates))
    current_steps = steps + 1
    eligible = np.asarray(eligible_indices, dtype=np.int64)
    prediction = np.asarray(predicted_state, dtype=np.float32)
    expected = (len(steps), int(features.node_count), len(eligible))
    if prediction.shape != expected:
        raise ValueError(f"JEPA state prediction shape {prediction.shape} != {expected}")
    if len(eligible) != len(np.unique(eligible)) or (
        len(eligible)
        and (eligible.min() < 0 or eligible.max() >= len(features.feature_names))
    ):
        raise ValueError("JEPA eligible feature indices are invalid")
    eligible_names = [str(features.feature_names[int(index)]) for index in eligible]
    stock_count = int(features.tradable_count)
    node_tickers = list(features.node_tickers or ())
    previous = features.features[steps][:, :, eligible].astype(np.float32)
    previous_available = features.available_mask[steps][:, :, eligible] > 0.5
    forecast_available = previous_available & np.isfinite(prediction)

    forecast_stock, forecast_stock_names = _masked_stock_moments(
        prediction[:, :stock_count],
        forecast_available[:, :stock_count],
        np.arange(len(eligible), dtype=np.int64),
        eligible_names,
        prefix="jepa_forecast_stock",
    )
    delta = prediction - previous
    delta_stock, delta_stock_names = _masked_stock_moments(
        delta[:, :stock_count],
        forecast_available[:, :stock_count] & previous_available[:, :stock_count],
        np.arange(len(eligible), dtype=np.int64),
        eligible_names,
        prefix="jepa_delta_stock",
    )

    known_global = set(open_known_feature_indices(features.feature_names).tolist())
    overlap_positions = np.asarray(
        [position for position, index in enumerate(eligible) if int(index) in known_global],
        dtype=np.int64,
    )
    current = features.features[current_steps][:, :, eligible].astype(np.float32)
    current_available = features.available_mask[current_steps][:, :, eligible] > 0.5
    innovation = current - prediction
    innovation_available = current_available & forecast_available
    innovation_stock, innovation_stock_names = _masked_stock_moments(
        innovation[:, :stock_count],
        innovation_available[:, :stock_count],
        overlap_positions,
        eligible_names,
        prefix="open_innovation_stock",
    )

    external_blocks = []
    external_names: list[str] = []
    for block_values, block_available, block_prefix in (
        (prediction, forecast_available, "jepa_forecast_external"),
        (
            delta,
            forecast_available & previous_available,
            "jepa_delta_external",
        ),
        (innovation, innovation_available, "open_innovation_external"),
    ):
        block, names = _external_owned_values(
            block_values,
            block_available,
            eligible_names,
            node_tickers,
            stock_count=stock_count,
            prefix=block_prefix,
        )
        external_blocks.append(block)
        external_names.extend(names)

    values = np.concatenate(
        (forecast_stock, delta_stock, innovation_stock, *external_blocks), axis=1
    )
    names = (
        *forecast_stock_names,
        *delta_stock_names,
        *innovation_stock_names,
        *external_names,
    )
    current_names = tuple(eligible_names[int(position)] for position in overlap_positions)
    return OpenDesign(values, tuple(names), current_names)


def _fit_standardized_matrix(
    values: np.ndarray,
    fit_rows: np.ndarray,
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError(f"{label} values must be a finite row matrix")
    fit = values[fit_rows]
    mean = fit.mean(axis=0)
    standard_deviation = fit.std(axis=0)
    keep = np.isfinite(standard_deviation) & (standard_deviation > 1e-8)
    if not keep.any():
        raise ValueError(f"{label} values have no varying fit columns")
    standardized = (values[:, keep] - mean[keep]) / standard_deviation[keep]
    return standardized, keep


def _fit_pca_scores(
    standardized: np.ndarray,
    fit_rows: np.ndarray,
    *,
    rank: int,
) -> tuple[np.ndarray, float]:
    fit = np.asarray(standardized[fit_rows], dtype=np.float64)
    maximum_rank = min(fit.shape[0] - 1, fit.shape[1])
    selected_rank = min(int(rank), maximum_rank)
    if selected_rank < 1:
        raise ValueError("PCA requires at least two fit rows and one varying column")
    _, singular_values, right = np.linalg.svd(fit, full_matrices=False)
    right = right[:selected_rank].copy()
    for index in range(len(right)):
        anchor = int(np.argmax(np.abs(right[index])))
        if right[index, anchor] < 0.0:
            right[index] *= -1.0
    scores = np.asarray(standardized @ right.T, dtype=np.float64)
    fit_scale = scores[fit_rows].std(axis=0)
    if not np.isfinite(fit_scale).all() or np.any(fit_scale <= 1e-8):
        raise ValueError("PCA produced a degenerate fit component")
    scores /= fit_scale
    total_variance = float(np.square(singular_values).sum())
    explained = (
        float(np.square(singular_values[:selected_rank]).sum() / total_variance)
        if total_variance > 1e-12
        else 0.0
    )
    return scores, explained


def fit_residual_jepa_pca_design(
    sensor_values: np.ndarray,
    jepa_values: np.ndarray,
    fit_rows: Sequence[int],
    *,
    rank: int = 64,
    sensor_rank: int = 64,
    ridge_alpha: float = 1.0,
) -> ResidualPCADesign:
    """Fit an unsupervised, fit-only JEPA block not explained by open sensors."""

    sensor_values = np.asarray(sensor_values)
    jepa_values = np.asarray(jepa_values)
    if sensor_values.ndim != 2 or jepa_values.ndim != 2:
        raise ValueError("residual JEPA PCA inputs must be row matrices")
    if len(sensor_values) != len(jepa_values):
        raise ValueError("sensor and JEPA PCA rows must align")
    rows = np.asarray(fit_rows, dtype=np.int64)
    if rows.ndim != 1 or len(rows) < 20 or len(np.unique(rows)) != len(rows):
        raise ValueError("residual JEPA PCA requires at least twenty unique fit rows")
    if rows.min() < 0 or rows.max() >= len(sensor_values):
        raise ValueError("residual JEPA PCA fit rows exceed the design")
    if int(rank) < 1 or int(sensor_rank) < 1:
        raise ValueError("residual JEPA PCA ranks must be positive")
    if not np.isfinite(ridge_alpha) or float(ridge_alpha) < 0.0:
        raise ValueError("residual JEPA PCA ridge alpha must be non-negative")

    sensor_standardized, sensor_keep = _fit_standardized_matrix(
        sensor_values, rows, label="sensor"
    )
    jepa_standardized, jepa_keep = _fit_standardized_matrix(
        jepa_values, rows, label="JEPA"
    )
    sensor_scores, sensor_explained = _fit_pca_scores(
        sensor_standardized, rows, rank=int(sensor_rank)
    )
    jepa_scores, jepa_explained = _fit_pca_scores(
        jepa_standardized, rows, rank=int(rank)
    )
    fit_sensor = sensor_scores[rows]
    fit_jepa = jepa_scores[rows]
    gram = fit_sensor.T @ fit_sensor
    gram.flat[:: len(gram) + 1] += float(ridge_alpha)
    coefficients = np.linalg.solve(gram, fit_sensor.T @ fit_jepa)
    residual = jepa_scores - sensor_scores @ coefficients
    residual_mean = residual[rows].mean(axis=0)
    residual_scale = residual[rows].std(axis=0)
    keep_residual = np.isfinite(residual_scale) & (residual_scale > 1e-8)
    if not keep_residual.any():
        raise ValueError("open sensors explain every retained JEPA PCA component")
    residual = (residual[:, keep_residual] - residual_mean[keep_residual]) / (
        residual_scale[keep_residual]
    )
    if not np.isfinite(residual).all():
        raise ValueError("residual JEPA PCA produced non-finite values")
    fit_cross_correlation = (
        sensor_scores[rows].T @ residual[rows] / max(len(rows) - 1, 1)
    )
    feature_names = tuple(
        f"jepa_sensor_residual_pc:{index:03d}" for index in range(residual.shape[1])
    )
    contract = {
        "fit_only": True,
        "target_used_for_projection": False,
        "fit_rows": int(len(rows)),
        "fit_step_min": int(rows.min()),
        "fit_step_max": int(rows.max()),
        "sensor_input_features": int(sensor_values.shape[1]),
        "sensor_retained_features": int(sensor_keep.sum()),
        "jepa_input_features": int(jepa_values.shape[1]),
        "jepa_retained_features": int(jepa_keep.sum()),
        "sensor_rank": int(sensor_scores.shape[1]),
        "requested_jepa_rank": int(rank),
        "retained_jepa_rank": int(residual.shape[1]),
        "ridge_alpha": float(ridge_alpha),
        "sensor_explained_variance_ratio": float(sensor_explained),
        "jepa_explained_variance_ratio": float(jepa_explained),
        "maximum_absolute_fit_sensor_residual_correlation": float(
            np.max(np.abs(fit_cross_correlation))
        ),
    }
    return ResidualPCADesign(
        residual.astype(np.float32), feature_names, contract
    )


def shuffled_feature_block(
    values: np.ndarray,
    split_rows: Mapping[str, Sequence[int]],
    *,
    seed: int,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("the JEPA placebo block must be a row matrix")
    output = np.empty_like(values)
    covered = np.zeros(len(values), dtype=bool)
    generator = np.random.default_rng(int(seed))
    for rows in split_rows.values():
        positions = np.asarray(rows, dtype=np.int64)
        if positions.ndim != 1 or (
            len(positions) and (positions.min() < 0 or positions.max() >= len(values))
        ):
            raise ValueError("placebo split rows exceed the JEPA block")
        if covered[positions].any():
            raise ValueError("placebo split rows must be disjoint")
        permutation = generator.permutation(positions)
        if len(positions) > 1 and np.array_equal(permutation, positions):
            permutation = np.roll(permutation, 1)
        output[positions] = values[permutation]
        covered[positions] = True
    if not covered.all():
        raise ValueError("placebo split rows must cover every JEPA block row")
    return output
