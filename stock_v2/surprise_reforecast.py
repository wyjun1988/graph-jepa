from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


SURPRISE_STATISTIC_NAMES = (
    "common_energy",
    "total_energy",
    "node_median_energy",
    "node_q75_energy",
    "node_participation",
    "state_coherence",
    "graph_edge_coherence",
    "systemic_surprise_energy",
)


@dataclass(frozen=True)
class ResidualSurpriseCalibration:
    feature_center: np.ndarray
    feature_scale: np.ndarray
    energy_threshold: float
    threshold_quantile: float
    min_nodes: int
    node_z_threshold: float
    clip: float

    def __post_init__(self) -> None:
        center = np.asarray(self.feature_center, dtype=np.float64)
        scale = np.asarray(self.feature_scale, dtype=np.float64)
        if center.ndim != 1 or center.shape != scale.shape:
            raise ValueError("surprise feature center and scale must be aligned vectors")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("surprise feature calibration must be finite")
        if (scale <= 0.0).any():
            raise ValueError("surprise feature scales must be positive")
        if not np.isfinite(self.energy_threshold) or self.energy_threshold < 0.0:
            raise ValueError("surprise energy threshold must be finite and non-negative")
        if not 0.0 < float(self.threshold_quantile) < 1.0:
            raise ValueError("surprise threshold quantile must be in (0, 1)")
        if int(self.min_nodes) < 2:
            raise ValueError("surprise calibration requires at least two nodes")
        if not np.isfinite(self.node_z_threshold) or self.node_z_threshold <= 0.0:
            raise ValueError("surprise node z threshold must be positive")
        if not np.isfinite(self.clip) or self.clip <= 0.0:
            raise ValueError("surprise clipping must be positive")


@dataclass(frozen=True)
class ResidualSurpriseDesign:
    values: np.ndarray
    feature_names: tuple[str, ...]
    is_surprise: np.ndarray


@dataclass(frozen=True)
class OpenShockTrajectory:
    horizons: tuple[int, ...]
    open_direction: np.ndarray
    open_energy: np.ndarray
    open_median_return: np.ndarray
    open_directional_breadth: np.ndarray
    remaining_market_return: np.ndarray
    aligned_remaining_return: np.ndarray
    aligned_total_return: np.ndarray
    impact_extension: np.ndarray
    aligned_close_breadth: np.ndarray
    node_continuation_fraction: np.ndarray
    market_mfe: np.ndarray
    market_mae: np.ndarray
    node_mfe: np.ndarray
    node_mae: np.ndarray
    time_to_market_peak: np.ndarray
    valid: np.ndarray


def _validate_rows(rows: Sequence[int], total: int, label: str) -> np.ndarray:
    result = np.asarray(rows, dtype=np.int64)
    if result.ndim != 1 or not len(result):
        raise ValueError(f"{label} rows must be a non-empty vector")
    if result.min() < 0 or result.max() >= int(total):
        raise ValueError(f"{label} rows exceed the time axis")
    if len(np.unique(result)) != len(result):
        raise ValueError(f"{label} rows must be unique")
    return result


def _robust_feature_location_scale(
    residuals: np.ndarray,
    valid: np.ndarray,
    fit_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    feature_count = residuals.shape[-1]
    center = np.zeros(feature_count, dtype=np.float64)
    scale = np.ones(feature_count, dtype=np.float64)
    for feature in range(feature_count):
        selected = residuals[fit_rows, :, feature]
        selected_valid = valid[fit_rows, :, feature] & np.isfinite(selected)
        values = selected[selected_valid].astype(np.float64)
        if values.size < 3:
            continue
        feature_center = float(np.median(values))
        absolute = np.abs(values - feature_center)
        feature_scale = float(1.4826 * np.median(absolute))
        if not np.isfinite(feature_scale) or feature_scale < 1e-6:
            feature_scale = float(np.sqrt(np.mean(np.square(values - feature_center))))
        if not np.isfinite(feature_scale) or feature_scale < 1e-6:
            feature_scale = 1.0
        center[feature] = feature_center
        scale[feature] = feature_scale
    return center, scale


def _edge_coherence(
    standardized: np.ndarray,
    valid: np.ndarray,
    common: np.ndarray,
    edge_index: np.ndarray | None,
    edge_weight: np.ndarray | None,
    stock_count: int,
) -> float:
    if edge_index is None:
        return float("nan")
    edges = np.asarray(edge_index, dtype=np.int64)
    if edges.ndim != 2 or edges.shape[0] != 2:
        raise ValueError("surprise edge_index must be shaped [2, edges]")
    selected = (
        (edges[0] >= 0)
        & (edges[1] >= 0)
        & (edges[0] < int(stock_count))
        & (edges[1] < int(stock_count))
    )
    edges = edges[:, selected]
    if not edges.shape[1]:
        return float("nan")
    direction = np.sign(common)
    useful_features = np.isfinite(common) & (np.abs(common) > 1e-8)
    if not useful_features.any():
        return 0.0
    projected = np.full(int(stock_count), np.nan, dtype=np.float64)
    for node in range(int(stock_count)):
        usable = valid[node] & useful_features & np.isfinite(standardized[node])
        if usable.any():
            projected[node] = float(
                np.mean(standardized[node, usable] * direction[usable])
            )
    source = projected[edges[0]]
    target = projected[edges[1]]
    usable_edges = np.isfinite(source) & np.isfinite(target)
    if not usable_edges.any():
        return float("nan")
    agreement = np.sign(source[usable_edges]) * np.sign(target[usable_edges])
    if edge_weight is None:
        return float(np.mean(agreement))
    weights = np.asarray(edge_weight, dtype=np.float64)
    if weights.shape != (selected.shape[0],):
        raise ValueError("surprise edge weights must contain one value per edge")
    signed_weights = weights[selected][usable_edges]
    usable_weights = np.isfinite(signed_weights) & (np.abs(signed_weights) > 0.0)
    if not usable_weights.any():
        return float(np.mean(agreement))
    expected_agreement = agreement[usable_weights] * np.sign(
        signed_weights[usable_weights]
    )
    return float(
        np.average(
            expected_agreement,
            weights=np.abs(signed_weights[usable_weights]),
        )
    )


def summarize_residual_surprise(
    residuals: np.ndarray,
    valid: np.ndarray,
    *,
    feature_center: np.ndarray,
    feature_scale: np.ndarray,
    stock_count: int,
    min_nodes: int = 20,
    node_z_threshold: float = 1.0,
    clip: float = 8.0,
    edge_index: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
) -> np.ndarray:
    residuals = np.asarray(residuals, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    center = np.asarray(feature_center, dtype=np.float64)
    scale = np.asarray(feature_scale, dtype=np.float64)
    if residuals.ndim != 3 or residuals.shape != valid.shape:
        raise ValueError("surprise residuals and validity must be [time, node, feature]")
    if center.shape != (residuals.shape[-1],) or scale.shape != center.shape:
        raise ValueError("surprise feature calibration does not match residual width")
    if not 2 <= int(stock_count) <= residuals.shape[1]:
        raise ValueError("surprise stock_count does not fit the node axis")
    if int(min_nodes) < 2 or int(min_nodes) > int(stock_count):
        raise ValueError("surprise min_nodes must fit the stock count")
    if (scale <= 0.0).any() or not np.isfinite(scale).all():
        raise ValueError("surprise feature scales must be finite and positive")
    standardized = np.clip(
        (residuals - center[None, None, :]) / scale[None, None, :],
        -float(clip),
        float(clip),
    )
    standardized = np.where(valid & np.isfinite(standardized), standardized, np.nan)
    output = np.full(
        (residuals.shape[0], len(SURPRISE_STATISTIC_NAMES)),
        np.nan,
        dtype=np.float64,
    )
    for time_index in range(residuals.shape[0]):
        values = standardized[time_index, : int(stock_count)]
        available = np.isfinite(values)
        feature_counts = available.sum(axis=0)
        usable_features = feature_counts >= int(min_nodes)
        node_counts = available[:, usable_features].sum(axis=1)
        minimum_features = max(1, int(np.ceil(max(usable_features.sum(), 1) * 0.25)))
        usable_nodes = node_counts >= minimum_features
        if not usable_features.any() or int(usable_nodes.sum()) < int(min_nodes):
            continue
        common = np.nanmedian(values[:, usable_features], axis=0)
        selected = values[usable_nodes][:, usable_features]
        total_energy = float(np.sqrt(np.nanmean(np.square(selected))))
        common_energy = float(np.sqrt(np.nanmean(np.square(common))))
        node_energy = np.sqrt(np.nanmean(np.square(selected), axis=1))
        node_median = float(np.nanmedian(node_energy))
        node_q75 = float(np.nanquantile(node_energy, 0.75))
        participation = float(np.mean(node_energy >= float(node_z_threshold)))
        coherence = (
            float(np.clip(common_energy / total_energy, 0.0, 1.0))
            if total_energy > 1e-12
            else 0.0
        )
        full_common = np.full(residuals.shape[-1], np.nan, dtype=np.float64)
        full_common[usable_features] = common
        graph_coherence = _edge_coherence(
            standardized[time_index, : int(stock_count)],
            np.isfinite(standardized[time_index, : int(stock_count)]),
            full_common,
            edge_index,
            edge_weight,
            int(stock_count),
        )
        systemic_energy = float(
            np.sqrt(
                common_energy**2
                + (node_median * np.sqrt(max(participation, 0.0))) ** 2
            )
        )
        output[time_index] = (
            common_energy,
            total_energy,
            node_median,
            node_q75,
            participation,
            coherence,
            graph_coherence,
            systemic_energy,
        )
    return output


def fit_residual_surprise_calibration(
    residuals: np.ndarray,
    valid: np.ndarray,
    fit_rows: Sequence[int],
    *,
    stock_count: int,
    threshold_quantile: float = 0.80,
    min_nodes: int = 20,
    node_z_threshold: float = 1.0,
    clip: float = 8.0,
    edge_index: np.ndarray | None = None,
    edge_weight: np.ndarray | None = None,
) -> tuple[ResidualSurpriseCalibration, ResidualSurpriseDesign]:
    residuals = np.asarray(residuals, dtype=np.float64)
    valid = np.asarray(valid, dtype=bool)
    if residuals.ndim != 3 or residuals.shape != valid.shape:
        raise ValueError("surprise residuals and validity must be aligned 3D arrays")
    rows = _validate_rows(fit_rows, len(residuals), "surprise fit")
    if not 0.0 < float(threshold_quantile) < 1.0:
        raise ValueError("surprise threshold quantile must be in (0, 1)")
    center, scale = _robust_feature_location_scale(residuals, valid, rows)
    values = summarize_residual_surprise(
        residuals,
        valid,
        feature_center=center,
        feature_scale=scale,
        stock_count=int(stock_count),
        min_nodes=int(min_nodes),
        node_z_threshold=float(node_z_threshold),
        clip=float(clip),
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    energy_index = SURPRISE_STATISTIC_NAMES.index("systemic_surprise_energy")
    fit_energy = values[rows, energy_index]
    fit_energy = fit_energy[np.isfinite(fit_energy)]
    if len(fit_energy) < 3:
        raise ValueError("too few valid fit rows to calibrate surprise energy")
    threshold = float(np.quantile(fit_energy, float(threshold_quantile)))
    calibration = ResidualSurpriseCalibration(
        feature_center=center,
        feature_scale=scale,
        energy_threshold=threshold,
        threshold_quantile=float(threshold_quantile),
        min_nodes=int(min_nodes),
        node_z_threshold=float(node_z_threshold),
        clip=float(clip),
    )
    is_surprise = np.isfinite(values[:, energy_index]) & (
        values[:, energy_index] >= threshold
    )
    return calibration, ResidualSurpriseDesign(
        values=values,
        feature_names=SURPRISE_STATISTIC_NAMES,
        is_surprise=is_surprise,
    )


def build_open_shock_trajectory(
    gap_open: np.ndarray,
    close_from_open: np.ndarray,
    valid_gap: np.ndarray,
    valid_close: np.ndarray,
    *,
    horizons: Sequence[int],
    min_nodes: int = 20,
) -> OpenShockTrajectory:
    gap_open = np.asarray(gap_open, dtype=np.float64)
    close_from_open = np.asarray(close_from_open, dtype=np.float64)
    valid_gap = np.asarray(valid_gap, dtype=bool)
    valid_close = np.asarray(valid_close, dtype=bool)
    normalized_horizons = tuple(int(value) for value in horizons)
    if gap_open.ndim != 2 or gap_open.shape != valid_gap.shape:
        raise ValueError("open shock gap and validity must be [time, stock]")
    expected = (gap_open.shape[0], len(normalized_horizons), gap_open.shape[1])
    if close_from_open.shape != expected or valid_close.shape != expected:
        raise ValueError("open shock close paths must be [time, horizon, stock]")
    if (
        not normalized_horizons
        or tuple(sorted(normalized_horizons)) != normalized_horizons
        or len(set(normalized_horizons)) != len(normalized_horizons)
        or normalized_horizons[0] < 1
    ):
        raise ValueError("open shock horizons must be positive, unique, and sorted")
    if not 2 <= int(min_nodes) <= gap_open.shape[1]:
        raise ValueError("open shock min_nodes must fit the stock axis")

    time_count, horizon_count, _ = close_from_open.shape
    open_direction = np.full(time_count, np.nan, dtype=np.float64)
    open_energy = np.full(time_count, np.nan, dtype=np.float64)
    open_median = np.full(time_count, np.nan, dtype=np.float64)
    open_breadth = np.full(time_count, np.nan, dtype=np.float64)
    shape = (time_count, horizon_count)
    remaining = np.full(shape, np.nan, dtype=np.float64)
    aligned_remaining = np.full(shape, np.nan, dtype=np.float64)
    aligned_total = np.full(shape, np.nan, dtype=np.float64)
    extension = np.full(shape, np.nan, dtype=np.float64)
    aligned_breadth = np.full(shape, np.nan, dtype=np.float64)
    continuation_fraction = np.full(shape, np.nan, dtype=np.float64)
    market_mfe = np.full(shape, np.nan, dtype=np.float64)
    market_mae = np.full(shape, np.nan, dtype=np.float64)
    node_mfe = np.full(shape, np.nan, dtype=np.float64)
    node_mae = np.full(shape, np.nan, dtype=np.float64)
    time_to_peak = np.full(shape, np.nan, dtype=np.float64)
    output_valid = np.zeros(shape, dtype=bool)

    for time_index in range(time_count):
        gap_valid = valid_gap[time_index] & np.isfinite(gap_open[time_index])
        if int(gap_valid.sum()) < int(min_nodes):
            continue
        gaps = gap_open[time_index, gap_valid]
        median_gap = float(np.median(gaps))
        mean_gap = float(np.mean(gaps))
        direction = float(np.sign(median_gap if abs(median_gap) > 1e-12 else mean_gap))
        if direction == 0.0:
            continue
        signed_breadth = float(np.mean(np.sign(gaps)))
        directional_breadth = float(direction * signed_breadth)
        median_absolute = float(np.median(np.abs(gaps)))
        energy = float(
            np.sqrt(
                median_gap**2
                + (median_absolute * max(directional_breadth, 0.0)) ** 2
            )
        )
        open_direction[time_index] = direction
        open_energy[time_index] = energy
        open_median[time_index] = median_gap
        open_breadth[time_index] = directional_breadth

        market_path = np.full(horizon_count, np.nan, dtype=np.float64)
        node_path = np.full((horizon_count, gap_open.shape[1]), np.nan, dtype=np.float64)
        for horizon_index in range(horizon_count):
            valid = (
                gap_valid
                & valid_close[time_index, horizon_index]
                & np.isfinite(close_from_open[time_index, horizon_index])
            )
            if int(valid.sum()) < int(min_nodes):
                continue
            from_open = close_from_open[time_index, horizon_index, valid]
            total = (
                (1.0 + gap_open[time_index, valid]) * (1.0 + from_open) - 1.0
            )
            market_from_open = float(np.median(from_open))
            market_total = float(np.median(total))
            remaining[time_index, horizon_index] = market_from_open
            aligned_remaining[time_index, horizon_index] = direction * market_from_open
            aligned_total[time_index, horizon_index] = direction * market_total
            extension[time_index, horizon_index] = (
                direction * market_total - abs(median_gap)
            )
            aligned_breadth[time_index, horizon_index] = float(
                np.mean(direction * np.sign(total))
            )
            continuation_fraction[time_index, horizon_index] = float(
                np.mean(direction * from_open > 0.0)
            )
            market_path[horizon_index] = direction * market_from_open
            node_path[horizon_index, valid] = direction * from_open
            available_market = np.flatnonzero(np.isfinite(market_path[: horizon_index + 1]))
            if not len(available_market):
                continue
            selected_market = market_path[available_market]
            peak_position = int(np.argmax(selected_market))
            market_mfe[time_index, horizon_index] = float(np.max(selected_market))
            market_mae[time_index, horizon_index] = float(np.min(selected_market))
            time_to_peak[time_index, horizon_index] = float(
                normalized_horizons[int(available_market[peak_position])]
            )
            selected_nodes = node_path[: horizon_index + 1]
            node_observed = np.isfinite(selected_nodes)
            usable_nodes = node_observed.any(axis=0)
            if int(usable_nodes.sum()) >= int(min_nodes):
                favorable = np.nanmax(selected_nodes[:, usable_nodes], axis=0)
                adverse = np.nanmin(selected_nodes[:, usable_nodes], axis=0)
                node_mfe[time_index, horizon_index] = float(np.median(favorable))
                node_mae[time_index, horizon_index] = float(np.median(adverse))
            output_valid[time_index, horizon_index] = True

    return OpenShockTrajectory(
        horizons=normalized_horizons,
        open_direction=open_direction,
        open_energy=open_energy,
        open_median_return=open_median,
        open_directional_breadth=open_breadth,
        remaining_market_return=remaining,
        aligned_remaining_return=aligned_remaining,
        aligned_total_return=aligned_total,
        impact_extension=extension,
        aligned_close_breadth=aligned_breadth,
        node_continuation_fraction=continuation_fraction,
        market_mfe=market_mfe,
        market_mae=market_mae,
        node_mfe=node_mfe,
        node_mae=node_mae,
        time_to_market_peak=time_to_peak,
        valid=output_valid,
    )
