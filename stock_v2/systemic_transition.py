from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


SYSTEMIC_TARGET_VERSION = "broad_systemic_v2_robust_20260714"


DEFAULT_SYSTEMIC_STATE_FEATURES = (
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "volatility_5d",
    "volatility_10d",
    "volatility_20d",
    "downside_volatility_20d",
    "volatility_ratio_20_60",
    "volume_z20",
    "value_z20",
    "value_ma20_log",
    "amihud_20d",
    "range_z20",
    "gap_open",
    "intraday_return",
    "market_corr_60d",
    "investor_foreign_flow_ratio_1d",
    "investor_institution_flow_ratio_1d",
    "investor_individual_flow_ratio_1d",
    "investor_pension_flow_ratio_1d",
    "investor_foreign_flow_ratio_5d",
    "investor_institution_flow_ratio_5d",
)


SYSTEMIC_FAMILIES = {
    "price_breadth": (
        "market_return",
        "mean_absolute_return",
        "breadth",
        "return_coherence",
    ),
    "market_risk": (
        "median_absolute_return",
        "q75_absolute_return",
        "robust_return_dispersion",
    ),
    "activity": (
        "volume_shock",
        "traded_value_shock",
    ),
    "graph_state": (
        "common_state_energy",
        "node_state_median_energy",
        "market_corr_change",
    ),
}


ABSOLUTE_COMPONENTS = frozenset(
    {
        "market_return",
        "breadth",
        "return_coherence",
        "volume_shock",
        "traded_value_shock",
        "market_corr_change",
    }
)


@dataclass(frozen=True)
class SystemicCalibration:
    component_center: dict[str, float]
    component_scale: dict[str, float]
    event_quantile: float
    event_threshold: float
    fit_event_rate: float
    broad_selloff_return_threshold: float
    broad_selloff_breadth_threshold: float
    volume_explosion_threshold: float
    value_explosion_threshold: float
    state_shift_threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def systemic_state_feature_indices(
    feature_names: Sequence[str],
    requested: Sequence[str] = DEFAULT_SYSTEMIC_STATE_FEATURES,
) -> tuple[np.ndarray, list[str]]:
    positions = {name: index for index, name in enumerate(feature_names)}
    selected_names = [name for name in requested if name in positions]
    if len(selected_names) < 8:
        raise ValueError("at least eight dynamic stock-state features are required")
    return (
        np.asarray([positions[name] for name in selected_names], dtype=np.int64),
        selected_names,
    )


def _finite_mean(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(finite.mean()) if finite.size else float("nan")


def _raw_feature_mean(
    raw_state: np.ndarray,
    available: np.ndarray,
    feature_names: Sequence[str],
    name: str,
    node_mask: np.ndarray,
    min_nodes: int,
) -> float:
    if name not in feature_names:
        return float("nan")
    index = feature_names.index(name)
    valid = (
        np.asarray(node_mask, dtype=bool)
        & (np.asarray(available)[:, index] > 0.5)
        & np.isfinite(np.asarray(raw_state)[:, index])
    )
    if int(valid.sum()) < int(min_nodes):
        return float("nan")
    return float(np.asarray(raw_state, dtype=np.float64)[valid, index].mean())


def transition_components(
    *,
    current_state: np.ndarray,
    future_state: np.ndarray,
    current_raw: np.ndarray,
    future_raw: np.ndarray,
    current_available: np.ndarray,
    future_available: np.ndarray,
    feature_names: Sequence[str],
    entry_path_returns: np.ndarray,
    node_mask: np.ndarray | None = None,
    state_feature_names: Sequence[str] = DEFAULT_SYSTEMIC_STATE_FEATURES,
    min_nodes: int = 20,
) -> dict[str, float | int]:
    """Summarize a future panel as a broad, market-level state transition.

    Equal-node means and median node displacement prevent one extreme stock from
    dominating the score. Inputs named ``state`` are training-normalized model
    states; ``raw`` inputs retain interpretable units such as rolling volume z.
    """

    current_state = np.asarray(current_state, dtype=np.float64)
    future_state = np.asarray(future_state, dtype=np.float64)
    current_raw = np.asarray(current_raw, dtype=np.float64)
    future_raw = np.asarray(future_raw, dtype=np.float64)
    current_available = np.asarray(current_available, dtype=bool)
    future_available = np.asarray(future_available, dtype=bool)
    if current_state.shape != future_state.shape:
        raise ValueError("current and future state shapes must match")
    if current_state.shape != current_raw.shape or current_state.shape != future_raw.shape:
        raise ValueError("normalized and raw state shapes must match")
    if current_state.shape != current_available.shape or current_state.shape != future_available.shape:
        raise ValueError("state and availability shapes must match")
    if current_state.ndim != 2 or current_state.shape[1] != len(feature_names):
        raise ValueError("state must be a node-by-feature matrix")

    if node_mask is None:
        node_mask = np.ones(current_state.shape[0], dtype=bool)
    node_mask = np.asarray(node_mask, dtype=bool)
    if node_mask.shape != (current_state.shape[0],):
        raise ValueError("node_mask must contain one value per stock node")

    path = np.asarray(entry_path_returns, dtype=np.float64)
    if path.shape != node_mask.shape:
        raise ValueError("entry_path_returns must contain one value per stock node")
    path_valid = node_mask & np.isfinite(path)
    if int(path_valid.sum()) < int(min_nodes):
        path_values = np.empty(0, dtype=np.float64)
    else:
        path_values = path[path_valid]

    if path_values.size:
        market_return = float(path_values.mean())
        mean_absolute_return = float(np.abs(path_values).mean())
        market_path_rms = float(np.sqrt(np.square(path_values).mean()))
        return_dispersion = float(path_values.std())
        absolute_returns = np.abs(path_values)
        median_absolute_return = float(np.median(absolute_returns))
        q75_absolute_return = float(np.quantile(absolute_returns, 0.75))
        return_median = float(np.median(path_values))
        robust_return_dispersion = float(
            1.4826 * np.median(np.abs(path_values - return_median))
        )
        breadth = float(np.sign(path_values).mean())
        return_coherence = (
            float(market_return / mean_absolute_return)
            if mean_absolute_return > 1e-12
            else 0.0
        )
        squared = np.square(path_values)
        return_concentration = (
            float(squared.max() / squared.sum()) if squared.sum() > 1e-16 else 0.0
        )
    else:
        market_return = float("nan")
        mean_absolute_return = float("nan")
        market_path_rms = float("nan")
        return_dispersion = float("nan")
        median_absolute_return = float("nan")
        q75_absolute_return = float("nan")
        robust_return_dispersion = float("nan")
        breadth = float("nan")
        return_coherence = float("nan")
        return_concentration = float("nan")

    feature_indices, selected_names = systemic_state_feature_indices(
        feature_names, state_feature_names
    )
    selected_current = current_state[:, feature_indices]
    selected_future = future_state[:, feature_indices]
    valid = (
        node_mask[:, None]
        & current_available[:, feature_indices]
        & future_available[:, feature_indices]
        & np.isfinite(selected_current)
        & np.isfinite(selected_future)
    )
    delta = selected_future - selected_current
    feature_counts = valid.sum(axis=0)
    usable_features = feature_counts >= int(min_nodes)
    feature_means = np.full(len(feature_indices), np.nan, dtype=np.float64)
    if usable_features.any():
        feature_means[usable_features] = np.divide(
            np.where(valid[:, usable_features], delta[:, usable_features], 0.0).sum(axis=0),
            feature_counts[usable_features],
        )
    common_state_energy = float(
        np.sqrt(np.nanmean(np.square(feature_means[usable_features])))
    ) if usable_features.any() else float("nan")

    observed_delta = delta[valid]
    total_state_energy = (
        float(np.sqrt(np.square(observed_delta).mean()))
        if observed_delta.size
        else float("nan")
    )
    node_counts = valid.sum(axis=1)
    minimum_node_features = max(3, int(math.ceil(len(feature_indices) * 0.25)))
    usable_nodes = node_mask & (node_counts >= minimum_node_features)
    node_energy = np.full(current_state.shape[0], np.nan, dtype=np.float64)
    if usable_nodes.any():
        node_energy[usable_nodes] = np.sqrt(
            np.divide(
                np.where(valid[usable_nodes], np.square(delta[usable_nodes]), 0.0).sum(axis=1),
                node_counts[usable_nodes],
            )
        )
    finite_node_energy = node_energy[np.isfinite(node_energy)]
    node_state_median_energy = (
        float(np.median(finite_node_energy))
        if finite_node_energy.size >= int(min_nodes)
        else float("nan")
    )
    node_state_q75_energy = (
        float(np.quantile(finite_node_energy, 0.75))
        if finite_node_energy.size >= int(min_nodes)
        else float("nan")
    )
    state_coherence = (
        float(common_state_energy / total_state_energy)
        if np.isfinite(common_state_energy)
        and np.isfinite(total_state_energy)
        and total_state_energy > 1e-12
        else float("nan")
    )

    volume_shock = _raw_feature_mean(
        future_raw, future_available, feature_names, "volume_z20", node_mask, min_nodes
    )
    traded_value_shock = _raw_feature_mean(
        future_raw, future_available, feature_names, "value_z20", node_mask, min_nodes
    )
    current_market_corr = _raw_feature_mean(
        current_raw,
        current_available,
        feature_names,
        "market_corr_60d",
        node_mask,
        min_nodes,
    )
    future_market_corr = _raw_feature_mean(
        future_raw,
        future_available,
        feature_names,
        "market_corr_60d",
        node_mask,
        min_nodes,
    )
    market_corr_change = future_market_corr - current_market_corr

    return {
        "observed_nodes": int(path_values.size),
        "state_features": int(len(selected_names)),
        "market_return": market_return,
        "mean_absolute_return": mean_absolute_return,
        "market_path_rms": market_path_rms,
        "return_dispersion": return_dispersion,
        "median_absolute_return": median_absolute_return,
        "q75_absolute_return": q75_absolute_return,
        "robust_return_dispersion": robust_return_dispersion,
        "breadth": breadth,
        "return_coherence": return_coherence,
        "return_concentration": return_concentration,
        "volume_shock": volume_shock,
        "traded_value_shock": traded_value_shock,
        "market_corr_change": float(market_corr_change),
        "common_state_energy": common_state_energy,
        "total_state_energy": total_state_energy,
        "node_state_median_energy": node_state_median_energy,
        "node_state_q75_energy": node_state_q75_energy,
        "state_coherence": state_coherence,
    }


def _component_magnitude(name: str, value: float) -> float:
    value = float(value)
    if not np.isfinite(value):
        return float("nan")
    return abs(value) if name in ABSOLUTE_COMPONENTS else max(value, 0.0)


def _robust_center_scale(values: Iterable[float], quantile: float) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size < 20:
        raise ValueError("at least twenty finite fit observations are required")
    center = float(np.median(array))
    upper = float(np.quantile(array, quantile))
    scale = upper - center
    if scale <= 1e-12:
        scale = float(np.std(array))
    if scale <= 1e-12:
        scale = 1.0
    return center, scale


def score_systemic_components(
    row: Mapping[str, float | int],
    calibration: SystemicCalibration,
) -> dict[str, float]:
    component_scores: dict[str, float] = {}
    for name, center in calibration.component_center.items():
        magnitude = _component_magnitude(name, float(row.get(name, float("nan"))))
        scale = float(calibration.component_scale[name])
        component_scores[name] = (
            float(np.clip((magnitude - float(center)) / scale, 0.0, 20.0))
            if np.isfinite(magnitude)
            else float("nan")
        )
    family_scores: dict[str, float] = {}
    for family, names in SYSTEMIC_FAMILIES.items():
        values = np.asarray(
            [component_scores.get(name, float("nan")) for name in names],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        family_scores[family] = (
            float(np.sqrt(np.square(values).mean()))
            if values.size
            else float("nan")
        )
    families = np.asarray(list(family_scores.values()), dtype=np.float64)
    families = families[np.isfinite(families)]
    overall = (
        float(np.sqrt(np.square(families).mean()))
        if families.size == len(SYSTEMIC_FAMILIES)
        else float("nan")
    )
    return {
        **{f"component:{name}": value for name, value in component_scores.items()},
        **{f"family:{name}": value for name, value in family_scores.items()},
        "systemic_energy": overall,
    }


def fit_systemic_calibration(
    rows: Sequence[Mapping[str, float | int]],
    *,
    event_quantile: float = 0.90,
) -> SystemicCalibration:
    if not 0.75 <= float(event_quantile) < 1.0:
        raise ValueError("event_quantile must be in [0.75, 1.0)")
    component_names = tuple(
        dict.fromkeys(name for names in SYSTEMIC_FAMILIES.values() for name in names)
    )
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in component_names:
        magnitudes = [
            _component_magnitude(name, float(row.get(name, float("nan"))))
            for row in rows
        ]
        centers[name], scales[name] = _robust_center_scale(
            magnitudes, float(event_quantile)
        )

    provisional = SystemicCalibration(
        component_center=centers,
        component_scale=scales,
        event_quantile=float(event_quantile),
        event_threshold=float("nan"),
        fit_event_rate=float("nan"),
        broad_selloff_return_threshold=float("nan"),
        broad_selloff_breadth_threshold=float("nan"),
        volume_explosion_threshold=float("nan"),
        value_explosion_threshold=float("nan"),
        state_shift_threshold=float("nan"),
    )
    scored = [score_systemic_components(row, provisional) for row in rows]
    energies = np.asarray([row["systemic_energy"] for row in scored], dtype=np.float64)
    finite_energy = energies[np.isfinite(energies)]
    if finite_energy.size < 20:
        raise ValueError("too few complete fit rows for systemic event calibration")
    event_threshold = float(np.quantile(finite_energy, float(event_quantile)))

    def quantile(name: str, value: float) -> float:
        array = np.asarray(
            [float(row.get(name, float("nan"))) for row in rows], dtype=np.float64
        )
        array = array[np.isfinite(array)]
        if array.size < 20:
            return float("nan")
        return float(np.quantile(array, value))

    state_scores = np.asarray(
        [row["family:graph_state"] for row in scored], dtype=np.float64
    )
    state_scores = state_scores[np.isfinite(state_scores)]
    return replace(
        provisional,
        event_threshold=event_threshold,
        fit_event_rate=float((finite_energy >= event_threshold).mean()),
        broad_selloff_return_threshold=quantile("market_return", 0.10),
        broad_selloff_breadth_threshold=quantile("breadth", 0.10),
        volume_explosion_threshold=quantile("volume_shock", 0.90),
        value_explosion_threshold=quantile("traded_value_shock", 0.90),
        state_shift_threshold=float(np.quantile(state_scores, 0.90)),
    )


def event_labels(
    row: Mapping[str, float | int],
    calibration: SystemicCalibration,
) -> dict[str, bool]:
    scored = score_systemic_components(row, calibration)
    market_return = float(row.get("market_return", float("nan")))
    breadth = float(row.get("breadth", float("nan")))
    volume = float(row.get("volume_shock", float("nan")))
    value = float(row.get("traded_value_shock", float("nan")))
    graph_state = float(scored["family:graph_state"])
    return {
        "systemic_event": bool(
            np.isfinite(scored["systemic_energy"])
            and scored["systemic_energy"] >= calibration.event_threshold
        ),
        "broad_selloff": bool(
            np.isfinite(market_return)
            and np.isfinite(breadth)
            and market_return <= calibration.broad_selloff_return_threshold
            and breadth <= calibration.broad_selloff_breadth_threshold
        ),
        "turnover_explosion": bool(
            (np.isfinite(volume) and volume >= calibration.volume_explosion_threshold)
            or (np.isfinite(value) and value >= calibration.value_explosion_threshold)
        ),
        "graph_state_shift": bool(
            np.isfinite(graph_state) and graph_state >= calibration.state_shift_threshold
        ),
    }


def derived_subtype_scores(
    row: Mapping[str, float | int],
    calibration: SystemicCalibration,
) -> dict[str, float]:
    """Derive subtype rankings from dense continuous component predictions.

    The subtype labels are deterministic threshold combinations of these same
    components. Ranking the continuous predictions avoids fitting a separate
    classifier from the much smaller set of rare positive event days.
    """

    def value(name: str) -> float:
        return float(row.get(name, float("nan")))

    def scale(name: str) -> float:
        return max(float(calibration.component_scale[name]), 1e-8)

    market_return = value("market_return")
    breadth = value("breadth")
    if np.isfinite(market_return) and np.isfinite(breadth):
        return_severity = (
            float(calibration.broad_selloff_return_threshold) - market_return
        ) / scale("market_return")
        breadth_severity = (
            float(calibration.broad_selloff_breadth_threshold) - breadth
        ) / scale("breadth")
        broad_selloff = float(min(return_severity, breadth_severity))
    else:
        broad_selloff = float("nan")

    volume = value("volume_shock")
    traded_value = value("traded_value_shock")
    turnover_candidates = []
    if np.isfinite(volume):
        turnover_candidates.append(
            (volume - float(calibration.volume_explosion_threshold))
            / scale("volume_shock")
        )
    if np.isfinite(traded_value):
        turnover_candidates.append(
            (traded_value - float(calibration.value_explosion_threshold))
            / scale("traded_value_shock")
        )
    turnover_explosion = (
        float(max(turnover_candidates)) if turnover_candidates else float("nan")
    )

    graph_state = float(
        score_systemic_components(row, calibration)["family:graph_state"]
    )
    graph_state_shift = (
        graph_state - float(calibration.state_shift_threshold)
        if np.isfinite(graph_state)
        else float("nan")
    )
    return {
        "broad_selloff": broad_selloff,
        "turnover_explosion": turnover_explosion,
        "graph_state_shift": graph_state_shift,
    }


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1.0
        start = end
    return ranks


def _roc_auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = labels > 0
    positive_count = int(positives.sum())
    negative_count = int((~positives).sum())
    if positive_count == 0 or negative_count == 0:
        return float("nan")
    ranks = _average_ranks(scores)
    return float(
        (ranks[positives].sum() - positive_count * (positive_count + 1) / 2.0)
        / (positive_count * negative_count)
    )


def _average_precision(scores: np.ndarray, labels: np.ndarray) -> float:
    positives = int(labels.sum())
    if positives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")[::-1]
    sorted_labels = labels[order].astype(np.float64)
    precision = np.cumsum(sorted_labels) / np.arange(1, len(labels) + 1)
    return float((precision * sorted_labels).sum() / positives)


def binary_ranking_metrics(
    labels: Sequence[bool | int],
    scores: Sequence[float],
    *,
    selection_rate: float | None = None,
) -> dict[str, float | int]:
    labels_array = np.asarray(labels, dtype=bool)
    scores_array = np.asarray(scores, dtype=np.float64)
    if labels_array.shape != scores_array.shape or labels_array.ndim != 1:
        raise ValueError("labels and scores must be aligned vectors")
    valid = np.isfinite(scores_array)
    labels_array = labels_array[valid]
    scores_array = scores_array[valid]
    if len(labels_array) < 20:
        raise ValueError("at least twenty finite ranking rows are required")
    event_rate = float(labels_array.mean())
    rate = event_rate if selection_rate is None else float(selection_rate)
    if not 0.0 < rate <= 1.0:
        raise ValueError("selection_rate must be in (0, 1]")
    selected_count = max(1, int(math.ceil(len(labels_array) * rate)))
    selected = np.argsort(scores_array, kind="mergesort")[-selected_count:]
    true_positive = int(labels_array[selected].sum())
    events = int(labels_array.sum())
    precision = true_positive / float(selected_count)
    recall = true_positive / float(events) if events else float("nan")
    return {
        "rows": int(len(labels_array)),
        "events": events,
        "event_rate": event_rate,
        "roc_auc": _roc_auc(scores_array, labels_array),
        "average_precision": _average_precision(scores_array, labels_array),
        "selected_count": int(selected_count),
        "precision_at_selection_rate": float(precision),
        "recall_at_selection_rate": float(recall),
        "lift_at_selection_rate": (
            float(precision / event_rate) if event_rate > 0.0 else float("nan")
        ),
    }


def systemic_score_metrics(
    actual_rows: Sequence[Mapping[str, float | int]],
    predicted_rows: Sequence[Mapping[str, float | int]],
    calibration: SystemicCalibration,
) -> dict[str, float | int]:
    if len(actual_rows) != len(predicted_rows):
        raise ValueError("actual and predicted rows must be aligned")
    actual_scores = np.asarray(
        [score_systemic_components(row, calibration)["systemic_energy"] for row in actual_rows],
        dtype=np.float64,
    )
    predicted_scores = np.asarray(
        [score_systemic_components(row, calibration)["systemic_energy"] for row in predicted_rows],
        dtype=np.float64,
    )
    valid = np.isfinite(actual_scores) & np.isfinite(predicted_scores)
    actual_scores = actual_scores[valid]
    predicted_scores = predicted_scores[valid]
    if len(actual_scores) < 20:
        raise ValueError("at least twenty aligned systemic scores are required")
    labels = actual_scores >= float(calibration.event_threshold)
    event_count = int(labels.sum())
    selection_rate = float(calibration.fit_event_rate)
    selected_count = max(1, int(math.ceil(len(labels) * selection_rate)))
    selected = np.argsort(predicted_scores, kind="mergesort")[-selected_count:]
    true_positive = int(labels[selected].sum())
    precision = true_positive / float(selected_count)
    recall = true_positive / float(event_count) if event_count else float("nan")
    event_rate = float(labels.mean())
    centered_actual = actual_scores - actual_scores.mean()
    centered_predicted = predicted_scores - predicted_scores.mean()
    denominator = float(
        np.sqrt(np.square(centered_actual).sum() * np.square(centered_predicted).sum())
    )
    correlation = (
        float(centered_actual @ centered_predicted / denominator)
        if denominator > 1e-12
        else float("nan")
    )
    total_mass = float(actual_scores.sum())
    captured_mass = float(actual_scores[selected].sum())

    actual_returns = np.asarray(
        [float(row.get("market_return", float("nan"))) for row in actual_rows],
        dtype=np.float64,
    )[valid]
    predicted_returns = np.asarray(
        [float(row.get("market_return", float("nan"))) for row in predicted_rows],
        dtype=np.float64,
    )[valid]
    event_direction_valid = labels & np.isfinite(actual_returns) & np.isfinite(predicted_returns)
    if event_direction_valid.any():
        correct = np.sign(actual_returns[event_direction_valid]) == np.sign(
            predicted_returns[event_direction_valid]
        )
        weights = actual_scores[event_direction_valid]
        weighted_direction = float(np.average(correct.astype(np.float64), weights=weights))
    else:
        weighted_direction = float("nan")

    ranking = binary_ranking_metrics(
        labels, predicted_scores, selection_rate=selection_rate
    )
    return {
        "rows": int(ranking["rows"]),
        "events": int(ranking["events"]),
        "event_rate": float(ranking["event_rate"]),
        "roc_auc": float(ranking["roc_auc"]),
        "average_precision": float(ranking["average_precision"]),
        "precision_at_fit_event_rate": float(precision),
        "recall_at_fit_event_rate": float(recall),
        "lift_at_fit_event_rate": (
            float(precision / event_rate) if event_rate > 0.0 else float("nan")
        ),
        "systemic_energy_correlation": correlation,
        "tail_mass_recall_at_fit_event_rate": (
            float(captured_mass / total_mass) if total_mass > 1e-12 else float("nan")
        ),
        "event_impact_weighted_market_direction_accuracy": weighted_direction,
        "selected_count": int(selected_count),
    }
