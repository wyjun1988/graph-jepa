from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Iterable, Mapping, Sequence

import numpy as np

from stock_v2.systemic_transition import (
    DEFAULT_SYSTEMIC_STATE_FEATURES,
    binary_ranking_metrics,
    systemic_state_feature_indices,
)


MARKET_TRANSITION_TARGET_VERSION = "market_transition_v6_systemic_impact_20260714"
MARKET_TRANSITION_IMPACT_METRIC_VERSION = (
    "market_transition_systemic_impact_mass_v2_20260714"
)


MARKET_TRANSITION_FAMILIES = {
    "price_co_movement": (
        "median_return",
        "median_absolute_return",
        "q75_absolute_return",
        "return_breadth",
        "robust_return_coherence",
    ),
    "market_activity": (
        "volume_median_z",
        "volume_q75_z",
        "volume_participation_z1",
        "volume_delta_median_z",
        "value_median_z",
        "value_q75_z",
        "value_participation_z1",
        "value_delta_median_z",
    ),
    "node_state": (
        "common_state_energy",
        "node_state_median_energy",
        "node_state_q75_energy",
        "state_change_participation",
        "state_feature_breadth",
    ),
    "topology": (
        "market_corr_level",
        "market_corr_change",
        "state_coherence",
    ),
}


ABSOLUTE_COMPONENTS = frozenset(
    {
        "median_return",
        "return_breadth",
        "robust_return_coherence",
    }
)
# A correlation increase represents stronger market coupling. A decrease is
# directional because a large negative jump can be the deterministic expiry of
# an old shock from the 60-session rolling window rather than a new transition.


EVENT_NAMES = (
    "systemic_event",
    "price_transition",
    "broad_selloff",
    "activity_transition",
    "node_state_transition",
    "topology_transition",
)


@dataclass(frozen=True)
class MarketTransitionCalibration:
    component_center: dict[str, float]
    component_scale: dict[str, float]
    component_scale_quantile: float
    family_event_threshold: dict[str, float]
    family_event_quantile: float
    fit_event_rate: float
    broad_selloff_return_threshold: float
    broad_selloff_breadth_threshold: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return values[np.isfinite(values)]


def _distribution(values: np.ndarray, minimum: int) -> tuple[float, float]:
    values = _finite(values)
    if values.size < int(minimum):
        return float("nan"), float("nan")
    return float(np.median(values)), float(np.quantile(values, 0.75))


def _feature_activity(
    *,
    current_raw: np.ndarray,
    future_raw: np.ndarray,
    current_available: np.ndarray,
    future_available: np.ndarray,
    feature_names: Sequence[str],
    name: str,
    node_mask: np.ndarray,
    min_nodes: int,
) -> dict[str, float]:
    prefix = "volume" if name == "volume_z20" else "value"
    if name not in feature_names:
        return {
            f"{prefix}_median_z": float("nan"),
            f"{prefix}_q75_z": float("nan"),
            f"{prefix}_participation_z1": float("nan"),
            f"{prefix}_delta_median_z": float("nan"),
        }
    index = feature_names.index(name)
    future_valid = (
        node_mask
        & future_available[:, index]
        & np.isfinite(future_raw[:, index])
    )
    future_values = future_raw[future_valid, index]
    median, q75 = _distribution(future_values, min_nodes)
    participation = (
        float(np.mean(future_values >= 1.0))
        if future_values.size >= int(min_nodes)
        else float("nan")
    )
    paired = (
        node_mask
        & current_available[:, index]
        & future_available[:, index]
        & np.isfinite(current_raw[:, index])
        & np.isfinite(future_raw[:, index])
    )
    deltas = future_raw[paired, index] - current_raw[paired, index]
    delta_median = (
        float(np.median(deltas))
        if deltas.size >= int(min_nodes)
        else float("nan")
    )
    return {
        f"{prefix}_median_z": median,
        f"{prefix}_q75_z": q75,
        f"{prefix}_participation_z1": participation,
        f"{prefix}_delta_median_z": delta_median,
    }


def _raw_feature_median(
    raw: np.ndarray,
    available: np.ndarray,
    feature_names: Sequence[str],
    name: str,
    node_mask: np.ndarray,
    min_nodes: int,
) -> float:
    if name not in feature_names:
        return float("nan")
    index = feature_names.index(name)
    valid = node_mask & available[:, index] & np.isfinite(raw[:, index])
    if int(valid.sum()) < int(min_nodes):
        return float("nan")
    return float(np.median(raw[valid, index]))


def market_transition_components(
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
    state_change_threshold: float = 0.75,
    feature_change_threshold: float = 0.25,
) -> dict[str, float | int]:
    """Build a robust market-wide transition target.

    Every scored component needs broad cross-sectional participation. A single
    stock can affect diagnostics, but cannot create a target event by itself.
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
    path_values = path[node_mask & np.isfinite(path)]
    if path_values.size < int(min_nodes):
        path_values = np.empty(0, dtype=np.float64)
    if path_values.size:
        median_return = float(np.median(path_values))
        absolute = np.abs(path_values)
        median_absolute_return = float(np.median(absolute))
        q75_absolute_return = float(np.quantile(absolute, 0.75))
        return_breadth = float(np.sign(path_values).mean())
        robust_return_coherence = (
            float(np.clip(median_return / median_absolute_return, -1.0, 1.0))
            if median_absolute_return > 1e-12
            else 0.0
        )
        market_return = float(np.mean(path_values))
        squared = np.square(path_values)
        return_concentration = (
            float(squared.max() / squared.sum()) if squared.sum() > 1e-16 else 0.0
        )
    else:
        median_return = float("nan")
        median_absolute_return = float("nan")
        q75_absolute_return = float("nan")
        return_breadth = float("nan")
        robust_return_coherence = float("nan")
        market_return = float("nan")
        return_concentration = float("nan")

    feature_indices, selected_names = systemic_state_feature_indices(
        feature_names, state_feature_names
    )
    current_selected = current_state[:, feature_indices]
    future_selected = future_state[:, feature_indices]
    valid = (
        node_mask[:, None]
        & current_available[:, feature_indices]
        & future_available[:, feature_indices]
        & np.isfinite(current_selected)
        & np.isfinite(future_selected)
    )
    delta = future_selected - current_selected
    feature_counts = valid.sum(axis=0)
    usable_features = feature_counts >= int(min_nodes)
    # A cross-sectional mean can turn one extreme stock into an apparent
    # market-wide state shift.  Per-feature medians require broad node
    # participation while retaining the common direction of the transition.
    feature_centers = np.full(len(feature_indices), np.nan, dtype=np.float64)
    if usable_features.any():
        feature_centers[usable_features] = np.nanmedian(
            np.where(valid[:, usable_features], delta[:, usable_features], np.nan),
            axis=0,
        )
    common_state_energy = (
        float(np.sqrt(np.mean(np.square(feature_centers[usable_features]))))
        if usable_features.any()
        else float("nan")
    )
    state_feature_breadth = (
        float(
            np.mean(
                np.abs(feature_centers[usable_features])
                >= float(feature_change_threshold)
            )
        )
        if usable_features.any()
        else float("nan")
    )
    observed_delta = delta[valid]
    total_state_energy = (
        float(np.sqrt(np.mean(np.square(observed_delta))))
        if observed_delta.size
        else float("nan")
    )
    node_counts = valid.sum(axis=1)
    minimum_features = max(3, int(np.ceil(len(feature_indices) * 0.25)))
    usable_nodes = node_mask & (node_counts >= minimum_features)
    node_energy = np.full(current_state.shape[0], np.nan, dtype=np.float64)
    if usable_nodes.any():
        node_energy[usable_nodes] = np.sqrt(
            np.divide(
                np.where(valid[usable_nodes], np.square(delta[usable_nodes]), 0.0).sum(axis=1),
                node_counts[usable_nodes],
            )
        )
    finite_node_energy = _finite(node_energy)
    if finite_node_energy.size >= int(min_nodes):
        node_state_median_energy = float(np.median(finite_node_energy))
        node_state_q75_energy = float(np.quantile(finite_node_energy, 0.75))
        state_change_participation = float(
            np.mean(finite_node_energy >= float(state_change_threshold))
        )
    else:
        node_state_median_energy = float("nan")
        node_state_q75_energy = float("nan")
        state_change_participation = float("nan")
    if (
        np.isfinite(common_state_energy)
        and np.isfinite(total_state_energy)
        and total_state_energy > 1e-12
    ):
        state_coherence = float(common_state_energy / total_state_energy)
    elif np.isfinite(common_state_energy) and total_state_energy == 0.0:
        state_coherence = 0.0
    else:
        state_coherence = float("nan")

    activity = {}
    for name in ("volume_z20", "value_z20"):
        activity.update(
            _feature_activity(
                current_raw=current_raw,
                future_raw=future_raw,
                current_available=current_available,
                future_available=future_available,
                feature_names=feature_names,
                name=name,
                node_mask=node_mask,
                min_nodes=min_nodes,
            )
        )
    current_market_corr = _raw_feature_median(
        current_raw,
        current_available,
        feature_names,
        "market_corr_60d",
        node_mask,
        min_nodes,
    )
    market_corr_level = _raw_feature_median(
        future_raw,
        future_available,
        feature_names,
        "market_corr_60d",
        node_mask,
        min_nodes,
    )

    return {
        "observed_nodes": int(path_values.size),
        "state_features": int(len(selected_names)),
        "market_return": market_return,
        "median_return": median_return,
        "median_absolute_return": median_absolute_return,
        "q75_absolute_return": q75_absolute_return,
        "return_breadth": return_breadth,
        "robust_return_coherence": robust_return_coherence,
        "return_concentration": return_concentration,
        **activity,
        "common_state_energy": common_state_energy,
        "total_state_energy": total_state_energy,
        "node_state_median_energy": node_state_median_energy,
        "node_state_q75_energy": node_state_q75_energy,
        "state_change_participation": state_change_participation,
        "state_feature_breadth": state_feature_breadth,
        "state_coherence": state_coherence,
        "market_corr_level": market_corr_level,
        "market_corr_change": float(market_corr_level - current_market_corr),
    }


def _component_magnitude(name: str, value: float) -> float:
    if not np.isfinite(value):
        return float("nan")
    if name in ABSOLUTE_COMPONENTS:
        return abs(float(value))
    return max(float(value), 0.0)


def _robust_center_scale(
    values: Iterable[float], scale_quantile: float
) -> tuple[float, float]:
    values = _finite(np.asarray(list(values), dtype=np.float64))
    if values.size < 20:
        raise ValueError("at least twenty finite fit observations are required")
    center = float(np.median(values))
    scale = float(np.quantile(values, scale_quantile) - center)
    if scale <= 1e-12:
        scale = float(np.std(values))
    return center, scale if scale > 1e-12 else 1.0


def score_market_transition(
    row: Mapping[str, float | int],
    calibration: MarketTransitionCalibration,
) -> dict[str, float]:
    component_scores: dict[str, float] = {}
    for name, center in calibration.component_center.items():
        magnitude = _component_magnitude(name, float(row.get(name, float("nan"))))
        component_scores[name] = (
            float(
                np.clip(
                    (magnitude - float(center)) / calibration.component_scale[name],
                    0.0,
                    20.0,
                )
            )
            if np.isfinite(magnitude)
            else float("nan")
        )
    family_scores: dict[str, float] = {}
    for family, names in MARKET_TRANSITION_FAMILIES.items():
        values = _finite(
            np.asarray([component_scores.get(name, np.nan) for name in names])
        )
        family_scores[family] = (
            float(np.sqrt(np.mean(np.square(values))))
            if values.size == len(names)
            else float("nan")
        )
    finite_families = _finite(np.asarray(list(family_scores.values())))
    systemic_energy = (
        float(np.max(finite_families))
        if finite_families.size == len(MARKET_TRANSITION_FAMILIES)
        else float("nan")
    )
    return {
        **{f"component:{name}": value for name, value in component_scores.items()},
        **{f"family:{name}": value for name, value in family_scores.items()},
        "systemic_energy": systemic_energy,
    }


def fit_market_transition_calibration(
    rows: Sequence[Mapping[str, float | int]],
    *,
    component_scale_quantile: float = 0.90,
    family_event_quantile: float = 0.95,
) -> MarketTransitionCalibration:
    if not 0.75 <= float(component_scale_quantile) < 1.0:
        raise ValueError("component_scale_quantile must be in [0.75, 1.0)")
    if not 0.85 <= float(family_event_quantile) < 1.0:
        raise ValueError("family_event_quantile must be in [0.85, 1.0)")
    names = tuple(
        dict.fromkeys(
            name for family_names in MARKET_TRANSITION_FAMILIES.values() for name in family_names
        )
    )
    centers: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in names:
        values = [_component_magnitude(name, float(row.get(name, np.nan))) for row in rows]
        centers[name], scales[name] = _robust_center_scale(
            values, float(component_scale_quantile)
        )
    provisional = MarketTransitionCalibration(
        component_center=centers,
        component_scale=scales,
        component_scale_quantile=float(component_scale_quantile),
        family_event_threshold={},
        family_event_quantile=float(family_event_quantile),
        fit_event_rate=float("nan"),
        broad_selloff_return_threshold=float("nan"),
        broad_selloff_breadth_threshold=float("nan"),
    )
    scored = [score_market_transition(row, provisional) for row in rows]
    thresholds = {}
    for family in MARKET_TRANSITION_FAMILIES:
        values = _finite(np.asarray([row[f"family:{family}"] for row in scored]))
        if values.size < 20:
            raise ValueError(f"too few fit observations for family {family}")
        thresholds[family] = float(np.quantile(values, float(family_event_quantile)))

    def quantile(name: str, value: float) -> float:
        values = _finite(np.asarray([float(row.get(name, np.nan)) for row in rows]))
        if values.size < 20:
            return float("nan")
        return float(np.quantile(values, value))

    calibrated = MarketTransitionCalibration(
        component_center=centers,
        component_scale=scales,
        component_scale_quantile=float(component_scale_quantile),
        family_event_threshold=thresholds,
        family_event_quantile=float(family_event_quantile),
        fit_event_rate=float("nan"),
        broad_selloff_return_threshold=quantile("median_return", 0.10),
        broad_selloff_breadth_threshold=quantile("return_breadth", 0.10),
    )
    labels = [market_transition_labels(row, calibrated)["systemic_event"] for row in rows]
    return MarketTransitionCalibration(
        **{
            **calibrated.to_dict(),
            "fit_event_rate": float(np.mean(labels)),
        }
    )


def market_transition_labels(
    row: Mapping[str, float | int],
    calibration: MarketTransitionCalibration,
) -> dict[str, bool]:
    score = score_market_transition(row, calibration)
    family_events = {
        family: bool(
            np.isfinite(score[f"family:{family}"])
            and score[f"family:{family}"] >= calibration.family_event_threshold[family]
        )
        for family in MARKET_TRANSITION_FAMILIES
    }
    median_return = float(row.get("median_return", np.nan))
    breadth = float(row.get("return_breadth", np.nan))
    broad_selloff = bool(
        np.isfinite(median_return)
        and np.isfinite(breadth)
        and median_return <= calibration.broad_selloff_return_threshold
        and breadth <= calibration.broad_selloff_breadth_threshold
    )
    return {
        "systemic_event": bool(broad_selloff or any(family_events.values())),
        "price_transition": family_events["price_co_movement"],
        "broad_selloff": broad_selloff,
        "activity_transition": family_events["market_activity"],
        "node_state_transition": family_events["node_state"],
        "topology_transition": family_events["topology"],
    }


def normalized_market_transition_impact(
    row: Mapping[str, float | int],
    calibration: MarketTransitionCalibration,
) -> dict[str, float]:
    """Return a comparable broad-impact score across transition families.

    Family scores have different fit distributions, so their raw magnitudes
    cannot be compared directly.  Each family is divided by its fit-only event
    threshold.  A broad selloff receives an impact floor of one so the
    continuous score and the systemic-event OR contract remain aligned.
    """

    scored = score_market_transition(row, calibration)
    family_salience = {
        family: float(scored[f"family:{family}"])
        / max(float(calibration.family_event_threshold[family]), 1e-12)
        for family in MARKET_TRANSITION_FAMILIES
    }
    finite = _finite(np.asarray(list(family_salience.values()), dtype=np.float64))
    if finite.size != len(MARKET_TRANSITION_FAMILIES):
        systemic_impact = float("nan")
    else:
        broad_selloff = market_transition_labels(row, calibration)["broad_selloff"]
        systemic_impact = float(max(float(np.max(finite)), float(broad_selloff)))
    return {
        **{
            f"normalized_family:{family}": value
            for family, value in family_salience.items()
        },
        "systemic_impact": systemic_impact,
    }


def transition_signature_metrics(
    actual_rows: Sequence[Mapping[str, float | int]],
    predicted_rows: Sequence[Mapping[str, float | int]],
    calibration: MarketTransitionCalibration,
) -> dict[str, float | int]:
    if len(actual_rows) != len(predicted_rows):
        raise ValueError("actual and predicted rows must align")
    actual_scores = [score_market_transition(row, calibration) for row in actual_rows]
    predicted_scores = [score_market_transition(row, calibration) for row in predicted_rows]
    labels = np.asarray(
        [market_transition_labels(row, calibration)["systemic_event"] for row in actual_rows],
        dtype=bool,
    )
    actual_energy = np.asarray(
        [
            normalized_market_transition_impact(row, calibration)[
                "systemic_impact"
            ]
            for row in actual_rows
        ]
    )
    predicted_energy = np.asarray(
        [
            normalized_market_transition_impact(row, calibration)[
                "systemic_impact"
            ]
            for row in predicted_rows
        ]
    )
    valid = np.isfinite(actual_energy) & np.isfinite(predicted_energy)
    ranking = binary_ranking_metrics(
        labels[valid], predicted_energy[valid], selection_rate=calibration.fit_event_rate
    )
    family_actual = np.asarray(
        [[row[f"family:{name}"] for name in MARKET_TRANSITION_FAMILIES] for row in actual_scores],
        dtype=np.float64,
    )
    family_predicted = np.asarray(
        [[row[f"family:{name}"] for name in MARKET_TRANSITION_FAMILIES] for row in predicted_scores],
        dtype=np.float64,
    )
    signature_valid = labels & np.isfinite(family_actual).all(axis=1) & np.isfinite(
        family_predicted
    ).all(axis=1)
    if signature_valid.any():
        actual = family_actual[signature_valid]
        predicted = family_predicted[signature_valid]
        numerator = np.sum(actual * predicted, axis=1)
        denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(predicted, axis=1)
        cosine = np.divide(
            numerator,
            denominator,
            out=np.full_like(numerator, np.nan),
            where=denominator > 1e-12,
        )
        mean_cosine = float(np.nanmean(cosine))
        dominant_accuracy = float(
            np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))
        )
    else:
        mean_cosine = float("nan")
        dominant_accuracy = float("nan")
    impact_correlation = (
        float(np.corrcoef(actual_energy[valid], predicted_energy[valid])[0, 1])
        if int(valid.sum()) >= 2
        and np.std(actual_energy[valid]) > 0
        and np.std(predicted_energy[valid]) > 0
        else float("nan")
    )
    return {
        **ranking,
        "systemic_impact_correlation": impact_correlation,
        "systemic_energy_correlation": impact_correlation,
        "event_transition_signature_cosine": mean_cosine,
        "event_dominant_family_accuracy": dominant_accuracy,
    }
