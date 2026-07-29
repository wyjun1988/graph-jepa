from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from stock_v2.graph_jepa import GraphBatch
from stock_v2.market_transition import (
    MARKET_TRANSITION_FAMILIES,
    MARKET_TRANSITION_TARGET_VERSION,
    MarketTransitionCalibration,
    fit_market_transition_calibration,
    market_transition_components,
    market_transition_labels,
    score_market_transition,
)


MARKET_TRANSITION_AUXILIARY_FAMILIES = tuple(MARKET_TRANSITION_FAMILIES)
MARKET_TRANSITION_AUXILIARY_WIDTH = 2 * len(
    MARKET_TRANSITION_AUXILIARY_FAMILIES
) + 2


@dataclass(frozen=True)
class MarketTransitionAuxiliaryTargets:
    by_horizon: dict[int, dict[int, np.ndarray]]
    calibration: dict[int, MarketTransitionCalibration]
    fit_rows: dict[int, int]
    fit_family_event_rate: dict[int, list[float]]
    fit_broad_selloff_rate: dict[int, float]
    fit_systemic_event_rate: dict[int, float]

    def contract_dict(self) -> dict[str, Any]:
        return {
            "target_version": MARKET_TRANSITION_TARGET_VERSION,
            "family_names": list(MARKET_TRANSITION_AUXILIARY_FAMILIES),
            "target_layout": [
                *[
                    f"log_normalized_salience:{name}"
                    for name in MARKET_TRANSITION_AUXILIARY_FAMILIES
                ],
                *[
                    f"event:{name}"
                    for name in MARKET_TRANSITION_AUXILIARY_FAMILIES
                ],
                "broad_selloff",
                "systemic_event",
            ],
            "horizons": {
                str(horizon): {
                    "fit_rows": int(self.fit_rows[horizon]),
                    "fit_family_event_rate": self.fit_family_event_rate[horizon],
                    "fit_broad_selloff_rate": float(
                        self.fit_broad_selloff_rate[horizon]
                    ),
                    "fit_systemic_event_rate": float(
                        self.fit_systemic_event_rate[horizon]
                    ),
                    "calibration": self.calibration[horizon].to_dict(),
                }
                for horizon in sorted(self.calibration)
            },
            "stock_nodes_only": True,
            "individual_node_maximum_used": False,
            "training_weight_contract": {
                "formula": "1 + 3 * min(systemic_impact, 3)",
                "systemic_impact": (
                    "max(family_salience, broad_selloff)"
                ),
                "weighted_terms": [
                    "family_regression",
                    "family_event",
                    "broad_selloff",
                    "systemic_event",
                ],
            },
            "live_orders_allowed": False,
        }


def market_transition_auxiliary_target(
    row: dict[str, float | int],
    calibration: MarketTransitionCalibration,
) -> np.ndarray:
    thresholds = np.asarray(
        [
            calibration.family_event_threshold[name]
            for name in MARKET_TRANSITION_AUXILIARY_FAMILIES
        ],
        dtype=np.float64,
    )
    if not np.isfinite(thresholds).all() or (thresholds <= 0.0).any():
        raise ValueError("market transition family thresholds must be positive")
    scored = score_market_transition(row, calibration)
    family = np.asarray(
        [
            scored[f"family:{name}"]
            for name in MARKET_TRANSITION_AUXILIARY_FAMILIES
        ],
        dtype=np.float64,
    )
    if not np.isfinite(family).all():
        raise ValueError("market transition family targets must be finite")
    salience = np.maximum(family / thresholds, 0.0)
    family_event = salience >= 1.0
    labels = market_transition_labels(row, calibration)
    return np.concatenate(
        (
            np.log1p(salience),
            family_event.astype(np.float64),
            np.asarray(
                [labels["broad_selloff"], labels["systemic_event"]],
                dtype=np.float64,
            ),
        )
    ).astype(np.float32)


def apply_market_transition_auxiliary_calibration(
    features,
    steps: Sequence[int] | np.ndarray,
    calibration: Mapping[int, MarketTransitionCalibration],
) -> dict[int, dict[int, np.ndarray]]:
    context_steps = np.asarray(steps, dtype=np.int64)
    output: dict[int, dict[int, np.ndarray]] = {}
    for raw_horizon, contract in calibration.items():
        horizon = int(raw_horizon)
        output[horizon] = {
            int(step): market_transition_auxiliary_target(
                _transition_row(features, int(step), horizon), contract
            )
            for step in context_steps
        }
    return output


def _transition_row(features, step: int, horizon: int) -> dict[str, float | int]:
    stock_count = int(features.tradable_count)
    target_step = int(step) + int(horizon)
    if target_step >= len(features.dates):
        raise ValueError("market transition target exceeds the available panel")
    return_index = features.feature_names.index("return_1d")
    current_available = features.available_mask[int(step), :stock_count] > 0.5
    future_available = features.available_mask[target_step, :stock_count] > 0.5
    path = np.asarray(
        features.target_return_paths[int(horizon)][int(step), :stock_count],
        dtype=np.float64,
    )
    node_mask = (
        current_available[:, return_index]
        & future_available[:, return_index]
        & np.isfinite(path)
    )
    return market_transition_components(
        current_state=features.features[int(step), :stock_count],
        future_state=features.features[target_step, :stock_count],
        current_raw=features.raw_features[int(step), :stock_count],
        future_raw=features.raw_features[target_step, :stock_count],
        current_available=current_available,
        future_available=future_available,
        feature_names=features.feature_names,
        entry_path_returns=path,
        node_mask=node_mask,
    )


def build_market_transition_auxiliary_targets(
    features,
    steps: Sequence[int] | np.ndarray,
    horizons: Sequence[int],
    *,
    component_scale_quantile: float = 0.90,
    family_event_quantile: float = 0.95,
) -> MarketTransitionAuxiliaryTargets:
    context_steps = np.asarray(steps, dtype=np.int64)
    normalized_horizons = tuple(sorted({int(value) for value in horizons}))
    if context_steps.size < 20:
        raise ValueError("at least twenty training contexts are required")
    if not normalized_horizons or normalized_horizons[0] < 1:
        raise ValueError("market transition horizons must be positive")

    by_horizon: dict[int, dict[int, np.ndarray]] = {}
    calibrations: dict[int, MarketTransitionCalibration] = {}
    fit_rows: dict[int, int] = {}
    family_rates: dict[int, list[float]] = {}
    broad_selloff_rates: dict[int, float] = {}
    systemic_rates: dict[int, float] = {}
    for horizon in normalized_horizons:
        rows = [
            _transition_row(features, int(step), horizon)
            for step in context_steps
        ]
        calibration = fit_market_transition_calibration(
            rows,
            component_scale_quantile=float(component_scale_quantile),
            family_event_quantile=float(family_event_quantile),
        )
        targets: dict[int, np.ndarray] = {}
        labels = []
        broad_selloff_labels = []
        systemic_labels = []
        for step, row in zip(context_steps, rows):
            target = market_transition_auxiliary_target(row, calibration)
            family_count = len(MARKET_TRANSITION_AUXILIARY_FAMILIES)
            family_event = target[family_count : 2 * family_count].astype(bool)
            targets[int(step)] = target
            labels.append(family_event)
            broad_selloff_labels.append(bool(target[-2]))
            systemic_labels.append(bool(target[-1]))
        label_matrix = np.asarray(labels, dtype=bool)
        by_horizon[horizon] = targets
        calibrations[horizon] = calibration
        fit_rows[horizon] = len(rows)
        family_rates[horizon] = label_matrix.mean(axis=0).astype(float).tolist()
        broad_selloff_rates[horizon] = float(np.mean(broad_selloff_labels))
        systemic_rates[horizon] = float(np.mean(systemic_labels))
    return MarketTransitionAuxiliaryTargets(
        by_horizon=by_horizon,
        calibration=calibrations,
        fit_rows=fit_rows,
        fit_family_event_rate=family_rates,
        fit_broad_selloff_rate=broad_selloff_rates,
        fit_systemic_event_rate=systemic_rates,
    )


def attach_market_transition_auxiliary_targets(
    batch: GraphBatch,
    targets: MarketTransitionAuxiliaryTargets,
    context_steps: Sequence[int] | np.ndarray,
    horizon: int,
) -> GraphBatch:
    steps = np.asarray(context_steps, dtype=np.int64)
    horizon = int(horizon)
    if horizon not in targets.by_horizon:
        raise ValueError(f"missing market transition target horizon {horizon}")
    graph_count = (
        int(batch.graph_index.max().item()) + 1
        if batch.graph_index is not None and batch.graph_index.numel()
        else 1
    )
    if graph_count != len(steps):
        raise ValueError("market transition targets do not align with graph batch")
    lookup = targets.by_horizon[horizon]
    try:
        values = np.stack([lookup[int(step)] for step in steps], axis=0)
    except KeyError as error:
        raise ValueError(
            f"missing market transition target for context step {error.args[0]}"
        ) from error
    if values.shape != (len(steps), MARKET_TRANSITION_AUXILIARY_WIDTH):
        raise ValueError("invalid market transition auxiliary target shape")
    batch.target_market_transition = torch.from_numpy(values.astype(np.float32))
    return batch
