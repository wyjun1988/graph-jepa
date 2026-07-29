from __future__ import annotations

import argparse
import copy
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from scripts.audit_market_transition_targets import _actual_rows
from scripts.audit_systemic_transition_targets import _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_latent_trajectory_path_head import (
    HORIZON_WEIGHTS,
    checkpoint_sha256,
    latent_trajectories,
    snapshot_batch,
)
from scripts.benchmark_systemic_transition_head import configured_horizon_text
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    pearson,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import parse_int_list
from stock_v2.market_transition import (
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
    MarketTransitionCalibration,
    binary_ranking_metrics,
    fit_market_transition_calibration,
    market_transition_labels,
    score_market_transition,
)
from stock_v2.market_transition_head import (
    FamilyQueryMarketTrajectoryHead,
    MARKET_COMPONENT_TARGETS,
    MARKET_EVENT_TARGETS,
    MARKET_FAMILY_TARGETS,
    MarketTrajectoryHead,
    trajectory_difference_loss,
    weighted_masked_smooth_l1,
)
from stock_v2.systemic_head import correlation_rank_loss


LOSS_WEIGHTS = {
    "components": 0.25,
    "families": 0.30,
    "family_rank": 0.15,
    "events": 0.20,
    "trajectory": 0.10,
}


FAMILY_EVENT_NAME = {
    "price_co_movement": "price_transition",
    "market_activity": "activity_transition",
    "node_state": "node_state_transition",
    "topology": "topology_transition",
}


def _require_finite_tensor(name: str, value: torch.Tensor) -> None:
    finite = torch.isfinite(value)
    if bool(finite.all()):
        return
    nonfinite = int((~finite).sum().detach().cpu())
    total = int(value.numel())
    raise FloatingPointError(f"{name} contains {nonfinite}/{total} non-finite values")


@dataclass(frozen=True)
class HorizonMarketContract:
    calibration: MarketTransitionCalibration
    component_mean: np.ndarray
    component_std: np.ndarray
    sample_weight_mean: float
    event_pos_weight: np.ndarray
    event_fit_rate: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration.to_dict(),
            "component_names": list(MARKET_COMPONENT_TARGETS),
            "family_names": list(MARKET_FAMILY_TARGETS),
            "event_names": list(MARKET_EVENT_TARGETS),
            "component_mean": self.component_mean.tolist(),
            "component_std": self.component_std.tolist(),
            "sample_weight_mean": float(self.sample_weight_mean),
            "event_pos_weight": self.event_pos_weight.tolist(),
            "event_fit_rate": self.event_fit_rate.tolist(),
        }


def _subsample(steps: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or len(steps) <= int(maximum):
        return np.asarray(steps, dtype=np.int64)
    positions = np.linspace(0, len(steps) - 1, int(maximum)).round().astype(np.int64)
    return np.asarray(steps, dtype=np.int64)[positions]


def _rows_lookup(rows):
    return {
        (int(row["step"]), int(row["horizon"])): row
        for row in rows
    }


def _component_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [float(row.get(name, np.nan)) for name in MARKET_COMPONENT_TARGETS]
            for row in rows
        ],
        dtype=np.float64,
    )


def _family_vector(row, calibration):
    score = score_market_transition(row, calibration)
    return np.asarray(
        [float(score[f"family:{name}"]) for name in MARKET_FAMILY_TARGETS],
        dtype=np.float64,
    )


def build_target_contracts(fit_rows, horizons):
    contracts = {}
    for horizon in horizons:
        selected = [row for row in fit_rows if int(row["horizon"]) == int(horizon)]
        calibration = fit_market_transition_calibration(
            selected,
            component_scale_quantile=0.90,
            family_event_quantile=0.95,
        )
        component = _component_matrix(selected)
        finite = np.isfinite(component)
        count = finite.sum(axis=0)
        means = np.divide(
            np.where(finite, component, 0.0).sum(axis=0),
            count,
            out=np.zeros(component.shape[1], dtype=np.float64),
            where=count > 0,
        )
        centered = np.where(finite, component - means[None, :], 0.0)
        stds = np.sqrt(
            np.divide(
                np.square(centered).sum(axis=0),
                count,
                out=np.ones(component.shape[1], dtype=np.float64),
                where=count > 0,
            )
        )
        stds = np.where(np.isfinite(stds) & (stds > 1e-8), stds, 1.0)
        family = np.stack(
            [_family_vector(row, calibration) for row in selected], axis=0
        )
        thresholds = np.asarray(
            [calibration.family_event_threshold[name] for name in MARKET_FAMILY_TARGETS],
            dtype=np.float64,
        )
        salience = np.nanmax(family / np.maximum(thresholds[None, :], 1e-8), axis=1)
        labels = np.asarray(
            [
                [
                    float(market_transition_labels(row, calibration)[name])
                    for name in MARKET_EVENT_TARGETS
                ]
                for row in selected
            ],
            dtype=np.float64,
        )
        selloff_index = MARKET_EVENT_TARGETS.index("broad_selloff")
        systemic_impact = np.maximum(salience, labels[:, selloff_index])
        raw_weight = 1.0 + 3.0 * np.minimum(systemic_impact, 3.0)
        positives = labels.sum(axis=0)
        negatives = len(labels) - positives
        pos_weight = np.clip(negatives / np.maximum(positives, 1.0), 1.0, 20.0)
        contracts[int(horizon)] = HorizonMarketContract(
            calibration=calibration,
            component_mean=means.astype(np.float32),
            component_std=stds.astype(np.float32),
            sample_weight_mean=float(np.mean(raw_weight)),
            event_pos_weight=pos_weight.astype(np.float32),
            event_fit_rate=labels.mean(axis=0).astype(np.float32),
        )
    return contracts


def build_target_arrays(rows, steps, horizons, contracts):
    lookup = _rows_lookup(rows)
    batch = len(steps)
    component = np.zeros(
        (batch, len(horizons), len(MARKET_COMPONENT_TARGETS)), dtype=np.float32
    )
    component_valid = np.zeros_like(component, dtype=bool)
    family_log = np.zeros(
        (batch, len(horizons), len(MARKET_FAMILY_TARGETS)), dtype=np.float32
    )
    family_valid = np.zeros_like(family_log, dtype=bool)
    labels = np.zeros(
        (batch, len(horizons), len(MARKET_EVENT_TARGETS)), dtype=np.float32
    )
    sample_weight = np.ones((batch, len(horizons)), dtype=np.float32)
    selected_rows: list[list[Mapping[str, Any]]] = []
    for position, step in enumerate(steps):
        date_rows = []
        for horizon_index, horizon in enumerate(horizons):
            row = lookup[(int(step), int(horizon))]
            date_rows.append(row)
            contract = contracts[int(horizon)]
            raw_component = _component_matrix([row])[0]
            valid = np.isfinite(raw_component)
            component[position, horizon_index] = np.where(
                valid,
                (raw_component - contract.component_mean) / contract.component_std,
                0.0,
            )
            component_valid[position, horizon_index] = valid
            family = _family_vector(row, contract.calibration)
            family_valid[position, horizon_index] = np.isfinite(family)
            family_log[position, horizon_index] = np.where(
                np.isfinite(family), np.log1p(np.maximum(family, 0.0)), 0.0
            )
            row_labels = market_transition_labels(row, contract.calibration)
            labels[position, horizon_index] = [
                float(row_labels[name]) for name in MARKET_EVENT_TARGETS
            ]
            thresholds = np.asarray(
                [
                    contract.calibration.family_event_threshold[name]
                    for name in MARKET_FAMILY_TARGETS
                ],
                dtype=np.float64,
            )
            salience = float(np.nanmax(family / np.maximum(thresholds, 1e-8)))
            row_labels = market_transition_labels(row, contract.calibration)
            systemic_impact = max(salience, float(row_labels["broad_selloff"]))
            raw_weight = 1.0 + 3.0 * min(systemic_impact, 3.0)
            horizon_weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
            sample_weight[position, horizon_index] = (
                raw_weight
                / max(float(contract.sample_weight_mean), 1e-8)
                * horizon_weight
            )
        selected_rows.append(date_rows)
    return {
        "rows": selected_rows,
        "components": component,
        "component_valid": component_valid,
        "family_log": family_log,
        "family_valid": family_valid,
        "labels": labels,
        "sample_weight": sample_weight,
    }


def _target_batch(targets, positions, device):
    index = np.asarray(positions, dtype=np.int64)
    return {
        "components": torch.as_tensor(targets["components"][index], device=device),
        "component_valid": torch.as_tensor(
            targets["component_valid"][index], dtype=torch.bool, device=device
        ),
        "family_log": torch.as_tensor(targets["family_log"][index], device=device),
        "family_valid": torch.as_tensor(
            targets["family_valid"][index], dtype=torch.bool, device=device
        ),
        "labels": torch.as_tensor(targets["labels"][index], device=device),
        "sample_weight": torch.as_tensor(
            targets["sample_weight"][index], device=device
        ),
    }


def _family_rank_loss(prediction, target, valid):
    losses = []
    for horizon_index in range(prediction.shape[1]):
        for family_index in range(prediction.shape[2]):
            selected = valid[:, horizon_index, family_index]
            if int(selected.sum()) >= 3:
                losses.append(
                    correlation_rank_loss(
                        prediction[selected, horizon_index, family_index],
                        target[selected, horizon_index, family_index],
                    )
                )
    return torch.stack(losses).mean() if losses else prediction.new_tensor(0.0)


def _weighted_focal_binary_loss(logits, labels, sample_weight):
    labels = labels.to(dtype=logits.dtype)
    probability = torch.sigmoid(logits)
    target_probability = torch.where(labels > 0.5, probability, 1.0 - probability)
    alpha = torch.where(
        labels > 0.5,
        torch.full_like(labels, 0.75),
        torch.full_like(labels, 0.25),
    )
    loss = (
        alpha
        * (1.0 - target_probability).square()
        * F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    )
    weight = sample_weight.to(dtype=loss.dtype)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def _event_loss(logits, labels, contracts, horizons, sample_weight):
    systemic = _weighted_focal_binary_loss(
        logits[..., 0], labels[..., 0], sample_weight
    )
    subtype_losses = []
    for index, horizon in enumerate(horizons):
        pos_weight = torch.as_tensor(
            contracts[int(horizon)].event_pos_weight[1:],
            dtype=logits.dtype,
            device=logits.device,
        )
        subtype = F.binary_cross_entropy_with_logits(
            logits[:, index, 1:],
            labels[:, index, 1:],
            pos_weight=pos_weight,
            reduction="none",
        )
        weight = sample_weight[:, index, None].to(dtype=subtype.dtype)
        subtype_losses.append(
            (subtype * weight).sum()
            / (weight.sum() * subtype.shape[-1]).clamp_min(1e-8)
        )
    return 0.40 * systemic + 0.60 * torch.stack(subtype_losses).mean()


def loss_terms(predictions, target, contracts, horizons):
    components, families, events = predictions
    terms = {
        "components": weighted_masked_smooth_l1(
            components,
            target["components"],
            target["component_valid"],
            target["sample_weight"],
        ),
        "families": weighted_masked_smooth_l1(
            families,
            target["family_log"],
            target["family_valid"],
            target["sample_weight"],
        ),
        "family_rank": _family_rank_loss(
            families, target["family_log"], target["family_valid"]
        ),
        "events": _event_loss(
            events,
            target["labels"],
            contracts,
            horizons,
            target["sample_weight"],
        ),
        "trajectory": trajectory_difference_loss(
            families,
            target["family_log"],
            target["family_valid"],
            target["sample_weight"],
            horizons,
        ),
    }
    loss = sum(LOSS_WEIGHTS[name] * terms[name] for name in LOSS_WEIGHTS)
    return loss, terms


def train_epoch(
    model,
    head,
    features,
    steps,
    targets,
    contracts,
    horizons,
    checkpoint_args,
    feature_args,
    edge_cache,
    optimizer,
    device,
    batch_size,
    seed,
):
    head.train()
    step_to_position = {int(step): index for index, step in enumerate(steps)}
    shuffled = np.random.default_rng(seed).permutation(steps)
    losses = []
    history = {name: [] for name in LOSS_WEIGHTS}
    for start in range(0, len(shuffled), int(batch_size)):
        selected_steps = np.asarray(
            shuffled[start : start + int(batch_size)], dtype=np.int64
        )
        positions = [step_to_position[int(step)] for step in selected_steps]
        batch = snapshot_batch(
            features, selected_steps, checkpoint_args, feature_args, edge_cache, device
        )
        with torch.no_grad():
            context, predicted = latent_trajectories(
                model, batch, horizons, checkpoint_args
            )
        output = head(
            context,
            predicted,
            batch_size=len(selected_steps),
            node_count=features.node_count,
            stock_count=features.tradable_count,
        )
        for name, value in zip(("components", "families", "events"), output):
            _require_finite_tensor(f"train_{name}", value)
        target = _target_batch(targets, positions, device)
        loss, terms = loss_terms(output, target, contracts, horizons)
        _require_finite_tensor("train_loss", loss)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        for name, parameter in head.named_parameters():
            if parameter.grad is not None:
                _require_finite_tensor(f"gradient:{name}", parameter.grad)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for name, value in terms.items():
            history[name].append(float(value.detach().cpu()))
    return float(np.mean(losses)), {
        name: float(np.mean(values)) for name, values in history.items()
    }


def predict_steps(
    model,
    head,
    features,
    steps,
    targets,
    contracts,
    horizons,
    checkpoint_args,
    feature_args,
    edge_cache,
    device,
    batch_size,
):
    head.eval()
    output = {int(horizon): [] for horizon in horizons}
    for start in range(0, len(steps), int(batch_size)):
        selected_steps = np.asarray(steps[start : start + int(batch_size)], dtype=np.int64)
        batch = snapshot_batch(
            features, selected_steps, checkpoint_args, feature_args, edge_cache, device
        )
        with torch.no_grad():
            context, predicted = latent_trajectories(
                model, batch, horizons, checkpoint_args
            )
            normalized_component, family_log, event_logits = head(
                context,
                predicted,
                batch_size=len(selected_steps),
                node_count=features.node_count,
                stock_count=features.tradable_count,
            )
            _require_finite_tensor("predict_components", normalized_component)
            _require_finite_tensor("predict_families", family_log)
            _require_finite_tensor("predict_events", event_logits)
        component_numpy = normalized_component.float().cpu().numpy()
        family_numpy = np.maximum(
            np.expm1(np.clip(family_log.float().cpu().numpy(), -5.0, 5.0)), 0.0
        )
        event_numpy = event_logits.float().cpu().numpy()
        for horizon_index, horizon in enumerate(horizons):
            contract = contracts[int(horizon)]
            raw_component = (
                component_numpy[:, horizon_index] * contract.component_std[None, :]
                + contract.component_mean[None, :]
            )
            for position, step in enumerate(selected_steps):
                output[int(horizon)].append(
                    {
                        "step": int(step),
                        "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                        "horizon": int(horizon),
                        "actual": targets["rows"][start + position][horizon_index],
                        "predicted": {
                            name: float(raw_component[position, component_index])
                            for component_index, name in enumerate(
                                MARKET_COMPONENT_TARGETS
                            )
                        },
                        "predicted_families": family_numpy[
                            position, horizon_index
                        ].tolist(),
                        "event_logits": event_numpy[position, horizon_index].tolist(),
                    }
                )
    return output


def _ranking_with_mass(labels, scores, actual_impact, selection_rate):
    metrics = binary_ranking_metrics(labels, scores, selection_rate=selection_rate)
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    actual_impact = np.asarray(actual_impact, dtype=np.float64)
    selected_count = int(metrics["selected_count"])
    selected = np.argsort(scores, kind="mergesort")[-selected_count:]
    total_mass = float(np.sum(actual_impact[labels]))
    captured_mass = float(np.sum(actual_impact[selected][labels[selected]]))
    metrics.update(
        {
            "fit_selection_rate": float(selection_rate),
            "average_precision_lift": (
                float(metrics["average_precision"]) / float(metrics["event_rate"])
                if float(metrics["event_rate"]) > 0.0
                else float("nan")
            ),
            "systemic_impact_mass_recall_at_fit_rate": (
                float(captured_mass / total_mass)
                if total_mass > 1e-12
                else float("nan")
            ),
        }
    )
    return metrics


def horizon_metrics(records, contract):
    calibration = contract.calibration
    actual_family = np.stack(
        [_family_vector(row["actual"], calibration) for row in records], axis=0
    )
    predicted_family = np.asarray(
        [row["predicted_families"] for row in records], dtype=np.float64
    )
    thresholds = np.asarray(
        [
            calibration.family_event_threshold[name]
            for name in MARKET_FAMILY_TARGETS
        ],
        dtype=np.float64,
    )
    actual_normalized = actual_family / np.maximum(thresholds[None, :], 1e-8)
    predicted_normalized = predicted_family / np.maximum(
        thresholds[None, :], 1e-8
    )
    labels = np.asarray(
        [
            market_transition_labels(row["actual"], calibration)["systemic_event"]
            for row in records
        ],
        dtype=bool,
    )
    selloff_labels = np.asarray(
        [
            market_transition_labels(row["actual"], calibration)["broad_selloff"]
            for row in records
        ],
        dtype=bool,
    )
    actual_impact = np.maximum(
        np.max(actual_normalized, axis=1), selloff_labels.astype(np.float64)
    )
    predicted_impact = np.max(predicted_normalized, axis=1)
    overall = _ranking_with_mass(
        labels,
        predicted_impact,
        actual_impact,
        float(calibration.fit_event_rate),
    )
    overall["systemic_impact_correlation"] = pearson(
        predicted_impact, actual_impact
    )
    family_metrics = {}
    family_correlations = {}
    for family_index, family in enumerate(MARKET_FAMILY_TARGETS):
        event_name = FAMILY_EVENT_NAME[family]
        event_index = MARKET_EVENT_TARGETS.index(event_name)
        family_labels = np.asarray(
            [
                market_transition_labels(row["actual"], calibration)[event_name]
                for row in records
            ],
            dtype=bool,
        )
        scores = np.asarray(
            [float(row["event_logits"][event_index]) for row in records]
        )
        family_metrics[family] = binary_ranking_metrics(
            family_labels,
            scores,
            selection_rate=max(float(contract.event_fit_rate[event_index]), 1e-6),
        )
        family_correlations[family] = pearson(
            predicted_family[:, family_index], actual_family[:, family_index]
        )
    selloff_index = MARKET_EVENT_TARGETS.index("broad_selloff")
    selloff = binary_ranking_metrics(
        selloff_labels,
        [float(row["event_logits"][selloff_index]) for row in records],
        selection_rate=max(float(contract.event_fit_rate[selloff_index]), 1e-6),
    )
    event_valid = labels & np.isfinite(actual_family).all(axis=1) & np.isfinite(
        predicted_family
    ).all(axis=1)
    if event_valid.any():
        actual = actual_family[event_valid]
        predicted = predicted_family[event_valid]
        denominator = np.linalg.norm(actual, axis=1) * np.linalg.norm(
            predicted, axis=1
        )
        cosine = np.divide(
            np.sum(actual * predicted, axis=1),
            denominator,
            out=np.full(len(actual), np.nan),
            where=denominator > 1e-12,
        )
        finite_cosine = cosine[np.isfinite(cosine)]
        signature_cosine = (
            float(finite_cosine.mean()) if finite_cosine.size else float("nan")
        )
        dominant_accuracy = float(
            np.mean(np.argmax(actual, axis=1) == np.argmax(predicted, axis=1))
        )
    else:
        signature_cosine = float("nan")
        dominant_accuracy = float("nan")
    return {
        "systemic_event": overall,
        "family_events": family_metrics,
        "family_intensity_correlation": family_correlations,
        "broad_selloff": selloff,
        "event_transition_signature_cosine": signature_cosine,
        "event_dominant_family_accuracy": dominant_accuracy,
    }


def trajectory_metrics(predictions, contracts, horizons, fit_event_rate):
    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for horizon in horizons:
        for row in predictions[int(horizon)]:
            by_date.setdefault(str(row["date"]), []).append(row)
    labels = []
    scores = []
    peak_matches = []
    correlations = []
    signature_cosines = []
    actual_path_impacts = []
    for rows in by_date.values():
        rows = sorted(rows, key=lambda row: horizons.index(int(row["horizon"])))
        actual = []
        predicted = []
        for row in rows:
            horizon = int(row["horizon"])
            calibration = contracts[horizon].calibration
            thresholds = np.asarray(
                [
                    calibration.family_event_threshold[name]
                    for name in MARKET_FAMILY_TARGETS
                ]
            )
            actual.append(_family_vector(row["actual"], calibration) / thresholds)
            predicted.append(
                np.asarray(row["predicted_families"], dtype=np.float64) / thresholds
            )
        actual = np.asarray(actual, dtype=np.float64)
        predicted = np.asarray(predicted, dtype=np.float64)
        actual_peak = np.max(actual, axis=1)
        predicted_peak = np.max(predicted, axis=1)
        horizon_labels = [
            market_transition_labels(
                row["actual"], contracts[int(row["horizon"])].calibration
            )
            for row in rows
        ]
        selloff = np.asarray(
            [label["broad_selloff"] for label in horizon_labels], dtype=bool
        )
        actual_peak = np.maximum(actual_peak, selloff.astype(np.float64))
        labels.append(any(label["systemic_event"] for label in horizon_labels))
        actual_path_impacts.append(float(np.max(actual_peak)))
        scores.append(float(np.max(predicted_peak)))
        correlations.append(pearson(predicted.reshape(-1), actual.reshape(-1)))
        denominator = np.linalg.norm(actual.reshape(-1)) * np.linalg.norm(
            predicted.reshape(-1)
        )
        signature_cosines.append(
            float(np.sum(actual * predicted) / denominator)
            if denominator > 1e-12
            else float("nan")
        )
        if labels[-1]:
            peak_matches.append(int(np.argmax(predicted_peak)) == int(np.argmax(actual_peak)))
    ranking = _ranking_with_mass(
        labels,
        scores,
        actual_path_impacts,
        selection_rate=min(float(fit_event_rate), 1.0),
    )
    finite_correlations = np.asarray(correlations, dtype=np.float64)
    finite_correlations = finite_correlations[np.isfinite(finite_correlations)]
    finite_signatures = np.asarray(signature_cosines, dtype=np.float64)
    finite_signatures = finite_signatures[np.isfinite(finite_signatures)]
    ranking.update(
        {
            "fit_event_rate": float(fit_event_rate),
            "peak_horizon_accuracy_on_events": (
                float(np.mean(peak_matches)) if peak_matches else float("nan")
            ),
            "mean_family_trajectory_correlation": (
                float(finite_correlations.mean())
                if finite_correlations.size
                else float("nan")
            ),
            "mean_transition_signature_cosine": (
                float(finite_signatures.mean())
                if finite_signatures.size
                else float("nan")
            ),
        }
    )
    return ranking


def fit_trajectory_event_rate(targets):
    return float(np.mean(np.any(targets["labels"][..., 0] > 0.5, axis=1)))


def validation_score(metrics, trajectory, horizons):
    def finite_or(value, fallback):
        value = float(value)
        return value if math.isfinite(value) else float(fallback)

    weighted = []
    weights = []
    for horizon in horizons:
        item = metrics[str(horizon)]
        overall = item["systemic_event"]
        family_auc_values = np.asarray(
            [value["roc_auc"] for value in item["family_events"].values()],
            dtype=np.float64,
        )
        family_auc_values = family_auc_values[np.isfinite(family_auc_values)]
        family_auc = (
            float(family_auc_values.mean()) if family_auc_values.size else 0.5
        )
        family_corr_values = np.asarray(
            list(item["family_intensity_correlation"].values()), dtype=np.float64
        )
        family_corr_values = family_corr_values[np.isfinite(family_corr_values)]
        family_corr = (
            float(family_corr_values.mean()) if family_corr_values.size else 0.0
        )
        ap_lift = finite_or(overall["average_precision_lift"], 1.0)
        overall_auc = finite_or(overall["roc_auc"], 0.5)
        signature = finite_or(item["event_transition_signature_cosine"], 0.0)
        selected_fraction = float(overall.get("selected_count", 1.0)) / max(
            float(overall.get("rows", 10.0)), 1.0
        )
        impact_mass_recall = finite_or(
            overall.get(
                "systemic_impact_mass_recall_at_fit_rate", selected_fraction
            ),
            selected_fraction,
        )
        impact_mass_lift = impact_mass_recall / max(selected_fraction, 1e-8)
        score = (
            0.20 * np.clip(2.0 * (overall_auc - 0.5), -1.0, 1.0)
            + 0.10 * np.clip(ap_lift - 1.0, -1.0, 1.0)
            + 0.20 * np.clip(2.0 * (family_auc - 0.5), -1.0, 1.0)
            + 0.15 * np.clip(family_corr, -1.0, 1.0)
            + 0.15
            * np.clip(
                2.0 * signature - 1.0,
                -1.0,
                1.0,
            )
            + 0.20 * np.clip(impact_mass_lift - 1.0, -1.0, 1.0)
        )
        weighted.append(score * float(HORIZON_WEIGHTS.get(int(horizon), 1.0)))
        weights.append(float(HORIZON_WEIGHTS.get(int(horizon), 1.0)))
    base = float(np.sum(weighted) / np.sum(weights))
    peak = finite_or(trajectory["peak_horizon_accuracy_on_events"], 0.20)
    return 0.90 * base + 0.10 * np.clip((peak - 0.20) / 0.80, -1.0, 1.0)


def summarize(predictions, contracts, horizons, fit_event_rate):
    horizon = {
        str(value): horizon_metrics(predictions[int(value)], contracts[int(value)])
        for value in horizons
    }
    trajectory = trajectory_metrics(
        predictions, contracts, horizons, fit_event_rate
    )
    return {
        "horizons": horizon,
        "trajectory": trajectory,
        "validation_formula_score": validation_score(
            horizon, trajectory, horizons
        ),
    }


def _daily_rows(predictions, contracts, horizons, split):
    output = []
    for horizon in horizons:
        calibration = contracts[int(horizon)].calibration
        thresholds = np.asarray(
            [calibration.family_event_threshold[name] for name in MARKET_FAMILY_TARGETS]
        )
        for row in predictions[int(horizon)]:
            actual_family = _family_vector(row["actual"], calibration)
            predicted_family = np.asarray(row["predicted_families"], dtype=np.float64)
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(np.asarray(row["event_logits"]), -30.0, 30.0))
            )
            labels = market_transition_labels(row["actual"], calibration)
            actual_normalized_salience = float(
                max(
                    np.max(actual_family / thresholds),
                    float(labels["broad_selloff"]),
                )
            )
            output.append(
                {
                    "split": split,
                    "date": row["date"],
                    "horizon": int(horizon),
                    "actual_systemic_energy": float(np.max(actual_family)),
                    "predicted_systemic_energy": float(np.max(predicted_family)),
                    "actual_normalized_salience": float(
                        actual_normalized_salience
                    ),
                    "predicted_normalized_salience": float(
                        np.max(predicted_family / thresholds)
                    ),
                    **{
                        f"actual_family_{name}": float(actual_family[index])
                        for index, name in enumerate(MARKET_FAMILY_TARGETS)
                    },
                    **{
                        f"predicted_family_{name}": float(predicted_family[index])
                        for index, name in enumerate(MARKET_FAMILY_TARGETS)
                    },
                    **{
                        f"actual_{name}": bool(labels[name])
                        for name in MARKET_EVENT_TARGETS
                    },
                    **{
                        f"probability_{name}": float(probabilities[index])
                        for index, name in enumerate(MARKET_EVENT_TARGETS)
                    },
                }
            )
    return sorted(output, key=lambda row: (row["date"], row["horizon"]))


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a joint market-transition trajectory head on frozen JEPA latents."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--family-query-pooling", action="store_true")
    parser.add_argument("--stock-quantile-pooling", action="store_true")
    parser.add_argument("--preserve-external-identity", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--eval-batch-size", type=int, default=32)
    parser.add_argument("--edge-cache-workers", type=int, default=16)
    parser.add_argument("--max-fit-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    if args.family_query_pooling and args.preserve_external_identity:
        raise ValueError(
            "family-query pooling already preserves external identity"
        )

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    horizons = parse_int_list(args.horizons)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(model_dir, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_args = dict(checkpoint.get("args", {}))
    validate_future_rollout_contract(checkpoint_args, horizons, False)
    feature_args = evaluator_namespace(args)
    feature_args.horizons = configured_horizon_text(checkpoint_args, horizons)
    feature_args.edge_cache_workers = int(args.edge_cache_workers)
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(features, checkpoint_args, horizons, int(args.validation_days))
    splits["fit"] = _subsample(splits["fit"], int(args.max_fit_steps))
    splits["validation"] = _subsample(
        splits["validation"], int(args.max_validation_steps)
    )
    splits["test"] = _subsample(splits["test"], int(args.max_test_steps))
    raw_rows = {
        name: _actual_rows(features, steps, horizons, name)
        for name, steps in splits.items()
    }
    contracts = build_target_contracts(raw_rows["fit"], horizons)
    targets = {
        name: build_target_arrays(raw_rows[name], steps, horizons, contracts)
        for name, steps in splits.items()
    }
    fit_event_rate = fit_trajectory_event_rate(targets["fit"])
    all_steps = np.unique(np.concatenate(list(splits.values())))
    edge_cache = build_evaluation_edge_cache(
        features, all_steps, checkpoint_args, feature_args
    )

    head_type = (
        FamilyQueryMarketTrajectoryHead
        if args.family_query_pooling
        else MarketTrajectoryHead
    )
    head_kwargs = {
        "projection_dim": int(args.projection_dim),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
        "heads": int(args.heads),
        "dropout": float(args.dropout),
        "stock_quantiles": bool(args.stock_quantile_pooling),
        "external_node_count": int(
            features.node_count - features.tradable_count
        ),
    }
    if not args.family_query_pooling:
        head_kwargs["preserve_external_identity"] = bool(
            args.preserve_external_identity
        )
    head = head_type(
        int(checkpoint_args["hidden_dim"]), horizons, **head_kwargs
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    history = []
    best_score = -math.inf
    best_state = None
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_loss, terms = train_epoch(
            model,
            head,
            features,
            splits["fit"],
            targets["fit"],
            contracts,
            horizons,
            checkpoint_args,
            feature_args,
            edge_cache,
            optimizer,
            device,
            int(args.batch_size),
            int(args.seed) + epoch,
        )
        validation_predictions = predict_steps(
            model,
            head,
            features,
            splits["validation"],
            targets["validation"],
            contracts,
            horizons,
            checkpoint_args,
            feature_args,
            edge_cache,
            device,
            int(args.eval_batch_size),
        )
        validation = summarize(
            validation_predictions, contracts, horizons, fit_event_rate
        )
        score = float(validation["validation_formula_score"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_terms": terms,
                "validation_score": score,
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"validation_market_score={score:+.6f}",
            flush=True,
        )
        if math.isfinite(score) and score > best_score + 1e-4:
            best_score = score
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("market trajectory head produced no valid checkpoint")
    head.load_state_dict(best_state)
    predictions = {
        split: predict_steps(
            model,
            head,
            features,
            splits[split],
            targets[split],
            contracts,
            horizons,
            checkpoint_args,
            feature_args,
            edge_cache,
            device,
            int(args.eval_batch_size),
        )
        for split in ("validation", "test")
    }
    metrics = {
        split: summarize(values, contracts, horizons, fit_event_rate)
        for split, values in predictions.items()
    }
    for split in predictions:
        _write_csv(
            output_dir / f"daily_{split}.csv",
            _daily_rows(predictions[split], contracts, horizons, split),
        )

    parent_sha = checkpoint_sha256(model_dir)
    summary = {
        "status": "complete",
        "role": "posthoc_frozen_jepa_joint_market_transition_head",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "model_dir": str(model_dir),
        "parent_model_sha256": parent_sha,
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "split_dates": {name: len(steps) for name, steps in splits.items()},
        "architecture": {
            "projection_dim": int(args.projection_dim),
            "hidden_dim": int(args.hidden_dim),
            "layers": int(args.layers),
            "heads": int(args.heads),
            "dropout": float(args.dropout),
            "joint_horizon_encoder": True,
            "pooling": (
                "family_query_cross_attention_over_nodes_q25_q75_and_external_ids"
                if args.family_query_pooling
                and args.stock_quantile_pooling
                else "family_query_cross_attention_over_nodes_and_external_ids"
                if args.family_query_pooling
                else
                "stock_mean_std_q25_median_q75_plus_ordered_external_nodes"
                if args.stock_quantile_pooling
                and args.preserve_external_identity
                else "stock_mean_std_q25_median_q75_plus_external_mean_std"
                if args.stock_quantile_pooling
                else "stock_mean_std_median_plus_ordered_external_nodes"
                if args.preserve_external_identity
                else "stock_mean_std_median_plus_external_mean_std"
            ),
            "family_query_pooling": bool(args.family_query_pooling),
            "family_query_count": (
                len(MARKET_FAMILY_TARGETS) if args.family_query_pooling else 0
            ),
            "stock_quantile_pooling": bool(args.stock_quantile_pooling),
            "preserve_external_identity": bool(
                args.preserve_external_identity or args.family_query_pooling
            ),
            "external_node_count": int(
                features.node_count - features.tradable_count
            ),
            "individual_node_max_pooling": False,
        },
        "loss_weights": LOSS_WEIGHTS,
        "sample_weight": "1 + 3 * min(max_family_threshold_ratio, 3)",
        "impact_weighted_event_loss": True,
        "target_contracts": {
            str(horizon): contracts[int(horizon)].to_dict() for horizon in horizons
        },
        "fit_cross_horizon_event_rate": fit_event_rate,
        "best_validation_score": best_score,
        "history": history,
        "metrics": metrics,
        "test_used_for_selection": False,
        "selection_status": "exploratory_only_requires_future_confirmation",
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "target_version": MARKET_TRANSITION_TARGET_VERSION,
            "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
            "parent_model_sha256": parent_sha,
            "horizons": horizons,
            "architecture": summary["architecture"],
            "loss_weights": LOSS_WEIGHTS,
            "target_contracts": summary["target_contracts"],
            "live_orders_allowed": False,
        },
        output_dir / "market_transition_head.pt",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "best_validation_score": best_score,
                "test_trajectory": metrics["test"]["trajectory"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
