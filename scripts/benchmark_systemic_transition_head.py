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

from scripts.audit_systemic_transition_targets import _actual_rows, _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace, newey_west_mean
from scripts.benchmark_latent_trajectory_path_head import (
    HORIZON_WEIGHTS,
    checkpoint_sha256,
    latent_trajectories,
    snapshot_batch,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    pearson,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import parse_int_list
from stock_v2.systemic_head import (
    SYSTEMIC_COMPONENT_TARGETS,
    SYSTEMIC_EVENT_TARGETS,
    SystemicTransitionHead,
    correlation_rank_loss,
    focal_binary_loss,
    weighted_smooth_l1_loss,
)
from stock_v2.systemic_transition import (
    SYSTEMIC_TARGET_VERSION,
    SystemicCalibration,
    binary_ranking_metrics,
    derived_subtype_scores,
    event_labels,
    fit_systemic_calibration,
    score_systemic_components,
    systemic_score_metrics,
)


LOSS_WEIGHTS = {
    "components": 0.30,
    "energy": 0.20,
    "energy_rank": 0.15,
    "event": 0.20,
    "subtypes": 0.15,
}


@dataclass(frozen=True)
class HorizonTargetContract:
    calibration: SystemicCalibration
    component_mean: np.ndarray
    component_std: np.ndarray
    sample_weight_mean: float
    subtype_pos_weight: np.ndarray
    event_fit_rate: np.ndarray

    def to_dict(self) -> dict[str, Any]:
        return {
            "calibration": self.calibration.to_dict(),
            "component_names": list(SYSTEMIC_COMPONENT_TARGETS),
            "component_mean": self.component_mean.tolist(),
            "component_std": self.component_std.tolist(),
            "sample_weight_mean": float(self.sample_weight_mean),
            "subtype_pos_weight": self.subtype_pos_weight.tolist(),
            "event_fit_rate": self.event_fit_rate.tolist(),
        }


def _subsample(steps: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or len(steps) <= maximum:
        return np.asarray(steps, dtype=np.int64)
    positions = np.linspace(0, len(steps) - 1, int(maximum)).round().astype(np.int64)
    return np.asarray(steps, dtype=np.int64)[positions]


def configured_horizon_text(checkpoint_args, fallback):
    value = checkpoint_args.get(
        "path_horizons", checkpoint_args.get("rollout_offsets", fallback)
    )
    if isinstance(value, str):
        return ",".join(part.strip() for part in value.split(",") if part.strip())
    return ",".join(str(int(item)) for item in value)


def _rows_by_horizon_and_step(rows):
    return {
        (int(row["horizon"]), int(row["step"])): row
        for row in rows
    }


def _component_matrix(rows: Sequence[Mapping[str, Any]]) -> np.ndarray:
    return np.asarray(
        [
            [float(row.get(name, float("nan"))) for name in SYSTEMIC_COMPONENT_TARGETS]
            for row in rows
        ],
        dtype=np.float64,
    )


def build_target_contracts(fit_rows, horizons):
    contracts = {}
    for horizon in horizons:
        selected = [row for row in fit_rows if int(row["horizon"]) == int(horizon)]
        calibration = fit_systemic_calibration(selected, event_quantile=0.90)
        matrix = _component_matrix(selected)
        means = np.nanmean(matrix, axis=0)
        stds = np.nanstd(matrix, axis=0)
        stds = np.where(np.isfinite(stds) & (stds > 1e-8), stds, 1.0)
        energies = np.asarray(
            [score_systemic_components(row, calibration)["systemic_energy"] for row in selected],
            dtype=np.float64,
        )
        raw_weight = 1.0 + 3.0 * np.minimum(
            energies / max(float(calibration.event_threshold), 1e-8), 3.0
        )
        labels = np.asarray(
            [
                [float(event_labels(row, calibration)[name]) for name in SYSTEMIC_EVENT_TARGETS]
                for row in selected
            ],
            dtype=np.float64,
        )
        positives = labels[:, 1:].sum(axis=0)
        negatives = len(labels) - positives
        subtype_pos_weight = np.clip(
            negatives / np.maximum(positives, 1.0), 1.0, 12.0
        )
        contracts[int(horizon)] = HorizonTargetContract(
            calibration=calibration,
            component_mean=means.astype(np.float32),
            component_std=stds.astype(np.float32),
            sample_weight_mean=float(np.nanmean(raw_weight)),
            subtype_pos_weight=subtype_pos_weight.astype(np.float32),
            event_fit_rate=labels.mean(axis=0).astype(np.float32),
        )
    return contracts


def _target_arrays(rows, steps, horizons, contracts):
    lookup = _rows_by_horizon_and_step(rows)
    result = {}
    for horizon in horizons:
        contract = contracts[int(horizon)]
        selected = [lookup[(int(horizon), int(step))] for step in steps]
        components_raw = _component_matrix(selected)
        component_valid = np.isfinite(components_raw)
        components = (
            components_raw - contract.component_mean[None, :]
        ) / contract.component_std[None, :]
        components = np.where(component_valid, components, 0.0).astype(np.float32)
        energy = np.asarray(
            [
                score_systemic_components(row, contract.calibration)["systemic_energy"]
                for row in selected
            ],
            dtype=np.float32,
        )
        labels = np.asarray(
            [
                [
                    float(event_labels(row, contract.calibration)[name])
                    for name in SYSTEMIC_EVENT_TARGETS
                ]
                for row in selected
            ],
            dtype=np.float32,
        )
        raw_weight = 1.0 + 3.0 * np.minimum(
            energy / max(float(contract.calibration.event_threshold), 1e-8), 3.0
        )
        result[int(horizon)] = {
            "rows": selected,
            "components": components,
            "component_valid": component_valid,
            "energy": energy,
            "log_energy": np.log1p(np.maximum(energy, 0.0)).astype(np.float32),
            "labels": labels,
            "sample_weight": (
                raw_weight / max(float(contract.sample_weight_mean), 1e-8)
            ).astype(np.float32),
        }
    return result


def _masked_component_loss(prediction, target, valid, sample_weight):
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    weight = sample_weight[:, None] * valid.to(dtype=error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1e-8)


def _batch_target(targets, horizon, positions, device):
    values = targets[int(horizon)]
    index = np.asarray(positions, dtype=np.int64)
    return {
        "components": torch.as_tensor(values["components"][index], device=device),
        "component_valid": torch.as_tensor(
            values["component_valid"][index], dtype=torch.bool, device=device
        ),
        "energy": torch.as_tensor(values["energy"][index], device=device),
        "log_energy": torch.as_tensor(values["log_energy"][index], device=device),
        "labels": torch.as_tensor(values["labels"][index], device=device),
        "sample_weight": torch.as_tensor(values["sample_weight"][index], device=device),
    }


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
    components_history = {name: [] for name in LOSS_WEIGHTS}
    for start in range(0, len(shuffled), int(batch_size)):
        selected_steps = np.asarray(
            shuffled[start : start + int(batch_size)], dtype=np.int64
        )
        positions = [step_to_position[int(step)] for step in selected_steps]
        batch = snapshot_batch(
            features,
            selected_steps,
            checkpoint_args,
            feature_args,
            edge_cache,
            device,
        )
        context, predicted = latent_trajectories(
            model, batch, horizons, checkpoint_args
        )
        horizon_losses = []
        horizon_weight_sum = 0.0
        batch_components = {name: [] for name in LOSS_WEIGHTS}
        for horizon in horizons:
            component_prediction, energy_prediction, event_logits = head(
                context,
                predicted[int(horizon)],
                batch_size=len(selected_steps),
                node_count=features.node_count,
                stock_count=features.tradable_count,
                horizon=int(horizon),
            )
            target = _batch_target(targets, horizon, positions, device)
            component_loss = _masked_component_loss(
                component_prediction,
                target["components"],
                target["component_valid"],
                target["sample_weight"],
            )
            energy_loss = weighted_smooth_l1_loss(
                energy_prediction, target["log_energy"], target["sample_weight"]
            )
            rank_loss = correlation_rank_loss(
                energy_prediction, target["log_energy"]
            )
            event_loss = focal_binary_loss(event_logits[:, 0], target["labels"][:, 0])
            subtype_pos_weight = torch.as_tensor(
                contracts[int(horizon)].subtype_pos_weight,
                dtype=event_logits.dtype,
                device=device,
            )
            subtype_loss = F.binary_cross_entropy_with_logits(
                event_logits[:, 1:],
                target["labels"][:, 1:],
                pos_weight=subtype_pos_weight,
            )
            values = {
                "components": component_loss,
                "energy": energy_loss,
                "energy_rank": rank_loss,
                "event": event_loss,
                "subtypes": subtype_loss,
            }
            loss = sum(LOSS_WEIGHTS[name] * values[name] for name in LOSS_WEIGHTS)
            weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
            horizon_losses.append(weight * loss)
            horizon_weight_sum += weight
            for name, value in values.items():
                batch_components[name].append(value)
        loss = torch.stack(horizon_losses).sum() / horizon_weight_sum
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for name, values in batch_components.items():
            components_history[name].append(
                float(torch.stack(values).mean().detach().cpu())
            )
    return float(np.mean(losses)), {
        name: float(np.mean(values)) for name, values in components_history.items()
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
            features,
            selected_steps,
            checkpoint_args,
            feature_args,
            edge_cache,
            device,
        )
        context, predicted = latent_trajectories(model, batch, horizons, checkpoint_args)
        with torch.no_grad():
            for horizon in horizons:
                normalized_components, log_energy, logits = head(
                    context,
                    predicted[int(horizon)],
                    batch_size=len(selected_steps),
                    node_count=features.node_count,
                    stock_count=features.tradable_count,
                    horizon=int(horizon),
                )
                contract = contracts[int(horizon)]
                raw_components = (
                    normalized_components.float().cpu().numpy()
                    * contract.component_std[None, :]
                    + contract.component_mean[None, :]
                )
                energy = np.maximum(
                    np.expm1(np.clip(log_energy.float().cpu().numpy(), -5.0, 5.0)),
                    0.0,
                )
                event_scores = logits.float().cpu().numpy()
                for position, step in enumerate(selected_steps):
                    predicted_row = {
                        name: float(raw_components[position, index])
                        for index, name in enumerate(SYSTEMIC_COMPONENT_TARGETS)
                    }
                    output[int(horizon)].append(
                        {
                            "step": int(step),
                            "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                            "horizon": int(horizon),
                            "actual": targets[int(horizon)]["rows"][start + position],
                            "predicted": predicted_row,
                            "predicted_energy": float(energy[position]),
                            "event_logits": event_scores[position].tolist(),
                        }
                    )
    return output


def _energy_metrics(records, contract):
    calibration = contract.calibration
    actual_energy = np.asarray(
        [
            score_systemic_components(row["actual"], calibration)["systemic_energy"]
            for row in records
        ],
        dtype=np.float64,
    )
    predicted_energy = np.asarray(
        [float(row["predicted_energy"]) for row in records], dtype=np.float64
    )
    labels = actual_energy >= float(calibration.event_threshold)
    ranking = binary_ranking_metrics(
        labels, predicted_energy, selection_rate=float(calibration.fit_event_rate)
    )
    selected_count = int(ranking["selected_count"])
    selected = np.argsort(predicted_energy, kind="mergesort")[-selected_count:]
    total_mass = float(actual_energy.sum())
    correlation = pearson(predicted_energy, actual_energy)
    actual_return = np.asarray(
        [float(row["actual"]["market_return"]) for row in records], dtype=np.float64
    )
    predicted_return = np.asarray(
        [float(row["predicted"]["market_return"]) for row in records], dtype=np.float64
    )
    direction_valid = labels & np.isfinite(actual_return) & np.isfinite(predicted_return)
    direction_accuracy = float("nan")
    if direction_valid.any():
        correct = np.sign(actual_return[direction_valid]) == np.sign(
            predicted_return[direction_valid]
        )
        direction_accuracy = float(
            np.average(correct.astype(np.float64), weights=actual_energy[direction_valid])
        )
    ranking.update(
        {
            "fit_selection_rate": float(calibration.fit_event_rate),
            "average_precision_lift": (
                float(ranking["average_precision"]) / float(ranking["event_rate"])
                if float(ranking["event_rate"]) > 0.0
                else float("nan")
            ),
            "systemic_energy_correlation": correlation,
            "tail_mass_recall_at_fit_event_rate": (
                float(actual_energy[selected].sum() / total_mass)
                if total_mass > 1e-12
                else float("nan")
            ),
            "event_impact_weighted_market_direction_accuracy": direction_accuracy,
        }
    )
    return ranking


def _subtype_metrics(records, contract):
    calibration = contract.calibration
    result = {}
    fit_rate_names = SYSTEMIC_EVENT_TARGETS[1:]
    for index, name in enumerate(fit_rate_names, start=1):
        labels = [event_labels(row["actual"], calibration)[name] for row in records]
        scores = [float(row["event_logits"][index]) for row in records]
        fit_rate = float(contract.event_fit_rate[index])
        result[name] = binary_ranking_metrics(
            labels, scores, selection_rate=max(fit_rate, 1e-6)
        )
    return result


def _derived_subtype_metrics(records, contract):
    calibration = contract.calibration
    result = {}
    for index, name in enumerate(SYSTEMIC_EVENT_TARGETS[1:], start=1):
        labels = [event_labels(row["actual"], calibration)[name] for row in records]
        scores = [
            derived_subtype_scores(row["predicted"], calibration)[name]
            for row in records
        ]
        result[name] = binary_ranking_metrics(
            labels,
            scores,
            selection_rate=max(float(contract.event_fit_rate[index]), 1e-6),
        )
    return result


def _validation_score(metrics_by_horizon, horizons):
    total = 0.0
    weight_sum = 0.0
    for horizon in horizons:
        item = metrics_by_horizon[str(horizon)]
        primary = item["energy_head"]
        subtype_auc = np.nanmean(
            [float(value["roc_auc"]) for value in item["subtypes"].values()]
        )
        selection_rate = float(primary["fit_selection_rate"])
        mass = float(primary["tail_mass_recall_at_fit_event_rate"])
        terms = {
            "auc": np.clip(2.0 * (float(primary["roc_auc"]) - 0.5), -1.0, 1.0),
            "ap": np.clip(float(primary["average_precision_lift"]) - 1.0, -1.0, 1.0),
            "mass": np.clip(
                (mass - selection_rate) / max(1.0 - selection_rate, 1e-6),
                -1.0,
                1.0,
            ),
            "correlation": np.clip(float(primary["systemic_energy_correlation"]), -1.0, 1.0),
            "subtypes": np.clip(2.0 * (float(subtype_auc) - 0.5), -1.0, 1.0),
        }
        score = (
            0.30 * terms["auc"]
            + 0.20 * terms["ap"]
            + 0.20 * terms["mass"]
            + 0.15 * terms["correlation"]
            + 0.15 * terms["subtypes"]
        )
        weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
        total += weight * score
        weight_sum += weight
    return float(total / weight_sum)


def summarize_predictions(predictions, contracts, horizons):
    result = {}
    for horizon in horizons:
        records = predictions[int(horizon)]
        result[str(horizon)] = {
            "energy_head": _energy_metrics(records, contracts[int(horizon)]),
            "component_derived": systemic_score_metrics(
                [row["actual"] for row in records],
                [row["predicted"] for row in records],
                contracts[int(horizon)].calibration,
            ),
            "subtypes": _subtype_metrics(records, contracts[int(horizon)]),
            "derived_subtypes": _derived_subtype_metrics(
                records, contracts[int(horizon)]
            ),
        }
    return result, _validation_score(result, horizons)


def trajectory_metrics(predictions, contracts, horizons, fit_event_rate):
    by_date: dict[str, list[Mapping[str, Any]]] = {}
    for horizon in horizons:
        for row in predictions[int(horizon)]:
            by_date.setdefault(str(row["date"]), []).append(row)
    labels = []
    predicted_scores = []
    peak_matches = []
    trajectory_correlations = []
    for rows in by_date.values():
        rows = sorted(rows, key=lambda row: int(row["horizon"]))
        actual = np.asarray(
            [
                score_systemic_components(
                    row["actual"], contracts[int(row["horizon"])].calibration
                )["systemic_energy"]
                / contracts[int(row["horizon"])].calibration.event_threshold
                for row in rows
            ],
            dtype=np.float64,
        )
        predicted = np.asarray(
            [
                float(row["predicted_energy"])
                / contracts[int(row["horizon"])].calibration.event_threshold
                for row in rows
            ],
            dtype=np.float64,
        )
        labels.append(bool(actual.max() >= 1.0))
        predicted_scores.append(float(predicted.max()))
        trajectory_correlations.append(pearson(predicted, actual))
        if labels[-1]:
            peak_matches.append(int(np.argmax(predicted)) == int(np.argmax(actual)))
    ranking = binary_ranking_metrics(
        labels, predicted_scores, selection_rate=min(float(fit_event_rate), 1.0)
    )
    finite = np.asarray(trajectory_correlations, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    ranking.update(
        {
            "fit_event_rate": float(fit_event_rate),
            "peak_horizon_accuracy_on_events": (
                float(np.mean(peak_matches)) if peak_matches else float("nan")
            ),
            "mean_trajectory_correlation": (
                float(finite.mean()) if finite.size else float("nan")
            ),
        }
    )
    return ranking


def fit_trajectory_event_rate(targets, contracts, horizons):
    date_count = len(targets[int(horizons[0])]["energy"])
    events = np.zeros(date_count, dtype=bool)
    for horizon in horizons:
        energy = np.asarray(targets[int(horizon)]["energy"], dtype=np.float64)
        if len(energy) != date_count:
            raise ValueError("fit trajectory targets are not date aligned")
        events |= energy >= float(contracts[int(horizon)].calibration.event_threshold)
    return float(events.mean())


def _daily_rows(predictions, contracts, horizons, split):
    output = []
    for horizon in horizons:
        calibration = contracts[int(horizon)].calibration
        for row in predictions[int(horizon)]:
            actual_scores = score_systemic_components(row["actual"], calibration)
            predicted_scores = score_systemic_components(row["predicted"], calibration)
            probabilities = 1.0 / (
                1.0 + np.exp(-np.clip(np.asarray(row["event_logits"]), -30.0, 30.0))
            )
            output.append(
                {
                    "split": split,
                    "date": row["date"],
                    "horizon": int(horizon),
                    "actual_systemic_energy": actual_scores["systemic_energy"],
                    "predicted_systemic_energy": row["predicted_energy"],
                    "component_predicted_systemic_energy": predicted_scores[
                        "systemic_energy"
                    ],
                    "actual_market_return": row["actual"]["market_return"],
                    "predicted_market_return": row["predicted"]["market_return"],
                    **{
                        f"actual_{name}": bool(
                            event_labels(row["actual"], calibration)[name]
                        )
                        for name in SYSTEMIC_EVENT_TARGETS
                    },
                    **{
                        f"probability_{name}": float(probabilities[index])
                        for index, name in enumerate(SYSTEMIC_EVENT_TARGETS)
                    },
                }
            )
    return sorted(output, key=lambda item: (item["date"], item["horizon"]))


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an impact-weighted systemic transition head on frozen JEPA rollout latents."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--horizon-dim", type=int, default=16)
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
    splits = _split_steps(
        features, checkpoint_args, horizons, int(args.validation_days)
    )
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
        name: _target_arrays(raw_rows[name], steps, horizons, contracts)
        for name, steps in splits.items()
    }
    fit_trajectory_rate = fit_trajectory_event_rate(
        targets["fit"], contracts, horizons
    )
    all_steps = np.unique(np.concatenate(list(splits.values())))
    edge_cache = build_evaluation_edge_cache(
        features, all_steps, checkpoint_args, feature_args
    )

    head = SystemicTransitionHead(
        int(checkpoint_args["hidden_dim"]),
        horizons,
        projection_dim=int(args.projection_dim),
        hidden_dim=int(args.hidden_dim),
        horizon_dim=int(args.horizon_dim),
        dropout=float(args.dropout),
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
        train_loss, train_components = train_epoch(
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
        validation_metrics, validation_score = summarize_predictions(
            validation_predictions, contracts, horizons
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_components": train_components,
                "validation_score": validation_score,
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"validation_systemic_score={validation_score:+.6f}",
            flush=True,
        )
        if math.isfinite(validation_score) and validation_score > best_score + 1e-4:
            best_score = validation_score
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("systemic head did not produce a valid validation checkpoint")
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
    summaries = {}
    for split in predictions:
        horizon_metrics, score = summarize_predictions(
            predictions[split], contracts, horizons
        )
        summaries[split] = {
            "horizons": horizon_metrics,
            "weighted_validation_formula_score": score,
            "trajectory": trajectory_metrics(
                predictions[split], contracts, horizons, fit_trajectory_rate
            ),
        }
        _write_csv(
            output_dir / f"daily_{split}.csv",
            _daily_rows(predictions[split], contracts, horizons, split),
        )

    parent_sha = checkpoint_sha256(model_dir)
    summary = {
        "status": "complete",
        "role": "posthoc_impact_weighted_frozen_jepa_systemic_transition_head",
        "target_version": SYSTEMIC_TARGET_VERSION,
        "model_dir": str(model_dir),
        "parent_model_sha256": parent_sha,
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "split_dates": {name: len(steps) for name, steps in splits.items()},
        "architecture": {
            "projection_dim": int(args.projection_dim),
            "hidden_dim": int(args.hidden_dim),
            "horizon_dim": int(args.horizon_dim),
            "dropout": float(args.dropout),
            "pooling": "stock_mean_std_median_plus_external_mean_std",
            "individual_node_max_pooling": False,
        },
        "loss_weights": LOSS_WEIGHTS,
        "systemic_sample_weight": "1 + 3 * min(actual_energy / fit_threshold, 3)",
        "target_contracts": {
            str(horizon): contracts[int(horizon)].to_dict() for horizon in horizons
        },
        "fit_cross_horizon_event_rate": fit_trajectory_rate,
        "best_validation_score": best_score,
        "history": history,
        "metrics": summaries,
        "fold2_used_for_selection": False,
        "selection_status": "exploratory_only_requires_future_confirmation",
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "target_version": SYSTEMIC_TARGET_VERSION,
            "parent_model_sha256": parent_sha,
            "horizons": horizons,
            "projection_dim": int(args.projection_dim),
            "hidden_dim": int(args.hidden_dim),
            "horizon_dim": int(args.horizon_dim),
            "dropout": float(args.dropout),
            "loss_weights": LOSS_WEIGHTS,
            "target_contracts": summary["target_contracts"],
            "live_orders_allowed": False,
        },
        output_dir / "systemic_transition_head.pt",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "best_validation_score": best_score,
                "test_trajectory": summaries["test"]["trajectory"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
