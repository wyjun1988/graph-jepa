from __future__ import annotations

import argparse
from collections import defaultdict
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_direct_baselines import evaluator_namespace, newey_west_mean
from scripts.benchmark_latent_trajectory_path_head import (
    checkpoint_sha256,
    chronological_splits,
    latent_trajectories,
    snapshot_batch,
    stock_rows,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    pearson,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import (
    date_indices,
    parse_int_list,
    rollout_steps_for_offset,
    temporal_training_indices,
)
from stock_v2.systemic_transition import (
    SYSTEMIC_TARGET_VERSION,
    SystemicCalibration,
    binary_ranking_metrics,
    event_labels,
    fit_systemic_calibration,
    score_systemic_components,
    systemic_score_metrics,
    transition_components,
)


METHODS = ("rollout", "no_rollout")


def _split_steps(features, checkpoint_args, horizons, validation_days):
    train_end = str(checkpoint_args["train_end"])
    max_horizon = max(horizons)
    edge_window = int(checkpoint_args.get("edge_window", 60))
    train_indices = date_indices(features.dates, end=train_end)
    train_steps = temporal_training_indices(
        train_indices,
        edge_window=edge_window,
        max_rollout_offset=max_horizon,
        total_steps=len(features.dates),
    )
    fit_steps, validation_steps = chronological_splits(
        train_steps, validation_days, max_horizon
    )
    test_steps = date_indices(
        features.dates,
        start=(pd.Timestamp(train_end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    test_steps = test_steps[
        (test_steps >= edge_window)
        & (test_steps <= len(features.dates) - 1 - max_horizon)
    ]
    return {"fit": fit_steps, "validation": validation_steps, "test": test_steps}


def _subsample(steps: np.ndarray, maximum: int) -> np.ndarray:
    if maximum <= 0 or len(steps) <= maximum:
        return np.asarray(steps, dtype=np.int64)
    positions = np.linspace(0, len(steps) - 1, int(maximum)).round().astype(np.int64)
    return np.asarray(steps, dtype=np.int64)[positions]


def _entry_paths_from_states(model, states, horizons):
    h1 = states[1]
    gap_index = int(model.gap_open_feature_index)
    intraday_index = int(model.intraday_return_feature_index)
    next_open_gap = h1[:, gap_index] * model.feature_stds[gap_index] + model.feature_means[
        gap_index
    ]
    output = {
        1: h1[:, intraday_index] * model.feature_stds[intraday_index]
        + model.feature_means[intraday_index]
    }
    for horizon in horizons:
        if int(horizon) == 1:
            continue
        return_index = int(model.return_feature_indices[int(horizon)])
        close_return = (
            states[int(horizon)][:, return_index] * model.feature_stds[return_index]
            + model.feature_means[return_index]
        )
        denominator = 1.0 + next_open_gap
        output[int(horizon)] = torch.where(
            denominator > 1e-6,
            (1.0 + close_return) / denominator.clamp_min(1e-6) - 1.0,
            torch.full_like(close_return, float("nan")),
        )
    return output


def _state_predictions(model, batch, context, predicted, horizons, checkpoint_args, rows):
    namespace = argparse.Namespace(**checkpoint_args)
    with torch.no_grad():
        rollout = {
            int(horizon): model.predict_temporal_state(
                batch,
                predicted[int(horizon)],
                rollout_steps=rollout_steps_for_offset(namespace, int(horizon)),
                z_context=context,
            )[rows]
            for horizon in horizons
        }
        no_rollout = {
            int(horizon): model.predict_temporal_state(
                batch,
                context,
                rollout_steps=rollout_steps_for_offset(namespace, int(horizon)),
                z_context=context,
            )[rows]
            for horizon in horizons
        }
    return {"rollout": rollout, "no_rollout": no_rollout}


def _score_steps(
    model,
    features,
    steps,
    horizons,
    checkpoint_args,
    feature_args,
    edge_cache,
    device,
    batch_size,
    split,
):
    stock_count = int(features.tradable_count)
    feature_count = len(features.feature_names)
    return_index = features.feature_names.index("return_1d")
    train_mean = np.asarray(features.train_mean, dtype=np.float64)
    train_std = np.asarray(features.train_std, dtype=np.float64)
    records: list[dict[str, Any]] = []
    for start in range(0, len(steps), int(batch_size)):
        selected_steps = np.asarray(steps[start : start + int(batch_size)], dtype=np.int64)
        batch = snapshot_batch(
            features, selected_steps, checkpoint_args, feature_args, edge_cache, device
        )
        context, predicted = latent_trajectories(
            model, batch, horizons, checkpoint_args
        )
        rows, _groups = stock_rows(
            len(selected_steps), features.node_count, stock_count, device
        )
        state_predictions = _state_predictions(
            model, batch, context, predicted, horizons, checkpoint_args, rows
        )
        path_predictions = {
            method: _entry_paths_from_states(model, states, horizons)
            for method, states in state_predictions.items()
        }
        state_numpy = {
            method: {
                int(horizon): values.detach()
                .float()
                .cpu()
                .numpy()
                .reshape(len(selected_steps), stock_count, feature_count)
                for horizon, values in states.items()
            }
            for method, states in state_predictions.items()
        }
        path_numpy = {
            method: {
                int(horizon): values.detach()
                .float()
                .cpu()
                .numpy()
                .reshape(len(selected_steps), stock_count)
                for horizon, values in paths.items()
            }
            for method, paths in path_predictions.items()
        }
        for position, step in enumerate(selected_steps):
            current_state = features.features[int(step), :stock_count]
            current_raw = features.raw_features[int(step), :stock_count]
            current_available = features.available_mask[int(step), :stock_count] > 0.5
            for horizon in horizons:
                target_step = int(step) + int(horizon)
                future_state = features.features[target_step, :stock_count]
                future_raw = features.raw_features[target_step, :stock_count]
                future_available = features.available_mask[target_step, :stock_count] > 0.5
                actual_path = np.asarray(
                    features.target_return_paths[int(horizon)][int(step), :stock_count],
                    dtype=np.float64,
                )
                node_mask = (
                    current_available[:, return_index]
                    & future_available[:, return_index]
                    & np.isfinite(actual_path)
                )
                actual = transition_components(
                    current_state=current_state,
                    future_state=future_state,
                    current_raw=current_raw,
                    future_raw=future_raw,
                    current_available=current_available,
                    future_available=future_available,
                    feature_names=features.feature_names,
                    entry_path_returns=actual_path,
                    node_mask=node_mask,
                )
                predictions = {}
                paired_available = current_available & future_available
                for method in METHODS:
                    predicted_state = state_numpy[method][int(horizon)][position]
                    predicted_raw = predicted_state * train_std[None, :] + train_mean[None, :]
                    predicted_path = path_numpy[method][int(horizon)][position].astype(
                        np.float64
                    )
                    predicted_path[~node_mask] = np.nan
                    predictions[method] = transition_components(
                        current_state=current_state,
                        future_state=predicted_state,
                        current_raw=current_raw,
                        future_raw=predicted_raw,
                        current_available=current_available,
                        future_available=paired_available,
                        feature_names=features.feature_names,
                        entry_path_returns=predicted_path,
                        node_mask=node_mask,
                    )
                records.append(
                    {
                        "split": str(split),
                        "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                        "target_date": str(pd.Timestamp(features.dates[target_step]).date()),
                        "step": int(step),
                        "horizon": int(horizon),
                        "actual": actual,
                        **predictions,
                    }
                )
        print(
            f"split={split} scored={min(start + int(batch_size), len(steps))}/{len(steps)}",
            flush=True,
        )
    return records


def _subtype_score(
    row: Mapping[str, float | int],
    calibration: SystemicCalibration,
    subtype: str,
) -> float:
    scored = score_systemic_components(row, calibration)
    if subtype == "systemic_event":
        return float(scored["systemic_energy"])
    if subtype == "broad_selloff":
        market_scale = max(
            abs(float(calibration.broad_selloff_return_threshold)), 1e-6
        )
        breadth_scale = max(
            abs(float(calibration.broad_selloff_breadth_threshold)), 1e-6
        )
        return -0.5 * (
            float(row.get("market_return", float("nan"))) / market_scale
            + float(row.get("breadth", float("nan"))) / breadth_scale
        )
    if subtype == "turnover_explosion":
        return float(
            max(
                scored["component:volume_shock"],
                scored["component:traded_value_shock"],
            )
        )
    if subtype == "graph_state_shift":
        return float(scored["family:graph_state"])
    raise ValueError(f"unknown subtype: {subtype}")


def _subtype_metrics(
    fit_actual,
    actual,
    predicted,
    calibration: SystemicCalibration,
):
    result = {}
    for subtype in (
        "systemic_event",
        "broad_selloff",
        "turnover_explosion",
        "graph_state_shift",
    ):
        fit_rate = float(
            np.mean([event_labels(row, calibration)[subtype] for row in fit_actual])
        )
        labels = [event_labels(row, calibration)[subtype] for row in actual]
        scores = [_subtype_score(row, calibration, subtype) for row in predicted]
        result[subtype] = binary_ranking_metrics(
            labels,
            scores,
            selection_rate=max(fit_rate, 1.0 / max(len(fit_actual), 1)),
        )
        result[subtype]["fit_event_rate"] = fit_rate
    return result


def _paired_rollout_improvement(records, calibration):
    differences = []
    for record in records:
        actual = score_systemic_components(record["actual"], calibration)[
            "systemic_energy"
        ]
        rollout = score_systemic_components(record["rollout"], calibration)[
            "systemic_energy"
        ]
        no_rollout = score_systemic_components(record["no_rollout"], calibration)[
            "systemic_energy"
        ]
        if np.isfinite(actual) and np.isfinite(rollout) and np.isfinite(no_rollout):
            differences.append(abs(no_rollout - actual) - abs(rollout - actual))
    return newey_west_mean(differences, lag=10)


def _trajectory_summary(
    records,
    calibrations: Mapping[int, SystemicCalibration],
    method: str,
    selection_rate: float,
):
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[str(record["date"])].append(record)
    labels = []
    predicted_peaks = []
    peak_correct = []
    path_correlations = []
    absolute_errors = []
    for date_records in grouped.values():
        ordered = sorted(date_records, key=lambda item: int(item["horizon"]))
        actual_ratios = []
        predicted_ratios = []
        horizons = []
        for record in ordered:
            horizon = int(record["horizon"])
            calibration = calibrations[horizon]
            actual_energy = score_systemic_components(
                record["actual"], calibration
            )["systemic_energy"]
            predicted_energy = score_systemic_components(
                record[method], calibration
            )["systemic_energy"]
            actual_ratios.append(actual_energy / calibration.event_threshold)
            predicted_ratios.append(predicted_energy / calibration.event_threshold)
            horizons.append(horizon)
        actual_array = np.asarray(actual_ratios, dtype=np.float64)
        predicted_array = np.asarray(predicted_ratios, dtype=np.float64)
        valid = np.isfinite(actual_array) & np.isfinite(predicted_array)
        if not valid.all() or not valid.any():
            continue
        actual_peak = float(actual_array.max())
        predicted_peak = float(predicted_array.max())
        event = actual_peak >= 1.0
        labels.append(event)
        predicted_peaks.append(predicted_peak)
        absolute_errors.append(abs(predicted_peak - actual_peak))
        path_correlations.append(pearson(predicted_array, actual_array))
        if event:
            peak_correct.append(
                int(horizons[int(np.argmax(predicted_array))])
                == int(horizons[int(np.argmax(actual_array))])
            )
    ranking = binary_ranking_metrics(
        labels, predicted_peaks, selection_rate=float(selection_rate)
    )
    finite_correlations = np.asarray(path_correlations, dtype=np.float64)
    finite_correlations = finite_correlations[np.isfinite(finite_correlations)]
    ranking.update(
        {
            "peak_horizon_accuracy_on_events": (
                float(np.mean(peak_correct)) if peak_correct else float("nan")
            ),
            "mean_trajectory_correlation": (
                float(finite_correlations.mean())
                if finite_correlations.size
                else float("nan")
            ),
            "mean_peak_intensity_absolute_error": float(np.mean(absolute_errors)),
        }
    )
    return ranking


def _flatten_records(records, calibrations):
    rows = []
    for record in records:
        horizon = int(record["horizon"])
        calibration = calibrations[horizon]
        row = {
            key: record[key]
            for key in ("split", "date", "target_date", "step", "horizon")
        }
        for role in ("actual",) + METHODS:
            values = {**record[role], **score_systemic_components(record[role], calibration)}
            row.update({f"{role}:{name}": value for name, value in values.items()})
        rows.append(row)
    return rows


def _write_csv(path: Path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate whether JEPA rollout predicts broad graph-state transitions."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--event-quantile", type=float, default=0.90)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--edge-cache-workers", type=int, default=16)
    parser.add_argument("--max-fit-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    horizons = parse_int_list(args.horizons)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, checkpoint = load_model(model_dir, device)
    model.eval()
    checkpoint_args = dict(checkpoint.get("args", {}))
    validate_future_rollout_contract(checkpoint_args, horizons, False)
    feature_args = evaluator_namespace(args)
    feature_args.edge_cache_workers = int(args.edge_cache_workers)
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    checkpoint_args.setdefault("temporal_offset", checkpoint_args.get("horizon", max(horizons)))
    checkpoint_args.setdefault("latent_rollout_steps", 1)
    splits = _split_steps(
        features, checkpoint_args, horizons, int(args.validation_days)
    )
    splits["fit"] = _subsample(splits["fit"], int(args.max_fit_steps))
    splits["validation"] = _subsample(
        splits["validation"], int(args.max_validation_steps)
    )
    splits["test"] = _subsample(splits["test"], int(args.max_test_steps))
    all_steps = np.unique(np.concatenate(list(splits.values())))
    edge_cache = build_evaluation_edge_cache(
        features, all_steps, checkpoint_args, feature_args
    )
    records = {
        split: _score_steps(
            model,
            features,
            steps,
            horizons,
            checkpoint_args,
            feature_args,
            edge_cache,
            device,
            int(args.batch_size),
            split,
        )
        for split, steps in splits.items()
    }

    calibrations: dict[int, SystemicCalibration] = {}
    horizons_summary = {}
    for horizon in horizons:
        fit_records = [
            record for record in records["fit"] if record["horizon"] == int(horizon)
        ]
        calibration = fit_systemic_calibration(
            [record["actual"] for record in fit_records],
            event_quantile=float(args.event_quantile),
        )
        calibrations[int(horizon)] = calibration
        horizons_summary[str(horizon)] = {"calibration": calibration.to_dict()}
        for split in ("validation", "test"):
            selected = [
                record
                for record in records[split]
                if record["horizon"] == int(horizon)
            ]
            actual = [record["actual"] for record in selected]
            horizons_summary[str(horizon)][split] = {
                method: {
                    "systemic": systemic_score_metrics(
                        actual,
                        [record[method] for record in selected],
                        calibration,
                    ),
                    "subtypes": _subtype_metrics(
                        [record["actual"] for record in fit_records],
                        actual,
                        [record[method] for record in selected],
                        calibration,
                    ),
                }
                for method in METHODS
            }
            horizons_summary[str(horizon)][split]["paired_rollout_vs_no_rollout"] = (
                _paired_rollout_improvement(selected, calibration)
            )

    fit_by_date: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records["fit"]:
        fit_by_date[str(record["date"])].append(record)
    fit_trajectory_labels = []
    for date_records in fit_by_date.values():
        ratios = []
        for record in date_records:
            calibration = calibrations[int(record["horizon"])]
            energy = score_systemic_components(record["actual"], calibration)[
                "systemic_energy"
            ]
            ratios.append(energy / calibration.event_threshold)
        fit_trajectory_labels.append(bool(np.nanmax(ratios) >= 1.0))
    trajectory_selection_rate = float(np.mean(fit_trajectory_labels))
    trajectory_summary = {
        split: {
            method: _trajectory_summary(
                records[split], calibrations, method, trajectory_selection_rate
            )
            for method in METHODS
        }
        for split in ("validation", "test")
    }
    trajectory_summary["fit_event_rate"] = trajectory_selection_rate

    flat_rows = _flatten_records(
        records["fit"] + records["validation"] + records["test"], calibrations
    )
    _write_csv(output_dir / "daily_systemic_rollout.csv", flat_rows)
    summary = {
        "status": "complete",
        "role": "posthoc_frozen_jepa_systemic_state_rollout_evaluation",
        "target_version": SYSTEMIC_TARGET_VERSION,
        "model_dir": str(model_dir),
        "parent_model_sha256": checkpoint_sha256(model_dir),
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "split_dates": {name: len(steps) for name, steps in splits.items()},
        "event_quantile": float(args.event_quantile),
        "headline_definition": (
            "equal-family RMS of broad price/breadth, market risk, activity, and "
            "graph-state transition scores; single-stock extremes are not a target"
        ),
        "horizons_summary": horizons_summary,
        "cross_horizon_trajectory": trajectory_summary,
        "selection_status": "exploratory_only_requires_future_confirmation",
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "test_dates": len(splits["test"]),
                "trajectory": trajectory_summary["test"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
