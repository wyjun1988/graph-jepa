from __future__ import annotations

import argparse
import csv
import json
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
    latent_trajectories,
    snapshot_batch,
    stock_rows,
)
from scripts.evaluate_major_market_trajectory import (
    add_actual_salience,
    fit_major_threshold,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    pearson,
    validate_future_rollout_contract,
)
from scripts.evaluate_systemic_state_rollout import _split_steps, _state_predictions
from scripts.run_real_backtest import parse_int_list
from stock_v2.market_transition import (
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
    systemic_state_feature_indices,
)


METHODS = ("rollout", "no_rollout")
HIGHER_IS_BETTER = (
    "common_feature_delta_cosine",
    "common_feature_delta_correlation",
    "feature_direction_accuracy",
    "node_delta_correlation",
    "node_energy_correlation",
)
LOWER_IS_BETTER = (
    "node_energy_mae",
    "node_energy_median_absolute_error",
    "node_energy_q75_absolute_error",
    "state_participation_absolute_error",
)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 2:
        return float("nan")
    left = left[valid]
    right = right[valid]
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    return float(np.dot(left, right) / denominator) if denominator > 1e-12 else float("nan")


def transition_shape_metrics(
    current: np.ndarray,
    actual_future: np.ndarray,
    predicted_future: np.ndarray,
    current_available: np.ndarray,
    future_available: np.ndarray,
    feature_indices: np.ndarray,
    *,
    node_mask: np.ndarray,
    min_nodes: int = 20,
    state_change_threshold: float = 0.75,
    feature_direction_threshold: float = 0.05,
) -> dict[str, float | int]:
    current = np.asarray(current, dtype=np.float64)[:, feature_indices]
    actual_future = np.asarray(actual_future, dtype=np.float64)[:, feature_indices]
    predicted_future = np.asarray(predicted_future, dtype=np.float64)[:, feature_indices]
    current_available = np.asarray(current_available, dtype=bool)[:, feature_indices]
    future_available = np.asarray(future_available, dtype=bool)[:, feature_indices]
    valid = (
        np.asarray(node_mask, dtype=bool)[:, None]
        & current_available
        & future_available
        & np.isfinite(current)
        & np.isfinite(actual_future)
        & np.isfinite(predicted_future)
    )
    actual_delta = actual_future - current
    predicted_delta = predicted_future - current

    counts = valid.sum(axis=0)
    usable_features = counts >= int(min_nodes)
    actual_common = np.full(len(feature_indices), np.nan, dtype=np.float64)
    predicted_common = np.full(len(feature_indices), np.nan, dtype=np.float64)
    if usable_features.any():
        actual_common[usable_features] = np.nanmedian(
            np.where(
                valid[:, usable_features],
                actual_delta[:, usable_features],
                np.nan,
            ),
            axis=0,
        )
        predicted_common[usable_features] = np.nanmedian(
            np.where(
                valid[:, usable_features],
                predicted_delta[:, usable_features],
                np.nan,
            ),
            axis=0,
        )
    directional = (
        usable_features
        & np.isfinite(actual_common)
        & np.isfinite(predicted_common)
        & (np.abs(actual_common) >= float(feature_direction_threshold))
    )
    direction_accuracy = (
        float(
            np.mean(
                np.sign(actual_common[directional])
                == np.sign(predicted_common[directional])
            )
        )
        if directional.any()
        else float("nan")
    )

    node_counts = valid.sum(axis=1)
    minimum_features = max(3, int(np.ceil(len(feature_indices) * 0.25)))
    usable_nodes = np.asarray(node_mask, dtype=bool) & (node_counts >= minimum_features)
    actual_energy = np.full(len(node_mask), np.nan, dtype=np.float64)
    predicted_energy = np.full(len(node_mask), np.nan, dtype=np.float64)
    if usable_nodes.any():
        actual_energy[usable_nodes] = np.sqrt(
            np.divide(
                np.where(valid[usable_nodes], np.square(actual_delta[usable_nodes]), 0.0).sum(axis=1),
                node_counts[usable_nodes],
            )
        )
        predicted_energy[usable_nodes] = np.sqrt(
            np.divide(
                np.where(valid[usable_nodes], np.square(predicted_delta[usable_nodes]), 0.0).sum(axis=1),
                node_counts[usable_nodes],
            )
        )
    node_valid = np.isfinite(actual_energy) & np.isfinite(predicted_energy)
    actual_participation = (
        float(np.mean(actual_energy[node_valid] >= float(state_change_threshold)))
        if int(node_valid.sum()) >= int(min_nodes)
        else float("nan")
    )
    predicted_participation = (
        float(np.mean(predicted_energy[node_valid] >= float(state_change_threshold)))
        if int(node_valid.sum()) >= int(min_nodes)
        else float("nan")
    )
    flattened = valid & np.isfinite(actual_delta) & np.isfinite(predicted_delta)
    node_energy_absolute_error = np.abs(
        predicted_energy[node_valid] - actual_energy[node_valid]
    )
    return {
        "observed_nodes": int(node_valid.sum()),
        "observed_features": int(usable_features.sum()),
        "common_feature_delta_cosine": _cosine(actual_common, predicted_common),
        "common_feature_delta_correlation": pearson(
            predicted_common[usable_features], actual_common[usable_features]
        ),
        "feature_direction_accuracy": direction_accuracy,
        "node_delta_correlation": pearson(
            predicted_delta[flattened], actual_delta[flattened]
        ),
        "node_energy_correlation": pearson(
            predicted_energy[node_valid], actual_energy[node_valid]
        ),
        "node_energy_mae": (
            float(np.mean(node_energy_absolute_error))
            if node_valid.any()
            else float("nan")
        ),
        "node_energy_median_absolute_error": (
            float(np.median(node_energy_absolute_error))
            if node_valid.any()
            else float("nan")
        ),
        "node_energy_q75_absolute_error": (
            float(np.quantile(node_energy_absolute_error, 0.75))
            if node_valid.any()
            else float("nan")
        ),
        "actual_state_participation": actual_participation,
        "predicted_state_participation": predicted_participation,
        "state_participation_absolute_error": (
            abs(predicted_participation - actual_participation)
            if np.isfinite(actual_participation) and np.isfinite(predicted_participation)
            else float("nan")
        ),
    }


def _major_dates(target_root: Path) -> tuple[set[str], float, float]:
    summary = json.loads((target_root / "summary.json").read_text())
    if summary["target_version"] != MARKET_TRANSITION_TARGET_VERSION:
        raise ValueError("major-node target audit version does not match")
    frame = pd.read_csv(target_root / "daily_market_transition_targets.csv")
    threshold, fit_rate = fit_major_threshold(frame, summary["calibrations"])
    test = frame[frame["split"] == "test"].reset_index(drop=True)
    test = add_actual_salience(test, summary["calibrations"])
    daily = test.groupby("date", sort=True)["actual_normalized_salience"].max()
    return set(daily[daily >= threshold].index.astype(str)), threshold, fit_rate


def score_test_steps(
    model,
    features,
    steps,
    horizons,
    checkpoint_args,
    feature_args,
    edge_cache,
    device,
    batch_size,
    major_dates,
):
    stock_count = int(features.tradable_count)
    feature_count = len(features.feature_names)
    return_index = features.feature_names.index("return_1d")
    feature_indices, feature_names = systemic_state_feature_indices(
        features.feature_names
    )
    records = []
    for start in range(0, len(steps), int(batch_size)):
        selected_steps = np.asarray(steps[start : start + int(batch_size)], dtype=np.int64)
        batch = snapshot_batch(
            features, selected_steps, checkpoint_args, feature_args, edge_cache, device
        )
        with torch.no_grad():
            context, predicted = latent_trajectories(
                model, batch, horizons, checkpoint_args
            )
            rows, _ = stock_rows(
                len(selected_steps), features.node_count, stock_count, device
            )
            state_predictions = _state_predictions(
                model,
                batch,
                context,
                predicted,
                horizons,
                checkpoint_args,
                rows,
            )
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
        for position, step in enumerate(selected_steps):
            date = str(pd.Timestamp(features.dates[int(step)]).date())
            current = features.features[int(step), :stock_count]
            current_available = features.available_mask[int(step), :stock_count] > 0.5
            for horizon in horizons:
                target_step = int(step) + int(horizon)
                future = features.features[target_step, :stock_count]
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
                record: dict[str, Any] = {
                    "date": date,
                    "target_date": str(pd.Timestamp(features.dates[target_step]).date()),
                    "step": int(step),
                    "horizon": int(horizon),
                    "major_trajectory_event": bool(date in major_dates),
                }
                for method in METHODS:
                    metrics = transition_shape_metrics(
                        current,
                        future,
                        state_numpy[method][int(horizon)][position],
                        current_available,
                        future_available,
                        feature_indices,
                        node_mask=node_mask,
                    )
                    record.update(
                        {f"{method}:{name}": value for name, value in metrics.items()}
                    )
                records.append(record)
        print(
            f"scored={min(start + int(batch_size), len(steps))}/{len(steps)}",
            flush=True,
        )
    return records, feature_names


def _mean_metrics(rows, method):
    result = {}
    for name in HIGHER_IS_BETTER + LOWER_IS_BETTER:
        values = np.asarray([float(row[f"{method}:{name}"]) for row in rows])
        values = values[np.isfinite(values)]
        result[name] = float(values.mean()) if values.size else float("nan")
    return result


def summarize_shape(records, horizons):
    major = [row for row in records if row["major_trajectory_event"]]
    by_horizon = {}
    for horizon in horizons:
        selected = [row for row in major if int(row["horizon"]) == int(horizon)]
        paired = {}
        for name in HIGHER_IS_BETTER:
            delta = [
                float(row[f"rollout:{name}"]) - float(row[f"no_rollout:{name}"])
                for row in selected
            ]
            paired[name] = newey_west_mean(delta, lag=int(horizon))
        for name in LOWER_IS_BETTER:
            delta = [
                float(row[f"no_rollout:{name}"]) - float(row[f"rollout:{name}"])
                for row in selected
            ]
            paired[name] = newey_west_mean(delta, lag=int(horizon))
        by_horizon[str(int(horizon))] = {
            "rows": len(selected),
            "rollout": _mean_metrics(selected, "rollout"),
            "no_rollout": _mean_metrics(selected, "no_rollout"),
            "paired_positive_favors_rollout": paired,
        }
    return {
        "test_rows": len(records),
        "major_event_dates": len(set(row["date"] for row in major)),
        "major_event_rows": len(major),
        "major_all_horizons": {
            "rollout": _mean_metrics(major, "rollout"),
            "no_rollout": _mean_metrics(major, "no_rollout"),
        },
        "by_horizon": by_horizon,
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate full node-state transition shape on major market paths."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--target-audit-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--edge-cache-workers", type=int, default=16)
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
    major_dates, threshold, fit_rate = _major_dates(Path(args.target_audit_root))
    model, checkpoint = load_model(model_dir, device)
    model.eval()
    checkpoint_args = dict(checkpoint.get("args", {}))
    validate_future_rollout_contract(checkpoint_args, horizons, False)
    feature_args = evaluator_namespace(args)
    feature_args.edge_cache_workers = int(args.edge_cache_workers)
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    checkpoint_args.setdefault(
        "temporal_offset", checkpoint_args.get("horizon", max(horizons))
    )
    checkpoint_args.setdefault("latent_rollout_steps", 1)
    splits = _split_steps(
        features, checkpoint_args, horizons, int(args.validation_days)
    )
    test_steps = np.asarray(splits["test"], dtype=np.int64)
    if int(args.max_test_steps) > 0 and len(test_steps) > int(args.max_test_steps):
        positions = np.linspace(0, len(test_steps) - 1, int(args.max_test_steps))
        test_steps = test_steps[np.round(positions).astype(np.int64)]
    edge_cache = build_evaluation_edge_cache(
        features, test_steps, checkpoint_args, feature_args
    )
    records, feature_names = score_test_steps(
        model,
        features,
        test_steps,
        horizons,
        checkpoint_args,
        feature_args,
        edge_cache,
        device,
        int(args.batch_size),
        major_dates,
    )
    summary = {
        "status": "complete",
        "role": "major_market_full_node_transition_shape_evaluation",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "model_dir": str(model_dir),
        "parent_model_sha256": checkpoint_sha256(model_dir),
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "state_features": feature_names,
        "fit_major_event_threshold": threshold,
        "fit_major_event_rate": fit_rate,
        "metrics": summarize_shape(records, horizons),
        "test_used_for_selection": False,
        "live_orders_allowed": False,
    }
    _write_csv(output_dir / "daily_node_transition_shape.csv", records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "major_event_dates": summary["metrics"]["major_event_dates"],
                "major_all_horizons": summary["metrics"]["major_all_horizons"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
