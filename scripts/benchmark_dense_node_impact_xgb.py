from __future__ import annotations

import argparse
import csv
import hashlib
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

from scripts.audit_systemic_transition_targets import _actual_rows, _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_latent_trajectory_path_head import HORIZON_WEIGHTS
from scripts.benchmark_qlib_lgb import load_context_matrix, validate_contract
from scripts.benchmark_systemic_transition_head import build_target_contracts
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.run_real_backtest import parse_int_list
from stock_v2.systemic_transition import (
    binary_ranking_metrics,
    derived_subtype_scores,
    event_labels,
)


VARIANTS = ("dense_absolute", "dense_impact_weighted")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _correlation(left: Sequence[float], right: Sequence[float]) -> float:
    x = np.asarray(left, dtype=np.float64)
    y = np.asarray(right, dtype=np.float64)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(valid.sum()) < 3:
        return float("nan")
    x = x[valid]
    y = y[valid]
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def aggregate_node_path_predictions(
    prediction: np.ndarray,
    available: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    prediction = np.asarray(prediction, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    if prediction.ndim != 2 or prediction.shape != available.shape:
        raise ValueError("node predictions and availability must be aligned panels")
    valid = available & np.isfinite(prediction)
    counts = valid.sum(axis=1)
    values = np.where(valid, prediction, 0.0)
    market_return = np.divide(
        values.sum(axis=1),
        counts,
        out=np.full(len(prediction), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    signs = np.where(valid, np.sign(prediction), 0.0)
    breadth = np.divide(
        signs.sum(axis=1),
        counts,
        out=np.full(len(prediction), np.nan, dtype=np.float64),
        where=counts > 0,
    )
    return market_return, breadth, counts.astype(np.int64)


def target_aligned_availability(
    current_available: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    current_available = np.asarray(current_available, dtype=bool)
    target = np.asarray(target)
    if current_available.shape != target.shape:
        raise ValueError("current availability and target panels must align")
    return current_available & np.isfinite(target)


def validate_target_panel_alignment(
    target: np.ndarray,
    current_available: np.ndarray,
    actual_rows: Sequence[Mapping[str, Any]],
    *,
    atol: float = 1e-7,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    if target.ndim != 2 or len(actual_rows) != len(target):
        raise ValueError("target panel and actual rows must align by date")
    available = target_aligned_availability(current_available, target)
    market_return, breadth, counts = aggregate_node_path_predictions(target, available)
    expected_return = np.asarray(
        [float(row["market_return"]) for row in actual_rows], dtype=np.float64
    )
    expected_breadth = np.asarray(
        [float(row["breadth"]) for row in actual_rows], dtype=np.float64
    )
    if np.any(counts <= 0):
        raise ValueError("target panel contains a date without an observable stock")
    if not np.allclose(market_return, expected_return, rtol=0.0, atol=atol):
        raise ValueError("dense target panel does not reconstruct systemic market return")
    if not np.allclose(breadth, expected_breadth, rtol=0.0, atol=atol):
        raise ValueError("dense target panel does not reconstruct systemic breadth")
    return available


def impact_day_weights(
    rows: Sequence[Mapping[str, Any]],
    calibration,
    *,
    maximum_extra_weight: float = 3.0,
) -> np.ndarray:
    scores = np.asarray(
        [
            derived_subtype_scores(row, calibration)["broad_selloff"]
            for row in rows
        ],
        dtype=np.float64,
    )
    severity = np.clip(np.where(np.isfinite(scores), scores, 0.0), 0.0, 3.0) / 3.0
    labels = np.asarray(
        [event_labels(row, calibration)["broad_selloff"] for row in rows],
        dtype=np.float64,
    )
    return (
        1.0
        + float(maximum_extra_weight) * np.maximum(severity, labels)
    ).astype(np.float32)


def evaluate_dense_predictions(
    prediction: np.ndarray,
    available: np.ndarray,
    actual_rows: Sequence[Mapping[str, Any]],
    contract,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(actual_rows) != len(prediction):
        raise ValueError("actual rows and prediction dates must align")
    predicted_return, predicted_breadth, counts = aggregate_node_path_predictions(
        prediction, available
    )
    actual_return = np.asarray(
        [float(row["market_return"]) for row in actual_rows], dtype=np.float64
    )
    actual_breadth = np.asarray(
        [float(row["breadth"]) for row in actual_rows], dtype=np.float64
    )
    labels = np.asarray(
        [
            event_labels(row, contract.calibration)["broad_selloff"]
            for row in actual_rows
        ],
        dtype=bool,
    )
    scores = np.asarray(
        [
            derived_subtype_scores(
                {"market_return": market, "breadth": breadth},
                contract.calibration,
            )["broad_selloff"]
            for market, breadth in zip(predicted_return, predicted_breadth)
        ],
        dtype=np.float64,
    )
    ranking = binary_ranking_metrics(
        labels,
        scores,
        selection_rate=max(float(contract.event_fit_rate[1]), 1e-6),
    )
    valid_direction = np.isfinite(predicted_return) & np.isfinite(actual_return)
    direction_accuracy = (
        float(
            (
                np.sign(predicted_return[valid_direction])
                == np.sign(actual_return[valid_direction])
            ).mean()
        )
        if valid_direction.any()
        else float("nan")
    )
    event_direction = valid_direction & labels
    broad_direction_accuracy = (
        float(
            (
                np.sign(predicted_return[event_direction])
                == np.sign(actual_return[event_direction])
            ).mean()
        )
        if event_direction.any()
        else float("nan")
    )
    metrics = {
        **ranking,
        "market_return_correlation": _correlation(predicted_return, actual_return),
        "breadth_correlation": _correlation(predicted_breadth, actual_breadth),
        "market_direction_accuracy": direction_accuracy,
        "broad_selloff_direction_accuracy": broad_direction_accuracy,
        "mean_observed_stocks": float(counts.mean()),
    }
    daily = [
        {
            "date": str(row["date"]),
            "actual_market_return": float(actual_return[index]),
            "predicted_market_return": float(predicted_return[index]),
            "actual_breadth": float(actual_breadth[index]),
            "predicted_breadth": float(predicted_breadth[index]),
            "actual_broad_selloff": bool(labels[index]),
            "broad_selloff_score": float(scores[index]),
            "observed_stocks": int(counts[index]),
        }
        for index, row in enumerate(actual_rows)
    ]
    return metrics, daily


def weighted_variant_score(horizons: Mapping[str, Mapping[str, Any]]) -> float:
    total = 0.0
    weight_sum = 0.0
    for raw_horizon, row in horizons.items():
        event_rate = max(float(row["event_rate"]), 1e-8)
        terms = (
            0.35 * np.clip(2.0 * (float(row["roc_auc"]) - 0.5), -1.0, 1.0)
            + 0.20
            * np.clip(float(row["average_precision"]) / event_rate - 1.0, -1.0, 1.0)
            + 0.20 * np.clip(float(row["market_return_correlation"]), -1.0, 1.0)
            + 0.10 * np.clip(float(row["breadth_correlation"]), -1.0, 1.0)
            + 0.15
            * np.clip(2.0 * (float(row["market_direction_accuracy"]) - 0.5), -1.0, 1.0)
        )
        weight = float(HORIZON_WEIGHTS.get(int(raw_horizon), 1.0))
        total += weight * float(terms)
        weight_sum += weight
    return float(total / weight_sum)


def dense_impact_gate(horizons: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    def weighted(name: str) -> float:
        total = 0.0
        weight_sum = 0.0
        for raw_horizon, row in horizons.items():
            weight = float(HORIZON_WEIGHTS.get(int(raw_horizon), 1.0))
            total += weight * float(row[name])
            weight_sum += weight
        return float(total / weight_sum)

    auc = weighted("roc_auc")
    recall = weighted("recall_at_selection_rate")
    correlation = weighted("market_return_correlation")
    direction = weighted("market_direction_accuracy")
    minimum_auc = min(float(row["roc_auc"]) for row in horizons.values())
    checks = {
        "weighted_broad_selloff_auc_at_least_0_60": auc >= 0.60,
        "weighted_broad_selloff_recall_at_least_0_25": recall >= 0.25,
        "weighted_market_return_correlation_at_least_0_10": correlation >= 0.10,
        "weighted_market_direction_accuracy_at_least_0_55": direction >= 0.55,
        "every_horizon_broad_selloff_auc_at_least_0_52": minimum_auc >= 0.52,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "values": {
            "weighted_broad_selloff_auc": auc,
            "weighted_broad_selloff_recall": recall,
            "weighted_market_return_correlation": correlation,
            "weighted_market_direction_accuracy": direction,
            "minimum_horizon_broad_selloff_auc": minimum_auc,
        },
    }


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train GPU XGBoost on dense absolute node returns for systemic breadth."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--variants", default=",".join(VARIANTS))
    parser.add_argument("--num-boost-round", type=int, default=600)
    parser.add_argument("--early-stopping-rounds", type=int, default=50)
    parser.add_argument("--max-depth", type=int, default=8)
    parser.add_argument("--eta", type=float, default=0.04)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    variants = tuple(value.strip() for value in args.variants.split(",") if value.strip())
    if not variants or any(value not in VARIANTS for value in variants):
        raise ValueError(f"variants must be selected from {VARIANTS}")
    horizons = tuple(parse_int_list(args.horizons))
    random.seed(int(args.seed))
    np.random.seed(int(args.seed))

    model_dir = Path(args.model_dir).resolve()
    bundle_dir = Path(args.bundle_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = model_dir / "graph_jepa_real.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    checkpoint_args = dict(checkpoint.get("args", {}))
    feature_args = evaluator_namespace(args)
    feature_args.horizons = args.horizons
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(features, checkpoint_args, horizons, validation_days=126)
    all_steps = np.unique(np.concatenate(tuple(splits.values()))).astype(np.int64)

    contract = validate_contract(bundle_dir)
    if contract.get("checkpoint_sha256") != sha256_file(checkpoint_path):
        raise ValueError("PIT bundle checkpoint differs from the requested JEPA fold")
    if tuple(int(value) for value in contract["horizons"]) != horizons:
        raise ValueError("PIT bundle horizons differ from the benchmark")
    arrays = np.load(bundle_dir / str(contract["arrays_file"]), allow_pickle=False)
    dates = np.asarray(arrays["dates"], dtype="datetime64[ns]")
    labels = np.asarray(arrays["labels"], dtype=np.float32)
    bundle_current_available = np.asarray(arrays["current_available"], dtype=bool)
    expected_dates = np.asarray(features.dates[all_steps], dtype="datetime64[ns]")
    if not np.array_equal(dates, expected_dates):
        raise ValueError("PIT bundle dates differ from rebuilt fold splits")
    stock_count = int(features.tradable_count)
    if int(contract["stocks"]) != stock_count:
        raise ValueError("PIT bundle stock count differs from the feature panel")
    context = load_context_matrix(bundle_dir, contract).reshape(
        len(dates), stock_count, int(contract["feature_count"])
    )
    horizon_position = {
        int(value): index for index, value in enumerate(contract["horizons"])
    }
    return_index = features.feature_names.index("return_1d")
    current_return_available = (
        features.available_mask[all_steps, :stock_count, return_index] > 0.5
    )
    systemic_available: dict[int, np.ndarray] = {}
    for horizon in horizons:
        horizon_labels = labels[horizon_position[int(horizon)]]
        rebuilt_labels = np.asarray(
            features.target_return_paths[int(horizon)][all_steps, :stock_count],
            dtype=np.float32,
        )
        if not np.array_equal(horizon_labels, rebuilt_labels, equal_nan=True):
            raise ValueError(f"PIT bundle labels differ from rebuilt horizon {horizon}")
        future_return_available = (
            features.available_mask[
                all_steps + int(horizon), :stock_count, return_index
            ]
            > 0.5
        )
        exact_available = target_aligned_availability(
            current_return_available & future_return_available,
            horizon_labels,
        )
        if np.any(exact_available & ~bundle_current_available):
            raise ValueError("systemic target mask exceeds PIT bundle current availability")
        systemic_available[int(horizon)] = exact_available
    step_to_date_position = {int(step): index for index, step in enumerate(all_steps)}
    split_positions = {
        name: np.asarray([step_to_date_position[int(step)] for step in steps], dtype=np.int64)
        for name, steps in splits.items()
    }
    raw_rows = {
        name: _actual_rows(features, steps, horizons, name)
        for name, steps in splits.items()
    }
    contracts = build_target_contracts(raw_rows["fit"], horizons)
    rows_by_split_horizon = {
        split: {
            int(horizon): [
                row for row in rows if int(row["horizon"]) == int(horizon)
            ]
            for horizon in horizons
        }
        for split, rows in raw_rows.items()
    }

    import xgboost as xgb

    parameters = {
        "objective": "reg:pseudohubererror",
        "eval_metric": "rmse",
        "tree_method": "hist",
        "device": str(args.device),
        "max_depth": int(args.max_depth),
        "eta": float(args.eta),
        "min_child_weight": 20.0,
        "subsample": 0.85,
        "colsample_bytree": 0.80,
        "lambda": 20.0,
        "alpha": 1.0,
        "max_bin": 256,
        "seed": int(args.seed),
        "nthread": 16,
    }
    results: dict[str, Any] = {}
    all_daily: list[dict[str, Any]] = []
    for variant in variants:
        variant_dir = output_dir / variant
        variant_dir.mkdir(parents=True, exist_ok=True)
        split_metrics = {"validation": {}, "test": {}}
        model_contracts = {}
        for horizon in horizons:
            fit_positions = split_positions["fit"]
            validation_positions = split_positions["validation"]
            test_positions = split_positions["test"]
            horizon_labels = labels[horizon_position[int(horizon)]]
            fit_label_panel = horizon_labels[fit_positions]
            validation_label_panel = horizon_labels[validation_positions]
            fit_available_panel = validate_target_panel_alignment(
                fit_label_panel,
                systemic_available[int(horizon)][fit_positions],
                rows_by_split_horizon["fit"][int(horizon)],
            )
            validation_available_panel = validate_target_panel_alignment(
                validation_label_panel,
                systemic_available[int(horizon)][validation_positions],
                rows_by_split_horizon["validation"][int(horizon)],
            )
            fit_label = fit_label_panel.reshape(-1)
            fit_available = fit_available_panel.reshape(-1)
            validation_label = validation_label_panel.reshape(-1)
            validation_available = validation_available_panel.reshape(-1)
            fit_mean = float(fit_label[fit_available].mean())
            fit_std = float(fit_label[fit_available].std())
            if not np.isfinite(fit_std) or fit_std < 1e-6:
                raise ValueError(f"invalid fit return scale for horizon {horizon}")
            fit_target = ((fit_label[fit_available] - fit_mean) / fit_std).astype(np.float32)
            validation_target = (
                (validation_label[validation_available] - fit_mean) / fit_std
            ).astype(np.float32)
            day_weight = impact_day_weights(
                rows_by_split_horizon["fit"][int(horizon)],
                contracts[int(horizon)].calibration,
            )
            row_weight = np.repeat(day_weight, stock_count)[fit_available]
            if variant == "dense_absolute":
                row_weight = np.ones_like(row_weight)

            fit_matrix = np.asarray(
                context[fit_positions].reshape(-1, context.shape[-1])[fit_available],
                dtype=np.float32,
            )
            validation_matrix = np.asarray(
                context[validation_positions]
                .reshape(-1, context.shape[-1])[validation_available],
                dtype=np.float32,
            )
            dtrain = xgb.QuantileDMatrix(
                fit_matrix,
                label=fit_target,
                weight=row_weight,
                max_bin=int(parameters["max_bin"]),
            )
            dvalidation = xgb.QuantileDMatrix(
                validation_matrix,
                label=validation_target,
                ref=dtrain,
                max_bin=int(parameters["max_bin"]),
            )
            booster = xgb.train(
                parameters,
                dtrain,
                num_boost_round=int(args.num_boost_round),
                evals=[(dtrain, "fit"), (dvalidation, "validation")],
                early_stopping_rounds=int(args.early_stopping_rounds),
                verbose_eval=50,
            )
            model_path = variant_dir / f"xgboost_absolute_h{int(horizon)}.json"
            booster.save_model(model_path)
            iteration_range = (0, int(booster.best_iteration) + 1)
            for split, positions in (
                ("validation", validation_positions),
                ("test", test_positions),
            ):
                split_matrix = np.asarray(
                    context[positions].reshape(-1, context.shape[-1]),
                    dtype=np.float32,
                )
                normalized_prediction = booster.inplace_predict(
                    split_matrix, iteration_range=iteration_range
                )
                prediction = (
                    normalized_prediction.reshape(len(positions), stock_count) * fit_std
                    + fit_mean
                )
                split_available = validate_target_panel_alignment(
                    horizon_labels[positions],
                    systemic_available[int(horizon)][positions],
                    rows_by_split_horizon[split][int(horizon)],
                )
                metrics, daily = evaluate_dense_predictions(
                    prediction,
                    split_available,
                    rows_by_split_horizon[split][int(horizon)],
                    contracts[int(horizon)],
                )
                split_metrics[split][str(int(horizon))] = metrics
                for row in daily:
                    row.update(
                        {
                            "variant": variant,
                            "split": split,
                            "horizon": int(horizon),
                        }
                    )
                all_daily.extend(daily)
            model_contracts[str(int(horizon))] = {
                "model_sha256": sha256_file(model_path),
                "best_iteration": int(booster.best_iteration),
                "fit_label_mean": fit_mean,
                "fit_label_std": fit_std,
                "fit_rows": int(fit_available.sum()),
                "validation_rows": int(validation_available.sum()),
            }
            del dtrain, dvalidation, booster, fit_matrix, validation_matrix

        validation_score = weighted_variant_score(split_metrics["validation"])
        results[variant] = {
            "validation_score": validation_score,
            "metrics": split_metrics,
            "test_gate": dense_impact_gate(split_metrics["test"]),
            "models": model_contracts,
        }

    selected_variant = max(
        results, key=lambda name: float(results[name]["validation_score"])
    )
    summary = {
        "schema_version": 1,
        "status": "complete",
        "role": "research_only_dense_absolute_node_impact_xgboost",
        "framework": "XGBoost GPU histogram",
        "xgboost_version": str(xgb.__version__),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "bundle_contract_sha256": contract["bundle_contract_sha256"],
        "stocks": stock_count,
        "features": int(contract["feature_count"]),
        "horizons": list(horizons),
        "variants": results,
        "validation_selected_variant": selected_variant,
        "selected_test_gate": results[selected_variant]["test_gate"],
        "selection_rule": "maximum fixed broad-impact validation score",
        "node_target_mask": (
            "current_and_future_return_1d_available_and_finite_entry_path_label"
        ),
        "test_used_for_selection": False,
        "live_orders_allowed": False,
        "parameters": parameters,
    }
    _write_csv(
        output_dir / "daily_metrics.csv",
        sorted(
            all_daily,
            key=lambda row: (
                row["variant"],
                row["split"],
                row["date"],
                row["horizon"],
            ),
        ),
    )
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "EXPERIMENT_COMPLETE").touch()
    print(
        json.dumps(
            {
                "status": "complete",
                "selected_variant": selected_variant,
                "selected_test_gate": summary["selected_test_gate"],
                "live_orders_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
