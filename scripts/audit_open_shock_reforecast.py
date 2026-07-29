from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.audit_systemic_transition_targets import _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_open_innovation_nowcast import (
    _split_rows,
    load_forecast_state,
)
from scripts.evaluate_node_prediction import build_features_from_ckpt
from stock_v2.open_innovation import (
    build_jepa_open_innovation_design,
    build_open_sensor_design,
    shuffled_feature_block,
)
from stock_v2.surprise_reforecast import (
    SURPRISE_STATISTIC_NAMES,
    build_open_shock_trajectory,
    fit_residual_surprise_calibration,
)


TARGET_NAMES = (
    "aligned_remaining_return",
    "impact_extension",
    "market_mfe",
    "market_mae",
    "node_mfe",
    "node_mae",
)

CORE_STATE_TERMS = (
    "return_",
    "gap_open",
    "intraday_return",
    "volatility",
    "volume_z",
    "value_z",
    "range_",
    "market_",
    "beta",
    "corr",
    "drawdown",
    "breakout",
    "amihud",
)


def parse_int_list(text: str) -> tuple[int, ...]:
    values = tuple(int(value.strip()) for value in str(text).split(",") if value.strip())
    if not values:
        raise ValueError("at least one horizon is required")
    return values


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_correlation(left: np.ndarray, right: np.ndarray) -> float:
    valid = np.isfinite(left) & np.isfinite(right)
    if int(valid.sum()) < 3:
        return float("nan")
    left = left[valid].astype(np.float64)
    right = right[valid].astype(np.float64)
    if left.std() <= 1e-12 or right.std() <= 1e-12:
        return float("nan")
    return float(np.corrcoef(left, right)[0, 1])


def regression_metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    actual = np.asarray(actual, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    valid = np.isfinite(actual) & np.isfinite(prediction)
    if int(valid.sum()) < 3:
        return {"rows": int(valid.sum())}
    actual = actual[valid]
    prediction = prediction[valid]
    error = prediction - actual
    zero_sse = float(np.sum(np.square(actual)))
    model_sse = float(np.sum(np.square(error)))
    magnitude = np.abs(actual)
    weighted = float(
        np.sum(magnitude * (np.sign(actual) == np.sign(prediction)))
        / max(float(magnitude.sum()), 1e-12)
    )
    tail_threshold = float(np.quantile(magnitude, 0.80))
    tail = magnitude >= tail_threshold
    return {
        "rows": int(len(actual)),
        "correlation": _finite_correlation(actual, prediction),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(np.square(error)))),
        "mse_skill_vs_zero": (
            float(1.0 - model_sse / zero_sse) if zero_sse > 1e-16 else float("nan")
        ),
        "direction_accuracy": float(np.mean(np.sign(actual) == np.sign(prediction))),
        "impact_weighted_direction_accuracy": weighted,
        "tail_rows": int(tail.sum()),
        "tail_correlation": _finite_correlation(actual[tail], prediction[tail]),
        "tail_mae": float(np.mean(np.abs(error[tail]))),
    }


def _prepare_design(
    values: np.ndarray,
    fit_rows: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 2:
        raise ValueError("reforecast design must be a matrix")
    fit = values[fit_rows]
    center = np.nanmedian(fit, axis=0)
    center = np.where(np.isfinite(center), center, 0.0)
    filled = np.where(np.isfinite(values), values, center[None, :])
    scale = np.std(filled[fit_rows], axis=0)
    usable = np.isfinite(scale) & (scale > 1e-8)
    if not usable.any():
        raise ValueError("reforecast design has no varying fit features")
    normalized = (filled[:, usable] - center[usable]) / scale[usable]
    # Keep a fixed ridge alpha comparable as the number of causal input columns
    # changes. Without this normalization, the dual Gram diagonal grows with the
    # design width and a small shock sample can interpolate thousands of inputs.
    normalized /= np.sqrt(float(normalized.shape[1]))
    return normalized.astype(np.float64), center, usable


def _contains_core_state(name: str) -> bool:
    feature = str(name).rsplit(":", 1)[-1]
    return any(term in feature for term in CORE_STATE_TERMS)


def select_compact_sensor_design(values, names):
    selected = []
    for index, raw_name in enumerate(names):
        name = str(raw_name)
        keep = (
            name.startswith("open_gap_raw_")
            or name.startswith("open_stock_mean:")
            or (name.startswith("open_stock_std:") and _contains_core_state(name))
            or (
                name.startswith(("previous_stock_mean:", "previous_stock_std:"))
                and _contains_core_state(name)
            )
            or name.startswith("open_external_value:")
        )
        if keep:
            selected.append(index)
    if not selected:
        raise ValueError("compact open-sensor design selected no features")
    return np.asarray(values)[:, selected], tuple(str(names[index]) for index in selected)


def select_compact_jepa_design(values, names):
    selected = []
    for index, raw_name in enumerate(names):
        name = str(raw_name)
        keep = (
            (
                name.startswith(
                    (
                        "jepa_forecast_stock_mean:",
                        "jepa_forecast_stock_std:",
                        "jepa_delta_stock_mean:",
                        "jepa_delta_stock_std:",
                    )
                )
                and _contains_core_state(name)
            )
            or (
                name.startswith(("open_innovation_stock_mean:", "open_innovation_stock_std:"))
                and name.endswith(":gap_open")
            )
            or name.startswith(
                (
                    "jepa_forecast_external_value:",
                    "open_innovation_external_value:",
                )
            )
        )
        if keep:
            selected.append(index)
    if not selected:
        raise ValueError("compact JEPA design selected no features")
    return np.asarray(values)[:, selected], tuple(str(names[index]) for index in selected)


def ridge_predictions(
    values: np.ndarray,
    target: np.ndarray,
    fit_rows: np.ndarray,
    *,
    alpha: float,
) -> np.ndarray:
    target = np.asarray(target, dtype=np.float64)
    fit_valid = fit_rows[np.isfinite(target[fit_rows])]
    if len(fit_valid) < 20:
        raise ValueError("reforecast ridge requires at least twenty fit targets")
    design, _center, _usable = _prepare_design(values, fit_valid)
    x = design[fit_valid]
    y = target[fit_valid]
    target_center = float(np.mean(y))
    centered_y = y - target_center
    gram = x @ x.T
    dual = np.linalg.solve(
        gram + float(alpha) * np.eye(len(fit_valid), dtype=np.float64),
        centered_y,
    )
    weights = x.T @ dual
    return target_center + design @ weights


def _build_close_paths(
    features,
    steps: np.ndarray,
    horizons: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    stock_count = int(features.tradable_count)
    current_steps = np.asarray(steps, dtype=np.int64) + 1
    gap_index = features.feature_names.index("gap_open")
    gap = np.asarray(
        features.raw_features[current_steps, :stock_count, gap_index],
        dtype=np.float64,
    )
    gap_valid = (
        features.available_mask[current_steps, :stock_count, gap_index] > 0.5
    ) & np.isfinite(gap)
    session_open = np.asarray(features.open[current_steps, :stock_count], dtype=np.float64)
    close_path = np.full(
        (len(steps), len(horizons), stock_count), np.nan, dtype=np.float64
    )
    close_valid = np.zeros_like(close_path, dtype=bool)
    for position, horizon in enumerate(horizons):
        close_steps = current_steps + int(horizon) - 1
        if int(close_steps.max()) >= len(features.dates):
            raise ValueError("reforecast close path exceeds available dates")
        close = np.asarray(features.close[close_steps, :stock_count], dtype=np.float64)
        valid = (
            gap_valid
            & np.isfinite(session_open)
            & (session_open > 0.0)
            & np.isfinite(close)
            & (close > 0.0)
        )
        close_path[:, position] = np.divide(
            close,
            session_open,
            out=np.full_like(close, np.nan),
            where=valid,
        ) - 1.0
        close_valid[:, position] = valid
    return gap, gap_valid, close_path, close_valid


def _target_matrix(trajectory) -> Mapping[str, np.ndarray]:
    return {
        "aligned_remaining_return": trajectory.aligned_remaining_return,
        "impact_extension": trajectory.impact_extension,
        "market_mfe": trajectory.market_mfe,
        "market_mae": trajectory.market_mae,
        "node_mfe": trajectory.node_mfe,
        "node_mae": trajectory.node_mae,
    }


def _evaluate_variants(
    variants: Mapping[str, np.ndarray],
    targets: Mapping[str, np.ndarray],
    split_rows: Mapping[str, np.ndarray],
    shock_mask: np.ndarray,
    horizons: Sequence[int],
    *,
    ridge_alpha: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    fit_shocks = split_rows["fit"][shock_mask[split_rows["fit"]]]
    if len(fit_shocks) < 20:
        raise ValueError("too few fit surprise events for conditional reforecast")
    for target_name, values in targets.items():
        target_output: dict[str, Any] = {}
        for horizon_index, horizon in enumerate(horizons):
            actual = np.asarray(values[:, horizon_index], dtype=np.float64)
            predictions = {
                name: ridge_predictions(
                    design,
                    actual,
                    fit_shocks,
                    alpha=float(ridge_alpha),
                )
                for name, design in variants.items()
            }
            horizon_output: dict[str, Any] = {}
            for split in ("validation", "test"):
                rows = split_rows[split]
                selected = rows[shock_mask[rows] & np.isfinite(actual[rows])]
                horizon_output[split] = {
                    name: regression_metrics(actual[selected], prediction[selected])
                    for name, prediction in predictions.items()
                }
                candidate = horizon_output[split]["open_sensors_plus_jepa"]
                baseline = horizon_output[split]["open_sensors"]
                placebo = horizon_output[split]["open_sensors_plus_shuffled_jepa"]
                horizon_output[split]["candidate_delta"] = {
                    "mse_skill_vs_direct": float(
                        candidate.get("mse_skill_vs_zero", float("nan"))
                        - baseline.get("mse_skill_vs_zero", float("nan"))
                    ),
                    "mse_skill_vs_placebo": float(
                        candidate.get("mse_skill_vs_zero", float("nan"))
                        - placebo.get("mse_skill_vs_zero", float("nan"))
                    ),
                    "mae_improvement_vs_direct": float(
                        baseline.get("mae", float("nan"))
                        - candidate.get("mae", float("nan"))
                    ),
                }
            target_output[str(int(horizon))] = horizon_output
        output[target_name] = target_output
    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit residual-conditioned forecasts after an observed KRX-open shock."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--forecast-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--surprise-quantile", type=float, default=0.80)
    parser.add_argument("--min-nodes", type=int, default=100)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--feature-mode", choices=("compact", "raw"), default="compact")
    parser.add_argument("--placebo-seed", type=int, default=701)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    forecast_cache_dir = Path(args.forecast_cache_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    horizons = parse_int_list(args.horizons)

    checkpoint = torch.load(
        model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    checkpoint_args = dict(checkpoint.get("args", {}))
    args.split_horizons = args.horizons
    feature_args = evaluator_namespace(args)
    feature_args.horizons = args.horizons
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(
        features, checkpoint_args, list(horizons), int(args.validation_days)
    )
    steps, split_rows = _split_rows(splits)
    predicted_state, cache_rows, cache_contract = load_forecast_state(
        forecast_cache_dir, model_dir, steps
    )
    stock_count = int(features.tradable_count)
    current_steps = steps + 1
    eligible = np.asarray(cache_contract["eligible_indices"], dtype=np.int64)
    gap_global = int(features.feature_names.index("gap_open"))
    gap_positions = np.flatnonzero(eligible == gap_global)
    if len(gap_positions) != 1:
        raise ValueError("JEPA forecast cache must contain exactly one gap_open feature")
    gap_position = int(gap_positions[0])
    current_gap_state = features.features[
        current_steps, :stock_count, gap_global
    ].astype(np.float32)
    current_gap_valid = (
        features.available_mask[current_steps, :stock_count, gap_global] > 0.5
    )
    predicted_gap_state = predicted_state[:, :stock_count, gap_position]
    residual = (current_gap_state - predicted_gap_state)[:, :, None]
    residual_valid = (
        current_gap_valid & np.isfinite(predicted_gap_state)
    )[:, :, None]
    calibration, surprise = fit_residual_surprise_calibration(
        residual,
        residual_valid,
        split_rows["fit"],
        stock_count=stock_count,
        threshold_quantile=float(args.surprise_quantile),
        min_nodes=int(args.min_nodes),
        edge_index=features.static_edge_index,
        edge_weight=features.static_edge_weight,
    )

    gap, gap_valid, close_path, close_valid = _build_close_paths(
        features, steps, horizons
    )
    trajectory = build_open_shock_trajectory(
        gap,
        close_path,
        gap_valid,
        close_valid,
        horizons=horizons,
        min_nodes=int(args.min_nodes),
    )
    shock_mask = surprise.is_surprise & np.isfinite(trajectory.open_direction)
    sensor = build_open_sensor_design(features, steps)
    jepa = build_jepa_open_innovation_design(
        features,
        steps,
        predicted_state,
        eligible,
    )
    surprise_values = np.where(np.isfinite(surprise.values), surprise.values, 0.0)
    if args.feature_mode == "compact":
        sensor_values, sensor_names = select_compact_sensor_design(
            sensor.values, sensor.feature_names
        )
        jepa_values, jepa_names = select_compact_jepa_design(
            jepa.values, jepa.feature_names
        )
    else:
        sensor_values = sensor.values
        sensor_names = tuple(sensor.feature_names)
        jepa_values = jepa.values
        jepa_names = tuple(jepa.feature_names)
    candidate_jepa = np.concatenate((jepa_values, surprise_values), axis=1)
    shuffled = shuffled_feature_block(
        candidate_jepa,
        split_rows,
        seed=int(args.placebo_seed),
    )
    variants = {
        "open_sensors": sensor_values,
        "open_sensors_plus_jepa": np.concatenate(
            (sensor_values, candidate_jepa), axis=1
        ),
        "open_sensors_plus_shuffled_jepa": np.concatenate(
            (sensor_values, shuffled), axis=1
        ),
    }
    targets = _target_matrix(trajectory)
    results = _evaluate_variants(
        variants,
        targets,
        split_rows,
        shock_mask,
        horizons,
        ridge_alpha=float(args.ridge_alpha),
    )

    intraday_position = np.flatnonzero(
        eligible == int(features.feature_names.index("intraday_return"))
    )
    stale_metrics: dict[str, Any] = {}
    if len(intraday_position) == 1:
        feature_index = int(features.feature_names.index("intraday_return"))
        predicted_intraday = (
            predicted_state[:, :stock_count, int(intraday_position[0])]
            * float(features.train_std[feature_index])
            + float(features.train_mean[feature_index])
        )
        stale_market = np.nanmedian(predicted_intraday, axis=1)
        stale_aligned = stale_market * trajectory.open_direction
        actual = trajectory.aligned_remaining_return[:, 0]
        for split in ("validation", "test"):
            rows = split_rows[split]
            selected = rows[shock_mask[rows] & np.isfinite(actual[rows])]
            stale_metrics[split] = regression_metrics(
                actual[selected], stale_aligned[selected]
            )

    event_counts = {
        name: int(shock_mask[rows].sum()) for name, rows in split_rows.items()
    }
    report = {
        "schema_version": 1,
        "status": "complete",
        "role": "diagnostic_only_observed_open_shock_conditional_reforecast",
        "checkpoint_sha256": sha256_file(model_dir / "graph_jepa_real.pt"),
        "forecast_cache_checkpoint_sha256": cache_contract["checkpoint_sha256"],
        "forecast_cache_rows_sha256": hashlib.sha256(
            np.ascontiguousarray(cache_rows).view(np.uint8)
        ).hexdigest(),
        "horizons": list(horizons),
        "stocks": stock_count,
        "rows": int(len(steps)),
        "split_rows": {name: int(len(rows)) for name, rows in split_rows.items()},
        "surprise_contract": {
            "residual": "observed_normalized_gap_open_minus_previous_close_h1_jepa_forecast",
            "fit_rows_only_calibration": True,
            "feature_center": calibration.feature_center.tolist(),
            "feature_scale": calibration.feature_scale.tolist(),
            "energy_threshold": float(calibration.energy_threshold),
            "threshold_quantile": float(calibration.threshold_quantile),
            "summary_features": list(SURPRISE_STATISTIC_NAMES),
            "event_counts": event_counts,
        },
        "target_contract": {
            "decision_time": "current_krx_open_after_gap_is_observed",
            "targets": list(TARGET_NAMES),
            "future_close_path_only": True,
            "minute_path_fabricated": False,
            "input_after_decision_time": False,
        },
        "model_contract": {
            "fit_only_on_surprise_rows": True,
            "fixed_ridge_alpha": float(args.ridge_alpha),
            "test_used_for_selection": False,
            "placebo_shuffled_within_each_split": True,
            "feature_mode": str(args.feature_mode),
            "raw_sensor_features": int(sensor.values.shape[1]),
            "selected_sensor_features": int(sensor_values.shape[1]),
            "raw_jepa_features": int(jepa.values.shape[1]),
            "selected_jepa_features": int(jepa_values.shape[1]),
            "selected_sensor_feature_names": list(sensor_names),
            "selected_jepa_feature_names": list(jepa_names),
            "design_scaled_by_inverse_sqrt_width": True,
        },
        "stale_h1_jepa": stale_metrics,
        "results": results,
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "EXPERIMENT_COMPLETE").touch()
    print(
        json.dumps(
            {
                "event_counts": event_counts,
                "stale_h1_jepa": stale_metrics,
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
