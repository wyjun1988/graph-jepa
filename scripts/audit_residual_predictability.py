from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.benchmark_direct_baselines import newey_west_mean
from scripts.evaluate_causal_innovation_filter import (
    REQUIRED_HORIZONS,
    build_parser as build_data_parser,
    sha256_file,
)
from scripts.evaluate_node_prediction import (
    as_namespace,
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    future_state_metrics,
    graph_edge_kwargs,
    load_model,
    select_steps,
    validate_future_rollout_contract,
    write_csv,
)
from scripts.run_real_backtest import parse_int_list, rollout_steps_for_offset
from stock_v2.real_features import make_real_snapshot


AUDIT_CANDIDATES = ("bias_only", "node_ar", "common_ar", "hybrid_ar")
DYNAMIC_CANDIDATES = ("node_ar", "common_ar", "hybrid_ar")
SELECTION_HORIZONS = (2, 3)


def _nanmedian(values: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmedian(values, axis=axis).astype(np.float32)


def fit_residual_models(
    errors: np.ndarray,
    horizon: int,
    fit_indices: np.ndarray,
    *,
    ridge: float,
    min_samples: int,
    gain_clip: float,
    bias_clip: float,
) -> dict[str, np.ndarray | list[int]]:
    """Fit shared feature-wise gains using only already-matured residuals."""

    if errors.ndim != 3:
        raise ValueError("residual errors must be [time, node, feature]")
    horizon = int(horizon)
    fit_indices = np.asarray(fit_indices, dtype=np.int64)
    if horizon < 1 or len(fit_indices) == 0:
        raise ValueError("residual fit requires a horizon and fit rows")
    if np.any(fit_indices - horizon < 0):
        raise ValueError("residual predictors must already be mature")
    if ridge < 0.0 or min_samples < 3:
        raise ValueError("invalid residual regression regularization")
    if gain_clip <= 0.0 or bias_clip <= 0.0:
        raise ValueError("residual coefficient clips must be positive")

    target = errors[fit_indices].astype(np.float64, copy=False)
    node_input = errors[fit_indices - horizon].astype(np.float64, copy=False)
    common_input = _nanmedian(node_input, axis=1).astype(
        np.float64,
        copy=False,
    )
    feature_count = errors.shape[2]
    bias = np.zeros(feature_count, dtype=np.float32)
    node_gain = np.zeros(feature_count, dtype=np.float32)
    common_gain = np.zeros(feature_count, dtype=np.float32)
    hybrid_node_gain = np.zeros(feature_count, dtype=np.float32)
    hybrid_common_gain = np.zeros(feature_count, dtype=np.float32)
    counts = {
        "bias": np.zeros(feature_count, dtype=np.int64),
        "node": np.zeros(feature_count, dtype=np.int64),
        "common": np.zeros(feature_count, dtype=np.int64),
        "hybrid": np.zeros(feature_count, dtype=np.int64),
    }

    for feature_index in range(feature_count):
        y = target[:, :, feature_index]
        valid_y = np.isfinite(y)
        counts["bias"][feature_index] = int(valid_y.sum())
        if counts["bias"][feature_index] >= min_samples:
            bias[feature_index] = np.float32(
                np.clip(np.mean(y[valid_y]), -bias_clip, bias_clip)
            )
        residual = y - float(bias[feature_index])

        x_node = node_input[:, :, feature_index]
        valid_node = valid_y & np.isfinite(x_node)
        counts["node"][feature_index] = int(valid_node.sum())
        if counts["node"][feature_index] >= min_samples:
            x = x_node[valid_node]
            r = residual[valid_node]
            denominator = float(np.mean(x * x) + ridge)
            node_gain[feature_index] = np.float32(
                np.clip(np.mean(x * r) / denominator, -gain_clip, gain_clip)
            )

        x_common_daily = common_input[:, feature_index]
        x_common = np.broadcast_to(x_common_daily[:, None], y.shape)
        valid_common = valid_y & np.isfinite(x_common)
        counts["common"][feature_index] = int(valid_common.sum())
        if counts["common"][feature_index] >= min_samples:
            x = x_common[valid_common]
            r = residual[valid_common]
            denominator = float(np.mean(x * x) + ridge)
            common_gain[feature_index] = np.float32(
                np.clip(np.mean(x * r) / denominator, -gain_clip, gain_clip)
            )

        valid_hybrid = valid_node & np.isfinite(x_common)
        counts["hybrid"][feature_index] = int(valid_hybrid.sum())
        if counts["hybrid"][feature_index] >= min_samples:
            node_values = x_node[valid_hybrid]
            common_values = x_common[valid_hybrid]
            r = residual[valid_hybrid]
            design = np.stack((node_values, common_values), axis=1)
            covariance = design.T @ design / len(design)
            covariance.flat[::3] += ridge
            right_hand_side = design.T @ r / len(design)
            try:
                coefficients = np.linalg.solve(covariance, right_hand_side)
            except np.linalg.LinAlgError:
                coefficients = np.zeros(2, dtype=np.float64)
            coefficients = np.clip(coefficients, -gain_clip, gain_clip)
            hybrid_node_gain[feature_index] = np.float32(coefficients[0])
            hybrid_common_gain[feature_index] = np.float32(coefficients[1])

    return {
        "bias": bias,
        "node_gain": node_gain,
        "common_gain": common_gain,
        "hybrid_node_gain": hybrid_node_gain,
        "hybrid_common_gain": hybrid_common_gain,
        "bias_counts": counts["bias"].tolist(),
        "node_counts": counts["node"].tolist(),
        "common_counts": counts["common"].tolist(),
        "hybrid_counts": counts["hybrid"].tolist(),
    }


def residual_correction(
    previous_error: np.ndarray,
    coefficients: dict[str, np.ndarray | list[int]],
    candidate: str,
    *,
    correction_clip: float,
) -> np.ndarray:
    if candidate not in AUDIT_CANDIDATES:
        raise ValueError(f"unknown residual audit candidate: {candidate}")
    previous_error = np.asarray(previous_error, dtype=np.float32)
    bias = np.asarray(coefficients["bias"], dtype=np.float32)
    if previous_error.ndim != 2 or bias.shape != (previous_error.shape[1],):
        raise ValueError("residual correction dimensions do not match")
    correction = np.broadcast_to(bias[None, :], previous_error.shape).copy()
    if candidate == "bias_only":
        return np.clip(correction, -correction_clip, correction_clip)

    common = _nanmedian(previous_error, axis=0)
    if candidate == "node_ar":
        gain = np.asarray(coefficients["node_gain"], dtype=np.float32)
        valid = np.isfinite(previous_error)
        dynamic = previous_error * gain[None, :]
        correction[valid] += dynamic[valid]
    elif candidate == "common_ar":
        gain = np.asarray(coefficients["common_gain"], dtype=np.float32)
        dynamic = common * gain
        valid_features = np.isfinite(dynamic)
        correction[:, valid_features] += dynamic[valid_features]
    else:
        node_gain = np.asarray(
            coefficients["hybrid_node_gain"],
            dtype=np.float32,
        )
        common_gain = np.asarray(
            coefficients["hybrid_common_gain"],
            dtype=np.float32,
        )
        node_dynamic = previous_error * node_gain[None, :]
        valid_node = np.isfinite(previous_error)
        correction[valid_node] += node_dynamic[valid_node]
        common_dynamic = common * common_gain
        valid_features = np.isfinite(common_dynamic)
        correction[:, valid_features] += common_dynamic[valid_features]
    return np.clip(correction, -correction_clip, correction_clip)


def corrected_model_sse(
    baseline_sse: float,
    target_error: np.ndarray,
    correction: np.ndarray,
) -> tuple[float, int]:
    target_error = np.asarray(target_error, dtype=np.float64)
    correction = np.asarray(correction, dtype=np.float64)
    if target_error.shape != correction.shape:
        raise ValueError("residual SSE inputs must have equal shapes")
    valid = np.isfinite(target_error) & np.isfinite(correction)
    if not valid.any():
        return float(baseline_sse), 0
    before = target_error[valid]
    after = before - correction[valid]
    adjusted = float(baseline_sse + np.sum(after * after - before * before))
    return adjusted, int(valid.sum())


def collect_baseline_errors(
    model,
    features,
    steps: np.ndarray,
    ckpt: dict[str, Any],
    ckpt_args: dict[str, Any],
    cli_args: argparse.Namespace,
    device: torch.device,
    edge_cache,
) -> tuple[dict[int, np.ndarray], dict[int, list[dict[str, Any]]], np.ndarray]:
    horizons = tuple(parse_int_list(cli_args.horizons))
    stock_count = int(features.tradable_count)
    feature_count = len(features.feature_names)
    temporal_weights = ckpt.get("temporal_state_feature_weights")
    if temporal_weights is None:
        eligible = np.ones(feature_count, dtype=bool)
    else:
        if torch.is_tensor(temporal_weights):
            temporal_weights = temporal_weights.detach().cpu().numpy()
        eligible = np.asarray(temporal_weights, dtype=np.float32) > 0.0
    if eligible.shape != (feature_count,):
        raise ValueError("temporal state weights do not match checkpoint features")
    eligible_indices = np.flatnonzero(eligible)
    errors = {
        horizon: np.full(
            (len(steps), stock_count, len(eligible_indices)),
            np.nan,
            dtype=np.float32,
        )
        for horizon in horizons
    }
    rows = {horizon: [] for horizon in horizons}
    edge_window = int(cli_args.edge_window or ckpt_args.get("edge_window", 60))
    edge_top_k = int(cli_args.edge_top_k or ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(
        cli_args.min_abs_corr
        if cli_args.min_abs_corr is not None
        else ckpt_args.get("min_abs_corr", 0.2)
    )
    rollout_args = dict(ckpt_args)
    rollout_args.setdefault(
        "temporal_offset",
        ckpt_args.get("horizon", max(horizons)),
    )
    rollout_args.setdefault("latent_rollout_steps", 1)
    rollout_namespace = as_namespace(rollout_args)

    for ordinal, raw_step in enumerate(steps):
        step = int(raw_step)
        batch = make_real_snapshot(
            features,
            step=step,
            full_observation=True,
            edge_window=edge_window,
            top_k=edge_top_k,
            min_abs_corr=min_abs_corr,
            **graph_edge_kwargs(ckpt_args, cli_args),
            edge_cache=edge_cache,
        ).to(device)
        with torch.no_grad():
            context = model.encode_temporal_context(batch)
        x0 = features.features[step, :stock_count]
        source_available = features.available_mask[step, :stock_count] > 0.5
        for horizon in horizons:
            target_step = step + int(horizon)
            target = features.features[target_step, :stock_count]
            target_available = (
                features.available_mask[target_step, :stock_count] > 0.5
            )
            rollout_steps = rollout_steps_for_offset(
                rollout_namespace,
                int(horizon),
            )
            with torch.no_grad():
                latent = model.rollout_latent(context, steps=rollout_steps)
                prediction = model.predict_temporal_state(
                    batch,
                    latent,
                    rollout_steps=rollout_steps,
                    z_context=context,
                ).detach().cpu().numpy()[:stock_count]
            metrics = future_state_metrics(
                prediction,
                target,
                x0,
                target_available,
                source_available,
            )
            if metrics is None:
                raise ValueError("residual audit baseline produced no cells")
            eligible_valid = (
                target_available[:, eligible_indices]
                & source_available[:, eligible_indices]
                & np.isfinite(prediction[:, eligible_indices])
                & np.isfinite(target[:, eligible_indices])
            )
            error = (
                target[:, eligible_indices] - prediction[:, eligible_indices]
            ).astype(np.float32, copy=False)
            errors[horizon][ordinal][eligible_valid] = error[eligible_valid]
            rows[horizon].append(
                {
                    "session_ordinal": int(ordinal),
                    "date": str(features.dates[step].date()),
                    "target_date": str(features.dates[target_step].date()),
                    "horizon": int(horizon),
                    "rollout_steps": int(rollout_steps),
                    **metrics,
                }
            )
        if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(steps):
            print(
                f"residual predictability collection={ordinal + 1}/{len(steps)}",
                flush=True,
            )
    return errors, rows, eligible_indices


def evaluate_candidates(
    errors: dict[int, np.ndarray],
    baseline_rows: dict[int, list[dict[str, Any]]],
    coefficients: dict[int, dict[str, np.ndarray | list[int]]],
    *,
    split_index: int,
    correction_clip: float,
) -> list[dict[str, Any]]:
    output = []
    for horizon, horizon_errors in errors.items():
        for ordinal in range(split_index, len(horizon_errors)):
            base = baseline_rows[horizon][ordinal]
            baseline_skill = float(base["mse_skill_vs_persistence"])
            output.append(
                {
                    "candidate": "baseline",
                    **base,
                    "eligible_corrected_cells": 0,
                    "correction_rms": 0.0,
                }
            )
            previous_error = horizon_errors[ordinal - int(horizon)]
            target_error = horizon_errors[ordinal]
            for candidate in AUDIT_CANDIDATES:
                correction = residual_correction(
                    previous_error,
                    coefficients[horizon],
                    candidate,
                    correction_clip=correction_clip,
                )
                model_sse, corrected_cells = corrected_model_sse(
                    float(base["model_sse"]),
                    target_error,
                    correction,
                )
                persistence_sse = float(base["persistence_sse"])
                skill = (
                    1.0 - model_sse / persistence_sse
                    if persistence_sse > 1e-12
                    else float("nan")
                )
                finite_correction = correction[np.isfinite(target_error)]
                output.append(
                    {
                        "candidate": candidate,
                        **base,
                        "model_sse": model_sse,
                        "mse_skill_vs_persistence": skill,
                        "skill_delta_vs_baseline": skill - baseline_skill,
                        "eligible_corrected_cells": corrected_cells,
                        "correction_rms": float(
                            np.sqrt(np.mean(finite_correction**2))
                        )
                        if len(finite_correction)
                        else 0.0,
                    }
                )
    return output


def _pooled_skill(rows: Sequence[dict[str, Any]]) -> float:
    model_sse = sum(float(row["model_sse"]) for row in rows)
    persistence_sse = sum(float(row["persistence_sse"]) for row in rows)
    return (
        float(1.0 - model_sse / persistence_sse)
        if persistence_sse > 1e-12
        else float("nan")
    )


def summarize_audit(
    rows: list[dict[str, Any]],
    horizons: Sequence[int],
    *,
    impact_quantile: float,
    pooled_floor: float,
) -> dict[str, Any]:
    baseline = {
        (int(row["session_ordinal"]), int(row["horizon"])): row
        for row in rows
        if row["candidate"] == "baseline"
    }
    by_candidate: dict[str, Any] = {}
    for candidate in AUDIT_CANDIDATES:
        candidate_rows = {
            (int(row["session_ordinal"]), int(row["horizon"])): row
            for row in rows
            if row["candidate"] == candidate
        }
        horizon_output = {}
        for horizon in horizons:
            keys = sorted(key for key in baseline if key[1] == int(horizon))
            base_rows = [baseline[key] for key in keys]
            filtered_rows = [candidate_rows[key] for key in keys]
            bias_rows = [
                row
                for row in rows
                if row["candidate"] == "bias_only"
                and int(row["horizon"]) == int(horizon)
            ]
            base_daily = np.asarray(
                [float(row["mse_skill_vs_persistence"]) for row in base_rows]
            )
            filtered_daily = np.asarray(
                [float(row["mse_skill_vs_persistence"]) for row in filtered_rows]
            )
            bias_daily = np.asarray(
                [float(row["mse_skill_vs_persistence"]) for row in bias_rows]
            )
            energy = np.asarray(
                [
                    float(row["persistence_sse"])
                    / float(row["observed_cells"])
                    for row in base_rows
                ]
            )
            threshold = float(np.quantile(energy, impact_quantile))
            top_mask = energy >= threshold
            horizon_output[str(int(horizon))] = {
                "rows": len(keys),
                "baseline_pooled_skill": _pooled_skill(base_rows),
                "candidate_pooled_skill": _pooled_skill(filtered_rows),
                "pooled_delta_vs_baseline": (
                    _pooled_skill(filtered_rows) - _pooled_skill(base_rows)
                ),
                "pooled_delta_vs_bias_only": (
                    _pooled_skill(filtered_rows) - _pooled_skill(bias_rows)
                ),
                "daily_delta_vs_baseline": newey_west_mean(
                    filtered_daily - base_daily,
                    lag=int(horizon),
                ),
                "top_impact_delta_vs_baseline": newey_west_mean(
                    (filtered_daily - base_daily)[top_mask],
                    lag=int(horizon),
                ),
                "daily_delta_vs_bias_only": newey_west_mean(
                    filtered_daily - bias_daily,
                    lag=int(horizon),
                ),
                "top_impact_delta_vs_bias_only": newey_west_mean(
                    (filtered_daily - bias_daily)[top_mask],
                    lag=int(horizon),
                ),
                "impact_threshold": threshold,
                "top_impact_rows": int(top_mask.sum()),
                "mean_correction_rms": float(
                    np.mean(
                        [float(row["correction_rms"]) for row in filtered_rows]
                    )
                ),
            }
        floor_pass = all(
            horizon_output[str(int(horizon))]["pooled_delta_vs_baseline"]
            >= pooled_floor
            for horizon in horizons
        )
        dynamic_pass = candidate in DYNAMIC_CANDIDATES and all(
            horizon_output[str(horizon)]["daily_delta_vs_baseline"]["mean"] > 0.0
            and horizon_output[str(horizon)][
                "top_impact_delta_vs_baseline"
            ]["mean"]
            > 0.0
            and horizon_output[str(horizon)]["daily_delta_vs_bias_only"]["mean"]
            > 0.0
            and horizon_output[str(horizon)][
                "top_impact_delta_vs_bias_only"
            ]["mean"]
            > 0.0
            for horizon in SELECTION_HORIZONS
        )
        score = float(
            min(
                horizon_output[str(horizon)][
                    "top_impact_delta_vs_bias_only"
                ]["mean"]
                for horizon in SELECTION_HORIZONS
            )
        )
        by_candidate[candidate] = {
            "horizons": horizon_output,
            "pooled_floor_passed": bool(floor_pass),
            "dynamic_increment_passed": bool(dynamic_pass),
            "selection_eligible": bool(floor_pass and dynamic_pass),
            "selection_score": score,
        }
    eligible = [
        (value["selection_score"], name)
        for name, value in by_candidate.items()
        if value["selection_eligible"]
    ]
    selected = max(eligible)[1] if eligible else None
    return {
        "impact_quantile": float(impact_quantile),
        "pooled_delta_floor": float(pooled_floor),
        "selection_horizons": list(SELECTION_HORIZONS),
        "candidates": by_candidate,
        "selected_candidate": selected,
        "selection_passed": selected is not None,
    }


def coefficient_summary(
    values: dict[int, dict[str, np.ndarray | list[int]]],
    feature_names: Sequence[str],
    eligible_indices: np.ndarray,
) -> dict[str, Any]:
    output = {}
    for horizon, coefficients in values.items():
        horizon_output = {}
        for key in (
            "bias",
            "node_gain",
            "common_gain",
            "hybrid_node_gain",
            "hybrid_common_gain",
        ):
            array = np.asarray(coefficients[key], dtype=np.float64)
            order = np.argsort(np.abs(array))[::-1][:10]
            horizon_output[key] = {
                "mean": float(np.mean(array)),
                "median": float(np.median(array)),
                "positive_fraction": float(np.mean(array > 0.0)),
                "negative_fraction": float(np.mean(array < 0.0)),
                "max_abs": float(np.max(np.abs(array))),
                "top_abs": [
                    {
                        "feature": str(feature_names[int(eligible_indices[index])]),
                        "value": float(array[index]),
                    }
                    for index in order
                ],
            }
        output[str(int(horizon))] = horizon_output
    return output


def save_selected_adapter(
    output_dir: Path,
    selected_candidate: str,
    coefficients: dict[int, dict[str, np.ndarray | list[int]]],
    *,
    horizons: Sequence[int],
    eligible_indices: np.ndarray,
    feature_names: Sequence[str],
    model_dir: Path,
    checkpoint: dict[str, Any],
    steps: np.ndarray,
    features,
    hyperparameters: dict[str, float | int],
) -> dict[str, Any]:
    arrays: dict[str, np.ndarray] = {
        "eligible_indices": np.asarray(eligible_indices, dtype=np.int64),
    }
    coefficient_keys = (
        "bias",
        "node_gain",
        "common_gain",
        "hybrid_node_gain",
        "hybrid_common_gain",
    )
    for horizon in horizons:
        for key in coefficient_keys:
            arrays[f"h{int(horizon)}_{key}"] = np.asarray(
                coefficients[int(horizon)][key],
                dtype=np.float32,
            )
    artifact_path = output_dir / "selected_adapter.npz"
    np.savez_compressed(artifact_path, **arrays)
    metadata = {
        "schema_version": 1,
        "selected_candidate": selected_candidate,
        "source_model_dir": str(model_dir),
        "source_checkpoint_sha256": sha256_file(model_dir / "graph_jepa_real.pt"),
        "source_train_data_manifest_sha256": checkpoint.get(
            "train_data_manifest", {}
        ).get("sha256"),
        "source_eval_start": str(features.dates[int(steps[0])].date()),
        "source_eval_end": str(features.dates[int(steps[-1])].date()),
        "source_latest_target_date_by_horizon": {
            str(int(horizon)): str(
                features.dates[int(steps[-1]) + int(horizon)].date()
            )
            for horizon in horizons
        },
        "fit_contract": {
            "configuration_selected_on_chronological_holdout": True,
            "coefficients_refit_on_all_source_fold": True,
            "target_fold_touched": False,
            "action_outputs_fed_back": False,
        },
        "horizons": [int(horizon) for horizon in horizons],
        "feature_names": [str(name) for name in feature_names],
        "eligible_indices": [int(index) for index in eligible_indices],
        "eligible_feature_names": [
            str(feature_names[int(index)]) for index in eligible_indices
        ],
        "hyperparameters": hyperparameters,
        "artifact_file": artifact_path.name,
        "artifact_sha256": sha256_file(artifact_path),
        "implementation_sha256": sha256_file(Path(__file__)),
        "live_orders_allowed": False,
    }
    metadata_path = output_dir / "selected_adapter.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2) + "\n",
        encoding="utf-8",
    )
    return {
        "candidate": selected_candidate,
        "artifact": str(artifact_path),
        "artifact_sha256": metadata["artifact_sha256"],
        "metadata": str(metadata_path),
        "metadata_sha256": sha256_file(metadata_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_data_parser()
    parser.description = (
        "Audit out-of-sample signed predictability in matured JEPA residuals."
    )
    parser.add_argument("--fit-fraction", type=float, default=0.5)
    parser.add_argument("--ridge", type=float, default=0.01)
    parser.add_argument("--min-samples", type=int, default=100)
    parser.add_argument("--gain-clip", type=float, default=1.0)
    parser.add_argument("--bias-clip", type=float, default=0.5)
    parser.add_argument("--correction-clip", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.evaluation_role != "calibration":
        raise ValueError("residual predictability audit is calibration-only")
    if not 0.35 <= args.fit_fraction <= 0.65:
        raise ValueError("fit fraction must remain between 0.35 and 0.65")
    args.max_steps = 0
    horizons = tuple(parse_int_list(args.horizons))
    if horizons != REQUIRED_HORIZONS:
        raise ValueError(
            f"residual audit requires horizons {list(REQUIRED_HORIZONS)}"
        )
    device = torch.device(args.device)
    model, checkpoint = load_model(args.model_dir, device)
    checkpoint_args = dict(checkpoint.get("args", {}))
    validate_future_rollout_contract(
        checkpoint_args,
        list(horizons),
        args.allow_extrapolated_horizons,
    )
    features, checkpoint_args = build_features_from_ckpt(checkpoint, args)
    steps = select_steps(features, checkpoint_args, args)
    if args.limit_steps:
        steps = steps[: args.limit_steps]
    if np.any(np.diff(steps) != 1):
        raise ValueError("residual audit requires contiguous evaluation steps")
    split_index = int(round(len(steps) * args.fit_fraction))
    if split_index <= 2 * max(horizons) or len(steps) - split_index < 20:
        raise ValueError("residual audit split is too small")
    edge_cache = build_evaluation_edge_cache(
        features,
        steps,
        checkpoint_args,
        args,
    )
    errors, baseline_rows, eligible_indices = collect_baseline_errors(
        model,
        features,
        steps,
        checkpoint,
        checkpoint_args,
        args,
        device,
        edge_cache,
    )
    coefficients = {}
    fit_contract = {}
    for horizon in horizons:
        fit_indices = np.arange(
            int(horizon),
            split_index - int(horizon),
            dtype=np.int64,
        )
        coefficients[horizon] = fit_residual_models(
            errors[horizon],
            int(horizon),
            fit_indices,
            ridge=args.ridge,
            min_samples=args.min_samples,
            gain_clip=args.gain_clip,
            bias_clip=args.bias_clip,
        )
        fit_contract[str(int(horizon))] = {
            "fit_context_start_ordinal": int(fit_indices[0]),
            "fit_context_end_ordinal": int(fit_indices[-1]),
            "fit_rows": int(len(fit_indices)),
            "embargo_sessions": int(horizon),
            "latest_fit_target_ordinal": int(fit_indices[-1] + horizon),
            "validation_start_ordinal": int(split_index),
        }
    rows = evaluate_candidates(
        errors,
        baseline_rows,
        coefficients,
        split_index=split_index,
        correction_clip=args.correction_clip,
    )
    summary = summarize_audit(
        rows,
        horizons,
        impact_quantile=args.impact_quantile,
        pooled_floor=-0.002,
    )
    output_dir = args.output_dir / args.model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in ("baseline", *AUDIT_CANDIDATES):
        write_csv(
            output_dir / f"validation_{candidate}.csv",
            [row for row in rows if row["candidate"] == candidate],
        )
    hyperparameters = {
        "ridge": float(args.ridge),
        "min_samples": int(args.min_samples),
        "gain_clip": float(args.gain_clip),
        "bias_clip": float(args.bias_clip),
        "correction_clip": float(args.correction_clip),
    }
    selected_adapter = None
    if summary["selection_passed"]:
        full_coefficients = {}
        for horizon in horizons:
            full_fit_indices = np.arange(
                int(horizon),
                len(steps),
                dtype=np.int64,
            )
            full_coefficients[horizon] = fit_residual_models(
                errors[horizon],
                int(horizon),
                full_fit_indices,
                ridge=args.ridge,
                min_samples=args.min_samples,
                gain_clip=args.gain_clip,
                bias_clip=args.bias_clip,
            )
        selected_adapter = save_selected_adapter(
            output_dir,
            str(summary["selected_candidate"]),
            full_coefficients,
            horizons=horizons,
            eligible_indices=eligible_indices,
            feature_names=features.feature_names,
            model_dir=args.model_dir,
            checkpoint=checkpoint,
            steps=steps,
            features=features,
            hyperparameters=hyperparameters,
        )
    report = {
        "schema_version": 1,
        "evaluation_role": "calibration_chronological_split",
        "causal_contract": {
            "fit_before_validation": True,
            "horizon_embargo": True,
            "input_is_matured_forecast_residual": True,
            "action_outputs_fed_back": False,
            "fold3_touched": False,
            "live_orders_allowed": False,
        },
        "model_dir": str(args.model_dir),
        "checkpoint_sha256": sha256_file(args.model_dir / "graph_jepa_real.pt"),
        "implementation_sha256": sha256_file(Path(__file__)),
        "runtime": {
            "python": sys.version,
            "torch": torch.__version__,
            "numpy": np.__version__,
            "device": str(device),
            "cuda_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(device)
                if device.type == "cuda"
                else None
            ),
        },
        "eval_steps": int(len(steps)),
        "split_index": int(split_index),
        "fit_fraction": float(args.fit_fraction),
        "validation_start": str(features.dates[int(steps[split_index])].date()),
        "validation_end": str(features.dates[int(steps[-1])].date()),
        "eligible_feature_count": int(len(eligible_indices)),
        "eligible_features": [
            str(features.feature_names[int(index)]) for index in eligible_indices
        ],
        "fit_contract": fit_contract,
        "hyperparameters": hyperparameters,
        "coefficient_summary": coefficient_summary(
            coefficients,
            features.feature_names,
            eligible_indices,
        ),
        "selection": summary,
        "selected_adapter": selected_adapter,
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    marker = "RESIDUAL_SIGNAL_SELECTED" if summary["selection_passed"] else "RESIDUAL_SIGNAL_REJECTED"
    (output_dir / marker).touch()
    print(
        json.dumps(
            {
                "selected_candidate": summary["selected_candidate"],
                "selection_passed": summary["selection_passed"],
                "output": str(output_dir / "summary.json"),
                "fold3_touched": False,
                "live_orders_allowed": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
