from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.audit_residual_predictability import (
    DYNAMIC_CANDIDATES,
    SELECTION_HORIZONS,
    collect_baseline_errors,
    corrected_model_sse,
    fit_residual_models,
    residual_correction,
    save_selected_adapter,
)
from scripts.benchmark_direct_baselines import newey_west_mean
from scripts.evaluate_causal_innovation_filter import (
    REQUIRED_HORIZONS,
    build_parser as build_data_parser,
    sha256_file,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    select_steps,
    validate_future_rollout_contract,
    write_csv,
)
from scripts.run_real_backtest import parse_int_list


def load_adapter(
    metadata_path: Path,
) -> tuple[dict[str, Any], dict[int, dict[str, np.ndarray]]]:
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported residual adapter schema")
    candidate = str(metadata.get("selected_candidate", ""))
    if candidate not in DYNAMIC_CANDIDATES:
        raise ValueError("residual adapter was not a selected dynamic candidate")
    if metadata.get("live_orders_allowed") is not False:
        raise ValueError("residual adapter safety metadata is invalid")
    artifact_path = metadata_path.parent / str(metadata["artifact_file"])
    if sha256_file(artifact_path) != metadata.get("artifact_sha256"):
        raise ValueError("residual adapter artifact hash mismatch")
    horizons = tuple(int(value) for value in metadata.get("horizons", []))
    if horizons != REQUIRED_HORIZONS:
        raise ValueError("residual adapter horizons do not match contract")
    coefficient_keys = (
        "bias",
        "node_gain",
        "common_gain",
        "hybrid_node_gain",
        "hybrid_common_gain",
    )
    coefficients = {}
    with np.load(artifact_path, allow_pickle=False) as archive:
        archived_indices = archive["eligible_indices"].astype(np.int64)
        expected_indices = np.asarray(
            metadata.get("eligible_indices", []),
            dtype=np.int64,
        )
        if not np.array_equal(archived_indices, expected_indices):
            raise ValueError("residual adapter eligible indices do not match")
        for horizon in horizons:
            coefficients[horizon] = {
                key: archive[f"h{horizon}_{key}"].astype(np.float32)
                for key in coefficient_keys
            }
    return metadata, coefficients


def evaluate_transfer(
    errors: dict[int, np.ndarray],
    baseline_rows: dict[int, list[dict[str, Any]]],
    coefficients: dict[int, dict[str, np.ndarray]],
    selected_candidate: str,
    *,
    correction_clip: float,
) -> list[dict[str, Any]]:
    rows = []
    for horizon, horizon_errors in errors.items():
        for ordinal, base in enumerate(baseline_rows[horizon]):
            rows.append(
                {
                    "candidate": "baseline",
                    **base,
                    "dynamic_available": bool(ordinal >= int(horizon)),
                    "eligible_corrected_cells": 0,
                    "correction_rms": 0.0,
                }
            )
            previous_error = (
                horizon_errors[ordinal - int(horizon)]
                if ordinal >= int(horizon)
                else np.full_like(horizon_errors[ordinal], np.nan)
            )
            target_error = horizon_errors[ordinal]
            for candidate in ("bias_only", selected_candidate):
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
                rows.append(
                    {
                        "candidate": candidate,
                        **base,
                        "model_sse": model_sse,
                        "mse_skill_vs_persistence": skill,
                        "dynamic_available": bool(ordinal >= int(horizon)),
                        "eligible_corrected_cells": corrected_cells,
                        "correction_rms": float(
                            np.sqrt(np.mean(finite_correction**2))
                        )
                        if len(finite_correction)
                        else 0.0,
                    }
                )
    return rows


def _pooled_skill(rows: Sequence[dict[str, Any]]) -> float:
    model_sse = sum(float(row["model_sse"]) for row in rows)
    persistence_sse = sum(float(row["persistence_sse"]) for row in rows)
    return (
        float(1.0 - model_sse / persistence_sse)
        if persistence_sse > 1e-12
        else float("nan")
    )


def summarize_transfer(
    rows: list[dict[str, Any]],
    selected_candidate: str,
    horizons: Sequence[int],
    *,
    impact_quantile: float,
    pooled_floor: float = -0.002,
) -> dict[str, Any]:
    indexed = {
        candidate: {
            (int(row["session_ordinal"]), int(row["horizon"])): row
            for row in rows
            if row["candidate"] == candidate
        }
        for candidate in ("baseline", "bias_only", selected_candidate)
    }
    if any(set(values) != set(indexed["baseline"]) for values in indexed.values()):
        raise ValueError("residual transfer rows are not aligned")
    output = {}
    for horizon in horizons:
        keys = sorted(
            key for key in indexed["baseline"] if key[1] == int(horizon)
        )
        base_rows = [indexed["baseline"][key] for key in keys]
        bias_rows = [indexed["bias_only"][key] for key in keys]
        selected_rows = [indexed[selected_candidate][key] for key in keys]
        base_daily = np.asarray(
            [float(row["mse_skill_vs_persistence"]) for row in base_rows]
        )
        bias_daily = np.asarray(
            [float(row["mse_skill_vs_persistence"]) for row in bias_rows]
        )
        selected_daily = np.asarray(
            [float(row["mse_skill_vs_persistence"]) for row in selected_rows]
        )
        energy = np.asarray(
            [
                float(row["persistence_sse"]) / float(row["observed_cells"])
                for row in base_rows
            ]
        )
        impact_threshold = float(np.quantile(energy, impact_quantile))
        impact_mask = energy >= impact_threshold
        dynamic_mask = np.asarray(
            [bool(row["dynamic_available"]) for row in selected_rows]
        )
        selected_delta = selected_daily - base_daily
        bias_delta = selected_daily - bias_daily
        output[str(int(horizon))] = {
            "rows": len(keys),
            "dynamic_rows": int(dynamic_mask.sum()),
            "baseline_pooled_skill": _pooled_skill(base_rows),
            "bias_only_pooled_skill": _pooled_skill(bias_rows),
            "selected_pooled_skill": _pooled_skill(selected_rows),
            "pooled_delta_vs_baseline": (
                _pooled_skill(selected_rows) - _pooled_skill(base_rows)
            ),
            "pooled_delta_vs_bias_only": (
                _pooled_skill(selected_rows) - _pooled_skill(bias_rows)
            ),
            "post_cold_start_pooled_delta_vs_baseline": (
                _pooled_skill(
                    [row for row, use in zip(selected_rows, dynamic_mask) if use]
                )
                - _pooled_skill(
                    [row for row, use in zip(base_rows, dynamic_mask) if use]
                )
            ),
            "daily_delta_vs_baseline": newey_west_mean(
                selected_delta,
                lag=int(horizon),
            ),
            "top_impact_delta_vs_baseline": newey_west_mean(
                selected_delta[impact_mask],
                lag=int(horizon),
            ),
            "post_cold_start_delta_vs_baseline": newey_west_mean(
                selected_delta[dynamic_mask],
                lag=int(horizon),
            ),
            "daily_delta_vs_bias_only": newey_west_mean(
                bias_delta,
                lag=int(horizon),
            ),
            "top_impact_delta_vs_bias_only": newey_west_mean(
                bias_delta[impact_mask],
                lag=int(horizon),
            ),
            "post_cold_start_delta_vs_bias_only": newey_west_mean(
                bias_delta[dynamic_mask],
                lag=int(horizon),
            ),
            "impact_threshold": impact_threshold,
            "top_impact_rows": int(impact_mask.sum()),
            "mean_correction_rms": float(
                np.mean([float(row["correction_rms"]) for row in selected_rows])
            ),
        }
    floor_pass = all(
        output[str(int(horizon))]["pooled_delta_vs_baseline"] >= pooled_floor
        for horizon in horizons
    )
    targeted_pass = all(
        output[str(horizon)]["daily_delta_vs_baseline"]["mean"] > 0.0
        and output[str(horizon)]["top_impact_delta_vs_baseline"]["mean"] > 0.0
        and output[str(horizon)]["post_cold_start_delta_vs_baseline"]["mean"]
        > 0.0
        and output[str(horizon)]["daily_delta_vs_bias_only"]["mean"] > 0.0
        and output[str(horizon)]["top_impact_delta_vs_bias_only"]["mean"] > 0.0
        and output[str(horizon)]["post_cold_start_delta_vs_bias_only"]["mean"]
        > 0.0
        for horizon in SELECTION_HORIZONS
    )
    return {
        "selected_candidate": selected_candidate,
        "impact_quantile": float(impact_quantile),
        "pooled_delta_floor": float(pooled_floor),
        "selection_horizons": list(SELECTION_HORIZONS),
        "horizons": output,
        "pooled_floor_passed": bool(floor_pass),
        "targeted_gate_passed": bool(targeted_pass),
        "transfer_gate_passed": bool(floor_pass and targeted_pass),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = build_data_parser()
    parser.description = "Evaluate a frozen residual adapter on an untouched fold."
    parser.add_argument("--adapter-metadata", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.evaluation_role != "test":
        raise ValueError("residual adapter transfer requires test role")
    args.max_steps = 0
    horizons = tuple(parse_int_list(args.horizons))
    if horizons != REQUIRED_HORIZONS:
        raise ValueError(
            f"residual transfer requires horizons {list(REQUIRED_HORIZONS)}"
        )
    metadata, coefficients = load_adapter(args.adapter_metadata)
    device = torch.device(args.device)
    model, checkpoint = load_model(args.model_dir, device)
    checkpoint_args = dict(checkpoint.get("args", {}))
    target_checkpoint_sha = sha256_file(args.model_dir / "graph_jepa_real.pt")
    if target_checkpoint_sha == metadata["source_checkpoint_sha256"]:
        raise ValueError("residual transfer target must use a different checkpoint")
    validate_future_rollout_contract(
        checkpoint_args,
        list(horizons),
        args.allow_extrapolated_horizons,
    )
    features, checkpoint_args = build_features_from_ckpt(checkpoint, args)
    if list(features.feature_names) != list(metadata["feature_names"]):
        raise ValueError("residual adapter feature schema does not match target")
    steps = select_steps(features, checkpoint_args, args)
    if args.limit_steps:
        steps = steps[: args.limit_steps]
    if np.any(np.diff(steps) != 1):
        raise ValueError("residual transfer requires contiguous evaluation steps")
    target_start = str(features.dates[int(steps[0])].date())
    latest_source_target = max(
        str(value)
        for value in metadata["source_latest_target_date_by_horizon"].values()
    )
    if latest_source_target >= target_start:
        raise ValueError("residual adapter source targets overlap target fold")
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
    if not np.array_equal(
        eligible_indices,
        np.asarray(metadata["eligible_indices"], dtype=np.int64),
    ):
        raise ValueError("residual adapter eligibility changed across folds")
    selected_candidate = str(metadata["selected_candidate"])
    rows = evaluate_transfer(
        errors,
        baseline_rows,
        coefficients,
        selected_candidate,
        correction_clip=float(metadata["hyperparameters"]["correction_clip"]),
    )
    summary = summarize_transfer(
        rows,
        selected_candidate,
        horizons,
        impact_quantile=args.impact_quantile,
    )
    output_dir = args.output_dir / args.model_dir.name
    output_dir.mkdir(parents=True, exist_ok=True)
    for candidate in ("baseline", "bias_only", selected_candidate):
        write_csv(
            output_dir / f"{candidate}.csv",
            [row for row in rows if row["candidate"] == candidate],
        )
    rollforward_adapter = None
    if summary["transfer_gate_passed"]:
        hyperparameters = dict(metadata["hyperparameters"])
        rollforward_coefficients = {}
        for horizon in horizons:
            fit_indices = np.arange(
                int(horizon),
                len(steps),
                dtype=np.int64,
            )
            rollforward_coefficients[horizon] = fit_residual_models(
                errors[horizon],
                int(horizon),
                fit_indices,
                ridge=float(hyperparameters["ridge"]),
                min_samples=int(hyperparameters["min_samples"]),
                gain_clip=float(hyperparameters["gain_clip"]),
                bias_clip=float(hyperparameters["bias_clip"]),
            )
        rollforward_adapter = save_selected_adapter(
            output_dir,
            selected_candidate,
            rollforward_coefficients,
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
        "evaluation_role": "untouched_fold_transfer_test",
        "causal_contract": {
            "source_targets_before_target_fold": True,
            "horizon_specific_cold_start": True,
            "matured_residual_only": True,
            "target_parameters_fitted": False,
            "action_outputs_fed_back": False,
            "live_orders_allowed": False,
        },
        "adapter_metadata": str(args.adapter_metadata),
        "adapter_metadata_sha256": sha256_file(args.adapter_metadata),
        "adapter_artifact_sha256": metadata["artifact_sha256"],
        "source_checkpoint_sha256": metadata["source_checkpoint_sha256"],
        "target_model_dir": str(args.model_dir),
        "target_checkpoint_sha256": target_checkpoint_sha,
        "implementation_sha256": sha256_file(Path(__file__)),
        "target_eval_start": target_start,
        "target_eval_end": str(features.dates[int(steps[-1])].date()),
        "target_eval_steps": int(len(steps)),
        "latest_source_target": latest_source_target,
        "selection": summary,
        "rollforward_adapter": rollforward_adapter,
        "live_orders_allowed": False,
    }
    report_path = output_dir / "summary.json"
    report_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    marker = (
        "TRANSFER_GATE_PASSED"
        if summary["transfer_gate_passed"]
        else "TRANSFER_GATE_FAILED"
    )
    (output_dir / marker).touch()
    print(
        json.dumps(
            {
                "selected_candidate": selected_candidate,
                "transfer_gate_passed": summary["transfer_gate_passed"],
                "output": str(report_path),
                "live_orders_allowed": False,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
