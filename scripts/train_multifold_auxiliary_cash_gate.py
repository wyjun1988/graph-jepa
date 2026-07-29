from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.linear_model import Ridge

from scripts.train_auxiliary_cash_gate import (
    evaluate_gate,
    fit_transform_contract,
    load_dataset,
    regression_metrics,
    validate_contract,
)


FEATURE_SETS = ("market_head", "auxiliary", "semantic")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a cash gate from multiple chronological OOS JEPA folds."
    )
    parser.add_argument("--development", action="append", required=True)
    parser.add_argument("--confirmation", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-set", action="append", default=[])
    parser.add_argument("--alphas", default="0.1,1,10,100,1000,10000,100000")
    parser.add_argument("--z-clip", type=float, default=8.0)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--stress-cost-bps", type=float, default=50.0)
    parser.add_argument("--confirmation-pristine", action="store_true")
    return parser.parse_args()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def design_matrix(dataset: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    names = np.concatenate(
        [dataset["model_feature_names"], dataset["market_feature_names"]]
    ).astype(str)
    values = np.concatenate(
        [dataset["model_features"], dataset["market_features"]], axis=1
    ).astype(np.float64)
    return values, names


def feature_mask(names: np.ndarray, feature_set: str) -> np.ndarray:
    market_head = np.char.startswith(names.astype(str), "market_head_")
    auxiliary = np.char.startswith(names.astype(str), "aux_")
    if feature_set == "market_head":
        result = market_head
    elif feature_set == "auxiliary":
        result = auxiliary
    elif feature_set == "semantic":
        result = market_head | auxiliary
    else:
        raise ValueError(f"unsupported feature set: {feature_set}")
    if not result.any():
        raise ValueError(f"feature set is empty: {feature_set}")
    return result


def transform_clipped(
    values: np.ndarray,
    contract: dict[str, np.ndarray],
    z_clip: float,
) -> np.ndarray:
    selected = values[:, contract["selected_indices"]]
    imputed = np.where(
        np.isfinite(selected),
        selected,
        contract["median"][None, :],
    )
    standardized = (
        (imputed - contract["mean"][None, :]) / contract["scale"][None, :]
    ).astype(np.float64)
    return np.clip(standardized, -float(z_clip), float(z_clip))


def validate_multifold_contract(
    development: list[dict[str, Any]],
    confirmation: dict[str, Any],
) -> None:
    if len(development) < 2:
        raise ValueError("at least two development folds are required")
    reference = development[0]
    for dataset in development[1:] + [confirmation]:
        validate_contract(reference, dataset)
    previous_end: np.datetime64 | None = None
    for dataset in development + [confirmation]:
        if int(dataset["schema_version"][0]) < 2:
            raise ValueError("cash-gate datasets must use schema version 2 or newer")
        dates = dataset["dates"].astype("datetime64[D]")
        if len(dates) < 2 or not np.all(dates[1:] > dates[:-1]):
            raise ValueError("each cash-gate fold must contain strictly increasing dates")
        if previous_end is not None and dates[0] <= previous_end:
            raise ValueError("cash-gate folds must be chronological and non-overlapping")
        previous_end = dates[-1]


def dataset_quality(dataset: dict[str, Any]) -> dict[str, int]:
    return {
        "rows": int(len(dataset["dates"])),
        "finite_targets": int(
            np.isfinite(dataset["candidate_gross_return"]).sum()
        ),
        "candidate_missing_targets": int(
            dataset["candidate_missing_target_count"].sum()
        ),
        "equal_weight_missing_targets": int(
            dataset["equal_weight_missing_target_count"].sum()
        ),
    }


def main() -> None:
    args = parse_args()
    if args.z_clip <= 0.0:
        raise ValueError("z-clip must be positive")
    development_paths = [Path(value) for value in args.development]
    confirmation_path = Path(args.confirmation)
    development = [load_dataset(path) for path in development_paths]
    confirmation = load_dataset(confirmation_path)
    validate_multifold_contract(development, confirmation)

    feature_sets = args.feature_set or list(FEATURE_SETS)
    invalid = sorted(set(feature_sets) - set(FEATURE_SETS))
    if invalid:
        raise ValueError(f"unsupported feature sets: {invalid}")
    alphas = sorted({float(value) for value in args.alphas.split(",") if value})
    if not alphas:
        raise ValueError("at least one alpha is required")

    matrices = []
    feature_names = None
    for dataset in development + [confirmation]:
        values, names = design_matrix(dataset)
        if feature_names is None:
            feature_names = names
        elif not np.array_equal(feature_names, names):
            raise ValueError("cash-gate full feature names disagree across folds")
        matrices.append(values)
    assert feature_names is not None

    training_datasets = development[:-1]
    validation_dataset = development[-1]
    training_values = np.concatenate(matrices[: len(training_datasets)], axis=0)
    validation_values = matrices[len(training_datasets)]
    confirmation_values = matrices[-1]
    training_target = np.concatenate(
        [dataset["candidate_gross_return"] for dataset in training_datasets]
    ).astype(np.float64)
    validation_target = validation_dataset["candidate_gross_return"].astype(
        np.float64
    )
    confirmation_target = confirmation["candidate_gross_return"].astype(np.float64)
    training_indices = np.flatnonzero(np.isfinite(training_target))
    validation_indices = np.flatnonzero(np.isfinite(validation_target))
    confirmation_indices = np.flatnonzero(np.isfinite(confirmation_target))

    candidates: list[dict[str, Any]] = []
    for feature_set in feature_sets:
        mask = feature_mask(feature_names, feature_set)
        fit_values = training_values[:, mask]
        held_out_values = validation_values[:, mask]
        contract = fit_transform_contract(fit_values, training_indices)
        transformed_fit = transform_clipped(fit_values, contract, args.z_clip)
        transformed_validation = transform_clipped(
            held_out_values, contract, args.z_clip
        )
        baseline_mean = float(training_target[training_indices].mean())
        for alpha in alphas:
            model = Ridge(alpha=alpha, solver="lsqr")
            model.fit(
                transformed_fit[training_indices],
                training_target[training_indices],
            )
            prediction = model.predict(transformed_validation)
            metrics = regression_metrics(
                prediction,
                validation_target,
                baseline_mean,
            )
            candidates.append(
                {
                    "feature_set": feature_set,
                    "alpha": alpha,
                    "input_feature_count": int(mask.sum()),
                    "used_feature_count": int(len(contract["selected_indices"])),
                    "metrics": metrics,
                    "prediction": prediction,
                    "model": model,
                    "contract": contract,
                }
            )
    selected = min(candidates, key=lambda row: float(row["metrics"]["mse"]))

    selected_mask = feature_mask(feature_names, str(selected["feature_set"]))
    strict_validation_prediction = np.asarray(
        selected["prediction"], dtype=np.float64
    )
    all_development_values = np.concatenate(
        matrices[: len(development)], axis=0
    )[:, selected_mask]
    all_development_target = np.concatenate(
        [dataset["candidate_gross_return"] for dataset in development]
    ).astype(np.float64)
    all_development_indices = np.flatnonzero(np.isfinite(all_development_target))
    final_contract = fit_transform_contract(
        all_development_values,
        all_development_indices,
    )
    transformed_development = transform_clipped(
        all_development_values,
        final_contract,
        args.z_clip,
    )
    transformed_confirmation = transform_clipped(
        confirmation_values[:, selected_mask],
        final_contract,
        args.z_clip,
    )
    final_model = Ridge(alpha=float(selected["alpha"]), solver="lsqr")
    final_model.fit(
        transformed_development[all_development_indices],
        all_development_target[all_development_indices],
    )
    confirmation_prediction = final_model.predict(transformed_confirmation)

    horizon = int(development[0]["horizon"][0])
    top_k = int(development[0]["top_k"][0])
    costs = sorted({float(args.cost_bps), float(args.stress_cost_bps)})
    validation_evaluations = {
        f"{cost:g}bps": evaluate_gate(
            validation_dataset,
            strict_validation_prediction,
            indices=validation_indices,
            horizon=horizon,
            cost_bps=cost,
        )
        for cost in costs
    }
    confirmation_evaluations = {
        f"{cost:g}bps": evaluate_gate(
            confirmation,
            confirmation_prediction,
            indices=confirmation_indices,
            horizon=horizon,
            cost_bps=cost,
        )
        for cost in costs
    }
    validation_regression = regression_metrics(
        strict_validation_prediction,
        validation_target,
        float(training_target[training_indices].mean()),
    )
    confirmation_regression = regression_metrics(
        confirmation_prediction,
        confirmation_target,
        float(all_development_target[all_development_indices].mean()),
    )

    primary_key = f"{float(args.cost_bps):g}bps"
    stress_key = f"{float(args.stress_cost_bps):g}bps"
    validation_primary = validation_evaluations[primary_key]
    confirmation_primary = confirmation_evaluations[primary_key]
    confirmation_stress = confirmation_evaluations[stress_key]
    checks = {
        "validation_positive_mse_skill": float(
            validation_regression["mse_skill_vs_development_mean"]
        )
        > 0.0,
        "validation_beats_cash": float(
            validation_primary["metrics"]["total_return"]
        )
        > float(validation_primary["metrics"]["risk_free_total_return"]),
        "confirmation_positive_correlation": float(
            confirmation_regression["correlation"]
        )
        > 0.0,
        "confirmation_primary_at_least_12_active_periods": int(
            confirmation_primary["active_periods"]
        )
        >= 12,
        "confirmation_primary_nondegenerate_exposure": 0.10
        <= float(confirmation_primary["active_fraction"])
        <= 0.80,
        "confirmation_primary_beats_cash": float(
            confirmation_primary["metrics"]["total_return"]
        )
        > float(confirmation_primary["metrics"]["risk_free_total_return"]),
        "confirmation_primary_positive_excess_sharpe": float(
            confirmation_primary["metrics"]["excess_sharpe"]
        )
        > 0.0,
        "confirmation_primary_positive_cash_premium": float(
            confirmation_primary["premiums"]["cash"]["mean"]
        )
        > 0.0,
        "confirmation_primary_drawdown_within_15pct": float(
            confirmation_primary["metrics"]["max_drawdown"]
        )
        >= -0.15,
        "confirmation_stress_beats_cash": float(
            confirmation_stress["metrics"]["total_return"]
        )
        > float(confirmation_stress["metrics"]["risk_free_total_return"]),
        "confirmation_stress_positive_excess_sharpe": float(
            confirmation_stress["metrics"]["excess_sharpe"]
        )
        > 0.0,
    }
    passed = all(checks.values())

    group_indices = np.flatnonzero(selected_mask)
    global_selected_indices = group_indices[final_contract["selected_indices"]]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / "cash_gate_head.npz"
    np.savez_compressed(
        artifact_path,
        schema_version=np.asarray([2], dtype=np.int64),
        horizon=np.asarray([horizon], dtype=np.int64),
        top_k=np.asarray([top_k], dtype=np.int64),
        feature_set=np.asarray([str(selected["feature_set"])]),
        alpha=np.asarray([float(selected["alpha"])], dtype=np.float64),
        z_clip=np.asarray([float(args.z_clip)], dtype=np.float64),
        feature_names=feature_names[global_selected_indices],
        selected_indices=global_selected_indices.astype(np.int64),
        median=final_contract["median"],
        mean=final_contract["mean"],
        scale=final_contract["scale"],
        coefficient=np.asarray(final_model.coef_, dtype=np.float64),
        intercept=np.asarray([float(final_model.intercept_)], dtype=np.float64),
        development_checkpoint_sha256=np.concatenate(
            [dataset["checkpoint_sha256"] for dataset in development]
        ),
        confirmation_checkpoint_sha256=confirmation["checkpoint_sha256"],
        live_orders_allowed=np.asarray([False]),
    )

    approval_scope = "none"
    if passed:
        approval_scope = (
            "read_only_shadow"
            if args.confirmation_pristine
            else "research_candidate_holdout_reused"
        )
    output = {
        "status": "pass" if passed else "blocked",
        "approval_scope": approval_scope,
        "live_orders_allowed": False,
        "confirmation_pristine": bool(args.confirmation_pristine),
        "horizon": horizon,
        "top_k": top_k,
        "selected_feature_set": str(selected["feature_set"]),
        "selected_alpha": float(selected["alpha"]),
        "z_clip": float(args.z_clip),
        "feature_count": int(len(global_selected_indices)),
        "checks": checks,
        "failures": [name for name, value in checks.items() if not value],
        "datasets": {
            "development": [
                {
                    "path": str(path),
                    "sha256": sha256_file(path),
                    "start": str(dataset["dates"][0]),
                    "end": str(dataset["dates"][-1]),
                    "quality": dataset_quality(dataset),
                }
                for path, dataset in zip(development_paths, development)
            ],
            "confirmation": {
                "path": str(confirmation_path),
                "sha256": sha256_file(confirmation_path),
                "start": str(confirmation["dates"][0]),
                "end": str(confirmation["dates"][-1]),
                "quality": dataset_quality(confirmation),
            },
        },
        "selection_candidates": [
            {
                "feature_set": row["feature_set"],
                "alpha": row["alpha"],
                "input_feature_count": row["input_feature_count"],
                "used_feature_count": row["used_feature_count"],
                "metrics": row["metrics"],
            }
            for row in candidates
        ],
        "strict_validation": {
            "regression": validation_regression,
            "evaluations": validation_evaluations,
        },
        "confirmation": {
            "regression": confirmation_regression,
            "evaluations": confirmation_evaluations,
        },
        "artifact": str(artifact_path),
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
