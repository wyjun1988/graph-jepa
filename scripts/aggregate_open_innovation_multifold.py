from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np


BASELINE = "open_sensors"
CANDIDATE = "open_sensors_plus_jepa"
REQUIRED_DAILY_PREDICTIONS = {
    "candidate_selection_validation_daily.csv",
    "candidate_refit_test_daily.csv",
    "modular_selection_validation_daily.csv",
    "modular_refit_test_daily.csv",
}


def parse_fold(value: str) -> tuple[str, Path]:
    name, separator, raw_path = str(value).partition("=")
    if not separator or not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("folds must use NAME=REPORT_DIR")
    return name.strip(), Path(raw_path.strip())


def bootstrap_mean_interval(
    values: Sequence[float], *, seed: int = 1701, samples: int = 100_000
) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("bootstrap values must contain at least two finite observations")
    generator = np.random.default_rng(int(seed))
    indices = generator.integers(0, len(values), size=(int(samples), len(values)))
    means = values[indices].mean(axis=1)
    return {
        "mean": float(values.mean()),
        "lower_95": float(np.quantile(means, 0.025)),
        "upper_95": float(np.quantile(means, 0.975)),
    }


def _test_metrics(summary: Mapping[str, Any], variant: str) -> Mapping[str, Any]:
    return summary["variants"][variant]["selected"]["metrics"]["test"]


def _metric(metrics: Mapping[str, Any], path: Sequence[str]) -> float:
    value: Any = metrics
    for key in path:
        value = value[key]
    return float(value)


def _compact_metrics(metrics: Mapping[str, Any]) -> dict[str, float]:
    return {
        "event_auc": _metric(metrics, ("energy", "roc_auc")),
        "event_ap_lift": _metric(metrics, ("energy", "average_precision_lift")),
        "energy_correlation": _metric(
            metrics, ("energy", "systemic_energy_correlation")
        ),
        "tail_mass_recall": _metric(
            metrics, ("energy", "tail_mass_recall_at_fit_event_rate")
        ),
        "impact_direction": _metric(
            metrics,
            ("energy", "event_impact_weighted_market_direction_accuracy"),
        ),
        "broad_selloff_recall": _metric(
            metrics,
            ("subtypes", "broad_selloff", "recall_at_selection_rate"),
        ),
        "minimum_subtype_auc": min(
            float(value["roc_auc"]) for value in metrics["subtypes"].values()
        ),
        "open_to_close_return_correlation": _metric(
            metrics, ("open_to_close_market_return", "correlation")
        ),
    }


def _schema4_modular_summary(
    name: str,
    summary: Mapping[str, Any],
    *,
    baseline_validation_score: float,
    baseline_test_score: float,
) -> dict[str, Any]:
    prediction_contract = summary.get("daily_prediction_contract")
    if not isinstance(prediction_contract, Mapping) or set(prediction_contract) != (
        REQUIRED_DAILY_PREDICTIONS
    ):
        raise ValueError(f"{name} lacks the split-safe schema-4 daily prediction contract")
    modular = summary.get("placebo_guarded_modular")
    if not isinstance(modular, Mapping):
        raise ValueError(f"{name} lacks the schema-4 placebo-guarded modular candidate")
    if modular.get("live_orders_allowed") is not False:
        raise ValueError(f"{name} modular candidate does not prohibit live orders")
    if modular.get("test_used_for_selection") is not False:
        raise ValueError(f"{name} modular candidate used test data for selection")

    minimum_margin = float(modular["minimum_validation_margin"])
    if not np.isfinite(minimum_margin) or minimum_margin < 0.0:
        raise ValueError(f"{name} has an invalid modular validation margin")
    head_selection = modular.get("head_selection")
    if not isinstance(head_selection, Mapping) or not head_selection:
        raise ValueError(f"{name} lacks modular head-selection evidence")
    enabled = [str(value) for value in modular.get("enabled_jepa_heads", [])]
    if len(enabled) != len(set(enabled)):
        raise ValueError(f"{name} repeats enabled modular JEPA heads")
    selected_from_evidence = set()
    for head, evidence in head_selection.items():
        source = str(evidence.get("source"))
        if source not in (BASELINE, CANDIDATE):
            raise ValueError(f"{name} has an invalid source for modular head {head}")
        margin = float(evidence["candidate_margin_over_reference"])
        if not np.isfinite(margin):
            raise ValueError(f"{name} has a non-finite modular margin for {head}")
        if source == CANDIDATE:
            if margin + 1e-12 < minimum_margin:
                raise ValueError(f"{name} enabled modular head {head} below its margin")
            selected_from_evidence.add(str(head))
    if set(enabled) != selected_from_evidence:
        raise ValueError(f"{name} enabled modular heads differ from selection evidence")

    metrics = modular.get("metrics")
    if not isinstance(metrics, Mapping) or set(metrics) != {"validation", "test"}:
        raise ValueError(f"{name} lacks split-safe modular metrics")
    validation_score = float(modular["validation_score"])
    test_score = float(modular["test_score"])
    if not np.isfinite((validation_score, test_score)).all():
        raise ValueError(f"{name} has non-finite modular scores")
    return {
        "enabled_jepa_heads": enabled,
        "minimum_validation_margin": minimum_margin,
        "validation_score": validation_score,
        "test_score": test_score,
        "minus_baseline_validation": validation_score - baseline_validation_score,
        "minus_baseline_test": test_score - baseline_test_score,
        "beats_baseline_validation": validation_score >= baseline_validation_score,
        "beats_baseline_test": test_score > baseline_test_score,
        "absolute_gate": modular["absolute_gate"],
        "reported_gate": modular["gate"],
        "metrics": _compact_metrics(metrics["test"]),
        "selection_contract_verified": True,
    }


def summarize_fold(name: str, summary: Mapping[str, Any]) -> dict[str, Any]:
    if summary.get("status") != "complete" or int(summary.get("schema_version", 0)) < 2:
        raise ValueError(f"{name} is not a completed refit report")
    if summary.get("live_orders_allowed") is not False:
        raise ValueError(f"{name} does not explicitly prohibit live orders")
    variants = summary["variants"]
    if BASELINE not in variants or CANDIDATE not in variants:
        raise ValueError(f"{name} lacks required open-nowcast variants")
    placebo_names = sorted(
        variant
        for variant in variants
        if variant.startswith("open_sensors_plus_shuffled_jepa_seed")
    )
    if len(placebo_names) < 5:
        raise ValueError(f"{name} requires at least five shuffled JEPA placebos")
    baseline = variants[BASELINE]["selected"]
    candidate = variants[CANDIDATE]["selected"]
    placebo_test = np.asarray(
        [float(variants[value]["selected"]["test_score"]) for value in placebo_names],
        dtype=np.float64,
    )
    placebo_validation = np.asarray(
        [
            float(variants[value]["selected"]["validation_score"])
            for value in placebo_names
        ],
        dtype=np.float64,
    )
    candidate_metrics = _test_metrics(summary, CANDIDATE)
    baseline_metrics = _test_metrics(summary, BASELINE)
    cache_contract = summary["forecast_cache_contract"]
    cache_horizons = [int(value) for value in cache_contract.get("horizons", [])]
    if 1 not in cache_horizons:
        raise ValueError(f"{name} forecast cache does not contain h1 state")
    if cache_contract.get("checkpoint_sha256") != summary["checkpoint_sha256"]:
        raise ValueError(f"{name} forecast cache checkpoint differs from its report")
    jepa_feature_mode = str(summary.get("jepa_feature_mode", "raw"))
    projection_contract = summary.get("jepa_feature_contract", {})
    if jepa_feature_mode == "sensor_residual_pca":
        if int(summary["schema_version"]) < 5 or not isinstance(
            projection_contract, Mapping
        ):
            raise ValueError(f"{name} lacks the schema-5 residual PCA contract")
        if projection_contract.get("fit_only") is not True:
            raise ValueError(f"{name} residual PCA is not fit-only")
        if projection_contract.get("target_used_for_projection") is not False:
            raise ValueError(f"{name} residual PCA used target information")
        if int(projection_contract["retained_jepa_rank"]) != int(
            summary["jepa_innovation_features"]
        ):
            raise ValueError(f"{name} residual PCA rank differs from its report")
        projection_semantics = (
            int(projection_contract["sensor_rank"]),
            int(projection_contract["requested_jepa_rank"]),
            float(projection_contract["ridge_alpha"]),
        )
    elif jepa_feature_mode == "raw":
        projection_semantics = None
    else:
        raise ValueError(f"{name} has an unknown JEPA feature mode")
    row = {
        "fold": name,
        "report_schema_version": int(summary["schema_version"]),
        "target_version": str(summary["target_version"]),
        "open_sensor_contract": str(summary.get("open_sensor_contract", "legacy")),
        "raw_open_gap_statistics": [
            str(value) for value in summary.get("raw_open_gap_statistics", [])
        ],
        "open_sensor_features": int(summary["open_sensor_features"]),
        "jepa_innovation_features": int(summary["jepa_innovation_features"]),
        "jepa_feature_mode": jepa_feature_mode,
        "jepa_projection_semantics": projection_semantics,
        "forecast_cache_kind": str(
            cache_contract.get("cache_kind", "legacy_multihorizon_state_cache")
        ),
        "forecast_semantics": "state_h1_temporal_forecast",
        "checkpoint_sha256": summary["checkpoint_sha256"],
        "test_dates": int(summary["split_dates"]["test"]),
        "candidate_config": candidate["config"],
        "baseline_config": baseline["config"],
        "candidate_validation_score": float(candidate["validation_score"]),
        "baseline_validation_score": float(baseline["validation_score"]),
        "candidate_test_score": float(candidate["test_score"]),
        "baseline_test_score": float(baseline["test_score"]),
        "candidate_minus_baseline_test": float(candidate["test_score"])
        - float(baseline["test_score"]),
        "placebo_validation_scores": placebo_validation.tolist(),
        "placebo_test_scores": placebo_test.tolist(),
        "candidate_minus_best_placebo_test": float(candidate["test_score"])
        - float(placebo_test.max()),
        "candidate_beats_baseline_validation": float(candidate["validation_score"])
        > float(baseline["validation_score"]),
        "candidate_beats_every_placebo_validation": float(
            candidate["validation_score"]
        )
        > float(placebo_validation.max()),
        "candidate_beats_baseline_test": float(candidate["test_score"])
        > float(baseline["test_score"]),
        "candidate_beats_every_placebo_test": float(candidate["test_score"])
        > float(placebo_test.max()),
        "absolute_gate": summary["gate"]["absolute"],
        "candidate_metrics": _compact_metrics(candidate_metrics),
        "baseline_metrics": _compact_metrics(baseline_metrics),
    }
    schema_version = int(summary["schema_version"])
    if schema_version >= 4:
        row["modular"] = _schema4_modular_summary(
            name,
            summary,
            baseline_validation_score=float(baseline["validation_score"]),
            baseline_test_score=float(baseline["test_score"]),
        )
    else:
        row["modular"] = None
    return row


def aggregate(folds: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(folds) != 5:
        raise ValueError("the rolling open-nowcast gate requires exactly five folds")
    experiment_contracts = {
        (
            int(item["report_schema_version"]),
            str(item["target_version"]),
            str(item["open_sensor_contract"]),
            tuple(str(value) for value in item["raw_open_gap_statistics"]),
            int(item["open_sensor_features"]),
            int(item["jepa_innovation_features"]),
            str(item["jepa_feature_mode"]),
            (
                tuple(item["jepa_projection_semantics"])
                if item["jepa_projection_semantics"] is not None
                else None
            ),
            str(item["forecast_semantics"]),
        )
        for item in folds
    }
    if len(experiment_contracts) != 1:
        raise ValueError(
            "every fold must use the same report, target, sensor, and cache contract"
        )
    checkpoints = [str(item["checkpoint_sha256"]) for item in folds]
    if len(set(checkpoints)) != len(checkpoints):
        raise ValueError("every fold must use a distinct checkpoint")
    candidate_delta = np.asarray(
        [float(item["candidate_minus_baseline_test"]) for item in folds],
        dtype=np.float64,
    )
    placebo_delta = np.asarray(
        [float(item["candidate_minus_best_placebo_test"]) for item in folds],
        dtype=np.float64,
    )
    return_delta = np.asarray(
        [
            float(item["candidate_metrics"]["open_to_close_return_correlation"])
            - float(item["baseline_metrics"]["open_to_close_return_correlation"])
            for item in folds
        ],
        dtype=np.float64,
    )
    absolute_passes = sum(bool(item["absolute_gate"]["passed"]) for item in folds)
    baseline_wins = sum(bool(item["candidate_beats_baseline_test"]) for item in folds)
    placebo_wins = sum(bool(item["candidate_beats_every_placebo_test"]) for item in folds)
    validation_baseline_wins = sum(
        bool(item["candidate_beats_baseline_validation"]) for item in folds
    )
    validation_placebo_wins = sum(
        bool(item["candidate_beats_every_placebo_validation"]) for item in folds
    )
    candidate_direct_interval = bootstrap_mean_interval(candidate_delta)
    candidate_placebo_interval = bootstrap_mean_interval(placebo_delta, seed=1702)
    candidate_return_interval = bootstrap_mean_interval(return_delta, seed=1703)
    candidate_checks = {
        "absolute_impact_gate_passes_every_fold": absolute_passes == len(folds),
        "candidate_beats_direct_test_in_at_least_four_folds": baseline_wins
        >= 4,
        "candidate_beats_every_placebo_test_in_every_fold": placebo_wins == len(folds),
        "candidate_beats_direct_validation_in_at_least_four_folds": validation_baseline_wins
        >= 4,
        "candidate_beats_every_placebo_validation_in_at_least_four_folds": validation_placebo_wins
        >= 4,
        "mean_candidate_minus_direct_test_is_positive": float(candidate_delta.mean()) > 0.0,
        "mean_candidate_minus_best_placebo_test_is_positive": float(
            placebo_delta.mean()
        )
        > 0.0,
        "lower_95_candidate_minus_direct_test_is_positive": float(
            candidate_direct_interval["lower_95"]
        )
        > 0.0,
        "lower_95_candidate_minus_best_placebo_test_is_positive": float(
            candidate_placebo_interval["lower_95"]
        )
        > 0.0,
    }
    candidate_result = {
        "absolute_gate_passes": int(absolute_passes),
        "direct_test_wins": int(baseline_wins),
        "placebo_test_wins": int(placebo_wins),
        "direct_validation_wins": int(validation_baseline_wins),
        "placebo_validation_wins": int(validation_placebo_wins),
        "minus_direct_test": candidate_direct_interval,
        "minus_best_placebo_test": candidate_placebo_interval,
        "minus_direct_return_correlation": candidate_return_interval,
        "gate": {
            "passed": all(candidate_checks.values()),
            "checks": candidate_checks,
            "failures": [
                name for name, passed in candidate_checks.items() if not passed
            ],
        },
    }

    modular_presence = [item.get("modular") is not None for item in folds]
    if any(modular_presence) and not all(modular_presence):
        raise ValueError("every fold must either include or omit schema-4 modular evidence")
    modular_result = None
    if all(modular_presence):
        modular_rows = [item["modular"] for item in folds]
        modular_delta = np.asarray(
            [float(item["minus_baseline_test"]) for item in modular_rows],
            dtype=np.float64,
        )
        modular_validation_delta = np.asarray(
            [float(item["minus_baseline_validation"]) for item in modular_rows],
            dtype=np.float64,
        )
        modular_return_delta = np.asarray(
            [
                float(item["metrics"]["open_to_close_return_correlation"])
                - float(fold["baseline_metrics"]["open_to_close_return_correlation"])
                for fold, item in zip(folds, modular_rows)
            ],
            dtype=np.float64,
        )
        modular_direct_interval = bootstrap_mean_interval(modular_delta, seed=1711)
        modular_validation_interval = bootstrap_mean_interval(
            modular_validation_delta, seed=1712
        )
        modular_return_interval = bootstrap_mean_interval(
            modular_return_delta, seed=1713
        )
        head_frequency = Counter(
            head for item in modular_rows for head in item["enabled_jepa_heads"]
        )
        modular_absolute_passes = sum(
            bool(item["absolute_gate"]["passed"]) for item in modular_rows
        )
        modular_test_wins = sum(
            bool(item["beats_baseline_test"]) for item in modular_rows
        )
        modular_validation_wins = sum(
            bool(item["beats_baseline_validation"]) for item in modular_rows
        )
        modular_head_folds = sum(
            bool(item["enabled_jepa_heads"]) for item in modular_rows
        )
        modular_reported_passes = sum(
            bool(item["reported_gate"]["passed"]) for item in modular_rows
        )
        stable_head_frequency = max(head_frequency.values(), default=0)
        modular_checks = {
            "selection_contract_verified_every_fold": all(
                bool(item["selection_contract_verified"]) for item in modular_rows
            ),
            "absolute_impact_gate_passes_every_fold": modular_absolute_passes
            == len(folds),
            "placebo_guarded_jepa_head_selected_in_at_least_four_folds": modular_head_folds
            >= 4,
            "same_jepa_head_selected_in_at_least_three_folds": stable_head_frequency
            >= 3,
            "modular_beats_direct_test_in_at_least_four_folds": modular_test_wins
            >= 4,
            "modular_not_below_direct_validation_in_at_least_four_folds": modular_validation_wins
            >= 4,
            "mean_modular_minus_direct_test_is_positive": float(
                modular_direct_interval["mean"]
            )
            > 0.0,
            "lower_95_modular_minus_direct_test_is_positive": float(
                modular_direct_interval["lower_95"]
            )
            > 0.0,
        }
        modular_result = {
            "absolute_gate_passes": int(modular_absolute_passes),
            "reported_fold_gate_passes": int(modular_reported_passes),
            "folds_with_enabled_jepa_heads": int(modular_head_folds),
            "direct_test_wins": int(modular_test_wins),
            "direct_validation_non_losses": int(modular_validation_wins),
            "head_selection_frequency": dict(sorted(head_frequency.items())),
            "minus_direct_test": modular_direct_interval,
            "minus_direct_validation": modular_validation_interval,
            "minus_direct_return_correlation": modular_return_interval,
            "gate": {
                "passed": all(modular_checks.values()),
                "checks": modular_checks,
                "failures": [
                    name for name, passed in modular_checks.items() if not passed
                ],
            },
        }

    qualified_modes = []
    if candidate_result["gate"]["passed"]:
        qualified_modes.append("full_candidate")
    if modular_result is not None and modular_result["gate"]["passed"]:
        qualified_modes.append("placebo_guarded_modular")
    if modular_result is None:
        deployment_gate = candidate_result["gate"]
    else:
        deployment_checks = {
            "at_least_one_prediction_mode_passes_multifold_gate": bool(
                qualified_modes
            )
        }
        deployment_gate = {
            "passed": all(deployment_checks.values()),
            "checks": deployment_checks,
            "failures": [
                name for name, passed in deployment_checks.items() if not passed
            ],
        }
    return {
        "folds": len(folds),
        "test_dates": int(sum(int(item["test_dates"]) for item in folds)),
        "candidate": candidate_result,
        "modular": modular_result,
        "qualified_modes": qualified_modes,
        # Preserve the schema-1 candidate fields for existing report consumers.
        "absolute_gate_passes": int(absolute_passes),
        "candidate_direct_test_wins": int(baseline_wins),
        "candidate_placebo_test_wins": int(placebo_wins),
        "candidate_direct_validation_wins": int(validation_baseline_wins),
        "candidate_placebo_validation_wins": int(validation_placebo_wins),
        "candidate_minus_direct_test": candidate_direct_interval,
        "candidate_minus_best_placebo_test": candidate_placebo_interval,
        "candidate_minus_direct_return_correlation": candidate_return_interval,
        "gate": deployment_gate,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate causal KRX-open JEPA innovation evidence across folds."
    )
    parser.add_argument("--fold", action="append", type=parse_fold, required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    names = [name for name, _ in args.fold]
    if len(set(names)) != len(names):
        raise ValueError("fold names must be unique")
    fold_rows = []
    for name, report_dir in args.fold:
        summary = json.loads((report_dir / "summary.json").read_text(encoding="utf-8"))
        fold_rows.append(summarize_fold(name, summary))
    result = aggregate(fold_rows)
    payload = {
        "schema_version": 2,
        "status": "complete",
        "role": "split_safe_open_innovation_multifold_gate",
        "folds": fold_rows,
        "aggregate": result,
        "decision": (
            "shadow_candidate" if result["gate"]["passed"] else "research_only"
        ),
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "AGGREGATION_COMPLETE").touch()
    print(json.dumps(payload["aggregate"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
