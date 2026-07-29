from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def evaluate_open_innovation_replication(
    contract: Mapping[str, Any],
    stability_summary: Mapping[str, Any],
    fold_summaries: Mapping[str, Mapping[str, Any]],
    aggregate_summary: Mapping[str, Any],
    observed_source_hashes: Mapping[str, str],
) -> dict[str, Any]:
    """Verify a predeclared seed replication without granting live eligibility."""

    if int(contract.get("schema_version", 0)) != 1:
        raise ValueError("open-innovation replication contract must use schema 1")
    if contract.get("eligible_as_unbiased_promotion_evidence") is not False:
        raise ValueError("this reused-test replication cannot be unbiased evidence")
    if contract.get("promotion_eligible_from_this_replication_alone") is not False:
        raise ValueError("replication alone cannot grant promotion eligibility")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("replication contract must prohibit live orders")

    folds = tuple(str(value) for value in contract.get("folds", ()))
    if len(folds) != 5 or len(set(folds)) != 5:
        raise ValueError("open-innovation replication requires five unique folds")
    if set(fold_summaries) != set(folds):
        raise ValueError("candidate fold summaries do not match the contract")

    benchmark = _mapping(contract.get("benchmark"), "benchmark")
    references = _mapping(contract.get("seed17_reference"), "seed17_reference")
    invariants = _mapping(contract.get("invariants"), "invariants")
    expected_sources = _mapping(contract.get("source_sha256"), "source_sha256")
    if set(observed_source_hashes) != set(expected_sources):
        raise ValueError("observed source files do not match the contract")

    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, actual: Any, expected: Any) -> None:
        checks.append(
            {
                "id": check_id,
                "passed": bool(passed),
                "actual": actual,
                "expected": expected,
            }
        )

    for path, expected in expected_sources.items():
        actual = str(observed_source_hashes[path])
        add(f"source:{path}", actual == str(expected), actual, str(expected))

    tolerance = float(invariants.get("open_sensor_score_absolute_tolerance", 0.0))
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("open-sensor score tolerance must be finite and nonnegative")
    expected_configs = set(str(value) for value in benchmark["model_configs"])
    expected_placebos = [int(value) for value in benchmark["placebo_seeds"]]
    checkpoint_hashes: list[str] = []
    fold_rows: dict[str, Any] = {}
    for fold in folds:
        summary = _mapping(fold_summaries[fold], f"candidate {fold}")
        reference = _mapping(references.get(fold), f"reference {fold}")
        add(f"{fold}:status", summary.get("status") == "complete", summary.get("status"), "complete")
        add(
            f"{fold}:schema",
            int(summary.get("schema_version", 0))
            == int(benchmark["report_schema_version"]),
            int(summary.get("schema_version", 0)),
            int(benchmark["report_schema_version"]),
        )
        add(
            f"{fold}:target_version",
            summary.get("target_version") == benchmark["target_version"],
            summary.get("target_version"),
            benchmark["target_version"],
        )
        add(
            f"{fold}:open_sensor_contract",
            summary.get("open_sensor_contract") == benchmark["open_sensor_contract"],
            summary.get("open_sensor_contract"),
            benchmark["open_sensor_contract"],
        )
        add(
            f"{fold}:jepa_feature_mode",
            summary.get("jepa_feature_mode") == benchmark["jepa_feature_mode"],
            summary.get("jepa_feature_mode"),
            benchmark["jepa_feature_mode"],
        )
        add(
            f"{fold}:placebo_seeds",
            [int(value) for value in summary.get("placebo_seeds", ())]
            == expected_placebos,
            summary.get("placebo_seeds"),
            expected_placebos,
        )
        add(
            f"{fold}:model_configs",
            set(str(value) for value in summary.get("model_configs", {}))
            == expected_configs,
            sorted(summary.get("model_configs", {})),
            sorted(expected_configs),
        )
        add(
            f"{fold}:live_orders",
            summary.get("live_orders_allowed") is False,
            summary.get("live_orders_allowed"),
            False,
        )

        split_dates = _mapping(summary.get("split_dates"), f"candidate {fold}.split_dates")
        expected_split = {
            "fit": int(reference["fit_dates"]),
            "validation": int(reference["validation_dates"]),
            "test": int(reference["test_dates"]),
        }
        actual_split = {name: int(split_dates.get(name, -1)) for name in expected_split}
        add(f"{fold}:split_dates", actual_split == expected_split, actual_split, expected_split)

        variants = _mapping(summary.get("variants"), f"candidate {fold}.variants")
        baseline_variant = _mapping(variants.get("open_sensors"), f"candidate {fold}.open_sensors")
        baseline = _mapping(baseline_variant.get("selected"), f"candidate {fold}.open_sensors.selected")
        expected_config = str(reference["open_sensor_selected_config"])
        validation_score = float(baseline.get("validation_score", float("nan")))
        test_score = float(baseline.get("test_score", float("nan")))
        expected_validation = float(reference["open_sensor_validation_score"])
        expected_test = float(reference["open_sensor_test_score"])
        add(
            f"{fold}:open_sensor_config",
            str(baseline.get("config")) == expected_config,
            baseline.get("config"),
            expected_config,
        )
        add(
            f"{fold}:open_sensor_validation_score",
            np.isfinite(validation_score)
            and abs(validation_score - expected_validation) <= tolerance,
            validation_score,
            expected_validation,
        )
        add(
            f"{fold}:open_sensor_test_score",
            np.isfinite(test_score) and abs(test_score - expected_test) <= tolerance,
            test_score,
            expected_test,
        )
        checkpoint_hash = str(summary.get("checkpoint_sha256", ""))
        checkpoint_hashes.append(checkpoint_hash)
        fold_rows[fold] = {
            "checkpoint_sha256": checkpoint_hash,
            "split_dates": actual_split,
            "open_sensor_selected_config": baseline.get("config"),
            "open_sensor_validation_score": validation_score,
            "open_sensor_test_score": test_score,
        }

    add(
        "distinct_checkpoint_per_fold",
        len(set(checkpoint_hashes)) == len(folds) and all(checkpoint_hashes),
        len(set(checkpoint_hashes)),
        len(folds),
    )

    aggregate_folds = aggregate_summary.get("folds")
    if not isinstance(aggregate_folds, list):
        raise ValueError("aggregate summary folds must be a list")
    aggregate_checkpoint_by_fold = {
        str(row["fold"]): str(row["checkpoint_sha256"])
        for row in aggregate_folds
        if isinstance(row, Mapping) and "fold" in row and "checkpoint_sha256" in row
    }
    add(
        "aggregate_checkpoint_binding",
        aggregate_checkpoint_by_fold
        == {fold: fold_rows[fold]["checkpoint_sha256"] for fold in folds},
        aggregate_checkpoint_by_fold,
        {fold: fold_rows[fold]["checkpoint_sha256"] for fold in folds},
    )
    aggregate = _mapping(aggregate_summary.get("aggregate"), "aggregate summary.aggregate")
    add(
        "aggregate_test_dates",
        int(aggregate.get("test_dates", -1))
        == int(benchmark["expected_test_dates_total"]),
        int(aggregate.get("test_dates", -1)),
        int(benchmark["expected_test_dates_total"]),
    )
    add(
        "aggregate_live_orders",
        aggregate_summary.get("live_orders_allowed") is False,
        aggregate_summary.get("live_orders_allowed"),
        False,
    )

    stability_gate = _mapping(stability_summary.get("gate"), "stability gate")
    aggregate_gate = _mapping(aggregate.get("gate"), "schema4 aggregate gate")
    add(
        "seed_stability_gate",
        stability_gate.get("passed") is True,
        stability_gate.get("passed"),
        True,
    )
    add(
        "schema4_multifold_gate",
        aggregate_gate.get("passed") is True,
        aggregate_gate.get("passed"),
        True,
    )

    passed = all(row["passed"] for row in checks)
    return {
        "schema_version": 1,
        "role": "seed29_open_innovation_replication_gate",
        "folds": fold_rows,
        "checks": checks,
        "gate": {
            "passed": passed,
            "failures": [row["id"] for row in checks if not row["passed"]],
        },
        "decision": (
            "advance_to_read_only_forward_shadow_validation"
            if passed
            else "research_only_replication_failed"
        ),
        "eligible_as_unbiased_promotion_evidence": False,
        "promotion_eligible_from_this_replication_alone": False,
        "live_orders_allowed": False,
    }
