from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from stock_v2.downstream_probes import newey_west_mean


KEY_COLUMNS = ("date", "horizon")


def _require_mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _normalize_frame(
    frame: pd.DataFrame,
    *,
    label: str,
    horizons: Sequence[int],
    metrics: Sequence[str],
    expected_scope: str,
    expected_feature_count: int,
) -> pd.DataFrame:
    required = {
        *KEY_COLUMNS,
        "state_target_scope",
        "state_target_feature_count",
        *metrics,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing columns: {missing}")
    result = frame.loc[:, sorted(required)].copy()
    result["date"] = pd.to_datetime(result["date"], errors="raise").dt.normalize()
    result["horizon"] = pd.to_numeric(result["horizon"], errors="raise").astype(int)
    if result.duplicated(list(KEY_COLUMNS)).any():
        raise ValueError(f"{label} has duplicate date/horizon rows")
    observed_horizons = sorted(result["horizon"].unique().tolist())
    if observed_horizons != sorted(int(value) for value in horizons):
        raise ValueError(
            f"{label} horizons {observed_horizons} do not match {list(horizons)}"
        )
    scopes = set(result["state_target_scope"].astype(str))
    if scopes != {expected_scope}:
        raise ValueError(f"{label} state target scopes do not match {expected_scope}")
    feature_counts = set(
        pd.to_numeric(result["state_target_feature_count"], errors="raise")
        .astype(int)
        .tolist()
    )
    if feature_counts != {int(expected_feature_count)}:
        raise ValueError(
            f"{label} state target feature counts do not match {expected_feature_count}"
        )
    for metric in metrics:
        result[metric] = pd.to_numeric(result[metric], errors="coerce")
        if not np.isfinite(result[metric].to_numpy(dtype=np.float64)).all():
            raise ValueError(f"{label} metric {metric} contains non-finite values")
    return result.sort_values(list(KEY_COLUMNS)).reset_index(drop=True)


def evaluate_seed_stability(
    contract: Mapping[str, Any],
    reference_folds: Mapping[str, pd.DataFrame],
    candidate_folds: Mapping[str, pd.DataFrame],
    direct_comparisons: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compare two encoder seeds under an immutable paired rollout contract."""

    if int(contract.get("schema_version", 0)) != 1:
        raise ValueError("seed stability contract schema_version must be 1")
    if contract.get("promotion_eligible_from_this_audit_alone") is not False:
        raise ValueError("seed stability diagnostics cannot be promotion eligible")
    if contract.get("live_orders_allowed") is not False:
        raise ValueError("seed stability diagnostics must prohibit live orders")

    folds = tuple(str(value) for value in contract.get("folds", ()))
    horizons = tuple(int(value) for value in contract.get("horizons", ()))
    if len(folds) != 5 or len(set(folds)) != 5:
        raise ValueError("seed stability requires five unique folds")
    if not horizons or len(set(horizons)) != len(horizons):
        raise ValueError("seed stability requires unique horizons")
    if set(reference_folds) != set(folds) or set(candidate_folds) != set(folds):
        raise ValueError("reference and candidate folds must match the contract")

    target = _require_mapping(contract.get("state_target"), "state_target")
    expected_scope = str(target.get("scope"))
    expected_feature_count = int(target.get("feature_count", 0))
    metric_contracts = _require_mapping(contract.get("metrics"), "metrics")
    metrics = tuple(str(value) for value in metric_contracts)
    if not metrics:
        raise ValueError("seed stability contract must declare metrics")

    normalized_reference: dict[str, pd.DataFrame] = {}
    normalized_candidate: dict[str, pd.DataFrame] = {}
    for fold in folds:
        normalized_reference[fold] = _normalize_frame(
            reference_folds[fold],
            label=f"reference {fold}",
            horizons=horizons,
            metrics=metrics,
            expected_scope=expected_scope,
            expected_feature_count=expected_feature_count,
        )
        normalized_candidate[fold] = _normalize_frame(
            candidate_folds[fold],
            label=f"candidate {fold}",
            horizons=horizons,
            metrics=metrics,
            expected_scope=expected_scope,
            expected_feature_count=expected_feature_count,
        )

    fold_rows: dict[str, Any] = {}
    aggregate_inputs: dict[int, list[pd.DataFrame]] = {value: [] for value in horizons}
    positive_cells = {metric: 0 for metric in metrics}
    total_cells = len(folds) * len(horizons)
    for fold in folds:
        reference = normalized_reference[fold]
        candidate = normalized_candidate[fold]
        if not reference.loc[:, list(KEY_COLUMNS)].equals(
            candidate.loc[:, list(KEY_COLUMNS)]
        ):
            raise ValueError(f"{fold} reference and candidate dates do not match")
        fold_horizons: dict[str, Any] = {}
        for horizon in horizons:
            reference_h = reference.loc[reference["horizon"] == horizon]
            candidate_h = candidate.loc[candidate["horizon"] == horizon]
            merged = reference_h.merge(
                candidate_h,
                on=list(KEY_COLUMNS),
                how="inner",
                validate="one_to_one",
                suffixes=("_reference", "_candidate"),
            )
            if len(merged) != len(reference_h):
                raise ValueError(f"{fold} h{horizon} paired rows are incomplete")
            aggregate_inputs[horizon].append(merged)
            metric_rows: dict[str, Any] = {}
            for metric in metrics:
                reference_values = merged[f"{metric}_reference"].to_numpy(
                    dtype=np.float64
                )
                candidate_values = merged[f"{metric}_candidate"].to_numpy(
                    dtype=np.float64
                )
                reference_summary = newey_west_mean(reference_values, lag=horizon)
                candidate_summary = newey_west_mean(candidate_values, lag=horizon)
                paired_summary = newey_west_mean(
                    candidate_values - reference_values,
                    lag=horizon,
                )
                candidate_mean = float(candidate_summary["mean"])
                if candidate_mean > 0.0:
                    positive_cells[metric] += 1
                metric_rows[metric] = {
                    "reference": reference_summary,
                    "candidate": candidate_summary,
                    "candidate_minus_reference": paired_summary,
                }
            fold_horizons[str(horizon)] = {
                "rows": int(len(merged)),
                "metrics": metric_rows,
            }
        fold_rows[fold] = {"horizons": fold_horizons}

    aggregate: dict[str, Any] = {}
    retention_by_metric: dict[str, dict[str, float]] = {metric: {} for metric in metrics}
    for horizon in horizons:
        merged = pd.concat(aggregate_inputs[horizon], ignore_index=True)
        if merged["date"].duplicated().any():
            raise ValueError(f"h{horizon} evaluation dates overlap across folds")
        metric_rows = {}
        for metric in metrics:
            reference_values = merged[f"{metric}_reference"].to_numpy(dtype=np.float64)
            candidate_values = merged[f"{metric}_candidate"].to_numpy(dtype=np.float64)
            reference_summary = newey_west_mean(reference_values, lag=horizon)
            candidate_summary = newey_west_mean(candidate_values, lag=horizon)
            paired_summary = newey_west_mean(
                candidate_values - reference_values,
                lag=horizon,
            )
            reference_mean = float(reference_summary["mean"])
            if reference_mean <= 0.0:
                raise ValueError(
                    f"reference {metric} mean must be positive at h{horizon}"
                )
            retention = float(candidate_summary["mean"]) / reference_mean
            retention_by_metric[metric][str(horizon)] = retention
            metric_rows[metric] = {
                "reference": reference_summary,
                "candidate": candidate_summary,
                "candidate_minus_reference": paired_summary,
                "retention_ratio": retention,
            }
        aggregate[str(horizon)] = {
            "rows": int(len(merged)),
            "metrics": metric_rows,
        }

    checks: list[dict[str, Any]] = []

    def add_check(check_id: str, passed: bool, actual: Any, threshold: Any) -> None:
        checks.append(
            {
                "id": check_id,
                "passed": bool(passed),
                "actual": actual,
                "threshold": threshold,
            }
        )

    for metric, raw_rules in metric_contracts.items():
        rules = _require_mapping(raw_rules, f"metrics.{metric}")
        minimum_cells = int(rules.get("minimum_positive_fold_horizon_cells", 0))
        add_check(
            f"{metric}:positive_fold_horizon_cells",
            positive_cells[metric] >= minimum_cells,
            positive_cells[metric],
            minimum_cells,
        )
        minimum_retention = float(rules.get("minimum_aggregate_retention_ratio", 0.0))
        for horizon in horizons:
            retention = retention_by_metric[metric][str(horizon)]
            add_check(
                f"{metric}:h{horizon}:aggregate_retention",
                retention >= minimum_retention,
                retention,
                minimum_retention,
            )

    significance = _require_mapping(
        contract.get("aggregate_significance"), "aggregate_significance"
    )
    significance_metric = str(significance.get("metric"))
    significance_horizon = int(significance.get("horizon", -1))
    minimum_mean = float(significance.get("minimum_mean", 0.0))
    minimum_t = float(significance.get("minimum_newey_west_t", 0.0))
    if significance_metric not in metrics or significance_horizon not in horizons:
        raise ValueError("aggregate significance metric/horizon is not declared")
    significance_row = aggregate[str(significance_horizon)]["metrics"][
        significance_metric
    ]["candidate"]
    add_check(
        "aggregate_significance:mean",
        float(significance_row["mean"]) >= minimum_mean,
        float(significance_row["mean"]),
        minimum_mean,
    )
    add_check(
        "aggregate_significance:newey_west_t",
        float(significance_row["newey_west_t"]) >= minimum_t,
        float(significance_row["newey_west_t"]),
        minimum_t,
    )

    direct_rules = _require_mapping(contract.get("direct_baseline"), "direct_baseline")
    direct_required = direct_rules.get("required") is True
    if direct_required and direct_comparisons is None:
        raise ValueError("direct baseline comparisons are required by the contract")
    if direct_comparisons is not None and set(direct_comparisons) != set(folds):
        raise ValueError("direct baseline comparison folds must match the contract")
    primary_horizons = {
        int(value) for value in direct_rules.get("primary_horizons", ())
    }
    if not primary_horizons.issubset(set(horizons)):
        raise ValueError("direct baseline primary horizons are not declared")
    direct_t_threshold = float(
        direct_rules.get("significant_direct_advantage_newey_west_t", 1.96)
    )
    direct_rows: dict[str, Any] = {}
    if direct_comparisons is not None:
        for fold in folds:
            comparison = direct_comparisons[fold]
            if comparison.get("state_target_scope") != expected_scope:
                raise ValueError(f"direct comparison {fold} uses the wrong target scope")
            if int(comparison.get("state_target_feature_count", 0)) != int(
                expected_feature_count
            ):
                raise ValueError(
                    f"direct comparison {fold} uses the wrong target feature count"
                )
            comparison_horizons = _require_mapping(
                comparison.get("horizons"), f"direct comparison {fold}.horizons"
            )
            if {int(value) for value in comparison_horizons} != set(horizons):
                raise ValueError(f"direct comparison {fold} horizons do not match")
            fold_direct_rows: dict[str, Any] = {}
            for horizon in horizons:
                horizon_row = _require_mapping(
                    comparison_horizons.get(str(horizon)),
                    f"direct comparison {fold} h{horizon}",
                )
                state_skill = _require_mapping(
                    horizon_row.get("state_skill"),
                    f"direct comparison {fold} h{horizon}.state_skill",
                )
                delta = _require_mapping(
                    state_skill.get("delta_direct_minus_jepa"),
                    f"direct comparison {fold} h{horizon}.delta",
                )
                mean = float(delta.get("mean", float("nan")))
                newey_west_t = float(delta.get("newey_west_t", float("nan")))
                if not np.isfinite((mean, newey_west_t)).all():
                    raise ValueError(
                        f"direct comparison {fold} h{horizon} is non-finite"
                    )
                if horizon in primary_horizons:
                    add_check(
                        f"direct_baseline:{fold}:h{horizon}:candidate_not_below_direct",
                        mean <= 0.0,
                        mean,
                        "direct_minus_candidate_mean <= 0",
                    )
                add_check(
                    f"direct_baseline:{fold}:h{horizon}:no_significant_direct_advantage",
                    newey_west_t < direct_t_threshold,
                    newey_west_t,
                    f"< {direct_t_threshold}",
                )
                fold_direct_rows[str(horizon)] = {
                    "direct_minus_candidate_mean": mean,
                    "direct_minus_candidate_newey_west_t": newey_west_t,
                }
            direct_rows[fold] = {"horizons": fold_direct_rows}

    passed = all(row["passed"] for row in checks)
    return {
        "schema_version": 1,
        "role": "paired_multifold_encoder_seed_stability_diagnostic",
        "reference_seed": int(contract["reference_seed"]),
        "candidate_seed": int(contract["candidate_seed"]),
        "state_target": {
            "scope": expected_scope,
            "feature_count": expected_feature_count,
        },
        "folds": fold_rows,
        "aggregate": aggregate,
        "positive_fold_horizon_cells": positive_cells,
        "total_fold_horizon_cells": total_cells,
        "direct_baseline": direct_rows,
        "gate": {
            "passed": passed,
            "checks": checks,
            "failures": [row["id"] for row in checks if not row["passed"]],
        },
        "decision": "stable_diagnostic" if passed else "unstable_diagnostic",
        "promotion_eligible_from_this_audit_alone": False,
        "live_orders_allowed": False,
    }
