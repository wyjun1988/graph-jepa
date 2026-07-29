from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from scripts.benchmark_direct_baselines import newey_west_mean


REQUIRED_HORIZONS = (1, 2, 3, 5, 10)
REQUIRED_LEGACY_CHECKS = (
    "frozen_preflight_verification",
    "stock_coverage",
    "evaluation_length",
    "current_imputation_skill",
    "future_state_skill",
    "rollout_dependency",
    "m1_max_latency",
    "shadow_safety",
    "dataset_integrity",
    "ohlcv_integrity",
)


def _check(
    check_id: str,
    passed: bool,
    value: Any,
    requirement: str,
    *,
    module: str,
) -> dict[str, Any]:
    return {
        "id": check_id,
        "module": module,
        "passed": bool(passed),
        "value": value,
        "requirement": requirement,
    }


def _legacy_module_checks(legacy_gate: Mapping[str, Any]) -> list[dict[str, Any]]:
    checks = list(legacy_gate.get("checks") or [])
    result = []
    for check_id in REQUIRED_LEGACY_CHECKS:
        selected = [row for row in checks if row.get("id") == check_id]
        result.append(
            _check(
                f"legacy_{check_id}",
                bool(selected) and all(bool(row.get("passed")) for row in selected),
                {
                    "instances": len(selected),
                    "failed": sum(not bool(row.get("passed")) for row in selected),
                },
                "all applicable instances from the frozen rolling gate pass",
                module="infrastructure" if check_id not in {
                    "current_imputation_skill",
                    "future_state_skill",
                    "rollout_dependency",
                } else "state",
            )
        )
    return result


def _state_checks(state_parity: Mapping[str, Any]) -> list[dict[str, Any]]:
    folds = state_parity.get("fold_results") or {}
    return [
        _check(
            "state_temporal_target_contract",
            state_parity.get("state_target_scope") == "checkpoint_temporal"
            and int(state_parity.get("state_target_feature_count", 0)) == 127,
            {
                "scope": state_parity.get("state_target_scope"),
                "features": state_parity.get("state_target_feature_count"),
            },
            "checkpoint temporal target scope with 127 supervised features",
            module="state",
        ),
        _check(
            "state_five_fold_direct_parity",
            len(folds) == 5
            and bool(state_parity.get("all_folds_contract_gate_passed"))
            and all(bool(row.get("fold_contract_gate_passed")) for row in folds.values()),
            {
                "folds": len(folds),
                "all_passed": state_parity.get("all_folds_contract_gate_passed"),
            },
            "all five folds pass the corrected 127-target JEPA versus direct comparison",
            module="state",
        ),
        _check(
            "state_artifact_is_non_trading",
            state_parity.get("live_orders_allowed") is False,
            state_parity.get("live_orders_allowed"),
            "live_orders_allowed=false",
            module="state",
        ),
    ]


def _return_checks(
    qlib_summaries: Mapping[str, Mapping[str, Any]],
    qlib_daily: Mapping[str, pd.DataFrame],
    *,
    aggregate_t_min: float,
    positive_fold_fraction_min: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels = sorted(qlib_summaries)
    reports_aligned = len(labels) == 5 and set(labels) == set(qlib_daily)
    artifact_contracts = []
    cells = []
    h10_frames = []
    positive_h10_folds = 0
    dates_are_unique = True

    if reports_aligned:
        for label in labels:
            summary = qlib_summaries[label]
            artifact_contracts.append(
                summary.get("live_orders_allowed") is False
                and summary.get("test_used_for_selection") is False
                and summary.get("role") == "research_only_qlib_lightgbm_baseline"
            )
            horizons = summary.get("horizons") or {}
            for horizon in REQUIRED_HORIZONS:
                metric = (
                    horizons.get(str(horizon), {})
                    .get("splits", {})
                    .get("test", {})
                    .get("return_path_ic_top300", {})
                )
                cells.append(
                    {
                        "fold": label,
                        "horizon": horizon,
                        "mean": metric.get("mean"),
                        "rows": metric.get("rows"),
                    }
                )

            daily = qlib_daily[label].copy()
            if "split" in daily:
                daily = daily.loc[daily["split"].astype(str) == "test"]
            daily["horizon"] = pd.to_numeric(daily["horizon"], errors="raise").astype(int)
            h10 = daily.loc[
                daily["horizon"] == 10,
                ["date", "return_path_ic_top300"],
            ].copy()
            if h10["date"].duplicated().any():
                dates_are_unique = False
            h10["fold"] = label
            h10_frames.append(h10)
            if float(h10["return_path_ic_top300"].mean()) > 0.0:
                positive_h10_folds += 1

    valid_cells = (
        len(cells) == 25
        and all(row["mean"] is not None and np.isfinite(float(row["mean"])) for row in cells)
    )
    all_positive = valid_cells and all(float(row["mean"]) > 0.0 for row in cells)
    aggregate = {
        "rows": 0,
        "mean": float("nan"),
        "newey_west_t": float("nan"),
    }
    if reports_aligned and h10_frames:
        combined = pd.concat(h10_frames, ignore_index=True)
        if combined["date"].duplicated().any():
            dates_are_unique = False
        combined = combined.sort_values("date")
        aggregate = newey_west_mean(
            combined["return_path_ic_top300"].to_numpy(dtype=np.float64),
            lag=10,
        )
    positive_fraction = positive_h10_folds / len(labels) if labels else 0.0

    checks = [
        _check(
            "return_five_fold_artifacts",
            reports_aligned and all(artifact_contracts),
            {"labels": labels, "artifact_contracts": artifact_contracts},
            "five aligned research-only Qlib reports with no test selection",
            module="return",
        ),
        _check(
            "return_all_fold_horizon_top300_positive",
            all_positive,
            {"positive": sum(float(row["mean"]) > 0.0 for row in cells if row["mean"] is not None), "total": len(cells)},
            "all 25 fold/horizon top300 IC means are positive",
            module="return",
        ),
        _check(
            "return_h10_dates_non_overlapping",
            reports_aligned and dates_are_unique,
            dates_are_unique,
            "test dates are unique within and across rolling folds",
            module="return",
        ),
        _check(
            "return_aggregate_h10_significance",
            float(aggregate.get("mean", float("nan"))) > 0.0
            and float(aggregate.get("newey_west_t", float("nan"))) >= float(aggregate_t_min),
            aggregate,
            f"aggregate top300 h10 IC mean>0 and Newey-West t>={aggregate_t_min}",
            module="return",
        ),
        _check(
            "return_positive_h10_fold_fraction",
            positive_fraction >= float(positive_fold_fraction_min),
            positive_fraction,
            f"positive h10 fold fraction>={positive_fold_fraction_min}",
            module="return",
        ),
    ]
    return checks, {"cells": cells, "aggregate_h10": aggregate}


def _impact_checks(
    comparison: Mapping[str, Any],
    stability: Mapping[str, Any],
) -> list[dict[str, Any]]:
    folds = comparison.get("folds") or {}
    absolute_folds = {
        label: bool((row.get("jepa_absolute_gate") or {}).get("passed"))
        for label, row in folds.items()
    }
    advantage = comparison.get("jepa_specific_advantage") or {}
    overall = stability.get("overall") or {}
    return [
        _check(
            "impact_absolute_multifold_gate",
            len(absolute_folds) >= 2
            and all(absolute_folds.values())
            and bool(comparison.get("jepa_absolute_historical_gate_passed")),
            absolute_folds,
            "JEPA systemic head passes every preregistered historical fold",
            module="impact",
        ),
        _check(
            "impact_jepa_specific_advantage",
            bool(advantage.get("passed")),
            advantage,
            "JEPA systemic head has positive transferable advantage over the direct comparator",
            module="impact",
        ),
        _check(
            "impact_seed_fold_stability",
            bool(stability.get("all_runs_passed"))
            and bool(overall.get("all_runs_passed")),
            {
                "runs": overall.get("runs"),
                "passed_runs": overall.get("passed_runs"),
                "all_runs_passed": overall.get("all_runs_passed"),
            },
            "all preregistered seed/fold stability runs pass",
            module="impact",
        ),
        _check(
            "impact_artifacts_are_non_trading",
            comparison.get("live_orders_allowed") is False
            and stability.get("live_orders_allowed") is False,
            {
                "comparison": comparison.get("live_orders_allowed"),
                "stability": stability.get("live_orders_allowed"),
            },
            "live_orders_allowed=false for every impact artifact",
            module="impact",
        ),
    ]


def _integration_checks(
    modular_latency: Mapping[str, Any] | None,
    *,
    latency_p95_ms_max: float,
) -> list[dict[str, Any]]:
    payload = modular_latency or {}
    p95 = payload.get("total_p95_ms")
    passed = (
        payload.get("status") == "pass"
        and payload.get("live_orders_allowed") is False
        and p95 is not None
        and float(p95) <= float(latency_p95_ms_max)
        and bool(payload.get("jepa_and_qlib_executed"))
    )
    return [
        _check(
            "integration_jepa_qlib_latency",
            passed,
            payload if payload else {"status": "missing"},
            f"combined JEPA and Qlib p95<={latency_p95_ms_max} ms with live orders disabled",
            module="integration",
        )
    ]


def evaluate_modular_shadow_gate(
    legacy_gate: Mapping[str, Any],
    state_parity: Mapping[str, Any],
    qlib_summaries: Mapping[str, Mapping[str, Any]],
    qlib_daily: Mapping[str, pd.DataFrame],
    impact_comparison: Mapping[str, Any],
    impact_stability: Mapping[str, Any],
    modular_latency: Mapping[str, Any] | None = None,
    *,
    aggregate_t_min: float = 1.96,
    positive_fold_fraction_min: float = 0.8,
    latency_p95_ms_max: float = 250.0,
) -> dict[str, Any]:
    checks = _legacy_module_checks(legacy_gate)
    checks.extend(_state_checks(state_parity))
    return_checks, return_evidence = _return_checks(
        qlib_summaries,
        qlib_daily,
        aggregate_t_min=aggregate_t_min,
        positive_fold_fraction_min=positive_fold_fraction_min,
    )
    checks.extend(return_checks)
    checks.extend(_impact_checks(impact_comparison, impact_stability))
    checks.extend(
        _integration_checks(
            modular_latency,
            latency_p95_ms_max=latency_p95_ms_max,
        )
    )

    modules = {}
    for module in sorted({str(row["module"]) for row in checks}):
        selected = [row for row in checks if row["module"] == module]
        failed = [row for row in selected if not row["passed"]]
        modules[module] = {
            "status": "pass" if not failed else "blocked",
            "passed": len(selected) - len(failed),
            "failed": len(failed),
            "total": len(selected),
            "failed_check_ids": [row["id"] for row in failed],
        }
    failed = [row for row in checks if not row["passed"]]
    return {
        "schema_version": 1,
        "status": "pass" if not failed else "blocked",
        "approval_scope": "read_only_shadow" if not failed else "none",
        "role": "modular_graph_jepa_shadow_qualification",
        "architecture": {
            "state_model": "graph_jepa_v6_500_stock_world_state",
            "return_model": "qlib_lightgbm_raw_pit_features",
            "impact_model": "graph_jepa_systemic_transition_head",
            "prediction_blending": False,
        },
        "modules": modules,
        "checks": checks,
        "return_evidence": return_evidence,
        "summary": {
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "total": len(checks),
        },
        "shadow_candidate_count": 1 if not failed else 0,
        "live_orders_allowed": False,
    }
