from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from scripts.benchmark_direct_baselines import newey_west_mean
from stock_v2.rolling_validation import REQUIRED_HORIZONS, validate_rolling_contract


def evaluate_rolling_gate(
    contract: Mapping[str, Any],
    frozen_verification: Mapping[str, Any],
    node_summaries: Mapping[str, Mapping[str, Any]],
    direct_comparisons: Mapping[str, Mapping[str, Any]],
    head_summaries: Mapping[str, Mapping[str, Any]],
    head_daily: Mapping[str, pd.DataFrame],
    qlib_comparisons: Mapping[str, Mapping[str, Any]],
    latency: Mapping[str, Any],
    safety: Mapping[str, Any],
    dataset_audit: Mapping[str, Any],
    ohlcv_audit: Mapping[str, Any],
) -> dict[str, Any]:
    validated = validate_rolling_contract(contract)
    labels = [str(row["label"]) for row in validated["folds"]]
    required_labels = set(labels)
    for role, values in (
        ("node", node_summaries),
        ("direct", direct_comparisons),
        ("head", head_summaries),
        ("head_daily", head_daily),
        ("qlib", qlib_comparisons),
    ):
        if set(values) != required_labels:
            raise ValueError(f"{role} fold labels do not match the contract")

    checks: list[dict[str, Any]] = []

    def add(check_id: str, passed: bool, value: Any, requirement: str, **meta: Any) -> None:
        checks.append(
            {
                "id": check_id,
                "passed": bool(passed),
                "value": value,
                "requirement": requirement,
                **meta,
            }
        )

    add(
        "frozen_preflight_verification",
        frozen_verification.get("status") == "pass"
        and frozen_verification.get("live_orders_allowed") is False
        and len(frozen_verification.get("verified_folds") or []) == len(labels),
        {
            "status": frozen_verification.get("status"),
            "folds": len(frozen_verification.get("verified_folds") or []),
        },
        "status=pass, all five folds verified, live orders false",
    )

    per_fold_gate = validated["gates"]["per_fold"]
    required_stock_nodes = int(
        (validated.get("sensor_gates") or {}).get(
            "stock_nodes_required",
            (validated.get("architecture") or {}).get("stock_nodes", 0),
        )
    )
    if required_stock_nodes <= 0:
        raise ValueError("rolling contract must declare a positive stock node count")
    direct_significant_losses = 0
    qlib_significant_losses = 0
    h10_frames = []
    positive_h10_folds = 0
    for fold in validated["folds"]:
        label = str(fold["label"])
        node = node_summaries[label]
        direct = direct_comparisons[label]
        head = head_summaries[label]
        qlib = qlib_comparisons[label]
        for role, payload in (("node", node), ("direct", direct), ("head", head), ("qlib", qlib)):
            add(
                "research_only_artifact",
                payload.get("live_orders_allowed") is False,
                payload.get("live_orders_allowed"),
                "live_orders_allowed=false",
                fold=label,
                artifact=role,
            )
        add(
            "stock_coverage",
            int(node.get("stock_node_count", 0)) == required_stock_nodes,
            int(node.get("stock_node_count", 0)),
            f"exactly {required_stock_nodes} stock nodes",
            fold=label,
        )
        eval_steps = int(node.get("eval_steps", 0))
        add(
            "evaluation_length",
            eval_steps >= int(per_fold_gate["minimum_evaluation_steps"]),
            eval_steps,
            f">={int(per_fold_gate['minimum_evaluation_steps'])} signal dates",
            fold=label,
        )
        current_skill = float(
            node["current_imputation"]["all"]["mse_skill_vs_zero"]["mean"]
        )
        add(
            "current_imputation_skill",
            current_skill > float(per_fold_gate["current_imputation_skill_min_exclusive"]),
            current_skill,
            ">0 versus zero-state baseline",
            fold=label,
        )

        for horizon in REQUIRED_HORIZONS:
            key = str(horizon)
            state_skill = float(
                node["future_rollout_by_horizon"][key][
                    "pooled_mse_skill_vs_persistence"
                ]
            )
            rollout_dependency = float(
                node["rollout_dependency_by_horizon"][key][
                    "pooled_mse_skill_vs_no_rollout"
                ]
            )
            direct_delta = direct["horizons"][key]["state_skill"][
                "delta_direct_minus_jepa"
            ]
            direct_margin = -float(direct_delta["mean"])
            if float(direct_delta["mean"]) > 0.0 and float(
                direct_delta["newey_west_t"]
            ) >= 1.96:
                direct_significant_losses += 1
            all_ic = float(head["horizons"][key]["all_stock"]["mean"])
            top_ic = float(head["horizons"][key]["top300"]["mean"])
            add(
                "future_state_skill",
                state_skill > float(per_fold_gate["persistence_node_skill_min_exclusive"]),
                state_skill,
                ">0 versus persistence",
                fold=label,
                horizon=horizon,
            )
            add(
                "rollout_dependency",
                rollout_dependency > float(per_fold_gate["rollout_dependency_min_exclusive"]),
                rollout_dependency,
                ">0 versus zero rollout innovation",
                fold=label,
                horizon=horizon,
            )
            add(
                "direct_node_margin",
                direct_margin > float(per_fold_gate["direct_node_margin_min_exclusive"]),
                direct_margin,
                "JEPA persistence skill strictly exceeds direct MLP",
                fold=label,
                horizon=horizon,
            )
            add(
                "path_ic_all",
                all_ic > float(per_fold_gate["path_ic_min_exclusive"]),
                all_ic,
                "mean path IC >0",
                fold=label,
                horizon=horizon,
            )
            add(
                "path_ic_top300",
                top_ic > float(per_fold_gate["path_ic_min_exclusive"]),
                top_ic,
                "mean top300 path IC >0",
                fold=label,
                horizon=horizon,
            )
            for metric in ("entry_path_ic", "entry_path_ic_top300"):
                qlib_row = qlib["horizons"][key][metric]
                if bool(qlib_row.get("qlib_significantly_superior")):
                    qlib_significant_losses += 1

        daily = head_daily[label].copy()
        h10 = daily.loc[
            pd.to_numeric(daily["horizon"], errors="raise").astype(int) == 10,
            ["date", "entry_path_ic_top300"],
        ].copy()
        if h10["date"].duplicated().any() or len(h10) != eval_steps:
            raise ValueError(f"{label} h10 daily path rows differ from node evaluation")
        h10["fold"] = label
        h10_frames.append(h10)
        if float(h10["entry_path_ic_top300"].mean()) > 0.0:
            positive_h10_folds += 1

    aggregate_gate = validated["gates"]["aggregate"]
    combined = pd.concat(h10_frames, ignore_index=True)
    if combined["date"].duplicated().any():
        raise ValueError("rolling fold signal dates overlap")
    combined = combined.sort_values("date")
    aggregate_h10 = newey_west_mean(
        combined["entry_path_ic_top300"].to_numpy(dtype=np.float64),
        lag=10,
    )
    positive_fold_fraction = positive_h10_folds / len(labels)
    add(
        "aggregate_top300_h10_significance",
        float(aggregate_h10["mean"]) > 0.0
        and float(aggregate_h10["newey_west_t"]) >= float(
            aggregate_gate["top300_h10_newey_west_t_min"]
        ),
        aggregate_h10,
        f"mean>0 and Newey-West t>={aggregate_gate['top300_h10_newey_west_t_min']}",
    )
    add(
        "positive_h10_fold_fraction",
        positive_fold_fraction >= float(aggregate_gate["positive_fold_fraction_min"]),
        positive_fold_fraction,
        f">={aggregate_gate['positive_fold_fraction_min']}",
    )
    add(
        "significant_direct_losses",
        direct_significant_losses <= int(aggregate_gate["significant_direct_losses_max"]),
        direct_significant_losses,
        f"<={aggregate_gate['significant_direct_losses_max']}",
    )
    add(
        "significant_qlib_losses",
        qlib_significant_losses == 0,
        qlib_significant_losses,
        "zero significantly superior Qlib path cells",
    )

    latency_limit = float(
        validated["gates"]["latency"]["m1_max_rollout10_p95_ms_max"]
    )
    latency_p95_ms = float(latency["total_sec"]["p95"]) * 1000.0
    add(
        "m1_max_latency",
        latency.get("live_orders_allowed") is False
        and int(latency.get("rollout_steps", 0)) == 10
        and latency_p95_ms <= latency_limit,
        latency_p95_ms,
        f"10-step p95 <= {latency_limit} ms",
    )
    required_safety = int(
        validated["gates"]["safety"]["required_shadow_safety_checks"]
    )
    add(
        "shadow_safety",
        safety.get("status") == "pass"
        and safety.get("live_orders_allowed") is False
        and int((safety.get("summary") or {}).get("passed", 0)) == required_safety
        and int((safety.get("summary") or {}).get("failed", -1)) == 0,
        safety.get("summary"),
        f"{required_safety}/{required_safety} read-only safety checks",
    )
    add(
        "dataset_integrity",
        dataset_audit.get("status") == "pass"
        and not (dataset_audit.get("blockers") or []),
        {
            "status": dataset_audit.get("status"),
            "blockers": dataset_audit.get("blockers"),
        },
        "status=pass and blockers empty",
    )
    add(
        "ohlcv_integrity",
        ohlcv_audit.get("status") == "pass"
        and not (ohlcv_audit.get("blockers") or [])
        and int(ohlcv_audit.get("verified_files", 0)) >= required_stock_nodes,
        {
            "status": ohlcv_audit.get("status"),
            "verified_files": ohlcv_audit.get("verified_files"),
            "blockers": ohlcv_audit.get("blockers"),
        },
        (
            "status=pass, blockers empty, and "
            f">={required_stock_nodes} verified files"
        ),
    )

    failed = [row for row in checks if not row["passed"]]
    return {
        "status": "pass" if not failed else "blocked",
        "approval_scope": "read_only_shadow" if not failed else "none",
        "role": "rolling_v6_read_only_shadow_qualification",
        "folds": len(labels),
        "aggregate_top300_h10": aggregate_h10,
        "checks": checks,
        "summary": {
            "passed": len(checks) - len(failed),
            "failed": len(failed),
            "total": len(checks),
        },
        "live_orders_allowed": False,
    }
