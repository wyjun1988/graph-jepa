from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import statistics
from typing import Any, Iterable


HORIZONS = (1, 2, 3, 5, 10)
HORIZON_WEIGHTS = {1: 2.0, 2: 2.0, 3: 1.0, 5: 1.0, 10: 1.0}
PRIMARY_METRICS = (
    "precision",
    "impact_lift",
    "captured_direction_accuracy",
    "predicted_bucket_direction_accuracy",
    "tail_ic",
    "magnitude_lift",
    "signed_ic",
    "realized_tail_mass_recall",
    "captured_impact_weighted_direction_accuracy",
    "signed_realized_tail_mass_capture",
)
FIXED_K_METRICS = (
    "impact_precision_at_k",
    "impact_lift_at_k",
    "direction_accuracy_at_k",
    "captured_direction_accuracy_at_k",
    "joint_correct_precision_at_k",
    "magnitude_lift_at_k",
    "realized_tail_mass_recall_at_k",
    "captured_impact_weighted_direction_accuracy_at_k",
    "signed_realized_tail_mass_capture_at_k",
)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_and_fold(path: Path, payload: dict[str, Any]) -> tuple[int, str]:
    text = f"{path} {payload.get('model_dir', '')}"
    seed_match = re.search(r"seed(\d+)", text)
    fold_match = re.search(r"fold([12])", text)
    if not seed_match or not fold_match:
        raise ValueError(f"cannot infer seed/fold from {path}")
    return int(seed_match.group(1)), f"fold{fold_match.group(1)}"


def weighted_primary_metric(
    payload: dict[str, Any], variant: str, metric: str
) -> float:
    weighted = 0.0
    weight_sum = 0.0
    for horizon in HORIZONS:
        metric_payload = payload["metrics"][str(horizon)]["top300"]["0.10"][
            variant
        ].get(metric)
        if not metric_payload:
            continue
        value = float(metric_payload["mean"])
        if not math.isfinite(value):
            continue
        weight = HORIZON_WEIGHTS[horizon]
        weighted += weight * value
        weight_sum += weight
    return weighted / weight_sum if weight_sum else float("nan")


def primary_score(payload: dict[str, Any], variant: str) -> float:
    precision = weighted_primary_metric(payload, variant, "precision")
    direction = weighted_primary_metric(
        payload, variant, "captured_direction_accuracy"
    )
    tail_ic = weighted_primary_metric(payload, variant, "tail_ic")
    if not all(math.isfinite(value) for value in (precision, direction, tail_ic)):
        return float("nan")
    impact_skill = (precision - 0.10) / 0.90
    direction_skill = 2.0 * (direction - 0.50)
    return 0.50 * impact_skill + 0.30 * direction_skill + 0.20 * tail_ic


def weighted_fixed_metric(
    payload: dict[str, Any],
    strategy: str,
    metric: str,
    count: int = 3,
    metric_root: str = "metrics",
) -> float:
    root = payload.get(metric_root)
    if not root:
        return float("nan")
    weighted = 0.0
    weight_sum = 0.0
    for horizon in HORIZONS:
        metric_payload = root[str(horizon)]["top300"][
            str(int(count))
        ][strategy].get(metric)
        if not metric_payload:
            continue
        value = float(metric_payload["mean"])
        if not math.isfinite(value):
            continue
        weight = HORIZON_WEIGHTS[horizon]
        weighted += weight * value
        weight_sum += weight
    return weighted / weight_sum if weight_sum else float("nan")


def describe(values: Iterable[float]) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": statistics.fmean(finite),
        "std": statistics.stdev(finite) if len(finite) > 1 else 0.0,
        "min": min(finite),
        "max": max(finite),
    }


def discover_jepa(root: Path) -> list[dict[str, Any]]:
    paths = list(root.glob("reports/impact_head_weight95_seed*/fold*/summary.json"))
    paths.extend(
        root.glob(
            "reports/walk_forward_causal453_path_multiseed_seed*_20260714/"
            "impact_head_weight95/fold*/summary.json"
        )
    )
    records = []
    seen = set()
    for path in sorted(paths):
        payload = load_json(path)
        seed, fold = seed_and_fold(path, payload)
        key = (seed, fold)
        if key in seen:
            continue
        seen.add(key)
        record = {
            "seed": seed,
            "fold": fold,
            "path": str(path),
            "weighted_impact_score": float(payload["weighted_impact_score"]),
            "internal_scores": {
                variant: primary_score(payload, variant)
                for variant in ("impact_head", "signed_abs", "base_jepa")
            },
            "metrics": {
                metric: weighted_primary_metric(payload, "impact_head", metric)
                for metric in PRIMARY_METRICS
            },
            "live_orders_allowed": bool(payload.get("live_orders_allowed", True)),
        }
        record["beats_internal"] = record["internal_scores"]["impact_head"] > max(
            record["internal_scores"]["signed_abs"],
            record["internal_scores"]["base_jepa"],
        )
        records.append(record)
    return records


def discover_fixed_k(root: Path) -> list[dict[str, Any]]:
    paths = list(root.glob("reports/impact_head_fixed_k_seed*/fold*/summary.json"))
    paths.extend(
        root.glob(
            "reports/walk_forward_causal453_path_multiseed_seed*_20260714/"
            "impact_fixed_k/fold*/summary.json"
        )
    )
    records = []
    seen = set()
    for path in sorted(paths):
        payload = load_json(path)
        seed, fold = seed_and_fold(path, payload)
        key = (seed, fold)
        if key in seen:
            continue
        seen.add(key)
        records.append(
            {
                "seed": seed,
                "fold": fold,
                "path": str(path),
                "strategies": {
                    strategy: {
                        metric: weighted_fixed_metric(payload, strategy, metric, 3)
                        for metric in FIXED_K_METRICS
                    }
                    for strategy in (
                        "impact_only",
                        "impact_then_confidence",
                        "joint_75_25",
                    )
                },
                "cross_horizon_strategies": {
                    strategy: {
                        metric: weighted_fixed_metric(
                            payload,
                            strategy,
                            metric,
                            3,
                            metric_root="cross_horizon_metrics",
                        )
                        for metric in FIXED_K_METRICS
                    }
                    for strategy in (
                        "impact_only",
                        "impact_then_confidence",
                        "joint_75_25",
                    )
                },
                "cross_horizon_selection_contract": payload.get(
                    "cross_horizon_selection_contract"
                ),
                "live_orders_allowed": bool(payload.get("live_orders_allowed", True)),
            }
        )
    return records


def discover_direct_state(root: Path) -> list[dict[str, Any]]:
    records = []
    for path in sorted(
        root.glob("reports/direct_state_impact_v1_20260714/fold*/summary.json")
    ):
        payload = load_json(path)
        fold_match = re.search(r"fold([12])", str(path))
        if not fold_match:
            continue
        records.append(
            {
                "fold": f"fold{fold_match.group(1)}",
                "path": str(path),
                "scores": {
                    key: float(value)
                    for key, value in payload["primary_scores"].items()
                },
                "live_orders_allowed": bool(payload.get("live_orders_allowed", True)),
            }
        )
    return records


def discover_equal_direct(root: Path) -> list[dict[str, Any]]:
    patterns = (
        "reports/direct_impact_equal_objective*_20260714/*/fold*/summary.json",
        "reports/direct_impact_equal_objective*_20260714/fold*/*/summary.json",
    )
    records = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            payload = load_json(path)
            fold_match = re.search(r"fold([12])", str(path))
            if not fold_match:
                continue
            fold = f"fold{fold_match.group(1)}"
            mode = "graph" if payload.get("uses_graph_neighbor_state") else "nograph"
            key = (str(path.parent.parent.parent), mode, fold)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "fold": fold,
                    "mode": mode,
                    "path": str(path),
                    "weighted_impact_score": float(payload["weighted_impact_score"]),
                    "metrics": {
                        metric: weighted_primary_metric(
                            payload, "direct_impact", metric
                        )
                        for metric in PRIMARY_METRICS
                    },
                    "live_orders_allowed": bool(payload.get("live_orders_allowed", True)),
                }
            )
    return records


def discover_equal_direct_fixed_k(root: Path) -> list[dict[str, Any]]:
    patterns = (
        "reports/direct_impact_fixed_k_m1pro_v1_20260714/*/fold*/summary.json",
        "reports/direct_impact_fixed_k_equal*_20260714/*/fold*/summary.json",
    )
    records = []
    seen = set()
    for pattern in patterns:
        for path in sorted(root.glob(pattern)):
            payload = load_json(path)
            fold_match = re.search(r"fold([12])", str(path))
            if not fold_match:
                continue
            fold = f"fold{fold_match.group(1)}"
            mode = "graph" if payload.get("uses_graph_neighbor_state") else "nograph"
            key = (mode, fold)
            if key in seen:
                continue
            seen.add(key)
            records.append(
                {
                    "fold": fold,
                    "mode": mode,
                    "path": str(path),
                    "strategies": {
                        strategy: {
                            metric: weighted_fixed_metric(payload, strategy, metric, 3)
                            for metric in FIXED_K_METRICS
                        }
                        for strategy in (
                            "impact_only",
                            "impact_then_confidence",
                            "joint_75_25",
                        )
                    },
                    "cross_horizon_strategies": {
                        strategy: {
                            metric: weighted_fixed_metric(
                                payload,
                                strategy,
                                metric,
                                3,
                                metric_root="cross_horizon_metrics",
                            )
                            for metric in FIXED_K_METRICS
                        }
                        for strategy in (
                            "impact_only",
                            "impact_then_confidence",
                            "joint_75_25",
                        )
                    },
                    "cross_horizon_selection_contract": payload.get(
                        "cross_horizon_selection_contract"
                    ),
                    "live_orders_allowed": bool(
                        payload.get("live_orders_allowed", True)
                    ),
                }
            )
    return records


def completed_seeds(records: list[dict[str, Any]]) -> list[int]:
    by_seed: dict[int, set[str]] = {}
    for record in records:
        by_seed.setdefault(int(record["seed"]), set()).add(str(record["fold"]))
    return sorted(seed for seed, folds in by_seed.items() if folds == {"fold1", "fold2"})


def selection_contract_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    contracts = [
        record["cross_horizon_selection_contract"]
        for record in records
        if record.get("cross_horizon_selection_contract")
    ]
    return {
        "available_records": len(contracts),
        "all_same_candidates_across_horizons": bool(contracts)
        and all(item.get("same_candidates_across_horizons") for item in contracts),
        "all_within_requested_k": bool(contracts)
        and all(item.get("within_requested_k") for item in contracts),
        "maximum_selected_count": (
            max(int(item.get("maximum_selected_count", 0)) for item in contracts)
            if contracts
            else None
        ),
    }


def build_gate(jepa: list[dict[str, Any]], fixed: list[dict[str, Any]]) -> dict[str, Any]:
    seeds = completed_seeds(jepa)
    selected = [record for record in jepa if record["seed"] in seeds]
    by_seed_lift = {
        seed: statistics.fmean(
            record["metrics"]["impact_lift"]
            for record in selected
            if record["seed"] == seed
        )
        for seed in seeds
    }
    internal_rate = (
        statistics.fmean(float(record["beats_internal"]) for record in selected)
        if selected
        else float("nan")
    )
    checks = {
        "three_complete_seeds": len(seeds) >= 3,
        "mean_impact_lift_ge_1_25": bool(selected)
        and statistics.fmean(r["metrics"]["impact_lift"] for r in selected) >= 1.25,
        "worst_seed_impact_lift_gt_1": bool(by_seed_lift)
        and min(by_seed_lift.values()) > 1.0,
        "mean_magnitude_lift_ge_1_10": bool(selected)
        and statistics.fmean(r["metrics"]["magnitude_lift"] for r in selected) >= 1.10,
        "fractional_captured_direction_gt_0_55": bool(selected)
        and statistics.fmean(
            r["metrics"]["captured_direction_accuracy"] for r in selected
        )
        > 0.55,
        "mean_tail_ic_positive": bool(selected)
        and statistics.fmean(r["metrics"]["tail_ic"] for r in selected) > 0.0,
        "internal_win_rate_ge_0_80": math.isfinite(internal_rate)
        and internal_rate >= 0.80,
        "all_live_order_flags_false": all(
            not record["live_orders_allowed"] for record in [*jepa, *fixed]
        ),
    }
    status = "pending" if len(seeds) < 3 else ("pass" if all(checks.values()) else "fail")
    return {
        "status": status,
        "complete_seeds": seeds,
        "seed_impact_lift": by_seed_lift,
        "internal_win_rate": internal_rate,
        "checks": checks,
    }


def aggregate(root: Path) -> dict[str, Any]:
    jepa = discover_jepa(root)
    fixed = discover_fixed_k(root)
    direct_state = discover_direct_state(root)
    equal_direct = discover_equal_direct(root)
    equal_direct_fixed = discover_equal_direct_fixed_k(root)
    metric_summary = {
        metric: describe(record["metrics"][metric] for record in jepa)
        for metric in PRIMARY_METRICS
    }
    fixed_summary = {
        strategy: {
            metric: describe(
                record["strategies"][strategy][metric] for record in fixed
            )
            for metric in FIXED_K_METRICS
        }
        for strategy in ("impact_only", "impact_then_confidence", "joint_75_25")
    }
    cross_horizon_fixed_summary = {
        strategy: {
            metric: describe(
                record["cross_horizon_strategies"][strategy][metric]
                for record in fixed
            )
            for metric in FIXED_K_METRICS
        }
        for strategy in ("impact_only", "impact_then_confidence", "joint_75_25")
    }
    equal_direct_fixed_summary = {
        mode: {
            strategy: {
                metric: describe(
                    record["strategies"][strategy][metric]
                    for record in equal_direct_fixed
                    if record["mode"] == mode
                )
                for metric in FIXED_K_METRICS
            }
            for strategy in (
                "impact_only",
                "impact_then_confidence",
                "joint_75_25",
            )
        }
        for mode in ("graph", "nograph")
    }
    equal_direct_cross_horizon_summary = {
        mode: {
            strategy: {
                metric: describe(
                    record["cross_horizon_strategies"][strategy][metric]
                    for record in equal_direct_fixed
                    if record["mode"] == mode
                )
                for metric in FIXED_K_METRICS
            }
            for strategy in (
                "impact_only",
                "impact_then_confidence",
                "joint_75_25",
            )
        }
        for mode in ("graph", "nograph")
    }
    return {
        "status": "complete",
        "scope": "read_only_impact_validation_aggregate",
        "jepa_records": jepa,
        "fixed_k_records": fixed,
        "frozen_direct_records": direct_state,
        "equal_objective_direct_records": equal_direct,
        "equal_objective_direct_fixed_k_records": equal_direct_fixed,
        "jepa_metric_summary": metric_summary,
        "fixed_k_summary": fixed_summary,
        "cross_horizon_fixed_k_summary": cross_horizon_fixed_summary,
        "equal_objective_direct_fixed_k_summary": equal_direct_fixed_summary,
        "equal_objective_direct_cross_horizon_summary": (
            equal_direct_cross_horizon_summary
        ),
        "cross_horizon_selection_contracts": {
            "jepa": selection_contract_summary(fixed),
            "equal_objective_direct": selection_contract_summary(
                equal_direct_fixed
            ),
        },
        "gate": build_gate(jepa, fixed),
        "live_orders_allowed": False,
    }


def fmt(value: Any, digits: int = 4) -> str:
    return "pending" if value is None or not math.isfinite(float(value)) else f"{float(value):.{digits}f}"


def markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# Impact validation aggregate",
        "",
        f"Gate status: **{payload['gate']['status']}**",
        f"Complete seeds: `{payload['gate']['complete_seeds']}`",
        "",
        "## Fractional top-10%",
        "",
        "| Metric | Mean | Std | Min | Max |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for metric, values in payload["jepa_metric_summary"].items():
        lines.append(
            f"| {metric} | {fmt(values['mean'])} | {fmt(values['std'])} | "
            f"{fmt(values['min'])} | {fmt(values['max'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-horizon K=3",
            "",
            "| Strategy | Impact precision | Direction | Captured direction | Joint correct | Magnitude lift |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy, metrics in payload["fixed_k_summary"].items():
        lines.append(
            f"| {strategy} | {fmt(metrics['impact_precision_at_k']['mean'])} | "
            f"{fmt(metrics['direction_accuracy_at_k']['mean'])} | "
            f"{fmt(metrics['captured_direction_accuracy_at_k']['mean'])} | "
            f"{fmt(metrics['joint_correct_precision_at_k']['mean'])} | "
            f"{fmt(metrics['magnitude_lift_at_k']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Per-horizon magnitude-weighted diagnostics",
            "",
            "| Strategy | Tail mass recall | Impact-weighted direction | Signed tail capture |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for strategy, metrics in payload["fixed_k_summary"].items():
        lines.append(
            f"| {strategy} | "
            f"{fmt(metrics['realized_tail_mass_recall_at_k']['mean'])} | "
            f"{fmt(metrics['captured_impact_weighted_direction_accuracy_at_k']['mean'])} | "
            f"{fmt(metrics['signed_realized_tail_mass_capture_at_k']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Cross-horizon daily K=3",
            "",
            "One three-name candidate set is selected per date after aggregating all horizons.",
            "",
            "| Strategy | Impact precision | Captured direction | Tail mass recall | Impact-weighted direction | Signed tail capture |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for strategy, metrics in payload["cross_horizon_fixed_k_summary"].items():
        lines.append(
            f"| {strategy} | {fmt(metrics['impact_precision_at_k']['mean'])} | "
            f"{fmt(metrics['captured_direction_accuracy_at_k']['mean'])} | "
            f"{fmt(metrics['realized_tail_mass_recall_at_k']['mean'])} | "
            f"{fmt(metrics['captured_impact_weighted_direction_accuracy_at_k']['mean'])} | "
            f"{fmt(metrics['signed_realized_tail_mass_capture_at_k']['mean'])} |"
        )
    lines.extend(
        [
            "",
            "## Equal-objective direct K=3",
            "",
            "| Mode | Strategy | Impact precision | Captured direction | Tail mass recall | Impact-weighted direction | Signed tail capture |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, strategies in payload[
        "equal_objective_direct_fixed_k_summary"
    ].items():
        for strategy, metrics in strategies.items():
            lines.append(
                f"| {mode} | {strategy} | "
                f"{fmt(metrics['impact_precision_at_k']['mean'])} | "
                f"{fmt(metrics['captured_direction_accuracy_at_k']['mean'])} | "
                f"{fmt(metrics['realized_tail_mass_recall_at_k']['mean'])} | "
                f"{fmt(metrics['captured_impact_weighted_direction_accuracy_at_k']['mean'])} | "
                f"{fmt(metrics['signed_realized_tail_mass_capture_at_k']['mean'])} |"
            )
    lines.extend(
        [
            "",
            "## Equal-objective direct cross-horizon daily K=3",
            "",
            "| Mode | Strategy | Impact precision | Captured direction | Tail mass recall | Impact-weighted direction | Signed tail capture |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for mode, strategies in payload[
        "equal_objective_direct_cross_horizon_summary"
    ].items():
        for strategy, metrics in strategies.items():
            lines.append(
                f"| {mode} | {strategy} | "
                f"{fmt(metrics['impact_precision_at_k']['mean'])} | "
                f"{fmt(metrics['captured_direction_accuracy_at_k']['mean'])} | "
                f"{fmt(metrics['realized_tail_mass_recall_at_k']['mean'])} | "
                f"{fmt(metrics['captured_impact_weighted_direction_accuracy_at_k']['mean'])} | "
                f"{fmt(metrics['signed_realized_tail_mass_capture_at_k']['mean'])} |"
            )
    lines.extend(["", "## Gate checks", ""])
    for name, passed in payload["gate"]["checks"].items():
        lines.append(f"- `{name}`: `{passed}`")
    lines.extend(["", "## Cross-horizon candidate contract", ""])
    for name, contract in payload["cross_horizon_selection_contracts"].items():
        lines.append(
            f"- `{name}`: records={contract['available_records']}, "
            f"same_set={contract['all_same_candidates_across_horizons']}, "
            f"within_k={contract['all_within_requested_k']}, "
            f"max_selected={contract['maximum_selected_count']}"
        )
    lines.extend(
        [
            "",
            f"Frozen direct records: {len(payload['frozen_direct_records'])}",
            f"Equal-objective direct records: {len(payload['equal_objective_direct_records'])}",
            f"Equal-objective direct fixed-K records: {len(payload['equal_objective_direct_fixed_k_records'])}",
            "",
            "This report cannot authorize live orders.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate impact validation artifacts.")
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument(
        "--output-dir", default="reports/impact_validation_aggregate_20260714"
    )
    args = parser.parse_args()
    root = Path(args.root).resolve()
    output_dir = root / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = aggregate(root)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "summary.md").write_text(markdown(payload), encoding="utf-8")
    print(
        json.dumps(
            {
                "gate": payload["gate"]["status"],
                "jepa_records": len(payload["jepa_records"]),
                "fixed_k_records": len(payload["fixed_k_records"]),
                "equal_direct_records": len(payload["equal_objective_direct_records"]),
                "equal_direct_fixed_k_records": len(
                    payload["equal_objective_direct_fixed_k_records"]
                ),
            }
        )
    )


if __name__ == "__main__":
    main()
