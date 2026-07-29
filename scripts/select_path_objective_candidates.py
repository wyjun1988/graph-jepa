from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HORIZONS = (1, 2, 3, 5, 10)
WEIGHTS = {1: 2.0, 2: 2.0, 3: 1.0, 5: 1.0, 10: 1.0}


def parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("candidates must use LABEL=SUMMARY_JSON")
    return label.strip(), Path(raw_path.strip())


def finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def summarize_candidate(label: str, path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    node_horizons = summary.get("future_rollout_by_horizon") or {}
    path_significance = (
        summary.get("realized_entry_path_correlation_significance") or {}
    )
    liquid_path_significance = (
        summary.get("realized_entry_path_liquidity_significance", {}).get(
            "top300",
            {},
        )
    )
    group_horizons = summary.get("future_rollout_by_horizon_feature_group") or {}
    rollout_horizons = summary.get("rollout_dependency_by_horizon") or {}
    node_skill: dict[str, float] = {}
    delta_corr: dict[str, float] = {}
    rollout_skill: dict[str, float] = {}
    path_ic: dict[str, float] = {}
    path_ic_top300: dict[str, float] = {}
    path_nw_t: dict[str, float] = {}
    path_nw_t_top300: dict[str, float] = {}
    path_skill_vs_zero: dict[str, float] = {}
    weighted_node = 0.0
    weighted_path = 0.0
    total_weight = 0.0
    for horizon in HORIZONS:
        key = str(horizon)
        return_group = f"feature:return_{horizon}d"
        if (
            key not in node_horizons
            or key not in path_significance
            or key not in liquid_path_significance
            or key not in group_horizons
            or key not in rollout_horizons
            or return_group not in group_horizons[key]
        ):
            raise ValueError(f"{path} is missing matched horizon {horizon}")
        node = finite(
            node_horizons[key]["pooled_mse_skill_vs_persistence"],
            f"{label}.h{horizon}.node_skill",
        )
        delta = finite(
            node_horizons[key]["delta_corr"]["mean"],
            f"{label}.h{horizon}.delta_corr",
        )
        ic = finite(
            path_significance[key]["mean_target_corr"],
            f"{label}.h{horizon}.path_ic",
        )
        nw_t = finite(
            path_significance[key]["newey_west_t_stat"],
            f"{label}.h{horizon}.path_nw_t",
        )
        liquid_ic = finite(
            liquid_path_significance[key]["mean_target_corr"],
            f"{label}.h{horizon}.path_ic_top300",
        )
        liquid_nw_t = finite(
            liquid_path_significance[key]["newey_west_t_stat"],
            f"{label}.h{horizon}.path_nw_t_top300",
        )
        return_skill = finite(
            group_horizons[key][return_group]["pooled_mse_skill_vs_zero"],
            f"{label}.h{horizon}.path_skill_vs_zero",
        )
        rollout = finite(
            rollout_horizons[key]["pooled_mse_skill_vs_no_rollout"],
            f"{label}.h{horizon}.rollout_skill",
        )
        node_skill[key] = node
        delta_corr[key] = delta
        rollout_skill[key] = rollout
        path_ic[key] = ic
        path_ic_top300[key] = liquid_ic
        path_nw_t[key] = nw_t
        path_nw_t_top300[key] = liquid_nw_t
        path_skill_vs_zero[key] = return_skill
        weight = WEIGHTS[horizon]
        weighted_node += weight * node
        weighted_path += weight * 0.5 * (ic + liquid_ic)
        total_weight += weight

    weighted_node /= total_weight
    weighted_path /= total_weight
    current_skill = finite(
        summary["current_imputation"]["all"]["mse_skill_vs_zero"]["mean"],
        f"{label}.current_skill",
    )
    node_eligible = (
        current_skill > 0.15
        and weighted_node > 0.05
        and all(value > 0.0 for value in node_skill.values())
        and all(value > 0.0 for value in delta_corr.values())
        and all(value > 0.0 for value in rollout_skill.values())
    )
    return {
        "label": label,
        "summary_path": str(path),
        "weighted_node_skill": weighted_node,
        "weighted_realized_entry_path_ic": weighted_path,
        "current_imputation_skill": current_skill,
        "node_eligible": node_eligible,
        "realized_entry_path_gate_passes_fold1": all(
            path_ic[str(horizon)] > 0.0
            and path_nw_t[str(horizon)] >= 1.96
            and path_ic_top300[str(horizon)] > 0.0
            and path_nw_t_top300[str(horizon)] >= 1.96
            for horizon in HORIZONS
        ),
        "node_skill": node_skill,
        "delta_corr": delta_corr,
        "rollout_skill_vs_no_rollout": rollout_skill,
        "realized_entry_path_ic": path_ic,
        "realized_entry_path_ic_top300": path_ic_top300,
        "realized_entry_path_newey_west_t": path_nw_t,
        "realized_entry_path_top300_newey_west_t": path_nw_t_top300,
        "horizon_return_state_skill_vs_zero": path_skill_vs_zero,
    }


def select(candidates: list[tuple[str, Path]]) -> dict[str, Any]:
    if len(candidates) < 2:
        raise ValueError("at least two objective candidates are required")
    rows = [summarize_candidate(label, path) for label, path in candidates]
    eligible = [row for row in rows if row["node_eligible"]]
    pool = eligible or rows
    selected = max(
        pool,
        key=lambda row: (
            row["weighted_realized_entry_path_ic"],
            row["weighted_node_skill"],
        ),
    )
    return {
        "status": "complete",
        "selection_fold": "fold1_only",
        "selection_metric": "weighted_mean_all_and_top300_entry_path_ic_with_node_and_rollout_floors",
        "horizon_weights": {str(key): value for key, value in WEIGHTS.items()},
        "node_skill_floor": 0.05,
        "current_imputation_skill_floor": 0.15,
        "selected_label": selected["label"],
        "used_node_eligible_pool": bool(eligible),
        "candidates": rows,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Path Objective Screen",
        "",
        "Selection uses Fold 1 only and requires positive state skill at every horizon when possible.",
        "",
        f"Selected: `{result['selected_label']}`",
        "",
        "| Candidate | Node eligible | Current | Node skill | All/top300 path IC | Fold 1 path gate |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["candidates"]:
        lines.append(
            "| {label} | {eligible} | {current:+.4f} | {node:+.4f} | {path:+.4f} | {gate} |".format(
                label=row["label"],
                eligible=str(row["node_eligible"]).lower(),
                current=row["current_imputation_skill"],
                node=row["weighted_node_skill"],
                path=row["weighted_realized_entry_path_ic"],
                gate=str(row["realized_entry_path_gate_passes_fold1"]).lower(),
            )
        )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a horizon-matched path objective on Fold 1."
    )
    parser.add_argument("--candidate", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    result = select(args.candidate)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "selection.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(result["selected_label"])


if __name__ == "__main__":
    main()
