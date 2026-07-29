from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HORIZONS = (1, 2, 3, 5, 10)
WEIGHTS = {1: 2.0, 2: 2.0, 3: 1.0, 5: 1.0, 10: 1.0}


def parse_candidate(value: str) -> tuple[str, float, float, Path]:
    metadata, separator, raw_path = value.partition("=")
    parts = metadata.split(",")
    if not separator or len(parts) != 3 or not raw_path.strip():
        raise argparse.ArgumentTypeError(
            "candidates must use LABEL,TEMPORAL_SCALE,STOCK_EDGE_SCALE=SUMMARY_JSON"
        )
    label = parts[0].strip()
    if not label:
        raise argparse.ArgumentTypeError("candidate labels must be non-empty")
    temporal_scale = float(parts[1])
    stock_edge_scale = float(parts[2])
    if (
        not math.isfinite(temporal_scale)
        or temporal_scale < 0.0
        or not math.isfinite(stock_edge_scale)
        or stock_edge_scale < 0.0
    ):
        raise argparse.ArgumentTypeError("edge scales must be finite and non-negative")
    return label, temporal_scale, stock_edge_scale, Path(raw_path.strip())


def finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def summarize_candidate(
    label: str,
    temporal_scale: float,
    stock_edge_scale: float,
    path: Path,
) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    horizons = summary.get("future_rollout_by_horizon") or {}
    skills: dict[str, float] = {}
    delta_correlations: dict[str, float] = {}
    weighted_skill = 0.0
    total_weight = 0.0
    for horizon in HORIZONS:
        key = str(horizon)
        if key not in horizons:
            raise ValueError(f"{path} is missing horizon {horizon}")
        skill = finite(
            horizons[key]["pooled_mse_skill_vs_persistence"],
            f"{label}.h{horizon}.skill",
        )
        delta_corr = finite(
            horizons[key]["delta_corr"]["mean"],
            f"{label}.h{horizon}.delta_corr",
        )
        skills[key] = skill
        delta_correlations[key] = delta_corr
        weight = WEIGHTS[horizon]
        weighted_skill += weight * skill
        total_weight += weight
    current_skill = finite(
        summary["current_imputation"]["all"]["mse_skill_vs_zero"]["mean"],
        f"{label}.current_skill",
    )
    return {
        "label": label,
        "summary_path": str(path),
        "eval_start": summary.get("eval_start"),
        "eval_end": summary.get("eval_end"),
        "eval_steps": int(summary.get("eval_steps", 0)),
        "train_data_manifest_sha256": summary.get("train_data_manifest_sha256"),
        "temporal_graph_neighbor_scale": temporal_scale,
        "temporal_stock_edge_scale": stock_edge_scale,
        "weighted_future_node_skill": weighted_skill / total_weight,
        "current_imputation_skill": current_skill,
        "eligible": (
            current_skill > 0.15
            and all(value > 0.0 for value in skills.values())
            and all(value > 0.0 for value in delta_correlations.values())
        ),
        "node_skill": skills,
        "delta_corr": delta_correlations,
    }


def select(candidates: list[tuple[str, float, float, Path]]) -> dict[str, Any]:
    if len(candidates) < 2:
        raise ValueError("at least two temporal architecture candidates are required")
    rows = [summarize_candidate(*candidate) for candidate in candidates]
    evaluation_contracts = {
        (
            row["eval_start"],
            row["eval_end"],
            row["eval_steps"],
            row["train_data_manifest_sha256"],
        )
        for row in rows
    }
    if len(evaluation_contracts) != 1:
        raise ValueError(
            "temporal architecture candidates must use one identical evaluation "
            f"and data contract; found={sorted(evaluation_contracts, key=str)}"
        )
    eval_start, eval_end, eval_steps, manifest = next(iter(evaluation_contracts))
    if not eval_start or not eval_end or eval_steps <= 0 or not manifest:
        raise ValueError("temporal architecture evaluation contract is incomplete")
    eligible = [row for row in rows if row["eligible"]]
    pool = eligible or rows
    selected = max(
        pool,
        key=lambda row: (
            row["weighted_future_node_skill"],
            row["current_imputation_skill"],
        ),
    )
    return {
        "status": "complete",
        "selection_fold": "fold1_only",
        "selection_metric": "weighted_future_node_skill",
        "horizon_weights": {str(key): value for key, value in WEIGHTS.items()},
        "selected_label": selected["label"],
        "selected_temporal_graph_neighbor_scale": selected[
            "temporal_graph_neighbor_scale"
        ],
        "selected_temporal_stock_edge_scale": selected[
            "temporal_stock_edge_scale"
        ],
        "candidates": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select temporal graph routing on Fold 1 node-state skill."
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
    lines = [
        "# Temporal Architecture Selection",
        "",
        f"Selected: `{result['selected_label']}`",
        "",
        "| Candidate | Temporal scale | Stock edge scale | Current | Future | Eligible |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["candidates"]:
        lines.append(
            "| {label} | {temporal:.2f} | {stock:.2f} | {current:+.4f} | "
            "{future:+.4f} | {eligible} |".format(
                label=row["label"],
                temporal=row["temporal_graph_neighbor_scale"],
                stock=row["temporal_stock_edge_scale"],
                current=row["current_imputation_skill"],
                future=row["weighted_future_node_skill"],
                eligible=str(row["eligible"]).lower(),
            )
        )
    (output_dir / "selection.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(result["selected_label"])


if __name__ == "__main__":
    main()
