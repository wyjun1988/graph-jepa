from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any


HORIZONS = (1, 2, 3, 5, 10)
HORIZON_WEIGHTS = {1: 2.0, 2: 2.0, 3: 1.0, 5: 1.0, 10: 1.0}


def parse_candidate(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("candidates must use LABEL=SUMMARY_JSON")
    return label.strip(), Path(raw_path.strip())


def finite_float(value: Any, field: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"{field} must be finite")
    return parsed


def summarize_candidate(label: str, path: Path) -> dict[str, Any]:
    summary = json.loads(path.read_text(encoding="utf-8"))
    by_horizon = summary.get("future_rollout_by_horizon") or {}
    return_significance = (
        summary.get("realized_entry_path_correlation_significance")
        or summary.get("horizon_return_state_correlation_significance")
        or summary.get("matched_path_return_correlation_significance")
        or summary.get("return_1d_correlation_significance")
        or {}
    )
    skills: dict[str, float] = {}
    delta_correlations: dict[str, float] = {}
    return_ic: dict[str, dict[str, float]] = {}
    weighted_skill = 0.0
    total_weight = 0.0
    for horizon in HORIZONS:
        key = str(horizon)
        if key not in by_horizon or key not in return_significance:
            raise ValueError(f"{path} is missing horizon {horizon}")
        horizon_row = by_horizon[key]
        skill = finite_float(
            horizon_row["pooled_mse_skill_vs_persistence"],
            f"{label}.h{horizon}.pooled_skill",
        )
        delta_corr = finite_float(
            horizon_row["delta_corr"]["mean"],
            f"{label}.h{horizon}.delta_corr",
        )
        significance = return_significance[key]
        skills[key] = skill
        delta_correlations[key] = delta_corr
        return_ic[key] = {
            "mean": finite_float(
                significance["mean_target_corr"],
                f"{label}.h{horizon}.return_ic",
            ),
            "newey_west_t": finite_float(
                significance["newey_west_t_stat"],
                f"{label}.h{horizon}.return_nw_t",
            ),
        }
        weight = HORIZON_WEIGHTS[horizon]
        weighted_skill += weight * skill
        total_weight += weight

    current_skill = finite_float(
        summary["current_imputation"]["all"]["mse_skill_vs_zero"]["mean"],
        f"{label}.current_skill",
    )
    return {
        "label": label,
        "summary_path": str(path),
        "eval_start": summary.get("eval_start"),
        "eval_end": summary.get("eval_end"),
        "eval_steps": int(summary.get("eval_steps", 0)),
        "evaluation_seed": summary.get("evaluation_seed"),
        "current_imputation_skill": current_skill,
        "pooled_skill_by_horizon": skills,
        "delta_corr_by_horizon": delta_correlations,
        "return_1d_ic_by_horizon": return_ic,
        "fold1_selection_score": weighted_skill / total_weight,
    }


def validate_evaluation_contract(rows: list[dict[str, Any]], fold_name: str) -> None:
    windows = {
        (row.get("eval_start"), row.get("eval_end"), int(row.get("eval_steps", 0)))
        for row in rows
    }
    if len(windows) != 1:
        raise ValueError(
            f"{fold_name} checkpoint candidates must use one identical evaluation window; "
            f"found={sorted(windows, key=str)}"
        )
    eval_start, eval_end, eval_steps = next(iter(windows))
    if not eval_start or not eval_end or eval_steps <= 0:
        raise ValueError(f"{fold_name} has an invalid evaluation window")
    seeds = {row.get("evaluation_seed") for row in rows}
    if len(seeds) != 1 or None in seeds:
        raise ValueError(
            f"{fold_name} checkpoint candidates must use one identical evaluation seed; "
            f"found={sorted(seeds, key=str)}"
        )


def summarize(
    fold1_candidates: list[tuple[str, Path]],
    fold2_candidates: list[tuple[str, Path]],
) -> dict[str, Any]:
    if len(fold1_candidates) < 2:
        raise ValueError("at least two Fold 1 checkpoint candidates are required")
    fold1 = [summarize_candidate(label, path) for label, path in fold1_candidates]
    validate_evaluation_contract(fold1, "Fold 1")
    labels = [row["label"] for row in fold1]
    if len(labels) != len(set(labels)):
        raise ValueError("Fold 1 candidate labels must be unique")

    # The first fold is the only checkpoint-selection set. List order breaks an
    # exact tie, which keeps selection deterministic without consulting Fold 2.
    selected = max(fold1, key=lambda row: row["fold1_selection_score"])
    fold2 = [summarize_candidate(label, path) for label, path in fold2_candidates]
    validate_evaluation_contract(fold2, "Fold 2")
    fold2_by_label = {row["label"]: row for row in fold2}
    if selected["label"] not in fold2_by_label:
        raise ValueError(
            f"selected checkpoint {selected['label']} has no Fold 2 confirmation"
        )
    return {
        "status": "complete",
        "live_orders_allowed": False,
        "selection_policy": {
            "selection_fold": "fold1_only",
            "metric": "weighted_mean_pooled_mse_skill_vs_persistence",
            "horizon_weights": {str(key): value for key, value in HORIZON_WEIGHTS.items()},
            "fold2_used_for_selection": False,
            "return_ic_used_for_selection": False,
        },
        "selected_label": selected["label"],
        "fold1_candidates": fold1,
        "fold2_candidates": fold2,
        "selected_fold2_confirmation": fold2_by_label[selected["label"]],
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Checkpoint Epoch Selection",
        "",
        "Selection uses Fold 1 node-state skill only. Fold 2 and return IC are confirmation diagnostics.",
        "",
        "Live orders allowed: `false`",
        "",
        f"Selected checkpoint: `{result['selected_label']}`",
        "",
        "| Fold | Checkpoint | Selection score | Current skill | h1 | h2 | h3 | h5 | h10 |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold_name, rows in (
        ("Fold 1", result["fold1_candidates"]),
        ("Fold 2", result["fold2_candidates"]),
    ):
        for row in rows:
            skills = row["pooled_skill_by_horizon"]
            lines.append(
                "| {fold} | {label} | {score:+.4f} | {current:+.4f} | "
                "{h1:+.4f} | {h2:+.4f} | {h3:+.4f} | {h5:+.4f} | {h10:+.4f} |".format(
                    fold=fold_name,
                    label=row["label"],
                    score=row["fold1_selection_score"],
                    current=row["current_imputation_skill"],
                    h1=skills["1"],
                    h2=skills["2"],
                    h3=skills["3"],
                    h5=skills["5"],
                    h10=skills["10"],
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a training epoch on Fold 1 and confirm it on Fold 2."
    )
    parser.add_argument("--fold1", action="append", type=parse_candidate, required=True)
    parser.add_argument("--fold2", action="append", type=parse_candidate, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    result = summarize(args.fold1, args.fold2)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "selection.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(json.dumps({"selected_label": result["selected_label"]}, sort_keys=True))


if __name__ == "__main__":
    main()
