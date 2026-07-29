from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stock_v2.systemic_transition import binary_ranking_metrics


CANDIDATE_NAMES = (
    "independent",
    "down_direction",
    "systemic_x_down",
    "independent_x_down",
    "independent_x_systemic_x_down",
)


def candidate_scores(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    required = {
        "probability_systemic_event",
        "probability_broad_selloff",
        "predicted_up_probability",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily frame is missing score columns: {sorted(missing)}")
    systemic = frame["probability_systemic_event"].to_numpy(dtype=np.float64)
    independent = frame["probability_broad_selloff"].to_numpy(dtype=np.float64)
    down = 1.0 - frame["predicted_up_probability"].to_numpy(dtype=np.float64)
    return {
        "independent": independent,
        "down_direction": down,
        "systemic_x_down": systemic * down,
        "independent_x_down": independent * down,
        "independent_x_systemic_x_down": independent * systemic * down,
    }


def _fixed_count_metrics(
    labels: np.ndarray,
    scores: np.ndarray,
    selected_count: int,
) -> dict[str, float | int]:
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    if labels.shape != scores.shape or labels.ndim != 1:
        raise ValueError("labels and scores must be aligned vectors")
    if not 0 < int(selected_count) <= len(labels):
        raise ValueError("selected_count must fit the score rows")
    ranking = binary_ranking_metrics(
        labels,
        scores,
        selection_rate=float(selected_count) / len(labels),
    )
    if int(ranking["selected_count"]) != int(selected_count):
        raise RuntimeError("fixed selection count changed during evaluation")
    return ranking


def evaluate_candidates(
    frame: pd.DataFrame,
    selected_counts: Mapping[int, int],
) -> dict[str, dict[str, Any]]:
    required = {"horizon", "actual_broad_selloff"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"daily frame is missing label columns: {sorted(missing)}")
    output: dict[str, dict[str, Any]] = {name: {} for name in CANDIDATE_NAMES}
    for horizon, rows in frame.groupby("horizon", sort=True):
        horizon = int(horizon)
        if horizon not in selected_counts:
            raise ValueError(f"missing fixed selection count for horizon {horizon}")
        labels = rows["actual_broad_selloff"].astype(bool).to_numpy()
        scores = candidate_scores(rows)
        for name in CANDIDATE_NAMES:
            output[name][str(horizon)] = _fixed_count_metrics(
                labels,
                scores[name],
                int(selected_counts[horizon]),
            )
    for name, horizons in output.items():
        auc = np.asarray(
            [float(row["roc_auc"]) for row in horizons.values()], dtype=np.float64
        )
        recall = np.asarray(
            [float(row["recall_at_selection_rate"]) for row in horizons.values()],
            dtype=np.float64,
        )
        average_precision = np.asarray(
            [float(row["average_precision"]) for row in horizons.values()],
            dtype=np.float64,
        )
        horizons["aggregate"] = {
            "mean_auc": float(np.nanmean(auc)),
            "minimum_auc": float(np.nanmin(auc)),
            "mean_average_precision": float(np.nanmean(average_precision)),
            "mean_recall": float(np.nanmean(recall)),
            "selection_score": float(
                0.60 * np.nanmean(auc)
                + 0.20 * np.nanmin(auc)
                + 0.20 * np.nanmean(recall)
            ),
        }
    return output


def _selected_counts(summary: Mapping[str, Any], split: str) -> dict[int, int]:
    horizons = summary["metrics"][split]["horizons"]
    return {
        int(horizon): int(values["subtypes"]["broad_selloff"]["selected_count"])
        for horizon, values in horizons.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Diagnose compositional broad-selloff scores without retraining."
    )
    parser.add_argument("--variant-dir", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    variant_dir = Path(args.variant_dir)
    summary = json.loads((variant_dir / "summary.json").read_text(encoding="utf-8"))
    if summary.get("live_orders_allowed") is not False:
        raise ValueError("input variant must prohibit live orders")
    validation = pd.read_csv(variant_dir / "daily_validation.csv")
    test = pd.read_csv(variant_dir / "daily_test.csv")
    validation_metrics = evaluate_candidates(
        validation, _selected_counts(summary, "validation")
    )
    selected = max(
        CANDIDATE_NAMES,
        key=lambda name: float(validation_metrics[name]["aggregate"]["selection_score"]),
    )
    test_metrics = evaluate_candidates(test, _selected_counts(summary, "test"))
    payload = {
        "schema_version": 1,
        "status": "complete",
        "role": "retrospective_composed_broad_selloff_diagnostic",
        "variant": summary.get("variant"),
        "candidate_formulas": {
            "independent": "p(broad_selloff)",
            "down_direction": "1-p(up)",
            "systemic_x_down": "p(systemic_event)*(1-p(up))",
            "independent_x_down": "p(broad_selloff)*(1-p(up))",
            "independent_x_systemic_x_down": (
                "p(broad_selloff)*p(systemic_event)*(1-p(up))"
            ),
        },
        "selection_rule": (
            "validation only: 0.60*mean_auc + 0.20*minimum_auc + "
            "0.20*mean_recall at the frozen per-horizon selection count"
        ),
        "validation_selected_candidate": selected,
        "validation": validation_metrics,
        "test": test_metrics,
        "selected_test_metrics": test_metrics[selected],
        "test_used_for_formula_selection": False,
        "retrospective_idea_development": True,
        "eligible_as_unbiased_promotion_evidence": False,
        "live_orders_allowed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "selected": selected,
                "validation": validation_metrics[selected]["aggregate"],
                "test": test_metrics[selected]["aggregate"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
