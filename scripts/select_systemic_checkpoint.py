from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stock_v2.market_transition import (
    MARKET_TRANSITION_FAMILIES,
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
)


HORIZONS = (1, 2, 3, 5, 10)
HORIZON_WEIGHTS = {1: 2.0, 2: 2.0, 3: 1.0, 5: 1.0, 10: 1.0}
SELECTION_WEIGHTS = {
    "node_skill": 0.15,
    "major_trajectory_auc": 0.20,
    "major_impact_mass_recall": 0.20,
    "peak_horizon_accuracy": 0.10,
    "systemic_event_auc": 0.10,
    "broad_selloff_auc": 0.10,
    "macro_family_auc": 0.15,
}


def parse_labeled_path(value: str) -> tuple[str, Path]:
    label, separator, raw_path = value.partition("=")
    if not separator or not label.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("inputs must use LABEL=SUMMARY_JSON")
    return label.strip(), Path(raw_path.strip())


def finite(value: Any, field: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field} must be finite")
    return result


def weighted(values: Mapping[str, float]) -> float:
    numerator = 0.0
    denominator = 0.0
    for horizon in HORIZONS:
        value = finite(values[str(horizon)], f"h{horizon}")
        weight = HORIZON_WEIGHTS[horizon]
        numerator += weight * value
        denominator += weight
    return numerator / denominator


def load_labeled(values: Sequence[tuple[str, Path]], role: str) -> dict[str, Path]:
    result = dict(values)
    if len(result) != len(values):
        raise ValueError(f"{role} labels must be unique")
    return result


def load_direct_node_skills(path: Path, role: str) -> dict[str, float]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("live_orders_allowed") is not False:
        raise ValueError(f"{role} direct baseline is not explicitly research-only")
    try:
        return {
            str(horizon): finite(
                payload["horizons"][str(horizon)]["metrics"][
                    "skill_vs_persistence"
                ],
                f"{role}.direct.h{horizon}",
            )
            for horizon in HORIZONS
        }
    except KeyError as exc:
        raise ValueError(f"{role} direct baseline is missing {exc}") from exc


def apply_direct_node_challenge(
    rows: Sequence[dict[str, Any]],
    direct_skills: Mapping[str, float],
) -> None:
    for row in rows:
        margins = {
            str(horizon): row["node_skill_by_horizon"][str(horizon)]
            - direct_skills[str(horizon)]
            for horizon in HORIZONS
        }
        passes = all(value > 0.0 for value in margins.values())
        row["direct_node_skill_by_horizon"] = dict(direct_skills)
        row["node_skill_margin_vs_direct_by_horizon"] = margins
        row["minimum_node_skill_margin_vs_direct"] = min(margins.values())
        row["passes_direct_node_challenge"] = passes
        row["eligible"] = bool(row["eligible"] and passes)


def summarize_candidate(
    label: str,
    node_path: Path,
    transition_path: Path,
) -> dict[str, Any]:
    node = json.loads(node_path.read_text(encoding="utf-8"))
    transition = json.loads(transition_path.read_text(encoding="utf-8"))
    for role, payload in (("node", node), ("transition", transition)):
        if payload.get("live_orders_allowed") is not False:
            raise ValueError(f"{label} {role} summary is not explicitly research-only")
    if transition.get("target_version") != MARKET_TRANSITION_TARGET_VERSION:
        raise ValueError(f"{label} transition target version differs")
    if (
        transition.get("impact_metric_version")
        != MARKET_TRANSITION_IMPACT_METRIC_VERSION
    ):
        raise ValueError(f"{label} transition impact metric version differs")

    node_skills = {
        str(horizon): finite(
            node["future_rollout_by_horizon"][str(horizon)][
                "pooled_mse_skill_vs_persistence"
            ],
            f"{label}.node.h{horizon}",
        )
        for horizon in HORIZONS
    }
    current_skill = finite(
        node["current_imputation"]["all"]["mse_skill_vs_zero"]["mean"],
        f"{label}.current_skill",
    )
    node_skill = weighted(node_skills)

    horizon_metrics = transition["horizon_metrics"]
    systemic_auc = weighted(
        {
            str(horizon): finite(
                horizon_metrics[str(horizon)]["systemic_event"]["roc_auc"],
                f"{label}.systemic.h{horizon}",
            )
            for horizon in HORIZONS
        }
    )
    broad_selloff_auc = weighted(
        {
            str(horizon): finite(
                horizon_metrics[str(horizon)]["broad_selloff"]["roc_auc"],
                f"{label}.selloff.h{horizon}",
            )
            for horizon in HORIZONS
        }
    )
    family_auc = {
        family: weighted(
            {
                str(horizon): finite(
                    horizon_metrics[str(horizon)]["family"][family]["event"][
                        "roc_auc"
                    ],
                    f"{label}.{family}.h{horizon}",
                )
                for horizon in HORIZONS
            }
        )
        for family in MARKET_TRANSITION_FAMILIES
    }
    macro_family_auc = sum(family_auc.values()) / len(family_auc)
    major = transition["major_path"]
    components = {
        "node_skill": node_skill,
        "major_trajectory_auc": finite(
            major["roc_auc"], f"{label}.major_trajectory_auc"
        ),
        "major_impact_mass_recall": finite(
            major["systemic_impact_mass_recall_at_selection_rate"],
            f"{label}.major_impact_mass_recall",
        ),
        "peak_horizon_accuracy": finite(
            major["peak_horizon_accuracy_on_major_events"],
            f"{label}.peak_horizon_accuracy",
        ),
        "systemic_event_auc": systemic_auc,
        "broad_selloff_auc": broad_selloff_auc,
        "macro_family_auc": macro_family_auc,
    }
    eligible = current_skill > 0.0 and all(value > 0.0 for value in node_skills.values())
    score = sum(SELECTION_WEIGHTS[name] * components[name] for name in components)
    return {
        "label": label,
        "node_summary_path": str(node_path),
        "transition_summary_path": str(transition_path),
        "eval_start": node.get("eval_start"),
        "eval_end": node.get("eval_end"),
        "eval_steps": int(node.get("eval_steps", 0)),
        "evaluation_seed": node.get("evaluation_seed"),
        "transition_test_steps": int(transition.get("test_steps", 0)),
        "current_imputation_skill": current_skill,
        "node_skill_by_horizon": node_skills,
        "family_auc": family_auc,
        "components": components,
        "eligible": eligible,
        "selection_score": score,
    }


def validate_fold(rows: Sequence[Mapping[str, Any]], fold: str) -> None:
    windows = {
        (row["eval_start"], row["eval_end"], row["eval_steps"])
        for row in rows
    }
    if len(windows) != 1:
        raise ValueError(f"{fold} candidates use different node evaluation windows")
    seeds = {row["evaluation_seed"] for row in rows}
    if len(seeds) != 1 or None in seeds:
        raise ValueError(f"{fold} candidates use different evaluation seeds")
    for row in rows:
        if row["eval_steps"] <= 0 or row["transition_test_steps"] != row["eval_steps"]:
            raise ValueError(f"{fold} candidate row counts do not align: {row['label']}")


def select(
    fold1_nodes: Sequence[tuple[str, Path]],
    fold1_transitions: Sequence[tuple[str, Path]],
    fold2_nodes: Sequence[tuple[str, Path]],
    fold2_transitions: Sequence[tuple[str, Path]],
    fold1_direct_summary: Path | None = None,
    fold2_direct_summary: Path | None = None,
) -> dict[str, Any]:
    if (fold1_direct_summary is None) != (fold2_direct_summary is None):
        raise ValueError("Fold 1 and Fold 2 direct summaries must be supplied together")
    paths = {
        "fold1_node": load_labeled(fold1_nodes, "Fold 1 node"),
        "fold1_transition": load_labeled(
            fold1_transitions, "Fold 1 transition"
        ),
        "fold2_node": load_labeled(fold2_nodes, "Fold 2 node"),
        "fold2_transition": load_labeled(
            fold2_transitions, "Fold 2 transition"
        ),
    }
    label_sets = {tuple(sorted(values)) for values in paths.values()}
    if len(label_sets) != 1:
        raise ValueError("Fold/checkpoint labels do not align")
    labels = [label for label, _path in fold1_nodes]
    if len(labels) < 2:
        raise ValueError("at least two checkpoint candidates are required")
    fold1 = [
        summarize_candidate(
            label,
            paths["fold1_node"][label],
            paths["fold1_transition"][label],
        )
        for label in labels
    ]
    fold2 = [
        summarize_candidate(
            label,
            paths["fold2_node"][label],
            paths["fold2_transition"][label],
        )
        for label in labels
    ]
    validate_fold(fold1, "Fold 1")
    validate_fold(fold2, "Fold 2")
    direct_challenge_enabled = fold1_direct_summary is not None
    if direct_challenge_enabled:
        assert fold1_direct_summary is not None
        assert fold2_direct_summary is not None
        apply_direct_node_challenge(
            fold1,
            load_direct_node_skills(fold1_direct_summary, "Fold 1"),
        )
        apply_direct_node_challenge(
            fold2,
            load_direct_node_skills(fold2_direct_summary, "Fold 2"),
        )
    eligible = [row for row in fold1 if row["eligible"]]
    if not eligible:
        requirement = (
            " and beats the direct MLP at every horizon"
            if direct_challenge_enabled
            else ""
        )
        raise ValueError(
            "no Fold 1 checkpoint has positive current and future node skill"
            + requirement
        )
    selected = max(eligible, key=lambda row: row["selection_score"])
    fold2_by_label = {row["label"]: row for row in fold2}
    return {
        "status": "complete",
        "role": (
            "fold1_only_deployment_checkpoint_selection"
            if direct_challenge_enabled
            else "fold1_only_systemic_checkpoint_selection"
        ),
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "selection_policy": {
            "selection_fold": "fold1_only",
            "fold2_used_for_selection": False,
            "eligibility": (
                "positive current skill, positive persistence skill, and strictly "
                "higher persistence skill than the direct MLP at h1,h2,h3,h5,h10"
                if direct_challenge_enabled
                else "positive current skill and positive persistence skill at h1,h2,h3,h5,h10"
            ),
            "direct_node_challenge_enabled": direct_challenge_enabled,
            "fold1_direct_summary": (
                str(fold1_direct_summary) if fold1_direct_summary else None
            ),
            "fold2_direct_summary": (
                str(fold2_direct_summary) if fold2_direct_summary else None
            ),
            "weights": SELECTION_WEIGHTS,
            "horizon_weights": {
                str(horizon): HORIZON_WEIGHTS[horizon] for horizon in HORIZONS
            },
        },
        "selected_label": selected["label"],
        "fold1_candidates": fold1,
        "fold2_candidates": fold2,
        "selected_fold2_confirmation": fold2_by_label[selected["label"]],
        "live_orders_allowed": False,
    }


def render_markdown(payload: Mapping[str, Any]) -> str:
    direct_enabled = payload["selection_policy"][
        "direct_node_challenge_enabled"
    ]
    lines = [
        "# Systemic Checkpoint Selection",
        "",
        "Selection uses Fold 1 only; Fold 2 is confirmation-only.",
        (
            "Eligibility requires beating the direct MLP at every rollout horizon."
            if direct_enabled
            else "Direct-MLP eligibility challenge was not supplied."
        ),
        "",
        "Live orders allowed: `false`",
        "",
        f"Selected checkpoint: `{payload['selected_label']}`",
        "",
        "| Fold | Epoch | Eligible | Direct pass | Min direct margin | Score | Node | Major AUC | Impact recall | Peak | Systemic AUC | Selloff AUC | Family AUC |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for fold, rows in (
        ("Fold 1", payload["fold1_candidates"]),
        ("Fold 2", payload["fold2_candidates"]),
    ):
        for row in rows:
            values = row["components"]
            direct_pass = row.get("passes_direct_node_challenge")
            direct_pass_text = (
                str(direct_pass).lower() if direct_pass is not None else "n/a"
            )
            min_margin = row.get("minimum_node_skill_margin_vs_direct")
            min_margin_text = f"{min_margin:.4f}" if min_margin is not None else "n/a"
            lines.append(
                f"| {fold} | {row['label']} | {str(row['eligible']).lower()} | "
                f"{direct_pass_text} | {min_margin_text} | "
                f"{row['selection_score']:.4f} | {values['node_skill']:.4f} | "
                f"{values['major_trajectory_auc']:.4f} | "
                f"{values['major_impact_mass_recall']:.4f} | "
                f"{values['peak_horizon_accuracy']:.4f} | "
                f"{values['systemic_event_auc']:.4f} | "
                f"{values['broad_selloff_auc']:.4f} | "
                f"{values['macro_family_auc']:.4f} |"
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a JEPA checkpoint on Fold 1 node and systemic metrics."
    )
    parser.add_argument("--fold1-node", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--fold1-transition", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--fold2-node", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--fold2-transition", action="append", type=parse_labeled_path, required=True)
    parser.add_argument("--fold1-direct-summary", type=Path, required=True)
    parser.add_argument("--fold2-direct-summary", type=Path, required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    payload = select(
        args.fold1_node,
        args.fold1_transition,
        args.fold2_node,
        args.fold2_transition,
        args.fold1_direct_summary,
        args.fold2_direct_summary,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "selection.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_dir / "selection.md").write_text(
        render_markdown(payload), encoding="utf-8"
    )
    print(json.dumps({"selected_label": payload["selected_label"], "live_orders_allowed": False}))


if __name__ == "__main__":
    main()
