from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def compact(summary: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_skill_vs_zero": summary["current_imputation"]["all"][
            "mse_skill_vs_zero"
        ]["mean"],
        "future_skill_vs_persistence": {
            str(horizon): row["pooled_mse_skill_vs_persistence"]
            for horizon, row in summary["future_rollout_by_horizon"].items()
        },
        "future_delta_correlation": {
            str(horizon): row["delta_corr"]["mean"]
            for horizon, row in summary["future_rollout_by_horizon"].items()
        },
    }


def difference(candidate: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "current_skill_vs_zero": (
            candidate["current_skill_vs_zero"]
            - baseline["current_skill_vs_zero"]
        ),
        "future_skill_vs_persistence": {
            horizon: candidate["future_skill_vs_persistence"][horizon]
            - baseline["future_skill_vs_persistence"][horizon]
            for horizon in candidate["future_skill_vs_persistence"]
        },
        "future_delta_correlation": {
            horizon: candidate["future_delta_correlation"][horizon]
            - baseline["future_delta_correlation"][horizon]
            for horizon in candidate["future_delta_correlation"]
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare baseline and aligned JEPA under identical mask policies."
    )
    for name in (
        "baseline-mixed",
        "candidate-mixed",
        "baseline-operational",
        "candidate-operational",
    ):
        parser.add_argument(f"--{name}", action="append", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    groups = {
        "mixed": (
            args.baseline_mixed,
            args.candidate_mixed,
        ),
        "operational_mixed": (
            args.baseline_operational,
            args.candidate_operational,
        ),
    }
    result: dict[str, Any] = {
        "status": "complete",
        "approval_scope": "research_only",
        "live_orders_allowed": False,
        "masks": {},
    }
    for mask_name, (baseline_paths, candidate_paths) in groups.items():
        if len(baseline_paths) != len(candidate_paths):
            raise ValueError(f"fold count mismatch for {mask_name}")
        fold_rows = []
        for fold, (baseline_path, candidate_path) in enumerate(
            zip(baseline_paths, candidate_paths), start=1
        ):
            baseline = compact(load(baseline_path))
            candidate = compact(load(candidate_path))
            fold_rows.append(
                {
                    "fold": fold,
                    "baseline": baseline,
                    "candidate": candidate,
                    "candidate_minus_baseline": difference(candidate, baseline),
                }
            )
        result["masks"][mask_name] = fold_rows
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
