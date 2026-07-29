from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stock_v2.downstream_probes import CONTINUOUS_TASKS, newey_west_mean


def paired_daily_difference(left: dict[str, Any], right: dict[str, Any], lag: int) -> dict[str, Any]:
    differences = [
        float(a) - float(b)
        for a, b in zip(left["daily_ic_values"], right["daily_ic_values"])
        if a is not None and b is not None
    ]
    return newey_west_mean(differences, lag=lag)


def paired_daily_values(left: dict[str, Any], right: dict[str, Any]) -> list[float]:
    return [
        float(a) - float(b)
        for a, b in zip(left["daily_ic_values"], right["daily_ic_values"])
        if a is not None and b is not None
    ]


def summarize(fold_paths: list[Path]) -> tuple[dict[str, Any], pd.DataFrame]:
    folds = [json.loads(path.read_text(encoding="utf-8")) for path in fold_paths]
    if any(fold.get("status") != "complete" for fold in folds):
        raise ValueError("every fold must be complete before representation promotion")
    rows: list[dict[str, Any]] = []
    path_positive = 0
    path_strong = 0
    path_vs_shuffled_positive = 0
    cross_task_positive = 0
    path_tests = 0
    task_tests = 0
    per_fold_path_wins: dict[str, int] = {}
    multitask_path_wins = 0
    pooled_path_vs_raw: dict[int, list[float]] = {}
    pooled_path_vs_shuffled: dict[int, list[float]] = {}
    for fold_index, fold in enumerate(folds, start=1):
        fold_name = f"fold{fold_index}"
        per_fold_path_wins[fold_name] = 0
        for raw_horizon, horizon_result in sorted(
            fold["results"].items(), key=lambda item: int(item[0])
        ):
            horizon = int(raw_horizon)
            multi = horizon_result["multi"]
            raw_metrics = multi["raw"]["metrics"]
            combined_metrics = multi["raw_latent"]["metrics"]
            shuffled_metrics = multi["raw_shuffled_latent"]["metrics"]
            for task in CONTINUOUS_TASKS:
                raw_task = raw_metrics["tasks"][task]
                combined_task = combined_metrics["tasks"][task]
                shuffled_task = shuffled_metrics["tasks"][task]
                raw_premium = paired_daily_difference(
                    combined_task, raw_task, lag=horizon
                )
                shuffled_premium = paired_daily_difference(
                    combined_task, shuffled_task, lag=horizon
                )
                rows.append(
                    {
                        "fold": fold_name,
                        "horizon": horizon,
                        "task": task,
                        "raw_ic": raw_task["daily_ic"]["mean"],
                        "raw_latent_ic": combined_task["daily_ic"]["mean"],
                        "shuffled_ic": shuffled_task["daily_ic"]["mean"],
                        "premium_vs_raw": raw_premium["mean"],
                        "premium_vs_raw_t": raw_premium["newey_west_t"],
                        "premium_vs_shuffled": shuffled_premium["mean"],
                        "premium_vs_shuffled_t": shuffled_premium["newey_west_t"],
                    }
                )
                task_tests += 1
                cross_task_positive += int(float(raw_premium["mean"]) > 0.0)
                if task == "path_return":
                    pooled_path_vs_raw.setdefault(horizon, []).extend(
                        paired_daily_values(combined_task, raw_task)
                    )
                    pooled_path_vs_shuffled.setdefault(horizon, []).extend(
                        paired_daily_values(combined_task, shuffled_task)
                    )
                    path_tests += 1
                    positive = float(raw_premium["mean"]) > 0.0
                    path_positive += int(positive)
                    per_fold_path_wins[fold_name] += int(positive)
                    path_strong += int(
                        positive and float(raw_premium["newey_west_t"]) >= 1.96
                    )
                    path_vs_shuffled_positive += int(
                        float(shuffled_premium["mean"]) > 0.0
                    )
            single_task = horizon_result["single"]["raw_latent"]["metrics"]["tasks"]["path_return"]
            multi_task = combined_metrics["tasks"]["path_return"]
            multitask_premium = paired_daily_difference(
                multi_task, single_task, lag=horizon
            )
            multitask_path_wins += int(float(multitask_premium["mean"]) > 0.0)

    path_required = max(1, int(np.ceil(path_tests * 0.70)))
    cross_task_required = max(1, int(np.ceil(task_tests * 0.60)))
    shuffled_required = path_required
    each_fold_required = max(1, int(np.ceil(path_tests / len(folds) * 0.40)))
    keep_shared = (
        path_positive >= path_required
        and path_vs_shuffled_positive >= shuffled_required
        and cross_task_positive >= cross_task_required
        and path_strong >= 2
        and all(value >= each_fold_required for value in per_fold_path_wins.values())
    )
    keep_auxiliary = (
        path_positive >= int(np.ceil(path_tests * 0.50))
        and cross_task_positive >= int(np.ceil(task_tests * 0.50))
    )
    decision = (
        "keep_as_shared_encoder"
        if keep_shared
        else "keep_as_auxiliary_feature"
        if keep_auxiliary
        else "no_representation_premium"
    )
    result = {
        "status": "complete",
        "approval_scope": "research_only",
        "live_orders_allowed": False,
        "folds": len(folds),
        "decision": decision,
        "criteria": {
            "path_positive_required": path_required,
            "path_vs_shuffled_positive_required": shuffled_required,
            "cross_task_positive_required": cross_task_required,
            "strong_path_premiums_required": 2,
            "per_fold_path_positive_required": each_fold_required,
        },
        "observed": {
            "path_tests": path_tests,
            "path_positive": path_positive,
            "path_strong": path_strong,
            "path_vs_shuffled_positive": path_vs_shuffled_positive,
            "task_tests": task_tests,
            "cross_task_positive": cross_task_positive,
            "per_fold_path_wins": per_fold_path_wins,
            "multitask_path_wins": multitask_path_wins,
            "pooled_path_premium_by_horizon": {
                str(horizon): {
                    "vs_raw": newey_west_mean(values, lag=horizon),
                    "vs_shuffled": newey_west_mean(
                        pooled_path_vs_shuffled[horizon], lag=horizon
                    ),
                }
                for horizon, values in sorted(pooled_path_vs_raw.items())
            },
        },
        "fold_summaries": [str(path) for path in fold_paths],
    }
    return result, pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize frozen encoder downstream probes.")
    parser.add_argument("--fold", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result, rows = summarize([Path(value) for value in args.fold])
    rows.to_csv(output_dir / "representation_premiums.csv", index=False)
    (output_dir / "decision.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# Frozen Encoder Downstream Decision",
        "",
        f"- Decision: `{result['decision']}`",
        f"- Path premium wins: {result['observed']['path_positive']}/{result['observed']['path_tests']}",
        f"- Strong path premium wins: {result['observed']['path_strong']}",
        f"- Wins versus shuffled latent: {result['observed']['path_vs_shuffled_positive']}/{result['observed']['path_tests']}",
        f"- Cross-task premium wins: {result['observed']['cross_task_positive']}/{result['observed']['task_tests']}",
        f"- Multi-task path wins over single-task: {result['observed']['multitask_path_wins']}/{result['observed']['path_tests']}",
        "- Approval scope: research only; live orders remain disabled.",
    ]
    (output_dir / "decision.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
