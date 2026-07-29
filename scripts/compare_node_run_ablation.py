from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.benchmark_direct_baselines import newey_west_mean


METRIC_COLUMNS = (
    "mse_skill_vs_persistence",
    "target_corr",
    "delta_corr",
    "target_sign_accuracy_abs_ge_0_10",
    "delta_sign_accuracy_abs_ge_0_10",
)
OPTIONAL_METRIC_COLUMNS = (
    "rollout_mse_skill_vs_no_rollout",
    "realized_entry_path_ic",
    "realized_entry_path_ic_top300",
)


def compare_metric(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
    horizon: int,
    column: str,
) -> dict[str, object]:
    left = baseline.loc[
        baseline["horizon"] == int(horizon), ["date", column]
    ].rename(columns={column: "baseline"})
    right = candidate.loc[
        candidate["horizon"] == int(horizon), ["date", column]
    ].rename(columns={column: "candidate"})
    paired = left.merge(right, on="date", how="inner").sort_values("date")
    if paired.empty:
        raise ValueError(f"no paired dates for horizon {horizon} and metric {column}")
    delta = (
        paired["candidate"].to_numpy(dtype=np.float64)
        - paired["baseline"].to_numpy(dtype=np.float64)
    )
    return {
        "baseline": newey_west_mean(
            paired["baseline"].to_numpy(dtype=np.float64), lag=int(horizon)
        ),
        "candidate": newey_west_mean(
            paired["candidate"].to_numpy(dtype=np.float64), lag=int(horizon)
        ),
        "delta_candidate_minus_baseline": newey_west_mean(delta, lag=int(horizon)),
    }


def compare_frames(
    baseline: pd.DataFrame,
    candidate: pd.DataFrame,
) -> dict[str, object]:
    required = {"date", "horizon", *METRIC_COLUMNS}
    for label, frame in (("baseline", baseline), ("candidate", candidate)):
        missing = sorted(required - set(frame.columns))
        if missing:
            raise ValueError(f"{label} is missing columns: {missing}")
    horizons = sorted(
        set(pd.to_numeric(baseline["horizon"], errors="raise").astype(int))
        & set(pd.to_numeric(candidate["horizon"], errors="raise").astype(int))
    )
    if not horizons:
        raise ValueError("baseline and candidate have no common horizons")
    metrics = METRIC_COLUMNS + tuple(
        column
        for column in OPTIONAL_METRIC_COLUMNS
        if column in baseline.columns and column in candidate.columns
    )
    return {
        "horizons": {
            str(horizon): {
                metric: compare_metric(baseline, candidate, horizon, metric)
                for metric in metrics
            }
            for horizon in horizons
        }
    }


def render_markdown(result: dict[str, object]) -> str:
    baseline_label = str(result["baseline_label"])
    candidate_label = str(result["candidate_label"])
    lines = [
        "# Paired Node-Run Ablation",
        "",
        f"Baseline: `{baseline_label}`",
        f"Candidate: `{candidate_label}`",
        "",
        "Delta is candidate minus baseline. Newey-West lag equals the horizon.",
        "",
        "| H | Base skill | Candidate | Delta | t | Base delta corr | Candidate | Delta | t |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for horizon, row in result["horizons"].items():
        skill = row["mse_skill_vs_persistence"]
        delta_corr = row["delta_corr"]
        lines.append(
            "| {h} | {sb:+.4f} | {sc:+.4f} | {sd:+.4f} | {st:+.2f} | "
            "{db:+.4f} | {dc:+.4f} | {dd:+.4f} | {dt:+.2f} |".format(
                h=horizon,
                sb=skill["baseline"]["mean"],
                sc=skill["candidate"]["mean"],
                sd=skill["delta_candidate_minus_baseline"]["mean"],
                st=skill["delta_candidate_minus_baseline"]["newey_west_t"],
                db=delta_corr["baseline"]["mean"],
                dc=delta_corr["candidate"]["mean"],
                dd=delta_corr["delta_candidate_minus_baseline"]["mean"],
                dt=delta_corr["delta_candidate_minus_baseline"]["newey_west_t"],
            )
        )
    first_row = next(iter(result["horizons"].values()))
    if "realized_entry_path_ic" in first_row:
        lines.extend(
            [
                "",
                "## Executable Path",
                "",
                "| H | Base all | Candidate | Delta | t | Base top300 | Candidate | Delta | t |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for horizon, row in result["horizons"].items():
            all_path = row["realized_entry_path_ic"]
            liquid_path = row["realized_entry_path_ic_top300"]
            lines.append(
                "| {h} | {ab:+.4f} | {ac:+.4f} | {ad:+.4f} | {at:+.2f} | "
                "{lb:+.4f} | {lc:+.4f} | {ld:+.4f} | {lt:+.2f} |".format(
                    h=horizon,
                    ab=all_path["baseline"]["mean"],
                    ac=all_path["candidate"]["mean"],
                    ad=all_path["delta_candidate_minus_baseline"]["mean"],
                    at=all_path["delta_candidate_minus_baseline"]["newey_west_t"],
                    lb=liquid_path["baseline"]["mean"],
                    lc=liquid_path["candidate"]["mean"],
                    ld=liquid_path["delta_candidate_minus_baseline"]["mean"],
                    lt=liquid_path["delta_candidate_minus_baseline"]["newey_west_t"],
                )
            )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare two node-rollout runs on paired dates."
    )
    parser.add_argument("--baseline-daily", required=True)
    parser.add_argument("--candidate-daily", required=True)
    parser.add_argument("--baseline-label", required=True)
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    result = compare_frames(
        pd.read_csv(args.baseline_daily),
        pd.read_csv(args.candidate_daily),
    )
    result.update(
        {
            "baseline_label": args.baseline_label,
            "candidate_label": args.candidate_label,
            "baseline_daily": args.baseline_daily,
            "candidate_daily": args.candidate_daily,
        }
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "comparison.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "comparison.md").write_text(
        render_markdown(result), encoding="utf-8"
    )
    print(f"wrote {output_dir / 'comparison.json'}")


if __name__ == "__main__":
    main()
