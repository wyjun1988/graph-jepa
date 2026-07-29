from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.compare_impact_fixed_k_paired import daily_paired_summary


FRACTIONAL_METRICS = (
    "precision",
    "captured_direction_accuracy",
    "tail_ic",
    "magnitude_lift",
    "realized_tail_mass_recall",
    "captured_impact_weighted_direction_accuracy",
    "signed_realized_tail_mass_capture",
)
FIXED_METRICS = (
    "impact_precision_at_k",
    "joint_correct_precision_at_k",
    "magnitude_lift_at_k",
    "realized_tail_mass_recall_at_k",
    "captured_impact_weighted_direction_accuracy_at_k",
    "signed_realized_tail_mass_capture_at_k",
)
STRATEGIES = ("impact_only", "impact_then_confidence", "joint_75_25")


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_fractional(path: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] != "top300" or row["variant"] != "impact_head":
                continue
            if abs(float(row["fraction"]) - 0.10) > 1e-9:
                continue
            key = (row["date"], int(row["horizon"]), "impact_head")
            if key in result:
                raise ValueError(f"duplicate fractional metric key in {path}: {key}")
            result[key] = {metric: _float(row[metric]) for metric in FRACTIONAL_METRICS}
    return result


def load_fixed(path: Path) -> dict[tuple[str, int, str], dict[str, float]]:
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] != "top300" or int(row["k"]) != 3:
                continue
            if row.get("selection_mode") != "cross_horizon":
                continue
            key = (row["date"], int(row["horizon"]), row["strategy"])
            if key in result:
                raise ValueError(f"duplicate fixed-K metric key in {path}: {key}")
            result[key] = {metric: _float(row[metric]) for metric in FIXED_METRICS}
    return result


def version_summary(
    v1_rows: dict[tuple[str, int, str], dict[str, float]],
    v2_rows: dict[tuple[str, int, str], dict[str, float]],
    *,
    strategy: str,
    metric: str,
) -> dict[str, Any]:
    paired = daily_paired_summary(
        v2_rows,
        v1_rows,
        strategy=strategy,
        metric=metric,
        nw_lag=10,
    )
    return {
        "paired_dates": paired["paired_dates"],
        "v1_mean": paired["direct_mean"],
        "v2_mean": paired["jepa_mean"],
        "v2_minus_v1": paired["delta"],
        "delta_ci95": paired["delta_ci95"],
        "mean_horizons_per_date": paired["mean_horizons_per_date"],
        "all_five_horizons_fraction": paired["all_five_horizons_fraction"],
    }


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "pending"
    return f"{number:.4f}" if math.isfinite(number) else "pending"


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Impact-direction v2 versus v1 paired comparison",
        "",
        "Deltas are v2 minus v1 on the same date and horizon. Horizons are combined",
        "per date with fixed 2:2:1:1:1 weights before a lag-10 Newey-West test.",
        "The scalar model-selection scores are intentionally not compared.",
        "",
    ]
    for fold, result in payload["folds"].items():
        lines.extend(
            [
                f"## {fold} fractional top-10%",
                "",
                "| Metric | v1 | v2 | Delta | NW t | Dates |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for metric, item in result["fractional"].items():
            lines.append(
                f"| {metric} | {_fmt(item['v1_mean'])} | {_fmt(item['v2_mean'])} | "
                f"{_fmt(item['v2_minus_v1'].get('mean'))} | "
                f"{_fmt(item['v2_minus_v1'].get('newey_west_t'))} | "
                f"{item['paired_dates']} |"
            )
        for strategy in STRATEGIES:
            lines.extend(
                [
                    "",
                    f"## {fold} cross-horizon K=3 / {strategy}",
                    "",
                    "| Metric | v1 | v2 | Delta | NW t | Dates |",
                    "| --- | ---: | ---: | ---: | ---: | ---: |",
                ]
            )
            for metric, item in result["cross_horizon_k3"][strategy].items():
                lines.append(
                    f"| {metric} | {_fmt(item['v1_mean'])} | {_fmt(item['v2_mean'])} | "
                    f"{_fmt(item['v2_minus_v1'].get('mean'))} | "
                    f"{_fmt(item['v2_minus_v1'].get('newey_west_t'))} | "
                    f"{item['paired_dates']} |"
                )
        lines.append("")
    lines.extend(
        [
            "This is a read-only model diagnostic and not a return estimate.",
            "Live orders remain disabled.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v1-root", type=Path, required=True)
    parser.add_argument("--v2-root", type=Path, required=True)
    parser.add_argument("--v1-fixed-root", type=Path, required=True)
    parser.add_argument("--v2-fixed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="fold1,fold2")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    folds = [value.strip() for value in args.folds.split(",") if value.strip()]
    output: dict[str, Any] = {}
    for fold in folds:
        v1_csv = args.v1_root / fold / "daily_impact_metrics.csv"
        v2_csv = args.v2_root / fold / "daily_impact_metrics.csv"
        v1_fixed_csv = args.v1_fixed_root / fold / "daily_fixed_k_metrics.csv"
        v2_fixed_csv = args.v2_fixed_root / fold / "daily_fixed_k_metrics.csv"
        if not all(
            path.is_file()
            for path in (v1_csv, v2_csv, v1_fixed_csv, v2_fixed_csv)
        ):
            continue
        v1_rows = load_fractional(v1_csv)
        v2_rows = load_fractional(v2_csv)
        v1_fixed_rows = load_fixed(v1_fixed_csv)
        v2_fixed_rows = load_fixed(v2_fixed_csv)
        output[fold] = {
            "fractional": {
                metric: version_summary(
                    v1_rows,
                    v2_rows,
                    strategy="impact_head",
                    metric=metric,
                )
                for metric in FRACTIONAL_METRICS
            },
            "cross_horizon_k3": {
                strategy: {
                    metric: version_summary(
                        v1_fixed_rows,
                        v2_fixed_rows,
                        strategy=strategy,
                        metric=metric,
                    )
                    for metric in FIXED_METRICS
                }
                for strategy in STRATEGIES
            },
        }

    payload = {
        "role": "read_only_paired_impact_direction_version_comparison",
        "delta_definition": "v2_minus_v1",
        "folds": output,
        "live_orders_allowed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        markdown_report(payload), encoding="utf-8"
    )
    print(json.dumps({"folds": len(output), "live_orders_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
