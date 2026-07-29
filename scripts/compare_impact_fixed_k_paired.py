from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
import sys
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from scripts.benchmark_direct_baselines import newey_west_mean


HORIZON_WEIGHTS = {1: 2.0, 2: 2.0, 3: 1.0, 5: 1.0, 10: 1.0}
METRICS = (
    "impact_precision_at_k",
    "joint_correct_precision_at_k",
    "magnitude_lift_at_k",
    "realized_tail_mass_recall_at_k",
    "captured_impact_weighted_direction_accuracy_at_k",
    "signed_realized_tail_mass_capture_at_k",
)
STRATEGIES = ("impact_only", "impact_then_confidence", "joint_75_25")
SELECTION_MODES = ("per_horizon", "cross_horizon")


def _float(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def load_metric_rows(
    path: Path,
    *,
    scope: str,
    count: int,
    selection_mode: str,
) -> dict[tuple[str, int, str], dict[str, float]]:
    rows: dict[tuple[str, int, str], dict[str, float]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["scope"] != scope:
                continue
            if int(row["k"]) != int(count):
                continue
            if row.get("selection_mode") != selection_mode:
                continue
            key = (row["date"], int(row["horizon"]), row["strategy"])
            if key in rows:
                raise ValueError(f"duplicate fixed-K metric key in {path}: {key}")
            rows[key] = {metric: _float(row[metric]) for metric in METRICS}
    return rows


def daily_paired_summary(
    jepa_rows: dict[tuple[str, int, str], dict[str, float]],
    direct_rows: dict[tuple[str, int, str], dict[str, float]],
    *,
    strategy: str,
    metric: str,
    nw_lag: int = 10,
) -> dict[str, Any]:
    grouped: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    for key in sorted(set(jepa_rows) & set(direct_rows)):
        date, horizon, row_strategy = key
        if row_strategy != strategy:
            continue
        jepa_value = float(jepa_rows[key][metric])
        direct_value = float(direct_rows[key][metric])
        if not (math.isfinite(jepa_value) and math.isfinite(direct_value)):
            continue
        grouped[date].append((int(horizon), jepa_value, direct_value))

    jepa_daily: list[float] = []
    direct_daily: list[float] = []
    horizon_counts: list[int] = []
    for date in sorted(grouped):
        values = grouped[date]
        weights = np.asarray(
            [HORIZON_WEIGHTS.get(horizon, 1.0) for horizon, _, _ in values],
            dtype=np.float64,
        )
        jepa = np.asarray([value for _, value, _ in values], dtype=np.float64)
        direct = np.asarray([value for _, _, value in values], dtype=np.float64)
        jepa_daily.append(float(np.average(jepa, weights=weights)))
        direct_daily.append(float(np.average(direct, weights=weights)))
        horizon_counts.append(len(values))

    deltas = np.asarray(jepa_daily, dtype=np.float64) - np.asarray(
        direct_daily, dtype=np.float64
    )
    delta_summary = newey_west_mean(deltas, lag=int(nw_lag))
    standard_error = float(delta_summary.get("newey_west_standard_error", float("nan")))
    delta_mean = float(delta_summary.get("mean", float("nan")))
    return {
        "paired_dates": len(deltas),
        "jepa_mean": float(np.mean(jepa_daily)) if jepa_daily else float("nan"),
        "direct_mean": float(np.mean(direct_daily)) if direct_daily else float("nan"),
        "delta": delta_summary,
        "delta_ci95": [
            delta_mean - 1.96 * standard_error,
            delta_mean + 1.96 * standard_error,
        ]
        if math.isfinite(delta_mean) and math.isfinite(standard_error)
        else [float("nan"), float("nan")],
        "mean_horizons_per_date": (
            float(np.mean(horizon_counts)) if horizon_counts else float("nan")
        ),
        "all_five_horizons_fraction": (
            float(np.mean(np.asarray(horizon_counts) == 5))
            if horizon_counts
            else float("nan")
        ),
    }


def compare_fold(
    jepa_csv: Path,
    direct_csv: Path,
    *,
    scope: str,
    count: int,
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for selection_mode in SELECTION_MODES:
        jepa_rows = load_metric_rows(
            jepa_csv, scope=scope, count=count, selection_mode=selection_mode
        )
        direct_rows = load_metric_rows(
            direct_csv, scope=scope, count=count, selection_mode=selection_mode
        )
        result[selection_mode] = {
            strategy: {
                metric: daily_paired_summary(
                    jepa_rows,
                    direct_rows,
                    strategy=strategy,
                    metric=metric,
                )
                for metric in METRICS
            }
            for strategy in STRATEGIES
        }
    return result


def _fmt(value: Any) -> str:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return "pending"
    return f"{number:.4f}" if math.isfinite(number) else "pending"


def markdown_report(payload: dict[str, Any]) -> str:
    lines = [
        "# Paired fixed-K impact comparison",
        "",
        "Each delta is JEPA minus the equal-objective direct challenger on the same",
        "date and horizon. Horizon rows are combined per date with fixed 2:2:1:1:1",
        "weights before a lag-10 Newey-West test.",
        "",
    ]
    for mode, folds in payload["comparisons"].items():
        for fold, selections in folds.items():
            for selection_mode in SELECTION_MODES:
                lines.extend(
                    [
                        f"## {mode} / {fold} / {selection_mode}",
                        "",
                        "| Strategy | Metric | JEPA | Direct | Delta | NW t | Dates |",
                        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
                    ]
                )
                for strategy in STRATEGIES:
                    for metric in METRICS:
                        item = selections[selection_mode][strategy][metric]
                        lines.append(
                            f"| {strategy} | {metric} | {_fmt(item['jepa_mean'])} | "
                            f"{_fmt(item['direct_mean'])} | "
                            f"{_fmt(item['delta'].get('mean'))} | "
                            f"{_fmt(item['delta'].get('newey_west_t'))} | "
                            f"{item['paired_dates']} |"
                        )
                lines.append("")
    lines.extend(
        [
            "This is a model-diagnostic comparison, not a return estimate.",
            "Live orders remain disabled.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jepa-root", type=Path, required=True)
    parser.add_argument("--direct-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--folds", default="fold1,fold2")
    parser.add_argument("--modes", default="graph,nograph")
    parser.add_argument("--scope", default="top300")
    parser.add_argument("--count", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    folds = [value.strip() for value in args.folds.split(",") if value.strip()]
    modes = [value.strip() for value in args.modes.split(",") if value.strip()]
    comparisons: dict[str, Any] = {}
    for mode in modes:
        mode_result: dict[str, Any] = {}
        for fold in folds:
            jepa_csv = args.jepa_root / fold / "daily_fixed_k_metrics.csv"
            direct_csv = args.direct_root / mode / fold / "daily_fixed_k_metrics.csv"
            if not jepa_csv.is_file() or not direct_csv.is_file():
                continue
            mode_result[fold] = compare_fold(
                jepa_csv,
                direct_csv,
                scope=args.scope,
                count=args.count,
            )
        if mode_result:
            comparisons[mode] = mode_result

    payload = {
        "role": "read_only_paired_impact_diagnostic",
        "scope": args.scope,
        "k": int(args.count),
        "horizon_weights": {str(key): value for key, value in HORIZON_WEIGHTS.items()},
        "comparisons": comparisons,
        "live_orders_allowed": False,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.output_dir / "summary.md").write_text(
        markdown_report(payload), encoding="utf-8"
    )
    print(json.dumps({"modes": len(comparisons), "live_orders_allowed": False}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
