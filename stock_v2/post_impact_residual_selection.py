from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd


RESIDUAL_SELECTION_METRICS = ("pearson", "skill_vs_zero_mse")


def normalize_cell(record: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(record["horizon"]),
        str(record["bucket"]),
        str(record["subset"]),
    )


def paired_daily_delta(
    candidate_rows: Sequence[Mapping[str, Any]],
    baseline_rows: Sequence[Mapping[str, Any]],
    metric: str,
) -> pd.DataFrame:
    if metric not in RESIDUAL_SELECTION_METRICS:
        raise ValueError(f"unsupported residual selection metric: {metric}")
    candidate = pd.DataFrame(candidate_rows)
    baseline = pd.DataFrame(baseline_rows)
    required = {"date", "count", metric}
    if not required.issubset(candidate.columns) or not required.issubset(
        baseline.columns
    ):
        raise ValueError("residual selection daily rows are missing required fields")
    candidate["date"] = pd.to_datetime(candidate["date"])
    baseline["date"] = pd.to_datetime(baseline["date"])
    if (
        candidate.empty
        or baseline.empty
        or candidate["date"].duplicated().any()
        or baseline["date"].duplicated().any()
    ):
        raise ValueError("residual selection daily dates are empty or duplicated")
    paired = candidate[["date", "count", metric]].merge(
        baseline[["date", "count", metric]],
        on="date",
        how="inner",
        validate="one_to_one",
        suffixes=("_candidate", "_baseline"),
    )
    if len(paired) != len(candidate) or len(paired) != len(baseline):
        raise ValueError("residual selection variants do not share exact daily dates")
    if not np.array_equal(
        paired["count_candidate"].to_numpy(),
        paired["count_baseline"].to_numpy(),
    ):
        raise ValueError("residual selection variants do not share sample counts")
    values = paired[[f"{metric}_candidate", f"{metric}_baseline"]].to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("residual selection daily metrics must be finite")
    paired["delta"] = values[:, 0] - values[:, 1]
    return paired[["date", "count_candidate", "delta"]].rename(
        columns={"count_candidate": "count"}
    )


def aggregate_variant_cells(
    daily_rows: Mapping[str, Any],
    candidate: str,
    baseline: str,
    cells: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    normalized = [normalize_cell(record) for record in cells]
    if not normalized or len(set(normalized)) != len(normalized):
        raise ValueError("residual selection cells must be unique and non-empty")
    output: dict[str, Any] = {}
    for metric in RESIDUAL_SELECTION_METRICS:
        strata: dict[str, Any] = {}
        pooled: list[np.ndarray] = []
        for horizon, bucket, subset in normalized:
            key = f"{horizon}|{bucket}|{subset}"
            paired = paired_daily_delta(
                daily_rows[candidate][horizon][bucket][subset],
                daily_rows[baseline][horizon][bucket][subset],
                metric,
            )
            values = paired["delta"].to_numpy(dtype=np.float64)
            strata[key] = {
                "rows": int(len(values)),
                "mean_delta": float(values.mean()),
                "positive_day_fraction": float(np.mean(values > 0.0)),
            }
            pooled.append(values)
        values = np.concatenate(pooled)
        output[metric] = {
            "rows": int(len(values)),
            "strata": int(len(strata)),
            "mean_delta": float(values.mean()),
            "positive_day_fraction": float(np.mean(values > 0.0)),
            "positive_strata": int(
                sum(record["mean_delta"] > 0.0 for record in strata.values())
            ),
            "per_stratum": strata,
        }
    return output


def select_residual_candidate(
    daily_rows: Mapping[str, Any],
    candidates: Sequence[str],
    *,
    baseline: str,
    primary_cells: Sequence[Mapping[str, Any]],
    fast_exit_cells: Sequence[Mapping[str, Any]],
    gates: Mapping[str, Any],
) -> dict[str, Any]:
    names = [str(value) for value in candidates]
    if not names or len(set(names)) != len(names) or baseline in names:
        raise ValueError("residual selection candidates must be unique and non-empty")
    records: dict[str, Any] = {}
    eligible: list[str] = []
    for name in names:
        primary = aggregate_variant_cells(
            daily_rows, name, baseline, primary_cells
        )
        fast_exit = aggregate_variant_cells(
            daily_rows, name, baseline, fast_exit_cells
        )
        minimum_days = all(
            int(record["rows"]) >= int(gates["minimum_days_per_stratum"])
            for record in primary["pearson"]["per_stratum"].values()
        ) and all(
            int(record["rows"]) >= int(gates["minimum_days_per_stratum"])
            for record in fast_exit["skill_vs_zero_mse"]["per_stratum"].values()
        )
        checks = {
            "minimum_days": minimum_days,
            "primary_pearson": float(primary["pearson"]["mean_delta"])
            >= float(gates["minimum_primary_pearson_delta"]),
            "primary_positive_strata": int(primary["pearson"]["positive_strata"])
            >= int(gates["minimum_primary_positive_strata"]),
            "primary_skill": float(primary["skill_vs_zero_mse"]["mean_delta"])
            >= float(gates["minimum_primary_skill_delta"]),
            "fast_exit_skill": float(
                fast_exit["skill_vs_zero_mse"]["mean_delta"]
            )
            >= -float(gates["maximum_fast_exit_skill_degradation"]),
        }
        is_eligible = all(checks.values())
        if is_eligible:
            eligible.append(name)
        records[name] = {
            "primary": primary,
            "fast_exit": fast_exit,
            "checks": checks,
            "eligible": is_eligible,
            "selection_score": float(primary["pearson"]["mean_delta"]),
        }
    selected = (
        max(
            eligible,
            key=lambda name: (
                records[name]["selection_score"],
                records[name]["primary"]["skill_vs_zero_mse"]["mean_delta"],
                name,
            ),
        )
        if eligible
        else None
    )
    return {
        "baseline": baseline,
        "candidate_order": names,
        "candidates": records,
        "eligible_candidates": eligible,
        "selected_candidate": selected,
        "selection_passed": selected is not None,
    }
