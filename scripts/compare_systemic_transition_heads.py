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

from scripts.benchmark_direct_baselines import newey_west_mean
from scripts.benchmark_latent_trajectory_path_head import HORIZON_WEIGHTS
from stock_v2.systemic_transition import SYSTEMIC_TARGET_VERSION


KEYS = ["date", "horizon"]


def _as_bool(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    return values.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()


def paired_fold_metrics(
    jepa: pd.DataFrame,
    direct: pd.DataFrame,
) -> tuple[dict[str, Any], pd.DataFrame]:
    required = {
        *KEYS,
        "actual_systemic_energy",
        "predicted_systemic_energy",
        "actual_systemic_event",
        "probability_systemic_event",
        "actual_market_return",
        "predicted_market_return",
    }
    for name, frame in (("jepa", jepa), ("direct", direct)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{name} daily systemic rows are missing: {sorted(missing)}")
        if frame.duplicated(KEYS).any():
            raise ValueError(f"{name} daily systemic rows contain duplicate keys")
    merged = jepa.merge(direct, on=KEYS, suffixes=("_jepa", "_direct"), how="inner")
    if len(merged) != len(jepa) or len(merged) != len(direct):
        raise ValueError("JEPA and direct daily systemic keys do not align")
    actual_jepa = merged["actual_systemic_energy_jepa"].to_numpy(dtype=np.float64)
    actual_direct = merged["actual_systemic_energy_direct"].to_numpy(dtype=np.float64)
    if not np.allclose(actual_jepa, actual_direct, rtol=1e-7, atol=1e-9, equal_nan=True):
        raise ValueError("JEPA and direct actual systemic targets differ")
    actual = actual_jepa
    jepa_prediction = merged["predicted_systemic_energy_jepa"].to_numpy(dtype=np.float64)
    direct_prediction = merged["predicted_systemic_energy_direct"].to_numpy(dtype=np.float64)
    jepa_error = np.abs(np.log1p(np.maximum(jepa_prediction, 0.0)) - np.log1p(actual))
    direct_error = np.abs(np.log1p(np.maximum(direct_prediction, 0.0)) - np.log1p(actual))
    merged["primary_error_delta_direct_minus_jepa"] = direct_error - jepa_error

    label = _as_bool(merged["actual_systemic_event_jepa"])
    label_direct = _as_bool(merged["actual_systemic_event_direct"])
    if not np.array_equal(label, label_direct):
        raise ValueError("JEPA and direct systemic event labels differ")
    jepa_probability = merged["probability_systemic_event_jepa"].to_numpy(dtype=np.float64)
    direct_probability = merged["probability_systemic_event_direct"].to_numpy(dtype=np.float64)
    merged["brier_delta_direct_minus_jepa"] = (
        np.square(direct_probability - label.astype(np.float64))
        - np.square(jepa_probability - label.astype(np.float64))
    )
    actual_return = merged["actual_market_return_jepa"].to_numpy(dtype=np.float64)
    direct_actual_return = merged["actual_market_return_direct"].to_numpy(dtype=np.float64)
    if not np.allclose(
        actual_return, direct_actual_return, rtol=1e-7, atol=1e-9, equal_nan=True
    ):
        raise ValueError("JEPA and direct market-return targets differ")
    jepa_return = merged["predicted_market_return_jepa"].to_numpy(dtype=np.float64)
    direct_return = merged["predicted_market_return_direct"].to_numpy(dtype=np.float64)
    direction_valid = label & np.isfinite(actual_return) & np.isfinite(jepa_return) & np.isfinite(direct_return)
    direction_delta = np.full(len(merged), np.nan, dtype=np.float64)
    direction_delta[direction_valid] = (
        (np.sign(jepa_return[direction_valid]) == np.sign(actual_return[direction_valid])).astype(np.float64)
        - (np.sign(direct_return[direction_valid]) == np.sign(actual_return[direction_valid])).astype(np.float64)
    )
    merged["event_direction_correct_delta_jepa_minus_direct"] = direction_delta

    by_horizon = {}
    for horizon, selected in merged.groupby("horizon", sort=True):
        by_horizon[str(int(horizon))] = {
            "primary_error": newey_west_mean(
                selected["primary_error_delta_direct_minus_jepa"].to_numpy(),
                lag=int(horizon),
            ),
            "brier": newey_west_mean(
                selected["brier_delta_direct_minus_jepa"].to_numpy(),
                lag=int(horizon),
            ),
            "event_direction": newey_west_mean(
                selected["event_direction_correct_delta_jepa_minus_direct"].to_numpy(),
                lag=int(horizon),
            ),
        }
    daily = (
        merged.groupby("date", sort=True)[
            ["primary_error_delta_direct_minus_jepa", "brier_delta_direct_minus_jepa"]
        ]
        .mean()
        .reset_index()
    )
    event_daily = (
        merged.groupby("date", sort=True)["event_direction_correct_delta_jepa_minus_direct"]
        .mean()
        .reset_index()
    )
    daily = daily.merge(event_daily, on="date", how="left")
    return (
        {
            "rows": len(merged),
            "dates": int(merged["date"].nunique()),
            "by_horizon": by_horizon,
            "date_level": {
                "primary_error": newey_west_mean(
                    daily["primary_error_delta_direct_minus_jepa"].to_numpy(), lag=10
                ),
                "brier": newey_west_mean(
                    daily["brier_delta_direct_minus_jepa"].to_numpy(), lag=10
                ),
                "event_direction": newey_west_mean(
                    daily["event_direction_correct_delta_jepa_minus_direct"].to_numpy(), lag=10
                ),
            },
        },
        daily,
    )


def _weighted_horizon_metric(summary: Mapping[str, Any], path: tuple[str, ...]) -> float:
    total = 0.0
    weight_sum = 0.0
    horizons = summary["metrics"]["test"]["horizons"]
    for raw_horizon, row in horizons.items():
        value: Any = row
        for key in path:
            value = value[key]
        weight = float(HORIZON_WEIGHTS.get(int(raw_horizon), 1.0))
        total += weight * float(value)
        weight_sum += weight
    return float(total / weight_sum)


def absolute_gate(summary: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = summary["metrics"]["test"]["trajectory"]
    trajectory_auc = float(trajectory["roc_auc"])
    trajectory_ap_lift = float(trajectory["average_precision"]) / max(
        float(trajectory["event_rate"]), 1e-8
    )
    energy_correlation = _weighted_horizon_metric(
        summary, ("energy_head", "systemic_energy_correlation")
    )
    tail_mass = _weighted_horizon_metric(
        summary, ("energy_head", "tail_mass_recall_at_fit_event_rate")
    )
    direction = _weighted_horizon_metric(
        summary,
        ("energy_head", "event_impact_weighted_market_direction_accuracy"),
    )
    selloff_recall = _weighted_horizon_metric(
        summary, ("subtypes", "broad_selloff", "recall_at_selection_rate")
    )
    subtype_auc_values = [
        float(metrics["roc_auc"])
        for horizon in summary["metrics"]["test"]["horizons"].values()
        for metrics in horizon["subtypes"].values()
    ]
    checks = {
        "trajectory_auc_at_least_0_60": trajectory_auc >= 0.60,
        "trajectory_ap_lift_at_least_1_50": trajectory_ap_lift >= 1.50,
        "weighted_energy_correlation_at_least_0_15": energy_correlation >= 0.15,
        "weighted_tail_mass_recall_at_least_0_20": tail_mass >= 0.20,
        "weighted_event_direction_at_least_0_55": direction >= 0.55,
        "weighted_broad_selloff_recall_at_least_0_25": selloff_recall >= 0.25,
        "all_subtype_auc_at_least_0_52": min(subtype_auc_values) >= 0.52,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "values": {
            "trajectory_auc": trajectory_auc,
            "trajectory_ap_lift": trajectory_ap_lift,
            "weighted_energy_correlation": energy_correlation,
            "weighted_tail_mass_recall": tail_mass,
            "weighted_event_direction_accuracy": direction,
            "weighted_broad_selloff_recall": selloff_recall,
            "minimum_subtype_auc": min(subtype_auc_values),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare systemic JEPA and same-objective direct heads on paired dates."
    )
    for fold in ("fold1", "fold2"):
        parser.add_argument(f"--jepa-{fold}", required=True)
        parser.add_argument(f"--direct-{fold}", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = {}
    pooled_daily = []
    for fold in ("fold1", "fold2"):
        jepa_dir = Path(getattr(args, f"jepa_{fold}"))
        direct_dir = Path(getattr(args, f"direct_{fold}"))
        jepa_summary = json.loads((jepa_dir / "summary.json").read_text(encoding="utf-8"))
        direct_summary = json.loads(
            (direct_dir / "summary.json").read_text(encoding="utf-8")
        )
        if jepa_summary.get("target_version") != SYSTEMIC_TARGET_VERSION:
            raise ValueError(f"{fold} JEPA target version mismatch")
        if direct_summary.get("target_version") != SYSTEMIC_TARGET_VERSION:
            raise ValueError(f"{fold} direct target version mismatch")
        paired, daily = paired_fold_metrics(
            pd.read_csv(jepa_dir / "daily_test.csv"),
            pd.read_csv(direct_dir / "daily_test.csv"),
        )
        daily.insert(0, "fold", fold)
        pooled_daily.append(daily)
        folds[fold] = {
            "paired": paired,
            "jepa_absolute_gate": absolute_gate(jepa_summary),
            "direct_absolute_gate": absolute_gate(direct_summary),
        }

    pooled = pd.concat(pooled_daily, ignore_index=True)
    pooled_primary = newey_west_mean(
        pooled["primary_error_delta_direct_minus_jepa"].to_numpy(), lag=10
    )
    advantage_checks = {
        "positive_primary_delta_fold1": float(
            folds["fold1"]["paired"]["date_level"]["primary_error"]["mean"]
        )
        > 0.0,
        "positive_primary_delta_fold2": float(
            folds["fold2"]["paired"]["date_level"]["primary_error"]["mean"]
        )
        > 0.0,
        "pooled_primary_newey_west_t_above_1_96": float(
            pooled_primary["newey_west_t"]
        )
        > 1.96,
    }
    jepa_historical_gate = all(
        folds[fold]["jepa_absolute_gate"]["passed"] for fold in ("fold1", "fold2")
    )
    output = {
        "status": "complete",
        "role": "paired_systemic_head_comparison",
        "target_version": SYSTEMIC_TARGET_VERSION,
        "folds": folds,
        "pooled_date_level_primary_error": pooled_primary,
        "jepa_specific_advantage": {
            "passed": all(advantage_checks.values()),
            "checks": advantage_checks,
            "failures": [
                name for name, passed in advantage_checks.items() if not passed
            ],
        },
        "jepa_absolute_historical_gate_passed": jepa_historical_gate,
        "decision": (
            "historical_research_gate_only_requires_future_confirmation"
            if jepa_historical_gate
            else "research_only"
        ),
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    pooled.to_csv(output_dir / "paired_daily.csv", index=False)
    print(
        json.dumps(
            {
                "jepa_historical_gate": jepa_historical_gate,
                "jepa_advantage": output["jepa_specific_advantage"]["passed"],
                "pooled_primary": pooled_primary,
                "live_orders_allowed": False,
            }
        )
    )


if __name__ == "__main__":
    main()
