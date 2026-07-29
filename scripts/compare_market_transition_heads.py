from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.benchmark_direct_baselines import newey_west_mean
from scripts.benchmark_latent_trajectory_path_head import HORIZON_WEIGHTS
from stock_v2.market_transition import (
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
)
from stock_v2.market_transition_head import MARKET_FAMILY_TARGETS


def _load_daily(path: Path, role: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"date", "horizon", "actual_systemic_event", "probability_systemic_event"}
    for family in MARKET_FAMILY_TARGETS:
        required.add(f"actual_family_{family}")
        required.add(f"predicted_family_{family}")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{role} daily rows are missing: {sorted(missing)}")
    if frame.duplicated(["date", "horizon"]).any():
        raise ValueError(f"{role} daily rows contain duplicate keys")
    return frame


def _as_bool(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    return values.astype(str).str.lower().eq("true").to_numpy(dtype=bool)


def paired_rows(jepa_path: Path, direct_path: Path) -> pd.DataFrame:
    jepa = _load_daily(jepa_path, "JEPA")
    direct = _load_daily(direct_path, "direct")
    merged = jepa.merge(
        direct,
        on=["date", "horizon"],
        suffixes=("_jepa", "_direct"),
        validate="one_to_one",
    )
    if len(merged) != len(jepa) or len(merged) != len(direct):
        raise ValueError("JEPA and direct daily keys do not align")
    jepa_errors = []
    direct_errors = []
    actual_vectors = []
    jepa_vectors = []
    direct_vectors = []
    for family in MARKET_FAMILY_TARGETS:
        actual = merged[f"actual_family_{family}_jepa"].to_numpy(dtype=np.float64)
        direct_actual = merged[f"actual_family_{family}_direct"].to_numpy(
            dtype=np.float64
        )
        if not np.allclose(actual, direct_actual, equal_nan=True, atol=1e-7):
            raise ValueError(f"JEPA and direct actual family targets differ: {family}")
        jepa_prediction = merged[f"predicted_family_{family}_jepa"].to_numpy(
            dtype=np.float64
        )
        direct_prediction = merged[f"predicted_family_{family}_direct"].to_numpy(
            dtype=np.float64
        )
        jepa_errors.append(np.abs(np.log1p(jepa_prediction) - np.log1p(actual)))
        direct_errors.append(
            np.abs(np.log1p(direct_prediction) - np.log1p(actual))
        )
        actual_vectors.append(actual)
        jepa_vectors.append(jepa_prediction)
        direct_vectors.append(direct_prediction)
    jepa_error = np.nanmean(np.stack(jepa_errors, axis=1), axis=1)
    direct_error = np.nanmean(np.stack(direct_errors, axis=1), axis=1)
    merged["equal_family_log_error_delta_direct_minus_jepa"] = (
        direct_error - jepa_error
    )

    labels = _as_bool(merged["actual_systemic_event_jepa"])
    direct_labels = _as_bool(merged["actual_systemic_event_direct"])
    if not np.array_equal(labels, direct_labels):
        raise ValueError("JEPA and direct systemic event labels differ")
    jepa_probability = merged["probability_systemic_event_jepa"].to_numpy(
        dtype=np.float64
    )
    direct_probability = merged["probability_systemic_event_direct"].to_numpy(
        dtype=np.float64
    )
    merged["systemic_brier_delta_direct_minus_jepa"] = np.square(
        direct_probability - labels
    ) - np.square(jepa_probability - labels)

    actual_matrix = np.stack(actual_vectors, axis=1)
    jepa_matrix = np.stack(jepa_vectors, axis=1)
    direct_matrix = np.stack(direct_vectors, axis=1)

    def cosine(prediction):
        denominator = np.linalg.norm(actual_matrix, axis=1) * np.linalg.norm(
            prediction, axis=1
        )
        return np.divide(
            np.sum(actual_matrix * prediction, axis=1),
            denominator,
            out=np.full(len(prediction), np.nan),
            where=denominator > 1e-12,
        )

    merged["signature_cosine_delta_jepa_minus_direct"] = cosine(
        jepa_matrix
    ) - cosine(direct_matrix)
    return merged


def paired_summary(frame: pd.DataFrame) -> dict[str, object]:
    metrics = (
        "equal_family_log_error_delta_direct_minus_jepa",
        "systemic_brier_delta_direct_minus_jepa",
        "signature_cosine_delta_jepa_minus_direct",
    )
    by_horizon = {}
    for horizon, selected in frame.groupby("horizon", sort=True):
        by_horizon[str(int(horizon))] = {
            name: newey_west_mean(selected[name].to_numpy(), lag=int(horizon))
            for name in metrics
        }
    daily = frame.groupby("date", sort=True)[list(metrics)].mean().reset_index()
    date_level = {
        name: newey_west_mean(daily[name].to_numpy(), lag=10) for name in metrics
    }
    return {
        "rows": len(frame),
        "dates": int(frame["date"].nunique()),
        "by_horizon": by_horizon,
        "date_level": date_level,
        "daily": daily,
    }


def _weighted(values: Mapping[str, float]) -> float:
    total = 0.0
    weight_sum = 0.0
    for horizon, value in values.items():
        if not np.isfinite(float(value)):
            continue
        weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
        total += weight * float(value)
        weight_sum += weight
    return float(total / weight_sum) if weight_sum else float("nan")


def historical_gate(
    summary: Mapping[str, object], major: Mapping[str, object]
) -> dict[str, object]:
    test = summary["metrics"]["test"]
    trajectory = test["trajectory"]
    ap_lift = float(major["average_precision_lift"])
    family_auc_by_name = {name: {} for name in MARKET_FAMILY_TARGETS}
    family_corr_by_name = {name: {} for name in MARKET_FAMILY_TARGETS}
    signature = {}
    systemic_auc = {}
    selloff_auc = {}
    impact_mass_lift = {}
    for horizon, item in test["horizons"].items():
        signature[horizon] = float(item["event_transition_signature_cosine"])
        systemic_auc[horizon] = float(item["systemic_event"]["roc_auc"])
        selloff_auc[horizon] = float(item["broad_selloff"]["roc_auc"])
        selected_fraction = float(item["systemic_event"]["selected_count"]) / max(
            float(item["systemic_event"]["rows"]), 1.0
        )
        impact_mass_lift[horizon] = float(
            item["systemic_event"]["systemic_impact_mass_recall_at_fit_rate"]
        ) / max(selected_fraction, 1e-8)
        for family in MARKET_FAMILY_TARGETS:
            family_auc_by_name[family][horizon] = float(
                item["family_events"][family]["roc_auc"]
            )
            family_corr_by_name[family][horizon] = float(
                item["family_intensity_correlation"][family]
            )
    family_auc = {
        name: _weighted(values) for name, values in family_auc_by_name.items()
    }
    family_corr = {
        name: _weighted(values) for name, values in family_corr_by_name.items()
    }
    macro_auc = float(np.nanmean(list(family_auc.values())))
    minimum_auc = float(np.nanmin(list(family_auc.values())))
    macro_corr = float(np.nanmean(list(family_corr.values())))
    weighted_signature = _weighted(signature)
    weighted_systemic_auc = _weighted(systemic_auc)
    weighted_selloff_auc = _weighted(selloff_auc)
    weighted_impact_mass_lift = _weighted(impact_mass_lift)
    checks = {
        "major_trajectory_auc_at_least_0_65": float(major["roc_auc"]) >= 0.65,
        "major_trajectory_ap_lift_at_least_1_50": ap_lift >= 1.50,
        "mean_systemic_auc_at_least_0_65": weighted_systemic_auc >= 0.65,
        "macro_family_auc_at_least_0_68": macro_auc >= 0.68,
        "minimum_family_auc_at_least_0_60": minimum_auc >= 0.60,
        "macro_family_correlation_at_least_0_20": macro_corr >= 0.20,
        "node_state_auc_at_least_0_75": family_auc["node_state"] >= 0.75,
        "market_activity_auc_at_least_0_60": family_auc["market_activity"] >= 0.60,
        "mean_broad_selloff_auc_at_least_0_60": weighted_selloff_auc >= 0.60,
        "transition_signature_cosine_at_least_0_62": weighted_signature >= 0.62,
        "peak_horizon_accuracy_above_0_20": float(
            major["peak_horizon_accuracy_on_major_events"]
        )
        > 0.20,
        "systemic_impact_mass_lift_at_least_1_50": weighted_impact_mass_lift
        >= 1.50,
        "major_systemic_impact_mass_recall_at_least_0_25": float(
            major["systemic_impact_mass_recall_at_major_rate"]
        )
        >= 0.25,
        "major_systemic_impact_mass_lift_at_least_2_00": float(
            major["systemic_impact_mass_lift_at_major_rate"]
        )
        >= 2.00,
    }
    return {
        "passed": bool(all(checks.values())),
        "checks": checks,
        "failures": [name for name, passed in checks.items() if not passed],
        "values": {
            "major_trajectory_auc": float(major["roc_auc"]),
            "major_trajectory_ap_lift": ap_lift,
            "major_trajectory_test_event_rate": float(major["event_rate"]),
            "fit_major_trajectory_event_rate": float(major["fit_major_event_rate"]),
            "weighted_systemic_auc": weighted_systemic_auc,
            "weighted_broad_selloff_auc": weighted_selloff_auc,
            "weighted_systemic_impact_mass_lift": weighted_impact_mass_lift,
            "major_systemic_impact_mass_recall": float(
                major["systemic_impact_mass_recall_at_major_rate"]
            ),
            "major_systemic_impact_mass_lift": float(
                major["systemic_impact_mass_lift_at_major_rate"]
            ),
            "family_auc": family_auc,
            "macro_family_auc": macro_auc,
            "minimum_family_auc": minimum_auc,
            "family_intensity_correlation": family_corr,
            "macro_family_intensity_correlation": macro_corr,
            "weighted_transition_signature_cosine": weighted_signature,
            "peak_horizon_accuracy": float(
                major["peak_horizon_accuracy_on_major_events"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Paired comparison of joint JEPA and direct market-transition heads."
    )
    parser.add_argument("--jepa-fold1", required=True)
    parser.add_argument("--direct-fold1", required=True)
    parser.add_argument("--jepa-fold2", required=True)
    parser.add_argument("--direct-fold2", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    folds = {
        "fold1": (Path(args.jepa_fold1), Path(args.direct_fold1)),
        "fold2": (Path(args.jepa_fold2), Path(args.direct_fold2)),
    }
    result = {}
    pooled_daily = []
    for fold, (jepa_root, direct_root) in folds.items():
        jepa_summary = json.loads((jepa_root / "summary.json").read_text())
        direct_summary = json.loads((direct_root / "summary.json").read_text())
        jepa_major = json.loads(
            (jepa_root / "major_trajectory" / "summary.json").read_text()
        )
        direct_major = json.loads(
            (direct_root / "major_trajectory" / "summary.json").read_text()
        )
        if jepa_summary["target_version"] != MARKET_TRANSITION_TARGET_VERSION:
            raise ValueError("JEPA target version does not match comparison contract")
        if direct_summary["target_version"] != MARKET_TRANSITION_TARGET_VERSION:
            raise ValueError("direct target version does not match comparison contract")
        if (
            jepa_summary.get("impact_metric_version")
            != MARKET_TRANSITION_IMPACT_METRIC_VERSION
        ):
            raise ValueError("JEPA impact metric version does not match")
        if (
            direct_summary.get("impact_metric_version")
            != MARKET_TRANSITION_IMPACT_METRIC_VERSION
        ):
            raise ValueError("direct impact metric version does not match")
        paired = paired_rows(
            jepa_root / "daily_test.csv", direct_root / "daily_test.csv"
        )
        paired.to_csv(output_dir / f"paired_{fold}.csv", index=False)
        comparison = paired_summary(paired)
        daily = comparison.pop("daily")
        daily.insert(0, "fold", fold)
        pooled_daily.append(daily)
        result[fold] = {
            "paired": comparison,
            "jepa_historical_gate": historical_gate(jepa_summary, jepa_major),
            "direct_historical_gate": historical_gate(direct_summary, direct_major),
        }
    pooled = pd.concat(pooled_daily, ignore_index=True)
    primary = newey_west_mean(
        pooled["equal_family_log_error_delta_direct_minus_jepa"].to_numpy(), lag=10
    )
    advantage_checks = {
        "positive_primary_delta_fold1": result["fold1"]["paired"]["date_level"][
            "equal_family_log_error_delta_direct_minus_jepa"
        ]["mean"]
        > 0.0,
        "positive_primary_delta_fold2": result["fold2"]["paired"]["date_level"][
            "equal_family_log_error_delta_direct_minus_jepa"
        ]["mean"]
        > 0.0,
        "pooled_primary_newey_west_t_above_1_96": float(primary["newey_west_t"])
        > 1.96,
        "major_systemic_impact_mass_lift_not_below_direct_fold1": float(
            result["fold1"]["jepa_historical_gate"]["values"][
                "major_systemic_impact_mass_lift"
            ]
        )
        >= float(
            result["fold1"]["direct_historical_gate"]["values"][
                "major_systemic_impact_mass_lift"
            ]
        ),
        "major_systemic_impact_mass_lift_not_below_direct_fold2": float(
            result["fold2"]["jepa_historical_gate"]["values"][
                "major_systemic_impact_mass_lift"
            ]
        )
        >= float(
            result["fold2"]["direct_historical_gate"]["values"][
                "major_systemic_impact_mass_lift"
            ]
        ),
    }
    summary = {
        "status": "complete",
        "role": "paired_joint_market_transition_head_comparison",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "folds": result,
        "pooled_date_level_equal_family_error": primary,
        "jepa_specific_advantage": {
            "passed": bool(all(advantage_checks.values())),
            "checks": advantage_checks,
            "failures": [
                name for name, passed in advantage_checks.items() if not passed
            ],
        },
        "jepa_historical_gate_passed": bool(
            all(result[fold]["jepa_historical_gate"]["passed"] for fold in result)
        ),
        "decision": "research_only",
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "jepa_historical_gate": summary["jepa_historical_gate_passed"],
                "jepa_advantage": summary["jepa_specific_advantage"]["passed"],
                "pooled_primary": primary,
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
