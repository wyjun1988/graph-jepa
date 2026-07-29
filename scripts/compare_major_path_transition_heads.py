from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from scripts.benchmark_direct_baselines import newey_west_mean
from scripts.compare_market_transition_heads import (
    _as_bool,
    _load_daily,
    historical_gate,
    paired_rows,
    paired_summary,
)
from stock_v2.market_transition import (
    MARKET_TRANSITION_TARGET_VERSION,
    binary_ranking_metrics,
)
from stock_v2.market_transition_head import MARKET_FAMILY_TARGETS


OBJECTIVE_VERSION = "major_path_v31_20260714"


def _sigmoid(value: np.ndarray) -> np.ndarray:
    value = np.clip(np.asarray(value, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-value))


def _thresholds(summary: Mapping[str, object], horizons: list[int]) -> np.ndarray:
    return np.asarray(
        [
            [
                float(
                    summary["target_contracts"][str(horizon)]["calibration"][
                        "family_event_threshold"
                    ][family]
                )
                for family in MARKET_FAMILY_TARGETS
            ]
            for horizon in horizons
        ],
        dtype=np.float64,
    )


def _validate_summary(summary: Mapping[str, object], role: str) -> None:
    if summary.get("target_version") != MARKET_TRANSITION_TARGET_VERSION:
        raise ValueError(f"{role} target version does not match")
    if summary.get("objective_version") != OBJECTIVE_VERSION:
        raise ValueError(f"{role} objective version does not match")
    if bool(summary.get("live_orders_allowed", True)):
        raise ValueError(f"{role} artifact is not research-only")


def major_path_rows(
    jepa_path: Path,
    direct_path: Path,
    jepa_summary: Mapping[str, object],
    direct_summary: Mapping[str, object],
) -> pd.DataFrame:
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

    horizons = sorted(int(value) for value in merged["horizon"].unique())
    jepa_thresholds = _thresholds(jepa_summary, horizons)
    direct_thresholds = _thresholds(direct_summary, horizons)
    if not np.allclose(jepa_thresholds, direct_thresholds, atol=1e-10):
        raise ValueError("JEPA and direct family thresholds differ")
    jepa_contract = jepa_summary["major_path_contract"]
    direct_contract = direct_summary["major_path_contract"]
    for name in ("event_threshold", "fit_event_rate", "logit_scale"):
        if not np.isclose(float(jepa_contract[name]), float(direct_contract[name])):
            raise ValueError(f"JEPA and direct major-path contracts differ: {name}")

    rows = []
    for date, selected in merged.groupby("date", sort=True):
        selected = selected.sort_values("horizon")
        selected_horizons = selected["horizon"].to_numpy(dtype=np.int64)
        if not np.array_equal(selected_horizons, np.asarray(horizons)):
            raise ValueError(f"incomplete horizon path on {date}")
        actual = []
        predicted_jepa = []
        predicted_direct = []
        for family in MARKET_FAMILY_TARGETS:
            family_actual = selected[f"actual_family_{family}_jepa"].to_numpy(
                dtype=np.float64
            )
            direct_actual = selected[
                f"actual_family_{family}_direct"
            ].to_numpy(dtype=np.float64)
            if not np.allclose(family_actual, direct_actual, atol=1e-7):
                raise ValueError(f"JEPA and direct actual paths differ: {family}")
            actual.append(family_actual)
            predicted_jepa.append(
                selected[f"predicted_family_{family}_jepa"].to_numpy(
                    dtype=np.float64
                )
            )
            predicted_direct.append(
                selected[f"predicted_family_{family}_direct"].to_numpy(
                    dtype=np.float64
                )
            )
        actual = np.stack(actual, axis=1) / jepa_thresholds
        predicted_jepa = np.stack(predicted_jepa, axis=1) / jepa_thresholds
        predicted_direct = np.stack(predicted_direct, axis=1) / jepa_thresholds
        actual_horizon = np.max(actual, axis=1)
        jepa_horizon = np.max(predicted_jepa, axis=1)
        direct_horizon = np.max(predicted_direct, axis=1)
        actual_path = float(np.max(actual_horizon))
        jepa_path_value = float(np.max(jepa_horizon))
        direct_path_value = float(np.max(direct_horizon))
        threshold = float(jepa_contract["event_threshold"])
        label = bool(actual_path >= threshold)
        rows.append(
            {
                "date": date,
                "actual_path_salience": actual_path,
                "predicted_path_salience_jepa": jepa_path_value,
                "predicted_path_salience_direct": direct_path_value,
                "actual_major_event": label,
                "actual_peak_horizon": horizons[int(np.argmax(actual_horizon))],
                "predicted_peak_horizon_jepa": horizons[
                    int(np.argmax(jepa_horizon))
                ],
                "predicted_peak_horizon_direct": horizons[
                    int(np.argmax(direct_horizon))
                ],
                "path_log_error_delta_direct_minus_jepa": abs(
                    np.log1p(direct_path_value) - np.log1p(actual_path)
                )
                - abs(np.log1p(jepa_path_value) - np.log1p(actual_path)),
            }
        )
    frame = pd.DataFrame(rows)
    scale = max(float(jepa_contract["logit_scale"]), 1e-8)
    labels = frame["actual_major_event"].to_numpy(dtype=bool)
    probability_jepa = _sigmoid(
        (frame["predicted_path_salience_jepa"].to_numpy() - threshold) / scale
    )
    probability_direct = _sigmoid(
        (frame["predicted_path_salience_direct"].to_numpy() - threshold) / scale
    )
    frame["major_brier_delta_direct_minus_jepa"] = np.square(
        probability_direct - labels
    ) - np.square(probability_jepa - labels)
    frame["peak_correct_delta_jepa_minus_direct"] = np.where(
        labels,
        (
            frame["predicted_peak_horizon_jepa"] == frame["actual_peak_horizon"]
        ).astype(float)
        - (
            frame["predicted_peak_horizon_direct"] == frame["actual_peak_horizon"]
        ).astype(float),
        np.nan,
    )
    return frame


def path_summary(
    frame: pd.DataFrame, fit_event_rate: float
) -> dict[str, object]:
    labels = _as_bool(frame["actual_major_event"])
    result = {
        "rows": len(frame),
        "events": int(labels.sum()),
        "paired": {
            name: newey_west_mean(frame[name].to_numpy(), lag=10)
            for name in (
                "path_log_error_delta_direct_minus_jepa",
                "major_brier_delta_direct_minus_jepa",
                "peak_correct_delta_jepa_minus_direct",
            )
        },
    }
    for role in ("jepa", "direct"):
        ranking = binary_ranking_metrics(
            labels,
            frame[f"predicted_path_salience_{role}"].to_numpy(dtype=np.float64),
            selection_rate=float(fit_event_rate),
        )
        event_rate = float(ranking["event_rate"])
        ranking["average_precision_lift"] = (
            float(ranking["average_precision"]) / event_rate
            if event_rate > 0.0
            else float("nan")
        )
        result[role] = ranking
    return result


def _historical_gate(summary: Mapping[str, object]) -> dict[str, object]:
    wrapped = {"metrics": {"test": summary["metrics"]["test"]["base"]}}
    return historical_gate(wrapped, summary["metrics"]["test"]["major_path"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare same-objective JEPA and direct major-path heads."
    )
    parser.add_argument("--jepa-fold1", required=True)
    parser.add_argument("--direct-fold1", required=True)
    parser.add_argument("--jepa-fold2", required=True)
    parser.add_argument("--direct-fold2", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    fold_paths = {
        "fold1": (Path(args.jepa_fold1), Path(args.direct_fold1)),
        "fold2": (Path(args.jepa_fold2), Path(args.direct_fold2)),
    }
    folds = {}
    family_daily = []
    path_daily = []
    for fold, (jepa_root, direct_root) in fold_paths.items():
        jepa_summary = json.loads((jepa_root / "summary.json").read_text())
        direct_summary = json.loads((direct_root / "summary.json").read_text())
        _validate_summary(jepa_summary, "JEPA")
        _validate_summary(direct_summary, "direct")
        paired = paired_rows(
            jepa_root / "daily_test.csv", direct_root / "daily_test.csv"
        )
        family = paired_summary(paired)
        daily = family.pop("daily")
        daily.insert(0, "fold", fold)
        family_daily.append(daily)
        paths = major_path_rows(
            jepa_root / "daily_test.csv",
            direct_root / "daily_test.csv",
            jepa_summary,
            direct_summary,
        )
        paths.insert(0, "fold", fold)
        paths.to_csv(output_dir / f"paired_major_path_{fold}.csv", index=False)
        path_daily.append(paths)
        folds[fold] = {
            "family_paired": family,
            "path_paired": path_summary(
                paths, float(jepa_summary["major_path_contract"]["fit_event_rate"])
            ),
            "jepa_historical_gate": _historical_gate(jepa_summary),
            "direct_historical_gate": _historical_gate(direct_summary),
        }

    pooled_family = pd.concat(family_daily, ignore_index=True)
    pooled_path = pd.concat(path_daily, ignore_index=True)
    primary = newey_west_mean(
        pooled_family[
            "equal_family_log_error_delta_direct_minus_jepa"
        ].to_numpy(),
        lag=10,
    )
    path_error = newey_west_mean(
        pooled_path["path_log_error_delta_direct_minus_jepa"].to_numpy(), lag=10
    )
    advantage_checks = {
        "positive_primary_delta_fold1": folds["fold1"]["family_paired"][
            "date_level"
        ]["equal_family_log_error_delta_direct_minus_jepa"]["mean"]
        > 0.0,
        "positive_primary_delta_fold2": folds["fold2"]["family_paired"][
            "date_level"
        ]["equal_family_log_error_delta_direct_minus_jepa"]["mean"]
        > 0.0,
        "pooled_primary_newey_west_t_above_1_96": float(primary["newey_west_t"])
        > 1.96,
    }
    summary = {
        "status": "complete",
        "role": "paired_major_path_transition_head_comparison",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "objective_version": OBJECTIVE_VERSION,
        "folds": folds,
        "pooled_date_level_equal_family_error": primary,
        "pooled_date_level_path_error": path_error,
        "jepa_specific_advantage": {
            "passed": bool(all(advantage_checks.values())),
            "checks": advantage_checks,
            "failures": [
                name for name, passed in advantage_checks.items() if not passed
            ],
        },
        "jepa_historical_gate_passed": bool(
            all(folds[fold]["jepa_historical_gate"]["passed"] for fold in folds)
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
                "pooled_family_error": primary,
                "pooled_path_error": path_error,
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
