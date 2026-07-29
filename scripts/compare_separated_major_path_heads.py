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
from scripts.compare_major_path_transition_heads import path_summary
from scripts.compare_market_transition_heads import (
    historical_gate,
    paired_rows,
    paired_summary,
)
from stock_v2.market_transition import (
    MARKET_TRANSITION_TARGET_VERSION,
    binary_ranking_metrics,
)
from stock_v2.separated_major_path import SEPARATED_MAJOR_OBJECTIVE_VERSION


def _load_major(path: Path, role: str) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {
        "date",
        "actual_major_event",
        "actual_path_salience",
        "predicted_path_salience",
        "major_logit",
        "actual_peak_horizon",
        "predicted_peak_horizon",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{role} major rows are missing: {sorted(missing)}")
    if frame["date"].duplicated().any():
        raise ValueError(f"{role} major rows contain duplicate dates")
    return frame


def _as_bool(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    return values.astype(str).str.lower().eq("true").to_numpy(dtype=bool)


def paired_major_rows(jepa_path: Path, direct_path: Path) -> pd.DataFrame:
    jepa = _load_major(jepa_path, "JEPA")
    direct = _load_major(direct_path, "direct")
    merged = jepa.merge(direct, on="date", suffixes=("_jepa", "_direct"))
    if len(merged) != len(jepa) or len(merged) != len(direct):
        raise ValueError("JEPA and direct major dates do not align")
    labels = _as_bool(merged["actual_major_event_jepa"])
    direct_labels = _as_bool(merged["actual_major_event_direct"])
    if not np.array_equal(labels, direct_labels):
        raise ValueError("JEPA and direct major labels differ")
    actual = merged["actual_path_salience_jepa"].to_numpy(dtype=np.float64)
    direct_actual = merged["actual_path_salience_direct"].to_numpy(dtype=np.float64)
    if not np.allclose(actual, direct_actual, atol=1e-7):
        raise ValueError("JEPA and direct actual path salience differs")
    actual_peak = merged["actual_peak_horizon_jepa"].to_numpy(dtype=np.int64)
    if not np.array_equal(
        actual_peak,
        merged["actual_peak_horizon_direct"].to_numpy(dtype=np.int64),
    ):
        raise ValueError("JEPA and direct actual peak horizons differ")
    jepa_path_value = merged["predicted_path_salience_jepa"].to_numpy(
        dtype=np.float64
    )
    direct_path_value = merged["predicted_path_salience_direct"].to_numpy(
        dtype=np.float64
    )
    merged["actual_path_salience"] = actual
    merged["actual_major_event"] = labels
    merged["path_log_error_delta_direct_minus_jepa"] = np.abs(
        np.log1p(direct_path_value) - np.log1p(actual)
    ) - np.abs(np.log1p(jepa_path_value) - np.log1p(actual))
    probability_jepa = 1.0 / (
        1.0 + np.exp(-np.clip(merged["major_logit_jepa"].to_numpy(), -30.0, 30.0))
    )
    probability_direct = 1.0 / (
        1.0
        + np.exp(-np.clip(merged["major_logit_direct"].to_numpy(), -30.0, 30.0))
    )
    merged["major_brier_delta_direct_minus_jepa"] = np.square(
        probability_direct - labels
    ) - np.square(probability_jepa - labels)
    merged["peak_correct_delta_jepa_minus_direct"] = np.where(
        labels,
        (
            merged["predicted_peak_horizon_jepa"]
            == merged["actual_peak_horizon_jepa"]
        ).astype(float)
        - (
            merged["predicted_peak_horizon_direct"]
            == merged["actual_peak_horizon_direct"]
        ).astype(float),
        np.nan,
    )
    merged["predicted_path_salience_jepa"] = jepa_path_value
    merged["predicted_path_salience_direct"] = direct_path_value
    return merged


def _validate_summary(summary: Mapping[str, object], role: str) -> None:
    if summary.get("target_version") != MARKET_TRANSITION_TARGET_VERSION:
        raise ValueError(f"{role} target version does not match")
    if summary.get("objective_version") != SEPARATED_MAJOR_OBJECTIVE_VERSION:
        raise ValueError(f"{role} objective version does not match")
    if bool(summary.get("live_orders_allowed", True)):
        raise ValueError(f"{role} artifact is not research-only")


def separated_path_summary(
    frame: pd.DataFrame, fit_event_rate: float
) -> dict[str, object]:
    result = path_summary(frame, fit_event_rate)
    labels = _as_bool(frame["actual_major_event"])
    result["path_salience_ranking"] = {
        "jepa": result.pop("jepa"),
        "direct": result.pop("direct"),
    }
    result["dedicated_event_ranking"] = {}
    for role in ("jepa", "direct"):
        ranking = binary_ranking_metrics(
            labels,
            frame[f"major_logit_{role}"].to_numpy(dtype=np.float64),
            selection_rate=float(fit_event_rate),
        )
        event_rate = float(ranking["event_rate"])
        ranking["average_precision_lift"] = (
            float(ranking["average_precision"]) / event_rate
            if event_rate > 0.0
            else float("nan")
        )
        result["dedicated_event_ranking"][role] = ranking
    return result


def _historical_gate(summary: Mapping[str, object]) -> dict[str, object]:
    wrapped = {"metrics": {"test": summary["metrics"]["test"]["base"]}}
    return historical_gate(wrapped, summary["metrics"]["test"]["major_path"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare same-objective separated JEPA and direct heads."
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
        family_rows = paired_rows(
            jepa_root / "daily_test.csv", direct_root / "daily_test.csv"
        )
        family = paired_summary(family_rows)
        daily = family.pop("daily")
        daily.insert(0, "fold", fold)
        family_daily.append(daily)
        paths = paired_major_rows(
            jepa_root / "daily_major_test.csv",
            direct_root / "daily_major_test.csv",
        )
        paths.insert(0, "fold", fold)
        paths.to_csv(output_dir / f"paired_major_path_{fold}.csv", index=False)
        path_daily.append(paths)
        folds[fold] = {
            "family_paired": family,
            "path_paired": separated_path_summary(
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
    checks = {
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
        "role": "paired_separated_major_path_head_comparison",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "objective_version": SEPARATED_MAJOR_OBJECTIVE_VERSION,
        "folds": folds,
        "pooled_date_level_equal_family_error": primary,
        "pooled_date_level_path_error": path_error,
        "jepa_specific_advantage": {
            "passed": bool(all(checks.values())),
            "checks": checks,
            "failures": [name for name, passed in checks.items() if not passed],
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
