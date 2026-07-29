from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stock_v2.market_transition import (
    MARKET_TRANSITION_FAMILIES,
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
    binary_ranking_metrics,
)


def add_actual_salience(
    frame: pd.DataFrame, calibrations: dict[str, object]
) -> pd.DataFrame:
    frame = frame.copy()
    values = np.full(len(frame), np.nan, dtype=np.float64)
    for horizon, selected_index in frame.groupby("horizon").groups.items():
        calibration = calibrations[str(int(horizon))]
        ratios = []
        for family in MARKET_TRANSITION_FAMILIES:
            threshold = float(calibration["family_event_threshold"][family])
            ratios.append(
                frame.loc[selected_index, f"family:{family}"].to_numpy(
                    dtype=np.float64
                )
                / max(threshold, 1e-8)
            )
        values[np.asarray(selected_index, dtype=np.int64)] = np.nanmax(
            np.stack(ratios, axis=1), axis=1
        )
        if "broad_selloff" in frame.columns:
            selloff = frame.loc[selected_index, "broad_selloff"].to_numpy(
                dtype=bool
            )
            values[np.asarray(selected_index, dtype=np.int64)] = np.maximum(
                values[np.asarray(selected_index, dtype=np.int64)],
                selloff.astype(np.float64),
            )
    frame["actual_normalized_salience"] = values
    return frame


def fit_major_threshold(
    target_frame: pd.DataFrame,
    calibrations: dict[str, object],
    quantile: float = 0.90,
) -> tuple[float, float]:
    fit = target_frame[target_frame["split"] == "fit"]
    fit = add_actual_salience(fit.reset_index(drop=True), calibrations)
    trajectory = fit.groupby("date", sort=True)["actual_normalized_salience"].max()
    threshold = float(np.quantile(trajectory.to_numpy(dtype=np.float64), quantile))
    return threshold, float(np.mean(trajectory >= threshold))


def evaluate_major_trajectory(
    target_frame: pd.DataFrame,
    prediction_frame: pd.DataFrame,
    calibrations: dict[str, object],
    *,
    quantile: float = 0.90,
) -> tuple[dict[str, object], pd.DataFrame]:
    threshold, fit_rate = fit_major_threshold(
        target_frame, calibrations, quantile=quantile
    )
    test_target = target_frame[target_frame["split"] == "test"].copy()
    test_target = add_actual_salience(test_target.reset_index(drop=True), calibrations)
    predicted = prediction_frame.copy()
    keys = ["date", "horizon"]
    if predicted.duplicated(keys).any() or test_target.duplicated(keys).any():
        raise ValueError("major trajectory inputs contain duplicate date-horizon keys")
    merged = test_target.merge(
        predicted[
            ["date", "horizon", "predicted_normalized_salience"]
        ],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    if len(merged) != len(test_target) or len(merged) != len(predicted):
        raise ValueError("target and prediction date-horizon keys do not align")
    daily = (
        merged.groupby("date", sort=True)
        .agg(
            actual_normalized_salience=("actual_normalized_salience", "max"),
            predicted_normalized_salience=("predicted_normalized_salience", "max"),
        )
        .reset_index()
    )
    daily["major_trajectory_event"] = (
        daily["actual_normalized_salience"] >= threshold
    )
    ranking = binary_ranking_metrics(
        daily["major_trajectory_event"].to_numpy(dtype=bool),
        daily["predicted_normalized_salience"].to_numpy(dtype=np.float64),
        selection_rate=fit_rate,
    )
    labels = daily["major_trajectory_event"].to_numpy(dtype=bool)
    predicted_impact = daily["predicted_normalized_salience"].to_numpy(
        dtype=np.float64
    )
    actual_impact = daily["actual_normalized_salience"].to_numpy(dtype=np.float64)
    selected_count = int(ranking["selected_count"])
    selected = np.argsort(predicted_impact, kind="mergesort")[-selected_count:]
    total_event_impact = float(actual_impact[labels].sum())
    captured_event_impact = float(actual_impact[selected][labels[selected]].sum())
    impact_mass_recall = (
        captured_event_impact / total_event_impact
        if total_event_impact > 1e-12
        else float("nan")
    )
    selected_fraction = selected_count / float(len(daily))
    event_rate = float(ranking["event_rate"])
    ranking["average_precision_lift"] = (
        float(ranking["average_precision"]) / event_rate
        if event_rate > 0.0
        else float("nan")
    )
    peak_matches = []
    for _, group in merged.groupby("date", sort=True):
        if float(group["actual_normalized_salience"].max()) < threshold:
            continue
        actual_peak = int(
            group.loc[group["actual_normalized_salience"].idxmax(), "horizon"]
        )
        predicted_peak = int(
            group.loc[group["predicted_normalized_salience"].idxmax(), "horizon"]
        )
        peak_matches.append(actual_peak == predicted_peak)
    summary = {
        **ranking,
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "major_event_quantile": float(quantile),
        "fit_major_event_threshold": threshold,
        "fit_major_event_rate": fit_rate,
        "systemic_impact_mass_recall_at_major_rate": impact_mass_recall,
        "systemic_impact_mass_lift_at_major_rate": (
            impact_mass_recall / selected_fraction
            if selected_fraction > 0.0 and np.isfinite(impact_mass_recall)
            else float("nan")
        ),
        "peak_horizon_accuracy_on_major_events": (
            float(np.mean(peak_matches)) if peak_matches else float("nan")
        ),
        "test_used_for_threshold": False,
        "live_orders_allowed": False,
    }
    return summary, daily


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate fit-top-tail major market-transition trajectories."
    )
    parser.add_argument("--target-audit-root", required=True)
    parser.add_argument("--prediction-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--major-event-quantile", type=float, default=0.90)
    args = parser.parse_args()

    target_root = Path(args.target_audit_root)
    prediction_root = Path(args.prediction_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    target_summary = json.loads((target_root / "summary.json").read_text())
    if target_summary["target_version"] != MARKET_TRANSITION_TARGET_VERSION:
        raise ValueError("target audit version does not match major-event contract")
    target_frame = pd.read_csv(target_root / "daily_market_transition_targets.csv")
    prediction_frame = pd.read_csv(prediction_root / "daily_test.csv")
    summary, daily = evaluate_major_trajectory(
        target_frame,
        prediction_frame,
        target_summary["calibrations"],
        quantile=float(args.major_event_quantile),
    )
    daily.to_csv(output_dir / "daily_major_trajectory.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
