from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Iterable, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

from stock_v2.market_transition import binary_ranking_metrics


TARGETS = {
    "systemic_event": "systemic_energy",
    "price_transition": "family:price_co_movement",
    "activity_transition": "family:market_activity",
    "node_state_transition": "family:node_state",
    "topology_transition": "family:topology",
}


def load_events(paths: Sequence[Path]) -> tuple[pd.DataFrame, dict[str, object]]:
    rows = []
    rejected = 0
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                value = json.loads(line)
                event = value.get("event", value)
                if not bool(event.get("sensor_accepted", True)):
                    rejected += 1
                    continue
                effective = value.get("effective_session") or event.get(
                    "effective_session"
                )
                ticker = value.get("ticker")
                if ticker is None:
                    affected = event.get("affected_nodes") or []
                    ticker = affected[0] if affected else None
                if not effective or not ticker:
                    rejected += 1
                    continue
                rows.append(
                    {
                        "date": pd.Timestamp(effective).normalize(),
                        "ticker": str(ticker),
                        "polarity": float(event.get("polarity", 0.0) or 0.0),
                        "magnitude": float(event.get("magnitude", 0.0) or 0.0),
                        "confidence": float(event.get("confidence", 0.0) or 0.0),
                        "event_type": str(event.get("event_type", "unknown")),
                        "source": str(value.get("source", "unknown")),
                    }
                )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ValueError("no accepted events with an effective session and ticker")
    frame.sort_values(["date", "ticker"], inplace=True, ignore_index=True)
    report = {
        "accepted_rows": len(frame),
        "rejected_rows": rejected,
        "first_effective_session": str(frame["date"].min().date()),
        "last_effective_session": str(frame["date"].max().date()),
        "tickers": int(frame["ticker"].nunique()),
        "nonneutral_rows": int((frame["polarity"].abs() > 1e-8).sum()),
        "nonzero_magnitude_rows": int((frame["magnitude"].abs() > 1e-8).sum()),
        "event_types": frame["event_type"].value_counts().to_dict(),
        "sources": frame["source"].value_counts().to_dict(),
    }
    return frame, report


def build_daily_sensors(
    events: pd.DataFrame,
    dates: Iterable[pd.Timestamp],
    *,
    universe_size: int,
    windows: Sequence[int] = (1, 3, 10),
) -> pd.DataFrame:
    dates = pd.DatetimeIndex(pd.to_datetime(list(dates))).normalize().unique().sort_values()
    grouped = {date: frame for date, frame in events.groupby("date", sort=False)}
    daily = []
    for date in dates:
        selected = grouped.get(date)
        if selected is None:
            daily.append(
                {
                    "date": date,
                    "event_count": 0.0,
                    "active_tickers": frozenset(),
                    "nonneutral_count": 0.0,
                    "absolute_polarity": 0.0,
                    "confidence_magnitude": 0.0,
                }
            )
            continue
        daily.append(
            {
                "date": date,
                "event_count": float(len(selected)),
                "active_tickers": frozenset(selected["ticker"].astype(str)),
                "nonneutral_count": float((selected["polarity"].abs() > 1e-8).sum()),
                "absolute_polarity": float(selected["polarity"].abs().sum()),
                "confidence_magnitude": float(
                    (selected["confidence"] * selected["magnitude"].abs()).sum()
                ),
            }
        )
    rows = []
    for index, item in enumerate(daily):
        row = {"date": item["date"]}
        for window in windows:
            selected = daily[max(0, index - int(window) + 1) : index + 1]
            active = set().union(*(value["active_tickers"] for value in selected))
            row[f"event_count_{window}d"] = float(
                sum(value["event_count"] for value in selected)
            )
            row[f"active_ticker_ratio_{window}d"] = float(
                len(active) / max(int(universe_size), 1)
            )
            row[f"nonneutral_count_{window}d"] = float(
                sum(value["nonneutral_count"] for value in selected)
            )
            row[f"absolute_polarity_{window}d"] = float(
                sum(value["absolute_polarity"] for value in selected)
            )
            row[f"confidence_magnitude_{window}d"] = float(
                sum(value["confidence_magnitude"] for value in selected)
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _as_bool(values: pd.Series) -> np.ndarray:
    if values.dtype == bool:
        return values.to_numpy(dtype=bool)
    return values.astype(str).str.lower().eq("true").to_numpy(dtype=bool)


def _spearman(left: pd.Series, right: pd.Series) -> float:
    valid = left.notna() & right.notna()
    if int(valid.sum()) < 3 or left[valid].nunique() < 2 or right[valid].nunique() < 2:
        return float("nan")
    return float(left[valid].corr(right[valid], method="spearman"))


def evaluate_sensors(targets: pd.DataFrame, sensor_names: Sequence[str]) -> dict:
    output = {}
    for horizon, horizon_rows in targets.groupby("horizon", sort=True):
        output[str(int(horizon))] = {}
        fit = horizon_rows[horizon_rows["split"] == "fit"]
        if fit.empty:
            raise ValueError(f"horizon {horizon} has no fit rows")
        for label_name, intensity_name in TARGETS.items():
            sensors = {}
            for sensor in sensor_names:
                fit_correlation = _spearman(fit[sensor], fit[intensity_name])
                direction = -1.0 if np.isfinite(fit_correlation) and fit_correlation < 0 else 1.0
                by_split = {}
                for split in ("fit", "validation", "test"):
                    selected = horizon_rows[horizon_rows["split"] == split]
                    labels = _as_bool(selected[label_name])
                    scores = direction * selected[sensor].to_numpy(dtype=np.float64)
                    valid = np.isfinite(scores)
                    valid_labels = labels[valid]
                    valid_scores = scores[valid]
                    event_rate = (
                        float(valid_labels.mean()) if valid_labels.size else float("nan")
                    )
                    if (
                        valid_scores.size < 2
                        or np.unique(valid_scores).size < 2
                        or np.unique(valid_labels).size < 2
                    ):
                        metrics = {
                            "roc_auc": float("nan"),
                            "average_precision": float("nan"),
                        }
                    else:
                        metrics = binary_ranking_metrics(
                            valid_labels,
                            valid_scores,
                            selection_rate=float(_as_bool(fit[label_name]).mean()),
                        )
                    by_split[split] = {
                        "rows": int(valid.sum()),
                        "intensity_spearman": _spearman(
                            selected[sensor] * direction, selected[intensity_name]
                        ),
                        "roc_auc": float(metrics["roc_auc"]),
                        "average_precision": float(metrics["average_precision"]),
                        "average_precision_lift": (
                            float(metrics["average_precision"]) / event_rate
                            if event_rate > 0.0
                            else float("nan")
                        ),
                    }
                sensors[sensor] = {
                    "fit_direction": int(direction),
                    "raw_fit_intensity_spearman": fit_correlation,
                    "splits": by_split,
                }
            finite_fit = [
                (name, value["splits"]["fit"]["roc_auc"])
                for name, value in sensors.items()
                if np.isfinite(value["splits"]["fit"]["roc_auc"])
            ]
            selected_name = max(finite_fit, key=lambda item: item[1])[0] if finite_fit else None
            output[str(int(horizon))][label_name] = {
                "fit_selected_sensor": selected_name,
                "fit_selected_metrics": sensors.get(selected_name) if selected_name else None,
                "sensors": sensors,
            }
    return output


def _aggregate_selected(evaluation: Mapping[str, object]) -> dict[str, object]:
    result = {}
    for target in TARGETS:
        values = {"fit": [], "validation": [], "test": []}
        selected = {}
        for horizon, horizon_values in evaluation.items():
            item = horizon_values[target]
            selected[horizon] = item["fit_selected_sensor"]
            metrics = item["fit_selected_metrics"]
            if metrics is None:
                continue
            for split in values:
                values[split].append(metrics["splits"][split]["roc_auc"])
        result[target] = {
            "fit_selected_sensor_by_horizon": selected,
            "macro_auc": {
                split: float(np.nanmean(split_values)) if split_values else float("nan")
                for split, split_values in values.items()
            },
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether causal event-count sensors lead broad market transitions."
    )
    parser.add_argument("--event-path", action="append", required=True)
    parser.add_argument("--target-csv", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--universe-size", type=int, default=500)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.read_csv(args.target_csv)
    required = {"date", "horizon", "split", *TARGETS, *TARGETS.values()}
    missing = required - set(targets.columns)
    if missing:
        raise ValueError(f"target rows are missing columns: {sorted(missing)}")
    targets["date"] = pd.to_datetime(targets["date"]).dt.normalize()
    events, event_report = load_events([Path(path) for path in args.event_path])
    sensors = build_daily_sensors(
        events,
        targets["date"].unique(),
        universe_size=int(args.universe_size),
    )
    sensor_names = [name for name in sensors.columns if name != "date"]
    merged = targets.merge(sensors, on="date", how="left", validate="many_to_one")
    evaluation = evaluate_sensors(merged, sensor_names)
    summary = {
        "status": "complete",
        "role": "causal_market_event_sensor_audit",
        "event_paths": args.event_path,
        "target_csv": args.target_csv,
        "event_input": event_report,
        "sensor_names": sensor_names,
        "evaluation": evaluation,
        "fit_selected_macro": _aggregate_selected(evaluation),
        "interpretation_contract": {
            "sensor_direction_selected_on_fit_only": True,
            "sensor_identity_selected_on_fit_only": True,
            "validation_and_test_not_used_for_selection": True,
            "event_availability_uses_effective_session": True,
        },
        "decision": "diagnostic_only",
        "live_orders_allowed": False,
    }
    sensors.to_csv(output_dir / "daily_event_sensors.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "event_input": event_report,
                "fit_selected_macro": summary["fit_selected_macro"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
