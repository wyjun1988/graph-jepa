from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_latent_trajectory_path_head import chronological_splits
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.run_real_backtest import date_indices, parse_int_list, temporal_training_indices
from stock_v2.systemic_transition import (
    SYSTEMIC_FAMILIES,
    SYSTEMIC_TARGET_VERSION,
    event_labels,
    fit_systemic_calibration,
    score_systemic_components,
    transition_components,
)


def _finite_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"count": 0}
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "q10": float(np.quantile(array, 0.10)),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
        "q99": float(np.quantile(array, 0.99)),
        "max": float(array.max()),
    }


def _split_steps(features, checkpoint_args, horizons, validation_days):
    train_end = str(checkpoint_args["train_end"])
    max_horizon = max(horizons)
    edge_window = int(checkpoint_args.get("edge_window", 60))
    train_indices = date_indices(features.dates, end=train_end)
    train_steps = temporal_training_indices(
        train_indices,
        edge_window=edge_window,
        max_rollout_offset=max_horizon,
        total_steps=len(features.dates),
    )
    fit_steps, validation_steps = chronological_splits(
        train_steps, validation_days, max_horizon
    )
    test_steps = date_indices(
        features.dates,
        start=(pd.Timestamp(train_end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    test_steps = test_steps[
        (test_steps >= edge_window)
        & (test_steps <= len(features.dates) - 1 - max_horizon)
    ]
    if not len(test_steps):
        raise ValueError("systemic target audit has no out-of-sample dates")
    return {
        "fit": fit_steps,
        "validation": validation_steps,
        "test": test_steps,
    }


def _actual_rows(features, steps: np.ndarray, horizons: Sequence[int], split: str):
    stock_count = int(features.tradable_count)
    return_index = features.feature_names.index("return_1d")
    rows: list[dict[str, Any]] = []
    for step in np.asarray(steps, dtype=np.int64):
        current_state = features.features[int(step), :stock_count]
        current_raw = features.raw_features[int(step), :stock_count]
        current_available = features.available_mask[int(step), :stock_count] > 0.5
        for horizon in horizons:
            target_step = int(step) + int(horizon)
            future_state = features.features[target_step, :stock_count]
            future_raw = features.raw_features[target_step, :stock_count]
            future_available = features.available_mask[target_step, :stock_count] > 0.5
            path = np.asarray(
                features.target_return_paths[int(horizon)][int(step), :stock_count],
                dtype=np.float64,
            )
            node_mask = (
                current_available[:, return_index]
                & future_available[:, return_index]
                & np.isfinite(path)
            )
            components = transition_components(
                current_state=current_state,
                future_state=future_state,
                current_raw=current_raw,
                future_raw=future_raw,
                current_available=current_available,
                future_available=future_available,
                feature_names=features.feature_names,
                entry_path_returns=path,
                node_mask=node_mask,
            )
            rows.append(
                {
                    "split": split,
                    "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                    "target_date": str(pd.Timestamp(features.dates[target_step]).date()),
                    "step": int(step),
                    "horizon": int(horizon),
                    **components,
                }
            )
    return rows


def _score_rows(rows, calibration):
    output = []
    for row in rows:
        scores = score_systemic_components(row, calibration)
        labels = event_labels(row, calibration)
        output.append({**row, **scores, **labels})
    return output


def _subtype_counts(rows):
    names = ("systemic_event", "broad_selloff", "turnover_explosion", "graph_state_shift")
    return {
        name: {
            "count": int(sum(bool(row[name]) for row in rows)),
            "rate": float(np.mean([bool(row[name]) for row in rows])) if rows else float("nan"),
        }
        for name in names
    }


def _horizon_summary(rows):
    components = tuple(
        dict.fromkeys(
            name for family_names in SYSTEMIC_FAMILIES.values() for name in family_names
        )
    )
    result = {
        "rows": len(rows),
        "events": _subtype_counts(rows),
        "systemic_energy": _finite_summary(
            [float(row["systemic_energy"]) for row in rows]
        ),
        "return_concentration": _finite_summary(
            [float(row["return_concentration"]) for row in rows]
        ),
        "components": {
            name: _finite_summary([float(row[name]) for row in rows])
            for name in components
        },
    }
    event_rows = [row for row in rows if row["systemic_event"]]
    result["event_dominance_audit"] = {
        "events": len(event_rows),
        "median_return_concentration": (
            float(np.nanmedian([float(row["return_concentration"]) for row in event_rows]))
            if event_rows
            else float("nan")
        ),
        "max_return_concentration": (
            float(np.nanmax([float(row["return_concentration"]) for row in event_rows]))
            if event_rows
            else float("nan")
        ),
        "median_observed_nodes": (
            float(np.median([int(row["observed_nodes"]) for row in event_rows]))
            if event_rows
            else float("nan")
        ),
    }
    return result


def _top_events(rows, limit: int):
    ranked = sorted(
        [row for row in rows if np.isfinite(float(row["systemic_energy"]))],
        key=lambda row: float(row["systemic_energy"]),
        reverse=True,
    )[: int(limit)]
    keys = (
        "date",
        "target_date",
        "horizon",
        "systemic_energy",
        "family:price_breadth",
        "family:market_risk",
        "family:activity",
        "family:graph_state",
        "market_return",
        "mean_absolute_return",
        "breadth",
        "volume_shock",
        "traded_value_shock",
        "common_state_energy",
        "node_state_median_energy",
        "return_concentration",
        "broad_selloff",
        "turnover_explosion",
        "graph_state_shift",
    )
    return [{key: row[key] for key in keys} for row in ranked]


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_markdown(path: Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# Systemic transition target audit",
        "",
        "This is a post-hoc research target audit. It cannot authorize live orders.",
        "",
        f"- Train end: `{summary['train_end']}`",
        f"- Horizons: `{summary['horizons']}`",
        f"- Live orders allowed: `{summary['live_orders_allowed']}`",
        "",
        "## Out-of-sample event rates",
        "",
    ]
    for horizon, item in summary["horizons_summary"].items():
        events = item["test"]["events"]
        lines.append(
            f"- h{horizon}: systemic={events['systemic_event']['rate']:.3f}, "
            f"selloff={events['broad_selloff']['rate']:.3f}, "
            f"turnover={events['turnover_explosion']['rate']:.3f}, "
            f"state_shift={events['graph_state_shift']['rate']:.3f}"
        )
    lines.extend(["", "## Highest-scoring out-of-sample transitions", ""])
    for row in summary["top_test_events"]:
        lines.append(
            f"- {row['date']} -> {row['target_date']} h{row['horizon']}: "
            f"energy={row['systemic_energy']:.3f}, market={row['market_return']:.4f}, "
            f"breadth={row['breadth']:.3f}, activity={row['family:activity']:.3f}, "
            f"state={row['family:graph_state']:.3f}, concentration={row['return_concentration']:.4f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit broad-market systemic transition targets without model predictions."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--event-quantile", type=float, default=0.90)
    parser.add_argument("--top-events", type=int, default=20)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    horizons = parse_int_list(args.horizons)
    feature_args = evaluator_namespace(args)
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(
        features, checkpoint_args, horizons, int(args.validation_days)
    )
    raw_rows = {
        name: _actual_rows(features, steps, horizons, name)
        for name, steps in splits.items()
    }

    scored_rows: dict[str, list[dict[str, Any]]] = {
        name: [] for name in raw_rows
    }
    calibrations = {}
    horizons_summary = {}
    for horizon in horizons:
        fit = [row for row in raw_rows["fit"] if row["horizon"] == int(horizon)]
        calibration = fit_systemic_calibration(
            fit, event_quantile=float(args.event_quantile)
        )
        calibrations[str(horizon)] = calibration.to_dict()
        horizons_summary[str(horizon)] = {}
        for split in raw_rows:
            selected = [
                row for row in raw_rows[split] if row["horizon"] == int(horizon)
            ]
            scored = _score_rows(selected, calibration)
            scored_rows[split].extend(scored)
            horizons_summary[str(horizon)][split] = _horizon_summary(scored)

    fit_last_target = max(
        int(row["step"]) + int(row["horizon"]) for row in raw_rows["fit"]
    )
    validation_first_step = min(int(row["step"]) for row in raw_rows["validation"])
    all_rows = scored_rows["fit"] + scored_rows["validation"] + scored_rows["test"]
    all_rows.sort(key=lambda row: (int(row["step"]), int(row["horizon"]), row["split"]))
    summary = {
        "status": "complete",
        "role": "posthoc_systemic_transition_target_audit",
        "target_version": SYSTEMIC_TARGET_VERSION,
        "model_dir": str(model_dir),
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "event_quantile": float(args.event_quantile),
        "split_dates": {name: len(steps) for name, steps in splits.items()},
        "leakage_contract": {
            "fit_last_target_step": fit_last_target,
            "validation_first_context_step": validation_first_step,
            "fit_targets_before_validation_context": bool(
                fit_last_target < validation_first_step
            ),
            "calibration_uses_fit_only": True,
            "test_used_for_selection": False,
        },
        "calibrations": calibrations,
        "horizons_summary": horizons_summary,
        "top_test_events": _top_events(scored_rows["test"], int(args.top_events)),
        "selection_status": "exploratory_only_requires_future_confirmation",
        "live_orders_allowed": False,
    }
    _write_csv(output_dir / "daily_systemic_targets.csv", all_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output_dir / "summary.md", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "test_dates": len(splits["test"]),
                "top_event": summary["top_test_events"][0],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
