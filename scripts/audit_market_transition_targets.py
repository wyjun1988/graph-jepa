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

from scripts.audit_systemic_transition_targets import _finite_summary, _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.run_real_backtest import parse_int_list
from stock_v2.market_transition import (
    EVENT_NAMES,
    MARKET_TRANSITION_FAMILIES,
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
    fit_market_transition_calibration,
    market_transition_components,
    market_transition_labels,
    normalized_market_transition_impact,
    score_market_transition,
)


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
            rows.append(
                {
                    "split": split,
                    "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                    "target_date": str(pd.Timestamp(features.dates[target_step]).date()),
                    "step": int(step),
                    "horizon": int(horizon),
                    **market_transition_components(
                        current_state=current_state,
                        future_state=future_state,
                        current_raw=current_raw,
                        future_raw=future_raw,
                        current_available=current_available,
                        future_available=future_available,
                        feature_names=features.feature_names,
                        entry_path_returns=path,
                        node_mask=node_mask,
                    ),
                }
            )
    return rows


def _score_rows(rows, calibration):
    return [
        {
            **row,
            **score_market_transition(row, calibration),
            **normalized_market_transition_impact(row, calibration),
            **market_transition_labels(row, calibration),
        }
        for row in rows
    ]


def _event_counts(rows):
    return {
        name: {
            "count": int(sum(bool(row[name]) for row in rows)),
            "rate": float(np.mean([bool(row[name]) for row in rows]))
            if rows
            else float("nan"),
        }
        for name in EVENT_NAMES
    }


def _family_overlap(rows):
    subtype_names = (
        "price_transition",
        "activity_transition",
        "node_state_transition",
        "topology_transition",
    )
    event_rows = [row for row in rows if row["systemic_event"]]
    overlap = np.asarray(
        [sum(bool(row[name]) for name in subtype_names) for row in event_rows],
        dtype=np.int64,
    )
    return {
        "events": len(event_rows),
        "exactly_one_family": int(np.sum(overlap == 1)),
        "multiple_families": int(np.sum(overlap > 1)),
        "zero_families": int(np.sum(overlap == 0)),
    }


def _horizon_summary(rows):
    component_names = tuple(
        dict.fromkeys(
            name
            for family_names in MARKET_TRANSITION_FAMILIES.values()
            for name in family_names
        )
    )
    event_rows = [row for row in rows if row["systemic_event"]]
    family_dominance = {name: 0 for name in MARKET_TRANSITION_FAMILIES}
    for row in event_rows:
        dominant = max(
            MARKET_TRANSITION_FAMILIES,
            key=lambda name: float(row[f"normalized_family:{name}"]),
        )
        family_dominance[dominant] += 1
    return {
        "rows": len(rows),
        "events": _event_counts(rows),
        "event_family_overlap": _family_overlap(rows),
        "event_family_dominance": family_dominance,
        "systemic_energy": _finite_summary(
            [float(row["systemic_energy"]) for row in rows]
        ),
        "systemic_impact": _finite_summary(
            [float(row["systemic_impact"]) for row in rows]
        ),
        "return_concentration": _finite_summary(
            [float(row["return_concentration"]) for row in rows]
        ),
        "event_return_concentration": _finite_summary(
            [float(row["return_concentration"]) for row in event_rows]
        ),
        "components": {
            name: _finite_summary([float(row[name]) for row in rows])
            for name in component_names
        },
    }


def _top_events(rows, limit: int):
    ranked = sorted(
        [row for row in rows if np.isfinite(float(row["systemic_impact"]))],
        key=lambda row: float(row["systemic_impact"]),
        reverse=True,
    )[: int(limit)]
    keys = (
        "date",
        "target_date",
        "horizon",
        "systemic_impact",
        "systemic_energy",
        "normalized_family:price_co_movement",
        "normalized_family:market_activity",
        "normalized_family:node_state",
        "normalized_family:topology",
        "family:price_co_movement",
        "family:market_activity",
        "family:node_state",
        "family:topology",
        "market_return",
        "median_return",
        "return_breadth",
        "volume_median_z",
        "volume_participation_z1",
        "value_median_z",
        "value_participation_z1",
        "node_state_median_energy",
        "state_change_participation",
        "market_corr_change",
        "return_concentration",
        *EVENT_NAMES,
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
        "# Market-wide transition target audit",
        "",
        "Individual-stock moves are diagnostics only. The target requires broad",
        "price co-movement, market activity, node-state movement, or topology change.",
        "",
        f"- Target: `{summary['target_version']}`",
        f"- Train end: `{summary['train_end']}`",
        f"- Live orders allowed: `{summary['live_orders_allowed']}`",
        "",
        "## Out-of-sample event rates",
        "",
    ]
    for horizon, item in summary["horizons_summary"].items():
        event = item["test"]["events"]
        lines.append(
            f"- h{horizon}: any={event['systemic_event']['rate']:.3f}, "
            f"price={event['price_transition']['rate']:.3f}, "
            f"activity={event['activity_transition']['rate']:.3f}, "
            f"state={event['node_state_transition']['rate']:.3f}, "
            f"topology={event['topology_transition']['rate']:.3f}"
        )
    lines.extend(["", "## Highest-scoring test transitions", ""])
    for row in summary["top_test_events"]:
        lines.append(
            f"- {row['date']} -> {row['target_date']} h{row['horizon']}: "
            f"impact={row['systemic_impact']:.3f}, "
            f"price={row['normalized_family:price_co_movement']:.3f}, "
            f"activity={row['normalized_family:market_activity']:.3f}, "
            f"state={row['normalized_family:node_state']:.3f}, "
            f"topology={row['normalized_family:topology']:.3f}"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit robust market-wide transition targets without predictions."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--component-scale-quantile", type=float, default=0.90)
    parser.add_argument("--family-event-quantile", type=float, default=0.95)
    parser.add_argument("--top-events", type=int, default=30)
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
    features, checkpoint_args = build_features_from_ckpt(
        checkpoint, evaluator_namespace(args)
    )
    splits = _split_steps(features, checkpoint_args, horizons, int(args.validation_days))
    raw_rows = {
        name: _actual_rows(features, steps, horizons, name)
        for name, steps in splits.items()
    }

    scored_rows = {name: [] for name in raw_rows}
    calibrations = {}
    horizons_summary = {}
    for horizon in horizons:
        fit = [row for row in raw_rows["fit"] if row["horizon"] == int(horizon)]
        calibration = fit_market_transition_calibration(
            fit,
            component_scale_quantile=float(args.component_scale_quantile),
            family_event_quantile=float(args.family_event_quantile),
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
    validation_first = min(int(row["step"]) for row in raw_rows["validation"])
    all_rows = scored_rows["fit"] + scored_rows["validation"] + scored_rows["test"]
    all_rows.sort(key=lambda row: (int(row["step"]), int(row["horizon"]), row["split"]))
    summary = {
        "status": "complete",
        "role": "posthoc_market_transition_target_audit",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "model_dir": str(model_dir),
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "component_scale_quantile": float(args.component_scale_quantile),
        "family_event_quantile": float(args.family_event_quantile),
        "split_dates": {name: len(steps) for name, steps in splits.items()},
        "leakage_contract": {
            "fit_last_target_step": fit_last_target,
            "validation_first_context_step": validation_first,
            "fit_targets_before_validation_context": bool(
                fit_last_target < validation_first
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
    _write_csv(output_dir / "daily_market_transition_targets.csv", all_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_markdown(output_dir / "summary.md", summary)
    print(
        json.dumps(
            {
                "status": "complete",
                "target_version": MARKET_TRANSITION_TARGET_VERSION,
                "test_dates": len(splits["test"]),
                "top_event": summary["top_test_events"][0],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
