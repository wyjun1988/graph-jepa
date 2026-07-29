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
import torch

from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.evaluate_node_prediction import build_features_from_ckpt
from stock_v2.market_transition import MARKET_TRANSITION_TARGET_VERSION
from stock_v2.systemic_transition import (
    DEFAULT_SYSTEMIC_STATE_FEATURES,
    systemic_state_feature_indices,
)


def state_feature_contributions(
    current_state: np.ndarray,
    future_state: np.ndarray,
    current_available: np.ndarray,
    future_available: np.ndarray,
    node_mask: np.ndarray,
    feature_names: Sequence[str],
    *,
    min_nodes: int = 20,
    change_threshold: float = 0.25,
) -> dict[str, Any]:
    indices, selected_names = systemic_state_feature_indices(
        feature_names, DEFAULT_SYSTEMIC_STATE_FEATURES
    )
    current = np.asarray(current_state, dtype=np.float64)[:, indices]
    future = np.asarray(future_state, dtype=np.float64)[:, indices]
    valid = (
        np.asarray(node_mask, dtype=bool)[:, None]
        & np.asarray(current_available, dtype=bool)[:, indices]
        & np.asarray(future_available, dtype=bool)[:, indices]
        & np.isfinite(current)
        & np.isfinite(future)
    )
    absolute_delta = np.abs(future - current)
    counts = valid.sum(axis=0)
    median_absolute = np.full(len(indices), np.nan, dtype=np.float64)
    q75_absolute = np.full(len(indices), np.nan, dtype=np.float64)
    participation = np.full(len(indices), np.nan, dtype=np.float64)
    for index in range(len(indices)):
        values = absolute_delta[valid[:, index], index]
        if values.size < int(min_nodes):
            continue
        median_absolute[index] = float(np.median(values))
        q75_absolute[index] = float(np.quantile(values, 0.75))
        participation[index] = float(np.mean(values >= float(change_threshold)))
    usable = np.isfinite(median_absolute)
    energy = np.where(usable, np.square(median_absolute), 0.0)
    total = float(energy.sum())
    shares = energy / total if total > 1e-12 else np.zeros_like(energy)
    positive = shares > 0.0
    effective_features = (
        float(1.0 / np.square(shares[positive]).sum())
        if positive.any()
        else 0.0
    )
    dominant_index = int(np.argmax(shares)) if shares.size else -1
    return {
        "feature_names": list(selected_names),
        "observed_nodes": int(np.asarray(node_mask, dtype=bool).sum()),
        "feature_observation_count": counts.astype(int).tolist(),
        "median_absolute_delta": median_absolute.tolist(),
        "q75_absolute_delta": q75_absolute.tolist(),
        "change_participation": participation.tolist(),
        "energy_share": shares.tolist(),
        "dominant_feature": (
            str(selected_names[dominant_index]) if dominant_index >= 0 else None
        ),
        "top_feature_share": (
            float(shares[dominant_index]) if dominant_index >= 0 else 0.0
        ),
        "effective_feature_count": effective_features,
        "feature_breadth_at_threshold": float(
            np.mean(median_absolute[usable] >= float(change_threshold))
        )
        if usable.any()
        else float("nan"),
    }


def _finite_summary(values: Sequence[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    if array.size == 0:
        return {"count": 0, "median": float("nan"), "q90": float("nan")}
    return {
        "count": int(array.size),
        "median": float(np.median(array)),
        "q90": float(np.quantile(array, 0.90)),
    }


def _aggregate(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"events": 0}
    names = list(records[0]["feature_names"])
    share_matrix = np.asarray([row["energy_share"] for row in records], dtype=np.float64)
    dominant_counts = {name: 0 for name in names}
    for row in records:
        dominant_counts[str(row["dominant_feature"])] += 1
    investor_indices = [
        index for index, name in enumerate(names) if name.startswith("investor_")
    ]
    investor_share = (
        share_matrix[:, investor_indices].sum(axis=1)
        if investor_indices
        else np.zeros(len(records), dtype=np.float64)
    )
    return {
        "events": len(records),
        "top_feature_share": _finite_summary(
            [float(row["top_feature_share"]) for row in records]
        ),
        "effective_feature_count": _finite_summary(
            [float(row["effective_feature_count"]) for row in records]
        ),
        "feature_breadth_at_threshold": _finite_summary(
            [float(row["feature_breadth_at_threshold"]) for row in records]
        ),
        "investor_feature_energy_share": _finite_summary(investor_share.tolist()),
        "dominant_feature_counts": dict(
            sorted(dominant_counts.items(), key=lambda item: item[1], reverse=True)
        ),
        "median_energy_share_by_feature": dict(
            sorted(
                zip(names, np.median(share_matrix, axis=0).astype(float)),
                key=lambda item: item[1],
                reverse=True,
            )
        ),
    }


def _write_csv(path: Path, records: Sequence[Mapping[str, Any]]) -> None:
    if not records:
        return
    names = list(records[0]["feature_names"])
    rows = []
    for record in records:
        row = {
            key: record[key]
            for key in (
                "split",
                "date",
                "target_date",
                "step",
                "horizon",
                "systemic_impact",
                "dominant_family",
                "dominant_feature",
                "top_feature_share",
                "effective_feature_count",
                "feature_breadth_at_threshold",
            )
        }
        for index, name in enumerate(names):
            row[f"energy_share:{name}"] = record["energy_share"][index]
            row[f"median_absolute_delta:{name}"] = record[
                "median_absolute_delta"
            ][index]
        rows.append(row)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit whether broad node-state targets depend on one feature."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--target-audit-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--major-event-quantile", type=float, default=0.90)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    target_root = Path(args.target_audit_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = torch.load(
        model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    features, _checkpoint_args = build_features_from_ckpt(
        checkpoint, evaluator_namespace(args)
    )
    with (target_root / "daily_market_transition_targets.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        target_rows = list(csv.DictReader(handle))
    fit_impact = np.asarray(
        [float(row["systemic_impact"]) for row in target_rows if row["split"] == "fit"],
        dtype=np.float64,
    )
    major_threshold = float(
        np.quantile(fit_impact[np.isfinite(fit_impact)], args.major_event_quantile)
    )
    candidates = [
        row
        for row in target_rows
        if row["split"] == "test"
        and float(row["systemic_impact"]) >= major_threshold
    ]
    stock_count = int(features.tradable_count)
    return_index = features.feature_names.index("return_1d")
    family_names = ("price_co_movement", "market_activity", "node_state", "topology")
    records = []
    for source in candidates:
        step = int(source["step"])
        horizon = int(source["horizon"])
        target_step = step + horizon
        current_available = features.available_mask[step, :stock_count] > 0.5
        future_available = features.available_mask[target_step, :stock_count] > 0.5
        path = np.asarray(
            features.target_return_paths[horizon][step, :stock_count],
            dtype=np.float64,
        )
        node_mask = (
            current_available[:, return_index]
            & future_available[:, return_index]
            & np.isfinite(path)
        )
        contribution = state_feature_contributions(
            features.features[step, :stock_count],
            features.features[target_step, :stock_count],
            current_available,
            future_available,
            node_mask,
            features.feature_names,
        )
        dominant_family = max(
            family_names,
            key=lambda name: float(source[f"normalized_family:{name}"]),
        )
        records.append(
            {
                "split": source["split"],
                "date": source["date"],
                "target_date": source["target_date"],
                "step": step,
                "horizon": horizon,
                "systemic_impact": float(source["systemic_impact"]),
                "dominant_family": dominant_family,
                **contribution,
            }
        )
    node_records = [row for row in records if row["dominant_family"] == "node_state"]
    summary = {
        "status": "complete",
        "role": "target_semantics_audit",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "major_event_quantile": float(args.major_event_quantile),
        "fit_major_event_threshold": major_threshold,
        "all_major_events": _aggregate(records),
        "node_state_dominant_major_events": _aggregate(node_records),
        "test_used_for_selection": False,
        "live_orders_allowed": False,
    }
    _write_csv(output_dir / "major_event_state_feature_contributions.csv", records)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
