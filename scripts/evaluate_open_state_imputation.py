from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Mapping, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.audit_systemic_transition_targets import _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    graph_edge_kwargs,
    load_model,
)
from scripts.run_real_backtest import parse_int_list, rollout_steps_for_offset
from stock_v2.graph_jepa import GraphBatch
from stock_v2.real_features import make_real_snapshot


OPEN_KNOWN_EXACT = frozenset(("gap_open",))
OPEN_KNOWN_PREFIXES = ("news_", "fund_", "investor_", "ext_")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def open_known_feature_indices(feature_names: Sequence[str]) -> np.ndarray:
    return np.asarray(
        [
            index
            for index, name in enumerate(feature_names)
            if str(name) in OPEN_KNOWN_EXACT
            or str(name).startswith(OPEN_KNOWN_PREFIXES)
        ],
        dtype=np.int64,
    )


def target_feature_groups(feature_names: Sequence[str]) -> dict[str, np.ndarray]:
    groups: dict[str, list[int]] = {
        "price_trend": [],
        "risk": [],
        "liquidity": [],
        "market_cross_section": [],
        "intraday": [],
        "other_close_state": [],
    }
    known = set(open_known_feature_indices(feature_names).tolist())
    for index, raw_name in enumerate(feature_names):
        if index in known:
            continue
        name = str(raw_name)
        if name == "intraday_return":
            groups["intraday"].append(index)
        elif any(token in name for token in ("volume", "value", "amihud")):
            groups["liquidity"].append(index)
        elif any(
            token in name
            for token in ("volatility", "range", "beta", "corr")
        ):
            groups["risk"].append(index)
        elif name.startswith("market_") or name.startswith("relative_") or name.startswith("cs_"):
            groups["market_cross_section"].append(index)
        elif any(
            token in name
            for token in ("return", "ma", "drawdown", "breakout", "position")
        ):
            groups["price_trend"].append(index)
        else:
            groups["other_close_state"].append(index)
    result = {
        name: np.asarray(indices, dtype=np.int64)
        for name, indices in groups.items()
        if indices
    }
    covered = np.concatenate(list(result.values())) if result else np.empty(0, dtype=np.int64)
    expected = np.asarray(
        [index for index in range(len(feature_names)) if index not in known],
        dtype=np.int64,
    )
    if sorted(covered.tolist()) != expected.tolist():
        raise ValueError("open target feature groups are not an exact partition")
    return result


def build_open_observations(
    previous: np.ndarray,
    previous_available: np.ndarray,
    current: np.ndarray,
    current_available: np.ndarray,
    forecast: np.ndarray,
    *,
    stock_count: int,
    known_feature_indices: Sequence[int],
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    previous = np.asarray(previous, dtype=np.float32)
    previous_available = np.asarray(previous_available, dtype=bool)
    current = np.asarray(current, dtype=np.float32)
    current_available = np.asarray(current_available, dtype=bool)
    forecast = np.asarray(forecast, dtype=np.float32)
    if not (
        previous.shape
        == previous_available.shape
        == current.shape
        == current_available.shape
        == forecast.shape
    ):
        raise ValueError("open observation arrays must have identical shapes")
    if previous.ndim != 2 or not 0 < int(stock_count) <= previous.shape[0]:
        raise ValueError("open observations require [node, feature] arrays")
    known = np.asarray(known_feature_indices, dtype=np.int64)
    if known.size and (known.min() < 0 or known.max() >= previous.shape[1]):
        raise ValueError("known feature indices exceed the feature axis")

    known_only = np.zeros_like(previous)
    known_only_mask = np.zeros_like(previous_available)
    stale = previous.copy()
    stale_mask = previous_available.copy()
    forecast_state = forecast.copy()
    forecast_mask = previous_available.copy()

    if known.size:
        stock_rows = np.arange(int(stock_count), dtype=np.int64)
        stock_known = np.ix_(stock_rows, known)
        known_values = current[stock_known]
        known_available = current_available[stock_known]
        for values, mask in (
            (known_only, known_only_mask),
            (stale, stale_mask),
            (forecast_state, forecast_mask),
        ):
            selected = values[stock_known].copy()
            selected_mask = mask[stock_known].copy()
            selected[known_available] = known_values[known_available]
            selected_mask[known_available] = True
            values[stock_known] = selected
            mask[stock_known] = selected_mask

    if int(stock_count) < previous.shape[0]:
        external_rows = slice(int(stock_count), previous.shape[0])
        for values, mask in (
            (known_only, known_only_mask),
            (stale, stale_mask),
            (forecast_state, forecast_mask),
        ):
            available = current_available[external_rows]
            external = values[external_rows].copy()
            external[available] = current[external_rows][available]
            values[external_rows] = external
            mask[external_rows] = available

    return {
        "open_known_only": (known_only, known_only_mask),
        "stale_plus_open": (stale, stale_mask),
        "forecast_plus_open": (forecast_state, forecast_mask),
    }


def _batch_with_observation(
    base: GraphBatch,
    values: np.ndarray,
    observed: np.ndarray,
) -> GraphBatch:
    values_tensor = torch.as_tensor(values, dtype=base.node_features.dtype)
    observed_tensor = torch.as_tensor(observed, dtype=base.feature_mask.dtype)
    return GraphBatch(
        node_features=values_tensor,
        feature_mask=observed_tensor,
        edge_index=base.edge_index,
        edge_weight=base.edge_weight,
        available_mask=observed_tensor,
        supervision_node_mask=base.supervision_node_mask,
    )


def _metric_row(prediction: np.ndarray, target: np.ndarray, mask: np.ndarray) -> dict[str, float | int]:
    valid = np.asarray(mask, dtype=bool) & np.isfinite(prediction) & np.isfinite(target)
    if not valid.any():
        return {"cells": 0, "mae": float("nan"), "mse": float("nan")}
    error = prediction[valid].astype(np.float64) - target[valid].astype(np.float64)
    return {
        "cells": int(valid.sum()),
        "mae": float(np.abs(error).mean()),
        "mse": float(np.square(error).mean()),
    }


def _aggregate(rows: Sequence[Mapping[str, Any]], variants: Sequence[str], groups: Sequence[str]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for variant in variants:
        result[variant] = {}
        for group in groups:
            selected = [row for row in rows if row["variant"] == variant and row["group"] == group]
            cells = sum(int(row["cells"]) for row in selected)
            if cells == 0:
                continue
            mse = sum(float(row["mse"]) * int(row["cells"]) for row in selected) / cells
            mae = sum(float(row["mae"]) * int(row["cells"]) for row in selected) / cells
            result[variant][group] = {"dates": len(selected), "cells": cells, "mae": mae, "mse": mse}
    persistence_mse = float(result["persistence"]["all_close_state"]["mse"])
    temporal_mse = float(result["temporal_forecast"]["all_close_state"]["mse"])
    for variant in variants:
        metrics = result[variant].get("all_close_state")
        if metrics is None:
            continue
        mse = float(metrics["mse"])
        metrics["skill_vs_persistence"] = 1.0 - mse / persistence_mse
        metrics["skill_vs_temporal_forecast"] = 1.0 - mse / temporal_mse
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate causal Korean-open current-state imputation."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--edge-cache-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    horizons = tuple(parse_int_list(args.horizons))
    model, checkpoint = load_model(model_dir, device)
    checkpoint_args = dict(checkpoint.get("args", {}))
    feature_args = evaluator_namespace(args)
    feature_args.horizons = args.horizons
    feature_args.edge_cache_workers = max(0, int(args.edge_cache_workers))
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(features, checkpoint_args, horizons, int(args.validation_days))
    steps = np.unique(np.concatenate((splits["validation"], splits["test"]))).astype(np.int64)
    if int(steps.min()) < 1:
        raise ValueError("open evaluation requires a previous trading day")
    edge_steps = np.unique(steps - 1)
    edge_cache = build_evaluation_edge_cache(
        features, edge_steps, checkpoint_args, feature_args
    )
    known_indices = open_known_feature_indices(features.feature_names)
    groups = target_feature_groups(features.feature_names)
    all_target_indices = np.concatenate(list(groups.values()))
    split_lookup = {
        int(step): split for split, values in splits.items() for step in values
    }
    variants = (
        "persistence",
        "temporal_forecast",
        "open_known_only",
        "stale_plus_open",
        "forecast_plus_open",
    )
    rows: list[dict[str, Any]] = []
    edge_window = int(checkpoint_args.get("edge_window", 60))
    edge_top_k = int(checkpoint_args.get("edge_top_k", 6))
    min_abs_corr = float(checkpoint_args.get("min_abs_corr", 0.2))
    rollout_args = argparse.Namespace(**checkpoint_args)

    model.eval()
    with torch.inference_mode():
        for ordinal, step in enumerate(steps):
            previous_step = int(step) - 1
            base = make_real_snapshot(
                features,
                step=previous_step,
                full_observation=True,
                edge_window=edge_window,
                top_k=edge_top_k,
                min_abs_corr=min_abs_corr,
                **graph_edge_kwargs(checkpoint_args, feature_args),
                edge_cache=edge_cache,
            )
            base_device = base.to(device)
            context = model.encode_temporal_context(base_device)
            rollout_step = rollout_steps_for_offset(rollout_args, 1)
            latent = model.rollout_latent(context, steps=rollout_step)
            temporal = model.predict_temporal_state(
                base_device,
                latent,
                rollout_steps=rollout_step,
                z_context=context,
            ).float().cpu().numpy()

            observations = build_open_observations(
                features.features[previous_step],
                features.available_mask[previous_step] > 0.5,
                features.features[int(step)],
                features.available_mask[int(step)] > 0.5,
                temporal,
                stock_count=features.tradable_count,
                known_feature_indices=known_indices,
            )
            predictions = {
                "persistence": features.features[previous_step],
                "temporal_forecast": temporal,
            }
            for name, (values, observed) in observations.items():
                batch = _batch_with_observation(base, values, observed).to(device)
                predictions[name] = model.infer_unobserved_state(batch).float().cpu().numpy()

            target = features.features[int(step), : features.tradable_count]
            common_available = (
                (features.available_mask[int(step), : features.tradable_count] > 0.5)
                & (features.available_mask[previous_step, : features.tradable_count] > 0.5)
            )
            for variant in variants:
                prediction = predictions[variant][: features.tradable_count]
                for group_name, indices in (
                    ("all_close_state", all_target_indices),
                    *groups.items(),
                ):
                    metric = _metric_row(
                        prediction[:, indices],
                        target[:, indices],
                        common_available[:, indices],
                    )
                    rows.append(
                        {
                            "split": split_lookup[int(step)],
                            "date": str(features.dates[int(step)].date()),
                            "step": int(step),
                            "variant": variant,
                            "group": group_name,
                            **metric,
                        }
                    )
            if (ordinal + 1) % 25 == 0 or ordinal + 1 == len(steps):
                print(f"open-state evaluation: {ordinal + 1}/{len(steps)}", flush=True)

    summaries = {}
    group_names = ("all_close_state", *groups)
    for split in ("validation", "test"):
        selected = [row for row in rows if row["split"] == split]
        summaries[split] = _aggregate(selected, variants, group_names)
    open_variants = ("open_known_only", "stale_plus_open", "forecast_plus_open")
    selected_variant = min(
        open_variants,
        key=lambda name: float(summaries["validation"][name]["all_close_state"]["mse"]),
    )
    selected_test = summaries["test"][selected_variant]["all_close_state"]
    payload = {
        "schema_version": 1,
        "status": "complete",
        "role": "research_only_korean_open_current_state_imputation",
        "model_dir": str(model_dir),
        "checkpoint_sha256": sha256_file(model_dir / "graph_jepa_real.pt"),
        "stocks": int(features.tradable_count),
        "nodes": int(features.node_count),
        "features": len(features.feature_names),
        "known_current_features": [features.feature_names[int(index)] for index in known_indices],
        "target_feature_groups": {
            name: [features.feature_names[int(index)] for index in indices]
            for name, indices in groups.items()
        },
        "split_dates": {name: int(len(values)) for name, values in splits.items()},
        "validation_selected_open_variant": selected_variant,
        "selected_test_metrics": selected_test,
        "open_imputation_gate": {
            "passed": (
                float(selected_test["skill_vs_persistence"]) > 0.0
                and float(selected_test["skill_vs_temporal_forecast"]) > 0.0
            ),
            "requirements": (
                "validation-selected open variant must beat persistence and the "
                "previous-close JEPA h1 forecast on the untouched test split"
            ),
        },
        "metrics": summaries,
        "causal_contract": {
            "current_close_volume_high_low_features_in_input": False,
            "current_gap_open_in_input": True,
            "lagged_news_fundamental_investor_external_in_input": True,
            "graph_edges_end_at_previous_trading_day": True,
            "test_used_for_selection": False,
        },
        "live_orders_allowed": False,
    }
    _write_csv(output_dir / "daily_metrics.csv", rows)
    (output_dir / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "EXPERIMENT_COMPLETE").touch()
    print(
        json.dumps(
            {
                "selected_variant": selected_variant,
                "selected_test_metrics": selected_test,
                "gate": payload["open_imputation_gate"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
