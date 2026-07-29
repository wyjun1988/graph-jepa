from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_direct_baselines import evaluator_namespace, newey_west_mean
from scripts.benchmark_impact_trajectory_head import ImpactTrajectoryHead
from scripts.benchmark_latent_trajectory_path_head import (
    HORIZON_WEIGHTS,
    batch_targets,
    blend_path_scores,
    checkpoint_sha256,
    latent_trajectories,
    snapshot_batch,
    state_entry_path_scores,
    stock_rows,
    top_liquidity_mask,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import date_indices, parse_int_list


STRATEGIES = ("impact_only", "impact_then_confidence", "joint_75_25")
FIXED_K_METRICS = (
    "impact_precision_at_k",
    "impact_recall_at_k",
    "impact_lift_at_k",
    "direction_accuracy_at_k",
    "captured_direction_accuracy_at_k",
    "joint_correct_precision_at_k",
    "selected_mean_abs_return_at_k",
    "magnitude_lift_at_k",
    "realized_tail_mass_recall_at_k",
    "captured_impact_weighted_direction_accuracy_at_k",
    "signed_realized_tail_mass_capture_at_k",
)


def rank_percentile(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    result = np.full(len(values), np.nan, dtype=np.float64)
    selected = np.flatnonzero(valid & np.isfinite(values))
    if not len(selected):
        return result
    order = np.argsort(values[selected], kind="stable")
    if len(selected) == 1:
        result[selected[order]] = 1.0
    else:
        result[selected[order]] = np.arange(len(selected), dtype=np.float64) / (
            len(selected) - 1
        )
    return result


def select_fixed_k(
    signed_score: np.ndarray,
    impact_score: np.ndarray,
    valid: np.ndarray,
    count: int,
    strategy: str,
    candidate_fraction: float = 0.10,
) -> np.ndarray:
    eligible = np.flatnonzero(
        valid & np.isfinite(signed_score) & np.isfinite(impact_score)
    )
    if not len(eligible):
        return np.asarray([], dtype=np.int64)
    count = min(max(1, int(count)), len(eligible))
    if strategy == "impact_only":
        order = np.argsort(impact_score[eligible], kind="stable")
        return eligible[order[-count:]]
    if strategy == "impact_then_confidence":
        candidate_count = max(
            count, int(math.ceil(len(eligible) * float(candidate_fraction)))
        )
        impact_order = np.argsort(impact_score[eligible], kind="stable")
        candidates = eligible[impact_order[-candidate_count:]]
        confidence_order = np.argsort(np.abs(signed_score[candidates]), kind="stable")
        return candidates[confidence_order[-count:]]
    if strategy == "joint_75_25":
        impact_rank = rank_percentile(impact_score, valid)
        confidence_rank = rank_percentile(np.abs(signed_score), valid)
        joint = 0.75 * impact_rank + 0.25 * confidence_rank
        order = np.argsort(joint[eligible], kind="stable")
        return eligible[order[-count:]]
    raise ValueError(f"unknown fixed-k strategy: {strategy}")


def fixed_k_metric_row(
    signed_score: np.ndarray,
    impact_score: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    count: int,
    strategy: str,
    realized_fraction: float = 0.10,
) -> dict[str, float]:
    eligible_mask = (
        valid
        & np.isfinite(signed_score)
        & np.isfinite(impact_score)
        & np.isfinite(target)
    )
    eligible = np.flatnonzero(eligible_mask)
    if len(eligible) < 10:
        return {name: float("nan") for name in FIXED_K_METRICS}
    selected = select_fixed_k(
        signed_score,
        impact_score,
        eligible_mask,
        count,
        strategy,
        candidate_fraction=realized_fraction,
    )
    return fixed_k_metrics_for_selected(
        signed_score,
        target,
        eligible_mask,
        selected,
        realized_fraction=realized_fraction,
    )


def fixed_k_metrics_for_selected(
    signed_score: np.ndarray,
    target: np.ndarray,
    eligible_mask: np.ndarray,
    selected: np.ndarray,
    realized_fraction: float = 0.10,
) -> dict[str, float]:
    eligible = np.flatnonzero(
        eligible_mask & np.isfinite(signed_score) & np.isfinite(target)
    )
    selected = np.asarray(selected, dtype=np.int64)
    if len(eligible) < 10 or not len(selected):
        return {name: float("nan") for name in FIXED_K_METRICS}
    selected = selected[eligible_mask[selected]]
    selected = selected[
        np.isfinite(signed_score[selected]) & np.isfinite(target[selected])
    ]
    if not len(selected):
        return {name: float("nan") for name in FIXED_K_METRICS}
    realized_count = max(1, int(math.ceil(len(eligible) * float(realized_fraction))))
    realized_order = np.argsort(np.abs(target[eligible]), kind="stable")
    realized = eligible[realized_order[-realized_count:]]
    realized_mask = np.zeros(len(target), dtype=bool)
    realized_mask[realized] = True
    captured = selected[realized_mask[selected]]
    direction_valid = target[selected] != 0.0
    selected_correct = np.zeros(len(selected), dtype=bool)
    selected_correct[direction_valid] = (
        np.sign(signed_score[selected][direction_valid])
        == np.sign(target[selected][direction_valid])
    )
    captured_direction_valid = target[captured] != 0.0
    captured_accuracy = (
        float(
            np.mean(
                np.sign(signed_score[captured][captured_direction_valid])
                == np.sign(target[captured][captured_direction_valid])
            )
        )
        if captured_direction_valid.any()
        else float("nan")
    )
    joint_correct = int(
        np.sum(realized_mask[selected] & selected_correct)
    )
    universe_mean = float(np.mean(np.abs(target[eligible])))
    selected_mean = float(np.mean(np.abs(target[selected])))
    realized_mass = float(np.sum(np.abs(target[realized])))
    captured_mass = float(np.sum(np.abs(target[captured])))
    captured_weights = np.abs(target[captured][captured_direction_valid])
    captured_correct = (
        np.sign(signed_score[captured][captured_direction_valid])
        == np.sign(target[captured][captured_direction_valid])
    )
    captured_weight_sum = float(captured_weights.sum())
    weighted_captured_accuracy = (
        float(np.sum(captured_weights * captured_correct) / captured_weight_sum)
        if captured_weight_sum > 0.0
        else float("nan")
    )
    direction_alignment = np.where(captured_correct, 1.0, -1.0)
    signed_captured_mass = float(np.sum(captured_weights * direction_alignment))
    return {
        "impact_precision_at_k": float(len(captured) / len(selected)),
        "impact_recall_at_k": float(len(captured) / realized_count),
        "impact_lift_at_k": float(len(captured) / len(selected) / realized_fraction),
        "direction_accuracy_at_k": (
            float(np.mean(selected_correct[direction_valid]))
            if direction_valid.any()
            else float("nan")
        ),
        "captured_direction_accuracy_at_k": captured_accuracy,
        "joint_correct_precision_at_k": float(joint_correct / len(selected)),
        "selected_mean_abs_return_at_k": selected_mean,
        "magnitude_lift_at_k": (
            selected_mean / universe_mean if universe_mean > 0.0 else float("nan")
        ),
        "realized_tail_mass_recall_at_k": (
            captured_mass / realized_mass if realized_mass > 0.0 else float("nan")
        ),
        "captured_impact_weighted_direction_accuracy_at_k": (
            weighted_captured_accuracy
        ),
        "signed_realized_tail_mass_capture_at_k": (
            signed_captured_mass / realized_mass
            if realized_mass > 0.0
            else float("nan")
        ),
    }


def aggregate_horizon_selection_scores(
    signed_scores: dict[int, np.ndarray],
    impact_scores: dict[int, np.ndarray],
    valid: np.ndarray,
    horizons: Sequence[int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    common_valid = np.asarray(valid, dtype=bool).copy()
    for horizon in horizons:
        common_valid &= np.isfinite(signed_scores[int(horizon)])
        common_valid &= np.isfinite(impact_scores[int(horizon)])
    aggregate_impact = np.full(len(valid), np.nan, dtype=np.float64)
    aggregate_confidence = np.full(len(valid), np.nan, dtype=np.float64)
    if not common_valid.any():
        return aggregate_confidence, aggregate_impact, common_valid
    impact_total = np.zeros(len(valid), dtype=np.float64)
    confidence_total = np.zeros(len(valid), dtype=np.float64)
    weight_sum = 0.0
    for horizon in horizons:
        weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
        impact_total += weight * np.nan_to_num(
            rank_percentile(impact_scores[int(horizon)], common_valid), nan=0.0
        )
        confidence_total += weight * np.nan_to_num(
            rank_percentile(np.abs(signed_scores[int(horizon)]), common_valid),
            nan=0.0,
        )
        weight_sum += weight
    aggregate_impact[common_valid] = impact_total[common_valid] / weight_sum
    aggregate_confidence[common_valid] = confidence_total[common_valid] / weight_sum
    return aggregate_confidence, aggregate_impact, common_valid


def score_steps(
    model,
    head,
    features,
    steps,
    horizons,
    ckpt_args,
    cli_args,
    edge_cache,
    device,
    batch_size,
    liquidity_top_k,
    latent_weight,
    counts,
) -> list[dict[str, Any]]:
    head.eval()
    rows: list[dict[str, Any]] = []
    for start in range(0, len(steps), batch_size):
        selected_steps = np.asarray(steps[start : start + batch_size], dtype=np.int64)
        batch = snapshot_batch(features, selected_steps, ckpt_args, cli_args, edge_cache, device)
        context, predicted = latent_trajectories(model, batch, horizons, ckpt_args)
        stock_rows_tensor, groups = stock_rows(
            len(selected_steps), features.node_count, features.tradable_count, device
        )
        stock_context = context[stock_rows_tensor]
        base_scores = state_entry_path_scores(
            model,
            batch,
            context,
            predicted,
            horizons,
            ckpt_args,
            stock_rows_tensor,
        )
        targets, _ = batch_targets(features, selected_steps, horizons, liquidity_top_k, device)
        predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        with torch.no_grad():
            for horizon in horizons:
                latent_signed, impact_logit = head(
                    stock_context,
                    predicted[int(horizon)][stock_rows_tensor],
                    int(horizon),
                )
                signed = blend_path_scores(
                    base_scores[int(horizon)],
                    latent_signed,
                    targets[int(horizon)],
                    groups,
                    latent_weight,
                )
                predictions[int(horizon)] = (
                    signed.float().cpu().numpy().reshape(
                        len(selected_steps), features.tradable_count
                    ),
                    impact_logit.float().cpu().numpy().reshape(
                        len(selected_steps), features.tradable_count
                    ),
                )
        liquidity_index = features.feature_names.index("value_ma20_log")
        for position, step in enumerate(selected_steps):
            date = str(pd.Timestamp(features.dates[int(step)]).date())
            liquidity = features.raw_features[
                int(step), : features.tradable_count, liquidity_index
            ]
            scopes = {
                "all": np.ones(features.tradable_count, dtype=bool),
                "top300": top_liquidity_mask(liquidity, liquidity_top_k),
            }
            date_targets = {
                int(horizon): np.asarray(
                    features.target_return_paths[int(horizon)][
                        int(step), : features.tradable_count
                    ],
                    dtype=np.float64,
                )
                for horizon in horizons
            }
            signed_by_horizon = {
                int(horizon): predictions[int(horizon)][0][position]
                for horizon in horizons
            }
            impact_by_horizon = {
                int(horizon): predictions[int(horizon)][1][position]
                for horizon in horizons
            }
            for horizon in horizons:
                target = date_targets[int(horizon)]
                signed, impact = predictions[int(horizon)]
                for scope, valid in scopes.items():
                    for count in counts:
                        for strategy in STRATEGIES:
                            metrics = fixed_k_metric_row(
                                signed[position],
                                impact[position],
                                target,
                                valid,
                                int(count),
                                strategy,
                            )
                            rows.append(
                                {
                                    "date": date,
                                    "horizon": int(horizon),
                                    "scope": scope,
                                    "k": int(count),
                                    "strategy": strategy,
                                    "selection_mode": "per_horizon",
                                    "selected_tickers": "",
                                    **metrics,
                                }
                            )
            for scope, scope_valid in scopes.items():
                aggregate_confidence, aggregate_impact, selection_valid = (
                    aggregate_horizon_selection_scores(
                        signed_by_horizon,
                        impact_by_horizon,
                        scope_valid,
                        horizons,
                    )
                )
                for count in counts:
                    for strategy in STRATEGIES:
                        selected = select_fixed_k(
                            aggregate_confidence,
                            aggregate_impact,
                            selection_valid,
                            int(count),
                            strategy,
                        )
                        selected_tickers = "|".join(
                            str(features.tickers[int(index)]) for index in selected
                        )
                        for horizon in horizons:
                            target = date_targets[int(horizon)]
                            eligible = (
                                scope_valid
                                & np.isfinite(signed_by_horizon[int(horizon)])
                                & np.isfinite(impact_by_horizon[int(horizon)])
                                & np.isfinite(target)
                            )
                            rows.append(
                                {
                                    "date": date,
                                    "horizon": int(horizon),
                                    "scope": scope,
                                    "k": int(count),
                                    "strategy": strategy,
                                    "selection_mode": "cross_horizon",
                                    "selected_tickers": selected_tickers,
                                    **fixed_k_metrics_for_selected(
                                        signed_by_horizon[int(horizon)],
                                        target,
                                        eligible,
                                        selected,
                                    ),
                                }
                            )
    return rows


def summarize_rows(
    rows: list[dict[str, Any]],
    horizons: Sequence[int],
    counts: Sequence[int],
    selection_mode: str = "per_horizon",
) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for horizon in horizons:
        horizon_result: dict[str, Any] = {}
        for scope in ("all", "top300"):
            scope_result: dict[str, Any] = {}
            for count in counts:
                strategy_result: dict[str, Any] = {}
                for strategy in STRATEGIES:
                    selected = [
                        row
                        for row in rows
                        if int(row["horizon"]) == int(horizon)
                        and row["scope"] == scope
                        and int(row["k"]) == int(count)
                        and row["strategy"] == strategy
                        and row.get("selection_mode", "per_horizon") == selection_mode
                    ]
                    metric_result: dict[str, Any] = {}
                    for metric in FIXED_K_METRICS:
                        finite = [
                            float(row[metric])
                            for row in selected
                            if math.isfinite(float(row[metric]))
                        ]
                        metric_result[metric] = (
                            newey_west_mean(finite, lag=int(horizon))
                            if finite
                            else {"rows": 0, "mean": float("nan")}
                        )
                    strategy_result[strategy] = metric_result
                scope_result[str(int(count))] = strategy_result
            horizon_result[scope] = scope_result
        summary[str(int(horizon))] = horizon_result
    return summary


def cross_horizon_selection_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[tuple[str, str, int, str], set[str]] = {}
    maximum = 0
    within_limit = True
    for row in rows:
        if row.get("selection_mode") != "cross_horizon":
            continue
        key = (
            str(row["date"]),
            str(row["scope"]),
            int(row["k"]),
            str(row["strategy"]),
        )
        value = str(row.get("selected_tickers", ""))
        groups.setdefault(key, set()).add(value)
        count = len([ticker for ticker in value.split("|") if ticker])
        maximum = max(maximum, count)
        within_limit = within_limit and count <= int(row["k"])
    return {
        "groups": len(groups),
        "same_candidates_across_horizons": all(
            len(values) == 1 for values in groups.values()
        ),
        "within_requested_k": bool(within_limit),
        "maximum_selected_count": int(maximum),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a frozen JEPA impact head at fixed operational counts."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--head-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--counts", default="1,3,5")
    parser.add_argument("--liquidity-top-k", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--edge-cache-workers", type=int, default=8)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    device = torch.device(args.device)
    horizons = parse_int_list(args.horizons)
    counts = parse_int_list(args.counts)
    model_dir = Path(args.model_dir)
    model, ckpt = load_model(model_dir, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    artifact = torch.load(args.head_path, map_location="cpu", weights_only=False)
    parent_sha = checkpoint_sha256(model_dir)
    if artifact.get("parent_model_sha256") != parent_sha:
        raise ValueError("impact head parent checkpoint hash does not match model")
    if [int(value) for value in artifact["horizons"]] != [int(value) for value in horizons]:
        raise ValueError("impact head horizon contract does not match request")
    head = ImpactTrajectoryHead(
        int(artifact["latent_dim"]),
        horizons,
        hidden_dim=int(artifact["hidden_dim"]),
        dropout=float(artifact["dropout"]),
    ).to(device)
    head.load_state_dict(artifact["state_dict"])
    ckpt_args = dict(ckpt.get("args", {}))
    validate_future_rollout_contract(ckpt_args, horizons, False)
    feature_args = evaluator_namespace(args)
    feature_args.edge_cache_workers = int(args.edge_cache_workers)
    features, ckpt_args = build_features_from_ckpt(ckpt, feature_args)
    train_end = str(ckpt_args["train_end"])
    max_horizon = max(horizons)
    edge_window = int(ckpt_args.get("edge_window", 60))
    test_steps = date_indices(
        features.dates,
        start=(pd.Timestamp(train_end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    test_steps = test_steps[
        (test_steps >= edge_window)
        & (test_steps <= len(features.dates) - 1 - max_horizon)
    ]
    if args.max_test_steps and len(test_steps) > args.max_test_steps:
        positions = np.linspace(0, len(test_steps) - 1, args.max_test_steps).round().astype(int)
        test_steps = test_steps[positions]
    edge_cache = build_evaluation_edge_cache(features, test_steps, ckpt_args, feature_args)
    rows = score_steps(
        model,
        head,
        features,
        test_steps,
        horizons,
        ckpt_args,
        feature_args,
        edge_cache,
        device,
        int(args.batch_size),
        int(args.liquidity_top_k),
        float(artifact["latent_blend_weight"]),
        counts,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "daily_fixed_k_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "horizon",
                "scope",
                "k",
                "strategy",
                "selection_mode",
                "selected_tickers",
                *FIXED_K_METRICS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "role": "frozen_jepa_impact_head_fixed_count_diagnostic",
        "model_dir": str(model_dir),
        "head_path": str(args.head_path),
        "parent_model_sha256": parent_sha,
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get("sha256"),
        "train_end": train_end,
        "test_dates": int(len(test_steps)),
        "counts": counts,
        "realized_impact_fraction": 0.10,
        "strategies": list(STRATEGIES),
        "metrics": summarize_rows(rows, horizons, counts),
        "cross_horizon_metrics": summarize_rows(
            rows, horizons, counts, selection_mode="cross_horizon"
        ),
        "selection_modes": ["per_horizon", "cross_horizon"],
        "cross_horizon_selection_contract": cross_horizon_selection_contract(rows),
        "posthoc_notice": (
            "Per-horizon fixed-count metrics were added after seed17 fold1 "
            "fractional results, and cross-horizon selection was added later. "
            "They are secondary diagnostics; fold1 cannot select a strategy "
            "for final gating."
        ),
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"test_dates": len(test_steps), "rows": len(rows)}))


if __name__ == "__main__":
    main()
