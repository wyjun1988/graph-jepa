from __future__ import annotations

import argparse
import copy
import csv
import json
import math
from pathlib import Path
import random
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F

from scripts.benchmark_direct_baselines import evaluator_namespace, newey_west_mean
from scripts.benchmark_latent_trajectory_path_head import (
    HORIZON_WEIGHTS,
    batch_targets,
    blend_path_scores,
    checkpoint_sha256,
    chronological_splits,
    grouped_correlation_loss,
    grouped_zscore,
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
    pearson,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import date_indices, parse_int_list, temporal_training_indices


DEFAULT_IMPACT_FRACTIONS = (0.05, 0.10, 0.20)
DEFAULT_LOSS_WEIGHTS = {
    "impact_rank": 0.30,
    "impact_focal": 0.25,
    "tail_rank": 0.30,
    "tail_direction": 0.10,
    "all_rank": 0.05,
}
METRIC_NAMES = (
    "precision",
    "recall",
    "impact_lift",
    "ndcg",
    "captured_direction_accuracy",
    "realized_tail_direction_accuracy",
    "predicted_bucket_direction_accuracy",
    "tail_ic",
    "signed_ic",
    "selected_mean_abs_return",
    "universe_mean_abs_return",
    "magnitude_lift",
    "realized_tail_mass_recall",
    "captured_impact_weighted_direction_accuracy",
    "signed_realized_tail_mass_capture",
)
VALIDATION_SCORE_MODES = ("impact_v1", "magnitude_v2")


class ImpactTrajectoryHead(nn.Module):
    def __init__(
        self,
        latent_dim: int,
        horizons: Sequence[int],
        hidden_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(horizon) for horizon in horizons)
        self.heads = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.LayerNorm(2 * int(latent_dim)),
                    nn.Linear(2 * int(latent_dim), int(hidden_dim)),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), 2),
                )
                for horizon in self.horizons
            }
        )

    def forward(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = torch.cat((context, predicted - context), dim=-1)
        output = self.heads[str(int(horizon))](features)
        return output[:, 0], output[:, 1]


def grouped_top_fraction_mask(
    values: torch.Tensor,
    valid: torch.Tensor,
    groups: torch.Tensor,
    fraction: float,
) -> torch.Tensor:
    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("fraction must be between zero and one")
    selected = torch.zeros_like(valid, dtype=torch.bool)
    if not valid.any():
        return selected
    for group in torch.unique(groups[valid]).tolist():
        candidates = torch.nonzero(
            valid & (groups == int(group)), as_tuple=False
        ).flatten()
        if candidates.numel() == 0:
            continue
        count = max(1, int(math.ceil(float(candidates.numel()) * float(fraction))))
        chosen = torch.topk(values[candidates], k=count, largest=True).indices
        selected[candidates[chosen]] = True
    return selected


def focal_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    valid: torch.Tensor,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    selected = valid & torch.isfinite(logits)
    if not selected.any():
        return logits.new_tensor(0.0)
    logits = logits[selected]
    labels = labels[selected].to(logits.dtype)
    probability = torch.sigmoid(logits)
    target_probability = torch.where(labels > 0.5, probability, 1.0 - probability)
    alpha_factor = torch.where(
        labels > 0.5,
        torch.full_like(labels, float(alpha)),
        torch.full_like(labels, 1.0 - float(alpha)),
    )
    cross_entropy = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (alpha_factor * (1.0 - target_probability).pow(float(gamma)) * cross_entropy).mean()


def tail_direction_loss(
    score: torch.Tensor,
    target: torch.Tensor,
    tail_mask: torch.Tensor,
    groups: torch.Tensor,
    magnitude_power: float = 0.0,
) -> torch.Tensor:
    valid = torch.isfinite(score) & torch.isfinite(target)
    selected = valid & tail_mask & (target != 0.0)
    if not selected.any():
        return score.new_tensor(0.0)
    normalized = grouped_zscore(score, valid, groups)
    direction = torch.sign(target[selected])
    losses = F.softplus(-direction * normalized[selected])
    if float(magnitude_power) <= 0.0:
        return losses.mean()
    magnitude = torch.log1p(target[selected].abs() * 100.0).clamp_min(1e-6)
    weights = magnitude.pow(float(magnitude_power))
    selected_groups = groups[selected]
    group_count = int(groups.max().item()) + 1
    sums = torch.zeros(group_count, dtype=weights.dtype, device=weights.device)
    counts = torch.zeros_like(sums)
    sums.index_add_(0, selected_groups, weights)
    counts.index_add_(0, selected_groups, torch.ones_like(weights))
    group_means = sums / counts.clamp_min(1.0)
    weights = (weights / group_means[selected_groups].clamp_min(1e-6)).clamp(0.25, 4.0)
    return (weights * losses).sum() / weights.sum().clamp_min(1e-6)


def _mean_pair(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return 0.5 * (first + second)


def train_epoch(
    model,
    head,
    features,
    steps,
    horizons,
    ckpt_args,
    cli_args,
    edge_cache,
    optimizer,
    device,
    batch_size,
    liquidity_top_k,
    latent_weight,
    impact_fraction,
    loss_weights,
    tail_direction_magnitude_power,
    seed,
) -> tuple[float, dict[str, float]]:
    head.train()
    shuffled = np.random.default_rng(seed).permutation(steps)
    total_losses: list[float] = []
    components: dict[str, list[float]] = {name: [] for name in DEFAULT_LOSS_WEIGHTS}
    for start in range(0, len(shuffled), batch_size):
        selected_steps = np.asarray(shuffled[start : start + batch_size], dtype=np.int64)
        batch = snapshot_batch(features, selected_steps, ckpt_args, cli_args, edge_cache, device)
        context, predicted = latent_trajectories(model, batch, horizons, ckpt_args)
        rows, groups = stock_rows(
            len(selected_steps), features.node_count, features.tradable_count, device
        )
        stock_context = context[rows]
        base_scores = state_entry_path_scores(
            model, batch, context, predicted, horizons, ckpt_args, rows
        )
        targets, liquid_mask = batch_targets(
            features, selected_steps, horizons, liquidity_top_k, device
        )
        horizon_losses = []
        horizon_weights = []
        batch_components: dict[str, list[torch.Tensor]] = {
            name: [] for name in DEFAULT_LOSS_WEIGHTS
        }
        for horizon in horizons:
            latent_signed, impact_logit = head(
                stock_context, predicted[int(horizon)][rows], int(horizon)
            )
            target = targets[int(horizon)]
            valid = torch.isfinite(target)
            liquid_valid = valid & liquid_mask
            signed_score = blend_path_scores(
                base_scores[int(horizon)], latent_signed, target, groups, latent_weight
            )
            magnitude_target = torch.log1p(target.abs() * 100.0)
            all_tail = grouped_top_fraction_mask(
                target.abs(), valid, groups, impact_fraction
            )
            liquid_tail = grouped_top_fraction_mask(
                target.abs(), liquid_valid, groups, impact_fraction
            )
            impact_rank = _mean_pair(
                grouped_correlation_loss(
                    impact_logit, magnitude_target, valid, groups
                ),
                grouped_correlation_loss(
                    impact_logit, magnitude_target, liquid_valid, groups
                ),
            )
            impact_focal = _mean_pair(
                focal_binary_loss(impact_logit, all_tail, valid),
                focal_binary_loss(impact_logit, liquid_tail, liquid_valid),
            )
            tail_rank = _mean_pair(
                grouped_correlation_loss(signed_score, target, all_tail, groups),
                grouped_correlation_loss(signed_score, target, liquid_tail, groups),
            )
            direction = _mean_pair(
                tail_direction_loss(
                    signed_score,
                    target,
                    all_tail,
                    groups,
                    tail_direction_magnitude_power,
                ),
                tail_direction_loss(
                    signed_score,
                    target,
                    liquid_tail,
                    groups,
                    tail_direction_magnitude_power,
                ),
            )
            all_rank = _mean_pair(
                grouped_correlation_loss(signed_score, target, valid, groups),
                grouped_correlation_loss(signed_score, target, liquid_valid, groups),
            )
            values = {
                "impact_rank": impact_rank,
                "impact_focal": impact_focal,
                "tail_rank": tail_rank,
                "tail_direction": direction,
                "all_rank": all_rank,
            }
            combined = sum(float(loss_weights[name]) * value for name, value in values.items())
            horizon_weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
            horizon_losses.append(horizon_weight * combined)
            horizon_weights.append(horizon_weight)
            for name, value in values.items():
                batch_components[name].append(value.detach())
        loss = torch.stack(horizon_losses).sum() / sum(horizon_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        total_losses.append(float(loss.detach().cpu()))
        for name, values in batch_components.items():
            components[name].append(float(torch.stack(values).mean().cpu()))
    return float(np.mean(total_losses)), {
        name: float(np.mean(values)) for name, values in components.items()
    }


def _safe_direction_accuracy(score: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(score) & np.isfinite(target) & (target != 0.0)
    if not valid.any():
        return float("nan")
    return float(np.mean(np.sign(score[valid]) == np.sign(target[valid])))


def _impact_weighted_direction_accuracy(
    score: np.ndarray, target: np.ndarray, selected: np.ndarray
) -> float:
    valid = selected & np.isfinite(score) & np.isfinite(target) & (target != 0.0)
    if not valid.any():
        return float("nan")
    weights = np.abs(target[valid])
    weight_sum = float(weights.sum())
    if weight_sum <= 0.0:
        return float("nan")
    correct = np.sign(score[valid]) == np.sign(target[valid])
    return float(np.sum(weights * correct) / weight_sum)


def _ndcg_at_k(impact_score: np.ndarray, relevance: np.ndarray, count: int) -> float:
    order = np.argsort(impact_score, kind="stable")[-count:][::-1]
    ideal = np.argsort(relevance, kind="stable")[-count:][::-1]
    discounts = 1.0 / np.log2(np.arange(count, dtype=np.float64) + 2.0)
    dcg = float(np.sum(relevance[order] * discounts))
    ideal_dcg = float(np.sum(relevance[ideal] * discounts))
    return dcg / ideal_dcg if ideal_dcg > 0.0 else float("nan")


def impact_metric_row(
    signed_score: np.ndarray,
    impact_score: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    fraction: float,
) -> dict[str, float]:
    selected = np.flatnonzero(
        valid & np.isfinite(signed_score) & np.isfinite(impact_score) & np.isfinite(target)
    )
    if len(selected) < 10:
        return {name: float("nan") for name in METRIC_NAMES}
    signed = signed_score[selected]
    impact = impact_score[selected]
    observed = target[selected]
    relevance = np.abs(observed)
    count = max(1, int(math.ceil(len(selected) * float(fraction))))
    predicted_local = np.argsort(impact, kind="stable")[-count:]
    realized_local = np.argsort(relevance, kind="stable")[-count:]
    predicted_mask = np.zeros(len(selected), dtype=bool)
    realized_mask = np.zeros(len(selected), dtype=bool)
    predicted_mask[predicted_local] = True
    realized_mask[realized_local] = True
    captured_mask = predicted_mask & realized_mask
    captured = int(captured_mask.sum())
    precision = float(captured / count)
    universe_mean = float(np.mean(relevance))
    selected_mean = float(np.mean(relevance[predicted_mask]))
    tail_mass = float(np.sum(relevance[realized_mask]))
    captured_mass = float(np.sum(relevance[captured_mask]))
    captured_valid = captured_mask & (observed != 0.0)
    direction_alignment = np.where(
        np.sign(signed[captured_valid]) == np.sign(observed[captured_valid]),
        1.0,
        -1.0,
    )
    signed_captured_mass = float(
        np.sum(relevance[captured_valid] * direction_alignment)
    )
    return {
        "precision": precision,
        "recall": precision,
        "impact_lift": precision / float(fraction),
        "ndcg": _ndcg_at_k(impact, relevance, count),
        "captured_direction_accuracy": _safe_direction_accuracy(
            signed[captured_mask], observed[captured_mask]
        ),
        "realized_tail_direction_accuracy": _safe_direction_accuracy(
            signed[realized_mask], observed[realized_mask]
        ),
        "predicted_bucket_direction_accuracy": _safe_direction_accuracy(
            signed[predicted_mask], observed[predicted_mask]
        ),
        "tail_ic": pearson(signed[realized_mask], observed[realized_mask]),
        "signed_ic": pearson(signed, observed),
        "selected_mean_abs_return": selected_mean,
        "universe_mean_abs_return": universe_mean,
        "magnitude_lift": selected_mean / universe_mean if universe_mean > 0.0 else float("nan"),
        "realized_tail_mass_recall": (
            captured_mass / tail_mass if tail_mass > 0.0 else float("nan")
        ),
        "captured_impact_weighted_direction_accuracy": (
            _impact_weighted_direction_accuracy(signed, observed, captured_mask)
        ),
        "signed_realized_tail_mass_capture": (
            signed_captured_mass / tail_mass if tail_mass > 0.0 else float("nan")
        ),
    }


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
    impact_fractions,
) -> list[dict[str, Any]]:
    head.eval()
    output_rows: list[dict[str, Any]] = []
    for start in range(0, len(steps), batch_size):
        selected_steps = np.asarray(steps[start : start + batch_size], dtype=np.int64)
        batch = snapshot_batch(features, selected_steps, ckpt_args, cli_args, edge_cache, device)
        context, predicted = latent_trajectories(model, batch, horizons, ckpt_args)
        selected_rows, groups = stock_rows(
            len(selected_steps), features.node_count, features.tradable_count, device
        )
        stock_context = context[selected_rows]
        base_scores = state_entry_path_scores(
            model, batch, context, predicted, horizons, ckpt_args, selected_rows
        )
        targets, _ = batch_targets(
            features, selected_steps, horizons, liquidity_top_k, device
        )
        predictions: dict[int, dict[str, np.ndarray]] = {}
        with torch.no_grad():
            for horizon in horizons:
                latent_signed, impact_logit = head(
                    stock_context,
                    predicted[int(horizon)][selected_rows],
                    int(horizon),
                )
                blended = blend_path_scores(
                    base_scores[int(horizon)],
                    latent_signed,
                    targets[int(horizon)],
                    groups,
                    latent_weight,
                )
                valid = torch.isfinite(targets[int(horizon)])
                normalized_base = grouped_zscore(base_scores[int(horizon)], valid, groups)
                predictions[int(horizon)] = {
                    "signed": blended.float().cpu().numpy().reshape(
                        len(selected_steps), features.tradable_count
                    ),
                    "impact": impact_logit.float().cpu().numpy().reshape(
                        len(selected_steps), features.tradable_count
                    ),
                    "base": normalized_base.float().cpu().numpy().reshape(
                        len(selected_steps), features.tradable_count
                    ),
                }
        liquidity_index = features.feature_names.index("value_ma20_log")
        for position, step in enumerate(selected_steps):
            date = str(pd.Timestamp(features.dates[int(step)]).date())
            liquidity = features.raw_features[
                int(step), : features.tradable_count, liquidity_index
            ]
            scope_masks = {
                "all": np.ones(features.tradable_count, dtype=bool),
                "top300": top_liquidity_mask(liquidity, liquidity_top_k),
            }
            for horizon in horizons:
                target = np.asarray(
                    features.target_return_paths[int(horizon)][
                        int(step), : features.tradable_count
                    ],
                    dtype=np.float64,
                )
                signed = predictions[int(horizon)]["signed"][position]
                impact = predictions[int(horizon)]["impact"][position]
                base = predictions[int(horizon)]["base"][position]
                variants = {
                    "impact_head": (signed, impact),
                    "signed_abs": (signed, np.abs(signed)),
                    "base_jepa": (base, np.abs(base)),
                }
                for scope, scope_mask in scope_masks.items():
                    for fraction in impact_fractions:
                        for variant, (variant_signed, variant_impact) in variants.items():
                            metrics = impact_metric_row(
                                variant_signed,
                                variant_impact,
                                target,
                                scope_mask,
                                fraction,
                            )
                            output_rows.append(
                                {
                                    "date": date,
                                    "horizon": int(horizon),
                                    "scope": scope,
                                    "fraction": float(fraction),
                                    "variant": variant,
                                    **metrics,
                                }
                            )
    return output_rows


def _metric_summary(values: list[float], lag: int) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "rows": 0,
            "mean": float("nan"),
            "newey_west_lag": int(lag),
            "newey_west_standard_error": float("nan"),
            "newey_west_t": float("nan"),
            "positive_fraction": float("nan"),
        }
    return newey_west_mean(finite, lag=int(lag))


def impact_validation_score(
    metrics: dict[str, Any],
    primary_fraction: float,
    mode: str = "impact_v1",
) -> float:
    precision = float(metrics["precision"]["mean"])
    tail_ic = float(metrics["tail_ic"]["mean"])
    impact_skill = (precision - float(primary_fraction)) / (
        1.0 - float(primary_fraction)
    )
    if mode == "impact_v1":
        direction = float(metrics["captured_direction_accuracy"]["mean"])
        if not all(
            math.isfinite(value) for value in (precision, direction, tail_ic)
        ):
            return float("nan")
        direction_skill = 2.0 * (direction - 0.5)
        return 0.50 * impact_skill + 0.30 * direction_skill + 0.20 * tail_ic
    if mode == "magnitude_v2":
        mass_recall = float(metrics["realized_tail_mass_recall"]["mean"])
        weighted_direction = float(
            metrics["captured_impact_weighted_direction_accuracy"]["mean"]
        )
        signed_mass = float(metrics["signed_realized_tail_mass_capture"]["mean"])
        if not all(
            math.isfinite(value)
            for value in (
                precision,
                mass_recall,
                weighted_direction,
                signed_mass,
                tail_ic,
            )
        ):
            return float("nan")
        mass_skill = (mass_recall - float(primary_fraction)) / (
            1.0 - float(primary_fraction)
        )
        weighted_direction_skill = 2.0 * (weighted_direction - 0.5)
        return (
            0.25 * impact_skill
            + 0.15 * mass_skill
            + 0.25 * weighted_direction_skill
            + 0.30 * signed_mass
            + 0.05 * tail_ic
        )
    raise ValueError(f"unsupported validation score mode: {mode}")


def primary_metric_contract(mode: str, primary_fraction: float) -> dict[str, Any]:
    if mode == "impact_v1":
        return {
            "mode": mode,
            "scope": "top300",
            "fraction": float(primary_fraction),
            "impact_skill_weight": 0.50,
            "captured_direction_skill_weight": 0.30,
            "tail_ic_weight": 0.20,
        }
    if mode == "magnitude_v2":
        return {
            "mode": mode,
            "scope": "top300",
            "fraction": float(primary_fraction),
            "impact_skill_weight": 0.25,
            "tail_mass_recall_skill_weight": 0.15,
            "impact_weighted_direction_skill_weight": 0.25,
            "signed_realized_tail_mass_capture_weight": 0.30,
            "tail_ic_weight": 0.05,
            "random_precision_and_tail_mass_baseline": float(primary_fraction),
            "random_direction_baseline": 0.50,
        }
    raise ValueError(f"unsupported validation score mode: {mode}")


def summarize_rows(
    rows: list[dict[str, Any]],
    horizons: Sequence[int],
    impact_fractions: Sequence[float],
    validation_score_mode: str = "impact_v1",
) -> tuple[dict[str, Any], float]:
    summary: dict[str, Any] = {}
    for horizon in horizons:
        horizon_result: dict[str, Any] = {}
        for scope in ("all", "top300"):
            scope_result: dict[str, Any] = {}
            for fraction in impact_fractions:
                fraction_key = f"{float(fraction):.2f}"
                variant_result: dict[str, Any] = {}
                for variant in ("impact_head", "signed_abs", "base_jepa"):
                    selected = [
                        row
                        for row in rows
                        if int(row["horizon"]) == int(horizon)
                        and row["scope"] == scope
                        and abs(float(row["fraction"]) - float(fraction)) < 1e-9
                        and row["variant"] == variant
                    ]
                    variant_result[variant] = {
                        metric: _metric_summary(
                            [float(row[metric]) for row in selected], int(horizon)
                        )
                        for metric in METRIC_NAMES
                    }
                scope_result[fraction_key] = variant_result
            horizon_result[scope] = scope_result
        summary[str(int(horizon))] = horizon_result

    primary_fraction = min(impact_fractions, key=lambda value: abs(float(value) - 0.10))
    primary_key = f"{float(primary_fraction):.2f}"
    weighted_score = 0.0
    weight_sum = 0.0
    for horizon in horizons:
        metrics = summary[str(int(horizon))]["top300"][primary_key]["impact_head"]
        score = impact_validation_score(
            metrics, float(primary_fraction), validation_score_mode
        )
        if not math.isfinite(score):
            continue
        weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
        weighted_score += weight * score
        weight_sum += weight
    return summary, weighted_score / weight_sum if weight_sum else float("nan")


def write_daily(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["date", "horizon", "scope", "fraction", "variant", *METRIC_NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an impact-weighted head on frozen JEPA latent trajectories."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--impact-fractions", default="0.05,0.10,0.20")
    parser.add_argument("--train-impact-fraction", type=float, default=0.10)
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--liquidity-top-k", type=int, default=300)
    parser.add_argument("--latent-blend-weight", type=float, default=0.5)
    parser.add_argument("--impact-rank-weight", type=float, default=0.30)
    parser.add_argument("--impact-focal-weight", type=float, default=0.25)
    parser.add_argument("--tail-rank-weight", type=float, default=0.30)
    parser.add_argument("--tail-direction-weight", type=float, default=0.10)
    parser.add_argument("--all-rank-weight", type=float, default=0.05)
    parser.add_argument("--tail-direction-magnitude-power", type=float, default=0.0)
    parser.add_argument(
        "--validation-score-mode",
        choices=VALIDATION_SCORE_MODES,
        default="impact_v1",
    )
    parser.add_argument("--max-fit-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--edge-cache-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    if not 0.0 <= args.latent_blend_weight <= 1.0:
        raise ValueError("--latent-blend-weight must be between zero and one")
    if not 0.0 < args.train_impact_fraction < 1.0:
        raise ValueError("--train-impact-fraction must be between zero and one")
    if args.tail_direction_magnitude_power < 0.0:
        raise ValueError("--tail-direction-magnitude-power must be non-negative")
    impact_fractions = tuple(float(value) for value in args.impact_fractions.split(","))
    if any(not 0.0 < value < 1.0 for value in impact_fractions):
        raise ValueError("all impact fractions must be between zero and one")
    loss_weights = {
        "impact_rank": float(args.impact_rank_weight),
        "impact_focal": float(args.impact_focal_weight),
        "tail_rank": float(args.tail_rank_weight),
        "tail_direction": float(args.tail_direction_weight),
        "all_rank": float(args.all_rank_weight),
    }
    weight_sum = sum(loss_weights.values())
    if weight_sum <= 0.0 or any(value < 0.0 for value in loss_weights.values()):
        raise ValueError("loss weights must be non-negative and have a positive sum")
    loss_weights = {name: value / weight_sum for name, value in loss_weights.items()}

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    horizons = parse_int_list(args.horizons)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model, ckpt = load_model(model_dir, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    ckpt_args = dict(ckpt.get("args", {}))
    validate_future_rollout_contract(ckpt_args, horizons, False)
    feature_args = evaluator_namespace(args)
    feature_args.edge_cache_workers = int(args.edge_cache_workers)
    features, ckpt_args = build_features_from_ckpt(ckpt, feature_args)
    train_end = str(ckpt_args["train_end"])
    max_horizon = max(horizons)
    edge_window = int(ckpt_args.get("edge_window", 60))
    train_indices = date_indices(features.dates, end=train_end)
    train_steps = temporal_training_indices(
        train_indices,
        edge_window=edge_window,
        max_rollout_offset=max_horizon,
        total_steps=len(features.dates),
    )
    fit_steps, validation_steps = chronological_splits(
        train_steps, args.validation_days, max_horizon
    )
    if args.max_fit_steps and len(fit_steps) > args.max_fit_steps:
        positions = np.linspace(0, len(fit_steps) - 1, args.max_fit_steps).round().astype(int)
        fit_steps = fit_steps[positions]
    if args.max_validation_steps and len(validation_steps) > args.max_validation_steps:
        positions = np.linspace(
            0, len(validation_steps) - 1, args.max_validation_steps
        ).round().astype(int)
        validation_steps = validation_steps[positions]
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
    all_steps = np.unique(np.concatenate((fit_steps, validation_steps, test_steps)))
    edge_cache = build_evaluation_edge_cache(features, all_steps, ckpt_args, feature_args)

    latent_dim = int(ckpt_args["hidden_dim"])
    head = ImpactTrajectoryHead(
        latent_dim,
        horizons,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        loss, components = train_epoch(
            model,
            head,
            features,
            fit_steps,
            horizons,
            ckpt_args,
            feature_args,
            edge_cache,
            optimizer,
            device,
            args.batch_size,
            args.liquidity_top_k,
            args.latent_blend_weight,
            args.train_impact_fraction,
            loss_weights,
            args.tail_direction_magnitude_power,
            args.seed + epoch,
        )
        validation_rows = score_steps(
            model,
            head,
            features,
            validation_steps,
            horizons,
            ckpt_args,
            feature_args,
            edge_cache,
            device,
            args.batch_size,
            args.liquidity_top_k,
            args.latent_blend_weight,
            impact_fractions,
        )
        _, validation_score = summarize_rows(
            validation_rows,
            horizons,
            impact_fractions,
            validation_score_mode=args.validation_score_mode,
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": loss,
                "train_components": components,
                "validation_impact_score": validation_score,
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={loss:.6f} "
            f"validation_impact_score={validation_score:+.6f}",
            flush=True,
        )
        if math.isfinite(validation_score) and validation_score > best_score + 1e-4:
            best_score = validation_score
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("impact head did not produce a valid checkpoint")
    head.load_state_dict(best_state)
    test_rows = score_steps(
        model,
        head,
        features,
        test_steps,
        horizons,
        ckpt_args,
        feature_args,
        edge_cache,
        device,
        args.batch_size,
        args.liquidity_top_k,
        args.latent_blend_weight,
        impact_fractions,
    )
    metrics, weighted_score = summarize_rows(
        test_rows,
        horizons,
        impact_fractions,
        validation_score_mode=args.validation_score_mode,
    )
    write_daily(output_dir / "daily_impact_metrics.csv", test_rows)
    parent_sha = checkpoint_sha256(model_dir)
    summary = {
        "status": "complete",
        "role": "frozen_jepa_impact_weighted_trajectory_head_research",
        "model_dir": str(model_dir),
        "parent_model_sha256": parent_sha,
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get("sha256"),
        "train_edge_manifest_sha256": ckpt.get("train_edge_manifest", {}).get("sha256"),
        "fold2_used_for_selection": False,
        "train_end": train_end,
        "fit_dates": len(fit_steps),
        "validation_dates": len(validation_steps),
        "test_dates": len(test_steps),
        "evaluation_seed": args.seed,
        "latent_blend_weight": args.latent_blend_weight,
        "train_impact_fraction": args.train_impact_fraction,
        "impact_fractions": list(impact_fractions),
        "loss_weights": loss_weights,
        "impact_weight_total": 1.0 - loss_weights["all_rank"],
        "tail_direction_magnitude_power": args.tail_direction_magnitude_power,
        "validation_score_mode": args.validation_score_mode,
        "primary_metric_contract": primary_metric_contract(
            args.validation_score_mode,
            min(impact_fractions, key=lambda value: abs(value - 0.10)),
        ),
        "metrics": metrics,
        "weighted_impact_score": weighted_score,
        "best_validation_impact_score": best_score,
        "history": history,
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "parent_model_sha256": parent_sha,
            "horizons": horizons,
            "latent_dim": latent_dim,
            "hidden_dim": args.hidden_dim,
            "dropout": args.dropout,
            "latent_blend_weight": args.latent_blend_weight,
            "train_impact_fraction": args.train_impact_fraction,
            "loss_weights": loss_weights,
            "tail_direction_magnitude_power": args.tail_direction_magnitude_power,
            "validation_score_mode": args.validation_score_mode,
            "best_validation_impact_score": best_score,
            "train_data_manifest_sha256": summary["train_data_manifest_sha256"],
            "train_edge_manifest_sha256": summary["train_edge_manifest_sha256"],
            "live_orders_allowed": False,
        },
        output_dir / "impact_trajectory_head.pt",
    )
    print(
        json.dumps(
            {"weighted_impact_score": weighted_score, "test_dates": len(test_steps)}
        )
    )


if __name__ == "__main__":
    main()
