from __future__ import annotations

import argparse
import copy
import csv
import hashlib
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

from scripts.benchmark_direct_baselines import evaluator_namespace, newey_west_mean
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    graph_edge_kwargs,
    load_model,
    pearson,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import (
    date_indices,
    parse_int_list,
    rollout_steps_for_offset,
    temporal_training_indices,
)
from stock_v2.graph_jepa import merge_graph_batches
from stock_v2.real_features import make_real_snapshot


HORIZON_WEIGHTS = {1: 2.0, 2: 2.0, 3: 1.0, 5: 1.0, 10: 1.0}


class LatentTrajectoryPathHead(nn.Module):
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
                    nn.Linear(int(hidden_dim), 1),
                )
                for horizon in self.horizons
            }
        )

    def forward(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        horizon: int,
    ) -> torch.Tensor:
        features = torch.cat((context, predicted - context), dim=-1)
        return self.heads[str(int(horizon))](features).squeeze(-1)


def grouped_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    selected = valid & torch.isfinite(prediction) & torch.isfinite(target)
    if not selected.any():
        return prediction.new_tensor(0.0)
    group_count = int(groups.max().item()) + 1
    selected_groups = groups[selected]
    pred = prediction[selected]
    observed = target[selected]
    counts = torch.zeros(group_count, dtype=prediction.dtype, device=prediction.device)
    counts.index_add_(0, selected_groups, torch.ones_like(pred))
    pred_sum = torch.zeros_like(counts).index_add_(0, selected_groups, pred)
    target_sum = torch.zeros_like(counts).index_add_(0, selected_groups, observed)
    pred_sq_sum = torch.zeros_like(counts).index_add_(0, selected_groups, pred.square())
    target_sq_sum = torch.zeros_like(counts).index_add_(0, selected_groups, observed.square())
    cross_sum = torch.zeros_like(counts).index_add_(0, selected_groups, pred * observed)
    safe_counts = counts.clamp_min(1.0)
    covariance = cross_sum - pred_sum * target_sum / safe_counts
    pred_variance = pred_sq_sum - pred_sum.square() / safe_counts
    target_variance = target_sq_sum - target_sum.square() / safe_counts
    usable = (counts >= 3.0) & (pred_variance > 1e-8) & (target_variance > 1e-8)
    if not usable.any():
        return prediction.new_tensor(0.0)
    correlation = covariance[usable] / torch.sqrt(
        pred_variance[usable] * target_variance[usable]
    ).clamp_min(1e-8)
    return 1.0 - correlation.clamp(-1.0, 1.0).mean()


def grouped_zscore(
    values: torch.Tensor,
    valid: torch.Tensor,
    groups: torch.Tensor,
) -> torch.Tensor:
    selected = valid & torch.isfinite(values)
    group_count = int(groups.max().item()) + 1
    selected_groups = groups[selected]
    selected_values = values[selected]
    counts = torch.zeros(group_count, dtype=values.dtype, device=values.device)
    counts.index_add_(0, selected_groups, torch.ones_like(selected_values))
    sums = torch.zeros_like(counts).index_add_(0, selected_groups, selected_values)
    means = sums / counts.clamp_min(1.0)
    centered = selected_values - means[selected_groups]
    variances = torch.zeros_like(counts).index_add_(0, selected_groups, centered.square())
    stds = torch.sqrt(variances / counts.clamp_min(1.0)).clamp_min(1e-6)
    result = torch.zeros_like(values)
    result[selected] = centered / stds[selected_groups]
    return result


def blend_path_scores(
    base: torch.Tensor,
    latent: torch.Tensor,
    target: torch.Tensor,
    groups: torch.Tensor,
    latent_weight: float,
) -> torch.Tensor:
    valid = torch.isfinite(target) & torch.isfinite(base) & torch.isfinite(latent)
    base_z = grouped_zscore(base, valid, groups)
    latent_z = grouped_zscore(latent, valid, groups)
    return (1.0 - float(latent_weight)) * base_z + float(latent_weight) * latent_z


def chronological_splits(
    train_steps: np.ndarray,
    validation_days: int,
    max_horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    if validation_days < 20 or len(train_steps) <= validation_days + 260:
        raise ValueError("training history is too short for the requested validation split")
    validation = np.asarray(train_steps[-validation_days:], dtype=np.int64)
    fit = np.asarray(
        train_steps[train_steps < int(validation[0]) - int(max_horizon)],
        dtype=np.int64,
    )
    if len(fit) < 260:
        raise ValueError("fit split is too short")
    return fit, validation


def top_liquidity_mask(values: np.ndarray, size: int) -> np.ndarray:
    result = np.zeros(len(values), dtype=bool)
    finite = np.flatnonzero(np.isfinite(values))
    if len(finite) == 0:
        return result
    order = np.argsort(values[finite], kind="stable")
    result[finite[order[-min(int(size), len(finite)) :]]] = True
    return result


def checkpoint_sha256(model_dir: Path) -> str:
    digest = hashlib.sha256()
    with (model_dir / "graph_jepa_real.pt").open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_batch(features, steps, ckpt_args, cli_args, edge_cache, device):
    batches = [
        make_real_snapshot(
            features,
            step=int(step),
            full_observation=True,
            edge_window=int(ckpt_args.get("edge_window", 60)),
            top_k=int(ckpt_args.get("edge_top_k", 6)),
            min_abs_corr=float(ckpt_args.get("min_abs_corr", 0.2)),
            **graph_edge_kwargs(ckpt_args, cli_args),
            edge_cache=edge_cache,
        )
        for step in steps
    ]
    return merge_graph_batches(batches).to(device)


def latent_trajectories(model, batch, horizons: Sequence[int], ckpt_args):
    rollout_args = dict(ckpt_args)
    rollout_args.setdefault("temporal_offset", ckpt_args.get("horizon", max(horizons)))
    rollout_args.setdefault("latent_rollout_steps", 1)
    namespace = argparse.Namespace(**rollout_args)
    requested = {
        int(horizon): rollout_steps_for_offset(namespace, int(horizon))
        for horizon in horizons
    }
    with torch.no_grad():
        context = model.encode_temporal_context(batch)
        current = context
        predicted: dict[int, torch.Tensor] = {}
        for step in range(1, max(requested.values()) + 1):
            current = model._predict_latent(current)
            for horizon, rollout_step in requested.items():
                if step == rollout_step:
                    predicted[horizon] = current
    return context.detach(), {key: value.detach() for key, value in predicted.items()}


def state_entry_path_scores(
    model,
    batch,
    context: torch.Tensor,
    predicted: dict[int, torch.Tensor],
    horizons: Sequence[int],
    ckpt_args: dict[str, Any],
    rows: torch.Tensor,
) -> dict[int, torch.Tensor]:
    rollout_args = dict(ckpt_args)
    rollout_args.setdefault("temporal_offset", ckpt_args.get("horizon", max(horizons)))
    rollout_args.setdefault("latent_rollout_steps", 1)
    namespace = argparse.Namespace(**rollout_args)
    with torch.no_grad():
        state = {
            int(horizon): model.predict_temporal_state(
                batch,
                predicted[int(horizon)],
                rollout_steps=rollout_steps_for_offset(namespace, int(horizon)),
                z_context=context,
            )[rows]
            for horizon in horizons
        }
    h1_state = state[1]
    gap_index = int(model.gap_open_feature_index)
    intraday_index = int(model.intraday_return_feature_index)
    next_open_gap = (
        h1_state[:, gap_index] * model.feature_stds[gap_index]
        + model.feature_means[gap_index]
    )
    scores: dict[int, torch.Tensor] = {
        1: (
            h1_state[:, intraday_index] * model.feature_stds[intraday_index]
            + model.feature_means[intraday_index]
        )
    }
    for horizon in horizons:
        if int(horizon) == 1:
            continue
        return_index = int(model.return_feature_indices[int(horizon)])
        close_return = (
            state[int(horizon)][:, return_index] * model.feature_stds[return_index]
            + model.feature_means[return_index]
        )
        denominator = 1.0 + next_open_gap
        scores[int(horizon)] = torch.where(
            denominator > 1e-6,
            (1.0 + close_return) / denominator.clamp_min(1e-6) - 1.0,
            torch.full_like(close_return, float("nan")),
        )
    return scores


def stock_rows(batch_size: int, node_count: int, stock_count: int, device):
    rows = np.concatenate(
        [np.arange(index * node_count, index * node_count + stock_count) for index in range(batch_size)]
    )
    groups = np.repeat(np.arange(batch_size, dtype=np.int64), stock_count)
    return (
        torch.as_tensor(rows, dtype=torch.long, device=device),
        torch.as_tensor(groups, dtype=torch.long, device=device),
    )


def batch_targets(features, steps, horizons: Sequence[int], liquidity_top_k: int, device):
    stock_count = int(features.tradable_count)
    liquidity_index = features.feature_names.index("value_ma20_log")
    targets = {
        int(horizon): torch.as_tensor(
            np.asarray(
                features.target_return_paths[int(horizon)][steps, :stock_count],
                dtype=np.float32,
            ).reshape(-1),
            device=device,
        )
        for horizon in horizons
    }
    liquid_masks = np.stack(
        [
            top_liquidity_mask(
                features.raw_features[int(step), :stock_count, liquidity_index],
                liquidity_top_k,
            )
            for step in steps
        ]
    )
    liquid_mask = torch.as_tensor(liquid_masks.reshape(-1), dtype=torch.bool, device=device)
    return targets, liquid_mask


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
    seed,
) -> float:
    head.train()
    shuffled = np.random.default_rng(seed).permutation(steps)
    losses = []
    for start in range(0, len(shuffled), batch_size):
        selected_steps = np.asarray(shuffled[start : start + batch_size], dtype=np.int64)
        batch = snapshot_batch(features, selected_steps, ckpt_args, cli_args, edge_cache, device)
        context, predicted = latent_trajectories(model, batch, horizons, ckpt_args)
        rows, groups = stock_rows(len(selected_steps), features.node_count, features.tradable_count, device)
        stock_context = context[rows]
        base_scores = state_entry_path_scores(
            model, batch, context, predicted, horizons, ckpt_args, rows
        )
        targets, liquid_mask = batch_targets(
            features, selected_steps, horizons, liquidity_top_k, device
        )
        weighted_losses = []
        weights = []
        for horizon in horizons:
            latent_score = head(
                stock_context, predicted[int(horizon)][rows], int(horizon)
            )
            target = targets[int(horizon)]
            valid = torch.isfinite(target)
            score = blend_path_scores(
                base_scores[int(horizon)],
                latent_score,
                target,
                groups,
                latent_weight,
            )
            all_loss = grouped_correlation_loss(score, target, valid, groups)
            liquid_loss = grouped_correlation_loss(
                score, target, valid & liquid_mask, groups
            )
            weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
            weighted_losses.append(weight * 0.5 * (all_loss + liquid_loss))
            weights.append(weight)
        loss = torch.stack(weighted_losses).sum() / sum(weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses))


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
):
    head.eval()
    rows: list[dict[str, Any]] = []
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
        targets, _liquid_mask = batch_targets(
            features, selected_steps, horizons, liquidity_top_k, device
        )
        liquidity_index = features.feature_names.index("value_ma20_log")
        with torch.no_grad():
            scores = {}
            for horizon in horizons:
                latent_score = head(
                    stock_context,
                    predicted[int(horizon)][selected_rows],
                    int(horizon),
                )
                blended = blend_path_scores(
                    base_scores[int(horizon)],
                    latent_score,
                    targets[int(horizon)],
                    groups,
                    latent_weight,
                )
                scores[int(horizon)] = (
                    blended.float()
                    .cpu()
                    .numpy()
                    .reshape(len(selected_steps), features.tradable_count)
                )
        for position, step in enumerate(selected_steps):
            liquidity = features.raw_features[
                int(step), : features.tradable_count, liquidity_index
            ]
            top_mask = top_liquidity_mask(liquidity, liquidity_top_k)
            for horizon in horizons:
                target = features.target_return_paths[int(horizon)][
                    int(step), : features.tradable_count
                ]
                score = scores[int(horizon)][position]
                rows.append(
                    {
                        "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                        "horizon": int(horizon),
                        "entry_path_ic": pearson(score, target),
                        "entry_path_ic_top300": pearson(score[top_mask], target[top_mask]),
                    }
                )
    return rows


def summarize_rows(rows: list[dict[str, Any]], horizons: Sequence[int]):
    result: dict[str, Any] = {}
    weighted = 0.0
    weight_sum = 0.0
    for horizon in horizons:
        selected = [row for row in rows if int(row["horizon"]) == int(horizon)]
        all_result = newey_west_mean(
            [float(row["entry_path_ic"]) for row in selected], lag=int(horizon)
        )
        top_result = newey_west_mean(
            [float(row["entry_path_ic_top300"]) for row in selected], lag=int(horizon)
        )
        result[str(horizon)] = {"all_stock": all_result, "top300": top_result}
        weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
        weighted += weight * 0.5 * (float(all_result["mean"]) + float(top_result["mean"]))
        weight_sum += weight
    return result, float(weighted / weight_sum)


def write_daily(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "horizon", "entry_path_ic", "entry_path_ic_top300"],
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a path-ranking head on frozen JEPA latent trajectories."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--liquidity-top-k", type=int, default=300)
    parser.add_argument("--latent-blend-weight", type=float, default=1.0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--edge-cache-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    if not 0.0 <= args.latent_blend_weight <= 1.0:
        raise ValueError("--latent-blend-weight must be between 0 and 1")

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
    head = LatentTrajectoryPathHead(
        latent_dim,
        horizons,
        hidden_dim=args.hidden_dim,
        dropout=args.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    history = []
    best_score = -math.inf
    best_state = None
    stale = 0
    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(
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
        )
        _validation_metrics, validation_score = summarize_rows(validation_rows, horizons)
        history.append(
            {"epoch": epoch, "train_loss": loss, "validation_weighted_path_ic": validation_score}
        )
        print(
            f"epoch={epoch:02d} train_loss={loss:.6f} validation_path_ic={validation_score:+.6f}",
            flush=True,
        )
        if validation_score > best_score + 1e-4:
            best_score = validation_score
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break
    if best_state is None:
        raise RuntimeError("latent path head did not produce a valid checkpoint")
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
    )
    metrics, weighted_score = summarize_rows(test_rows, horizons)
    write_daily(output_dir / "daily_metrics.csv", test_rows)
    parent_sha = checkpoint_sha256(model_dir)
    summary = {
        "status": "complete",
        "role": "frozen_jepa_latent_trajectory_path_head_research",
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
        "horizons": metrics,
        "weighted_path_ic": weighted_score,
        "best_validation_path_ic": best_score,
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
            "best_validation_path_ic": best_score,
            "train_data_manifest_sha256": summary["train_data_manifest_sha256"],
            "train_edge_manifest_sha256": summary["train_edge_manifest_sha256"],
            "live_orders_allowed": False,
        },
        output_dir / "latent_trajectory_path_head.pt",
    )
    print(json.dumps({"weighted_path_ic": weighted_score, "test_dates": len(test_steps)}))


if __name__ == "__main__":
    main()
