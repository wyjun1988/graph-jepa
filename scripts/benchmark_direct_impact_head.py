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

from scripts.benchmark_direct_baselines import (
    build_context_layout,
    evaluator_namespace,
    load_or_build_context_matrix,
    newey_west_mean,
    rows_for_steps,
)
from scripts.benchmark_direct_state_mlp import (
    amp_context,
    amp_grad_scaler,
    prepare_context_tensor,
    resolve_device,
)
from scripts.benchmark_impact_trajectory_head import (
    DEFAULT_LOSS_WEIGHTS,
    METRIC_NAMES,
    VALIDATION_SCORE_MODES,
    focal_binary_loss,
    grouped_top_fraction_mask,
    impact_validation_score,
    impact_metric_row,
    primary_metric_contract,
    tail_direction_loss,
)
from scripts.benchmark_latent_trajectory_path_head import (
    HORIZON_WEIGHTS,
    grouped_correlation_loss,
    top_liquidity_mask,
)
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.run_real_backtest import date_indices, parse_int_list, temporal_training_indices


DIRECT_VARIANTS = ("direct_impact", "direct_signed_abs", "momentum")


class DirectImpactHead(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        hidden_dim: int = 256,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(horizon) for horizon in horizons)
        self.heads = nn.ModuleDict(
            {
                str(horizon): nn.Sequential(
                    nn.LayerNorm(int(input_dim)),
                    nn.Linear(int(input_dim), int(hidden_dim)),
                    nn.SiLU(),
                    nn.Dropout(float(dropout)),
                    nn.Linear(int(hidden_dim), 2),
                )
                for horizon in self.horizons
            }
        )

    def forward(self, context: torch.Tensor, horizon: int) -> tuple[torch.Tensor, torch.Tensor]:
        output = self.heads[str(int(horizon))](context)
        return output[:, 0], output[:, 1]


def _mean_pair(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
    return 0.5 * (first + second)


def train_epoch(
    head: DirectImpactHead,
    context: torch.Tensor,
    targets: dict[int, torch.Tensor],
    liquid_mask: torch.Tensor,
    horizons: Sequence[int],
    optimizer,
    scaler,
    amp_enabled: bool,
    batch_dates: int,
    impact_fraction: float,
    loss_weights: dict[str, float],
    tail_direction_magnitude_power: float,
    seed: int,
) -> tuple[float, dict[str, float]]:
    head.train()
    date_count, stock_count, _ = context.shape
    order = np.random.default_rng(seed).permutation(date_count)
    losses: list[float] = []
    components: dict[str, list[float]] = {name: [] for name in DEFAULT_LOSS_WEIGHTS}
    for start in range(0, len(order), int(batch_dates)):
        selected = torch.as_tensor(
            order[start : start + int(batch_dates)], dtype=torch.long, device=context.device
        )
        dates_in_batch = int(selected.numel())
        x = context[selected].reshape(dates_in_batch * stock_count, -1)
        groups = torch.arange(dates_in_batch, device=context.device).repeat_interleave(
            stock_count
        )
        liquid = liquid_mask[selected].reshape(-1)
        horizon_losses = []
        horizon_weights = []
        batch_components: dict[str, list[torch.Tensor]] = {
            name: [] for name in DEFAULT_LOSS_WEIGHTS
        }
        with amp_context(amp_enabled):
            for horizon in horizons:
                signed, impact = head(x, int(horizon))
                signed = signed.float()
                impact = impact.float()
                target = targets[int(horizon)][selected].reshape(-1)
                valid = torch.isfinite(target)
                liquid_valid = valid & liquid
                magnitude = torch.log1p(target.abs() * 100.0)
                all_tail = grouped_top_fraction_mask(
                    target.abs(), valid, groups, impact_fraction
                )
                liquid_tail = grouped_top_fraction_mask(
                    target.abs(), liquid_valid, groups, impact_fraction
                )
                values = {
                    "impact_rank": _mean_pair(
                        grouped_correlation_loss(impact, magnitude, valid, groups),
                        grouped_correlation_loss(
                            impact, magnitude, liquid_valid, groups
                        ),
                    ),
                    "impact_focal": _mean_pair(
                        focal_binary_loss(impact, all_tail, valid),
                        focal_binary_loss(impact, liquid_tail, liquid_valid),
                    ),
                    "tail_rank": _mean_pair(
                        grouped_correlation_loss(signed, target, all_tail, groups),
                        grouped_correlation_loss(signed, target, liquid_tail, groups),
                    ),
                    "tail_direction": _mean_pair(
                        tail_direction_loss(
                            signed,
                            target,
                            all_tail,
                            groups,
                            tail_direction_magnitude_power,
                        ),
                        tail_direction_loss(
                            signed,
                            target,
                            liquid_tail,
                            groups,
                            tail_direction_magnitude_power,
                        ),
                    ),
                    "all_rank": _mean_pair(
                        grouped_correlation_loss(signed, target, valid, groups),
                        grouped_correlation_loss(signed, target, liquid_valid, groups),
                    ),
                }
                combined = sum(
                    float(loss_weights[name]) * value for name, value in values.items()
                )
                horizon_weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
                horizon_losses.append(horizon_weight * combined)
                horizon_weights.append(horizon_weight)
                for name, value in values.items():
                    batch_components[name].append(value.detach())
            loss = torch.stack(horizon_losses).sum() / sum(horizon_weights)
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        losses.append(float(loss.detach().cpu()))
        for name, values in batch_components.items():
            components[name].append(float(torch.stack(values).mean().cpu()))
    return float(np.mean(losses)), {
        name: float(np.mean(values)) for name, values in components.items()
    }


def momentum_scores(features, steps: np.ndarray, horizon: int) -> np.ndarray:
    feature_name = "intraday_return" if int(horizon) == 1 else f"return_{int(horizon)}d"
    index = features.feature_names.index(feature_name)
    return np.asarray(
        features.raw_features[steps, : features.tradable_count, index], dtype=np.float64
    )


def score_rows(
    head: DirectImpactHead,
    context: torch.Tensor,
    targets: dict[int, torch.Tensor],
    features,
    steps: np.ndarray,
    horizons: Sequence[int],
    fractions: Sequence[float],
    liquidity_top_k: int,
    batch_dates: int,
    amp_enabled: bool,
) -> list[dict[str, Any]]:
    head.eval()
    date_count, stock_count, _ = context.shape
    predictions: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    signed_chunks: dict[int, list[np.ndarray]] = {int(h): [] for h in horizons}
    impact_chunks: dict[int, list[np.ndarray]] = {int(h): [] for h in horizons}
    with torch.inference_mode():
        for start in range(0, date_count, int(batch_dates)):
            x = context[start : start + int(batch_dates)]
            batch_count = len(x)
            flat = x.reshape(batch_count * stock_count, -1)
            with amp_context(amp_enabled):
                for horizon in horizons:
                    signed, impact = head(flat, int(horizon))
                    signed_chunks[int(horizon)].append(
                        signed.float().cpu().numpy().reshape(batch_count, stock_count)
                    )
                    impact_chunks[int(horizon)].append(
                        impact.float().cpu().numpy().reshape(batch_count, stock_count)
                    )
    for horizon in horizons:
        predictions[int(horizon)] = (
            np.concatenate(signed_chunks[int(horizon)], axis=0),
            np.concatenate(impact_chunks[int(horizon)], axis=0),
        )
    rows: list[dict[str, Any]] = []
    liquidity_index = features.feature_names.index("value_ma20_log")
    momentum = {int(h): momentum_scores(features, steps, int(h)) for h in horizons}
    for position, step in enumerate(steps):
        liquidity = features.raw_features[int(step), :stock_count, liquidity_index]
        scopes = {
            "all": np.ones(stock_count, dtype=bool),
            "top300": top_liquidity_mask(liquidity, liquidity_top_k),
        }
        for horizon in horizons:
            signed, impact = predictions[int(horizon)]
            target = targets[int(horizon)][position].float().cpu().numpy()
            variants = {
                "direct_impact": (signed[position], impact[position]),
                "direct_signed_abs": (signed[position], np.abs(signed[position])),
                "momentum": (momentum[int(horizon)][position], np.abs(momentum[int(horizon)][position])),
            }
            for scope, valid in scopes.items():
                for fraction in fractions:
                    for variant, (variant_signed, variant_impact) in variants.items():
                        rows.append(
                            {
                                "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                                "horizon": int(horizon),
                                "scope": scope,
                                "fraction": float(fraction),
                                "variant": variant,
                                **impact_metric_row(
                                    variant_signed,
                                    variant_impact,
                                    target,
                                    valid,
                                    float(fraction),
                                ),
                            }
                        )
    return rows


def metric_summary(values: list[float], lag: int) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"rows": 0, "mean": float("nan"), "newey_west_lag": int(lag)}
    return newey_west_mean(finite, lag=int(lag))


def summarize_rows(
    rows: list[dict[str, Any]],
    horizons: Sequence[int],
    fractions: Sequence[float],
    validation_score_mode: str = "impact_v1",
) -> tuple[dict[str, Any], float]:
    summary: dict[str, Any] = {}
    for horizon in horizons:
        horizon_result: dict[str, Any] = {}
        for scope in ("all", "top300"):
            scope_result: dict[str, Any] = {}
            for fraction in fractions:
                variant_result: dict[str, Any] = {}
                for variant in DIRECT_VARIANTS:
                    selected = [
                        row
                        for row in rows
                        if int(row["horizon"]) == int(horizon)
                        and row["scope"] == scope
                        and abs(float(row["fraction"]) - float(fraction)) < 1e-9
                        and row["variant"] == variant
                    ]
                    variant_result[variant] = {
                        metric: metric_summary(
                            [float(row[metric]) for row in selected], int(horizon)
                        )
                        for metric in METRIC_NAMES
                    }
                scope_result[f"{float(fraction):.2f}"] = variant_result
            horizon_result[scope] = scope_result
        summary[str(int(horizon))] = horizon_result

    primary_fraction = min(fractions, key=lambda value: abs(float(value) - 0.10))
    key = f"{float(primary_fraction):.2f}"
    weighted = 0.0
    weight_sum = 0.0
    for horizon in horizons:
        metrics = summary[str(int(horizon))]["top300"][key]["direct_impact"]
        score = impact_validation_score(
            metrics, float(primary_fraction), validation_score_mode
        )
        if not math.isfinite(score):
            continue
        weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
        weighted += weight * score
        weight_sum += weight
    return summary, weighted / weight_sum if weight_sum else float("nan")


def split_targets(features, steps: np.ndarray, horizons: Sequence[int], device: torch.device):
    return {
        int(horizon): torch.as_tensor(
            np.asarray(
                features.target_return_paths[int(horizon)][
                    steps, : features.tradable_count
                ],
                dtype=np.float32,
            ),
            device=device,
        )
        for horizon in horizons
    }


def split_liquidity_mask(features, steps: np.ndarray, liquidity_top_k: int, device):
    index = features.feature_names.index("value_ma20_log")
    values = np.stack(
        [
            top_liquidity_mask(
                features.raw_features[int(step), : features.tradable_count, index],
                liquidity_top_k,
            )
            for step in steps
        ]
    )
    return torch.as_tensor(values, dtype=torch.bool, device=device)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train an impact-weighted direct context challenger."
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
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--batch-dates", type=int, default=8)
    parser.add_argument("--eval-batch-dates", type=int, default=32)
    parser.add_argument("--liquidity-top-k", type=int, default=300)
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
    parser.add_argument("--without-graph", action="store_true")
    parser.add_argument("--feature-workers", type=int, default=16)
    parser.add_argument("--max-fit-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--context-cache", default=None)
    args = parser.parse_args()

    if not 0.0 < args.train_impact_fraction < 1.0:
        raise ValueError("--train-impact-fraction must be between zero and one")
    if args.tail_direction_magnitude_power < 0.0:
        raise ValueError("--tail-direction-magnitude-power must be non-negative")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    horizons = parse_int_list(args.horizons)
    fractions = tuple(float(value) for value in args.impact_fractions.split(","))
    device = resolve_device(str(args.device))
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

    model_dir = Path(args.model_dir)
    ckpt = torch.load(
        model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    features, ckpt_args = build_features_from_ckpt(ckpt, evaluator_namespace(args))
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
    validation_steps = np.asarray(train_steps[-int(args.validation_days) :], dtype=np.int64)
    fit_steps = np.asarray(
        train_steps[train_steps < int(validation_steps[0]) - max_horizon], dtype=np.int64
    )
    if len(fit_steps) < 260:
        raise ValueError("fit split is too short")
    test_steps = date_indices(
        features.dates,
        start=(pd.Timestamp(train_end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    test_steps = test_steps[
        (test_steps >= edge_window)
        & (test_steps <= len(features.dates) - 1 - max_horizon)
    ]
    if args.max_fit_steps and len(fit_steps) > args.max_fit_steps:
        positions = np.linspace(0, len(fit_steps) - 1, args.max_fit_steps).round().astype(int)
        fit_steps = fit_steps[positions]
    if args.max_validation_steps and len(validation_steps) > args.max_validation_steps:
        positions = np.linspace(
            0, len(validation_steps) - 1, args.max_validation_steps
        ).round().astype(int)
        validation_steps = validation_steps[positions]
    if args.max_test_steps and len(test_steps) > args.max_test_steps:
        positions = np.linspace(0, len(test_steps) - 1, args.max_test_steps).round().astype(int)
        test_steps = test_steps[positions]
    all_steps = np.unique(np.concatenate((fit_steps, validation_steps, test_steps)))
    positions = {int(step): index for index, step in enumerate(all_steps)}
    layout = build_context_layout(features, fit_steps, include_calendar=False)
    matrix = load_or_build_context_matrix(
        features,
        all_steps,
        layout,
        ckpt,
        ckpt_args,
        workers=max(1, int(args.feature_workers)),
        cache_path=Path(args.context_cache) if args.context_cache else None,
    )
    input_dim = int(
        layout.base_feature_count if args.without_graph else layout.total_feature_count
    )
    use_fp16 = device.type == "cuda" and bool(args.amp)
    split_steps = {
        "fit": fit_steps,
        "validation": validation_steps,
        "test": test_steps,
    }
    contexts = {}
    targets = {}
    liquidity_masks = {}
    for name, steps in split_steps.items():
        matrix_rows = rows_for_steps(steps, positions, features.tradable_count)
        contexts[name] = prepare_context_tensor(
            matrix, matrix_rows, input_dim, device, use_fp16
        ).reshape(len(steps), features.tradable_count, input_dim)
        targets[name] = split_targets(features, steps, horizons, device)
        liquidity_masks[name] = split_liquidity_mask(
            features, steps, int(args.liquidity_top_k), device
        )

    head = DirectImpactHead(
        input_dim,
        horizons,
        hidden_dim=int(args.hidden_dim),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    amp_enabled = device.type == "cuda" and bool(args.amp)
    scaler = amp_grad_scaler(amp_enabled)
    history: list[dict[str, Any]] = []
    best_score = -math.inf
    best_state = None
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        loss, components = train_epoch(
            head,
            contexts["fit"],
            targets["fit"],
            liquidity_masks["fit"],
            horizons,
            optimizer,
            scaler,
            amp_enabled,
            int(args.batch_dates),
            float(args.train_impact_fraction),
            loss_weights,
            float(args.tail_direction_magnitude_power),
            int(args.seed) + epoch,
        )
        validation_rows = score_rows(
            head,
            contexts["validation"],
            targets["validation"],
            features,
            validation_steps,
            horizons,
            fractions,
            int(args.liquidity_top_k),
            int(args.eval_batch_dates),
            amp_enabled,
        )
        _, validation_score = summarize_rows(
            validation_rows,
            horizons,
            fractions,
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
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("direct impact head did not produce a checkpoint")
    head.load_state_dict(best_state)
    test_rows = score_rows(
        head,
        contexts["test"],
        targets["test"],
        features,
        test_steps,
        horizons,
        fractions,
        int(args.liquidity_top_k),
        int(args.eval_batch_dates),
        amp_enabled,
    )
    metrics, weighted_score = summarize_rows(
        test_rows,
        horizons,
        fractions,
        validation_score_mode=args.validation_score_mode,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "daily_impact_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["date", "horizon", "scope", "fraction", "variant", *METRIC_NAMES],
        )
        writer.writeheader()
        writer.writerows(test_rows)
    summary = {
        "status": "complete",
        "role": "equal_objective_direct_context_impact_challenger",
        "model_dir": str(model_dir),
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get("sha256"),
        "train_edge_manifest_sha256": ckpt.get("train_edge_manifest", {}).get("sha256"),
        "train_end": train_end,
        "fit_dates": int(len(fit_steps)),
        "validation_dates": int(len(validation_steps)),
        "test_dates": int(len(test_steps)),
        "uses_graph_neighbor_state": not bool(args.without_graph),
        "input_dim": input_dim,
        "hidden_dim": int(args.hidden_dim),
        "loss_weights": loss_weights,
        "tail_direction_magnitude_power": float(
            args.tail_direction_magnitude_power
        ),
        "validation_score_mode": args.validation_score_mode,
        "primary_metric_contract": primary_metric_contract(
            args.validation_score_mode,
            min(fractions, key=lambda value: abs(value - 0.10)),
        ),
        "weighted_impact_score": weighted_score,
        "best_validation_impact_score": best_score,
        "history": history,
        "metrics": metrics,
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "input_dim": input_dim,
            "horizons": horizons,
            "hidden_dim": int(args.hidden_dim),
            "dropout": float(args.dropout),
            "uses_graph_neighbor_state": not bool(args.without_graph),
            "loss_weights": loss_weights,
            "tail_direction_magnitude_power": float(
                args.tail_direction_magnitude_power
            ),
            "validation_score_mode": args.validation_score_mode,
            "train_data_manifest_sha256": summary["train_data_manifest_sha256"],
            "live_orders_allowed": False,
        },
        output_dir / "direct_impact_head.pt",
    )
    print(json.dumps({"weighted_impact_score": weighted_score, "test_dates": len(test_steps)}))


if __name__ == "__main__":
    main()
