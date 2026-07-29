from __future__ import annotations

import argparse
from contextlib import nullcontext
from copy import deepcopy
from dataclasses import dataclass
import gc
import json
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
    _correlation,
    build_context_layout,
    evaluator_namespace,
    load_or_build_context_matrix,
    newey_west_mean,
    rows_for_steps,
)
from scripts.evaluate_node_prediction import (
    build_features_from_ckpt,
    derive_entry_path_return,
    feature_group_indices,
    future_state_metrics,
    state_target_feature_mask,
)
from scripts.run_real_backtest import date_indices, parse_int_list, temporal_training_indices


class ResidualStateMLP(nn.Module):
    """Direct supervised baseline initialized to the persistence forecast."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        layers: int = 3,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if layers < 1:
            raise ValueError("layers must be positive")
        blocks: list[nn.Module] = []
        width = input_dim
        for _ in range(layers):
            blocks.extend(
                [
                    nn.Linear(width, hidden_dim),
                    nn.SiLU(),
                    nn.LayerNorm(hidden_dim),
                    nn.Dropout(dropout),
                ]
            )
            width = hidden_dim
        self.trunk = nn.Sequential(*blocks)
        self.delta_head = nn.Linear(width, output_dim)
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)
        self.output_dim = int(output_dim)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        current_state = context[:, : self.output_dim]
        return current_state + self.delta_head(self.trunk(context))


@dataclass(frozen=True)
class SplitContext:
    steps: np.ndarray
    matrix_rows: np.ndarray


def resolve_device(value: str) -> torch.device:
    if value != "auto":
        return torch.device(value)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def target_state_arrays(
    features,
    steps: np.ndarray,
    horizon: int,
    target_feature_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    stock_count = features.tradable_count
    target_steps = np.asarray(steps, dtype=np.int64) + int(horizon)
    target = features.features[target_steps, :stock_count].reshape(
        -1, len(features.feature_names)
    ).astype(np.float32, copy=False)
    current_available = features.available_mask[steps, :stock_count].reshape(
        target.shape
    ) > 0.5
    target_available = features.available_mask[target_steps, :stock_count].reshape(
        target.shape
    ) > 0.5
    valid = current_available & target_available & np.isfinite(target)
    if target_feature_mask is not None:
        target_feature_mask = np.asarray(target_feature_mask, dtype=bool)
        if target_feature_mask.shape != (len(features.feature_names),):
            raise ValueError("target feature mask does not match the feature schema")
        valid &= target_feature_mask[None, :]
    return target, valid


def masked_sse(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if prediction.shape != target.shape or target.shape != valid.shape:
        raise ValueError("prediction, target, and valid arrays must have the same shape")
    error = np.where(valid, prediction - target, 0.0).astype(np.float64)
    return np.sum(error * error, axis=0), valid.sum(axis=0).astype(np.int64)


def sign_accuracy(
    prediction_delta: np.ndarray,
    target_delta: np.ndarray,
    valid: np.ndarray,
    threshold: float = 0.10,
) -> float:
    selected = (
        valid
        & np.isfinite(prediction_delta)
        & np.isfinite(target_delta)
        & (np.abs(target_delta) >= float(threshold))
    )
    if not selected.any():
        return float("nan")
    return float(
        (np.sign(prediction_delta[selected]) == np.sign(target_delta[selected])).mean()
    )


def pooled_skill(
    prediction: np.ndarray,
    current: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float | int]:
    model_sse = float(masked_sse(prediction, target, valid)[0].sum())
    persistence_sse = float(masked_sse(current, target, valid)[0].sum())
    zero_sse = float(masked_sse(np.zeros_like(target), target, valid)[0].sum())
    return {
        "observed_cells": int(valid.sum()),
        "model_sse": model_sse,
        "persistence_sse": persistence_sse,
        "zero_sse": zero_sse,
        "skill_vs_persistence": (
            float(1.0 - model_sse / persistence_sse)
            if persistence_sse > 1e-12
            else float("nan")
        ),
        "skill_vs_zero": (
            float(1.0 - model_sse / zero_sse) if zero_sse > 1e-12 else float("nan")
        ),
    }


def _tensor_from_numpy(
    values: np.ndarray,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    tensor = torch.from_numpy(np.ascontiguousarray(values))
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device=device)


def prepare_context_tensor(
    matrix: np.ndarray,
    rows: np.ndarray,
    feature_count: int,
    device: torch.device,
    use_fp16: bool,
) -> torch.Tensor:
    values = np.asarray(matrix[rows, :feature_count], dtype=np.float32)
    tensor = _tensor_from_numpy(
        values,
        device,
        dtype=torch.float16 if use_fp16 else torch.float32,
    )
    del values
    gc.collect()
    return tensor


def amp_context(enabled: bool):
    if not enabled:
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.float16)


def amp_grad_scaler(enabled: bool):
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def validation_sse(
    model: nn.Module,
    context: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    batch_size: int,
    amp_enabled: bool,
) -> tuple[float, float]:
    model.eval()
    model_sse = 0.0
    persistence_sse = 0.0
    with torch.inference_mode():
        for start in range(0, len(context), batch_size):
            end = min(start + batch_size, len(context))
            x = context[start:end]
            y = target[start:end]
            mask = valid[start:end]
            with amp_context(amp_enabled):
                prediction = model(x)
            difference = prediction.float() - y
            persistence = x[:, : model.output_dim].float() - y
            model_sse += float((difference.square() * mask).sum().item())
            persistence_sse += float((persistence.square() * mask).sum().item())
    return model_sse, persistence_sse


def fit_model(
    model: ResidualStateMLP,
    fit_context: torch.Tensor,
    fit_target: torch.Tensor,
    fit_valid: torch.Tensor,
    validation_context: torch.Tensor,
    validation_target: torch.Tensor,
    validation_valid: torch.Tensor,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[ResidualStateMLP, dict[str, Any]]:
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    amp_enabled = device.type == "cuda" and bool(args.amp)
    scaler = amp_grad_scaler(amp_enabled)
    generator = torch.Generator(device=device.type)
    generator.manual_seed(int(args.seed))
    best_skill = -float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    history: list[dict[str, float | int]] = []
    stale_epochs = 0

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        order = torch.randperm(len(fit_context), generator=generator, device=device)
        loss_sum = 0.0
        observed = 0
        for start in range(0, len(order), int(args.batch_size)):
            rows = order[start : start + int(args.batch_size)]
            x = fit_context[rows]
            y = fit_target[rows]
            mask = fit_valid[rows]
            optimizer.zero_grad(set_to_none=True)
            with amp_context(amp_enabled):
                prediction = model(x)
                squared_error = (prediction.float() - y).square()
                denominator = mask.sum().clamp_min(1)
                loss = (squared_error * mask).sum() / denominator
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.max_grad_norm))
            scaler.step(optimizer)
            scaler.update()
            count = int(mask.sum().item())
            loss_sum += float(loss.item()) * count
            observed += count

        model_sse, persistence_sse = validation_sse(
            model,
            validation_context,
            validation_target,
            validation_valid,
            int(args.batch_size),
            amp_enabled,
        )
        skill = (
            float(1.0 - model_sse / persistence_sse)
            if persistence_sse > 1e-12
            else -float("inf")
        )
        row = {
            "epoch": int(epoch),
            "train_mse": float(loss_sum / max(observed, 1)),
            "validation_model_sse": model_sse,
            "validation_persistence_sse": persistence_sse,
            "validation_skill_vs_persistence": skill,
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} train_mse={row['train_mse']:.6f} "
            f"validation_skill={skill:+.6f}",
            flush=True,
        )
        if skill > best_skill + float(args.min_delta):
            best_skill = skill
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone()
                for name, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
            if stale_epochs >= int(args.patience):
                break

    if best_state is None:
        raise RuntimeError("direct state MLP did not produce a checkpoint")
    model.load_state_dict(best_state)
    return model, {
        "best_epoch": int(best_epoch),
        "best_validation_skill_vs_persistence": float(best_skill),
        "history": history,
    }


def predict_batches(
    model: ResidualStateMLP,
    context: torch.Tensor,
    batch_size: int,
    amp_enabled: bool,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for start in range(0, len(context), batch_size):
            end = min(start + batch_size, len(context))
            with amp_context(amp_enabled):
                prediction = model(context[start:end])
            chunks.append(prediction.float().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def evaluate_predictions(
    features,
    steps: np.ndarray,
    horizon: int,
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    next_open_gap_prediction: np.ndarray | None = None,
) -> tuple[
    dict[str, Any],
    dict[str, dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    stock_count = features.tradable_count
    feature_count = len(features.feature_names)
    current = features.features[steps, :stock_count].reshape(-1, feature_count).astype(
        np.float32, copy=False
    )
    pooled = pooled_skill(prediction, current, target, valid)
    model_feature_sse, feature_cells = masked_sse(prediction, target, valid)
    persistence_feature_sse, _ = masked_sse(current, target, valid)
    zero_feature_sse, _ = masked_sse(np.zeros_like(target), target, valid)
    feature_rows: list[dict[str, Any]] = []
    for index, name in enumerate(features.feature_names):
        feature_rows.append(
            {
                "horizon": int(horizon),
                "feature": str(name),
                "observed_cells": int(feature_cells[index]),
                "model_mse": (
                    float(model_feature_sse[index] / feature_cells[index])
                    if feature_cells[index] > 0
                    else float("nan")
                ),
                "persistence_mse": (
                    float(persistence_feature_sse[index] / feature_cells[index])
                    if feature_cells[index] > 0
                    else float("nan")
                ),
                "zero_mse": (
                    float(zero_feature_sse[index] / feature_cells[index])
                    if feature_cells[index] > 0
                    else float("nan")
                ),
                "skill_vs_persistence": (
                    float(1.0 - model_feature_sse[index] / persistence_feature_sse[index])
                    if persistence_feature_sse[index] > 1e-12
                    else float("nan")
                ),
            }
        )

    return_feature = f"return_{int(horizon)}d"
    if return_feature not in features.feature_names:
        raise ValueError(
            f"direct state evaluation requires matched path feature {return_feature}"
        )
    return_index = features.feature_names.index(return_feature)
    path = features.target_return_paths[int(horizon)][steps, :stock_count]
    liquidity_index = features.feature_names.index("value_ma20_log")
    liquidity = features.raw_features[steps, :stock_count, liquidity_index]
    prediction_3d = prediction.reshape(len(steps), stock_count, feature_count)
    target_3d = target.reshape(prediction_3d.shape)
    current_3d = current.reshape(prediction_3d.shape)
    valid_3d = valid.reshape(prediction_3d.shape)
    if next_open_gap_prediction is None:
        if int(horizon) != 1 or "gap_open" not in features.feature_names:
            raise ValueError(
                "direct entry-path evaluation requires the horizon-1 gap forecast"
            )
        gap_index = features.feature_names.index("gap_open")
        next_open_gap_prediction = (
            prediction[:, gap_index] * float(features.train_std[gap_index])
            + float(features.train_mean[gap_index])
        )
    next_open_gap_2d = np.asarray(next_open_gap_prediction, dtype=np.float64).reshape(
        len(steps), stock_count
    )
    groups = feature_group_indices(list(features.feature_names))
    group_daily_metrics: dict[str, list[dict[str, float]]] = {
        group_name: [] for group_name in groups
    }
    daily_rows: list[dict[str, Any]] = []
    for position, step in enumerate(steps):
        cell_valid = valid_3d[position]
        model_error = np.where(
            cell_valid,
            prediction_3d[position] - target_3d[position],
            0.0,
        )
        persistence_error = np.where(
            cell_valid,
            current_3d[position] - target_3d[position],
            0.0,
        )
        day_model_sse = float(np.sum(model_error.astype(np.float64) ** 2))
        day_persistence_sse = float(np.sum(persistence_error.astype(np.float64) ** 2))
        prediction_delta = prediction_3d[position] - current_3d[position]
        target_delta = target_3d[position] - current_3d[position]
        return_valid = cell_valid[:, return_index]
        state_score = prediction_3d[position, :, return_index]
        close_return_score = (
            state_score * float(features.train_std[return_index])
            + float(features.train_mean[return_index])
        )
        if int(horizon) == 1:
            intraday_index = features.feature_names.index("intraday_return")
            path_score = (
                prediction_3d[position, :, intraday_index]
                * float(features.train_std[intraday_index])
                + float(features.train_mean[intraday_index])
            )
        else:
            path_score = derive_entry_path_return(
                close_return_score,
                next_open_gap_2d[position],
            )
        path_valid = return_valid & np.isfinite(path[position]) & np.isfinite(path_score)
        finite_liquidity = np.flatnonzero(np.isfinite(liquidity[position]))
        if len(finite_liquidity) > 300:
            order = np.argsort(liquidity[position, finite_liquidity], kind="stable")
            finite_liquidity = finite_liquidity[order[-300:]]
        top300 = np.zeros(stock_count, dtype=bool)
        top300[finite_liquidity] = True
        state = target_3d[position, :, return_index]
        state_metrics = future_state_metrics(
            prediction_3d[position],
            target_3d[position],
            current_3d[position],
            cell_valid,
        )
        if state_metrics is None:
            continue
        for group_name, indices in groups.items():
            group_metrics = future_state_metrics(
                prediction_3d[position][:, indices],
                target_3d[position][:, indices],
                current_3d[position][:, indices],
                cell_valid[:, indices],
            )
            if group_metrics is not None:
                group_daily_metrics[group_name].append(group_metrics)
        daily_rows.append(
            {
                "horizon": int(horizon),
                "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                "all_state_skill_vs_persistence": (
                    float(1.0 - day_model_sse / day_persistence_sse)
                    if day_persistence_sse > 1e-12
                    else float("nan")
                ),
                "all_state_target_corr": _correlation(
                    prediction_3d[position][cell_valid],
                    target_3d[position][cell_valid],
                ),
                "all_state_delta_corr": _correlation(
                    prediction_delta[cell_valid],
                    target_delta[cell_valid],
                ),
                "all_state_delta_sign_accuracy_abs_ge_0_10": sign_accuracy(
                    prediction_delta,
                    target_delta,
                    cell_valid,
                ),
                "all_state_target_sign_accuracy_abs_ge_0_10": float(
                    state_metrics["target_sign_accuracy_abs_ge_0_10"]
                ),
                "return_state_ic": _correlation(state_score[return_valid], state[return_valid]),
                "return_state_ic_top300": _correlation(
                    state_score[return_valid & top300], state[return_valid & top300]
                ),
                "return_path_ic": _correlation(path_score[path_valid], path[position, path_valid]),
                "return_path_ic_top300": _correlation(
                    path_score[path_valid & top300], path[position, path_valid & top300]
                ),
                "return_score_feature": return_feature,
            }
        )

    pooled.update(
        {
            metric: newey_west_mean(
                [float(row[metric]) for row in daily_rows], lag=int(horizon)
            )
            for metric in (
                "all_state_skill_vs_persistence",
                "all_state_target_corr",
                "all_state_delta_corr",
                "all_state_delta_sign_accuracy_abs_ge_0_10",
                "all_state_target_sign_accuracy_abs_ge_0_10",
                "return_state_ic",
                "return_state_ic_top300",
                "return_path_ic",
                "return_path_ic_top300",
            )
        }
    )
    group_summary: dict[str, dict[str, Any]] = {}
    for group_name, indices in groups.items():
        group_prediction = prediction[:, indices]
        group_current = current[:, indices]
        group_target = target[:, indices]
        group_valid = valid[:, indices]
        group_result = pooled_skill(
            group_prediction,
            group_current,
            group_target,
            group_valid,
        )
        for metric in (
            "mse_skill_vs_persistence",
            "target_corr",
            "delta_corr",
            "target_sign_accuracy_abs_ge_0_10",
            "delta_sign_accuracy_abs_ge_0_10",
        ):
            group_result[metric] = newey_west_mean(
                [float(row[metric]) for row in group_daily_metrics[group_name]],
                lag=int(horizon),
            )
        group_summary[group_name] = group_result
    return pooled, group_summary, daily_rows, feature_rows


def _context_splits(
    fit_steps: np.ndarray,
    validation_steps: np.ndarray,
    test_steps: np.ndarray,
    step_positions: dict[int, int],
    stock_count: int,
) -> dict[str, SplitContext]:
    return {
        "fit": SplitContext(
            fit_steps, rows_for_steps(fit_steps, step_positions, stock_count)
        ),
        "validation": SplitContext(
            validation_steps,
            rows_for_steps(validation_steps, step_positions, stock_count),
        ),
        "test": SplitContext(
            test_steps, rows_for_steps(test_steps, step_positions, stock_count)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Challenge Graph-JEPA all-state rollout with a direct residual MLP."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument(
        "--state-target-scope",
        choices=["all", "checkpoint_temporal"],
        default="all",
        help=(
            "Train and score every state feature or exactly the checkpoint's "
            "non-zero temporal targets."
        ),
    )
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--epochs", type=int, default=16)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--without-graph", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--feature-workers", type=int, default=24)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--context-cache", default=None)
    args = parser.parse_args()

    set_seed(int(args.seed))
    device = resolve_device(str(args.device))
    horizons = parse_int_list(args.horizons)
    max_horizon = max(horizons)
    model_dir = Path(args.model_dir)
    ckpt = torch.load(model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False)
    feature_args = deepcopy(args)
    feature_args.horizons = args.horizons
    features, ckpt_args = build_features_from_ckpt(ckpt, evaluator_namespace(feature_args))
    target_feature_mask = state_target_feature_mask(
        features.feature_names,
        ckpt.get("temporal_state_feature_weights"),
        args.state_target_scope,
    )
    train_end = str(ckpt_args.get("train_end", "2023-12-29"))
    edge_window = int(ckpt_args.get("edge_window", 60))
    train_indices = date_indices(features.dates, end=train_end)
    train_steps = temporal_training_indices(
        train_indices,
        edge_window=edge_window,
        max_rollout_offset=max_horizon,
        total_steps=len(features.dates),
    )
    validation_steps = train_steps[-int(args.validation_days) :]
    fit_steps = train_steps[train_steps < int(validation_steps[0]) - max_horizon]
    test_steps = date_indices(
        features.dates,
        start=(pd.Timestamp(train_end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    test_steps = test_steps[
        (test_steps >= edge_window)
        & (test_steps <= len(features.dates) - 1 - max_horizon)
    ]
    if len(fit_steps) < 260:
        raise ValueError("fit split is too short")

    all_steps = np.unique(np.concatenate([fit_steps, validation_steps, test_steps])).astype(
        np.int64
    )
    step_positions = {int(step): position for position, step in enumerate(all_steps)}
    layout = build_context_layout(features, fit_steps, include_calendar=False)
    context_matrix = load_or_build_context_matrix(
        features,
        all_steps,
        layout,
        ckpt,
        ckpt_args,
        workers=max(1, int(args.feature_workers)),
        cache_path=Path(args.context_cache) if args.context_cache else None,
    )
    feature_count = (
        layout.base_feature_count
        if args.without_graph
        else layout.total_feature_count
    )
    splits = _context_splits(
        fit_steps,
        validation_steps,
        test_steps,
        step_positions,
        features.tradable_count,
    )
    use_fp16 = device.type == "cuda" and bool(args.amp)
    print(
        f"direct state panel: stocks={features.tradable_count} states={len(features.feature_names)} "
        f"fit={len(fit_steps)} validation={len(validation_steps)} test={len(test_steps)} "
        f"inputs={feature_count} device={device} fp16_context={use_fp16}",
        flush=True,
    )
    context_tensors = {
        name: prepare_context_tensor(
            context_matrix,
            split.matrix_rows,
            feature_count,
            device,
            use_fp16,
        )
        for name, split in splits.items()
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
        "role": "research_only_direct_state_baseline",
        "test_used_for_selection": False,
        "live_orders_allowed": False,
        "checkpoint": str(model_dir),
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get("sha256"),
        "train_end": train_end,
        "fit_start": str(features.dates[int(fit_steps[0])].date()),
        "fit_end": str(features.dates[int(fit_steps[-1])].date()),
        "validation_start": str(features.dates[int(validation_steps[0])].date()),
        "validation_end": str(features.dates[int(validation_steps[-1])].date()),
        "test_start": str(features.dates[int(test_steps[0])].date()),
        "test_end": str(features.dates[int(test_steps[-1])].date()),
        "fit_dates": int(len(fit_steps)),
        "validation_dates": int(len(validation_steps)),
        "test_dates": int(len(test_steps)),
        "stocks": int(features.tradable_count),
        "states": int(len(features.feature_names)),
        "state_target_scope": args.state_target_scope,
        "state_target_feature_count": int(target_feature_mask.sum()),
        "state_target_features": [
            str(name)
            for name, selected in zip(features.feature_names, target_feature_mask)
            if selected
        ],
        "input_features": int(feature_count),
        "uses_graph_neighbor_state": not bool(args.without_graph),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
        "horizons": {},
    }
    daily_rows: list[dict[str, Any]] = []
    feature_rows: list[dict[str, Any]] = []
    test_next_open_gap_prediction: np.ndarray | None = None

    for horizon in horizons:
        print(f"training all-state residual MLP horizon={horizon}", flush=True)
        split_targets: dict[str, torch.Tensor] = {}
        split_valid: dict[str, torch.Tensor] = {}
        target_numpy: dict[str, np.ndarray] = {}
        valid_numpy: dict[str, np.ndarray] = {}
        for name, split in splits.items():
            target, valid = target_state_arrays(
                features,
                split.steps,
                horizon,
                target_feature_mask=target_feature_mask,
            )
            if name == "test":
                target_numpy[name] = target
                valid_numpy[name] = valid
            split_targets[name] = _tensor_from_numpy(target, device, dtype=torch.float32)
            split_valid[name] = _tensor_from_numpy(valid, device, dtype=torch.bool)

        model = ResidualStateMLP(
            input_dim=feature_count,
            output_dim=len(features.feature_names),
            hidden_dim=int(args.hidden_dim),
            layers=int(args.layers),
            dropout=float(args.dropout),
        ).to(device)
        parameter_count = int(sum(parameter.numel() for parameter in model.parameters()))
        model, fit_metadata = fit_model(
            model,
            context_tensors["fit"],
            split_targets["fit"],
            split_valid["fit"],
            context_tensors["validation"],
            split_targets["validation"],
            split_valid["validation"],
            args,
            device,
        )
        prediction = predict_batches(
            model,
            context_tensors["test"],
            int(args.batch_size),
            device.type == "cuda" and bool(args.amp),
        )
        if int(horizon) == 1:
            gap_index = features.feature_names.index("gap_open")
            test_next_open_gap_prediction = (
                prediction[:, gap_index] * float(features.train_std[gap_index])
                + float(features.train_mean[gap_index])
            )
        metrics, group_metrics, horizon_daily, horizon_features = evaluate_predictions(
            features,
            test_steps,
            horizon,
            prediction,
            target_numpy["test"],
            valid_numpy["test"],
            next_open_gap_prediction=test_next_open_gap_prediction,
        )
        for row in horizon_daily:
            row["state_target_scope"] = args.state_target_scope
            row["state_target_feature_count"] = int(target_feature_mask.sum())
        fit_metadata.update(
            {
                "parameter_count": parameter_count,
                "metrics": metrics,
                "feature_group_metrics": group_metrics,
            }
        )
        summary["horizons"][str(horizon)] = fit_metadata
        daily_rows.extend(horizon_daily)
        feature_rows.extend(horizon_features)
        torch.save(
            {
                "model_state_dict": {
                    name: value.detach().cpu() for name, value in model.state_dict().items()
                },
                "input_dim": feature_count,
                "output_dim": len(features.feature_names),
                "feature_names": list(features.feature_names),
                "base_feature_names": layout.base_feature_names,
                "graph_feature_names": (
                    [] if args.without_graph else layout.graph_feature_names
                ),
                "horizon": int(horizon),
                "state_target_scope": args.state_target_scope,
                "state_target_feature_count": int(target_feature_mask.sum()),
                "fit_metadata": fit_metadata,
            },
            output_dir / f"direct_state_mlp_h{horizon}.pt",
        )
        (output_dir / "summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        pd.DataFrame(daily_rows).to_csv(output_dir / "daily_metrics.csv", index=False)
        pd.DataFrame(feature_rows).to_csv(output_dir / "feature_metrics.csv", index=False)
        print(
            f"result horizon={horizon} state_skill={metrics['skill_vs_persistence']:+.6f} "
            f"return_ic={metrics['return_state_ic']['mean']:+.6f} "
            f"path_ic={metrics['return_path_ic']['mean']:+.6f}",
            flush=True,
        )
        del model, prediction, split_targets, split_valid, target_numpy, valid_numpy
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    print(f"wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
