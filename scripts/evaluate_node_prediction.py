from __future__ import annotations

import argparse
import csv
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.run_real_backtest import (
    date_indices,
    filter_history_for_training,
    parse_int_list,
    rollout_steps_for_offset,
)
from stock_v2.data_contract import validate_checkpoint_panel
from stock_v2.event_features import (
    build_event_feature_frames,
    build_event_theme_exposure,
    build_event_ticker_coverage,
)
from stock_v2.fundamental_features import build_fundamental_feature_frames, load_fundamental_observations
from stock_v2.external_factors import (
    POLICY_RATE_FACTOR_NAMES,
    build_external_feature_frames,
    build_external_node_feature_frames,
    fetch_external_factor_closes,
    resolve_external_factors,
)
from stock_v2.external_etf_nodes import (
    load_external_etf_node_inputs,
    merge_external_node_inputs,
)
from stock_v2.kiwoom_investor import build_investor_feature_frames, load_investor_flow_frames
from stock_v2.graph_jepa import StockGraphJEPA
from stock_v2.market_data import (
    fetch_krx_ohlcv,
    load_universe_manifest,
    make_ohlcv_panel,
    select_krx_universe_from_listing,
    select_universe,
)
from stock_v2.real_features import build_edge_tensor, build_feature_panel, make_real_snapshot
from stock_v2.static_edges import build_industry_edge_arrays, load_industry_codes


def as_namespace(mapping: dict[str, Any]) -> argparse.Namespace:
    return argparse.Namespace(**mapping)


def ablate_policy_rate_state(features) -> list[str]:
    policy_ids = {f"EXT:{name}" for name in POLICY_RATE_FACTOR_NAMES}
    node_ids = list(features.node_tickers or [])
    selected = [index for index, node_id in enumerate(node_ids) if node_id in policy_ids]
    if not selected:
        raise ValueError("policy-rate ablation requested but no policy-rate nodes were found")
    features.features[:, selected, :] = 0.0
    return [node_ids[index] for index in selected]


def event_edge_setting(ckpt_args: dict[str, Any], cli_args: argparse.Namespace, name: str, default: Any) -> Any:
    cli_value = getattr(cli_args, name, None)
    if cli_value is not None:
        return cli_value
    return ckpt_args.get(name, default)


def graph_edge_kwargs(ckpt_args: dict[str, Any], cli_args: argparse.Namespace) -> dict[str, float | int | str]:
    return {
        "correlation_mode": str(event_edge_setting(ckpt_args, cli_args, "edge_correlation_mode", "signed")),
        "event_top_k": int(event_edge_setting(ckpt_args, cli_args, "event_edge_top_k", 0) or 0),
        "event_min_weight": float(event_edge_setting(ckpt_args, cli_args, "event_edge_min_weight", 0.05)),
        "event_scale": float(event_edge_setting(ckpt_args, cli_args, "event_edge_scale", 0.25)),
        "partial_corr_top_k": int(event_edge_setting(ckpt_args, cli_args, "partial_corr_top_k", 0) or 0),
        "partial_corr_min_abs": float(event_edge_setting(ckpt_args, cli_args, "partial_corr_min_abs", 0.10)),
        "partial_corr_mode": str(event_edge_setting(ckpt_args, cli_args, "partial_corr_mode", "signed")),
        "partial_corr_scale": float(event_edge_setting(ckpt_args, cli_args, "partial_corr_scale", 0.50)),
        "lead_lag_top_k": int(event_edge_setting(ckpt_args, cli_args, "lead_lag_top_k", 0) or 0),
        "lead_lag_days": int(event_edge_setting(ckpt_args, cli_args, "lead_lag_days", 1) or 1),
        "lead_lag_min_abs_corr": float(event_edge_setting(ckpt_args, cli_args, "lead_lag_min_abs_corr", 0.08)),
        "lead_lag_mode": str(event_edge_setting(ckpt_args, cli_args, "lead_lag_mode", "signed")),
        "lead_lag_scale": float(event_edge_setting(ckpt_args, cli_args, "lead_lag_scale", 0.50)),
        "policy_rate_edge_scale": float(
            event_edge_setting(ckpt_args, cli_args, "policy_rate_edge_scale", 0.0)
        ),
        "ownership_edge_scale": float(
            event_edge_setting(ckpt_args, cli_args, "ownership_edge_scale", 0.0)
        ),
        "sequence_window": int(
            event_edge_setting(ckpt_args, cli_args, "sequence_window", 0) or 0
        ),
    }


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    x = a[valid].astype(np.float64)
    y = b[valid].astype(np.float64)
    sx = x.std()
    sy = y.std()
    if sx < 1e-12 or sy < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    x = a[valid].astype(np.float64)
    y = b[valid].astype(np.float64)
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    if denom < 1e-12:
        return float("nan")
    return float(np.dot(x, y) / denom)


def sign_accuracy(a: np.ndarray, b: np.ndarray, threshold: float) -> float:
    valid = np.isfinite(a) & np.isfinite(b) & (np.abs(b) >= threshold)
    if valid.sum() == 0:
        return float("nan")
    return float((np.sign(a[valid]) == np.sign(b[valid])).mean())


def feature_group_indices(feature_names: list[str]) -> dict[str, list[int]]:
    """Partition stock state features into interpretable forecast groups."""

    groups: dict[str, list[int]] = {
        "returns": [],
        "technical": [],
        "risk_liquidity": [],
        "market": [],
        "news": [],
        "fundamental": [],
        "investor": [],
        "other": [],
    }
    for index, name in enumerate(feature_names):
        if name.startswith("fund_"):
            groups["fundamental"].append(index)
        elif name.startswith("investor_"):
            groups["investor"].append(index)
        elif name.startswith("news_"):
            groups["news"].append(index)
        elif name.startswith("market_"):
            groups["market"].append(index)
        elif name.startswith("return_") or "relative_return" in name or "cs_rank_return" in name:
            groups["returns"].append(index)
        elif name.startswith("ma") or "drawdown" in name or "breakout" in name or "range_position" in name:
            groups["technical"].append(index)
        elif (
            "volatility" in name
            or "volume" in name
            or "value" in name
            or "amihud" in name
            or "range" in name
            or "gap_open" in name
            or "intraday" in name
        ):
            groups["risk_liquidity"].append(index)
        else:
            groups["other"].append(index)
    return {name: indices for name, indices in groups.items() if indices}


def top_finite_indices(values: np.ndarray, size: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    finite_indices = np.flatnonzero(np.isfinite(values))
    if size <= 0 or finite_indices.size == 0:
        return np.empty((0,), dtype=np.int64)
    order = np.argsort(values[finite_indices], kind="stable")[::-1]
    return finite_indices[order[: min(int(size), finite_indices.size)]].astype(np.int64)


def derive_entry_path_return(
    horizon_close_return: np.ndarray,
    next_open_gap: np.ndarray,
) -> np.ndarray:
    """Convert predicted close-to-close return and next-open gap to entry PnL."""

    horizon_close_return = np.asarray(horizon_close_return, dtype=np.float64)
    next_open_gap = np.asarray(next_open_gap, dtype=np.float64)
    if horizon_close_return.shape != next_open_gap.shape:
        raise ValueError("horizon return and next-open gap must have the same shape")
    denominator = 1.0 + next_open_gap
    valid = (
        np.isfinite(horizon_close_return)
        & np.isfinite(next_open_gap)
        & (denominator > 1e-6)
    )
    result = np.full(horizon_close_return.shape, np.nan, dtype=np.float64)
    result[valid] = (
        (1.0 + horizon_close_return[valid]) / denominator[valid] - 1.0
    )
    return result


def realized_entry_path_correlation_metrics(
    predicted_entry_path: np.ndarray,
    realized_path: np.ndarray,
    valid: np.ndarray,
    liquidity_subsets: dict[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Correlate a horizon return-state forecast with the executable path."""

    predicted_entry_path = np.asarray(predicted_entry_path)
    realized_path = np.asarray(realized_path)
    valid = np.asarray(valid, dtype=bool)
    if (
        predicted_entry_path.ndim != 1
        or realized_path.shape != predicted_entry_path.shape
        or valid.shape != realized_path.shape
    ):
        raise ValueError("predicted and realized paths must be aligned vectors")
    metrics = {
        "realized_entry_path_ic": pearson(
            predicted_entry_path[valid],
            realized_path[valid],
        )
    }
    subsets = liquidity_subsets or {}
    for size in (100, 300):
        indices = subsets.get(
            f"liquidity_top{size}:return_1d",
            np.empty(0, dtype=np.int64),
        )
        selected = np.zeros(predicted_entry_path.shape[0], dtype=bool)
        selected[np.asarray(indices, dtype=np.int64)] = True
        selected &= valid
        metrics[f"realized_entry_path_ic_top{size}"] = pearson(
            predicted_entry_path[selected],
            realized_path[selected],
        )
    return metrics


def future_state_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    current: np.ndarray,
    target_available: np.ndarray,
    current_available: np.ndarray | None = None,
) -> dict[str, float] | None:
    """Score future state where both persistence endpoints are observed."""

    if current_available is None:
        current_available = np.ones_like(target_available, dtype=bool)
    valid = (
        np.asarray(target_available, dtype=bool)
        & np.asarray(current_available, dtype=bool)
        & np.isfinite(prediction)
        & np.isfinite(target)
        & np.isfinite(current)
    )
    if not valid.any():
        return None
    pred = prediction[valid]
    observed_target = target[valid]
    observed_current = current[valid]
    err = pred - observed_target
    base_err = observed_current - observed_target
    zero_err = -observed_target
    model_sse = float(np.sum(err ** 2))
    persistence_sse = float(np.sum(base_err ** 2))
    zero_baseline_sse = float(np.sum(zero_err ** 2))
    prediction_sse = float(np.sum(pred ** 2))
    prediction_target_cross = float(np.sum(pred * observed_target))
    model_mse = model_sse / float(len(err))
    baseline_mse = persistence_sse / float(len(base_err))
    zero_baseline_mse = zero_baseline_sse / float(len(zero_err))
    pred_delta = pred - observed_current
    actual_delta = observed_target - observed_current
    return {
        "observed_cells": int(valid.sum()),
        "model_mae": float(np.mean(np.abs(err))),
        "persistence_mae": float(np.mean(np.abs(base_err))),
        "zero_baseline_mae": float(np.mean(np.abs(zero_err))),
        "model_rmse": float(np.sqrt(model_mse)),
        "persistence_rmse": float(np.sqrt(baseline_mse)),
        "zero_baseline_rmse": float(np.sqrt(zero_baseline_mse)),
        "mse_skill_vs_persistence": float(1.0 - model_mse / baseline_mse) if baseline_mse > 1e-12 else float("nan"),
        "mse_skill_vs_zero": (
            float(1.0 - model_mse / zero_baseline_mse)
            if zero_baseline_mse > 1e-12
            else float("nan")
        ),
        "model_sse": model_sse,
        "persistence_sse": persistence_sse,
        "zero_baseline_sse": zero_baseline_sse,
        "prediction_sse": prediction_sse,
        "prediction_target_cross": prediction_target_cross,
        "state_r2": r2_like(pred, observed_target),
        "target_corr": pearson(pred, observed_target),
        "target_cosine": cosine(pred, observed_target),
        "target_sign_accuracy_abs_ge_0_10": sign_accuracy(pred, observed_target, 0.10),
        "delta_corr": pearson(pred_delta, actual_delta),
        "delta_cosine": cosine(pred_delta, actual_delta),
        "delta_sign_accuracy_abs_ge_0_10": sign_accuracy(pred_delta, actual_delta, 0.10),
    }


def state_target_feature_mask(
    feature_names: Sequence[str],
    temporal_state_feature_weights: Any,
    scope: str,
) -> np.ndarray:
    """Resolve the exact feature set scored by a future-state challenger."""

    if scope == "all":
        return np.ones(len(feature_names), dtype=bool)
    if scope != "checkpoint_temporal":
        raise ValueError(f"unknown state target scope: {scope}")
    if temporal_state_feature_weights is None:
        raise ValueError(
            "checkpoint_temporal scope requires temporal state feature weights"
        )
    weights = temporal_state_feature_weights
    if torch.is_tensor(weights):
        weights = weights.detach().cpu().numpy()
    weights = np.asarray(weights, dtype=np.float32)
    if weights.shape != (len(feature_names),):
        raise ValueError(
            "temporal state feature weights do not match the feature schema"
        )
    if not np.isfinite(weights).all() or (weights < 0.0).any():
        raise ValueError(
            "temporal state feature weights must be finite and non-negative"
        )
    selected = weights > 0.0
    if not selected.any():
        raise ValueError("checkpoint temporal target scope is empty")
    return selected


def cross_sectional_state_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    target_available: np.ndarray,
    current_available: np.ndarray,
) -> dict[str, float] | None:
    """Score demeaned cross-sectional state, separating alpha from market level."""

    valid = (
        np.asarray(target_available, dtype=bool)
        & np.asarray(current_available, dtype=bool)
        & np.isfinite(prediction)
        & np.isfinite(target)
    )
    if valid.sum() < 3:
        return None
    pred = np.asarray(prediction, dtype=np.float64)[valid]
    observed = np.asarray(target, dtype=np.float64)[valid]
    pred = pred - pred.mean()
    observed = observed - observed.mean()
    prediction_sse = float(np.sum(pred ** 2))
    target_sse = float(np.sum(observed ** 2))
    cross = float(np.sum(pred * observed))
    model_sse = float(np.sum((pred - observed) ** 2))
    return {
        "cs_observed_cells": int(valid.sum()),
        "cs_prediction_sse": prediction_sse,
        "cs_target_sse": target_sse,
        "cs_prediction_target_cross": cross,
        "cs_model_sse": model_sse,
        "cs_skill_vs_zero": (
            float(1.0 - model_sse / target_sse) if target_sse > 1e-12 else float("nan")
        ),
    }


def r2_like(pred: np.ndarray, target: np.ndarray) -> float:
    valid = np.isfinite(pred) & np.isfinite(target)
    if valid.sum() < 3:
        return float("nan")
    y = target[valid].astype(np.float64)
    p = pred[valid].astype(np.float64)
    denom = ((y - y.mean()) ** 2).sum()
    if denom < 1e-12:
        return float("nan")
    return float(1.0 - ((p - y) ** 2).sum() / denom)


def aggregate(values: list[float]) -> dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=np.float64)
    if arr.size == 0:
        return {"mean": float("nan"), "median": float("nan"), "count": 0}
    return {"mean": float(arr.mean()), "median": float(np.median(arr)), "count": int(arr.size)}


def load_model(model_dir: Path, device: torch.device) -> tuple[StockGraphJEPA, dict[str, Any]]:
    ckpt = torch.load(model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False)
    ckpt_args = dict(ckpt.get("args", {}))
    model = StockGraphJEPA(
        num_features=len(ckpt["feature_names"]),
        hidden_dim=int(ckpt_args.get("hidden_dim", 128)),
        num_layers=int(ckpt_args.get("layers", 3)),
        ema_decay=float(ckpt_args.get("ema_decay", 0.98)),
        latent_loss_weight=float(ckpt_args.get("latent_loss_weight", 1.0)),
        state_loss_weight=float(ckpt_args.get("state_loss_weight", 0.35)),
        current_imputation_loss_weight=float(
            ckpt_args.get("current_imputation_loss_weight", 0.0)
        ),
        hidden_completion_loss_weight=float(
            ckpt_args.get("hidden_completion_weight", 0.0)
        ),
        temporal_state_mode=str(ckpt_args.get("temporal_state_mode", "direct")),
        feature_names=[n for n in ckpt["feature_names"] if not str(n).startswith("fund_yoy_")],
        temporal_residual_short_steps=int(ckpt_args.get("temporal_residual_short_steps", 2)),
        temporal_head_steps=ckpt_args.get("temporal_head_steps"),
        temporal_state_feature_weights=ckpt.get("temporal_state_feature_weights"),
        temporal_state_context_skip=bool(
            ckpt_args.get("temporal_state_context_skip", False)
        ),
        # 없으면 위 불리언에서 유도된다(하위호환). 이걸 빼먹으면 context 모드로
        # 학습한 체크포인트를 폭 2 로 재구성해 load_state_dict 가 크기 불일치로 죽는다.
        temporal_head_input=ckpt_args.get("temporal_head_input"),
        hybrid_fast_direct=bool(ckpt_args.get("hybrid_fast_direct", False)),
        return_correlation_loss_weight=float(
            ckpt_args.get("return_correlation_loss_weight", 0.0)
        ),
        entry_path_correlation_loss_weight=float(
            ckpt_args.get("entry_path_correlation_loss_weight", 0.0)
        ),
        feature_means=ckpt.get("train_mean"),
        feature_stds=ckpt.get("train_std"),
        normalize_predictor_output=bool(ckpt_args.get("normalize_predictor_output", False)),
        sequence_window=int(ckpt_args.get("sequence_window", 0) or 0),
        sequence_layers=int(ckpt_args.get("sequence_layers", 2) or 2),
        sequence_heads=int(ckpt_args.get("sequence_heads", 8) or 8),
        sequence_residual=bool(ckpt_args.get("sequence_residual", False)),
        graph_neighbor_scale=float(ckpt_args.get("graph_neighbor_scale", 1.0)),
        temporal_graph_neighbor_scale=ckpt_args.get("temporal_graph_neighbor_scale"),
        temporal_stock_edge_scale=float(
            ckpt_args.get("temporal_stock_edge_scale", 1.0)
        ),
        global_stock_context=bool(
            ckpt_args.get("global_stock_context", False)
        ),
        downstream_auxiliary_loss_weight=float(
            ckpt_args.get("downstream_auxiliary_loss_weight", 0.0)
        ),
        downstream_auxiliary_task_weights=ckpt_args.get(
            "downstream_auxiliary_task_weights"
        ),
        downstream_market_loss_weight=float(
            ckpt_args.get("downstream_market_loss_weight", 0.0)
        ),
        downstream_market_cost_bps=float(
            ckpt_args.get("downstream_market_cost_bps", 50.0)
        ),
        downstream_transition_loss_weight=float(
            ckpt_args.get("downstream_transition_loss_weight", 0.0)
        ),
        downstream_transition_pooling=str(
            ckpt_args.get("downstream_transition_pooling", "mean")
        ),
        temporal_impact_loss_mix=float(
            ckpt_args.get("temporal_impact_loss_mix", 0.0)
        ),
    ).to(device)
    model.load_state_dict(ckpt["model_state"], strict=True)
    model.eval()
    return model, ckpt


def validate_future_rollout_contract(
    ckpt_args: dict[str, Any],
    horizons: list[int],
    allow_extrapolated_horizons: bool,
) -> None:
    if ckpt_args.get("pretrain_task") != "temporal":
        raise ValueError(
            "future rollout evaluation requires a checkpoint trained with --pretrain-task temporal"
        )
    configured = ckpt_args.get("rollout_offsets", [])
    trained_offsets = set(parse_int_list(configured) if isinstance(configured, str) else configured)
    missing = sorted(set(horizons) - {int(offset) for offset in trained_offsets})
    if missing and not allow_extrapolated_horizons:
        raise ValueError(
            "future rollout horizons were not directly supervised during training: "
            f"{missing}. Pass --allow-extrapolated-horizons only for exploratory evaluation."
        )


def build_features_from_ckpt(ckpt: dict[str, Any], cli_args: argparse.Namespace):
    ckpt_args = dict(ckpt.get("args", {}))
    checkpoint_tickers = list(ckpt.get("tickers", []))
    checkpoint_names = dict(ckpt.get("names", {}))
    if checkpoint_tickers and not cli_args.override_universe:
        universe = [(ticker, checkpoint_names.get(ticker, ticker)) for ticker in checkpoint_tickers]
    elif cli_args.universe_manifest:
        universe = load_universe_manifest(cli_args.universe_manifest)
    else:
        max_tickers = int(cli_args.max_tickers or ckpt_args.get("max_tickers", len(checkpoint_tickers) or 100))
        universe_name = str(cli_args.universe or ckpt_args.get("universe", "krx"))
        if universe_name == "krx":
            universe = select_krx_universe_from_listing(max_tickers)
        else:
            universe = select_universe(max_tickers)
    names = dict(universe)
    start = cli_args.start or ckpt_args.get("start", "2020-01-01")
    end = cli_args.end or ckpt_args.get("end", None)
    train_end = cli_args.train_end or ckpt_args.get("train_end", "2023-12-29")
    edge_window = int(cli_args.edge_window or ckpt_args.get("edge_window", 60))
    min_train_rows_value = cli_args.min_train_rows if cli_args.min_train_rows is not None else ckpt_args.get("min_train_rows")
    if min_train_rows_value is None:
        min_train_rows_value = max(260, edge_window + 120)
    min_train_rows = int(min_train_rows_value)
    raw = fetch_krx_ohlcv(
        universe=universe,
        start=start,
        end=end,
        cache_dir=cli_args.cache_dir or ckpt_args.get("cache_dir", "data/cache"),
        refresh=cli_args.refresh,
    )
    raw = filter_history_for_training(raw, train_end=train_end, min_train_rows=min_train_rows)
    panel = make_ohlcv_panel(raw, names=names)
    horizons = parse_int_list(cli_args.horizons)
    event_paths = list(cli_args.event_path or ckpt_args.get("event_path", []) or [])
    event_feature_frames = None
    event_feature_names: list[str] = []
    event_ticker_coverage = None
    event_coverage_mode = str(
        ckpt_args.get("event_coverage_mode", "legacy_all_observed")
    )
    event_theme_exposure = None
    event_theme_names = []
    fundamental_feature_frames = None
    investor_feature_frames = None
    if event_paths:
        event_feature_frames = build_event_feature_frames(
            dates=panel.close.index,
            tickers=panel.tickers,
            event_paths=event_paths,
            half_life_days=float(cli_args.event_half_life_days or ckpt_args.get("event_half_life_days", 5.0)),
            lag_days=int(cli_args.event_lag_days if cli_args.event_lag_days is not None else ckpt_args.get("event_lag_days", 1)),
            max_decay_days=int(cli_args.event_max_decay_days if cli_args.event_max_decay_days is not None else ckpt_args.get("event_max_decay_days", 60)),
        )
        event_feature_names = list(event_feature_frames)
        if event_coverage_mode == "mask_uncovered":
            event_ticker_coverage = build_event_ticker_coverage(
                dates=panel.close.index,
                tickers=panel.tickers,
                event_paths=event_paths,
            )
        if int(event_edge_setting(ckpt_args, cli_args, "event_edge_top_k", 0) or 0) > 0:
            event_theme_exposure, event_theme_names = build_event_theme_exposure(
                dates=panel.close.index,
                tickers=panel.tickers,
                event_paths=event_paths,
                half_life_days=float(cli_args.event_half_life_days or ckpt_args.get("event_half_life_days", 5.0)),
                lag_days=int(cli_args.event_lag_days if cli_args.event_lag_days is not None else ckpt_args.get("event_lag_days", 1)),
                max_decay_days=int(cli_args.event_max_decay_days if cli_args.event_max_decay_days is not None else ckpt_args.get("event_max_decay_days", 60)),
                max_themes=int(event_edge_setting(ckpt_args, cli_args, "event_edge_max_themes", 96)),
                min_theme_count=int(event_edge_setting(ckpt_args, cli_args, "event_edge_min_theme_count", 2)),
            )
    fundamental_paths = list(cli_args.fundamental_path or ckpt_args.get("fundamental_path", []) or [])
    if fundamental_paths:
        fundamental_feature_frames = build_fundamental_feature_frames(
            dates=panel.close.index,
            tickers=panel.tickers,
            observations=load_fundamental_observations(fundamental_paths),
            lag_days=int(
                cli_args.fundamental_lag_days
                if cli_args.fundamental_lag_days is not None
                else ckpt_args.get("fundamental_lag_days", 1)
            ),
        )
    investor_cache_dir = cli_args.investor_cache_dir or ckpt_args.get("investor_cache_dir")
    if investor_cache_dir:
        investor_flow_frames = load_investor_flow_frames(
            cache_dir=investor_cache_dir,
            dates=panel.close.index,
            tickers=panel.tickers,
        )
        observed_close = panel.close.where(panel.price_observed)
        observed_volume = panel.volume.where(panel.price_observed)
        investor_feature_frames = build_investor_feature_frames(
            investor_flow_frames,
            traded_value=observed_close * observed_volume,
            lag_days=int(
                cli_args.investor_flow_lag_days
                if cli_args.investor_flow_lag_days is not None
                else ckpt_args.get("investor_flow_lag_days", 1)
            ),
        )
    cli_external_symbols = list(cli_args.external_symbol or [])
    ckpt_external_symbols = list(ckpt_args.get("external_symbol", []) or [])
    external_symbols = cli_external_symbols if cli_external_symbols else ckpt_external_symbols
    external_preset = cli_args.external_preset
    if external_preset is None:
        external_preset = ckpt_args.get("external_preset", "none")
    external_node_mode = str(ckpt_args.get("external_node_mode", "features") or "features")
    external_factors = resolve_external_factors(str(external_preset or "none"), external_symbols)
    external_node_feature_frames = None
    external_node_returns = None
    external_node_names = {}
    if external_factors:
        factor_closes = fetch_external_factor_closes(
            external_factors,
            start=start,
            end=end,
            cache_dir=cli_args.external_cache_dir or ckpt_args.get("external_cache_dir", "data/external_cache"),
            refresh=cli_args.refresh,
        )
        external_lag_days = int(
            cli_args.external_lag_days
            if cli_args.external_lag_days is not None
            else ckpt_args.get("external_lag_days", 1)
        )
        if external_node_mode in {"features", "both"}:
            external_feature_frames = build_external_feature_frames(
                dates=panel.close.index,
                tickers=panel.tickers,
                factor_closes=factor_closes,
                lag_days=external_lag_days,
            )
            if external_feature_frames:
                event_feature_frames = dict(event_feature_frames or {})
                event_feature_frames.update(external_feature_frames)
        if external_node_mode in {"nodes", "both"}:
            external_node_feature_frames, external_node_returns, external_node_names = build_external_node_feature_frames(
                dates=panel.close.index,
                factor_closes=factor_closes,
                lag_days=external_lag_days,
            )
    external_etf_panel = (
        cli_args.external_etf_panel
        if cli_args.external_etf_panel is not None
        else ckpt_args.get("external_etf_panel")
    )
    if external_etf_panel:
        etf_inputs = load_external_etf_node_inputs(
            external_etf_panel,
            panel.close.index,
            krx_cutoff_local_time="15:30",
        )
        (
            external_node_feature_frames,
            external_node_returns,
            external_node_names,
        ) = merge_external_node_inputs(
            external_node_feature_frames,
            external_node_returns,
            external_node_names,
            etf_inputs,
        )
    industry_profile_paths = list(cli_args.industry_profile_path or ckpt_args.get("industry_profile_path", []) or [])
    static_edge_index = None
    static_edge_weight = None
    if industry_profile_paths:
        industry_codes = load_industry_codes(industry_profile_paths)
        static_edge_index, static_edge_weight, _industry_stats = build_industry_edge_arrays(
            panel.tickers,
            industry_codes,
            prefix_length=int(
                cli_args.industry_prefix_length
                if cli_args.industry_prefix_length is not None
                else ckpt_args.get("industry_prefix_length", 2)
            ),
            scale=float(
                cli_args.industry_edge_scale
                if cli_args.industry_edge_scale is not None
                else ckpt_args.get("industry_edge_scale", 0.20)
            ),
        )
    features = build_feature_panel(
        panel,
        horizon=max(horizons),
        train_end=train_end,
        require_targets=False,
        feature_names=[
            n for n in ckpt["feature_names"]
            if not str(n).startswith(("fund_yoy_", "earn_", "retlag_"))
        ],
        event_feature_frames=event_feature_frames,
        event_feature_names=event_feature_names,
        event_ticker_coverage=event_ticker_coverage,
        fundamental_feature_frames=fundamental_feature_frames,
        investor_feature_frames=investor_feature_frames,
        external_node_feature_frames=external_node_feature_frames,
        external_node_returns=external_node_returns,
        external_node_names=external_node_names,
        event_theme_exposure=event_theme_exposure,
        event_theme_names=event_theme_names,
        static_edge_index=static_edge_index,
        static_edge_weight=static_edge_weight,
        path_horizons=horizons,
    )
    _fy_path = dict(ckpt.get("args", {})).get("fund_yoy_input_path")
    if _fy_path:
        from stock_v2.fund_yoy_inputs import augment_panel_with_fund_yoy
        features = augment_panel_with_fund_yoy(
            features, _fy_path,
            str(dict(ckpt.get("args", {})).get("fund_yoy_input_mode", "own")),
            str(train_end),
        )
    # Feature-panel augmentations that ran at training time must be replayed here,
    # or the rebuilt panel is narrower than the checkpoint and the encoder errors.
    if bool(ckpt_args.get("earnings_features", False)):
        from stock_v2.earnings_features import augment_panel_with_earnings
        _fp = ckpt_args.get("fundamental_path")
        _fp = _fp[0] if isinstance(_fp, list) and _fp else _fp
        features = augment_panel_with_earnings(
            features, _fp,
            horizon=int(ckpt_args.get("horizon", 10) or 10),
            train_end=str(ckpt_args.get("train_end") or ""),
        )
    _nlags = int(ckpt_args.get("return_lag_features", 0) or 0)
    if _nlags > 0:
        from stock_v2.earnings_features import augment_panel_with_return_lags
        features = augment_panel_with_return_lags(
            features, n_lags=_nlags, train_end=str(ckpt_args.get("train_end") or ""))
    validate_checkpoint_panel(
        ckpt,
        features,
        train_end,
        allow_unverified_legacy=cli_args.allow_unverified_legacy,
    )
    # Ownership edges must be reattached at eval time or the graph the encoder was
    # trained with silently degrades to the correlation-only one.
    _own = ckpt_args.get("ownership_edge_path") or getattr(cli_args, "ownership_edge_path", None)
    if _own and float(ckpt_args.get("ownership_edge_scale", 0.0) or 0.0) > 0.0:
        from stock_v2.ownership_edges import attach_ownership_edges
        features = attach_ownership_edges(features, _own)
    return features, ckpt_args


def select_steps(features, ckpt_args: dict[str, Any], cli_args: argparse.Namespace) -> np.ndarray:
    train_end = cli_args.train_end or ckpt_args.get("train_end", "2023-12-29")
    edge_window = int(cli_args.edge_window or ckpt_args.get("edge_window", 60))
    max_horizon = max(parse_int_list(cli_args.horizons))
    test_indices = date_indices(
        features.dates,
        start=(pd.Timestamp(train_end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    test_indices = test_indices[(test_indices >= edge_window) & (test_indices <= len(features.dates) - 1 - max_horizon)]
    if cli_args.max_steps and len(test_indices) > cli_args.max_steps:
        positions = np.linspace(0, len(test_indices) - 1, cli_args.max_steps).round().astype(int)
        test_indices = test_indices[positions]
    return test_indices


def build_evaluation_edge_cache(features, steps, ckpt_args, cli_args):
    """Build each causal graph edge set once for both evaluation passes."""

    worker_count = max(0, int(cli_args.edge_cache_workers))
    if worker_count == 0:
        return None
    edge_window = int(cli_args.edge_window or ckpt_args.get("edge_window", 60))
    edge_top_k = int(cli_args.edge_top_k or ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(
        cli_args.min_abs_corr if cli_args.min_abs_corr is not None else ckpt_args.get("min_abs_corr", 0.2)
    )
    edge_kwargs = graph_edge_kwargs(ckpt_args, cli_args)
    step_values = np.unique(np.asarray(steps, dtype=np.int64)).tolist()

    def build_one(step: int):
        return int(step), build_edge_tensor(
            features,
            step=int(step),
            edge_window=edge_window,
            top_k=edge_top_k,
            min_abs_corr=min_abs_corr,
            **edge_kwargs,
        )

    workers = min(worker_count, len(step_values))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            pairs = list(executor.map(build_one, step_values))
    else:
        pairs = [build_one(step) for step in step_values]
    cache = dict(pairs)
    print(
        f"evaluation edge cache: steps={len(cache)} "
        f"edges={sum(int(weight.numel()) for _index, weight in cache.values())} workers={workers}",
        flush=True,
    )
    return cache


def latent_snapshot_metrics(latent: torch.Tensor) -> dict[str, float]:
    """Measure cross-node latent diversity without an expensive full SVD."""

    values = latent.detach().float()
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("latent health requires at least two node vectors")
    centered = values - values.mean(dim=0, keepdim=True)
    std = centered.pow(2).mean(dim=0).sqrt()
    variance = std.square()
    normalized = torch.nn.functional.normalize(values, dim=-1)
    vector_sum = normalized.sum(dim=0)
    node_count = values.shape[0]
    mean_pairwise_cosine = (
        vector_sum.square().sum() - float(node_count)
    ) / float(node_count * (node_count - 1))
    participation = variance.sum().square() / variance.square().sum().clamp_min(1e-12)
    feature_count = float(values.shape[1])
    return {
        "latent_norm_mean": float(values.norm(dim=-1).mean().cpu()),
        "cross_node_std_mean": float(std.mean().cpu()),
        "active_dimension_fraction": float((std > 1e-3).float().mean().cpu()),
        "variance_participation_ratio": float((participation / feature_count).cpu()),
        "mean_pairwise_cosine": float(mean_pairwise_cosine.cpu()),
    }


def evaluate_latent_health(model, features, steps, ckpt_args, cli_args, device, edge_cache=None):
    sample_count = min(12, len(steps))
    positions = np.linspace(0, len(steps) - 1, sample_count).round().astype(int)
    sampled_steps = np.asarray(steps, dtype=np.int64)[positions]
    horizons = parse_int_list(cli_args.horizons)
    rollout_args = dict(ckpt_args)
    rollout_args.setdefault("temporal_offset", ckpt_args.get("horizon", max(horizons)))
    rollout_args.setdefault("latent_rollout_steps", 1)
    ns = as_namespace(rollout_args)
    rows = []
    for step in sampled_steps:
        batch = make_real_snapshot(
            features,
            step=int(step),
            full_observation=True,
            edge_window=int(cli_args.edge_window or ckpt_args.get("edge_window", 60)),
            top_k=int(cli_args.edge_top_k or ckpt_args.get("edge_top_k", 6)),
            min_abs_corr=float(
                cli_args.min_abs_corr
                if cli_args.min_abs_corr is not None
                else ckpt_args.get("min_abs_corr", 0.2)
            ),
            **graph_edge_kwargs(ckpt_args, cli_args),
            edge_cache=edge_cache,
        ).to(device)
        with torch.no_grad():
            context = model.encode_temporal_context(batch)
        stock_context = context[: features.tradable_count]
        rows.append(
            {
                "date": str(features.dates[int(step)].date()),
                "state": "context",
                "horizon": 0,
                "displacement_ratio": 0.0,
                **latent_snapshot_metrics(stock_context),
            }
        )
        context_rms = stock_context.square().mean().sqrt().clamp_min(1e-12)
        for horizon in horizons:
            rollout_steps = rollout_steps_for_offset(ns, int(horizon))
            with torch.no_grad():
                predicted = model.rollout_latent(context, steps=rollout_steps)
            stock_predicted = predicted[: features.tradable_count]
            displacement = (stock_predicted - stock_context).square().mean().sqrt() / context_rms
            rows.append(
                {
                    "date": str(features.dates[int(step)].date()),
                    "state": f"h{horizon}",
                    "horizon": int(horizon),
                    "displacement_ratio": float(displacement.cpu()),
                    **latent_snapshot_metrics(stock_predicted),
                }
            )
    return rows


def evaluate_current_imputation(model, features, steps, ckpt_args, cli_args, device, edge_cache=None):
    rows = []
    edge_window = int(cli_args.edge_window or ckpt_args.get("edge_window", 60))
    edge_top_k = int(cli_args.edge_top_k or ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(cli_args.min_abs_corr if cli_args.min_abs_corr is not None else ckpt_args.get("min_abs_corr", 0.2))
    hide_ratio = float(cli_args.hide_ratio if cli_args.hide_ratio is not None else ckpt_args.get("hide_ratio", 0.3))
    for ordinal, step in enumerate(steps):
        batch = make_real_snapshot(
            features,
            step=int(step),
            hide_ratio=hide_ratio,
            edge_window=edge_window,
            top_k=edge_top_k,
            min_abs_corr=min_abs_corr,
            **graph_edge_kwargs(ckpt_args, cli_args),
            seed=cli_args.seed + ordinal,
            full_observation=False,
            mask_strategy=cli_args.mask_strategy,
            edge_cache=edge_cache,
        ).to(device)
        with torch.no_grad():
            pred = model.infer_unobserved_state(batch).detach().cpu().numpy()
        stock_count = features.tradable_count
        pred = pred[:stock_count]
        target = batch.node_features.detach().cpu().numpy()[:stock_count]
        hidden = (batch.feature_mask.detach().cpu().numpy() < 0.5)[:stock_count]
        available = (batch.available_mask.detach().cpu().numpy() > 0.5)[:stock_count]
        mask = hidden & available
        if not mask.any():
            continue
        err = pred[mask] - target[mask]
        base_err = 0.0 - target[mask]
        model_mse = float(np.mean(err ** 2))
        baseline_mse = float(np.mean(base_err ** 2))
        rows.append({
            "date": str(features.dates[int(step)].date()),
            "hidden_cells": int(mask.sum()),
            "model_mae": float(np.mean(np.abs(err))),
            "baseline_zero_mae": float(np.mean(np.abs(base_err))),
            "model_rmse": float(np.sqrt(model_mse)),
            "baseline_zero_rmse": float(np.sqrt(baseline_mse)),
            "mse_skill_vs_zero": float(1.0 - model_mse / baseline_mse) if baseline_mse > 1e-12 else float("nan"),
            "r2_hidden": r2_like(pred[mask], target[mask]),
        })
    return rows


def evaluate_future_rollout(
    model,
    features,
    steps,
    ckpt_args,
    cli_args,
    device,
    edge_cache=None,
    return_forecast_writer=None,
):
    rows = []
    feature_rows = []
    horizons = parse_int_list(cli_args.horizons)
    edge_window = int(cli_args.edge_window or ckpt_args.get("edge_window", 60))
    edge_top_k = int(cli_args.edge_top_k or ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(cli_args.min_abs_corr if cli_args.min_abs_corr is not None else ckpt_args.get("min_abs_corr", 0.2))
    rollout_args = dict(ckpt_args)
    rollout_args.setdefault("pretrain_task", ckpt_args.get("pretrain_task", "temporal"))
    rollout_args.setdefault("temporal_offset", ckpt_args.get("temporal_offset", ckpt_args.get("horizon", max(horizons))))
    rollout_args.setdefault("latent_rollout_steps", ckpt_args.get("latent_rollout_steps", 1))
    ns = as_namespace(rollout_args)
    target_feature_mask = state_target_feature_mask(
        features.feature_names,
        model.temporal_state_feature_weights,
        cli_args.state_target_scope,
    )
    target_feature_count = int(target_feature_mask.sum())
    groups = feature_group_indices(features.feature_names)
    diagnostic_features = (
        "return_1d",
        "return_2d",
        "return_3d",
        "return_5d",
        "return_10d",
        "return_20d",
        "gap_open",
        "intraday_return",
        "cs_rank_return_20d",
        "volatility_20d",
        "downside_volatility_20d",
    )
    groups.update(
        {
            f"feature:{name}": [features.feature_names.index(name)]
            for name in diagnostic_features
            if name in features.feature_names
        }
    )
    return_1d_index = (
        features.feature_names.index("return_1d")
        if "return_1d" in features.feature_names
        else None
    )
    if "gap_open" not in features.feature_names:
        raise ValueError("future entry-path evaluation requires gap_open state")
    gap_open_index = features.feature_names.index("gap_open")
    if "intraday_return" not in features.feature_names:
        raise ValueError("future entry-path evaluation requires intraday_return state")
    intraday_return_index = features.feature_names.index("intraday_return")
    liquidity_index = (
        features.feature_names.index("value_ma20_log")
        if "value_ma20_log" in features.feature_names
        else None
    )
    for step in steps:
        batch = make_real_snapshot(
            features,
            step=int(step),
            full_observation=True,
            edge_window=edge_window,
            top_k=edge_top_k,
            min_abs_corr=min_abs_corr,
            **graph_edge_kwargs(ckpt_args, cli_args),
            edge_cache=edge_cache,
        ).to(device)
        with torch.no_grad():
            context = model.encode_temporal_context(batch)
        stock_count = features.tradable_count
        x0 = features.features[int(step), :stock_count]
        next_open_gap_prediction: np.ndarray | None = None
        liquidity_subsets: dict[str, np.ndarray] = {}
        if return_1d_index is not None and liquidity_index is not None:
            liquidity_values = features.raw_features[
                int(step), :stock_count, liquidity_index
            ]
            liquidity_subsets = {
                f"liquidity_top{size}:return_1d": top_finite_indices(
                    liquidity_values,
                    size,
                )
                for size in (100, 300)
            }
        for horizon in horizons:
            target = features.features[int(step) + int(horizon), :stock_count]
            horizon_return_name = f"return_{int(horizon)}d"
            if horizon_return_name not in features.feature_names:
                raise ValueError(
                    f"future evaluation requires horizon state {horizon_return_name}"
                )
            horizon_return_index = features.feature_names.index(horizon_return_name)
            steps_forward = rollout_steps_for_offset(ns, int(horizon)) if hasattr(model, "rollout_latent") else 1
            with torch.no_grad():
                z_pred = model.rollout_latent(context, steps=max(1, int(steps_forward)))
                pred = model.predict_temporal_state(
                    batch,
                    z_pred,
                    rollout_steps=max(1, int(steps_forward)),
                    z_context=context,
                ).detach().cpu().numpy()[:stock_count]
                pred_no_rollout = model.predict_temporal_state(
                    batch,
                    context,
                    rollout_steps=max(1, int(steps_forward)),
                    z_context=context,
                ).detach().cpu().numpy()[:stock_count]
            target_available = features.available_mask[int(step) + int(horizon), :stock_count] > 0.5
            current_available = features.available_mask[int(step), :stock_count] > 0.5
            scored_target_available = target_available & target_feature_mask[None, :]
            scored_current_available = current_available & target_feature_mask[None, :]
            horizon_close_prediction = (
                pred[:, horizon_return_index]
                * float(features.train_std[horizon_return_index])
                + float(features.train_mean[horizon_return_index])
            )
            if int(horizon) == 1:
                next_open_gap_prediction = (
                    pred[:, gap_open_index] * float(features.train_std[gap_open_index])
                    + float(features.train_mean[gap_open_index])
                )
            if next_open_gap_prediction is None:
                raise RuntimeError("horizon 1 must be evaluated before longer entry paths")
            if int(horizon) == 1:
                predicted_entry_path = (
                    pred[:, intraday_return_index]
                    * float(features.train_std[intraday_return_index])
                    + float(features.train_mean[intraday_return_index])
                )
            else:
                predicted_entry_path = derive_entry_path_return(
                    horizon_close_prediction,
                    next_open_gap_prediction,
                )
            path_returns = features.target_return_paths.get(int(horizon))
            if path_returns is None:
                raise ValueError(f"future evaluation is missing entry path horizon {horizon}")
            realized_path = path_returns[int(step), :stock_count]
            path_valid = (
                target_available[:, horizon_return_index]
                & current_available[:, horizon_return_index]
                & np.isfinite(predicted_entry_path)
                & np.isfinite(realized_path)
            )
            path_metrics = realized_entry_path_correlation_metrics(
                predicted_entry_path,
                realized_path,
                path_valid,
                liquidity_subsets,
            )
            if return_forecast_writer is not None and return_1d_index is not None:
                return_valid = (
                    target_available[:, return_1d_index]
                    & current_available[:, return_1d_index]
                    & np.isfinite(pred[:, return_1d_index])
                    & np.isfinite(target[:, return_1d_index])
                    & np.isfinite(x0[:, return_1d_index])
                )
                return_scale = float(features.train_std[return_1d_index])
                return_mean = float(features.train_mean[return_1d_index])
                optional_forecast_names = (
                    "return_5d",
                    "cs_rank_return_20d",
                    "volatility_20d",
                    "downside_volatility_20d",
                )
                optional_forecasts = {}
                for feature_name in optional_forecast_names:
                    if feature_name not in features.feature_names:
                        continue
                    feature_index = features.feature_names.index(feature_name)
                    optional_forecasts[feature_name] = (
                        pred[:, feature_index] * float(features.train_std[feature_index])
                        + float(features.train_mean[feature_index])
                    )
                liquidity_values = (
                    features.raw_features[int(step), :stock_count, liquidity_index]
                    if liquidity_index is not None
                    else np.full(stock_count, np.nan, dtype=np.float32)
                )
                forecast_rows = []
                for node_index in np.flatnonzero(return_valid):
                    realized_path_return = (
                        float(path_returns[int(step), int(node_index)])
                        if np.isfinite(path_returns[int(step), int(node_index)])
                        else float("nan")
                    )
                    forecast_rows.append(
                        {
                            "date": str(features.dates[int(step)].date()),
                            "target_date": str(
                                features.dates[int(step) + int(horizon)].date()
                            ),
                            "horizon": int(horizon),
                            "ticker": features.tickers[int(node_index)],
                            "prediction_return_1d": float(
                                pred[int(node_index), return_1d_index] * return_scale
                                + return_mean
                            ),
                            "target_return_1d": float(
                                target[int(node_index), return_1d_index] * return_scale
                                + return_mean
                            ),
                            "current_return_1d": float(
                                x0[int(node_index), return_1d_index] * return_scale
                                + return_mean
                            ),
                            "realized_path_return": realized_path_return,
                            "prediction_entry_path_return": float(
                                predicted_entry_path[int(node_index)]
                            ),
                            "current_value_ma20_log": float(
                                liquidity_values[int(node_index)]
                            ),
                            **{
                                f"prediction_{feature_name}": float(
                                    values[int(node_index)]
                                )
                                for feature_name, values in optional_forecasts.items()
                            },
                        }
                    )
                return_forecast_writer.writerows(forecast_rows)
            metrics = future_state_metrics(
                pred,
                target,
                x0,
                scored_target_available,
                scored_current_available,
            )
            no_rollout_metrics = future_state_metrics(
                pred_no_rollout,
                target,
                x0,
                scored_target_available,
                scored_current_available,
            )
            if metrics is None or no_rollout_metrics is None:
                continue
            no_rollout_sse = float(no_rollout_metrics["model_sse"])
            rollout_dependency = {
                "no_rollout_model_sse": no_rollout_sse,
                "rollout_mse_skill_vs_no_rollout": (
                    float(1.0 - float(metrics["model_sse"]) / no_rollout_sse)
                    if no_rollout_sse > 1e-12
                    else float("nan")
                ),
            }
            rows.append({
                "date": str(features.dates[int(step)].date()),
                "horizon": int(horizon),
                "rollout_steps": int(max(1, steps_forward)),
                "state_target_scope": cli_args.state_target_scope,
                "state_target_feature_count": target_feature_count,
                **path_metrics,
                **rollout_dependency,
                **metrics,
            })
            for group_name, indices in groups.items():
                group_metrics = future_state_metrics(
                    pred[:, indices],
                    target[:, indices],
                    x0[:, indices],
                    target_available[:, indices],
                    current_available[:, indices],
                )
                if group_metrics is None:
                    continue
                row = {
                    "date": str(features.dates[int(step)].date()),
                    "horizon": int(horizon),
                    "rollout_steps": int(max(1, steps_forward)),
                    "feature_group": group_name,
                    **group_metrics,
                }
                if group_name == "feature:return_1d":
                    cs_metrics = cross_sectional_state_metrics(
                        pred[:, indices],
                        target[:, indices],
                        target_available[:, indices],
                        current_available[:, indices],
                    )
                    if cs_metrics is not None:
                        row.update(cs_metrics)
                feature_rows.append(row)
            if return_1d_index is not None:
                for group_name, node_indices in liquidity_subsets.items():
                    if node_indices.size == 0:
                        continue
                    liquid_metrics = future_state_metrics(
                        pred[node_indices, return_1d_index : return_1d_index + 1],
                        target[node_indices, return_1d_index : return_1d_index + 1],
                        x0[node_indices, return_1d_index : return_1d_index + 1],
                        target_available[
                            node_indices,
                            return_1d_index : return_1d_index + 1,
                        ],
                        current_available[
                            node_indices,
                            return_1d_index : return_1d_index + 1,
                        ],
                    )
                    if liquid_metrics is not None:
                        row = {
                            "date": str(features.dates[int(step)].date()),
                            "horizon": int(horizon),
                            "rollout_steps": int(max(1, steps_forward)),
                            "feature_group": group_name,
                            **liquid_metrics,
                        }
                        cs_metrics = cross_sectional_state_metrics(
                            pred[node_indices, return_1d_index : return_1d_index + 1],
                            target[node_indices, return_1d_index : return_1d_index + 1],
                            target_available[
                                node_indices,
                                return_1d_index : return_1d_index + 1,
                            ],
                            current_available[
                                node_indices,
                                return_1d_index : return_1d_index + 1,
                            ],
                        )
                        if cs_metrics is not None:
                            row.update(cs_metrics)
                        feature_rows.append(row)
    return rows, feature_rows


def summarize_rows(rows: list[dict[str, Any]], metric_names: list[str], group_key: str | None = None):
    if group_key is None:
        groups = {"all": rows}
    else:
        groups: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            groups.setdefault(str(row[group_key]), []).append(row)
    out = {}
    for group, items in groups.items():
        out[group] = {metric: aggregate([float(row.get(metric, float("nan"))) for row in items]) for metric in metric_names}
        out[group]["rows"] = len(items)
    return out


def pooled_future_skill(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Pool squared errors before taking the skill ratio.

    Averaging date-level ratios overweights dates whose persistence error is
    near zero, especially for sparse or slowly changing point-in-time
    fundamentals. The pooled statistic is the headline model-selection metric;
    date-level mean and median remain in the report as diagnostics.
    """

    model_sse = sum(float(row.get("model_sse", 0.0)) for row in rows if np.isfinite(row.get("model_sse", float("nan"))))
    persistence_sse = sum(
        float(row.get("persistence_sse", 0.0))
        for row in rows
        if np.isfinite(row.get("persistence_sse", float("nan")))
    )
    zero_baseline_sse = sum(
        float(row.get("zero_baseline_sse", 0.0))
        for row in rows
        if np.isfinite(row.get("zero_baseline_sse", float("nan")))
    )
    observed_cells = sum(int(row.get("observed_cells", 0)) for row in rows)
    return {
        "model_sse": float(model_sse),
        "persistence_sse": float(persistence_sse),
        "zero_baseline_sse": float(zero_baseline_sse),
        "observed_cells": int(observed_cells),
        "pooled_mse_skill_vs_persistence": (
            float(1.0 - model_sse / persistence_sse)
            if persistence_sse > 1e-12
            else float("nan")
        ),
        "pooled_mse_skill_vs_zero": (
            float(1.0 - model_sse / zero_baseline_sse)
            if zero_baseline_sse > 1e-12
            else float("nan")
        ),
    }


def pooled_rollout_dependency(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    full_sse = sum(
        float(row.get("model_sse", 0.0))
        for row in rows
        if np.isfinite(row.get("model_sse", float("nan")))
    )
    no_rollout_sse = sum(
        float(row.get("no_rollout_model_sse", 0.0))
        for row in rows
        if np.isfinite(row.get("no_rollout_model_sse", float("nan")))
    )
    return {
        "full_rollout_model_sse": float(full_sse),
        "no_rollout_model_sse": float(no_rollout_sse),
        "pooled_mse_skill_vs_no_rollout": (
            float(1.0 - full_sse / no_rollout_sse)
            if no_rollout_sse > 1e-12
            else float("nan")
        ),
    }


def summarize_future_rows(rows: list[dict[str, Any]], metric_names: list[str], group_key: str | None = None):
    summary = summarize_rows(rows, metric_names, group_key=group_key)
    if group_key is None:
        grouped = {"all": rows}
    else:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            grouped.setdefault(str(row[group_key]), []).append(row)
    for group, items in grouped.items():
        summary[group].update(pooled_future_skill(items))
    return summary


def forward_calibration_summary(
    rows: list[dict[str, Any]],
    feature_group: str = "feature:return_1d",
) -> dict[str, dict[str, float | int | str]]:
    """Fit a shrink-only scale on the first half and score the later half."""

    selected = [row for row in rows if row.get("feature_group") == feature_group]
    horizons = sorted({int(row["horizon"]) for row in selected})
    result: dict[str, dict[str, float | int | str]] = {}
    for horizon in horizons:
        horizon_rows = sorted(
            (row for row in selected if int(row["horizon"]) == horizon),
            key=lambda row: str(row["date"]),
        )
        split = len(horizon_rows) // 2
        if split < 1 or split >= len(horizon_rows):
            continue
        calibration_rows = horizon_rows[:split]
        evaluation_rows = horizon_rows[split:]
        calibration_prediction_sse = sum(float(row["prediction_sse"]) for row in calibration_rows)
        calibration_cross = sum(float(row["prediction_target_cross"]) for row in calibration_rows)
        scale = (
            calibration_cross / calibration_prediction_sse
            if calibration_prediction_sse > 1e-12
            else 0.0
        )
        scale = float(np.clip(scale, 0.0, 1.0))
        prediction_sse = sum(float(row["prediction_sse"]) for row in evaluation_rows)
        target_sse = sum(float(row["zero_baseline_sse"]) for row in evaluation_rows)
        cross = sum(float(row["prediction_target_cross"]) for row in evaluation_rows)
        model_sse = sum(float(row["model_sse"]) for row in evaluation_rows)
        calibrated_sse = scale * scale * prediction_sse - 2.0 * scale * cross + target_sse
        result[str(horizon)] = {
            "calibration_start": str(calibration_rows[0]["date"]),
            "calibration_end": str(calibration_rows[-1]["date"]),
            "evaluation_start": str(evaluation_rows[0]["date"]),
            "evaluation_end": str(evaluation_rows[-1]["date"]),
            "calibration_rows": len(calibration_rows),
            "evaluation_rows": len(evaluation_rows),
            "scale": scale,
            "uncalibrated_skill_vs_zero": (
                float(1.0 - model_sse / target_sse) if target_sse > 1e-12 else float("nan")
            ),
            "calibrated_skill_vs_zero": (
                float(1.0 - calibrated_sse / target_sse)
                if target_sse > 1e-12
                else float("nan")
            ),
        }
    return result


def correlation_significance_summary(
    rows: list[dict[str, Any]],
    feature_group: str = "feature:return_1d",
) -> dict[str, dict[str, float | int]]:
    """Report horizon-aware Newey-West significance for daily cross-sectional IC."""

    selected = [row for row in rows if row.get("feature_group") == feature_group]
    result: dict[str, dict[str, float | int]] = {}
    for horizon in sorted({int(row["horizon"]) for row in selected}):
        values = np.asarray(
            [
                float(row["target_corr"])
                for row in sorted(
                    (row for row in selected if int(row["horizon"]) == horizon),
                    key=lambda row: str(row["date"]),
                )
            ],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size < 3:
            continue
        centered = values - values.mean()
        long_variance = float(centered @ centered / values.size)
        max_lag = min(horizon, values.size - 1)
        for lag in range(1, max_lag + 1):
            weight = 1.0 - lag / (max_lag + 1.0)
            covariance = float(centered[lag:] @ centered[:-lag] / values.size)
            long_variance += 2.0 * weight * covariance
        standard_error = float(np.sqrt(max(long_variance, 0.0) / values.size))
        mean = float(values.mean())
        result[str(horizon)] = {
            "rows": int(values.size),
            "mean_target_corr": mean,
            "newey_west_lag": int(max_lag),
            "newey_west_standard_error": standard_error,
            "newey_west_t_stat": (
                float(mean / standard_error) if standard_error > 1e-12 else float("nan")
            ),
            "positive_day_fraction": float((values > 0.0).mean()),
        }
    return result


def daily_metric_correlation_significance(
    rows: list[dict[str, Any]],
    metric_name: str,
) -> dict[str, dict[str, float | int]]:
    """Report horizon-aware Newey-West significance for a daily metric."""

    result: dict[str, dict[str, float | int]] = {}
    for horizon in sorted({int(row["horizon"]) for row in rows}):
        values = np.asarray(
            [
                float(row[metric_name])
                for row in sorted(
                    (row for row in rows if int(row["horizon"]) == horizon),
                    key=lambda row: str(row["date"]),
                )
            ],
            dtype=np.float64,
        )
        values = values[np.isfinite(values)]
        if values.size < 3:
            continue
        centered = values - values.mean()
        long_variance = float(centered @ centered / values.size)
        max_lag = min(horizon, values.size - 1)
        for lag in range(1, max_lag + 1):
            weight = 1.0 - lag / (max_lag + 1.0)
            covariance = float(centered[lag:] @ centered[:-lag] / values.size)
            long_variance += 2.0 * weight * covariance
        standard_error = float(np.sqrt(max(long_variance, 0.0) / values.size))
        mean = float(values.mean())
        result[str(horizon)] = {
            "rows": int(values.size),
            "mean": mean,
            "mean_target_corr": mean,
            "newey_west_lag": int(max_lag),
            "newey_west_standard_error": standard_error,
            "newey_west_t_stat": (
                float(mean / standard_error) if standard_error > 1e-12 else float("nan")
            ),
            "positive_day_fraction": float((values > 0.0).mean()),
        }
    return result


def matched_path_return_correlation_significance(
    rows: list[dict[str, Any]],
    horizons: list[int],
) -> dict[str, dict[str, float | int]]:
    """Score each horizon's close-to-close return state.

    At target date ``t+h``, ``return_hd`` is the close-to-close return from
    context date ``t``. It is not the executable path, which enters at the next
    open and is reported separately.
    """

    result: dict[str, dict[str, float | int]] = {}
    for horizon in horizons:
        group = f"feature:return_{int(horizon)}d"
        horizon_result = correlation_significance_summary(
            rows,
            feature_group=group,
        )
        key = str(int(horizon))
        if key not in horizon_result:
            raise ValueError(
                f"future evaluation is missing horizon return state {group}"
            )
        result[key] = horizon_result[key]
    return result


def forward_cross_sectional_calibration_summary(
    rows: list[dict[str, Any]],
    feature_group: str = "feature:return_1d",
) -> dict[str, dict[str, float | int | str]]:
    """Fit alpha shrinkage on earlier demeaned predictions and score later dates."""

    selected = [
        row
        for row in rows
        if row.get("feature_group") == feature_group and "cs_prediction_sse" in row
    ]
    result: dict[str, dict[str, float | int | str]] = {}
    for horizon in sorted({int(row["horizon"]) for row in selected}):
        horizon_rows = sorted(
            (row for row in selected if int(row["horizon"]) == horizon),
            key=lambda row: str(row["date"]),
        )
        split = len(horizon_rows) // 2
        if split < 1 or split >= len(horizon_rows):
            continue
        calibration_rows = horizon_rows[:split]
        evaluation_rows = horizon_rows[split:]
        calibration_prediction_sse = sum(
            float(row["cs_prediction_sse"]) for row in calibration_rows
        )
        calibration_cross = sum(
            float(row["cs_prediction_target_cross"]) for row in calibration_rows
        )
        scale = (
            calibration_cross / calibration_prediction_sse
            if calibration_prediction_sse > 1e-12
            else 0.0
        )
        scale = float(np.clip(scale, 0.0, 1.0))
        prediction_sse = sum(float(row["cs_prediction_sse"]) for row in evaluation_rows)
        target_sse = sum(float(row["cs_target_sse"]) for row in evaluation_rows)
        cross = sum(float(row["cs_prediction_target_cross"]) for row in evaluation_rows)
        model_sse = sum(float(row["cs_model_sse"]) for row in evaluation_rows)
        calibrated_sse = scale * scale * prediction_sse - 2.0 * scale * cross + target_sse
        result[str(horizon)] = {
            "calibration_start": str(calibration_rows[0]["date"]),
            "calibration_end": str(calibration_rows[-1]["date"]),
            "evaluation_start": str(evaluation_rows[0]["date"]),
            "evaluation_end": str(evaluation_rows[-1]["date"]),
            "calibration_rows": len(calibration_rows),
            "evaluation_rows": len(evaluation_rows),
            "scale": scale,
            "uncalibrated_skill_vs_zero": (
                float(1.0 - model_sse / target_sse) if target_sse > 1e-12 else float("nan")
            ),
            "calibrated_skill_vs_zero": (
                float(1.0 - calibrated_sse / target_sse)
                if target_sse > 1e-12
                else float("nan")
            ),
        }
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate node-state prediction on historical backdata.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", default="reports/node_prediction_eval")
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument(
        "--state-target-scope",
        choices=["all", "checkpoint_temporal"],
        default="all",
        help=(
            "Score all observed state features or exactly the checkpoint's "
            "non-zero temporal training targets."
        ),
    )
    parser.add_argument(
        "--mask-strategy",
        choices=[
            "random_cell",
            "feature_group",
            "node_block",
            "mixed",
            "operational_mixed",
        ],
        default="mixed",
    )
    parser.add_argument("--hide-ratio", type=float, default=None)
    parser.add_argument("--max-steps", type=int, default=180)
    parser.add_argument(
        "--save-return-forecasts",
        action="store_true",
        help="Stream per-stock return-state forecasts for shadow-score calibration.",
    )
    parser.add_argument(
        "--ablate-policy-rate-state",
        action="store_true",
        help="Set policy-rate node inputs to their normalized training mean while retaining graph edges.",
    )
    parser.add_argument("--edge-cache-workers", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-end", default=None)
    parser.add_argument("--universe", default=None)
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--max-tickers", type=int, default=None)
    parser.add_argument("--override-universe", action="store_true")
    parser.add_argument("--allow-unverified-legacy", action="store_true")
    parser.add_argument("--allow-extrapolated-horizons", action="store_true")
    parser.add_argument("--edge-window", type=int, default=None)
    parser.add_argument("--edge-top-k", type=int, default=None)
    parser.add_argument("--min-abs-corr", type=float, default=None)
    parser.add_argument("--edge-correlation-mode", choices=["signed", "abs", "positive", "negative", "none"], default=None)
    parser.add_argument("--partial-corr-top-k", type=int, default=None)
    parser.add_argument("--partial-corr-min-abs", type=float, default=None)
    parser.add_argument("--partial-corr-mode", choices=["signed", "abs", "positive", "negative"], default=None)
    parser.add_argument("--partial-corr-scale", type=float, default=None)
    parser.add_argument("--lead-lag-top-k", type=int, default=None)
    parser.add_argument("--lead-lag-days", type=int, default=None)
    parser.add_argument("--lead-lag-min-abs-corr", type=float, default=None)
    parser.add_argument("--lead-lag-mode", choices=["signed", "abs", "positive", "negative"], default=None)
    parser.add_argument("--lead-lag-scale", type=float, default=None)
    parser.add_argument("--policy-rate-edge-scale", type=float, default=None)
    parser.add_argument("--min-train-rows", type=int, default=None)
    parser.add_argument("--event-path", action="append", default=[], help="Override/add JSONL event feature paths for event checkpoints")
    parser.add_argument("--event-half-life-days", type=float, default=None)
    parser.add_argument("--event-lag-days", type=int, default=None)
    parser.add_argument("--event-max-decay-days", type=int, default=None)
    parser.add_argument("--event-edge-top-k", type=int, default=None)
    parser.add_argument("--event-edge-min-weight", type=float, default=None)
    parser.add_argument("--event-edge-scale", type=float, default=None)
    parser.add_argument("--event-edge-max-themes", type=int, default=None)
    parser.add_argument("--event-edge-min-theme-count", type=int, default=None)
    parser.add_argument("--industry-profile-path", action="append", default=[])
    parser.add_argument("--industry-prefix-length", type=int, default=None)
    parser.add_argument("--industry-edge-scale", type=float, default=None)
    parser.add_argument("--fundamental-path", action="append", default=[])
    parser.add_argument("--fundamental-lag-days", type=int, default=None)
    parser.add_argument("--investor-cache-dir", default=None)
    parser.add_argument("--investor-flow-lag-days", type=int, default=None)
    parser.add_argument(
        "--external-preset",
        choices=["none", "kr_global", "kr_global_rates"],
        default=None,
    )
    parser.add_argument("--external-symbol", action="append", default=[])
    parser.add_argument("--external-lag-days", type=int, default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--external-etf-panel", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    model, ckpt = load_model(model_dir, device)
    ckpt_args = dict(ckpt.get("args", {}))
    validate_future_rollout_contract(
        ckpt_args,
        parse_int_list(args.horizons),
        args.allow_extrapolated_horizons,
    )
    features, ckpt_args = build_features_from_ckpt(ckpt, args)
    state_target_mask = state_target_feature_mask(
        features.feature_names,
        ckpt.get("temporal_state_feature_weights"),
        args.state_target_scope,
    )
    ablated_policy_nodes: list[str] = []
    if args.ablate_policy_rate_state:
        ablated_policy_nodes = ablate_policy_rate_state(features)
    steps = select_steps(features, ckpt_args, args)
    if len(steps) == 0:
        raise ValueError("no evaluation steps selected")

    out_dir = Path(args.output_dir) / model_dir.name
    out_dir.mkdir(parents=True, exist_ok=True)
    edge_cache = build_evaluation_edge_cache(features, steps, ckpt_args, args)
    current_rows = evaluate_current_imputation(model, features, steps, ckpt_args, args, device, edge_cache=edge_cache)
    return_forecast_path = out_dir / "return_1d_forecasts.csv"
    return_forecast_file = None
    return_forecast_writer = None
    if args.save_return_forecasts:
        return_forecast_file = return_forecast_path.open("w", encoding="utf-8", newline="")
        return_forecast_writer = csv.DictWriter(
            return_forecast_file,
            fieldnames=[
                "date",
                "target_date",
                "horizon",
                "ticker",
                "prediction_return_1d",
                "target_return_1d",
                "current_return_1d",
                "realized_path_return",
                "prediction_entry_path_return",
                "current_value_ma20_log",
                "prediction_return_5d",
                "prediction_cs_rank_return_20d",
                "prediction_volatility_20d",
                "prediction_downside_volatility_20d",
            ],
        )
        return_forecast_writer.writeheader()
    try:
        future_rows, future_feature_rows = evaluate_future_rollout(
            model,
            features,
            steps,
            ckpt_args,
            args,
            device,
            edge_cache=edge_cache,
            return_forecast_writer=return_forecast_writer,
        )
    finally:
        if return_forecast_file is not None:
            return_forecast_file.close()
    latent_health_rows = evaluate_latent_health(
        model,
        features,
        steps,
        ckpt_args,
        args,
        device,
        edge_cache=edge_cache,
    )

    write_csv(out_dir / "current_imputation.csv", current_rows)
    write_csv(out_dir / "future_rollout.csv", future_rows)
    write_csv(out_dir / "future_rollout_by_feature_group.csv", future_feature_rows)
    write_csv(out_dir / "latent_health.csv", latent_health_rows)

    rollout_dependency_significance = daily_metric_correlation_significance(
        future_rows,
        "rollout_mse_skill_vs_no_rollout",
    )
    summary = {
        "live_orders_allowed": False,
        "model_dir": str(model_dir),
        "features": len(features.feature_names),
        "tickers": len(features.tickers),
        "nodes": features.node_count,
        "stock_node_count": features.tradable_count,
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get("sha256"),
        "eval_steps": int(len(steps)),
        "eval_start": str(features.dates[int(steps[0])].date()),
        "eval_end": str(features.dates[int(steps[-1])].date()),
        "evaluation_seed": int(args.seed),
        "mask_strategy": args.mask_strategy,
        "horizons": parse_int_list(args.horizons),
        "state_target_scope": args.state_target_scope,
        "state_target_feature_count": int(state_target_mask.sum()),
        "state_target_features": [
            str(name)
            for name, selected in zip(features.feature_names, state_target_mask)
            if selected
        ],
        "ablated_policy_rate_nodes": ablated_policy_nodes,
        "return_1d_forecasts_file": (
            str(return_forecast_path) if args.save_return_forecasts else None
        ),
        "current_imputation": summarize_rows(
            current_rows,
            ["model_mae", "baseline_zero_mae", "model_rmse", "baseline_zero_rmse", "mse_skill_vs_zero", "r2_hidden"],
        ),
        "future_skill_metric": "pooled_mse_skill_vs_persistence",
        "future_rollout_by_horizon": summarize_future_rows(
            future_rows,
            [
                "model_mae",
                "persistence_mae",
                "zero_baseline_mae",
                "model_rmse",
                "persistence_rmse",
                "zero_baseline_rmse",
                "mse_skill_vs_persistence",
                "mse_skill_vs_zero",
                "rollout_mse_skill_vs_no_rollout",
                "state_r2",
                "target_corr",
                "target_cosine",
                "target_sign_accuracy_abs_ge_0_10",
                "delta_corr",
                "delta_cosine",
                "delta_sign_accuracy_abs_ge_0_10",
            ],
            group_key="horizon",
        ),
        "rollout_dependency_by_horizon": {
            str(horizon): {
                **pooled_rollout_dependency(
                    [
                        row
                        for row in future_rows
                        if int(row["horizon"]) == int(horizon)
                    ]
                ),
                "daily_significance": rollout_dependency_significance.get(
                    str(horizon),
                    {},
                ),
            }
            for horizon in parse_int_list(args.horizons)
        },
        "future_rollout_by_horizon_feature_group": {
            str(horizon): summarize_future_rows(
                [row for row in future_feature_rows if int(row["horizon"]) == int(horizon)],
                [
                    "observed_cells",
                    "model_mae",
                    "persistence_mae",
                    "zero_baseline_mae",
                    "model_rmse",
                    "persistence_rmse",
                    "zero_baseline_rmse",
                    "mse_skill_vs_persistence",
                    "mse_skill_vs_zero",
                    "state_r2",
                    "target_corr",
                    "target_cosine",
                    "target_sign_accuracy_abs_ge_0_10",
                    "delta_corr",
                    "delta_cosine",
                    "delta_sign_accuracy_abs_ge_0_10",
                ],
                group_key="feature_group",
            )
            for horizon in parse_int_list(args.horizons)
        },
        "return_1d_forward_calibration": forward_calibration_summary(future_feature_rows),
        "return_1d_correlation_significance": correlation_significance_summary(
            future_feature_rows
        ),
        "horizon_return_state_correlation_significance": (
            matched_path_return_correlation_significance(
                future_feature_rows,
                parse_int_list(args.horizons),
            )
        ),
        "realized_entry_path_correlation_significance": (
            daily_metric_correlation_significance(
                future_rows,
                "realized_entry_path_ic",
            )
        ),
        "realized_entry_path_liquidity_significance": {
            f"top{size}": daily_metric_correlation_significance(
                future_rows,
                f"realized_entry_path_ic_top{size}",
            )
            for size in (100, 300)
        },
        "return_1d_cross_sectional_calibration": (
            forward_cross_sectional_calibration_summary(future_feature_rows)
        ),
        "return_1d_liquidity_diagnostics": {
            group: {
                "forward_calibration": forward_calibration_summary(
                    future_feature_rows,
                    feature_group=group,
                ),
                "correlation_significance": correlation_significance_summary(
                    future_feature_rows,
                    feature_group=group,
                ),
                "cross_sectional_calibration": forward_cross_sectional_calibration_summary(
                    future_feature_rows,
                    feature_group=group,
                ),
            }
            for group in (
                "liquidity_top100:return_1d",
                "liquidity_top300:return_1d",
            )
        },
        "latent_health": summarize_rows(
            latent_health_rows,
            [
                "displacement_ratio",
                "latent_norm_mean",
                "cross_node_std_mean",
                "active_dimension_fraction",
                "variance_participation_ratio",
                "mean_pairwise_cosine",
            ],
            group_key="state",
        ),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
