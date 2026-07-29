from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import joblib
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import Ridge, SGDRegressor
import torch

from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.run_real_backtest import date_indices, parse_int_list, temporal_training_indices
from stock_v2.real_features import build_edge_tensor

CONTEXT_CACHE_PART_BYTES = 480 * 1024 * 1024


@dataclass(frozen=True)
class ContextLayout:
    mask_feature_indices: np.ndarray
    external_positions: np.ndarray
    base_feature_names: list[str]
    graph_feature_names: list[str]
    include_calendar: bool = False

    @property
    def base_feature_count(self) -> int:
        return len(self.base_feature_names)

    @property
    def total_feature_count(self) -> int:
        return len(self.base_feature_names) + len(self.graph_feature_names)


def parse_name_list(value: str) -> list[str]:
    return [item.strip() for item in str(value).split(",") if item.strip()]


def evaluator_namespace(args: argparse.Namespace) -> SimpleNamespace:
    """Provide the evaluator's exact feature-rebuild contract without model inference."""

    return SimpleNamespace(
        override_universe=False,
        universe_manifest=None,
        universe=None,
        max_tickers=None,
        start=None,
        end=None,
        train_end=None,
        edge_window=None,
        min_train_rows=None,
        cache_dir=args.cache_dir,
        refresh=False,
        horizons=args.horizons,
        event_path=[],
        event_half_life_days=None,
        event_lag_days=None,
        event_max_decay_days=None,
        event_edge_top_k=None,
        event_edge_min_weight=None,
        event_edge_scale=None,
        event_edge_max_themes=None,
        event_edge_min_theme_count=None,
        fundamental_path=[],
        fundamental_lag_days=None,
        investor_cache_dir=None,
        investor_flow_lag_days=None,
        external_symbol=[],
        external_preset=None,
        external_lag_days=None,
        external_cache_dir=args.external_cache_dir,
        # build_features_from_ckpt reads these unconditionally. They were added
        # to the evaluator for the US-ETF node work and never mirrored here,
        # which silently broke every caller of this namespace -- including the
        # daily prospective chain, whose cache build imports it. None means
        # "fall back to whatever the checkpoint recorded", which is the same
        # behaviour every other field here uses.
        external_etf_panel=None,
        external_etf_symbols=None,
        industry_profile_path=[],
        industry_prefix_length=None,
        industry_edge_scale=None,
        allow_unverified_legacy=False,
        edge_top_k=None,
        min_abs_corr=None,
        edge_correlation_mode=None,
        partial_corr_top_k=None,
        partial_corr_min_abs=None,
        partial_corr_mode=None,
        partial_corr_scale=None,
        lead_lag_top_k=None,
        lead_lag_days=None,
        lead_lag_min_abs_corr=None,
        lead_lag_mode=None,
        lead_lag_scale=None,
        policy_rate_edge_scale=None,
    )


def build_context_layout(
    features,
    train_steps: np.ndarray,
    include_calendar: bool = False,
) -> ContextLayout:
    stock_count = features.tradable_count
    feature_names = list(features.feature_names)
    stock_availability = features.available_mask[train_steps, :stock_count]
    availability_fraction = stock_availability.mean(axis=(0, 1))
    mask_indices = np.flatnonzero(
        (availability_fraction > 0.0) & (availability_fraction < 0.9999)
    ).astype(np.int64)

    if features.node_count > stock_count:
        external_available = features.available_mask[
            train_steps, stock_count:, :
        ].any(axis=0)
        external_positions = np.argwhere(external_available).astype(np.int64)
    else:
        external_positions = np.empty((0, 2), dtype=np.int64)

    base_names = [f"own:{name}" for name in feature_names]
    base_names.extend(f"available:{feature_names[index]}" for index in mask_indices)
    base_names.extend(f"market_mean:{name}" for name in feature_names)
    base_names.extend(f"market_std:{name}" for name in feature_names)
    node_ids = list(features.node_tickers or [])
    for relative_node, feature_index in external_positions:
        node_index = stock_count + int(relative_node)
        node_id = node_ids[node_index] if node_index < len(node_ids) else f"EXT:{relative_node}"
        base_names.append(f"external:{node_id}:{feature_names[int(feature_index)]}")
    if include_calendar:
        base_names.extend(
            ["calendar:year", "calendar:month_sin", "calendar:month_cos", "calendar:dow_sin", "calendar:dow_cos"]
        )
    graph_names = [f"neighbor:{name}" for name in feature_names]
    return ContextLayout(
        mask_feature_indices=mask_indices,
        external_positions=external_positions,
        base_feature_names=base_names,
        graph_feature_names=graph_names,
        include_calendar=bool(include_calendar),
    )


def _edge_settings(ckpt_args: dict[str, Any]) -> dict[str, Any]:
    return {
        "edge_window": int(ckpt_args.get("edge_window", 60)),
        "top_k": int(ckpt_args.get("edge_top_k", 6)),
        "min_abs_corr": float(ckpt_args.get("min_abs_corr", 0.20)),
        "correlation_mode": str(ckpt_args.get("edge_correlation_mode", "signed")),
        "event_top_k": int(ckpt_args.get("event_edge_top_k", 0) or 0),
        "event_min_weight": float(ckpt_args.get("event_edge_min_weight", 0.05)),
        "event_scale": float(ckpt_args.get("event_edge_scale", 0.25)),
        "partial_corr_top_k": int(ckpt_args.get("partial_corr_top_k", 0) or 0),
        "partial_corr_min_abs": float(ckpt_args.get("partial_corr_min_abs", 0.10)),
        "partial_corr_mode": str(ckpt_args.get("partial_corr_mode", "signed")),
        "partial_corr_scale": float(ckpt_args.get("partial_corr_scale", 0.50)),
        "lead_lag_top_k": int(ckpt_args.get("lead_lag_top_k", 0) or 0),
        "lead_lag_days": int(ckpt_args.get("lead_lag_days", 1) or 1),
        "lead_lag_min_abs_corr": float(ckpt_args.get("lead_lag_min_abs_corr", 0.08)),
        "lead_lag_mode": str(ckpt_args.get("lead_lag_mode", "signed")),
        "lead_lag_scale": float(ckpt_args.get("lead_lag_scale", 0.50)),
        "policy_rate_edge_scale": float(ckpt_args.get("policy_rate_edge_scale", 0.0)),
    }


def graph_neighbor_state(features, step: int, ckpt_args: dict[str, Any]) -> np.ndarray:
    edge_index, edge_weight = build_edge_tensor(
        features,
        step=int(step),
        **_edge_settings(ckpt_args),
    )
    if edge_index.numel() == 0:
        return np.zeros(
            (features.tradable_count, len(features.feature_names)), dtype=np.float32
        )
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    weight = edge_weight.numpy().astype(np.float32, copy=False)
    adjacency = sparse.csr_matrix(
        (weight, (dst, src)),
        shape=(features.node_count, features.node_count),
        dtype=np.float32,
    )
    absolute_adjacency = sparse.csr_matrix(
        (np.abs(weight), (dst, src)),
        shape=(features.node_count, features.node_count),
        dtype=np.float32,
    )
    node_state = features.features[int(step)].astype(np.float32, copy=False)
    aggregate = np.asarray(adjacency @ node_state, dtype=np.float32)
    degree = np.asarray(
        absolute_adjacency @ np.ones(features.node_count, dtype=np.float32)
    ).reshape(-1, 1)
    aggregate /= np.maximum(degree, 1.0)
    return aggregate[: features.tradable_count]


def _market_moments(state: np.ndarray, available: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    count = available.sum(axis=0).clip(min=1.0)
    mean = (state * available).sum(axis=0) / count
    centered = (state - mean) * available
    variance = (centered * centered).sum(axis=0) / count
    return mean.astype(np.float32), np.sqrt(variance).astype(np.float32)


def context_rows_for_step(
    features,
    step: int,
    layout: ContextLayout,
    neighbor_state: np.ndarray,
) -> np.ndarray:
    stock_count = features.tradable_count
    state = features.features[int(step), :stock_count].astype(np.float32, copy=False)
    available = features.available_mask[int(step), :stock_count].astype(np.float32, copy=False)
    market_mean, market_std = _market_moments(state, available)

    parts = [state]
    if layout.mask_feature_indices.size:
        parts.append(available[:, layout.mask_feature_indices])
    parts.append(np.broadcast_to(market_mean, state.shape))
    parts.append(np.broadcast_to(market_std, state.shape))

    if layout.external_positions.size:
        external = features.features[int(step), stock_count:]
        external_values = external[
            layout.external_positions[:, 0], layout.external_positions[:, 1]
        ].astype(np.float32, copy=False)
        parts.append(np.broadcast_to(external_values, (stock_count, len(external_values))))

    if layout.include_calendar:
        date = pd.Timestamp(features.dates[int(step)])
        year = np.float32((date.year - 2020) / 10.0)
        month_angle = 2.0 * np.pi * (date.month - 1) / 12.0
        dow_angle = 2.0 * np.pi * date.dayofweek / 5.0
        calendar = np.asarray(
            [year, np.sin(month_angle), np.cos(month_angle), np.sin(dow_angle), np.cos(dow_angle)],
            dtype=np.float32,
        )
        parts.append(np.broadcast_to(calendar, (stock_count, len(calendar))))
    parts.append(neighbor_state.astype(np.float32, copy=False))
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def build_context_matrix(
    features,
    steps: np.ndarray,
    layout: ContextLayout,
    ckpt_args: dict[str, Any],
    workers: int,
) -> np.ndarray:
    stock_count = features.tradable_count
    matrix = np.empty(
        (len(steps) * stock_count, layout.total_feature_count), dtype=np.float32
    )

    def build_neighbor(raw_step: int) -> np.ndarray:
        return graph_neighbor_state(features, int(raw_step), ckpt_args)

    if workers > 1:
        executor = ThreadPoolExecutor(max_workers=workers)
        neighbor_iter = executor.map(build_neighbor, [int(step) for step in steps])
    else:
        executor = None
        neighbor_iter = map(build_neighbor, [int(step) for step in steps])
    try:
        for position, (step, neighbor) in enumerate(zip(steps, neighbor_iter)):
            start = position * stock_count
            matrix[start : start + stock_count] = context_rows_for_step(
                features, int(step), layout, neighbor
            )
            if (position + 1) % 100 == 0 or position + 1 == len(steps):
                print(f"context features: {position + 1}/{len(steps)} dates", flush=True)
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    return matrix


def context_cache_contract(
    ckpt: dict[str, Any],
    steps: np.ndarray,
    layout: ContextLayout,
    stock_count: int,
) -> dict[str, Any]:
    feature_payload = json.dumps(
        {
            "base": layout.base_feature_names,
            "graph": layout.graph_feature_names,
        },
        ensure_ascii=False,
        sort_keys=True,
    ).encode("utf-8")
    return {
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get("sha256"),
        "train_edge_manifest_sha256": ckpt.get("train_edge_manifest", {}).get("sha256"),
        "steps": [int(step) for step in steps],
        "stock_count": int(stock_count),
        "feature_count": int(layout.total_feature_count),
        "feature_contract_sha256": hashlib.sha256(feature_payload).hexdigest(),
    }


def save_context_matrix_cache(
    cache_path: Path,
    matrix: np.ndarray,
    contract: dict[str, Any],
    *,
    max_part_bytes: int = CONTEXT_CACHE_PART_BYTES,
) -> None:
    if matrix.ndim != 2 or matrix.dtype != np.float32:
        raise ValueError("context cache matrix must be a 2D float32 array")
    if max_part_bytes <= 0:
        raise ValueError("max_part_bytes must be positive")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    parts_dir = cache_path.with_suffix(cache_path.suffix + ".parts")
    payload: dict[str, Any] = {
        "contract": contract,
        "shape": [int(value) for value in matrix.shape],
        "dtype": str(matrix.dtype),
    }
    if matrix.nbytes <= int(max_part_bytes):
        np.save(cache_path, matrix)
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        payload.update({"format": "single_npy", "parts": []})
    else:
        if cache_path.exists():
            cache_path.unlink()
        if parts_dir.exists():
            shutil.rmtree(parts_dir)
        parts_dir.mkdir(parents=True)
        bytes_per_row = int(matrix.shape[1]) * int(matrix.dtype.itemsize)
        rows_per_part = max(1, int(max_part_bytes) // max(bytes_per_row, 1))
        parts = []
        for index, start in enumerate(range(0, len(matrix), rows_per_part)):
            end = min(start + rows_per_part, len(matrix))
            part_name = f"part_{index:04d}.npy"
            np.save(parts_dir / part_name, matrix[start:end])
            parts.append({"file": part_name, "start": int(start), "end": int(end)})
        payload.update({"format": "chunked_npy", "parts": parts})
    temporary_metadata = metadata_path.with_suffix(metadata_path.suffix + ".tmp")
    temporary_metadata.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary_metadata.replace(metadata_path)


def load_context_matrix_cache(
    cache_path: Path,
    contract: dict[str, Any],
) -> np.ndarray | None:
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not metadata_path.exists():
        return None
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata == contract:
        return np.load(cache_path, mmap_mode="r") if cache_path.exists() else None
    if metadata.get("contract") != contract:
        return None
    expected_shape = tuple(int(value) for value in metadata.get("shape", ()))
    if len(expected_shape) != 2 or metadata.get("dtype") != "float32":
        raise ValueError("context feature cache metadata has an invalid shape or dtype")
    cache_format = metadata.get("format")
    if cache_format == "single_npy":
        if not cache_path.exists():
            raise ValueError("single-file context feature cache is missing")
        matrix = np.load(cache_path, mmap_mode="r")
    elif cache_format == "chunked_npy":
        parts_dir = cache_path.with_suffix(cache_path.suffix + ".parts")
        matrix = np.empty(expected_shape, dtype=np.float32)
        cursor = 0
        for part in metadata.get("parts", []):
            start = int(part["start"])
            end = int(part["end"])
            if start != cursor or end <= start or end > expected_shape[0]:
                raise ValueError("context feature cache parts are not contiguous")
            part_path = parts_dir / str(part["file"])
            if not part_path.exists():
                raise ValueError(f"context feature cache part is missing: {part_path}")
            values = np.load(part_path, mmap_mode="r")
            if values.shape != (end - start, expected_shape[1]) or values.dtype != np.float32:
                raise ValueError("context feature cache part shape or dtype does not match")
            matrix[start:end] = values
            cursor = end
        if cursor != expected_shape[0]:
            raise ValueError("context feature cache parts do not cover every row")
    else:
        raise ValueError(f"unknown context feature cache format: {cache_format}")
    if matrix.shape != expected_shape or matrix.dtype != np.float32:
        raise ValueError("context feature cache shape or dtype does not match its contract")
    return matrix


def load_or_build_context_matrix(
    features,
    steps: np.ndarray,
    layout: ContextLayout,
    ckpt: dict[str, Any],
    ckpt_args: dict[str, Any],
    workers: int,
    cache_path: Path | None,
) -> np.ndarray:
    contract = context_cache_contract(ckpt, steps, layout, features.tradable_count)
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json") if cache_path else None
    if cache_path and metadata_path and metadata_path.exists():
        matrix = load_context_matrix_cache(cache_path, contract)
        if matrix is not None:
            expected_shape = (
                len(steps) * features.tradable_count,
                layout.total_feature_count,
            )
            if matrix.shape != expected_shape or matrix.dtype != np.float32:
                raise ValueError("context feature cache shape or dtype does not match its contract")
            print(f"loaded context feature cache: {cache_path}", flush=True)
            return matrix
        print(f"ignoring stale context feature cache: {cache_path}", flush=True)

    matrix = build_context_matrix(features, steps, layout, ckpt_args, workers=workers)
    if cache_path and metadata_path:
        save_context_matrix_cache(cache_path, matrix, contract)
        print(f"saved context feature cache: {cache_path}", flush=True)
    return matrix


def rows_for_steps(
    selected_steps: Sequence[int],
    step_positions: dict[int, int],
    stock_count: int,
) -> np.ndarray:
    rows = [
        np.arange(step_positions[int(step)] * stock_count, (step_positions[int(step)] + 1) * stock_count)
        for step in selected_steps
    ]
    return np.concatenate(rows).astype(np.int64) if rows else np.empty(0, dtype=np.int64)


def cross_sectional_zscore(values: np.ndarray, valid: np.ndarray, block_size: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float32)
    for start in range(0, len(values), block_size):
        end = start + block_size
        block_valid = valid[start:end] & np.isfinite(values[start:end])
        if block_valid.sum() < 3:
            continue
        block = values[start:end][block_valid].astype(np.float64)
        std = float(block.std())
        if std < 1e-12:
            continue
        block_result = result[start:end]
        block_result[block_valid] = ((block - block.mean()) / std).astype(np.float32)
    return result


def grouped_rank_labels(values: np.ndarray, valid: np.ndarray, block_size: int, bins: int = 10) -> np.ndarray:
    result = np.full(values.shape, -1, dtype=np.int32)
    for start in range(0, len(values), block_size):
        end = start + block_size
        local_valid = valid[start:end] & np.isfinite(values[start:end])
        indices = np.flatnonzero(local_valid)
        if len(indices) < bins:
            continue
        order = np.argsort(values[start:end][indices], kind="stable")
        labels = np.floor(np.arange(len(indices)) * bins / len(indices)).astype(np.int32)
        labels = np.minimum(labels, bins - 1)
        local = result[start:end]
        local[indices[order]] = labels
    return result


def group_sizes(row_indices: np.ndarray, row_step_ids: np.ndarray) -> list[int]:
    if len(row_indices) == 0:
        return []
    steps = row_step_ids[row_indices]
    if np.any(steps[1:] < steps[:-1]):
        raise ValueError("ranking rows must be sorted by date group")
    return np.unique(steps, return_counts=True)[1].astype(int).tolist()


def newey_west_mean(values: Sequence[float], lag: int) -> dict[str, float | int]:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3:
        return {"rows": int(len(arr)), "mean": float("nan"), "newey_west_t": float("nan")}
    centered = arr - arr.mean()
    long_variance = float(centered @ centered / len(arr))
    max_lag = min(max(0, int(lag)), len(arr) - 1)
    for offset in range(1, max_lag + 1):
        weight = 1.0 - offset / (max_lag + 1.0)
        covariance = float(centered[offset:] @ centered[:-offset] / len(arr))
        long_variance += 2.0 * weight * covariance
    standard_error = float(np.sqrt(max(long_variance, 0.0) / len(arr)))
    mean = float(arr.mean())
    return {
        "rows": int(len(arr)),
        "mean": mean,
        "newey_west_lag": int(max_lag),
        "newey_west_standard_error": standard_error,
        "newey_west_t": float(mean / standard_error) if standard_error > 1e-12 else float("nan"),
        "positive_fraction": float((arr > 0.0).mean()),
    }


def _correlation(a: np.ndarray, b: np.ndarray) -> float:
    valid = np.isfinite(a) & np.isfinite(b)
    if valid.sum() < 3:
        return float("nan")
    x = a[valid].astype(np.float64)
    y = b[valid].astype(np.float64)
    if x.std() < 1e-12 or y.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _decile_spread(score: np.ndarray, realized: np.ndarray) -> float:
    valid = np.isfinite(score) & np.isfinite(realized)
    if valid.sum() < 20:
        return float("nan")
    s = score[valid]
    r = realized[valid]
    count = max(1, len(s) // 10)
    order = np.argsort(s, kind="stable")
    return float(r[order[-count:]].mean() - r[order[:count]].mean())


def fit_affine(prediction: np.ndarray, target: np.ndarray, valid: np.ndarray) -> tuple[float, float]:
    mask = valid & np.isfinite(prediction) & np.isfinite(target)
    if mask.sum() < 3:
        return 0.0, 0.0
    x = prediction[mask].astype(np.float64)
    y = target[mask].astype(np.float64)
    variance = float(np.sum((x - x.mean()) ** 2))
    slope = float(np.sum((x - x.mean()) * (y - y.mean())) / variance) if variance > 1e-12 else 0.0
    intercept = float(y.mean() - slope * x.mean())
    return intercept, slope


def build_targets(features, steps: np.ndarray, horizon: int) -> dict[str, np.ndarray]:
    stock_count = features.tradable_count
    return_index = features.feature_names.index("return_1d")
    target_steps = steps + int(horizon)
    current = features.features[steps, :stock_count, return_index].reshape(-1).astype(np.float32)
    state = features.features[target_steps, :stock_count, return_index].reshape(-1).astype(np.float32)
    current_available = features.available_mask[steps, :stock_count, return_index].reshape(-1) > 0.5
    target_available = features.available_mask[target_steps, :stock_count, return_index].reshape(-1) > 0.5
    path = features.target_return_paths[int(horizon)][steps, :stock_count].reshape(-1).astype(np.float32)
    valid_state = current_available & target_available & np.isfinite(current) & np.isfinite(state)
    valid_path = current_available & np.isfinite(path)
    return_scale = float(features.train_std[return_index])
    return_mean = float(features.train_mean[return_index])
    return {
        "current_state": current,
        "target_state": state,
        "target_return_raw": state * return_scale + return_mean,
        "target_path": path,
        "valid_state": valid_state,
        "valid_path": valid_path,
    }


def _lightgbm_parameters(args: argparse.Namespace, ranker: bool) -> dict[str, Any]:
    common = {
        "n_estimators": int(args.rounds),
        "learning_rate": float(args.learning_rate),
        "num_leaves": int(args.num_leaves),
        "max_depth": -1,
        "min_child_samples": int(args.min_child_samples),
        "subsample": 0.80,
        "subsample_freq": 1,
        "colsample_bytree": 0.75,
        "reg_alpha": 1.0,
        "reg_lambda": 5.0,
        "max_bin": 127,
        "random_state": int(args.seed),
        "n_jobs": int(args.threads),
        "verbosity": -1,
    }
    if ranker:
        common.update({"objective": "lambdarank", "metric": "ndcg", "label_gain": list(range(10))})
    else:
        common.update({"objective": "regression_l2", "metric": "l2"})
    return common


def fit_baseline(
    model_name: str,
    X: np.ndarray,
    fit_rows: np.ndarray,
    validation_rows: np.ndarray,
    regression_target: np.ndarray,
    rank_target: np.ndarray,
    row_step_ids: np.ndarray,
    args: argparse.Namespace,
) -> tuple[Any, dict[str, Any]]:
    use_graph = "graph" in model_name
    feature_count = X.shape[1] if use_graph else int(args.base_feature_count)
    X_fit = X[fit_rows, :feature_count]
    X_validation = X[validation_rows, :feature_count]
    metadata: dict[str, Any] = {"feature_count": int(feature_count)}

    if model_name == "ridge":
        model = Ridge(alpha=float(args.ridge_alpha), solver="lsqr", tol=1e-4)
        model.fit(X_fit, regression_target[fit_rows])
    elif model_name == "elastic_net":
        model = SGDRegressor(
            loss="squared_error",
            penalty="elasticnet",
            alpha=float(args.elastic_alpha),
            l1_ratio=float(args.elastic_l1_ratio),
            max_iter=40,
            tol=1e-4,
            shuffle=True,
            random_state=int(args.seed),
            average=True,
        )
        model.fit(X_fit, regression_target[fit_rows])
    elif model_name.startswith("lightgbm"):
        import lightgbm as lgb

        is_ranker = "rank" in model_name
        parameters = _lightgbm_parameters(args, ranker=is_ranker)
        model = lgb.LGBMRanker(**parameters) if is_ranker else lgb.LGBMRegressor(**parameters)
        callbacks = [
            lgb.early_stopping(int(args.early_stopping_rounds), verbose=False),
            lgb.log_evaluation(100),
        ]
        if is_ranker:
            model.fit(
                X_fit,
                rank_target[fit_rows],
                group=group_sizes(fit_rows, row_step_ids),
                eval_set=[(X_validation, rank_target[validation_rows])],
                eval_group=[group_sizes(validation_rows, row_step_ids)],
                eval_at=[10],
                callbacks=callbacks,
            )
        else:
            model.fit(
                X_fit,
                regression_target[fit_rows],
                eval_set=[(X_validation, regression_target[validation_rows])],
                callbacks=callbacks,
            )
        metadata["best_iteration"] = int(model.best_iteration_ or args.rounds)
    elif model_name in {"catboost", "catboost_graph"}:
        from catboost import CatBoostRegressor

        model = CatBoostRegressor(
            iterations=int(args.rounds),
            depth=int(args.catboost_depth),
            learning_rate=float(args.learning_rate),
            loss_function="RMSE",
            l2_leaf_reg=5.0,
            random_seed=int(args.seed),
            task_type=str(args.catboost_task_type).upper(),
            devices="0" if str(args.catboost_task_type).lower() == "gpu" else None,
            od_type="Iter",
            od_wait=int(args.early_stopping_rounds),
            verbose=100,
            allow_writing_files=False,
        )
        model.fit(
            X_fit,
            regression_target[fit_rows],
            eval_set=(X_validation, regression_target[validation_rows]),
            use_best_model=True,
        )
        metadata["best_iteration"] = int(model.get_best_iteration())
    else:
        raise ValueError(f"unknown baseline model: {model_name}")
    return model, metadata


def predict_model(model: Any, X: np.ndarray, rows: np.ndarray, feature_count: int) -> np.ndarray:
    return np.asarray(model.predict(X[rows, :feature_count]), dtype=np.float32)


def evaluate_predictions(
    model_name: str,
    target_kind: str,
    horizon: int,
    test_steps: np.ndarray,
    prediction: np.ndarray,
    calibrated_prediction: np.ndarray,
    targets: dict[str, np.ndarray],
    liquidity: np.ndarray,
    stock_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    daily_rows: list[dict[str, Any]] = []
    state_model_sse = 0.0
    state_persistence_sse = 0.0
    state_zero_sse = 0.0
    path_model_sse = 0.0
    path_zero_sse = 0.0
    for position, step in enumerate(test_steps):
        start = position * stock_count
        end = start + stock_count
        score = prediction[start:end]
        calibrated = calibrated_prediction[start:end]
        state = targets["target_state"][start:end]
        current = targets["current_state"][start:end]
        raw_return = targets["target_return_raw"][start:end]
        path = targets["target_path"][start:end]
        valid_state = targets["valid_state"][start:end] & np.isfinite(score)
        valid_path = targets["valid_path"][start:end] & np.isfinite(score)
        liquid_indices = np.flatnonzero(np.isfinite(liquidity[start:end]))
        if len(liquid_indices) > 300:
            order = np.argsort(liquidity[start:end][liquid_indices], kind="stable")
            liquid_indices = liquid_indices[order[-300:]]
        liquid_state = np.zeros(stock_count, dtype=bool)
        liquid_state[liquid_indices] = True

        row = {
            "model": model_name,
            "target_kind": target_kind,
            "horizon": int(horizon),
            "date": str(pd.Timestamp(step).date()) if not isinstance(step, (int, np.integer)) else int(step),
            "state_ic": _correlation(score[valid_state], state[valid_state]),
            "state_ic_top300": _correlation(score[valid_state & liquid_state], state[valid_state & liquid_state]),
            "path_ic": _correlation(score[valid_path], path[valid_path]),
            "path_ic_top300": _correlation(score[valid_path & liquid_state], path[valid_path & liquid_state]),
            "next_return_decile_spread": _decile_spread(score[valid_state], raw_return[valid_state]),
            "path_decile_spread": _decile_spread(score[valid_path], path[valid_path]),
        }
        daily_rows.append(row)
        if target_kind == "state" and valid_state.any():
            state_model_sse += float(np.sum((calibrated[valid_state] - state[valid_state]) ** 2))
            state_persistence_sse += float(np.sum((current[valid_state] - state[valid_state]) ** 2))
            state_zero_sse += float(np.sum(state[valid_state] ** 2))
        if target_kind == "path" and valid_path.any():
            path_model_sse += float(np.sum((calibrated[valid_path] - path[valid_path]) ** 2))
            path_zero_sse += float(np.sum(path[valid_path] ** 2))

    metrics = {}
    for name in (
        "state_ic",
        "state_ic_top300",
        "path_ic",
        "path_ic_top300",
        "next_return_decile_spread",
        "path_decile_spread",
    ):
        metrics[name] = newey_west_mean([float(row[name]) for row in daily_rows], lag=horizon)
    metrics.update(
        {
            "state_skill_vs_persistence": (
                float(1.0 - state_model_sse / state_persistence_sse)
                if state_persistence_sse > 1e-12
                else float("nan")
            ),
            "state_skill_vs_zero": (
                float(1.0 - state_model_sse / state_zero_sse)
                if state_zero_sse > 1e-12
                else float("nan")
            ),
            "path_skill_vs_zero": (
                float(1.0 - path_model_sse / path_zero_sse)
                if path_zero_sse > 1e-12
                else float("nan")
            ),
        }
    )
    return metrics, daily_rows


def save_model(model: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(model, "booster_"):
        model.booster_.save_model(str(path.with_suffix(".txt")))
    elif model.__class__.__module__.startswith("catboost"):
        model.save_model(str(path.with_suffix(".cbm")))
    else:
        joblib.dump(model, path.with_suffix(".joblib"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark direct return baselines on a Graph-JEPA data contract.")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--models", default="ridge,elastic_net,lightgbm,lightgbm_rank,lightgbm_graph_rank")
    parser.add_argument("--target-kinds", default="state")
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--feature-workers", type=int, default=24)
    parser.add_argument("--threads", type=int, default=24)
    parser.add_argument("--rounds", type=int, default=800)
    parser.add_argument("--early-stopping-rounds", type=int, default=60)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--num-leaves", type=int, default=31)
    parser.add_argument("--min-child-samples", type=int, default=400)
    parser.add_argument("--ridge-alpha", type=float, default=10.0)
    parser.add_argument("--elastic-alpha", type=float, default=1e-5)
    parser.add_argument("--elastic-l1-ratio", type=float, default=0.05)
    parser.add_argument("--catboost-depth", type=int, default=8)
    parser.add_argument("--catboost-task-type", choices=["cpu", "gpu"], default="gpu")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--context-cache", default=None)
    parser.add_argument("--include-calendar", action="store_true")
    args = parser.parse_args()

    model_names = parse_name_list(args.models)
    target_kinds = parse_name_list(args.target_kinds)
    unknown_targets = sorted(set(target_kinds) - {"state", "path"})
    if unknown_targets:
        raise ValueError(f"unknown target kinds: {unknown_targets}")
    horizons = parse_int_list(args.horizons)

    model_dir = Path(args.model_dir)
    ckpt = torch.load(model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False)
    features, ckpt_args = build_features_from_ckpt(ckpt, evaluator_namespace(args))
    train_end = str(ckpt_args.get("train_end", "2023-12-29"))
    edge_window = int(ckpt_args.get("edge_window", 60))
    max_horizon = max(horizons)
    train_indices = date_indices(features.dates, end=train_end)
    train_steps = temporal_training_indices(
        train_indices,
        edge_window=edge_window,
        max_rollout_offset=max_horizon,
        total_steps=len(features.dates),
    )
    test_steps = date_indices(
        features.dates,
        start=(pd.Timestamp(train_end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    test_steps = test_steps[
        (test_steps >= edge_window) & (test_steps <= len(features.dates) - 1 - max_horizon)
    ]
    if args.max_test_steps > 0 and len(test_steps) > args.max_test_steps:
        positions = np.linspace(0, len(test_steps) - 1, args.max_test_steps).round().astype(int)
        test_steps = test_steps[positions]
    if len(train_steps) <= args.validation_days + max_horizon:
        raise ValueError("not enough training dates for validation split")
    validation_steps = train_steps[-int(args.validation_days) :]
    fit_steps = train_steps[train_steps < int(validation_steps[0]) - max_horizon]
    if len(fit_steps) < 260:
        raise ValueError("fit split is too short")

    all_steps = np.unique(np.concatenate([fit_steps, validation_steps, test_steps])).astype(np.int64)
    step_positions = {int(step): position for position, step in enumerate(all_steps)}
    stock_count = features.tradable_count
    layout = build_context_layout(
        features,
        fit_steps,
        include_calendar=bool(args.include_calendar),
    )
    print(
        f"baseline panel: dates={len(features.dates)} stocks={stock_count} features={len(features.feature_names)} "
        f"fit={len(fit_steps)} validation={len(validation_steps)} test={len(test_steps)} "
        f"base_columns={layout.base_feature_count} graph_columns={len(layout.graph_feature_names)}",
        flush=True,
    )
    X = load_or_build_context_matrix(
        features,
        all_steps,
        layout,
        ckpt,
        ckpt_args,
        workers=max(1, int(args.feature_workers)),
        cache_path=Path(args.context_cache) if args.context_cache else None,
    )
    args.base_feature_count = layout.base_feature_count
    row_step_ids = np.repeat(np.arange(len(all_steps), dtype=np.int32), stock_count)
    fit_matrix_rows = rows_for_steps(fit_steps, step_positions, stock_count)
    validation_matrix_rows = rows_for_steps(validation_steps, step_positions, stock_count)
    test_matrix_rows = rows_for_steps(test_steps, step_positions, stock_count)
    liquidity_index = features.feature_names.index("value_ma20_log")
    liquidity = features.raw_features[test_steps, :stock_count, liquidity_index].reshape(-1)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {
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
        "stocks": int(stock_count),
        "base_feature_count": int(layout.base_feature_count),
        "graph_feature_count": int(len(layout.graph_feature_names)),
        "models": {},
    }
    daily_rows: list[dict[str, Any]] = []

    for horizon in horizons:
        full_targets = build_targets(features, all_steps, horizon)
        state_cs = cross_sectional_zscore(
            full_targets["target_state"], full_targets["valid_state"], stock_count
        )
        path_cs = cross_sectional_zscore(
            full_targets["target_path"], full_targets["valid_path"], stock_count
        )
        state_rank = grouped_rank_labels(
            full_targets["target_state"], full_targets["valid_state"], stock_count
        )
        path_rank = grouped_rank_labels(
            full_targets["target_path"], full_targets["valid_path"], stock_count
        )
        test_targets = build_targets(features, test_steps, horizon)

        for target_kind in target_kinds:
            regression_target = state_cs if target_kind == "state" else path_cs
            rank_target = state_rank if target_kind == "state" else path_rank
            target_valid = full_targets["valid_state"] if target_kind == "state" else full_targets["valid_path"]
            fit_rows = fit_matrix_rows[
                target_valid[fit_matrix_rows]
                & np.isfinite(regression_target[fit_matrix_rows])
                & (rank_target[fit_matrix_rows] >= 0)
            ]
            validation_rows = validation_matrix_rows[
                target_valid[validation_matrix_rows]
                & np.isfinite(regression_target[validation_matrix_rows])
                & (rank_target[validation_matrix_rows] >= 0)
            ]
            for model_name in model_names:
                print(
                    f"training model={model_name} target={target_kind} horizon={horizon} "
                    f"fit_rows={len(fit_rows)} validation_rows={len(validation_rows)}",
                    flush=True,
                )
                model, fit_metadata = fit_baseline(
                    model_name,
                    X,
                    fit_rows,
                    validation_rows,
                    regression_target,
                    rank_target,
                    row_step_ids,
                    args,
                )
                feature_count = int(fit_metadata["feature_count"])
                validation_prediction = predict_model(
                    model, X, validation_rows, feature_count
                )
                calibration_target = (
                    full_targets["target_state"] if target_kind == "state" else full_targets["target_path"]
                )
                intercept, slope = fit_affine(
                    validation_prediction,
                    calibration_target[validation_rows],
                    np.ones(len(validation_rows), dtype=bool),
                )
                test_prediction = predict_model(model, X, test_matrix_rows, feature_count)
                calibrated = intercept + slope * test_prediction
                model_key = f"{target_kind}:{model_name}"
                metrics, model_daily_rows = evaluate_predictions(
                    model_key,
                    target_kind,
                    horizon,
                    features.dates[test_steps],
                    test_prediction,
                    calibrated.astype(np.float32),
                    test_targets,
                    liquidity,
                    stock_count,
                )
                fit_metadata.update(
                    {
                        "calibration_intercept": intercept,
                        "calibration_slope": slope,
                        "metrics": metrics,
                    }
                )
                summary["models"].setdefault(model_key, {})[str(horizon)] = fit_metadata
                daily_rows.extend(model_daily_rows)
                save_model(
                    model,
                    output_dir / "models" / f"{target_kind}_{model_name}_h{horizon}",
                )
                pd.DataFrame(daily_rows).to_csv(output_dir / "daily_metrics.csv", index=False)
                (output_dir / "summary.json").write_text(
                    json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                print(
                    f"result model={model_key} h={horizon} "
                    f"state_ic={metrics['state_ic']['mean']:.5f} "
                    f"path_ic={metrics['path_ic']['mean']:.5f} "
                    f"top300_path_ic={metrics['path_ic_top300']['mean']:.5f}",
                    flush=True,
                )

    pd.DataFrame(daily_rows).to_csv(output_dir / "daily_metrics.csv", index=False)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output_dir / "feature_names.json").write_text(
        json.dumps(
            {
                "base": layout.base_feature_names,
                "graph": layout.graph_feature_names,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {output_dir / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
