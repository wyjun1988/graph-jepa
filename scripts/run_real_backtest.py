from __future__ import annotations

import argparse
from concurrent.futures import Executor, ThreadPoolExecutor
from contextlib import nullcontext
import hashlib
import json
import pickle
from pathlib import Path
import sys
import time
from typing import Callable, Dict, List, Mapping, Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from stock_v2.backtest import format_pct, run_path_rank_backtest, run_rank_backtest
from stock_v2.data_contract import (
    build_training_data_diagnostics,
    build_training_data_manifest,
    quantized_array_sha256,
)
from stock_v2.event_features import (
    build_event_feature_frames,
    build_event_theme_exposure,
    build_event_ticker_coverage,
)
from stock_v2.fundamental_features import (
    build_fundamental_feature_frames,
    fundamental_coverage,
    load_fundamental_observations,
)
from stock_v2.external_factors import (
    POLICY_RATE_FACTORS,
    build_external_feature_frames,
    build_external_node_feature_frames,
    build_risk_free_period_returns,
    fetch_external_factor_closes,
    resolve_external_factors,
)
from stock_v2.external_etf_nodes import (
    load_external_etf_node_inputs,
    merge_external_node_inputs,
)
from stock_v2.kiwoom_investor import (
    build_investor_feature_frames,
    investor_feature_coverage,
    load_investor_flow_frames,
)
from stock_v2.graph_jepa import (
    DOWNSTREAM_AUXILIARY_TASKS,
    GraphBatch,
    StockGraphJEPA,
    merge_graph_batches,
)
from stock_v2.market_transition_auxiliary import (
    MarketTransitionAuxiliaryTargets,
    attach_market_transition_auxiliary_targets,
    build_market_transition_auxiliary_targets,
)
from stock_v2.market_data import (
    fetch_krx_ohlcv,
    load_universe_manifest,
    make_ohlcv_panel,
    select_krx_universe_from_listing,
    select_universe,
)
from stock_v2.real_features import (
    EDGE_WEIGHT_QUANTIZATION,
    build_edge_tensor,
    build_feature_panel,
    make_real_snapshot,
)
from stock_v2.static_edges import build_industry_edge_arrays, load_industry_codes


def training_autocast(device: torch.device, amp_dtype: str):
    if device.type != "cuda" or amp_dtype == "none":
        return nullcontext()
    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[amp_dtype]
    return torch.autocast(device_type="cuda", dtype=dtype)


def make_training_grad_scaler(device: torch.device, amp_dtype: str):
    enabled = device.type == "cuda" and amp_dtype == "float16"
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        try:
            return torch.amp.GradScaler("cuda", enabled=enabled)
        except TypeError:
            return torch.amp.GradScaler(enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def date_indices(dates: pd.DatetimeIndex, start: str | None = None, end: str | None = None) -> np.ndarray:
    mask = np.ones(len(dates), dtype=bool)
    if start:
        mask &= dates >= pd.Timestamp(start)
    if end:
        mask &= dates <= pd.Timestamp(end)
    return np.flatnonzero(mask)


def parse_int_list(value: str) -> List[int]:
    parsed = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        item = int(part)
        if item < 1:
            raise ValueError("horizons must be positive")
        parsed.append(item)
    if not parsed:
        raise ValueError("empty horizon list")
    return sorted(set(parsed))


def parse_nonnegative_float_list(value: str) -> List[float]:
    parsed = []
    for part in value.split(","):
        part = part.strip()
        if not part:
            continue
        item = float(part)
        if item < 0.0:
            raise ValueError("rollout loss weights must be non-negative")
        parsed.append(item)
    if not parsed or sum(parsed) <= 0.0:
        raise ValueError("rollout loss weights must have a positive sum")
    return parsed


def build_state_feature_weights(
    feature_names: List[str],
    specs: List[str],
) -> List[float]:
    """Build an explicit per-feature reconstruction-loss weight vector."""

    weights = np.ones(len(feature_names), dtype=np.float32)
    feature_index = {name: index for index, name in enumerate(feature_names)}
    for spec in specs:
        name, separator, raw_weight = str(spec).partition("=")
        name = name.strip()
        if not separator or not name or not raw_weight.strip():
            raise ValueError("state feature weights must use FEATURE=WEIGHT")
        if name not in feature_index:
            raise ValueError(f"unknown state feature weight: {name}")
        weight = float(raw_weight)
        if not np.isfinite(weight) or weight < 0.0:
            raise ValueError("state feature weights must be finite and non-negative")
        weights[feature_index[name]] = np.float32(weight)
    if float(weights.sum()) <= 0.0:
        raise ValueError("state feature weights must contain a positive weight")
    return weights.tolist()


def build_temporal_state_feature_weights(
    feature_names: List[str],
    state_feature_weights: List[float],
    excluded_prefixes: List[str],
) -> List[float]:
    """Keep exogenous sensors as context while excluding future innovations."""

    weights = np.asarray(state_feature_weights, dtype=np.float32).copy()
    if weights.shape != (len(feature_names),):
        raise ValueError("state_feature_weights must match feature_names")
    for raw_prefix in excluded_prefixes:
        prefix = str(raw_prefix).strip()
        if not prefix:
            raise ValueError("temporal excluded feature prefixes must be non-empty")
        matched = [
            index
            for index, feature_name in enumerate(feature_names)
            if feature_name.startswith(prefix)
        ]
        if not matched:
            raise ValueError(
                f"temporal excluded feature prefix matched nothing: {prefix}"
            )
        weights[matched] = 0.0
    if float(weights.sum()) <= 0.0:
        raise ValueError("temporal state feature weights must contain a positive weight")
    return weights.tolist()


def rollout_steps_for_offset(args: argparse.Namespace, offset: int) -> int:
    if args.pretrain_task != "temporal":
        return 0
    return max(1, int(round(offset * args.latent_rollout_steps / max(args.temporal_offset, 1))))


def graph_edge_kwargs(args: argparse.Namespace) -> dict[str, float | int | str]:
    return {
        "correlation_mode": str(getattr(args, "edge_correlation_mode", "signed")),
        "event_top_k": int(getattr(args, "event_edge_top_k", 0) or 0),
        "event_min_weight": float(getattr(args, "event_edge_min_weight", 0.05)),
        "event_scale": float(getattr(args, "event_edge_scale", 0.25)),
        "partial_corr_top_k": int(getattr(args, "partial_corr_top_k", 0) or 0),
        "partial_corr_min_abs": float(getattr(args, "partial_corr_min_abs", 0.10)),
        "partial_corr_mode": str(getattr(args, "partial_corr_mode", "signed")),
        "partial_corr_scale": float(getattr(args, "partial_corr_scale", 0.50)),
        "lead_lag_top_k": int(getattr(args, "lead_lag_top_k", 0) or 0),
        "lead_lag_days": int(getattr(args, "lead_lag_days", 1) or 1),
        "lead_lag_min_abs_corr": float(getattr(args, "lead_lag_min_abs_corr", 0.08)),
        "lead_lag_mode": str(getattr(args, "lead_lag_mode", "signed")),
        "lead_lag_scale": float(getattr(args, "lead_lag_scale", 0.50)),
        "policy_rate_edge_scale": float(getattr(args, "policy_rate_edge_scale", 0.0)),
        "ownership_edge_scale": float(getattr(args, "ownership_edge_scale", 0.0)),
        "sequence_window": int(getattr(args, "sequence_window", 0) or 0),
        "factor_sensitivity_top_k": int(
            getattr(args, "factor_sensitivity_top_k", 0) or 0
        ),
        "factor_sensitivity_min_abs_corr": float(
            getattr(args, "factor_sensitivity_min_abs_corr", 0.15)
        ),
        "factor_sensitivity_mode": str(
            getattr(args, "factor_sensitivity_mode", "signed")
        ),
        "factor_sensitivity_scale": float(
            getattr(args, "factor_sensitivity_scale", 0.50)
        ),
        "factor_sensitivity_permute_seed": int(
            getattr(args, "factor_sensitivity_permute_seed", 0) or 0
        ),
    }


HIDDEN_COMPLETION_CHANNELS = (
    "investor_foreign_flow_ratio_1d",
    "investor_institution_flow_ratio_1d",
)

# phase-4 (2026-07-26): fundamentals-hidden completion. (date, ticker) -> (g_rev, g_oi)
# labels exist ONLY inside the undisclosed window (period_end < day < available_at) —
# the value the market genuinely cannot observe yet; known post-hoc for training.
_FUND_HIDDEN: dict | None = None


def load_fund_hidden_targets(path: str, min_peers: int) -> None:
    global _FUND_HIDDEN
    import pandas as _pd
    df = _pd.read_csv(path)
    if min_peers > 0:
        df = df[df["n_peers"] >= min_peers]
    _FUND_HIDDEN = {(str(d), str(t).zfill(6)): (r, o)
                    for d, t, r, o in zip(df["date"], df["ticker"], df["g_rev"], df["g_oi"])}
    print(f"fund-hidden targets: {len(_FUND_HIDDEN):,} firm-days (min_peers={min_peers})", flush=True)


def attach_hidden_completion_targets(
    batch: "GraphBatch",
    features,
    context_steps: np.ndarray,
) -> "GraphBatch":
    """Attach t's structurally-hidden flow (disclosed t+1) to the context batch.

    The panel's flow feature is shift(1), so the value AT ROW t is t-1's flow and
    the value at row t+1 is t's flow. The target for decision step t is therefore
    the row t+1 value -- read only as a label, never fed to the encoder at t.
    Per-date cross-sectionally standardized so the loss is scale-comparable and
    the cross-sectional IC is preserved.
    """

    steps = np.asarray(context_steps, dtype=np.int64)
    node_count = int(features.node_count)
    stock_count = int(features.tradable_count)
    names = list(features.feature_names)
    channels = [names.index(n) for n in HIDDEN_COMPLETION_CHANNELS if n in names]
    if len(channels) != len(HIDDEN_COMPLETION_CHANNELS):
        raise ValueError(
            "hidden completion needs the investor flow channels in the panel; "
            f"found {len(channels)} of {len(HIDDEN_COMPLETION_CHANNELS)}"
        )
    raw = getattr(features, "raw_features", None)
    if raw is None:
        raise ValueError("hidden completion requires features.raw_features")
    width = len(HIDDEN_COMPLETION_CHANNELS)
    n_dates = raw.shape[0]
    values = np.full((len(steps), node_count, width), np.nan, dtype=np.float32)
    for position, step in enumerate(steps):
        nxt = int(step) + 1
        if nxt >= n_dates:
            continue
        for ci, feat_idx in enumerate(channels):
            target = np.asarray(raw[nxt, :stock_count, feat_idx], dtype=np.float64)
            finite = np.isfinite(target)
            if finite.sum() < 3:
                continue
            mean = float(target[finite].mean())
            std = float(target[finite].std())
            if not np.isfinite(std) or std < 1e-12:
                continue
            standardized = np.full(stock_count, np.nan, dtype=np.float64)
            standardized[finite] = (target[finite] - mean) / std
            values[position, :stock_count, ci] = standardized
    if _FUND_HIDDEN is not None:
        tickers = [str(t).zfill(6) for t in list(features.tickers)[:stock_count]]
        fvals = np.full((len(steps), node_count, 2), np.nan, dtype=np.float32)
        for position, step in enumerate(steps):
            day = str(features.dates[int(step)])[:10]
            col = np.full((stock_count, 2), np.nan, dtype=np.float64)
            for si, tk in enumerate(tickers):
                hit = _FUND_HIDDEN.get((day, tk))
                if hit is not None:
                    col[si, 0], col[si, 1] = hit
            for ci in range(2):
                finite = np.isfinite(col[:, ci])
                if finite.sum() < 3:
                    continue
                mean = float(col[finite, ci].mean()); std = float(col[finite, ci].std())
                if not np.isfinite(std) or std < 1e-12:
                    continue
                out = np.full(stock_count, np.nan, dtype=np.float64)
                out[finite] = (col[finite, ci] - mean) / std
                fvals[position, :stock_count, ci] = out
        values = np.concatenate([values, fvals], axis=2)
        width += 2
    batch.hidden_target = torch.from_numpy(values.reshape(-1, width))
    return batch


def install_privileged_hidden(target_batch, features, target_steps):
    """Overwrite the hidden flow channels of a TARGET batch with their true
    future-disclosed values, so the EMA teacher encodes the complete state.

    target_steps are the future decision rows (t+offset). The flow that is true
    ON that row is disclosed one row later (shift(1)), so the truth is
    features.features[step + 1] -- the normalized value the teacher input uses.
    Only finite disclosures overwrite; a missing one keeps the lagged value.
    """

    steps = np.asarray(target_steps, dtype=np.int64)
    node_count = int(features.node_count)
    stock_count = int(features.tradable_count)
    names = list(features.feature_names)
    channels = [names.index(n) for n in HIDDEN_COMPLETION_CHANNELS if n in names]
    if not channels:
        raise ValueError("privileged teacher needs the investor flow channels in the panel")
    norm = features.features
    n_dates = norm.shape[0]
    node_features = target_batch.node_features.clone()
    for position, step in enumerate(steps):
        nxt = int(step) + 1
        if nxt >= n_dates:
            continue
        base = position * node_count
        for feat_idx in channels:
            true_flow = np.asarray(norm[nxt, :stock_count, feat_idx], dtype=np.float32)
            column = torch.from_numpy(true_flow).to(node_features.dtype)
            finite = torch.isfinite(column)
            if not bool(finite.any()):
                continue
            rows = base + torch.arange(stock_count)[finite]
            node_features[rows, feat_idx] = column[finite]
    target_batch.node_features = node_features
    return target_batch


def make_training_batch(
    features,
    steps: np.ndarray,
    args: argparse.Namespace,
    epoch: int,
    full_observation: bool = False,
    executor: Optional[Executor] = None,
    edge_cache: Optional[Dict[int, tuple[torch.Tensor, torch.Tensor]]] = None,
) -> GraphBatch:
    def make_snapshot(step: int) -> GraphBatch:
        return make_real_snapshot(
            features,
            step=int(step),
            hide_ratio=args.hide_ratio,
            full_observation=full_observation,
            edge_window=args.edge_window,
            top_k=args.edge_top_k,
            min_abs_corr=args.min_abs_corr,
            **graph_edge_kwargs(args),
            seed=None if full_observation else args.seed + epoch * 100_000 + int(step),
            mask_strategy=args.mask_strategy,
            edge_cache=edge_cache,
        )

    step_values = [int(step) for step in steps]
    if executor is not None and len(step_values) > 1:
        snapshots = list(executor.map(make_snapshot, step_values))
    else:
        snapshots = [make_snapshot(step) for step in step_values]
    return merge_graph_batches(snapshots)


def attach_entry_path_targets(
    batch: GraphBatch,
    features,
    context_steps: np.ndarray,
    horizon: int,
) -> GraphBatch:
    steps = np.asarray(context_steps, dtype=np.int64)
    node_count = int(features.node_count)
    stock_count = int(features.tradable_count)
    expected_nodes = len(steps) * node_count
    if batch.node_features.shape[0] != expected_nodes:
        raise ValueError("entry path targets do not align with the merged graph batch")
    source = features.target_return_paths.get(int(horizon))
    if source is None:
        raise ValueError(f"missing target entry path horizon {horizon}")
    values = np.full((len(steps), node_count), np.nan, dtype=np.float32)
    values[:, :stock_count] = np.asarray(source[steps, :stock_count], dtype=np.float32)
    batch.target_entry_path = torch.from_numpy(values.reshape(-1))
    return batch


# Trailing window for the plan's causal scale, in sessions, and the minimum
# finite observations it must contain. 60 sessions is the edge window the graph
# already uses, so it needs no separate justification; the floor keeps a scale
# from being fitted on a handful of survivors early in a fold.
CAUSAL_PLAN_SCALE_LOOKBACK = 60
CAUSAL_PLAN_SCALE_MIN_OBSERVATIONS = 30


def attach_downstream_targets(
    batch: GraphBatch,
    features,
    context_steps: np.ndarray,
    horizon: int,
) -> GraphBatch:
    """Attach causal, cross-sectionally standardized specialist targets."""

    steps = np.asarray(context_steps, dtype=np.int64)
    node_count = int(features.node_count)
    stock_count = int(features.tradable_count)
    expected_nodes = len(steps) * node_count
    if batch.node_features.shape[0] != expected_nodes:
        raise ValueError("downstream targets do not align with the graph batch")
    horizon = int(horizon)
    if horizon < 1:
        raise ValueError("downstream target horizon must be positive")
    task_count = len(DOWNSTREAM_AUXILIARY_TASKS)
    values = np.full((len(steps), node_count, task_count), np.nan, dtype=np.float32)
    causal_scales = np.full((len(steps), node_count, 2), np.nan, dtype=np.float32)
    scales = np.full((len(steps), node_count, 2), np.nan, dtype=np.float32)
    path_source = features.target_return_paths.get(horizon)
    if path_source is None:
        raise ValueError(f"missing target entry path horizon {horizon}")

    for position, step in enumerate(steps):
        entry = np.asarray(
            features.open[int(step) + 1, :stock_count], dtype=np.float64
        )
        close_path = np.asarray(
            features.close[
                int(step) + 1 : int(step) + horizon + 1,
                :stock_count,
            ],
            dtype=np.float64,
        )
        entry_valid = np.isfinite(entry) & (entry > 0.0)
        price_valid = entry_valid & np.isfinite(close_path).all(axis=0)
        path_returns = np.divide(
            close_path,
            entry[None, :],
            out=np.full_like(close_path, np.nan),
            where=entry_valid[None, :],
        ) - 1.0
        raw = np.full((stock_count, task_count), np.nan, dtype=np.float64)
        raw[:, 0] = np.asarray(
            path_source[int(step), :stock_count], dtype=np.float64
        )
        if horizon > 1 and price_valid.any():
            raw[price_valid, 1] = np.max(
                path_returns[:, price_valid], axis=0
            )
            raw[price_valid, 2] = np.min(
                path_returns[:, price_valid], axis=0
            )
        future_returns = np.asarray(
            features.returns_1d[
                int(step) + 1 : int(step) + horizon + 1,
                :stock_count,
            ],
            dtype=np.float64,
        )
        returns_valid = np.isfinite(future_returns).all(axis=0)
        if returns_valid.any():
            raw[returns_valid, 3] = np.sqrt(
                np.mean(np.square(future_returns[:, returns_valid]), axis=0)
            )

        # Intent 2. The NET displacement per day, anchored on the decision date's
        # close -- not the path's roughness, which is task 3. A shock that
        # oscillates violently and ends where it started has high realized
        # volatility and zero continuation, and the two tasks must be able to
        # say so independently.
        decision_close = np.asarray(
            features.close[int(step), :stock_count], dtype=np.float64
        )
        horizon_close = np.asarray(
            features.close[int(step) + horizon, :stock_count], dtype=np.float64
        )
        continuation_valid = (
            np.isfinite(decision_close)
            & (decision_close > 0.0)
            & np.isfinite(horizon_close)
        )
        if continuation_valid.any():
            raw[continuation_valid, 4] = (
                np.abs(
                    horizon_close[continuation_valid] / decision_close[continuation_valid]
                    - 1.0
                )
                / float(horizon)
            )
        for task_index in range(task_count):
            target = raw[:, task_index]
            valid = np.isfinite(target)
            if valid.sum() < 3:
                continue
            mean = float(target[valid].mean())
            std = float(target[valid].std())
            if not np.isfinite(std) or std < 1e-12:
                continue
            values[position, :stock_count, task_index] = (
                (target - mean) / std
            ).astype(np.float32)
            if task_index == 0:
                # Carry the path task's per-date scale so the plan loss can
                # invert the standardization back to raw return units.
                scales[position, :stock_count, 0] = np.float32(mean)
                scales[position, :stock_count, 1] = np.float32(std)

        # The plan cannot use the scale above: it is this date's REALIZED
        # cross-sectional mean, and since the head's output is zero-mean across
        # the cross-section, it would be the only thing separating the horizons
        # in the plan's decision variable. argmax over it is hindsight -- a rule
        # that ignores the model entirely and takes argmax_h mean_h(t) scored
        # +0.01404 over hold10, matching the leaky arm's +0.01416 plan_adv.
        #
        # This one is estimated from paths that had already finished: a path
        # entered at s for horizon h is unknown until s+h, so at date `step` the
        # last usable entry is step-h (step-h-1 here, one session of margin).
        causal_last = int(step) - horizon - 1
        causal_first = max(0, causal_last - CAUSAL_PLAN_SCALE_LOOKBACK + 1)
        if causal_last >= causal_first:
            window = np.asarray(
                path_source[causal_first : causal_last + 1, :stock_count],
                dtype=np.float64,
            )
            finite = window[np.isfinite(window)]
            if finite.size >= CAUSAL_PLAN_SCALE_MIN_OBSERVATIONS:
                causal_mean = float(finite.mean())
                causal_std = float(finite.std())
                if np.isfinite(causal_std) and causal_std >= 1e-12:
                    causal_scales[position, :stock_count, 0] = np.float32(causal_mean)
                    causal_scales[position, :stock_count, 1] = np.float32(causal_std)

    batch.target_downstream_scale = torch.from_numpy(
        scales.reshape(-1, 2)
    )
    batch.target_downstream_causal_scale = torch.from_numpy(
        causal_scales.reshape(-1, 2)
    )
    batch.target_downstream = torch.from_numpy(
        values.reshape(expected_nodes, task_count)
    )
    return batch


def filter_history_for_training(
    raw: Dict[str, pd.DataFrame],
    train_end: str,
    min_train_rows: int,
) -> Dict[str, pd.DataFrame]:
    cutoff = pd.Timestamp(train_end)
    kept: Dict[str, pd.DataFrame] = {}
    dropped = 0
    for ticker, frame in raw.items():
        train_rows = frame.loc[frame.index <= cutoff]
        if len(train_rows) >= min_train_rows:
            kept[ticker] = frame
        else:
            dropped += 1
    if dropped:
        print(f"dropped {dropped} tickers with fewer than {min_train_rows} rows before train_end", flush=True)
    return kept


def temporal_training_indices(
    train_indices: np.ndarray,
    edge_window: int,
    max_rollout_offset: int,
    total_steps: int,
) -> np.ndarray:
    """Select temporal contexts whose entire supervised path stays in train."""

    if len(train_indices) == 0:
        return np.asarray([], dtype=np.int64)
    max_offset = max(1, int(max_rollout_offset))
    last_train_step = int(train_indices.max())
    last_context_step = min(last_train_step - max_offset, int(total_steps) - 1 - max_offset)
    return train_indices[(train_indices >= int(edge_window)) & (train_indices <= last_context_step)]


def validate_expected_training_manifest(
    manifest: Mapping[str, object],
    expected_sha256: str | None,
) -> None:
    expected = str(expected_sha256 or "").strip().lower()
    if not expected:
        return
    actual = str(manifest.get("sha256") or "").strip().lower()
    if actual != expected:
        raise RuntimeError(
            "training panel manifest does not match the frozen preflight "
            f"(expected={expected}, actual={actual})"
        )


def build_training_edge_manifest(
    features,
    edge_cache: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[str, object]:
    """Fingerprint every causal edge index and weight used during training."""

    step_rows: list[dict[str, object]] = []
    total_edges = 0
    for step in sorted(int(value) for value in edge_cache):
        edge_index, edge_weight = edge_cache[step]
        canonical_index = np.ascontiguousarray(
            edge_index.detach().cpu().numpy(),
            dtype="<i8",
        )
        weights = edge_weight.detach().cpu().numpy()
        edge_count = int(canonical_index.shape[1])
        total_edges += edge_count
        step_rows.append(
            {
                "step": step,
                "date": str(pd.Timestamp(features.dates[step]).date()),
                "edges": edge_count,
                "edge_index_sha256": hashlib.sha256(
                    canonical_index.tobytes()
                ).hexdigest(),
                "edge_weight_sha256": quantized_array_sha256(
                    "edge_weight",
                    weights,
                ),
            }
        )
    payload: dict[str, object] = {
        "schema_version": 1,
        "weight_quantization": EDGE_WEIGHT_QUANTIZATION,
        "steps": step_rows,
        "step_count": len(step_rows),
        "total_edges": total_edges,
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return {**payload, "sha256": hashlib.sha256(encoded).hexdigest()}


def validate_expected_training_edge_manifest(
    manifest: Mapping[str, object],
    expected_sha256: str | None,
) -> None:
    expected = str(expected_sha256 or "").strip().lower()
    if not expected:
        return
    actual = str(manifest.get("sha256") or "").strip().lower()
    if actual != expected:
        raise RuntimeError(
            "training edge manifest does not match the frozen preflight "
            f"(expected={expected}, actual={actual})"
        )


def build_training_edge_cache(
    features,
    usable_steps: np.ndarray,
    args: argparse.Namespace,
) -> Dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Precompute causal dynamic edges once for all temporal training snapshots."""

    required = [usable_steps]
    if args.pretrain_task == "temporal":
        required.extend(usable_steps + int(offset) for offset in args.rollout_offsets)
    steps = np.unique(np.concatenate(required)).astype(np.int64)
    edge_kwargs = graph_edge_kwargs(args)

    def build_one(step: int) -> tuple[int, tuple[torch.Tensor, torch.Tensor]]:
        return int(step), build_edge_tensor(
            features,
            step=int(step),
            edge_window=args.edge_window,
            top_k=args.edge_top_k,
            min_abs_corr=args.min_abs_corr,
            **edge_kwargs,
        )

    worker_count = min(max(1, int(args.snapshot_workers)), len(steps))
    if worker_count > 1:
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            pairs = list(executor.map(build_one, steps.tolist()))
    else:
        pairs = [build_one(int(step)) for step in steps]
    cache = dict(pairs)
    edge_manifest = build_training_edge_manifest(features, cache)
    reports_dir = Path(str(getattr(args, "reports_dir", "reports")))
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "training_edge_manifest.json").write_text(
        json.dumps(edge_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    validate_expected_training_edge_manifest(
        edge_manifest,
        getattr(args, "expected_training_edge_manifest_sha256", ""),
    )
    print(
        f"training edge cache: steps={len(cache)} "
        f"edges={edge_manifest['total_edges']} workers={worker_count} "
        f"sha256={str(edge_manifest['sha256'])[:16]}",
        flush=True,
    )
    return cache


def train_jepa(
    model: StockGraphJEPA,
    features,
    train_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
    transition_targets: MarketTransitionAuxiliaryTargets | None = None,
    checkpoint_callback: Callable[[int, List[Dict[str, float]]], None] | None = None,
) -> List[Dict[str, float]]:
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=1e-4)
    n_train = sum(p.numel() for p in trainable); n_all = sum(p.numel() for p in model.parameters())
    print(f"optimizer: trainable {n_train:,}/{n_all:,} params ({100*n_train/max(n_all,1):.0f}%)", flush=True)
    scaler = make_training_grad_scaler(device, args.amp_dtype)
    history: List[Dict[str, float]] = []
    usable = train_indices[train_indices >= args.edge_window]
    if args.pretrain_task == "temporal":
        usable = temporal_training_indices(
            train_indices,
            edge_window=args.edge_window,
            max_rollout_offset=max(args.rollout_offsets),
            total_steps=len(features.dates),
        )
        if len(usable) == 0:
            raise ValueError("no temporal training contexts remain after the train-end guard")
        print(
            f"temporal train guard: contexts={len(usable)} "
            f"last_context={features.dates[int(usable[-1])].date()} "
            f"max_offset={max(args.rollout_offsets)}",
            flush=True,
        )
    edge_cache = (
        build_training_edge_cache(features, usable, args)
        if getattr(args, "cache_training_edges", True)
        else None
    )
    worker_count = min(args.snapshot_workers, args.train_batch_size)
    snapshot_executor = ThreadPoolExecutor(max_workers=worker_count) if worker_count > 1 else None

    completed_optimizer_steps = 0
    stop_after_epoch = False
    for epoch in range(1, args.epochs + 1):
        epoch_started = time.perf_counter()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        rng = np.random.default_rng(args.seed + epoch)
        shuffled = rng.permutation(usable)
        total = 0.0
        latent = 0.0
        latent_std_acc = 0.0
        latent_pr_acc = 0.0
        state = 0.0
        mae = 0.0
        return_corr = 0.0
        entry_path_corr = 0.0
        downstream_auxiliary = 0.0
        downstream_market = 0.0
        downstream_transition = 0.0
        temporal_impact_weight = 0.0
        current_imputation = 0.0
        current_imputation_mae = 0.0
        # Generic so a diagnostic added to plan_timing_loss later shows up here
        # without another patch. Empty unless the plan loss is on.
        plan_totals: dict[str, float] = {}
        count = 0
        optimizer_steps = 0

        for batch_start in range(0, len(shuffled), args.train_batch_size):
            steps = shuffled[batch_start : batch_start + args.train_batch_size]
            batch = make_training_batch(
                features,
                steps,
                args,
                epoch,
                executor=snapshot_executor,
                edge_cache=edge_cache,
            )
            if getattr(args, "fund_hidden_target_path", None) and _FUND_HIDDEN is None:
                load_fund_hidden_targets(args.fund_hidden_target_path, int(args.fund_hidden_min_peers))
            if float(getattr(args, "hidden_completion_weight", 0.0)) > 0.0:
                batch = attach_hidden_completion_targets(batch, features, steps)
            batch = batch.to(device)
            with training_autocast(device, args.amp_dtype):
                if args.pretrain_task == "temporal":
                    target_batches = []
                    rollout_steps = []
                    for offset in args.rollout_offsets:
                        target_batch = make_training_batch(
                            features,
                            steps + int(offset),
                            args,
                            epoch,
                            full_observation=True,
                            executor=snapshot_executor,
                            edge_cache=edge_cache,
                        )
                        if (
                            args.entry_path_correlation_loss_weight > 0.0
                            or args.downstream_market_loss_weight > 0.0
                        ):
                            target_batch = attach_entry_path_targets(
                                target_batch,
                                features,
                                steps,
                                int(offset),
                            )
                        if (
                            args.downstream_auxiliary_loss_weight > 0.0
                            or args.downstream_plan_loss_weight > 0.0
                        ):
                            target_batch = attach_downstream_targets(
                                target_batch,
                                features,
                                steps,
                                int(offset),
                            )
                        if (
                            args.downstream_transition_loss_weight > 0.0
                            or args.temporal_impact_loss_mix > 0.0
                        ):
                            if transition_targets is None:
                                raise ValueError(
                                    "temporal impact supervision requires fitted targets"
                                )
                            target_batch = attach_market_transition_auxiliary_targets(
                                target_batch,
                                transition_targets,
                                steps,
                                int(offset),
                            )
                        if getattr(args, "privileged_hidden_teacher", False):
                            target_batch = install_privileged_hidden(
                                target_batch, features, steps + int(offset)
                            )
                        target_batches.append(target_batch.to(device))
                        rollout_steps.append(
                            rollout_steps_for_offset(args, int(offset))
                        )
                    loss, metrics = model.temporal_multi_loss(
                        batch,
                        target_batches,
                        rollout_steps=rollout_steps,
                        rollout_loss_weights=args.rollout_loss_weights,
                        target_horizons=args.rollout_offsets,
                    )
                else:
                    loss, metrics = model.loss(batch)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            model.update_target_encoder(decay=args.ema_decay ** len(steps))
            optimizer_steps += 1
            completed_optimizer_steps += 1

            total += metrics["loss"] * len(steps)
            latent += metrics["latent_loss"] * len(steps)
            latent_std_acc += metrics.get("latent_std", 0.0) * len(steps)
            latent_pr_acc += metrics.get("latent_participation", 0.0) * len(steps)
            state += metrics["state_loss"] * len(steps)
            mae += metrics["masked_mae"] * len(steps)
            return_corr += metrics.get("return_corr_loss", 0.0) * len(steps)
            entry_path_corr += metrics.get("entry_path_corr_loss", 0.0) * len(steps)
            downstream_auxiliary += metrics.get(
                "downstream_auxiliary_loss", 0.0
            ) * len(steps)
            downstream_market += metrics.get(
                "downstream_market_loss", 0.0
            ) * len(steps)
            downstream_transition += metrics.get(
                "downstream_transition_loss", 0.0
            ) * len(steps)
            temporal_impact_weight += metrics.get(
                "temporal_impact_weight_mean", 1.0
            ) * len(steps)
            current_imputation += metrics.get("current_imputation_loss", 0.0) * len(steps)
            current_imputation_mae += metrics.get("current_imputation_mae", 0.0) * len(steps)
            for plan_key, plan_value in metrics.items():
                if plan_key.startswith("plan_"):
                    plan_totals[plan_key] = plan_totals.get(plan_key, 0.0) + float(
                        plan_value
                    ) * len(steps)
            count += len(steps)
            if (
                args.max_train_steps > 0
                and completed_optimizer_steps >= args.max_train_steps
            ):
                stop_after_epoch = True
                break

        epoch_seconds = time.perf_counter() - epoch_started
        row = {
            "epoch": epoch,
            "loss": total / max(count, 1),
            "latent_loss": latent / max(count, 1),
            "latent_std": latent_std_acc / max(count, 1),
            "latent_participation": latent_pr_acc / max(count, 1),
            "state_loss": state / max(count, 1),
            "masked_mae": mae / max(count, 1),
            "return_corr_loss": return_corr / max(count, 1),
            "entry_path_corr_loss": entry_path_corr / max(count, 1),
            "downstream_auxiliary_loss": downstream_auxiliary / max(count, 1),
            "downstream_market_loss": downstream_market / max(count, 1),
            "downstream_transition_loss": downstream_transition / max(count, 1),
            "temporal_impact_weight_mean": temporal_impact_weight
            / max(count, 1),
            **{
                plan_key: plan_total / max(count, 1)
                for plan_key, plan_total in plan_totals.items()
            },
            "current_imputation_loss": current_imputation / max(count, 1),
            "current_imputation_mae": current_imputation_mae / max(count, 1),
            "epoch_seconds": epoch_seconds,
            "samples_per_second": count / max(epoch_seconds, 1e-9),
            "optimizer_steps": optimizer_steps,
            "peak_cuda_memory_mib": (
                torch.cuda.max_memory_allocated(device) / (1024 * 1024)
                if device.type == "cuda"
                else 0.0
            ),
        }
        history.append(row)
        print(
            f"epoch={epoch:02d} loss={row['loss']:.4f} "
            f"latent={row['latent_loss']:.4f} state={row['state_loss']:.4f} "
            f"masked_mae={row['masked_mae']:.4f} "
            f"return_corr={row['return_corr_loss']:.4f} "
            f"entry_path_corr={row['entry_path_corr_loss']:.4f} "
            f"downstream_aux={row['downstream_auxiliary_loss']:.4f} "
            f"downstream_market={row['downstream_market_loss']:.4f} "
            f"downstream_transition={row['downstream_transition_loss']:.4f} "
            f"impact_weight={row['temporal_impact_weight_mean']:.4f} "
            f"current_impute={row['current_imputation_loss']:.4f} "
            f"current_mae={row['current_imputation_mae']:.4f} "
            + (
                f"plan_adv={row['plan_advantage_mean']:+.5f} "
                f"plan_oracle_adv={row['plan_oracle_advantage_mean']:+.5f} "
                f"plan_entropy={row['plan_weight_entropy']:.4f} "
                if "plan_advantage_mean" in row
                else ""
            )
            + f"seconds={row['epoch_seconds']:.1f} "
            f"samples_per_sec={row['samples_per_second']:.2f} "
            f"optimizer_steps={row['optimizer_steps']} "
            f"peak_vram_mib={row['peak_cuda_memory_mib']:.0f}",
            flush=True,
        )
        if checkpoint_callback is not None:
            checkpoint_callback(epoch, history)
        if stop_after_epoch:
            print(
                f"stopped after max_train_steps={args.max_train_steps}",
                flush=True,
            )
            break

    if snapshot_executor is not None:
        snapshot_executor.shutdown(wait=True)
    return history


@torch.no_grad()
def encode_all(
    model: StockGraphJEPA,
    features,
    args: argparse.Namespace,
    device: torch.device,
    rollout_steps: int = 0,
) -> np.ndarray:
    model.eval()
    encoded = np.zeros((len(features.dates), features.node_count, args.hidden_dim), dtype=np.float32)
    for step in range(len(features.dates)):
        batch = make_real_snapshot(
            features,
            step=step,
            full_observation=True,
            edge_window=args.edge_window,
            top_k=args.edge_top_k,
            min_abs_corr=args.min_abs_corr,
            **graph_edge_kwargs(args),
        ).to(device)
        context = (
            model.encode_context(batch)
            if rollout_steps <= 0
            else model.encode_temporal_context(batch)
        )
        latent = context if rollout_steps <= 0 else model.rollout_latent(context, steps=rollout_steps)
        encoded[step] = latent.detach().cpu().numpy()
    return encoded


def fit_return_model(
    design: np.ndarray,
    targets: np.ndarray,
    train_indices: np.ndarray,
) -> object:
    x_train = []
    y_train = []
    for step in train_indices:
        x_step = design[step]
        y_step = targets[step]
        valid = np.isfinite(x_step).all(axis=1) & np.isfinite(y_step)
        x_train.append(x_step[valid])
        y_train.append(y_step[valid])

    X = np.vstack(x_train)
    y = np.concatenate(y_train)
    model = make_pipeline(StandardScaler(), Ridge(alpha=2.0))
    model.fit(X, y)
    return model


def predict_scores(model: object, design: np.ndarray) -> np.ndarray:
    scores = np.zeros((design.shape[0], design.shape[1]), dtype=np.float32)
    for step in range(design.shape[0]):
        X = design[step]
        scores[step] = model.predict(X).astype(np.float32)
    return scores


def build_path_predictions(
    model: StockGraphJEPA,
    features,
    train_indices: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> tuple[Dict[int, object], Dict[int, np.ndarray]]:
    models: Dict[int, object] = {}
    predictions: Dict[int, np.ndarray] = {}
    for horizon in args.path_horizons_list:
        rollout_steps = rollout_steps_for_offset(args, int(horizon))
        rolled_embeddings = encode_all(model, features, args, device, rollout_steps=rollout_steps)
        design = np.concatenate([rolled_embeddings, features.features], axis=2)
        target = features.target_return_paths[int(horizon)]
        horizon_model = fit_return_model(design, target, train_indices)
        models[int(horizon)] = horizon_model
        predictions[int(horizon)] = predict_scores(horizon_model, design)
    return models, predictions


def neighbor_support(
    features,
    step: int,
    node_scores: np.ndarray,
    args: argparse.Namespace,
) -> np.ndarray:
    edge_index, edge_weight = build_edge_tensor(
        features,
        step=step,
        edge_window=args.edge_window,
        top_k=args.edge_top_k,
        min_abs_corr=args.min_abs_corr,
        **graph_edge_kwargs(args),
    )
    support = np.zeros(node_scores.shape[0], dtype=np.float32)
    denom = np.zeros(node_scores.shape[0], dtype=np.float32)
    if edge_index.numel() == 0:
        return support
    src = edge_index[0].numpy()
    dst = edge_index[1].numpy()
    weights = edge_weight.numpy().astype(np.float32)
    for s, d, w in zip(src, dst, weights):
        if np.isfinite(node_scores[s]):
            support[d] += float(w) * float(node_scores[s])
            denom[d] += abs(float(w))
    valid = denom > 1e-6
    support[valid] /= denom[valid]
    return support


def path_aware_scores(
    features,
    predictions: Dict[int, np.ndarray],
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    horizons = sorted(predictions)
    path = np.stack([predictions[h] for h in horizons], axis=2)
    scores = np.full(path.shape[:2], np.nan, dtype=np.float32)
    exit_horizons = np.full(path.shape[:2], horizons[-1], dtype=np.int32)
    peak_returns = np.full(path.shape[:2], np.nan, dtype=np.float32)
    for step in range(path.shape[0]):
        step_path = path[step]
        if not np.isfinite(step_path).any():
            continue
        peak_idx = np.nanargmax(np.where(np.isfinite(step_path), step_path, -np.inf), axis=1)
        for node_idx, h_idx in enumerate(peak_idx.tolist()):
            curve = step_path[node_idx]
            if not np.isfinite(curve).any():
                continue
            peak = float(curve[h_idx])
            exit_horizons[step, node_idx] = int(horizons[h_idx])
            peak_returns[step, node_idx] = peak
            prefix = curve[: h_idx + 1]
            suffix = curve[h_idx:]
            downside = abs(min(0.0, float(np.nanmin(prefix)))) if np.isfinite(prefix).any() else 0.0
            giveback = max(0.0, peak - float(np.nanmin(suffix))) if np.isfinite(suffix).any() else 0.0
            scores[step, node_idx] = peak - args.path_downside_weight * downside - args.path_giveback_weight * giveback
        support = np.zeros(path.shape[1], dtype=np.float32)
        for horizon_idx, horizon in enumerate(horizons):
            support_for_h = neighbor_support(features, step, step_path[:, horizon_idx], args)
            support[exit_horizons[step] == horizon] = support_for_h[exit_horizons[step] == horizon]
        scores[step] += args.path_support_weight * support
    return scores, exit_horizons, peak_returns


def momentum_scores(features, name: str = "return_20d") -> np.ndarray:
    return features.features[:, :, features.feature_index(name)]


def write_summary(
    path: Path,
    args: argparse.Namespace,
    features,
    history: List[Dict[str, float]],
    metrics: Dict[str, Dict[str, float]],
    raw_metrics: Dict[str, Dict[str, float]],
    jepa_only_metrics: Dict[str, Dict[str, float]],
    momentum_metrics: Dict[str, Dict[str, float]],
    trades: Dict[str, pd.DataFrame],
) -> None:
    final = history[-1] if history else {}
    lines = [
        "# stock-v2 Real Data Graph-JEPA Backtest",
        "",
        "## Setup",
        "",
        f"- Universe: {len(features.tickers)} KRX large-cap stocks",
        f"- Date range after feature cleaning: {features.dates.min().date()} to {features.dates.max().date()}",
        f"- Train end: {args.train_end}",
        f"- Horizon: {args.horizon} trading days",
        f"- Top K: {args.top_k}",
        f"- Roundtrip cost: {args.cost_bps:.1f} bps",
        f"- JEPA epochs: {args.epochs}",
        f"- Hidden dim: {args.hidden_dim}",
        "",
        "## Pretraining",
        "",
        f"- Final loss: {final.get('loss', float('nan')):.4f}",
        f"- Final latent loss: {final.get('latent_loss', float('nan')):.4f}",
        f"- Final masked state MAE: {final.get('masked_mae', float('nan')):.4f}",
        "",
        "## Backtest Metrics",
        "",
        "| Strategy | Periods | Total Return | CAGR | Sharpe | Excess CAGR | Excess Sharpe | Max DD | Hit Rate | Avg Period |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    merged = {
        "graph_jepa_ridge": metrics.get("graph_jepa_ridge", {}),
        "jepa_only_ridge": jepa_only_metrics.get("jepa_only_ridge", {}),
        "raw_ridge": raw_metrics.get("raw_ridge", {}),
        "momentum_20d": momentum_metrics.get("momentum_20d", {}),
        "equal_weight_benchmark": metrics.get("equal_weight_benchmark", {}),
    }
    for name, row in merged.items():
        excess_cagr = format_pct(row["excess_cagr"]) if "excess_cagr" in row else "n/a"
        excess_sharpe = f"{row['excess_sharpe']:+.2f}" if "excess_sharpe" in row else "n/a"
        lines.append(
            f"| {name} | {int(row.get('periods', 0))} | "
            f"{format_pct(row.get('total_return', 0.0))} | "
            f"{format_pct(row.get('cagr', 0.0))} | "
            f"{row.get('sharpe', 0.0):+.2f} | "
            f"{excess_cagr} | "
            f"{excess_sharpe} | "
            f"{format_pct(row.get('max_drawdown', 0.0))} | "
            f"{format_pct(row.get('hit_rate', 0.0))} | "
            f"{format_pct(row.get('avg_period_return', 0.0))} |"
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- This is a research backtest, not a trading recommendation.",
            "- The signal is trained only on dates up to the train-end date.",
            "- The rebalance schedule is non-overlapping by horizon to reduce overlap bias.",
            f"- Event feature paths: {', '.join(args.event_path) if getattr(args, 'event_path', []) else 'none'}.",
            (
                f"- Event co-theme edges: top_k={getattr(args, 'event_edge_top_k', 0)} "
                f"scale={getattr(args, 'event_edge_scale', 0.25):.3f} "
                f"min_weight={getattr(args, 'event_edge_min_weight', 0.05):.3f}."
            ),
            "",
            "## Last Graph-JEPA Trades",
            "",
        ]
    )
    graph_trades = trades.get("graph_jepa_ridge", pd.DataFrame())
    if graph_trades.empty:
        lines.append("- No trades generated.")
    else:
        for _, row in graph_trades.tail(10).iterrows():
            lines.append(
                f"- {pd.Timestamp(row['date']).date()}: {row['selected']} "
                f"return={format_pct(float(row['period_return']))}"
            )

    for title, key in [
        ("Last JEPA-Only Trades", "jepa_only_ridge"),
        ("Last Raw Ridge Trades", "raw_ridge"),
        ("Last Momentum Trades", "momentum_20d"),
    ]:
        lines.extend(["", f"## {title}", ""])
        frame = trades.get(key, pd.DataFrame())
        if frame.empty:
            lines.append("- No trades generated.")
        else:
            for _, row in frame.tail(5).iterrows():
                lines.append(
                    f"- {pd.Timestamp(row['date']).date()}: {row['selected']} "
                    f"return={format_pct(float(row['period_return']))}"
                )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real KRX Graph-JEPA backtest.")
    parser.add_argument("--start", default="2020-01-01")
    parser.add_argument("--end", default=None)
    parser.add_argument("--train-end", default="2023-12-29")
    parser.add_argument("--universe", choices=["manual", "krx"], default="manual")
    parser.add_argument("--universe-manifest", default=None)
    parser.add_argument("--max-tickers", type=int, default=28)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--cost-bps", type=float, default=30.0)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--hide-ratio", type=float, default=0.30)
    parser.add_argument(
        "--mask-strategy",
        choices=[
            "random_cell",
            "feature_group",
            "node_block",
            "mixed",
            "operational_mixed",
        ],
        default="random_cell",
    )
    parser.add_argument("--path-horizons", default="1,2,3,5,10")
    parser.add_argument("--path-support-weight", type=float, default=0.35)
    parser.add_argument("--path-downside-weight", type=float, default=0.50)
    parser.add_argument("--path-giveback-weight", type=float, default=0.15)
    parser.add_argument("--edge-window", type=int, default=60)
    parser.add_argument("--edge-top-k", type=int, default=6)
    parser.add_argument("--min-abs-corr", type=float, default=0.20)
    parser.add_argument("--edge-correlation-mode", choices=["signed", "abs", "positive", "negative", "none"], default="signed")
    parser.add_argument("--graph-neighbor-scale", type=float, default=1.0)
    parser.add_argument("--temporal-graph-neighbor-scale", type=float, default=None)
    parser.add_argument("--temporal-stock-edge-scale", type=float, default=1.0)
    parser.add_argument("--global-stock-context", action="store_true")
    parser.add_argument("--partial-corr-top-k", type=int, default=0)
    parser.add_argument("--partial-corr-min-abs", type=float, default=0.10)
    parser.add_argument("--partial-corr-mode", choices=["signed", "abs", "positive", "negative"], default="signed")
    parser.add_argument("--partial-corr-scale", type=float, default=0.50)
    parser.add_argument("--lead-lag-top-k", type=int, default=0)
    parser.add_argument("--lead-lag-days", type=int, default=1)
    parser.add_argument("--lead-lag-min-abs-corr", type=float, default=0.08)
    parser.add_argument("--lead-lag-mode", choices=["signed", "abs", "positive", "negative"], default="signed")
    parser.add_argument("--lead-lag-scale", type=float, default=0.50)
    parser.add_argument("--policy-rate-edge-scale", type=float, default=0.0)
    parser.add_argument("--ownership-edge-scale", type=float, default=0.0)
    parser.add_argument("--ownership-edge-path", default=None)
    parser.add_argument("--earnings-features", action="store_true")
    parser.add_argument("--return-lag-features", type=int, default=0)
    parser.add_argument("--sequence-window", type=int, default=0)
    parser.add_argument("--sequence-layers", type=int, default=2)
    parser.add_argument("--sequence-heads", type=int, default=8)
    parser.add_argument("--sequence-residual", action="store_true")
    parser.add_argument("--factor-sensitivity-top-k", type=int, default=0)
    parser.add_argument("--factor-sensitivity-min-abs-corr", type=float, default=0.15)
    parser.add_argument(
        "--factor-sensitivity-mode",
        default="signed",
        choices=["signed", "positive", "negative", "abs", "none"],
    )
    parser.add_argument("--factor-sensitivity-scale", type=float, default=0.50)
    parser.add_argument("--factor-sensitivity-permute-seed", type=int, default=0)
    parser.add_argument("--min-train-rows", type=int, default=None)
    parser.add_argument("--min-usable-train-rows", type=int, default=120)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--snapshot-workers", type=int, default=1)
    parser.add_argument(
        "--amp-dtype",
        choices=["none", "float16", "bfloat16"],
        default="none",
        help="CUDA training autocast dtype; ignored on non-CUDA devices.",
    )
    parser.add_argument(
        "--max-train-steps",
        type=int,
        default=0,
        help="Stop after this many optimizer steps; zero runs all epochs.",
    )
    parser.add_argument("--cache-training-edges", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--ema-decay", type=float, default=0.98)
    parser.add_argument("--latent-loss-weight", type=float, default=1.0)
    parser.add_argument("--state-loss-weight", type=float, default=0.35)
    parser.add_argument("--current-imputation-loss-weight", type=float, default=0.0)
    parser.add_argument("--temporal-state-context-skip", action="store_true")
    parser.add_argument(
        "--temporal-head-input",
        choices=["context_skip", "future", "context"],
        default=None,
        help="예측 헤드 입력. context_skip=[현재, 미래-현재] (기존 기본), "
             "future=미래만, context=현재만(미래 잠재 미사용). "
             "미지정 시 --temporal-state-context-skip 으로부터 유도 (하위호환).",
    )
    parser.add_argument(
        "--state-feature-weight",
        action="append",
        default=[],
        metavar="FEATURE=WEIGHT",
        help="Override a normalized state reconstruction-loss weight; repeatable.",
    )
    parser.add_argument(
        "--temporal-exclude-feature-prefix",
        action="append",
        default=[],
        help=(
            "Keep matching features as context/current targets but persist them "
            "instead of predicting future innovations; repeatable."
        ),
    )
    parser.add_argument("--return-correlation-loss-weight", type=float, default=0.0)
    parser.add_argument(
        "--entry-path-correlation-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--downstream-plan-loss-weight", type=float, default=0.0)
    parser.add_argument("--plan-temperature", type=float, default=0.01)
    parser.add_argument("--plan-buy-sell", action="store_true")
    parser.add_argument("--plan-permute-seed", type=int, default=0)
    parser.add_argument(
        "--downstream-auxiliary-loss-weight",
        type=float,
        default=0.0,
    )
    parser.add_argument("--downstream-path-weight", type=float, default=1.0)
    parser.add_argument("--downstream-mfe-weight", type=float, default=0.25)
    parser.add_argument("--downstream-mae-weight", type=float, default=0.25)
    parser.add_argument("--downstream-volatility-weight", type=float, default=1.0)
    parser.add_argument("--downstream-continuation-weight", type=float, default=1.0)
    parser.add_argument("--hidden-completion-weight", type=float, default=0.0)
    parser.add_argument("--latent-variance-weight", type=float, default=0.0,
                        help="VICReg variance hinge on predicted latents (anti-collapse)")
    parser.add_argument("--latent-covariance-weight", type=float, default=0.0,
                        help="VICReg covariance penalty (decorrelate latent dims)")
    parser.add_argument("--latent-variance-target", type=float, default=1.0)
    parser.add_argument("--imputation-anchor", action="store_true",
                        help="Stage-1 JEPA: current-imputation loss NOT multiplied by state_weight (independent mask-reconstruction anchor)")
    parser.add_argument("--init-encoder-from", default=None,
                        help="Stage-2 probing: load context/target encoder + predictor from this checkpoint")
    parser.add_argument("--freeze-encoder", action="store_true",
                        help="Stage-2 probing: freeze encoder+predictor; train only heads")
    parser.add_argument("--fund-yoy-input-path", default=None,
                        help="daily GT-input table (build_fund_yoy_inputs): own disclosed YoY as input channels")
    parser.add_argument("--fund-yoy-input-mode", default="own", choices=["own", "own_peer"])
    parser.add_argument("--fund-hidden-target-path", default=None,
                        help="firm-day undisclosed-quarter YoY labels (csv.gz from build_fund_hidden_targets); adds 2 completion channels")
    parser.add_argument("--fund-hidden-min-peers", type=int, default=3,
                        help="supervise only firm-days with >= this many earlier-disclosed industry peers")
    parser.add_argument("--privileged-hidden-teacher", action="store_true")
    parser.add_argument(
        "--downstream-market-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for the date-level absolute market return and "
            "cost-exceedance head."
        ),
    )
    parser.add_argument(
        "--downstream-market-cost-bps",
        type=float,
        default=50.0,
        help="Round-trip cost hurdle used by the market exposure classifier.",
    )
    parser.add_argument(
        "--downstream-transition-loss-weight",
        type=float,
        default=0.0,
        help=(
            "Weight for fit-calibrated broad price, activity, node-state, "
            "and topology transition targets."
        ),
    )
    parser.add_argument(
        "--downstream-transition-pooling",
        choices=["mean", "robust", "robust_projected"],
        default="mean",
    )
    parser.add_argument(
        "--temporal-impact-loss-mix",
        type=float,
        default=0.0,
        help=(
            "Blend fit-calibrated broad/systemic transition weights into "
            "temporal node-state and latent losses; 0 disables and 1 uses "
            "only the impact-weighted reduction."
        ),
    )
    parser.add_argument("--normalize-predictor-output", action="store_true")
    parser.add_argument(
        "--temporal-state-mode",
        choices=[
            "direct",
            "residual_mixed",
            "horizon_hybrid",
            "horizon_residual_heads",
        ],
        default="direct",
    )
    parser.add_argument("--temporal-residual-short-steps", type=int, default=2)
    parser.add_argument(
        "--hybrid-fast-direct",
        action="store_true",
        help="Predict fast one-day features directly in horizon_hybrid mode.",
    )
    parser.add_argument("--pretrain-task", choices=["masked", "temporal"], default="masked")
    parser.add_argument("--temporal-offset", type=int, default=None)
    parser.add_argument("--latent-rollout-steps", type=int, default=1)
    # 2026-08-03: 수급 랭킹 손실(fr_s 실험). 기본 0.0 = 현행과 동일.
    # docs/DESIGN_FLOW_RANK_HEAD_20260803.md
    parser.add_argument("--flow-rank-loss-weight", type=float, default=0.0,
                        help="지정 피처의 횡단면 순위를 직접 맞추는 보조 손실 가중")
    parser.add_argument("--flow-rank-features",
                        default="investor_pension_flow_ratio_1d",
                        help="쉼표 구분. --flow-rank-loss-weight > 0 일 때만 쓰인다")
    parser.add_argument("--rollout-offsets", default="")
    parser.add_argument(
        "--rollout-loss-weights",
        default="",
        help="Optional non-negative comma-separated temporal-loss weights, aligned to --rollout-offsets.",
    )
    parser.add_argument("--event-path", action="append", default=[], help="JSONL market/news events to include as node features; repeatable")
    parser.add_argument("--event-half-life-days", type=float, default=5.0)
    parser.add_argument("--event-lag-days", type=int, default=1)
    parser.add_argument("--event-max-decay-days", type=int, default=60)
    parser.add_argument(
        "--event-coverage-mode",
        choices=["mask_uncovered", "legacy_all_observed"],
        default="mask_uncovered",
    )
    parser.add_argument("--require-event-sensors", action="store_true")
    parser.add_argument("--min-event-coverage", type=float, default=0.50)
    parser.add_argument("--event-edge-top-k", type=int, default=0)
    parser.add_argument("--event-edge-min-weight", type=float, default=0.05)
    parser.add_argument("--event-edge-scale", type=float, default=0.25)
    parser.add_argument("--event-edge-max-themes", type=int, default=96)
    parser.add_argument("--event-edge-min-theme-count", type=int, default=2)
    parser.add_argument("--industry-profile-path", action="append", default=[])
    parser.add_argument("--industry-prefix-length", type=int, default=2)
    parser.add_argument("--industry-edge-scale", type=float, default=0.20)
    parser.add_argument("--require-industry-edges", action="store_true")
    parser.add_argument("--fundamental-path", action="append", default=[], help="Point-in-time fundamental JSONL; repeatable")
    parser.add_argument("--fundamental-lag-days", type=int, default=1)
    parser.add_argument("--require-fundamental-sensors", action="store_true")
    parser.add_argument("--min-fundamental-coverage", type=float, default=0.50)
    parser.add_argument("--investor-cache-dir", default=None)
    parser.add_argument("--investor-flow-lag-days", type=int, default=1)
    parser.add_argument("--require-investor-sensors", action="store_true")
    parser.add_argument("--min-investor-coverage", type=float, default=0.50)
    parser.add_argument(
        "--external-preset",
        choices=["none", "kr_global", "kr_global_rates"],
        default="none",
    )
    parser.add_argument("--external-symbol", action="append", default=[], help="External factor as SYMBOL or SYMBOL:name; repeatable")
    parser.add_argument("--external-node-mode", choices=["features", "nodes", "both"], default="features")
    parser.add_argument("--external-lag-days", type=int, default=1)
    parser.add_argument("--external-cache-dir", default="data/external_cache")
    parser.add_argument("--require-all-external-factors", action="store_true")
    parser.add_argument(
        "--external-etf-panel",
        default=None,
        help=(
            "Frozen cross-source US ETF panel used as causal, input-only graph "
            "nodes at the KRX 15:30 cutoff."
        ),
    )
    parser.add_argument(
        "--risk-free-source",
        choices=["none", "bok_base_rate"],
        default="bok_base_rate",
        help="Effective policy-rate series used for excess-return diagnostics.",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--refresh", action="store_true")
    parser.add_argument("--cache-dir", default="data/cache")
    parser.add_argument("--expected-training-manifest-sha256", default="")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument(
        "--training-manifest-schema-version",
        type=int,
        choices=[1, 2, 3, 4],
        default=3,
    )
    parser.add_argument("--expected-training-edge-manifest-sha256", default="")
    parser.add_argument("--edge-manifest-only", action="store_true")
    parser.add_argument("--reports-dir", default="reports")
    parser.add_argument("--models-dir", default="models")
    parser.add_argument("--skip-return-backtest", action="store_true")
    parser.add_argument(
        "--checkpoint-epochs",
        default="",
        help="Comma-separated milestone epochs to preserve as complete model checkpoints",
    )
    parser.add_argument(
        "--final-refit",
        action="store_true",
        help="Train a node-only deployment checkpoint with no held-out test rows; requires --skip-return-backtest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.final_refit and not args.skip_return_backtest:
        raise ValueError("--final-refit requires --skip-return-backtest")
    if args.temporal_offset is None:
        args.temporal_offset = args.horizon
    if args.latent_rollout_steps < 1:
        raise ValueError("--latent-rollout-steps must be >= 1")
    if args.train_batch_size < 1:
        raise ValueError("--train-batch-size must be >= 1")
    if args.max_train_steps < 0:
        raise ValueError("--max-train-steps must be >= 0")
    if args.downstream_auxiliary_loss_weight < 0.0:
        raise ValueError("--downstream-auxiliary-loss-weight must be non-negative")
    if args.downstream_market_loss_weight < 0.0:
        raise ValueError("--downstream-market-loss-weight must be non-negative")
    if args.downstream_market_cost_bps < 0.0:
        raise ValueError("--downstream-market-cost-bps must be non-negative")
    if args.downstream_transition_loss_weight < 0.0:
        raise ValueError(
            "--downstream-transition-loss-weight must be non-negative"
        )
    if not 0.0 <= args.temporal_impact_loss_mix <= 1.0:
        raise ValueError("--temporal-impact-loss-mix must be between 0 and 1")
    if (
        (
            args.downstream_transition_loss_weight > 0.0
            or args.temporal_impact_loss_mix > 0.0
        )
        and args.pretrain_task != "temporal"
    ):
        raise ValueError(
            "transition and impact losses require --pretrain-task temporal"
        )
    args.downstream_auxiliary_task_weights = [
        float(args.downstream_path_weight),
        float(args.downstream_mfe_weight),
        float(args.downstream_mae_weight),
        float(args.downstream_volatility_weight),
        float(args.downstream_continuation_weight),
    ]
    if any(weight < 0.0 for weight in args.downstream_auxiliary_task_weights):
        raise ValueError("downstream task weights must be non-negative")
    if sum(args.downstream_auxiliary_task_weights) <= 0.0:
        raise ValueError("downstream task weights must contain a positive weight")
    if args.snapshot_workers < 1:
        raise ValueError("--snapshot-workers must be >= 1")
    if not 0.0 <= args.min_event_coverage <= 1.0:
        raise ValueError("--min-event-coverage must be between 0 and 1")
    args.path_horizons_list = sorted(set(parse_int_list(args.path_horizons) + [int(args.horizon)]))
    args.rollout_offsets = parse_int_list(args.rollout_offsets) if args.rollout_offsets else [int(args.temporal_offset)]
    args.rollout_loss_weights = (
        parse_nonnegative_float_list(args.rollout_loss_weights)
        if args.rollout_loss_weights
        else [1.0] * len(args.rollout_offsets)
    )
    args.checkpoint_epochs = parse_int_list(args.checkpoint_epochs) if args.checkpoint_epochs else []
    if len(args.rollout_loss_weights) != len(args.rollout_offsets):
        raise ValueError("--rollout-loss-weights must have one value per --rollout-offsets entry")
    # 헤드 구조는 사전학습 태스크와 무관하게 동일해야 한다. masked(공간 JEPA)로
    # 사전학습한 인코더를 --init-encoder-from 으로 시간축 헤드 학습에 넘기려면
    # 두 단계의 아키텍처가 일치해야 하므로, 여기서는 항상 시간축 공식을 쓴다.
    _head_span = max(int(args.temporal_offset), 1)
    if int(getattr(args, "sequence_window", 0) or 0) > 0 and getattr(args, "global_stock_context", False):
        raise ValueError("--sequence-window does not support --global-stock-context yet")
    args.flow_rank_features_list = [
        name.strip() for name in str(args.flow_rank_features).split(",") if name.strip()
    ]
    if args.flow_rank_loss_weight > 0.0 and not args.flow_rank_features_list:
        raise ValueError("--flow-rank-loss-weight > 0 requires --flow-rank-features")
    args.temporal_head_steps = sorted(
        {
            max(1, int(round(int(offset) * args.latent_rollout_steps / _head_span)))
            for offset in set(args.rollout_offsets) | set(args.path_horizons_list)
        }
    )
    if any(epoch < 1 or epoch >= args.epochs for epoch in args.checkpoint_epochs):
        raise ValueError("--checkpoint-epochs entries must be >= 1 and less than --epochs")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    reports_dir = Path(args.reports_dir)
    models_dir = Path(args.models_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    if args.universe_manifest:
        universe = load_universe_manifest(args.universe_manifest)
    elif args.universe == "krx":
        universe = select_krx_universe_from_listing(args.max_tickers)
    else:
        universe = select_universe(args.max_tickers)
    names = dict(universe)
    print(f"fetching {len(universe)} KRX tickers from {args.universe} universe...", flush=True)
    raw = fetch_krx_ohlcv(
        universe=universe,
        start=args.start,
        end=args.end,
        cache_dir=args.cache_dir,
        refresh=args.refresh,
    )
    raw = filter_history_for_training(
        raw,
        train_end=args.train_end,
        min_train_rows=args.min_train_rows or max(260, args.edge_window + args.min_usable_train_rows),
    )
    if len(raw) < max(args.top_k * 2, 8):
        raise ValueError("too few tickers remain after training-history filter")
    panel = make_ohlcv_panel(raw, names=names)
    event_feature_frames = None
    event_feature_names: list[str] = []
    event_ticker_coverage = None
    event_source_coverage = None
    event_theme_exposure = None
    event_theme_names = []
    fundamental_feature_frames = None
    investor_feature_frames = None
    external_node_feature_frames = None
    external_node_returns = None
    external_node_names = {}
    static_edge_index = None
    static_edge_weight = None
    external_factors = []
    factor_closes = {}
    if args.event_path:
        event_feature_frames = build_event_feature_frames(
            dates=panel.close.index,
            tickers=panel.tickers,
            event_paths=args.event_path,
            half_life_days=args.event_half_life_days,
            lag_days=args.event_lag_days,
            max_decay_days=args.event_max_decay_days,
        )
        event_feature_names = list(event_feature_frames)
        if args.event_coverage_mode == "mask_uncovered" or args.require_event_sensors:
            event_source_coverage = build_event_ticker_coverage(
                dates=panel.close.index,
                tickers=panel.tickers,
                event_paths=args.event_path,
            )
            if args.event_coverage_mode == "mask_uncovered":
                event_ticker_coverage = event_source_coverage
        coverage = (
            float(event_source_coverage.any(axis=0).mean())
            if event_source_coverage is not None
            else 1.0
        )
        print(
            f"event features: paths={len(args.event_path)} features={len(event_feature_frames)} "
            f"coverage_mode={args.event_coverage_mode} "
            f"covered_tickers={len(panel.tickers) if event_source_coverage is None else int(event_source_coverage.any(axis=0).sum())} "
            f"coverage={coverage:.3f}",
            flush=True,
        )
        if args.require_event_sensors and coverage < args.min_event_coverage:
            raise RuntimeError(
                f"event coverage {coverage:.3f} is below required minimum "
                f"{args.min_event_coverage:.3f}"
            )
    elif args.require_event_sensors:
        raise RuntimeError("--require-event-sensors requires at least one --event-path")
    if args.event_path and args.event_edge_top_k > 0:
        event_theme_exposure, event_theme_names = build_event_theme_exposure(
            dates=panel.close.index,
            tickers=panel.tickers,
            event_paths=args.event_path,
            half_life_days=args.event_half_life_days,
            lag_days=args.event_lag_days,
            max_decay_days=args.event_max_decay_days,
            max_themes=args.event_edge_max_themes,
            min_theme_count=args.event_edge_min_theme_count,
        )
        print(
            f"event edges: themes={len(event_theme_names)} top_k={args.event_edge_top_k} "
            f"scale={args.event_edge_scale}",
            flush=True,
        )
    if args.fundamental_path:
        fundamental_feature_frames = build_fundamental_feature_frames(
            dates=panel.close.index,
            tickers=panel.tickers,
            observations=load_fundamental_observations(args.fundamental_path),
            lag_days=args.fundamental_lag_days,
        )
        coverage = fundamental_coverage(
            fundamental_feature_frames,
            eligible_mask=panel.price_observed,
        )
        print(
            f"fundamental sensors: paths={len(args.fundamental_path)} "
            f"features={len(fundamental_feature_frames)} coverage={coverage:.3f}",
            flush=True,
        )
        if args.require_fundamental_sensors and coverage < args.min_fundamental_coverage:
            raise RuntimeError(
                f"fundamental coverage {coverage:.3f} is below required minimum {args.min_fundamental_coverage:.3f}"
            )
    elif args.require_fundamental_sensors:
        raise RuntimeError("--require-fundamental-sensors requires at least one --fundamental-path")
    if args.investor_cache_dir:
        investor_flow_frames = load_investor_flow_frames(
            cache_dir=args.investor_cache_dir,
            dates=panel.close.index,
            tickers=panel.tickers,
        )
        observed_close = panel.close.where(panel.price_observed)
        observed_volume = panel.volume.where(panel.price_observed)
        investor_feature_frames = build_investor_feature_frames(
            investor_flow_frames,
            traded_value=observed_close * observed_volume,
            lag_days=args.investor_flow_lag_days,
        )
        investor_coverage = investor_feature_coverage(
            investor_feature_frames,
            eligible_mask=panel.price_observed & panel.volume.gt(0.0),
        )
        print(
            f"investor sensors: cache={args.investor_cache_dir} "
            f"features={len(investor_feature_frames)} coverage={investor_coverage:.3f} "
            f"lag_days={args.investor_flow_lag_days}",
            flush=True,
        )
        if args.require_investor_sensors and investor_coverage < args.min_investor_coverage:
            raise RuntimeError(
                f"investor coverage {investor_coverage:.3f} is below required minimum "
                f"{args.min_investor_coverage:.3f}"
            )
    elif args.require_investor_sensors:
        raise RuntimeError("--require-investor-sensors requires --investor-cache-dir")
    if args.industry_profile_path:
        industry_codes = load_industry_codes(args.industry_profile_path)
        static_edge_index, static_edge_weight, industry_stats = build_industry_edge_arrays(
            panel.tickers,
            industry_codes,
            prefix_length=args.industry_prefix_length,
            scale=args.industry_edge_scale,
        )
        print(
            f"industry edges: paths={len(args.industry_profile_path)} "
            f"matched={industry_stats['matched_tickers']} groups={industry_stats['industry_groups']} "
            f"edges={industry_stats['edges']} scale={args.industry_edge_scale}",
            flush=True,
        )
        if args.require_industry_edges and industry_stats["edges"] == 0:
            raise RuntimeError("industry profiles produced no static industry edges")
    elif args.require_industry_edges:
        raise RuntimeError("--require-industry-edges requires --industry-profile-path")
    external_factors = resolve_external_factors(args.external_preset, args.external_symbol)
    if external_factors:
        factor_closes = fetch_external_factor_closes(
            external_factors,
            start=args.start,
            end=args.end,
            cache_dir=args.external_cache_dir,
            refresh=args.refresh,
        )
        missing_factor_names = [factor.name for factor in external_factors if factor.name not in factor_closes]
        if args.require_all_external_factors and missing_factor_names:
            raise RuntimeError(
                "required external factors were unavailable: " + ", ".join(missing_factor_names)
            )
        external_feature_frames = {}
        if args.external_node_mode in {"features", "both"}:
            external_feature_frames = build_external_feature_frames(
                dates=panel.close.index,
                tickers=panel.tickers,
                factor_closes=factor_closes,
                lag_days=args.external_lag_days,
            )
            if external_feature_frames:
                event_feature_frames = dict(event_feature_frames or {})
                event_feature_frames.update(external_feature_frames)
        if args.external_node_mode in {"nodes", "both"}:
            external_node_feature_frames, external_node_returns, external_node_names = build_external_node_feature_frames(
                dates=panel.close.index,
                factor_closes=factor_closes,
                lag_days=args.external_lag_days,
            )
        print(
            f"external factors: requested={len(external_factors)} loaded={len(factor_closes)} "
            f"features={len(external_feature_frames)} "
            f"node_features={len(external_node_feature_frames or {})} "
            f"nodes={0 if external_node_returns is None else external_node_returns.shape[1]} "
            f"mode={args.external_node_mode} lag_days={args.external_lag_days}",
            flush=True,
        )
    if args.external_etf_panel:
        etf_inputs = load_external_etf_node_inputs(
            args.external_etf_panel,
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
        (reports_dir / "external_etf_node_audit.json").write_text(
            json.dumps(etf_inputs.audit, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            f"US ETF nodes: nodes={etf_inputs.audit['nodes']} "
            f"features={len(etf_inputs.feature_frames)} "
            f"fresh_events={etf_inputs.audit['source_events_visible']} "
            f"holiday_bundles={etf_inputs.audit['bundled_holiday_events']} "
            "cutoff=15:30 Asia/Seoul mode=input_only",
            flush=True,
        )
    features = build_feature_panel(
        panel,
        horizon=args.horizon,
        train_end=args.train_end,
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
        path_horizons=args.path_horizons_list,
    )
    if getattr(args, "fund_yoy_input_path", None):
        from stock_v2.fund_yoy_inputs import augment_panel_with_fund_yoy
        features = augment_panel_with_fund_yoy(
            features, args.fund_yoy_input_path, args.fund_yoy_input_mode, args.train_end)
    if getattr(args, "ownership_edge_path", None):
        from stock_v2.ownership_edges import attach_ownership_edges
        features = attach_ownership_edges(features, args.ownership_edge_path)
    if getattr(args, "earnings_features", False):
        from stock_v2.earnings_features import augment_panel_with_earnings
        features = augment_panel_with_earnings(
            features, args.fundamental_path[0] if isinstance(args.fundamental_path, list) else args.fundamental_path,
            horizon=args.horizon, train_end=args.train_end)
    if int(getattr(args, "return_lag_features", 0) or 0) > 0:
        from stock_v2.earnings_features import augment_panel_with_return_lags
        features = augment_panel_with_return_lags(
            features, n_lags=int(args.return_lag_features), train_end=args.train_end)
    print(
        f"feature panel: dates={len(features.dates)} tickers={len(features.tickers)} "
        f"nodes={features.node_count} features={len(features.feature_names)}",
        flush=True,
    )
    train_data_manifest = build_training_data_manifest(
        features,
        args.train_end,
        schema_version=args.training_manifest_schema_version,
    )
    print(
        f"training data manifest: sha256={train_data_manifest['sha256'][:16]} "
        f"rows={len(train_data_manifest['dates'])}",
        flush=True,
    )
    validate_expected_training_manifest(
        train_data_manifest,
        args.expected_training_manifest_sha256,
    )
    (reports_dir / "training_data_manifest.json").write_text(
        json.dumps(train_data_manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    training_data_diagnostics = build_training_data_diagnostics(
        features,
        args.train_end,
    )
    (reports_dir / "training_data_diagnostics.json").write_text(
        json.dumps(training_data_diagnostics, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if args.manifest_only and not args.edge_manifest_only:
        print(json.dumps(train_data_manifest, ensure_ascii=False, indent=2), flush=True)
        return

    train_indices = date_indices(features.dates, end=args.train_end)
    test_indices = date_indices(
        features.dates,
        start=(pd.Timestamp(args.train_end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    test_indices = test_indices[test_indices >= args.edge_window]
    usable_train_indices = train_indices[train_indices >= args.edge_window]
    if len(usable_train_indices) < args.min_usable_train_rows:
        raise ValueError(
            "not enough usable train rows after feature cleaning: "
            f"usable={len(usable_train_indices)}, required={args.min_usable_train_rows}. "
            "Increase --min-train-rows or reduce universe size."
        )
    if args.final_refit and len(test_indices) != 0:
        raise ValueError(
            "--final-refit requires --train-end to cover the final valid feature row; "
            f"found {len(test_indices)} held-out rows"
        )
    if not args.final_refit and len(test_indices) < args.horizon * 4:
        raise ValueError("not enough test rows after train split")

    if args.edge_manifest_only:
        edge_steps = usable_train_indices
        if args.pretrain_task == "temporal":
            edge_steps = temporal_training_indices(
                train_indices,
                edge_window=args.edge_window,
                max_rollout_offset=max(args.rollout_offsets),
                total_steps=len(features.dates),
            )
        if len(edge_steps) == 0:
            raise ValueError("no training contexts remain for the edge manifest")
        build_training_edge_cache(features, edge_steps, args)
        return

    transition_targets = None
    if (
        args.downstream_transition_loss_weight > 0.0
        or args.temporal_impact_loss_mix > 0.0
    ):
        transition_steps = temporal_training_indices(
            train_indices,
            edge_window=args.edge_window,
            max_rollout_offset=max(args.rollout_offsets),
            total_steps=len(features.dates),
        )
        transition_targets = build_market_transition_auxiliary_targets(
            features,
            transition_steps,
            args.rollout_offsets,
        )
        transition_contract = transition_targets.contract_dict()
        transition_contract["temporal_impact_loss"] = {
            "mix": float(args.temporal_impact_loss_mix),
            "weighted_terms": ["temporal_latent", "temporal_node_state"],
            "normalization": "within_batch_weighted_mean",
            "calibration_scope": "fit_rows_only",
        }
        (reports_dir / "market_transition_auxiliary_contract.json").write_text(
            json.dumps(transition_contract, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(
            "market transition auxiliary: "
            + ", ".join(
                f"h{horizon}=systemic_rate_"
                f"{transition_targets.fit_systemic_event_rate[int(horizon)]:.3f}"
                f"/selloff_rate_"
                f"{transition_targets.fit_broad_selloff_rate[int(horizon)]:.3f}"
                for horizon in args.rollout_offsets
            ),
            flush=True,
        )

    device = torch.device(args.device)
    state_feature_weights = build_state_feature_weights(
        features.feature_names,
        args.state_feature_weight,
    )
    temporal_state_feature_weights = build_temporal_state_feature_weights(
        features.feature_names,
        state_feature_weights,
        args.temporal_exclude_feature_prefix,
    )
    if args.state_feature_weight:
        print(
            "state feature weights: " + ", ".join(args.state_feature_weight),
            flush=True,
        )
    if args.temporal_exclude_feature_prefix:
        print(
            "temporal excluded feature prefixes: "
            + ", ".join(args.temporal_exclude_feature_prefix),
            flush=True,
        )
    model = StockGraphJEPA(
        num_features=len(features.feature_names),
        hidden_dim=args.hidden_dim,
        num_layers=args.layers,
        ema_decay=args.ema_decay,
        latent_loss_weight=args.latent_loss_weight,
        state_loss_weight=args.state_loss_weight,
        current_imputation_loss_weight=args.current_imputation_loss_weight,
        imputation_standalone=bool(getattr(args, "imputation_anchor", False)),
        sequence_window=int(getattr(args, "sequence_window", 0) or 0),
        sequence_layers=int(getattr(args, "sequence_layers", 2) or 2),
        sequence_heads=int(getattr(args, "sequence_heads", 8) or 8),
        sequence_residual=bool(getattr(args, "sequence_residual", False)),
        latent_variance_weight=args.latent_variance_weight,
        latent_covariance_weight=args.latent_covariance_weight,
        latent_variance_target=args.latent_variance_target,
        hidden_completion_loss_weight=args.hidden_completion_weight,
        hidden_completion_width=(len(HIDDEN_COMPLETION_CHANNELS) + 2) if args.fund_hidden_target_path else None,
        temporal_state_mode=args.temporal_state_mode,
        feature_names=features.feature_names,
        temporal_residual_short_steps=args.temporal_residual_short_steps,
        temporal_head_steps=args.temporal_head_steps,
        state_feature_weights=state_feature_weights,
        temporal_state_feature_weights=temporal_state_feature_weights,
        temporal_state_context_skip=args.temporal_state_context_skip,
        temporal_head_input=args.temporal_head_input,
        hybrid_fast_direct=args.hybrid_fast_direct,
        return_correlation_loss_weight=args.return_correlation_loss_weight,
        entry_path_correlation_loss_weight=(
            args.entry_path_correlation_loss_weight
        ),
        flow_rank_loss_weight=args.flow_rank_loss_weight,
        flow_rank_features=args.flow_rank_features_list,
        feature_means=features.train_mean,
        feature_stds=features.train_std,
        normalize_predictor_output=args.normalize_predictor_output,
        graph_neighbor_scale=args.graph_neighbor_scale,
        temporal_graph_neighbor_scale=args.temporal_graph_neighbor_scale,
        temporal_stock_edge_scale=args.temporal_stock_edge_scale,
        global_stock_context=args.global_stock_context,
        downstream_plan_loss_weight=float(
            getattr(args, "downstream_plan_loss_weight", 0.0)
        ),
        plan_temperature=float(getattr(args, "plan_temperature", 0.01)),
        plan_buy_sell=bool(getattr(args, "plan_buy_sell", False)),
        plan_permute_seed=int(getattr(args, "plan_permute_seed", 0) or 0),
        downstream_auxiliary_loss_weight=(
            args.downstream_auxiliary_loss_weight
        ),
        downstream_auxiliary_task_weights=(
            args.downstream_auxiliary_task_weights
        ),
        downstream_market_loss_weight=args.downstream_market_loss_weight,
        downstream_market_cost_bps=args.downstream_market_cost_bps,
        downstream_transition_loss_weight=(
            args.downstream_transition_loss_weight
        ),
        downstream_transition_pooling=args.downstream_transition_pooling,
        temporal_impact_loss_mix=args.temporal_impact_loss_mix,
    ).to(device)

    if getattr(args, "init_encoder_from", None):
        _ck = torch.load(args.init_encoder_from, map_location=device, weights_only=False)
        _sd = _ck["model_state"] if isinstance(_ck, dict) and "model_state" in _ck else _ck
        _keep = {k: v for k, v in _sd.items()
                 if k.split(".")[0] in ("context_encoder", "target_encoder", "predictor")}
        _missing, _unexp = model.load_state_dict(_keep, strict=False)
        print(f"init-encoder-from: loaded {len(_keep)} tensors "
              f"(enc/pred), missing={len(_missing)} unexpected={len(_unexp)}", flush=True)
    if getattr(args, "freeze_encoder", False):
        _frozen = 0
        for _n, _pm in model.named_parameters():
            if _n.split(".")[0] in ("context_encoder", "target_encoder", "predictor"):
                _pm.requires_grad_(False); _frozen += _pm.numel()
        print(f"freeze-encoder: froze {_frozen:,} params (enc/pred); heads train", flush=True)

    def checkpoint_payload(history_rows: List[Dict[str, float]], checkpoint_epoch: int) -> Dict[str, object]:
        return {
            "model_state": model.state_dict(),
            "feature_names": features.feature_names,
            "tickers": features.tickers,
            "node_tickers": features.node_tickers,
            "stock_node_count": features.stock_node_count,
            "names": features.names,
            "train_mean": features.train_mean,
            "train_std": features.train_std,
            "temporal_state_feature_weights": temporal_state_feature_weights,
            "train_data_manifest": train_data_manifest,
            "train_edge_manifest": json.loads(
                (reports_dir / "training_edge_manifest.json").read_text(
                    encoding="utf-8"
                )
            ),
            "loaded_external_factor_names": list(factor_closes),
            "args": vars(args),
            "history": list(history_rows),
            "checkpoint_epoch": int(checkpoint_epoch),
            "market_transition_auxiliary_contract": (
                None
                if transition_targets is None
                else transition_targets.contract_dict()
            ),
        }

    checkpoint_epochs = set(args.checkpoint_epochs)

    def save_milestone(epoch: int, history_rows: List[Dict[str, float]]) -> None:
        if epoch not in checkpoint_epochs:
            return
        milestone_dir = models_dir / f"epoch_{epoch:03d}"
        milestone_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            checkpoint_payload(history_rows, epoch),
            milestone_dir / "graph_jepa_real.pt",
        )
        print(f"saved milestone checkpoint: epoch={epoch}", flush=True)

    history = train_jepa(
        model,
        features,
        train_indices,
        args,
        device,
        transition_targets=transition_targets,
        checkpoint_callback=save_milestone,
    )
    trained_epochs = int(history[-1]["epoch"]) if history else 0
    torch.save(
        checkpoint_payload(history, trained_epochs),
        models_dir / "graph_jepa_real.pt",
    )
    pd.DataFrame(history).to_csv(reports_dir / "pretrain_history.csv", index=False)
    if args.skip_return_backtest:
        metadata = {
            "config": vars(args),
            "tickers": features.tickers,
            "node_tickers": features.node_tickers,
            "stock_node_count": features.stock_node_count,
            "feature_names": features.feature_names,
            "train_rows": int(len(train_indices)),
            "test_rows": int(len(test_indices)),
            "train_data_manifest": train_data_manifest,
            "node_only": True,
        }
        (reports_dir / "node_training_metadata.json").write_text(
            json.dumps(metadata, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print("skipping return backtest; checkpoint is ready for node-state evaluation", flush=True)
        return

    risk_free_returns_by_horizon = None
    if args.risk_free_source == "bok_base_rate":
        bok_rate = factor_closes.get("bok_base_rate")
        if bok_rate is None:
            bok_rate_closes = fetch_external_factor_closes(
                [POLICY_RATE_FACTORS[0]],
                start=args.start,
                end=args.end,
                cache_dir=args.external_cache_dir,
                refresh=args.refresh,
            )
            bok_rate = bok_rate_closes.get("bok_base_rate")
        if bok_rate is None:
            raise RuntimeError("effective BOK base-rate history is required for return evaluation")
        risk_free_returns_by_horizon = build_risk_free_period_returns(
            features.dates,
            bok_rate,
            args.path_horizons_list,
        )
        observed_risk_free = int(
            np.isfinite(risk_free_returns_by_horizon[int(args.horizon)]).sum()
        )
        print(
            f"risk-free diagnostics: source=bok_base_rate convention=ACT/365-effective "
            f"horizon={args.horizon} observed={observed_risk_free}",
            flush=True,
        )

    print("encoding graph states...", flush=True)
    embeddings = encode_all(model, features, args, device)
    graph_design = np.concatenate([embeddings, features.features], axis=2)
    jepa_only_design = embeddings
    raw_design = features.features

    return_model = fit_return_model(graph_design, features.target_returns, train_indices)
    jepa_only_model = fit_return_model(jepa_only_design, features.target_returns, train_indices)
    raw_model = fit_return_model(raw_design, features.target_returns, train_indices)

    scores = predict_scores(return_model, graph_design)
    jepa_only_scores = predict_scores(jepa_only_model, jepa_only_design)
    raw_scores = predict_scores(raw_model, raw_design)

    print("training path-aware rollout heads...", flush=True)
    path_models, path_predictions = build_path_predictions(model, features, train_indices, args, device)
    path_scores, path_exit_horizons, path_peak_returns = path_aware_scores(features, path_predictions, args)

    trades, metrics = run_rank_backtest(
        dates=features.dates,
        tickers=features.tickers,
        scores=scores,
        target_returns=features.target_returns,
        test_indices=test_indices,
        top_k=args.top_k,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        strategy_name="graph_jepa_ridge",
        risk_free_returns=(
            risk_free_returns_by_horizon[int(args.horizon)]
            if risk_free_returns_by_horizon is not None
            else None
        ),
    )
    jepa_only_trades, jepa_only_metrics = run_rank_backtest(
        dates=features.dates,
        tickers=features.tickers,
        scores=jepa_only_scores,
        target_returns=features.target_returns,
        test_indices=test_indices,
        top_k=args.top_k,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        strategy_name="jepa_only_ridge",
        risk_free_returns=(
            risk_free_returns_by_horizon[int(args.horizon)]
            if risk_free_returns_by_horizon is not None
            else None
        ),
    )
    raw_trades, raw_metrics = run_rank_backtest(
        dates=features.dates,
        tickers=features.tickers,
        scores=raw_scores,
        target_returns=features.target_returns,
        test_indices=test_indices,
        top_k=args.top_k,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        strategy_name="raw_ridge",
        risk_free_returns=(
            risk_free_returns_by_horizon[int(args.horizon)]
            if risk_free_returns_by_horizon is not None
            else None
        ),
    )
    mom_trades, mom_metrics = run_rank_backtest(
        dates=features.dates,
        tickers=features.tickers,
        scores=momentum_scores(features),
        target_returns=features.target_returns,
        test_indices=test_indices,
        top_k=args.top_k,
        horizon=args.horizon,
        cost_bps=args.cost_bps,
        strategy_name="momentum_20d",
        risk_free_returns=(
            risk_free_returns_by_horizon[int(args.horizon)]
            if risk_free_returns_by_horizon is not None
            else None
        ),
    )
    path_trades, path_metrics = run_path_rank_backtest(
        dates=features.dates,
        tickers=features.tickers,
        scores=path_scores,
        exit_horizons=path_exit_horizons,
        target_returns_by_horizon=features.target_return_paths,
        test_indices=test_indices,
        top_k=args.top_k,
        rebalance_stride=max(args.path_horizons_list),
        cost_bps=args.cost_bps,
        strategy_name="path_aware_rollout_ridge",
        risk_free_returns_by_horizon=risk_free_returns_by_horizon,
    )

    combined_metrics = {
        **metrics,
        **jepa_only_metrics,
        **raw_metrics,
        **mom_metrics,
        **path_metrics,
        "config": vars(args),
        "tickers": features.tickers,
        "node_tickers": features.node_tickers,
        "stock_node_count": features.stock_node_count,
        "feature_names": features.feature_names,
        "train_data_manifest": train_data_manifest,
        "train_rows": int(len(train_indices)),
        "test_rows": int(len(test_indices)),
    }
    trades.to_csv(reports_dir / "graph_jepa_trades.csv", index=False)
    jepa_only_trades.to_csv(reports_dir / "jepa_only_trades.csv", index=False)
    raw_trades.to_csv(reports_dir / "raw_ridge_trades.csv", index=False)
    mom_trades.to_csv(reports_dir / "momentum_trades.csv", index=False)
    path_trades.to_csv(reports_dir / "path_aware_rollout_trades.csv", index=False)
    (reports_dir / "real_backtest_metrics.json").write_text(
        json.dumps(combined_metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    with (models_dir / "return_models.pkl").open("wb") as file:
        pickle.dump(
            {
                "graph_jepa_ridge": return_model,
                "jepa_only_ridge": jepa_only_model,
                "raw_ridge": raw_model,
                "feature_names": features.feature_names,
                "tickers": features.tickers,
                "node_tickers": features.node_tickers,
                "stock_node_count": features.stock_node_count,
                "names": features.names,
                "train_mean": features.train_mean,
                "train_std": features.train_std,
                "train_data_manifest": train_data_manifest,
                "loaded_external_factor_names": list(factor_closes),
                "args": vars(args),
                "history": history,
            },
            file,
        )
    write_summary(
        reports_dir / "real_backtest_summary.md",
        args=args,
        features=features,
        history=history,
        metrics=metrics,
        raw_metrics=raw_metrics,
        jepa_only_metrics=jepa_only_metrics,
        momentum_metrics=mom_metrics,
        trades={
            "graph_jepa_ridge": trades,
            "jepa_only_ridge": jepa_only_trades,
            "raw_ridge": raw_trades,
            "momentum_20d": mom_trades,
        },
    )

    print("metrics:", json.dumps({k: v for k, v in combined_metrics.items() if isinstance(v, dict)}, indent=2), flush=True)
    print(f"wrote {reports_dir / 'real_backtest_summary.md'}", flush=True)


if __name__ == "__main__":
    main()
