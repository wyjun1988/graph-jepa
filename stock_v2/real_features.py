from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from stock_v2.ownership_edges import build_ownership_edge_tensor

from stock_v2.external_factors import POLICY_RATE_FACTOR_NAMES
from stock_v2.graph_jepa import GraphBatch, make_feature_mask, make_structured_feature_mask
from stock_v2.market_data import OhlcvPanel


EDGE_WEIGHT_QUANTIZATION = 1e-4


@dataclass
class FeaturePanel:
    dates: pd.DatetimeIndex
    tickers: List[str]
    names: Dict[str, str]
    feature_names: List[str]
    features: np.ndarray
    raw_features: np.ndarray
    available_mask: np.ndarray
    returns_1d: np.ndarray
    target_returns: np.ndarray
    target_return_paths: Dict[int, np.ndarray]
    open: np.ndarray
    close: np.ndarray
    train_mean: np.ndarray
    train_std: np.ndarray
    event_theme_exposure: np.ndarray | None = None
    event_theme_names: List[str] | None = None
    static_edge_index: np.ndarray | None = None
    static_edge_weight: np.ndarray | None = None
    node_tickers: List[str] | None = None
    node_names: Dict[str, str] | None = None
    stock_node_count: int | None = None
    execution_close: np.ndarray | None = None

    def feature_index(self, name: str) -> int:
        return self.feature_names.index(name)

    @property
    def node_count(self) -> int:
        return int(self.features.shape[1])

    @property
    def tradable_count(self) -> int:
        return int(self.stock_node_count or len(self.tickers))


def _zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    mean = frame.rolling(window).mean()
    std = frame.rolling(window).std()
    return (frame - mean) / std.replace(0.0, np.nan)


def build_feature_panel(
    panel: OhlcvPanel,
    horizon: int = 5,
    train_end: str = "2023-12-31",
    require_targets: bool = True,
    feature_names: List[str] | None = None,
    event_feature_frames: Dict[str, pd.DataFrame] | None = None,
    event_feature_names: List[str] | None = None,
    event_ticker_coverage: pd.DataFrame | None = None,
    fundamental_feature_frames: Dict[str, pd.DataFrame] | None = None,
    investor_feature_frames: Dict[str, pd.DataFrame] | None = None,
    external_node_feature_frames: Dict[str, pd.DataFrame] | None = None,
    external_node_returns: pd.DataFrame | None = None,
    external_node_names: Dict[str, str] | None = None,
    event_theme_exposure: np.ndarray | None = None,
    event_theme_names: List[str] | None = None,
    static_edge_index: np.ndarray | None = None,
    static_edge_weight: np.ndarray | None = None,
    path_horizons: List[int] | None = None,
    warmup_rows: int = 80,
    min_valid_targets: int = 4,
) -> FeaturePanel:
    """Build normalized node-state tensors from OHLCV.

    Features only use information available at the signal date. Targets use next
    day's open as entry and `horizon` trading-day close as exit.
    """

    close = panel.close
    execution_close = panel.execution_close.reindex(
        index=panel.close.index,
        columns=panel.close.columns,
    )
    open_ = panel.open
    high = panel.high
    low = panel.low
    price_observed = panel.price_observed.reindex(index=close.index, columns=close.columns).fillna(False).astype(bool)
    close_observed = close.where(price_observed)
    open_observed = open_.where(price_observed)
    high_observed = high.where(price_observed)
    low_observed = low.where(price_observed)
    volume = panel.volume.where(price_observed).replace(0.0, np.nan)

    ret1 = close_observed.pct_change(1, fill_method=None)
    ret2 = close_observed.pct_change(2, fill_method=None)
    ret3 = close_observed.pct_change(3, fill_method=None)
    ret5 = close_observed.pct_change(5, fill_method=None)
    ret10 = close_observed.pct_change(10, fill_method=None)
    ret20 = close_observed.pct_change(20, fill_method=None)
    ret60 = close_observed.pct_change(60, fill_method=None)
    ret120 = close_observed.pct_change(120, fill_method=None)
    ma5_gap = close_observed / close_observed.rolling(5).mean() - 1.0
    ma10_gap = close_observed / close_observed.rolling(10).mean() - 1.0
    ma20_gap = close_observed / close_observed.rolling(20).mean() - 1.0
    ma60_gap = close_observed / close_observed.rolling(60).mean() - 1.0
    ma120_gap = close_observed / close_observed.rolling(120).mean() - 1.0
    vol5 = ret1.rolling(5).std()
    vol10 = ret1.rolling(10).std()
    vol20 = ret1.rolling(20).std()
    vol60 = ret1.rolling(60).std()
    downside_vol20 = ret1.where(ret1 < 0.0, 0.0).rolling(20).std()
    vol_ratio = vol20 / vol60.replace(0.0, np.nan)
    traded_value = volume * close
    volume_z20 = _zscore(np.log1p(volume), 20)
    volume_z60 = _zscore(np.log1p(volume), 60)
    value_z20 = _zscore(np.log1p(traded_value), 20)
    value_z60 = _zscore(np.log1p(traded_value), 60)
    value_ma20_log = np.log1p(traded_value.rolling(20).mean())
    amihud_20 = (ret1.abs() / traded_value.replace(0.0, np.nan)).rolling(20).mean()
    range_pct = (high_observed - low_observed) / close_observed
    range_z20 = _zscore(range_pct, 20)
    gap_open = open_observed / close_observed.shift(1) - 1.0
    intraday_return = close_observed / open_observed - 1.0
    drawdown20 = close_observed / close_observed.rolling(20).max() - 1.0
    breakout20 = close_observed / close_observed.rolling(20).min() - 1.0
    drawdown120 = close_observed / close_observed.rolling(120).max() - 1.0
    breakout120 = close_observed / close_observed.rolling(120).min() - 1.0
    high120 = close_observed.rolling(120).max()
    low120 = close_observed.rolling(120).min()
    range_position120 = (close_observed - low120) / (high120 - low120).replace(0.0, np.nan)
    market_ret = pd.DataFrame(
        np.repeat(ret1.mean(axis=1).to_numpy()[:, None], len(panel.tickers), axis=1),
        index=close.index,
        columns=close.columns,
    )
    market_ret5 = pd.DataFrame(
        np.repeat(ret5.mean(axis=1).to_numpy()[:, None], len(panel.tickers), axis=1),
        index=close.index,
        columns=close.columns,
    )
    relative_return20 = ret20.sub(ret20.mean(axis=1), axis=0)
    cs_rank_return20 = ret20.rank(axis=1, pct=True) - 0.5
    cs_rank_value20 = value_z20.rank(axis=1, pct=True) - 0.5
    market_series = ret1.mean(axis=1)
    market_var60 = market_series.rolling(60).var()
    beta60 = ret1.rolling(60).cov(market_series).div(market_var60, axis=0)
    corr_market60 = ret1.rolling(60).corr(market_series)

    all_feature_frames: Dict[str, pd.DataFrame] = {
        "return_1d": ret1,
        "return_2d": ret2,
        "return_3d": ret3,
        "return_5d": ret5,
        "return_10d": ret10,
        "return_20d": ret20,
        "return_60d": ret60,
        "return_120d": ret120,
        "ma5_gap": ma5_gap,
        "ma10_gap": ma10_gap,
        "ma20_gap": ma20_gap,
        "ma60_gap": ma60_gap,
        "ma120_gap": ma120_gap,
        "volatility_5d": vol5,
        "volatility_10d": vol10,
        "volatility_20d": vol20,
        "volatility_60d": vol60,
        "downside_volatility_20d": downside_vol20,
        "volatility_ratio_20_60": vol_ratio,
        "volume_z20": volume_z20,
        "volume_z60": volume_z60,
        "value_z20": value_z20,
        "value_z60": value_z60,
        "value_ma20_log": value_ma20_log,
        "amihud_20d": amihud_20,
        "range_pct": range_pct,
        "range_z20": range_z20,
        "gap_open": gap_open,
        "intraday_return": intraday_return,
        "drawdown_20d": drawdown20,
        "breakout_20d": breakout20,
        "drawdown_120d": drawdown120,
        "breakout_120d": breakout120,
        "range_position_120d": range_position120,
        "market_return_1d": market_ret,
        "market_return_5d": market_ret5,
        "relative_return_20d": relative_return20,
        "cs_rank_return_20d": cs_rank_return20,
        "cs_rank_value_20d": cs_rank_value20,
        "market_beta_60d": beta60,
        "market_corr_60d": corr_market60,
    }
    if event_feature_frames:
        for event_name, event_frame in event_feature_frames.items():
            all_feature_frames[event_name] = event_frame.reindex(index=close.index, columns=close.columns).fillna(0.0)
    if fundamental_feature_frames:
        for feature_name, feature_frame in fundamental_feature_frames.items():
            all_feature_frames[feature_name] = feature_frame.reindex(index=close.index, columns=close.columns)
    if investor_feature_frames:
        for feature_name, feature_frame in investor_feature_frames.items():
            all_feature_frames[feature_name] = feature_frame.reindex(index=close.index, columns=close.columns)
    if external_node_feature_frames:
        for feature_name in external_node_feature_frames:
            if feature_name not in all_feature_frames:
                all_feature_frames[feature_name] = pd.DataFrame(np.nan, index=close.index, columns=close.columns)
    if feature_names is not None:
        missing = [name for name in feature_names if name not in all_feature_frames]
        if missing:
            raise ValueError(f"unknown feature names requested: {missing}")
        feature_frames = {name: all_feature_frames[name] for name in feature_names}
    else:
        feature_frames = all_feature_frames
    feature_names = list(feature_frames)
    stock_raw = np.stack([feature_frames[name].to_numpy(dtype=np.float32) for name in feature_names], axis=-1)
    stock_raw = np.where(np.isfinite(stock_raw), stock_raw, np.nan).astype(np.float32)
    if event_ticker_coverage is not None:
        covered = event_ticker_coverage.reindex(
            index=close.index,
            columns=close.columns,
        ).fillna(False).to_numpy(dtype=bool)
        masked_event_names = set(event_feature_names or (event_feature_frames or {}).keys())
        for feature_index, name in enumerate(feature_names):
            if name in masked_event_names:
                stock_raw[:, :, feature_index] = np.where(
                    covered,
                    stock_raw[:, :, feature_index],
                    np.nan,
                )
    stock_available_mask = np.isfinite(stock_raw) & price_observed.to_numpy(dtype=bool)[:, :, None]

    external_node_ids: list[str] = []
    external_raw = None
    if external_node_feature_frames:
        seen_nodes: list[str] = []
        for frame in external_node_feature_frames.values():
            for column in frame.columns:
                column_name = str(column)
                if column_name not in seen_nodes:
                    seen_nodes.append(column_name)
        if external_node_returns is not None:
            for column in external_node_returns.columns:
                column_name = str(column)
                if column_name not in seen_nodes:
                    seen_nodes.append(column_name)
        external_node_ids = seen_nodes
        if external_node_ids:
            external_arrays = []
            for feature_name in feature_names:
                frame = external_node_feature_frames.get(feature_name)
                if frame is None:
                    values = np.full((len(close.index), len(external_node_ids)), np.nan, dtype=np.float32)
                else:
                    values = frame.reindex(index=close.index, columns=external_node_ids).to_numpy(dtype=np.float32)
                external_arrays.append(values)
            external_raw = np.stack(external_arrays, axis=-1)
            external_raw = np.where(np.isfinite(external_raw), external_raw, np.nan).astype(np.float32)

    raw = stock_raw if external_raw is None else np.concatenate([stock_raw, external_raw], axis=1)
    available_mask = (
        stock_available_mask
        if external_raw is None
        else np.concatenate([stock_available_mask, np.isfinite(external_raw)], axis=1)
    ).astype(np.float32)

    if path_horizons is None:
        path_horizons = [horizon]
    path_horizons = sorted({int(h) for h in path_horizons if int(h) >= 1} | {int(horizon)})
    entry = open_observed.shift(-1)
    target_frames = {
        h: (close_observed.shift(-h) / entry - 1.0).where(
            price_observed & price_observed.shift(-1).eq(True) & price_observed.shift(-h).eq(True)
        )
        for h in path_horizons
    }
    stock_target = target_frames[int(horizon)].to_numpy(dtype=np.float32)
    stock_target_paths_raw = {
        h: frame.to_numpy(dtype=np.float32)
        for h, frame in target_frames.items()
    }
    if external_node_ids:
        target = np.concatenate(
            [
                stock_target,
                np.full((stock_target.shape[0], len(external_node_ids)), np.nan, dtype=np.float32),
            ],
            axis=1,
        )
        target_paths_raw = {
            h: np.concatenate(
                [
                    arr,
                    np.full((arr.shape[0], len(external_node_ids)), np.nan, dtype=np.float32),
                ],
                axis=1,
            )
            for h, arr in stock_target_paths_raw.items()
        }
    else:
        target = stock_target
        target_paths_raw = stock_target_paths_raw

    train_cutoff = pd.Timestamp(train_end)
    train_rows = close.index <= train_cutoff
    if train_rows.sum() < 80:
        raise ValueError(f"not enough training rows before {train_end}")

    train_values = raw[train_rows]
    train_mean = np.nanmean(train_values, axis=(0, 1), keepdims=True)
    train_std = np.nanstd(train_values, axis=(0, 1), keepdims=True)
    train_mean = np.where(np.isfinite(train_mean), train_mean, 0.0)
    train_std = np.where(np.isfinite(train_std) & (train_std >= 1e-6), train_std, 1.0)
    normalized = (raw - train_mean) / train_std
    normalized = np.where(np.isfinite(normalized), normalized, 0.0)

    warmed_up = np.arange(len(close.index)) >= warmup_rows
    finite_features = np.isfinite(normalized).all(axis=(1, 2)) & warmed_up
    if require_targets:
        finite_target_counts = [np.isfinite(arr).sum(axis=1) >= min_valid_targets for arr in target_paths_raw.values()]
        finite_targets = np.logical_and.reduce(finite_target_counts)
    else:
        finite_targets = np.ones(len(close.index), dtype=bool)
    valid_rows = finite_features & finite_targets if require_targets else finite_features

    if valid_rows.sum() < 160:
        raise ValueError("too few valid rows after feature/target construction")

    stock_returns_1d = ret1.to_numpy(dtype=np.float32)
    if external_node_ids:
        if external_node_returns is None:
            external_returns = np.full((len(close.index), len(external_node_ids)), np.nan, dtype=np.float32)
        else:
            external_returns = external_node_returns.reindex(index=close.index, columns=external_node_ids).to_numpy(dtype=np.float32)
        returns_1d = np.concatenate([stock_returns_1d, external_returns], axis=1)
        open_values = np.concatenate(
            [
                open_observed.to_numpy(dtype=np.float32),
                np.full((len(close.index), len(external_node_ids)), np.nan, dtype=np.float32),
            ],
            axis=1,
        )
        close_values = np.concatenate(
            [
                close_observed.to_numpy(dtype=np.float32),
                np.full((len(close.index), len(external_node_ids)), np.nan, dtype=np.float32),
            ],
            axis=1,
        )
        execution_close_values = np.concatenate(
            [
                execution_close.to_numpy(dtype=np.float32),
                np.full((len(close.index), len(external_node_ids)), np.nan, dtype=np.float32),
            ],
            axis=1,
        )
    else:
        returns_1d = stock_returns_1d
        open_values = open_observed.to_numpy(dtype=np.float32)
        close_values = close_observed.to_numpy(dtype=np.float32)
        execution_close_values = execution_close.to_numpy(dtype=np.float32)

    dates = close.index[valid_rows]
    aligned_event_theme_exposure = None
    if event_theme_exposure is not None:
        if event_theme_exposure.shape[:2] != (len(close.index), len(panel.tickers)):
            raise ValueError(
                "event_theme_exposure shape must be "
                f"(dates={len(close.index)}, tickers={len(panel.tickers)}, themes); "
                f"got {event_theme_exposure.shape}"
            )
        aligned_event_theme_exposure = np.nan_to_num(event_theme_exposure[valid_rows], nan=0.0).astype(np.float32)
    node_tickers = list(panel.tickers) + list(external_node_ids)
    node_names = dict(panel.names)
    if external_node_ids:
        node_names.update({node_id: (external_node_names or {}).get(node_id, node_id) for node_id in external_node_ids})
    if static_edge_index is not None or static_edge_weight is not None:
        if static_edge_index is None or static_edge_weight is None:
            raise ValueError("static edge index and weights must be provided together")
        static_edge_index = np.asarray(static_edge_index, dtype=np.int64)
        static_edge_weight = np.asarray(static_edge_weight, dtype=np.float32)
        if static_edge_index.shape[0] != 2 or static_edge_index.shape[1] != static_edge_weight.shape[0]:
            raise ValueError("static edge index/weight shapes do not match")
        if static_edge_index.size and (
            static_edge_index.min() < 0 or static_edge_index.max() >= len(panel.tickers)
        ):
            raise ValueError("static edges must reference stock node indices only")
    return FeaturePanel(
        dates=dates,
        tickers=list(panel.tickers),
        names=panel.names,
        feature_names=feature_names,
        features=normalized[valid_rows].astype(np.float32),
        raw_features=raw[valid_rows].astype(np.float32),
        available_mask=available_mask[valid_rows].astype(np.float32),
        returns_1d=returns_1d[valid_rows].astype(np.float32),
        target_returns=target[valid_rows].astype(np.float32),
        target_return_paths={h: arr[valid_rows].astype(np.float32) for h, arr in target_paths_raw.items()},
        open=open_values[valid_rows].astype(np.float32),
        close=close_values[valid_rows].astype(np.float32),
        train_mean=train_mean.squeeze().astype(np.float32),
        train_std=train_std.squeeze().astype(np.float32),
        event_theme_exposure=aligned_event_theme_exposure,
        event_theme_names=list(event_theme_names or []),
        static_edge_index=static_edge_index,
        static_edge_weight=static_edge_weight,
        node_tickers=node_tickers,
        node_names=node_names,
        stock_node_count=len(panel.tickers),
        execution_close=execution_close_values[valid_rows].astype(np.float32),
    )


def build_event_edge_tensor(
    features: FeaturePanel,
    step: int,
    top_k: int = 4,
    min_weight: float = 0.05,
    scale: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    exposure = features.event_theme_exposure
    if exposure is None or top_k <= 0 or scale <= 0.0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    if step < 0 or step >= exposure.shape[0] or exposure.shape[2] == 0:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    node_theme = np.nan_to_num(exposure[step], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    norms = np.linalg.norm(node_theme, axis=1, keepdims=True)
    active = norms[:, 0] > 1e-8
    if int(active.sum()) < 2:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )

    normalized = np.divide(node_theme, norms, out=np.zeros_like(node_theme), where=norms > 1e-8)
    similarity = normalized @ normalized.T
    np.fill_diagonal(similarity, 0.0)

    src_nodes: list[int] = []
    dst_nodes: list[int] = []
    weights: list[float] = []
    for src in np.flatnonzero(active):
        row = similarity[int(src)]
        candidates = np.flatnonzero(row >= float(min_weight))
        if candidates.size == 0:
            continue
        ranked = candidates[np.argsort(row[candidates])[-int(top_k) :]]
        for dst in ranked:
            weight = float(row[int(dst)]) * float(scale)
            if weight > 0.0:
                src_nodes.append(int(src))
                dst_nodes.append(int(dst))
                weights.append(weight)

    if not weights:
        return (
            np.zeros((2, 0), dtype=np.int64),
            np.zeros((0,), dtype=np.float32),
        )
    return (
        np.asarray([src_nodes, dst_nodes], dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def _empty_edge_arrays() -> tuple[np.ndarray, np.ndarray]:
    return (
        np.zeros((2, 0), dtype=np.int64),
        np.zeros((0,), dtype=np.float32),
    )


def _edge_arrays_from_matrix(
    matrix: np.ndarray,
    top_k: int,
    min_abs_weight: float,
    mode: str = "signed",
    scale: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    if top_k <= 0 or scale <= 0.0 or matrix.size == 0:
        return _empty_edge_arrays()
    values = np.nan_to_num(matrix.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    values = (
        np.rint(values.astype(np.float64) / EDGE_WEIGHT_QUANTIZATION)
        * EDGE_WEIGHT_QUANTIZATION
    ).astype(np.float32)
    np.fill_diagonal(values, 0.0)
    srcs: list[int] = []
    dsts: list[int] = []
    weights: list[float] = []
    for src in range(values.shape[0]):
        row = values[src]
        if mode == "positive":
            candidates = np.flatnonzero(row >= float(min_abs_weight))
            scores = row[candidates]
        elif mode == "negative":
            candidates = np.flatnonzero(row <= -float(min_abs_weight))
            scores = np.abs(row[candidates])
        else:
            candidates = np.flatnonzero(np.abs(row) >= float(min_abs_weight))
            scores = np.abs(row[candidates])
        if candidates.size == 0:
            continue
        order = np.lexsort((candidates, scores))
        ranked = np.sort(candidates[order[-int(top_k) :]])
        for dst in ranked:
            weight = float(values[src, int(dst)])
            if mode == "abs":
                weight = abs(weight)
            if abs(weight) <= 0.0:
                continue
            srcs.append(src)
            dsts.append(int(dst))
            weights.append(weight * float(scale))
    if not weights:
        return _empty_edge_arrays()
    return (
        np.asarray([srcs, dsts], dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def build_correlation_edge_tensor(
    history: np.ndarray,
    top_k: int = 6,
    min_abs_corr: float = 0.20,
    mode: str = "signed",
) -> tuple[np.ndarray, np.ndarray]:
    if mode == "none" or history.shape[0] < 3:
        return _empty_edge_arrays()
    centered = history - history.mean(axis=0, keepdims=True)
    std = centered.std(axis=0, ddof=1, keepdims=True)
    normalized = np.divide(
        centered,
        std,
        out=np.zeros_like(centered, dtype=np.float32),
        where=std > 1e-8,
    )
    corr = normalized.T @ normalized / max(1, history.shape[0] - 1)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return _edge_arrays_from_matrix(
        corr,
        top_k=top_k,
        min_abs_weight=min_abs_corr,
        mode=mode,
    )


def build_partial_correlation_edge_tensor(
    history: np.ndarray,
    top_k: int = 0,
    min_abs_corr: float = 0.10,
    mode: str = "signed",
    scale: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    if top_k <= 0 or history.shape[0] < 4:
        return _empty_edge_arrays()
    covariance = np.cov(history, rowvar=False)
    if covariance.ndim != 2:
        return _empty_edge_arrays()
    covariance = np.nan_to_num(covariance, nan=0.0, posinf=0.0, neginf=0.0)
    covariance = covariance + np.eye(covariance.shape[0], dtype=np.float32) * 1e-6
    precision = np.linalg.pinv(covariance)
    diag = np.sqrt(np.clip(np.diag(precision), 1e-12, None))
    partial = -precision / np.outer(diag, diag)
    np.fill_diagonal(partial, 0.0)
    return _edge_arrays_from_matrix(
        partial,
        top_k=top_k,
        min_abs_weight=min_abs_corr,
        mode=mode,
        scale=scale,
    )


def build_lead_lag_edge_tensor(
    history: np.ndarray,
    lag_days: int = 1,
    top_k: int = 0,
    min_abs_corr: float = 0.08,
    mode: str = "signed",
    scale: float = 0.50,
) -> tuple[np.ndarray, np.ndarray]:
    lag = max(1, int(lag_days))
    if top_k <= 0 or history.shape[0] <= lag + 3:
        return _empty_edge_arrays()
    leaders = history[:-lag]
    followers = history[lag:]
    leaders = leaders - leaders.mean(axis=0, keepdims=True)
    followers = followers - followers.mean(axis=0, keepdims=True)
    leader_std = leaders.std(axis=0, keepdims=True)
    follower_std = followers.std(axis=0, keepdims=True)
    leaders = np.divide(leaders, leader_std, out=np.zeros_like(leaders), where=leader_std > 1e-8)
    followers = np.divide(followers, follower_std, out=np.zeros_like(followers), where=follower_std > 1e-8)
    matrix = leaders.T @ followers / max(1, leaders.shape[0] - 1)
    np.fill_diagonal(matrix, 0.0)
    return _edge_arrays_from_matrix(
        matrix,
        top_k=top_k,
        min_abs_weight=min_abs_corr,
        mode=mode,
        scale=scale,
    )


def build_policy_rate_broadcast_edge_tensor(
    features: FeaturePanel,
    scale: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Broadcast observed policy state into stocks without sparse-change correlation."""

    if scale <= 0.0 or not features.node_tickers or features.tradable_count <= 0:
        return _empty_edge_arrays()
    policy_ids = {f"EXT:{name}" for name in POLICY_RATE_FACTOR_NAMES}
    policy_indices = [
        index
        for index, node_id in enumerate(features.node_tickers)
        if node_id in policy_ids
    ]
    if not policy_indices:
        return _empty_edge_arrays()
    stock_indices = np.arange(features.tradable_count, dtype=np.int64)
    sources = np.repeat(np.asarray(policy_indices, dtype=np.int64), features.tradable_count)
    destinations = np.tile(stock_indices, len(policy_indices))
    weights = np.full(sources.shape[0], float(scale), dtype=np.float32)
    return np.stack([sources, destinations]), weights


def build_factor_sensitivity_edge_tensor(
    features: "FeaturePanel",
    history: np.ndarray,
    top_k: int = 0,
    min_abs_corr: float = 0.15,
    mode: str = "signed",
    scale: float = 0.50,
    permute_seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Selective external-factor -> stock edges by trailing return sensitivity.

    Unlike `build_policy_rate_broadcast_edge_tensor`, which sends every policy
    node to every stock with one uniform weight, this connects each external
    factor only to the `top_k` stocks whose trailing returns actually co-move
    with it. `history` is the same window the sibling builders receive, so the
    causality convention is unchanged.

    The matrix is factors x stocks and therefore rectangular, so it cannot use
    `_edge_arrays_from_matrix` (that helper zeroes a diagonal, which is
    meaningless here and would silently drop real factor/stock pairs).
    """

    if top_k <= 0 or scale <= 0.0 or mode == "none":
        return _empty_edge_arrays()
    if not features.node_tickers or features.tradable_count <= 0:
        return _empty_edge_arrays()
    if history.ndim != 2 or history.shape[0] < 3:
        return _empty_edge_arrays()
    stock_count = int(features.tradable_count)
    external_indices = [
        index
        for index, node_id in enumerate(features.node_tickers)
        if str(node_id).startswith("EXT:")
    ]
    if not external_indices or history.shape[1] <= stock_count:
        return _empty_edge_arrays()
    external_indices = [index for index in external_indices if index < history.shape[1]]
    if not external_indices:
        return _empty_edge_arrays()

    values = np.nan_to_num(history.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    centered = values - values.mean(axis=0, keepdims=True)
    deviation = centered.std(axis=0, keepdims=True)
    normalized = np.divide(
        centered, deviation, out=np.zeros_like(centered), where=deviation > 1e-8
    )
    factors = normalized[:, external_indices]
    stocks = normalized[:, :stock_count]
    matrix = factors.T @ stocks / max(1, normalized.shape[0] - 1)
    matrix = np.clip(matrix, -1.0, 1.0)
    matrix = (
        np.rint(matrix / EDGE_WEIGHT_QUANTIZATION) * EDGE_WEIGHT_QUANTIZATION
    ).astype(np.float32)

    srcs: list[int] = []
    dsts: list[int] = []
    weights: list[float] = []
    for row, source in enumerate(external_indices):
        line = matrix[row]
        if mode == "positive":
            candidates = np.flatnonzero(line >= float(min_abs_corr))
            scores = line[candidates]
        elif mode == "negative":
            candidates = np.flatnonzero(line <= -float(min_abs_corr))
            scores = np.abs(line[candidates])
        else:
            candidates = np.flatnonzero(np.abs(line) >= float(min_abs_corr))
            scores = np.abs(line[candidates])
        if candidates.size == 0:
            continue
        order = np.lexsort((candidates, scores))
        ranked = np.sort(candidates[order[-int(top_k) :]])
        for destination in ranked:
            weight = float(line[int(destination)])
            if mode == "abs":
                weight = abs(weight)
            if abs(weight) <= 0.0:
                continue
            srcs.append(int(source))
            dsts.append(int(destination))
            weights.append(weight * float(scale))
    if not weights:
        return _empty_edge_arrays()
    if int(permute_seed) > 0:
        # Placebo: keep sources, per-factor edge counts, and the weight multiset
        # exactly as selected, and permute only the destination stocks. Each
        # factor gets one permutation table keyed by the seed and its source
        # index, so the mapping is identical across steps, workers, and reruns,
        # and distinct destinations stay distinct.
        tables = {
            source: np.random.default_rng(
                abs(int(permute_seed)) * 1_000_003 + int(source)
            ).permutation(int(stock_count))
            for source in sorted(set(srcs))
        }
        dsts = [
            int(tables[source][destination])
            for source, destination in zip(srcs, dsts)
        ]
    return (
        np.asarray([srcs, dsts], dtype=np.int64),
        np.asarray(weights, dtype=np.float32),
    )


def build_edge_tensor(
    features: FeaturePanel,
    step: int,
    edge_window: int = 60,
    top_k: int = 6,
    min_abs_corr: float = 0.20,
    correlation_mode: str = "signed",
    event_top_k: int = 0,
    event_min_weight: float = 0.05,
    event_scale: float = 0.25,
    partial_corr_top_k: int = 0,
    partial_corr_min_abs: float = 0.10,
    partial_corr_mode: str = "signed",
    partial_corr_scale: float = 0.50,
    lead_lag_top_k: int = 0,
    lead_lag_days: int = 1,
    lead_lag_min_abs_corr: float = 0.08,
    lead_lag_mode: str = "signed",
    lead_lag_scale: float = 0.50,
    policy_rate_edge_scale: float = 0.0,
    factor_sensitivity_top_k: int = 0,
    factor_sensitivity_min_abs_corr: float = 0.15,
    factor_sensitivity_mode: str = "signed",
    factor_sensitivity_scale: float = 0.50,
    factor_sensitivity_permute_seed: int = 0,
    ownership_edge_scale: float = 0.0,
    sequence_window: int = 0,   # accepted for kwargs-compatibility with make_real_snapshot; edges ignore it
) -> tuple[torch.Tensor, torch.Tensor]:
    start = max(0, step - edge_window)
    history = np.nan_to_num(features.returns_1d[start : step + 1], nan=0.0, posinf=0.0, neginf=0.0)
    edge_parts: list[tuple[np.ndarray, np.ndarray]] = []
    edge_parts.append(
        build_correlation_edge_tensor(
            history,
            top_k=top_k,
            min_abs_corr=min_abs_corr,
            mode=correlation_mode,
        )
    )
    edge_parts.append(
        build_partial_correlation_edge_tensor(
            history,
            top_k=partial_corr_top_k,
            min_abs_corr=partial_corr_min_abs,
            mode=partial_corr_mode,
            scale=partial_corr_scale,
        )
    )
    edge_parts.append(
        build_lead_lag_edge_tensor(
            history,
            lag_days=lead_lag_days,
            top_k=lead_lag_top_k,
            min_abs_corr=lead_lag_min_abs_corr,
            mode=lead_lag_mode,
            scale=lead_lag_scale,
        )
    )
    event_edge_index, event_edge_weight = build_event_edge_tensor(
        features,
        step=step,
        top_k=event_top_k,
        min_weight=event_min_weight,
        scale=event_scale,
    )
    edge_parts.append((event_edge_index, event_edge_weight))
    edge_parts.append(
        build_policy_rate_broadcast_edge_tensor(
            features,
            scale=policy_rate_edge_scale,
        )
    )
    edge_parts.append(
        build_factor_sensitivity_edge_tensor(
            features,
            history,
            top_k=factor_sensitivity_top_k,
            min_abs_corr=factor_sensitivity_min_abs_corr,
            mode=factor_sensitivity_mode,
            scale=factor_sensitivity_scale,
            permute_seed=factor_sensitivity_permute_seed,
        )
    )
    edge_parts.append(
        build_ownership_edge_tensor(features, step=step, scale=ownership_edge_scale)
    )
    if features.static_edge_index is not None and features.static_edge_weight is not None:
        edge_parts.append((features.static_edge_index, features.static_edge_weight))

    edge_indices = [edge_index for edge_index, edge_weight in edge_parts if edge_weight.size]
    edge_weights = [edge_weight.astype(np.float32) for _edge_index, edge_weight in edge_parts if edge_weight.size]
    if not edge_weights:
        edge_index, edge_weight = _empty_edge_arrays()
    else:
        edge_index = np.concatenate(edge_indices, axis=1)
        edge_weight = np.concatenate(edge_weights)
        edge_weight = (
            np.rint(edge_weight.astype(np.float64) / EDGE_WEIGHT_QUANTIZATION)
            * EDGE_WEIGHT_QUANTIZATION
        ).astype(np.float32)
        order = np.lexsort((edge_weight, edge_index[1], edge_index[0]))
        edge_index = edge_index[:, order]
        edge_weight = edge_weight[order]
    return (
        torch.tensor(edge_index, dtype=torch.long),
        torch.tensor(edge_weight, dtype=torch.float32),
    )


def make_real_snapshot(
    features: FeaturePanel,
    step: int,
    hide_ratio: float = 0.30,
    edge_window: int = 60,
    top_k: int = 6,
    min_abs_corr: float = 0.20,
    correlation_mode: str = "signed",
    event_top_k: int = 0,
    event_min_weight: float = 0.05,
    event_scale: float = 0.25,
    partial_corr_top_k: int = 0,
    partial_corr_min_abs: float = 0.10,
    partial_corr_mode: str = "signed",
    partial_corr_scale: float = 0.50,
    lead_lag_top_k: int = 0,
    lead_lag_days: int = 1,
    lead_lag_min_abs_corr: float = 0.08,
    lead_lag_mode: str = "signed",
    lead_lag_scale: float = 0.50,
    policy_rate_edge_scale: float = 0.0,
    factor_sensitivity_top_k: int = 0,
    factor_sensitivity_min_abs_corr: float = 0.15,
    factor_sensitivity_mode: str = "signed",
    factor_sensitivity_scale: float = 0.50,
    factor_sensitivity_permute_seed: int = 0,
    ownership_edge_scale: float = 0.0,
    sequence_window: int = 0,
    seed: int | None = None,
    full_observation: bool = False,
    mask_strategy: str = "random_cell",
    edge_cache: Dict[int, tuple[torch.Tensor, torch.Tensor]] | None = None,
) -> GraphBatch:
    x = torch.tensor(features.features[step], dtype=torch.float32)
    available_mask = torch.tensor(features.available_mask[step], dtype=torch.float32)
    supervision_node_mask = torch.zeros(x.shape[0], dtype=torch.float32)
    supervision_node_mask[: features.tradable_count] = 1.0
    if full_observation:
        feature_mask = available_mask.clone()
    else:
        generator = None if seed is None else torch.Generator().manual_seed(seed)
        if mask_strategy == "random_cell":
            feature_mask = make_feature_mask(x, hide_ratio=hide_ratio, generator=generator)
        else:
            feature_mask = make_structured_feature_mask(
                x,
                feature_names=features.feature_names,
                hide_ratio=hide_ratio,
                strategy=mask_strategy,
                generator=generator,
            )
        feature_mask = torch.where(available_mask > 0.5, feature_mask, torch.zeros_like(feature_mask))
        if features.tradable_count < x.shape[0]:
            feature_mask[features.tradable_count :] = available_mask[features.tradable_count :]
    cached_edges = None if edge_cache is None else edge_cache.get(int(step))
    if cached_edges is None:
        edge_index, edge_weight = build_edge_tensor(
            features,
            step=step,
            edge_window=edge_window,
            top_k=top_k,
            min_abs_corr=min_abs_corr,
            correlation_mode=correlation_mode,
            event_top_k=event_top_k,
            event_min_weight=event_min_weight,
            event_scale=event_scale,
            partial_corr_top_k=partial_corr_top_k,
            partial_corr_min_abs=partial_corr_min_abs,
            partial_corr_mode=partial_corr_mode,
            partial_corr_scale=partial_corr_scale,
            lead_lag_top_k=lead_lag_top_k,
            lead_lag_days=lead_lag_days,
            lead_lag_min_abs_corr=lead_lag_min_abs_corr,
            lead_lag_mode=lead_lag_mode,
            lead_lag_scale=lead_lag_scale,
            policy_rate_edge_scale=policy_rate_edge_scale,
            factor_sensitivity_top_k=factor_sensitivity_top_k,
            factor_sensitivity_min_abs_corr=factor_sensitivity_min_abs_corr,
            factor_sensitivity_mode=factor_sensitivity_mode,
            factor_sensitivity_scale=factor_sensitivity_scale,
            factor_sensitivity_permute_seed=factor_sensitivity_permute_seed,
            ownership_edge_scale=ownership_edge_scale,
        )
    else:
        edge_index, edge_weight = cached_edges
    node_sequence = None
    if int(sequence_window) > 0:
        # Trailing window of encoder inputs [values*mask, mask], newest last.
        # Past rows were fully observable at decision time, so they carry only
        # the availability mask; the mask-training game (feature_mask) applies
        # to the CURRENT row alone -- last month's tape is known, today's may
        # have gaps. Steps before listing replicate the oldest available row
        # rather than fabricating zeros.
        w = int(sequence_window)
        rows = []
        for back in range(w - 1, -1, -1):
            s_idx = step - back
            if s_idx < 0:
                s_idx = max(step - (w - 1), 0)
            xv = torch.tensor(features.features[s_idx], dtype=torch.float32)
            av = torch.tensor(features.available_mask[s_idx], dtype=torch.float32)
            m = feature_mask if s_idx == step else av
            rows.append(torch.cat([xv * m, m], dim=-1))
        node_sequence = torch.stack(rows, dim=1)
    return GraphBatch(
        node_features=x,
        feature_mask=feature_mask,
        edge_index=edge_index,
        edge_weight=edge_weight,
        available_mask=available_mask,
        supervision_node_mask=supervision_node_mask,
        node_sequence=node_sequence,
    )
