from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch

from stock_v2.edge_update import RollingCorrelationEdgeUpdater
from stock_v2.graph_jepa import GraphBatch, make_feature_mask


@dataclass
class SyntheticMarket:
    node_features: np.ndarray
    returns: np.ndarray
    sectors: np.ndarray
    edge_index: np.ndarray
    edge_weight: np.ndarray
    feature_names: List[str]


def generate_synthetic_market(
    num_steps: int = 160,
    num_nodes: int = 36,
    num_features: int = 8,
    num_sectors: int = 6,
    seed: int = 7,
) -> SyntheticMarket:
    """Create a small dynamic market with hidden market/sector factors."""

    rng = np.random.default_rng(seed)
    sectors = np.arange(num_nodes) % num_sectors

    market = np.zeros(num_steps, dtype=np.float32)
    sector_factors = np.zeros((num_steps, num_sectors), dtype=np.float32)
    returns = np.zeros((num_steps, num_nodes), dtype=np.float32)

    for t in range(1, num_steps):
        market[t] = 0.92 * market[t - 1] + rng.normal(0.0, 0.018)
        sector_factors[t] = 0.84 * sector_factors[t - 1] + rng.normal(0.0, 0.025, size=num_sectors)
        for node in range(num_nodes):
            sector = sectors[node]
            lead = returns[t - 1, (node - 1) % num_nodes]
            returns[t, node] = (
                0.22 * returns[t - 1, node]
                + 0.45 * market[t]
                + 0.55 * sector_factors[t, sector]
                + 0.10 * lead
                + rng.normal(0.0, 0.015)
            )

    volume_z = np.abs(returns) * 12.0 + rng.normal(0.0, 0.2, size=returns.shape)
    volatility = np.zeros_like(returns)
    for t in range(num_steps):
        start = max(0, t - 10)
        volatility[t] = returns[start : t + 1].std(axis=0)

    news = np.zeros_like(returns)
    flow = np.zeros_like(returns)
    for t in range(2, num_steps):
        news[t] = 0.55 * np.sign(sector_factors[t, sectors]) + rng.normal(0.0, 0.4, size=num_nodes)
        flow[t] = 0.6 * flow[t - 1] + 4.0 * returns[t] + rng.normal(0.0, 0.25, size=num_nodes)

    base_features = [
        returns,
        np.roll(returns, 1, axis=0),
        volume_z,
        volatility,
        news,
        flow,
        market[:, None].repeat(num_nodes, axis=1),
        sector_factors[:, sectors],
    ]
    stacked = np.stack(base_features[:num_features], axis=-1).astype(np.float32)

    mean = stacked.mean(axis=(0, 1), keepdims=True)
    std = stacked.std(axis=(0, 1), keepdims=True) + 1e-6
    node_features = (stacked - mean) / std

    updater = RollingCorrelationEdgeUpdater(window=40, top_k=4, min_abs_corr=0.2)
    edge_index, edge_weight = updater.build_edges(returns)

    names = [
        "return_1d",
        "return_lag_1d",
        "volume_z",
        "volatility_10d",
        "news_score",
        "flow_score",
        "market_return",
        "sector_factor",
    ][:num_features]

    return SyntheticMarket(
        node_features=node_features,
        returns=returns,
        sectors=sectors,
        edge_index=edge_index,
        edge_weight=edge_weight,
        feature_names=names,
    )


def make_snapshot(
    market: SyntheticMarket,
    step: int,
    hide_ratio: float = 0.30,
    seed: int | None = None,
) -> GraphBatch:
    generator = None if seed is None else torch.Generator().manual_seed(seed)
    x = torch.tensor(market.node_features[step], dtype=torch.float32)
    feature_mask = make_feature_mask(x, hide_ratio=hide_ratio, generator=generator)
    return GraphBatch(
        node_features=x,
        feature_mask=feature_mask,
        edge_index=torch.tensor(market.edge_index, dtype=torch.long),
        edge_weight=torch.tensor(market.edge_weight, dtype=torch.float32),
    )
