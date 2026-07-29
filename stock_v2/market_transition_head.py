from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from stock_v2.market_transition import (
    EVENT_NAMES,
    MARKET_TRANSITION_FAMILIES,
)


MARKET_COMPONENT_TARGETS = tuple(
    dict.fromkeys(
        name
        for family_names in MARKET_TRANSITION_FAMILIES.values()
        for name in family_names
    )
)
MARKET_FAMILY_TARGETS = tuple(MARKET_TRANSITION_FAMILIES)
MARKET_EVENT_TARGETS = EVENT_NAMES
MARKET_COMPONENT_FAMILY_INDEX = tuple(
    next(
        family_index
        for family_index, family in enumerate(MARKET_FAMILY_TARGETS)
        if component in MARKET_TRANSITION_FAMILIES[family]
    )
    for component in MARKET_COMPONENT_TARGETS
)


class _TrajectoryEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        *,
        hidden_dim: int,
        layers: int,
        heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        normalized_horizons = tuple(int(value) for value in horizons)
        if not normalized_horizons or len(set(normalized_horizons)) != len(
            normalized_horizons
        ):
            raise ValueError("horizons must be unique and non-empty")
        if int(hidden_dim) % int(heads):
            raise ValueError("hidden_dim must be divisible by heads")
        self.horizons = normalized_horizons
        self.input_projector = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(hidden_dim)),
        )
        self.horizon_embedding = nn.Embedding(
            len(normalized_horizons), int(hidden_dim)
        )
        layer = nn.TransformerEncoderLayer(
            d_model=int(hidden_dim),
            nhead=int(heads),
            dim_feedforward=2 * int(hidden_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.temporal_encoder = nn.TransformerEncoder(
            layer, num_layers=int(layers), norm=nn.LayerNorm(int(hidden_dim))
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        if sequence.ndim != 3 or sequence.shape[1] != len(self.horizons):
            raise ValueError("trajectory input must be batch-by-horizon-by-feature")
        hidden = self.input_projector(sequence)
        indices = torch.arange(
            len(self.horizons), dtype=torch.long, device=sequence.device
        )
        hidden = hidden + self.horizon_embedding(indices)[None, :, :]
        return self.temporal_encoder(hidden)


class MarketTrajectoryHead(nn.Module):
    """Joint market-transition readout over all frozen JEPA rollout horizons."""

    def __init__(
        self,
        latent_dim: int,
        horizons: Sequence[int],
        *,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.10,
        component_count: int = len(MARKET_COMPONENT_TARGETS),
        family_count: int = len(MARKET_FAMILY_TARGETS),
        event_count: int = len(MARKET_EVENT_TARGETS),
        stock_quantiles: bool = False,
        preserve_external_identity: bool = False,
        external_node_count: int = 0,
    ) -> None:
        super().__init__()
        if int(latent_dim) < 1 or int(projection_dim) < 1:
            raise ValueError("latent and projection dimensions must be positive")
        self.latent_dim = int(latent_dim)
        self.horizons = tuple(int(value) for value in horizons)
        self.stock_quantiles = bool(stock_quantiles)
        self.preserve_external_identity = bool(preserve_external_identity)
        self.external_node_count = int(external_node_count)
        if self.external_node_count < 0:
            raise ValueError("external node count cannot be negative")
        if self.preserve_external_identity and self.external_node_count < 1:
            raise ValueError(
                "preserving external identity requires a positive external node count"
            )
        self.node_projector = nn.Sequential(
            nn.LayerNorm(2 * int(latent_dim)),
            nn.Linear(2 * int(latent_dim), int(projection_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(projection_dim)),
            nn.Dropout(float(dropout)),
        )
        stock_stat_count = 5 if self.stock_quantiles else 3
        external_stat_count = (
            self.external_node_count if self.preserve_external_identity else 2
        )
        pooled_dim = (stock_stat_count + external_stat_count) * int(projection_dim)
        self.trajectory_encoder = _TrajectoryEncoder(
            pooled_dim,
            self.horizons,
            hidden_dim=int(hidden_dim),
            layers=int(layers),
            heads=int(heads),
            dropout=float(dropout),
        )
        self.component_head = nn.Linear(int(hidden_dim), int(component_count))
        self.family_head = nn.Linear(int(hidden_dim), int(family_count))
        self.event_head = nn.Linear(int(hidden_dim), int(event_count))

    def _pool(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        *,
        batch_size: int,
        node_count: int,
        stock_count: int,
    ) -> torch.Tensor:
        if context.shape != predicted.shape or context.ndim != 2:
            raise ValueError("context and predicted latent rows must align")
        if context.shape[1] != self.latent_dim:
            raise ValueError("latent width does not match the trajectory head")
        if int(batch_size) * int(node_count) != context.shape[0]:
            raise ValueError("batch and node dimensions do not match latent rows")
        if not 0 < int(stock_count) <= int(node_count):
            raise ValueError("stock_count must be within each graph")
        projected = self.node_projector(
            torch.cat((context, predicted - context), dim=-1)
        ).reshape(int(batch_size), int(node_count), -1)
        stock = projected[:, : int(stock_count)]
        stock_mean = stock.mean(dim=1)
        stock_std = stock.std(dim=1, unbiased=False)
        stock_median = stock.median(dim=1).values
        stock_features = [stock_mean, stock_std]
        if self.stock_quantiles:
            stock_features.extend(
                (
                    torch.quantile(stock, 0.25, dim=1),
                    stock_median,
                    torch.quantile(stock, 0.75, dim=1),
                )
            )
        else:
            stock_features.append(stock_median)
        external = projected[:, int(stock_count) :]
        if self.preserve_external_identity:
            if external.shape[1] != self.external_node_count:
                raise ValueError("external node count differs from the trajectory head")
            external_features = [external.flatten(start_dim=1)]
        elif external.shape[1]:
            external_mean = external.mean(dim=1)
            external_std = external.std(dim=1, unbiased=False)
            external_features = [external_mean, external_std]
        else:
            external_mean = torch.zeros_like(stock_mean)
            external_std = torch.zeros_like(stock_mean)
            external_features = [external_mean, external_std]
        return torch.cat((*stock_features, *external_features), dim=-1)

    def forward(
        self,
        context: torch.Tensor,
        predicted: Mapping[int, torch.Tensor],
        *,
        batch_size: int,
        node_count: int,
        stock_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if set(int(key) for key in predicted) != set(self.horizons):
            raise ValueError("predicted latents must contain every configured horizon")
        pooled = torch.stack(
            [
                self._pool(
                    context,
                    predicted[int(horizon)],
                    batch_size=int(batch_size),
                    node_count=int(node_count),
                    stock_count=int(stock_count),
                )
                for horizon in self.horizons
            ],
            dim=1,
        )
        hidden = self.trajectory_encoder(pooled)
        return (
            self.component_head(hidden),
            self.family_head(hidden),
            self.event_head(hidden),
        )


class FamilyQueryMarketTrajectoryHead(nn.Module):
    """Family-aligned market tokens that attend to every frozen graph node."""

    def __init__(
        self,
        latent_dim: int,
        horizons: Sequence[int],
        *,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.10,
        component_count: int = len(MARKET_COMPONENT_TARGETS),
        family_count: int = len(MARKET_FAMILY_TARGETS),
        event_count: int = len(MARKET_EVENT_TARGETS),
        stock_quantiles: bool = True,
        external_node_count: int = 0,
    ) -> None:
        super().__init__()
        normalized_horizons = tuple(int(value) for value in horizons)
        if not normalized_horizons or len(set(normalized_horizons)) != len(
            normalized_horizons
        ):
            raise ValueError("horizons must be unique and non-empty")
        if int(latent_dim) < 1 or int(projection_dim) < 1:
            raise ValueError("latent and projection dimensions must be positive")
        if int(projection_dim) % int(heads) or int(hidden_dim) % int(heads):
            raise ValueError("projection_dim and hidden_dim must be divisible by heads")
        if int(family_count) != len(MARKET_FAMILY_TARGETS):
            raise ValueError("family query count must match market transition families")
        if int(component_count) != len(MARKET_COMPONENT_TARGETS):
            raise ValueError("component count must match market transition targets")
        if int(external_node_count) < 0:
            raise ValueError("external node count cannot be negative")

        self.latent_dim = int(latent_dim)
        self.horizons = normalized_horizons
        self.projection_dim = int(projection_dim)
        self.hidden_dim = int(hidden_dim)
        self.stock_quantiles = bool(stock_quantiles)
        self.external_node_count = int(external_node_count)
        self.family_count = int(family_count)
        self.node_projector = nn.Sequential(
            nn.LayerNorm(2 * self.latent_dim),
            nn.Linear(2 * self.latent_dim, self.projection_dim),
            nn.SiLU(),
            nn.LayerNorm(self.projection_dim),
            nn.Dropout(float(dropout)),
        )

        self.stat_count = 5 if self.stock_quantiles else 3
        self.node_type_embedding = nn.Embedding(3, self.projection_dim)
        self.stat_identity_embedding = nn.Embedding(
            self.stat_count, self.projection_dim
        )
        self.external_identity_embedding = (
            nn.Embedding(self.external_node_count, self.projection_dim)
            if self.external_node_count
            else None
        )
        self.family_query_embedding = nn.Embedding(
            self.family_count, self.projection_dim
        )
        self.memory_norm = nn.LayerNorm(self.projection_dim)
        self.query_norm = nn.LayerNorm(self.projection_dim)
        self.cross_attention = nn.MultiheadAttention(
            self.projection_dim,
            int(heads),
            dropout=float(dropout),
            batch_first=True,
        )
        self.cross_attention_dropout = nn.Dropout(float(dropout))
        self.query_feed_forward = nn.Sequential(
            nn.LayerNorm(self.projection_dim),
            nn.Linear(self.projection_dim, 2 * self.projection_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(2 * self.projection_dim, self.projection_dim),
            nn.Dropout(float(dropout)),
        )

        self.token_projector = nn.Sequential(
            nn.LayerNorm(self.projection_dim),
            nn.Linear(self.projection_dim, self.hidden_dim),
            nn.SiLU(),
            nn.LayerNorm(self.hidden_dim),
        )
        self.horizon_embedding = nn.Embedding(
            len(self.horizons), self.hidden_dim
        )
        self.family_embedding = nn.Embedding(self.family_count, self.hidden_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=self.hidden_dim,
            nhead=int(heads),
            dim_feedforward=2 * self.hidden_dim,
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.trajectory_encoder = nn.TransformerEncoder(
            layer,
            num_layers=int(layers),
            norm=nn.LayerNorm(self.hidden_dim),
        )
        self.component_heads = nn.ModuleList(
            nn.Linear(self.hidden_dim, 1) for _ in range(int(component_count))
        )
        self.family_heads = nn.ModuleList(
            nn.Linear(self.hidden_dim, 1) for _ in range(self.family_count)
        )
        self.event_head = nn.Sequential(
            nn.LayerNorm(self.family_count * self.hidden_dim),
            nn.Linear(self.family_count * self.hidden_dim, int(event_count)),
        )

    def _memory(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        *,
        batch_size: int,
        node_count: int,
        stock_count: int,
    ) -> torch.Tensor:
        if context.shape != predicted.shape or context.ndim != 2:
            raise ValueError("context and predicted latent rows must align")
        if context.shape[1] != self.latent_dim:
            raise ValueError("latent width does not match the family-query head")
        if int(batch_size) * int(node_count) != context.shape[0]:
            raise ValueError("batch and node dimensions do not match latent rows")
        if not 0 < int(stock_count) <= int(node_count):
            raise ValueError("stock_count must be within each graph")
        observed_external = int(node_count) - int(stock_count)
        if observed_external != self.external_node_count:
            raise ValueError("external node count differs from the family-query head")

        projected = self.node_projector(
            torch.cat((context, predicted - context), dim=-1)
        ).reshape(int(batch_size), int(node_count), self.projection_dim)
        device = projected.device
        stock = projected[:, : int(stock_count)]
        stock_type = self.node_type_embedding(
            torch.zeros(int(stock_count), dtype=torch.long, device=device)
        )[None, :, :]
        memories = [stock + stock_type]

        external = projected[:, int(stock_count) :]
        if self.external_node_count:
            external_positions = torch.arange(
                self.external_node_count, dtype=torch.long, device=device
            )
            external_type = self.node_type_embedding(
                torch.ones(
                    self.external_node_count, dtype=torch.long, device=device
                )
            )
            external_identity = self.external_identity_embedding(external_positions)
            memories.append(
                external
                + external_type[None, :, :]
                + external_identity[None, :, :]
            )

        stock_mean = stock.mean(dim=1)
        stock_std = stock.std(dim=1, unbiased=False)
        stock_median = stock.median(dim=1).values
        if self.stock_quantiles:
            statistics = torch.stack(
                (
                    stock_mean,
                    stock_std,
                    torch.quantile(stock, 0.25, dim=1),
                    stock_median,
                    torch.quantile(stock, 0.75, dim=1),
                ),
                dim=1,
            )
        else:
            statistics = torch.stack(
                (stock_mean, stock_std, stock_median), dim=1
            )
        stat_positions = torch.arange(
            self.stat_count, dtype=torch.long, device=device
        )
        stat_type = self.node_type_embedding(
            torch.full(
                (self.stat_count,), 2, dtype=torch.long, device=device
            )
        )
        statistics = (
            statistics
            + stat_type[None, :, :]
            + self.stat_identity_embedding(stat_positions)[None, :, :]
        )
        memories.append(statistics)
        return self.memory_norm(torch.cat(memories, dim=1))

    def _family_tokens(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        *,
        batch_size: int,
        node_count: int,
        stock_count: int,
    ) -> torch.Tensor:
        memory = self._memory(
            context,
            predicted,
            batch_size=int(batch_size),
            node_count=int(node_count),
            stock_count=int(stock_count),
        )
        family_positions = torch.arange(
            self.family_count, dtype=torch.long, device=context.device
        )
        query = self.family_query_embedding(family_positions)[None, :, :].expand(
            int(batch_size), -1, -1
        )
        attended, _ = self.cross_attention(
            self.query_norm(query), memory, memory, need_weights=False
        )
        query = query + self.cross_attention_dropout(attended)
        return query + self.query_feed_forward(query)

    def forward(
        self,
        context: torch.Tensor,
        predicted: Mapping[int, torch.Tensor],
        *,
        batch_size: int,
        node_count: int,
        stock_count: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if set(int(key) for key in predicted) != set(self.horizons):
            raise ValueError("predicted latents must contain every configured horizon")
        tokens = torch.stack(
            [
                self._family_tokens(
                    context,
                    predicted[int(horizon)],
                    batch_size=int(batch_size),
                    node_count=int(node_count),
                    stock_count=int(stock_count),
                )
                for horizon in self.horizons
            ],
            dim=1,
        )
        hidden = self.token_projector(tokens)
        device = context.device
        horizon_positions = torch.arange(
            len(self.horizons), dtype=torch.long, device=device
        )
        family_positions = torch.arange(
            self.family_count, dtype=torch.long, device=device
        )
        hidden = (
            hidden
            + self.horizon_embedding(horizon_positions)[None, :, None, :]
            + self.family_embedding(family_positions)[None, None, :, :]
        )
        encoded = self.trajectory_encoder(
            hidden.reshape(int(batch_size), -1, self.hidden_dim)
        ).reshape(
            int(batch_size),
            len(self.horizons),
            self.family_count,
            self.hidden_dim,
        )

        components = torch.stack(
            [
                head(encoded[:, :, int(family_index), :]).squeeze(-1)
                for head, family_index in zip(
                    self.component_heads, MARKET_COMPONENT_FAMILY_INDEX
                )
            ],
            dim=-1,
        )
        families = torch.stack(
            [
                head(encoded[:, :, family_index, :]).squeeze(-1)
                for family_index, head in enumerate(self.family_heads)
            ],
            dim=-1,
        )
        events = self.event_head(encoded.flatten(start_dim=2))
        return components, families, events


class DirectMarketTrajectoryHead(nn.Module):
    """Same trajectory outputs from causal observable graph summaries."""

    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        *,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.10,
        component_count: int = len(MARKET_COMPONENT_TARGETS),
        family_count: int = len(MARKET_FAMILY_TARGETS),
        event_count: int = len(MARKET_EVENT_TARGETS),
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(value) for value in horizons)
        self.trajectory_encoder = _TrajectoryEncoder(
            int(input_dim),
            self.horizons,
            hidden_dim=int(hidden_dim),
            layers=int(layers),
            heads=int(heads),
            dropout=float(dropout),
        )
        self.component_head = nn.Linear(int(hidden_dim), int(component_count))
        self.family_head = nn.Linear(int(hidden_dim), int(family_count))
        self.event_head = nn.Linear(int(hidden_dim), int(event_count))

    def forward(
        self, values: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 2:
            raise ValueError("direct market input must be a row matrix")
        sequence = values[:, None, :].expand(-1, len(self.horizons), -1)
        hidden = self.trajectory_encoder(sequence)
        return (
            self.component_head(hidden),
            self.family_head(hidden),
            self.event_head(hidden),
        )


class PooledMarketTrajectoryHead(nn.Module):
    """Joint readout over cached robust graph-transition representations."""

    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        *,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.10,
        component_count: int = len(MARKET_COMPONENT_TARGETS),
        family_count: int = len(MARKET_FAMILY_TARGETS),
        event_count: int = len(MARKET_EVENT_TARGETS),
    ) -> None:
        super().__init__()
        self.horizons = tuple(int(value) for value in horizons)
        self.trajectory_encoder = _TrajectoryEncoder(
            int(input_dim),
            self.horizons,
            hidden_dim=int(hidden_dim),
            layers=int(layers),
            heads=int(heads),
            dropout=float(dropout),
        )
        self.component_head = nn.Linear(int(hidden_dim), int(component_count))
        self.family_head = nn.Linear(int(hidden_dim), int(family_count))
        self.event_head = nn.Linear(int(hidden_dim), int(event_count))

    def forward(
        self, sequence: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        hidden = self.trajectory_encoder(sequence)
        return (
            self.component_head(hidden),
            self.family_head(hidden),
            self.event_head(hidden),
        )


def weighted_masked_smooth_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("prediction, target, and valid shapes must match")
    if sample_weight.shape != prediction.shape[:2]:
        raise ValueError("sample weights must be batch-by-horizon")
    error = F.smooth_l1_loss(prediction, target, reduction="none")
    weight = sample_weight[..., None].to(error.dtype) * valid.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1e-8)


def trajectory_difference_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid: torch.Tensor,
    sample_weight: torch.Tensor,
    horizons: Sequence[int],
) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.shape != valid.shape:
        raise ValueError("trajectory tensors must have matching shapes")
    if prediction.ndim != 3 or prediction.shape[1] != len(horizons):
        raise ValueError("trajectory tensors must be batch-by-horizon-by-family")
    gaps = torch.as_tensor(
        [int(right) - int(left) for left, right in zip(horizons[:-1], horizons[1:])],
        dtype=prediction.dtype,
        device=prediction.device,
    ).clamp_min(1.0)
    predicted_slope = (prediction[:, 1:] - prediction[:, :-1]) / gaps[None, :, None]
    target_slope = (target[:, 1:] - target[:, :-1]) / gaps[None, :, None]
    slope_valid = valid[:, 1:] & valid[:, :-1]
    slope_weight = 0.5 * (sample_weight[:, 1:] + sample_weight[:, :-1])
    error = F.smooth_l1_loss(predicted_slope, target_slope, reduction="none")
    weight = slope_weight[..., None].to(error.dtype) * slope_valid.to(error.dtype)
    return (error * weight).sum() / weight.sum().clamp_min(1e-8)
