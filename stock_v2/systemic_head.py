from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
import torch.nn.functional as F


SYSTEMIC_COMPONENT_TARGETS = (
    "market_return",
    "mean_absolute_return",
    "median_absolute_return",
    "q75_absolute_return",
    "breadth",
    "return_coherence",
    "robust_return_dispersion",
    "volume_shock",
    "traded_value_shock",
    "common_state_energy",
    "node_state_median_energy",
    "market_corr_change",
)

SYSTEMIC_EVENT_TARGETS = (
    "systemic_event",
    "broad_selloff",
    "turnover_explosion",
    "graph_state_shift",
)


class SystemicTransitionHead(nn.Module):
    """A robust graph-level readout over frozen context and rollout latents."""

    def __init__(
        self,
        latent_dim: int,
        horizons: Sequence[int],
        *,
        projection_dim: int = 128,
        hidden_dim: int = 256,
        horizon_dim: int = 16,
        dropout: float = 0.10,
        component_count: int = len(SYSTEMIC_COMPONENT_TARGETS),
        event_count: int = len(SYSTEMIC_EVENT_TARGETS),
    ) -> None:
        super().__init__()
        if latent_dim < 1 or projection_dim < 1 or hidden_dim < 1 or horizon_dim < 1:
            raise ValueError("systemic head dimensions must be positive")
        normalized_horizons = tuple(int(value) for value in horizons)
        if not normalized_horizons or len(set(normalized_horizons)) != len(
            normalized_horizons
        ):
            raise ValueError("horizons must be unique and non-empty")
        self.latent_dim = int(latent_dim)
        self.horizons = normalized_horizons
        self.horizon_to_index = {
            horizon: index for index, horizon in enumerate(normalized_horizons)
        }
        self.node_projector = nn.Sequential(
            nn.LayerNorm(2 * int(latent_dim)),
            nn.Linear(2 * int(latent_dim), int(projection_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(projection_dim)),
            nn.Dropout(float(dropout)),
        )
        self.horizon_embedding = nn.Embedding(len(normalized_horizons), int(horizon_dim))
        pooled_dim = 5 * int(projection_dim) + int(horizon_dim)
        self.trunk = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.component_head = nn.Linear(int(hidden_dim), int(component_count))
        self.energy_head = nn.Linear(int(hidden_dim), 1)
        self.event_head = nn.Linear(int(hidden_dim), int(event_count))

    def forward(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        *,
        batch_size: int,
        node_count: int,
        stock_count: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.shape != predicted.shape or context.ndim != 2:
            raise ValueError("context and predicted must be aligned node matrices")
        if context.shape[1] != self.latent_dim:
            raise ValueError("latent width does not match the systemic head")
        if int(batch_size) * int(node_count) != context.shape[0]:
            raise ValueError("batch and node dimensions do not match latent rows")
        if not 0 < int(stock_count) <= int(node_count):
            raise ValueError("stock_count must be within each graph")
        if int(horizon) not in self.horizon_to_index:
            raise ValueError(f"unsupported horizon: {horizon}")

        node_input = torch.cat((context, predicted - context), dim=-1)
        projected = self.node_projector(node_input).reshape(
            int(batch_size), int(node_count), -1
        )
        stock = projected[:, : int(stock_count)]
        stock_mean = stock.mean(dim=1)
        stock_std = stock.std(dim=1, unbiased=False)
        stock_median = stock.median(dim=1).values
        external = projected[:, int(stock_count) :]
        if external.shape[1]:
            external_mean = external.mean(dim=1)
            external_std = external.std(dim=1, unbiased=False)
        else:
            external_mean = torch.zeros_like(stock_mean)
            external_std = torch.zeros_like(stock_mean)
        horizon_index = torch.full(
            (int(batch_size),),
            self.horizon_to_index[int(horizon)],
            dtype=torch.long,
            device=context.device,
        )
        horizon_state = self.horizon_embedding(horizon_index)
        pooled = torch.cat(
            (
                stock_mean,
                stock_std,
                stock_median,
                external_mean,
                external_std,
                horizon_state,
            ),
            dim=-1,
        )
        hidden = self.trunk(pooled)
        return (
            self.component_head(hidden),
            self.energy_head(hidden).squeeze(-1),
            self.event_head(hidden),
        )


class DirectSystemicTransitionHead(nn.Module):
    """Same-output comparator using only causal observable graph summaries."""

    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        *,
        hidden_dim: int = 256,
        horizon_dim: int = 16,
        dropout: float = 0.10,
        component_count: int = len(SYSTEMIC_COMPONENT_TARGETS),
        event_count: int = len(SYSTEMIC_EVENT_TARGETS),
    ) -> None:
        super().__init__()
        if input_dim < 1 or hidden_dim < 1 or horizon_dim < 1:
            raise ValueError("direct systemic head dimensions must be positive")
        normalized_horizons = tuple(int(value) for value in horizons)
        if not normalized_horizons or len(set(normalized_horizons)) != len(
            normalized_horizons
        ):
            raise ValueError("horizons must be unique and non-empty")
        self.horizon_to_index = {
            horizon: index for index, horizon in enumerate(normalized_horizons)
        }
        self.input_projector = nn.Sequential(
            nn.LayerNorm(int(input_dim)),
            nn.Linear(int(input_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
        )
        self.horizon_embedding = nn.Embedding(len(normalized_horizons), int(horizon_dim))
        self.trunk = nn.Sequential(
            nn.Linear(int(hidden_dim) + int(horizon_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.component_head = nn.Linear(int(hidden_dim), int(component_count))
        self.energy_head = nn.Linear(int(hidden_dim), 1)
        self.event_head = nn.Linear(int(hidden_dim), int(event_count))

    def forward(
        self, values: torch.Tensor, horizon: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if values.ndim != 2:
            raise ValueError("direct systemic input must be a row matrix")
        if int(horizon) not in self.horizon_to_index:
            raise ValueError(f"unsupported horizon: {horizon}")
        hidden = self.input_projector(values)
        horizon_index = torch.full(
            (len(values),),
            self.horizon_to_index[int(horizon)],
            dtype=torch.long,
            device=values.device,
        )
        hidden = self.trunk(
            torch.cat((hidden, self.horizon_embedding(horizon_index)), dim=-1)
        )
        return (
            self.component_head(hidden),
            self.energy_head(hidden).squeeze(-1),
            self.event_head(hidden),
        )


class CausalMemorySystemicTransitionHead(nn.Module):
    """Robust latent readout fused with causal observable and residual memory."""

    def __init__(
        self,
        latent_dim: int,
        auxiliary_dim: int,
        horizons: Sequence[int],
        *,
        projection_dim: int = 128,
        auxiliary_projection_dim: int = 128,
        hidden_dim: int = 384,
        horizon_dim: int = 16,
        dropout: float = 0.10,
        component_count: int = len(SYSTEMIC_COMPONENT_TARGETS),
        event_count: int = len(SYSTEMIC_EVENT_TARGETS),
    ) -> None:
        super().__init__()
        dimensions = (
            latent_dim,
            auxiliary_dim,
            projection_dim,
            auxiliary_projection_dim,
            hidden_dim,
            horizon_dim,
        )
        if any(int(value) < 1 for value in dimensions):
            raise ValueError("causal memory systemic head dimensions must be positive")
        normalized_horizons = tuple(int(value) for value in horizons)
        if not normalized_horizons or len(set(normalized_horizons)) != len(
            normalized_horizons
        ):
            raise ValueError("horizons must be unique and non-empty")
        self.latent_dim = int(latent_dim)
        self.auxiliary_dim = int(auxiliary_dim)
        self.horizons = normalized_horizons
        self.horizon_to_index = {
            horizon: index for index, horizon in enumerate(normalized_horizons)
        }
        self.node_projector = nn.Sequential(
            nn.LayerNorm(2 * int(latent_dim)),
            nn.Linear(2 * int(latent_dim), int(projection_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(projection_dim)),
            nn.Dropout(float(dropout)),
        )
        self.stock_attention = nn.Linear(int(projection_dim), 1)
        self.external_attention = nn.Linear(int(projection_dim), 1)
        self.auxiliary_projector = nn.Sequential(
            nn.LayerNorm(int(auxiliary_dim)),
            nn.Linear(int(auxiliary_dim), int(auxiliary_projection_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(auxiliary_projection_dim)),
            nn.Dropout(float(dropout)),
        )
        self.horizon_embedding = nn.Embedding(len(normalized_horizons), int(horizon_dim))
        pooled_dim = (
            10 * int(projection_dim)
            + int(auxiliary_projection_dim)
            + int(horizon_dim)
        )
        self.trunk = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, int(hidden_dim)),
            nn.SiLU(),
            nn.LayerNorm(int(hidden_dim)),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
        )
        self.component_heads = nn.ModuleDict(
            {
                str(horizon): nn.Linear(int(hidden_dim), int(component_count))
                for horizon in normalized_horizons
            }
        )
        self.energy_heads = nn.ModuleDict(
            {
                str(horizon): nn.Linear(int(hidden_dim), 1)
                for horizon in normalized_horizons
            }
        )
        self.event_heads = nn.ModuleDict(
            {
                str(horizon): nn.Linear(int(hidden_dim), int(event_count))
                for horizon in normalized_horizons
            }
        )
        self.direction_heads = nn.ModuleDict(
            {
                str(horizon): nn.Linear(int(hidden_dim), 1)
                for horizon in normalized_horizons
            }
        )

    @staticmethod
    def _robust_pool(values: torch.Tensor, attention: nn.Linear) -> torch.Tensor:
        if values.ndim != 3 or values.shape[1] < 1:
            raise ValueError("robust pooling requires at least one node")
        mean = values.mean(dim=1)
        std = values.std(dim=1, unbiased=False)
        median = values.median(dim=1).values
        q10 = torch.quantile(values, 0.10, dim=1)
        q90 = torch.quantile(values, 0.90, dim=1)
        logits = attention(values).squeeze(-1).clamp(-8.0, 8.0)
        weights = torch.softmax(logits, dim=1)
        attended = (values * weights.unsqueeze(-1)).sum(dim=1)
        return torch.cat((mean, std, median, q10, q90, attended), dim=-1)

    def forward(
        self,
        context: torch.Tensor,
        predicted: torch.Tensor,
        auxiliary: torch.Tensor,
        *,
        batch_size: int,
        node_count: int,
        stock_count: int,
        horizon: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.shape != predicted.shape or context.ndim != 2:
            raise ValueError("context and predicted must be aligned node matrices")
        if context.shape[1] != self.latent_dim:
            raise ValueError("latent width does not match the causal memory head")
        if int(batch_size) * int(node_count) != context.shape[0]:
            raise ValueError("batch and node dimensions do not match latent rows")
        if auxiliary.shape != (int(batch_size), self.auxiliary_dim):
            raise ValueError("auxiliary matrix does not match the configured width")
        if not 0 < int(stock_count) <= int(node_count):
            raise ValueError("stock_count must be within each graph")
        if int(horizon) not in self.horizon_to_index:
            raise ValueError(f"unsupported horizon: {horizon}")

        node_input = torch.cat((context, predicted - context), dim=-1)
        projected = self.node_projector(node_input).reshape(
            int(batch_size), int(node_count), -1
        )
        stock = projected[:, : int(stock_count)]
        stock_state = self._robust_pool(stock, self.stock_attention)
        external = projected[:, int(stock_count) :]
        if external.shape[1]:
            external_mean = external.mean(dim=1)
            external_std = external.std(dim=1, unbiased=False)
            external_median = external.median(dim=1).values
            external_logits = self.external_attention(external).squeeze(-1).clamp(-8.0, 8.0)
            external_weights = torch.softmax(external_logits, dim=1)
            external_attended = (
                external * external_weights.unsqueeze(-1)
            ).sum(dim=1)
        else:
            width = projected.shape[-1]
            external_mean = projected.new_zeros((int(batch_size), width))
            external_std = torch.zeros_like(external_mean)
            external_median = torch.zeros_like(external_mean)
            external_attended = torch.zeros_like(external_mean)
        auxiliary_state = self.auxiliary_projector(auxiliary)
        horizon_index = torch.full(
            (int(batch_size),),
            self.horizon_to_index[int(horizon)],
            dtype=torch.long,
            device=context.device,
        )
        pooled = torch.cat(
            (
                stock_state,
                external_mean,
                external_std,
                external_median,
                external_attended,
                auxiliary_state,
                self.horizon_embedding(horizon_index),
            ),
            dim=-1,
        )
        hidden = self.trunk(pooled)
        key = str(int(horizon))
        return (
            self.component_heads[key](hidden),
            self.energy_heads[key](hidden).squeeze(-1),
            self.event_heads[key](hidden),
            self.direction_heads[key](hidden).squeeze(-1),
        )


def weighted_smooth_l1_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    sample_weight: torch.Tensor,
) -> torch.Tensor:
    if prediction.shape != target.shape:
        raise ValueError("prediction and target shapes must match")
    if sample_weight.shape != prediction.shape[:1]:
        raise ValueError("sample_weight must contain one value per row")
    loss = F.smooth_l1_loss(prediction, target, reduction="none")
    if loss.ndim > 1:
        loss = loss.mean(dim=tuple(range(1, loss.ndim)))
    weight = sample_weight.to(dtype=loss.dtype).clamp_min(0.0)
    return (loss * weight).sum() / weight.sum().clamp_min(1e-8)


def correlation_rank_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    valid = torch.isfinite(prediction) & torch.isfinite(target)
    if int(valid.sum()) < 3:
        return prediction.new_tensor(0.0)
    prediction = prediction[valid]
    target = target[valid]
    prediction = prediction - prediction.mean()
    target = target - target.mean()
    prediction_scale = torch.sqrt(prediction.square().sum() + 1e-8)
    target_scale = torch.sqrt(target.square().sum() + 1e-8)
    denominator = prediction_scale * target_scale
    return 1.0 - (prediction * target).sum() / denominator


def focal_binary_loss(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    alpha: float = 0.75,
    gamma: float = 2.0,
) -> torch.Tensor:
    if logits.shape != labels.shape:
        raise ValueError("logit and label shapes must match")
    labels = labels.to(dtype=logits.dtype)
    probability = torch.sigmoid(logits)
    target_probability = torch.where(labels > 0.5, probability, 1.0 - probability)
    alpha_factor = torch.where(
        labels > 0.5,
        torch.full_like(labels, float(alpha)),
        torch.full_like(labels, 1.0 - float(alpha)),
    )
    cross_entropy = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    return (
        alpha_factor * (1.0 - target_probability).pow(float(gamma)) * cross_entropy
    ).mean()
