from __future__ import annotations

from typing import Mapping, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from stock_v2.market_transition_head import (
    DirectMarketTrajectoryHead,
    MarketTrajectoryHead,
)
from stock_v2.systemic_head import correlation_rank_loss, focal_binary_loss


SEPARATED_MAJOR_OBJECTIVE_VERSION = "separated_major_path_v32_20260714"


SEPARATED_MAJOR_LOSS_WEIGHTS = {
    "components": 0.15,
    "families": 0.25,
    "family_rank": 0.10,
    "events": 0.10,
    "trajectory": 0.10,
    "horizon_salience": 0.075,
    "major_rank": 0.075,
    "major_focal": 0.075,
    "peak_horizon": 0.075,
}


class _SeparatedMajorReadout(nn.Module):
    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.horizon_salience_head = nn.Linear(int(hidden_dim), 1)
        self.peak_head = nn.Linear(int(hidden_dim), 1)
        self.major_event_head = nn.Sequential(
            nn.LayerNorm(2 * int(hidden_dim)),
            nn.Linear(2 * int(hidden_dim), int(hidden_dim)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_dim), 1),
        )

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        if hidden.ndim != 3:
            raise ValueError("major readout requires batch-by-horizon hidden states")
        return {
            "horizon_log_salience": F.softplus(
                self.horizon_salience_head(hidden).squeeze(-1)
            ),
            "major_logit": self.major_event_head(
                torch.cat((hidden.mean(dim=1), hidden.amax(dim=1)), dim=-1)
            ).squeeze(-1),
            "peak_logits": self.peak_head(hidden).squeeze(-1),
        }


class SeparatedMajorMarketTrajectoryHead(nn.Module):
    """Joint family forecasts plus independent path-event and timing heads."""

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
    ) -> None:
        super().__init__()
        self.base = MarketTrajectoryHead(
            latent_dim,
            horizons,
            projection_dim=int(projection_dim),
            hidden_dim=int(hidden_dim),
            layers=int(layers),
            heads=int(heads),
            dropout=float(dropout),
        )
        self.major = _SeparatedMajorReadout(int(hidden_dim), float(dropout))

    @property
    def horizons(self) -> tuple[int, ...]:
        return self.base.horizons

    def forward(
        self,
        context: torch.Tensor,
        predicted: Mapping[int, torch.Tensor],
        *,
        batch_size: int,
        node_count: int,
        stock_count: int,
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        pooled = torch.stack(
            [
                self.base._pool(
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
        hidden = self.base.trajectory_encoder(pooled)
        base = (
            self.base.component_head(hidden),
            self.base.family_head(hidden),
            self.base.event_head(hidden),
        )
        return base, self.major(hidden)


class SeparatedMajorDirectTrajectoryHead(nn.Module):
    """Same separated outputs from robust causal snapshot summaries."""

    def __init__(
        self,
        input_dim: int,
        horizons: Sequence[int],
        *,
        hidden_dim: int = 256,
        layers: int = 2,
        heads: int = 8,
        dropout: float = 0.10,
    ) -> None:
        super().__init__()
        self.base = DirectMarketTrajectoryHead(
            int(input_dim),
            horizons,
            hidden_dim=int(hidden_dim),
            layers=int(layers),
            heads=int(heads),
            dropout=float(dropout),
        )
        self.major = _SeparatedMajorReadout(int(hidden_dim), float(dropout))

    @property
    def horizons(self) -> tuple[int, ...]:
        return self.base.horizons

    def forward(
        self, values: torch.Tensor
    ) -> tuple[
        tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        dict[str, torch.Tensor],
    ]:
        if values.ndim != 2:
            raise ValueError("direct market input must be a row matrix")
        sequence = values[:, None, :].expand(-1, len(self.horizons), -1)
        hidden = self.base.trajectory_encoder(sequence)
        base = (
            self.base.component_head(hidden),
            self.base.family_head(hidden),
            self.base.event_head(hidden),
        )
        return base, self.major(hidden)


class BaseOutputView(nn.Module):
    """Expose a separated head through the existing three-output evaluator."""

    def __init__(self, head: nn.Module) -> None:
        super().__init__()
        self.head = head

    def forward(self, *args, **kwargs):
        base, _major = self.head(*args, **kwargs)
        return base


def separated_major_loss_terms(
    output: Mapping[str, torch.Tensor],
    target: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    predicted_horizon_log = output["horizon_log_salience"]
    target_horizon_log = torch.log1p(target["horizon_salience"].clamp_min(0.0))
    if predicted_horizon_log.shape != target_horizon_log.shape:
        raise ValueError("predicted and target horizon salience must align")
    path_log = predicted_horizon_log.amax(dim=1)
    target_path_log = torch.log1p(target["path_salience"].clamp_min(0.0))
    major = target["major_label"] > 0.5
    return {
        "horizon_salience": F.smooth_l1_loss(
            predicted_horizon_log, target_horizon_log
        ),
        "major_rank": correlation_rank_loss(path_log, target_path_log),
        "major_focal": focal_binary_loss(
            output["major_logit"], target["major_label"]
        ),
        "peak_horizon": (
            F.cross_entropy(
                output["peak_logits"][major],
                target["peak_horizon_index"][major],
            )
            if major.any()
            else path_log.new_tensor(0.0)
        ),
    }
