from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping, Sequence

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from stock_v2.intraday_trajectory import INTRADAY_TRAJECTORY_TARGET_NAMES


MODEL_VARIANTS = ("direct", "state", "latent")
GRAPH_MESSAGE_FUSIONS = ("shared", "long_horizon_residual")


@dataclass(frozen=True)
class RobustArrayScaler:
    center: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        center = np.asarray(self.center, dtype=np.float64)
        scale = np.asarray(self.scale, dtype=np.float64)
        if center.ndim != 1 or center.shape != scale.shape:
            raise ValueError("robust scaler center and scale must be aligned vectors")
        if not np.isfinite(center).all() or not np.isfinite(scale).all():
            raise ValueError("robust scaler values must be finite")
        if (scale <= 0.0).any():
            raise ValueError("robust scaler scales must be positive")


@dataclass(frozen=True)
class PostImpactPrediction:
    node: torch.Tensor
    systemic: torch.Tensor
    node_presence: torch.Tensor


@dataclass
class RegressionMetricAccumulator:
    count: int = 0
    absolute_error_sum: float = 0.0
    squared_error_sum: float = 0.0
    target_squared_sum: float = 0.0
    prediction_sum: float = 0.0
    target_sum: float = 0.0
    prediction_squared_sum: float = 0.0
    cross_sum: float = 0.0
    direction_count: int = 0
    direction_correct: int = 0

    def update(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
        selected: np.ndarray | None = None,
    ) -> None:
        prediction = np.asarray(prediction)
        target = np.asarray(target)
        if prediction.shape != target.shape:
            raise ValueError("regression metric arrays must be aligned")
        valid = np.isfinite(prediction) & np.isfinite(target)
        if selected is not None:
            selected = np.asarray(selected, dtype=bool)
            if selected.shape != prediction.shape:
                raise ValueError("regression metric selection mask is not aligned")
            valid &= selected
        if not valid.any():
            return
        predicted = prediction[valid].astype(np.float64)
        realized = target[valid].astype(np.float64)
        residual = predicted - realized
        self.count += int(len(realized))
        self.absolute_error_sum += float(np.abs(residual).sum())
        self.squared_error_sum += float(np.square(residual).sum())
        self.target_squared_sum += float(np.square(realized).sum())
        self.prediction_sum += float(predicted.sum())
        self.target_sum += float(realized.sum())
        self.prediction_squared_sum += float(np.square(predicted).sum())
        self.cross_sum += float((predicted * realized).sum())
        nonzero = realized != 0.0
        self.direction_count += int(nonzero.sum())
        self.direction_correct += int(
            (np.sign(predicted[nonzero]) == np.sign(realized[nonzero])).sum()
        )

    def metrics(self) -> dict[str, float | int]:
        if self.count < 3:
            return {
                "count": int(self.count),
                "mae": float("nan"),
                "mse": float("nan"),
                "pearson": float("nan"),
                "skill_vs_zero_mse": float("nan"),
                "direction_accuracy": float("nan"),
            }
        count = float(self.count)
        prediction_variation = (
            self.prediction_squared_sum - self.prediction_sum**2 / count
        )
        target_variation = (
            self.target_squared_sum - self.target_sum**2 / count
        )
        covariance = self.cross_sum - self.prediction_sum * self.target_sum / count
        pearson = float("nan")
        if prediction_variation > count * 1e-24 and target_variation > count * 1e-24:
            pearson = covariance / math.sqrt(
                prediction_variation * target_variation
            )
            pearson = float(np.clip(pearson, -1.0, 1.0))
        mse = self.squared_error_sum / count
        skill = (
            1.0 - self.squared_error_sum / self.target_squared_sum
            if self.target_squared_sum > 1e-12
            else float("nan")
        )
        direction_accuracy = (
            self.direction_correct / self.direction_count
            if self.direction_count
            else float("nan")
        )
        return {
            "count": int(self.count),
            "mae": self.absolute_error_sum / count,
            "mse": mse,
            "pearson": pearson,
            "skill_vs_zero_mse": skill,
            "direction_accuracy": direction_accuracy,
        }


def fit_robust_array_scaler(
    values: np.ndarray,
    available: np.ndarray,
    *,
    rows: Sequence[int] | None = None,
    minimum_count: int = 20,
    minimum_scale: float = 1e-6,
) -> RobustArrayScaler:
    values = np.asarray(values, dtype=np.float64)
    available = np.asarray(available, dtype=bool)
    if values.shape != available.shape or values.ndim < 2:
        raise ValueError("robust scaler inputs must be aligned arrays with a feature axis")
    selected = values if rows is None else values[np.asarray(rows, dtype=np.int64)]
    selected_available = (
        available if rows is None else available[np.asarray(rows, dtype=np.int64)]
    )
    width = values.shape[-1]
    center = np.zeros(width, dtype=np.float64)
    scale = np.ones(width, dtype=np.float64)
    for feature in range(width):
        valid = selected_available[..., feature] & np.isfinite(selected[..., feature])
        observed = selected[..., feature][valid]
        if len(observed) < int(minimum_count):
            continue
        location = float(np.median(observed))
        dispersion = float(1.4826 * np.median(np.abs(observed - location)))
        if not np.isfinite(dispersion) or dispersion < float(minimum_scale):
            dispersion = float(np.std(observed))
        center[feature] = location
        scale[feature] = max(dispersion, float(minimum_scale))
    return RobustArrayScaler(center=center, scale=scale)


def normalize_with_mask(
    values: torch.Tensor,
    available: torch.Tensor,
    center: torch.Tensor,
    scale: torch.Tensor,
    *,
    clip: float = 10.0,
) -> torch.Tensor:
    if values.shape != available.shape:
        raise ValueError("normalization values and masks must be aligned")
    if center.shape != scale.shape or center.shape != (values.shape[-1],):
        raise ValueError("normalization vectors do not match the feature width")
    valid = available.bool() & torch.isfinite(values)
    normalized = ((values - center) / scale.clamp_min(1e-8)).clamp(
        -float(clip), float(clip)
    )
    return torch.where(valid, normalized, torch.zeros_like(normalized))


class CausalPostImpactReforecast(nn.Module):
    """Reforecast post-shock paths without reading future intraday states."""

    def __init__(
        self,
        *,
        node_feature_dim: int,
        stale_state_dim: int,
        latent_dim: int,
        horizons: Sequence[str],
        systemic_target_dim: int,
        variant: str = "latent",
        hidden_dim: int = 192,
        latent_projection_dim: int = 192,
        temporal_layers: int = 2,
        dropout: float = 0.10,
        surprise_dim: int = 0,
        graph_message_dim: int = 0,
        graph_message_fusion: str = "shared",
        target_names: Sequence[str] = INTRADAY_TRAJECTORY_TARGET_NAMES,
        output_scales: Sequence[float] | None = None,
    ) -> None:
        super().__init__()
        if variant not in MODEL_VARIANTS:
            raise ValueError(f"variant must be one of {MODEL_VARIANTS}")
        if int(node_feature_dim) <= 0 or int(hidden_dim) <= 0:
            raise ValueError("node and hidden dimensions must be positive")
        if variant in {"state", "latent"} and int(stale_state_dim) <= 0:
            raise ValueError("state and latent variants require stale state features")
        if variant == "latent" and int(latent_dim) <= 0:
            raise ValueError("latent variant requires a positive latent dimension")
        self.variant = variant
        self.horizons = tuple(str(value) for value in horizons)
        self.target_names = tuple(str(value) for value in target_names)
        if not self.horizons or not self.target_names:
            raise ValueError("post-impact model requires horizons and targets")
        required_targets = set(INTRADAY_TRAJECTORY_TARGET_NAMES)
        if not required_targets.issubset(self.target_names):
            raise ValueError("post-impact target contract is missing required path labels")
        self.node_feature_dim = int(node_feature_dim)
        self.stale_state_dim = int(stale_state_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.surprise_dim = int(surprise_dim)
        self.graph_message_dim = int(graph_message_dim)
        if self.graph_message_dim < 0:
            raise ValueError("graph message dimension must be non-negative")
        if str(graph_message_fusion) not in GRAPH_MESSAGE_FUSIONS:
            raise ValueError(
                f"graph message fusion must be one of {GRAPH_MESSAGE_FUSIONS}"
            )
        self.graph_message_fusion = str(graph_message_fusion)
        if (
            self.graph_message_fusion == "long_horizon_residual"
            and self.graph_message_dim <= 0
        ):
            raise ValueError(
                "long-horizon residual fusion requires graph message inputs"
            )
        if self.graph_message_fusion == "long_horizon_residual" and (
            "5m" not in self.horizons
            or not any(label != "5m" for label in self.horizons)
        ):
            raise ValueError(
                "long-horizon residual fusion requires protected 5m and longer horizons"
            )
        scales = np.ones(len(self.target_names), dtype=np.float32)
        if output_scales is not None:
            scales = np.asarray(output_scales, dtype=np.float32)
            if scales.shape != (len(self.target_names),):
                raise ValueError("output scales do not match the target contract")
            if not np.isfinite(scales).all() or (scales <= 0.0).any():
                raise ValueError("output scales must be finite and positive")
        self.register_buffer("output_scales", torch.as_tensor(scales))

        daily_width = 0
        if variant in {"state", "latent"}:
            daily_width += self.stale_state_dim
        if variant == "latent":
            self.latent_projector = nn.Sequential(
                nn.LayerNorm(2 * self.latent_dim),
                nn.Linear(2 * self.latent_dim, int(latent_projection_dim)),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            )
            daily_width += int(latent_projection_dim)
        else:
            self.latent_projector = None

        shared_message_width = (
            2 * self.graph_message_dim
            if self.graph_message_fusion == "shared"
            else 0
        )
        node_input_width = 2 * self.node_feature_dim + shared_message_width + daily_width
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(node_input_width),
            nn.Linear(node_input_width, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.temporal = nn.GRU(
            input_size=self.hidden_dim,
            hidden_size=self.hidden_dim,
            num_layers=int(temporal_layers),
            batch_first=True,
            dropout=float(dropout) if int(temporal_layers) > 1 else 0.0,
            bidirectional=False,
        )
        market_input_width = 2 * self.hidden_dim + self.surprise_dim
        self.market_encoder = nn.Sequential(
            nn.LayerNorm(market_input_width),
            nn.Linear(market_input_width, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.node_fusion = nn.Sequential(
            nn.LayerNorm(2 * self.hidden_dim),
            nn.Linear(2 * self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
        )
        self.node_head = nn.Linear(
            self.hidden_dim, len(self.horizons) * len(self.target_names)
        )
        self.systemic_head = nn.Linear(
            self.hidden_dim, len(self.horizons) * int(systemic_target_dim)
        )
        if self.graph_message_fusion == "long_horizon_residual":
            self.message_encoder = nn.Sequential(
                nn.LayerNorm(2 * self.graph_message_dim),
                nn.Linear(2 * self.graph_message_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            )
            self.message_temporal = nn.GRU(
                input_size=self.hidden_dim,
                hidden_size=self.hidden_dim,
                num_layers=1,
                batch_first=True,
                bidirectional=False,
            )
            self.message_fusion = nn.Sequential(
                nn.LayerNorm(2 * self.hidden_dim),
                nn.Linear(2 * self.hidden_dim, self.hidden_dim),
                nn.SiLU(),
                nn.Dropout(float(dropout)),
            )
            self.message_node_head = nn.Linear(
                self.hidden_dim, len(self.horizons) * len(self.target_names)
            )
            nn.init.zeros_(self.message_node_head.weight)
            nn.init.zeros_(self.message_node_head.bias)
            self.register_buffer(
                "message_horizon_mask",
                torch.as_tensor(
                    [0.0 if label == "5m" else 1.0 for label in self.horizons],
                    dtype=torch.float32,
                ),
            )
        else:
            self.message_encoder = None
            self.message_temporal = None
            self.message_fusion = None
            self.message_node_head = None

    def _daily_context(
        self,
        stale_state: torch.Tensor | None,
        context_latent: torch.Tensor | None,
        predicted_delta: torch.Tensor | None,
    ) -> torch.Tensor | None:
        blocks: list[torch.Tensor] = []
        if self.variant in {"state", "latent"}:
            if stale_state is None or stale_state.ndim != 3:
                raise ValueError("state and latent variants require [batch,node,state]")
            if stale_state.shape[-1] != self.stale_state_dim:
                raise ValueError("stale state width does not match the model")
            blocks.append(stale_state)
        if self.variant == "latent":
            if context_latent is None or predicted_delta is None:
                raise ValueError("latent variant requires context and predicted delta")
            if context_latent.shape != predicted_delta.shape or context_latent.ndim != 3:
                raise ValueError("daily latent arrays must be aligned [batch,node,latent]")
            if context_latent.shape[-1] != self.latent_dim:
                raise ValueError("daily latent width does not match the model")
            blocks.append(
                self.latent_projector(torch.cat((context_latent, predicted_delta), dim=-1))
            )
        return torch.cat(blocks, dim=-1) if blocks else None

    def _physical_node_targets(self, raw: torch.Tensor) -> torch.Tensor:
        index = {name: self.target_names.index(name) for name in self.target_names}
        endpoint = (
            raw[..., index["endpoint_return"]]
            * self.output_scales[index["endpoint_return"]]
        )
        transformed = {
            name: raw[..., position] * self.output_scales[position]
            for name, position in index.items()
        }
        transformed["endpoint_return"] = endpoint
        transformed["mfe"] = endpoint + F.softplus(
            raw[..., index["mfe"]]
        ) * self.output_scales[index["mfe"]]
        transformed["mae"] = endpoint - F.softplus(
            raw[..., index["mae"]]
        ) * self.output_scales[index["mae"]]
        minimum_realized_absolute = torch.where(
            endpoint >= 0.0,
            torch.log1p(endpoint.clamp_min(0.0)),
            -endpoint,
        )
        transformed["realized_absolute_return"] = (
            minimum_realized_absolute
            + F.softplus(raw[..., index["realized_absolute_return"]])
            * self.output_scales[index["realized_absolute_return"]]
        )
        transformed["future_range"] = F.softplus(
            raw[..., index["future_range"]]
        ) * self.output_scales[index["future_range"]]
        transformed["time_to_peak_fraction"] = torch.sigmoid(
            raw[..., index["time_to_peak_fraction"]]
        )
        transformed["time_to_trough_fraction"] = torch.sigmoid(
            raw[..., index["time_to_trough_fraction"]]
        )
        return torch.stack(
            [transformed[name] for name in self.target_names], dim=-1
        )

    def forward(
        self,
        node_values: torch.Tensor,
        node_available: torch.Tensor,
        *,
        stale_state: torch.Tensor | None = None,
        context_latent: torch.Tensor | None = None,
        predicted_delta: torch.Tensor | None = None,
        surprise_values: torch.Tensor | None = None,
        graph_neighbor_values: torch.Tensor | None = None,
        graph_neighbor_available: torch.Tensor | None = None,
    ) -> PostImpactPrediction:
        if node_values.ndim != 4 or node_values.shape != node_available.shape:
            raise ValueError("intraday nodes must be aligned [batch,time,node,feature]")
        if node_values.shape[-1] != self.node_feature_dim:
            raise ValueError("intraday node width does not match the model")
        batch, time_count, node_count, _feature_count = node_values.shape
        if self.surprise_dim:
            if surprise_values is None or surprise_values.shape != (
                batch,
                time_count,
                self.surprise_dim,
            ):
                raise ValueError("surprise values do not match [batch,time,surprise]")
        elif surprise_values is not None and surprise_values.shape[:2] != (
            batch,
            time_count,
        ):
            raise ValueError("unexpected surprise time axes")

        valid = node_available.bool() & torch.isfinite(node_values)
        sanitized = torch.where(valid, node_values, torch.zeros_like(node_values))
        node_presence = valid.float().mean(dim=-1) >= 0.25
        blocks = (sanitized, valid.to(sanitized.dtype))
        graph_sanitized: torch.Tensor | None = None
        graph_valid: torch.Tensor | None = None
        if self.graph_message_dim:
            expected = (batch, time_count, node_count, self.graph_message_dim)
            if (
                graph_neighbor_values is None
                or graph_neighbor_available is None
                or graph_neighbor_values.shape != expected
                or graph_neighbor_available.shape != expected
            ):
                raise ValueError(
                    "graph neighbor values and masks must match "
                    "[batch,time,node,graph_message]"
                )
            graph_valid = graph_neighbor_available.bool() & torch.isfinite(
                graph_neighbor_values
            )
            graph_sanitized = torch.where(
                graph_valid,
                graph_neighbor_values,
                torch.zeros_like(graph_neighbor_values),
            )
            if self.graph_message_fusion == "shared":
                blocks = blocks + (
                    graph_sanitized,
                    graph_valid.to(graph_sanitized.dtype),
                )
        elif graph_neighbor_values is not None or graph_neighbor_available is not None:
            raise ValueError("model was constructed without graph message inputs")
        daily = self._daily_context(stale_state, context_latent, predicted_delta)
        if daily is not None:
            if daily.shape[:2] != (batch, node_count):
                raise ValueError("daily context node axes do not match intraday inputs")
            blocks = blocks + (daily[:, None].expand(-1, time_count, -1, -1),)
        encoded = self.node_encoder(torch.cat(blocks, dim=-1))
        temporal_input = encoded.permute(0, 2, 1, 3).reshape(
            batch * node_count, time_count, self.hidden_dim
        )
        temporal, _hidden = self.temporal(temporal_input)
        temporal = temporal.reshape(
            batch, node_count, time_count, self.hidden_dim
        ).permute(0, 2, 1, 3)

        weights = node_presence.to(temporal.dtype)[..., None]
        count = weights.sum(dim=2).clamp_min(1.0)
        market_mean = (temporal * weights).sum(dim=2) / count
        centered = (temporal - market_mean[:, :, None]) * weights
        market_std = torch.sqrt(centered.square().sum(dim=2) / count + 1e-8)
        market_blocks = [market_mean, market_std]
        if self.surprise_dim:
            market_blocks.append(surprise_values)
        market = self.market_encoder(torch.cat(market_blocks, dim=-1))
        fused = self.node_fusion(
            torch.cat(
                (temporal, market[:, :, None].expand(-1, -1, node_count, -1)),
                dim=-1,
            )
        )
        raw_node = self.node_head(fused).reshape(
            batch,
            time_count,
            node_count,
            len(self.horizons),
            len(self.target_names),
        )
        if self.graph_message_fusion == "long_horizon_residual":
            if (
                graph_sanitized is None
                or graph_valid is None
                or self.message_encoder is None
                or self.message_temporal is None
                or self.message_fusion is None
                or self.message_node_head is None
            ):
                raise RuntimeError("long-horizon message adapter is incomplete")
            message_presence = graph_valid.any(dim=-1)
            message_encoded = self.message_encoder(
                torch.cat(
                    (graph_sanitized, graph_valid.to(graph_sanitized.dtype)),
                    dim=-1,
                )
            )
            message_encoded = message_encoded * message_presence[..., None]
            message_temporal_input = message_encoded.permute(0, 2, 1, 3).reshape(
                batch * node_count, time_count, self.hidden_dim
            )
            message_temporal, _message_hidden = self.message_temporal(
                message_temporal_input
            )
            message_temporal = message_temporal.reshape(
                batch, node_count, time_count, self.hidden_dim
            ).permute(0, 2, 1, 3)
            message_history_present = message_presence.to(torch.int32).cumsum(
                dim=1
            ) > 0
            message_fused = self.message_fusion(
                torch.cat((message_temporal, fused.detach()), dim=-1)
            )
            raw_message = self.message_node_head(message_fused).reshape(
                batch,
                time_count,
                node_count,
                len(self.horizons),
                len(self.target_names),
            )
            raw_message = raw_message * message_history_present[
                ..., None, None
            ].to(raw_message.dtype)
            raw_node = raw_node + raw_message * self.message_horizon_mask[
                None, None, None, :, None
            ].to(raw_message.dtype)
        node = self._physical_node_targets(raw_node)
        systemic = self.systemic_head(market).reshape(
            batch, time_count, len(self.horizons), -1
        )
        return PostImpactPrediction(
            node=node,
            systemic=systemic,
            node_presence=node_presence,
        )


def impact_weighted_multitask_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
    *,
    target_scale: torch.Tensor,
    horizon_weights: torch.Tensor | None = None,
    event_weights: torch.Tensor | None = None,
    target_weights: Mapping[str, float] | None = None,
    target_names: Sequence[str] = INTRADAY_TRAJECTORY_TARGET_NAMES,
    huber_delta: float = 1.0,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if prediction.shape != target.shape or prediction.shape != available.shape:
        raise ValueError("multitask prediction, target, and masks must be aligned")
    if prediction.ndim != 5:
        raise ValueError("node multitask arrays must be [batch,time,node,horizon,target]")
    if target_scale.shape != (prediction.shape[-1],):
        raise ValueError("target scales do not match the target width")
    valid = available.bool() & torch.isfinite(prediction) & torch.isfinite(target)
    residual = torch.where(
        valid,
        (prediction - target) / target_scale.clamp_min(1e-8),
        torch.zeros_like(prediction),
    )
    absolute = residual.abs()
    delta = float(huber_delta)
    loss = torch.where(
        absolute <= delta,
        0.5 * residual.square(),
        delta * (absolute - 0.5 * delta),
    )
    loss = torch.where(valid, loss, torch.zeros_like(loss))
    weights = torch.ones_like(loss)
    if horizon_weights is not None:
        if horizon_weights.shape != (prediction.shape[-2],):
            raise ValueError("horizon weights do not match the horizon width")
        weights = weights * horizon_weights[None, None, None, :, None]
    if event_weights is not None:
        if event_weights.shape == prediction.shape[:2]:
            weights = weights * event_weights[:, :, None, None, None]
        elif event_weights.shape == (
            prediction.shape[0],
            prediction.shape[1],
            prediction.shape[3],
        ):
            weights = weights * event_weights[:, :, None, :, None]
        else:
            raise ValueError(
                "event weights must match [batch,time] or [batch,time,horizon]"
            )
    names = tuple(str(name) for name in target_names)
    if len(names) != prediction.shape[-1]:
        raise ValueError("target names do not match the target width")
    configured = dict(target_weights or {})
    per_target = torch.as_tensor(
        [float(configured.get(name, 1.0)) for name in names],
        dtype=prediction.dtype,
        device=prediction.device,
    )
    weights = weights * per_target[None, None, None, None, :]
    selected_weights = torch.where(valid, weights, torch.zeros_like(weights))
    total = (loss * selected_weights).sum() / selected_weights.sum().clamp_min(1.0)
    components: dict[str, torch.Tensor] = {}
    for index, name in enumerate(names):
        selected = valid[..., index]
        component_weight = selected_weights[..., index]
        components[name] = (
            (loss[..., index] * component_weight).sum()
            / component_weight.sum().clamp_min(1.0)
            if selected.any()
            else prediction.new_tensor(0.0)
        )
    return total, components


def grouped_node_correlation_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    available: torch.Tensor,
    event_mask: torch.Tensor,
    horizon_mask: torch.Tensor,
    *,
    minimum_nodes: int = 100,
) -> torch.Tensor:
    """Average cross-sectional correlation loss within causal time/horizon groups."""
    if prediction.shape != target.shape or prediction.shape != available.shape:
        raise ValueError("node correlation arrays must be aligned")
    if prediction.ndim != 4:
        raise ValueError("node correlation arrays must be [batch,time,node,horizon]")
    if event_mask.shape != prediction.shape[:2]:
        raise ValueError("event mask must match the batch and time axes")
    if horizon_mask.shape != (prediction.shape[-1],):
        raise ValueError("horizon mask must match the horizon axis")
    if int(minimum_nodes) < 3:
        raise ValueError("node correlation groups require at least three nodes")

    selected = (
        available.bool()
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & event_mask.bool()[:, :, None, None]
        & horizon_mask.bool()[None, None, None, :]
    )
    if not selected.any():
        return prediction.new_tensor(0.0)

    batch, time_count, node_count, horizon_count = prediction.shape
    group_count = batch * time_count * horizon_count
    groups = torch.arange(
        group_count, device=prediction.device, dtype=torch.long
    ).reshape(batch, time_count, 1, horizon_count)
    groups = groups.expand(-1, -1, node_count, -1)[selected]
    compute_dtype = torch.promote_types(prediction.dtype, target.dtype)
    if compute_dtype in {torch.float16, torch.bfloat16}:
        compute_dtype = torch.float32
    predicted = prediction[selected].to(dtype=compute_dtype)
    observed = target[selected].to(dtype=compute_dtype)
    counts = torch.bincount(groups, minlength=group_count).to(compute_dtype)
    prediction_sum = torch.zeros(
        group_count, dtype=compute_dtype, device=prediction.device
    ).index_add_(0, groups, predicted)
    target_sum = torch.zeros_like(prediction_sum).index_add_(0, groups, observed)
    prediction_square_sum = torch.zeros_like(prediction_sum).index_add_(
        0, groups, predicted.square()
    )
    target_square_sum = torch.zeros_like(prediction_sum).index_add_(
        0, groups, observed.square()
    )
    cross_sum = torch.zeros_like(prediction_sum).index_add_(
        0, groups, predicted * observed
    )
    safe_counts = counts.clamp_min(1.0)
    covariance = cross_sum - prediction_sum * target_sum / safe_counts
    prediction_variance = (
        prediction_square_sum - prediction_sum.square() / safe_counts
    ).clamp_min(0.0)
    target_variance = (
        target_square_sum - target_sum.square() / safe_counts
    ).clamp_min(0.0)
    usable = (counts >= float(minimum_nodes)) & (target_variance > 1e-8)
    if not usable.any():
        return prediction.new_tensor(0.0)
    denominator = torch.sqrt(
        (prediction_variance[usable] + 1e-8)
        * (target_variance[usable] + 1e-8)
    )
    correlation = covariance[usable] / denominator.clamp_min(1e-8)
    return 1.0 - correlation.clamp(-1.0, 1.0).mean()
