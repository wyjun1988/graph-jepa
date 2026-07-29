from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Dict, Optional, Sequence, Tuple

import torch
from torch.utils.checkpoint import checkpoint
from torch import Tensor, nn
import torch.nn.functional as F


PATH_RETURN_TASK_INDEX = 0

from stock_v2.plan_timing import plan_timing_loss

DOWNSTREAM_AUXILIARY_TASKS = (
    "path_return",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "realized_volatility",
    # Intent 2: |close(t+h)/close(t) - 1| / h -- the node's future move per day.
    # Paired at scoring time with the observed |return_1d(t)| to form the frozen
    # continuation contract's ratio score, which is what keeps the question
    # "did THIS shock continue" rather than "is this stock volatile".
    "continuation_rate",
)


MARKET_TRANSITION_AUXILIARY_FAMILIES = (
    "price_co_movement",
    "market_activity",
    "node_state",
    "topology",
)
MARKET_TRANSITION_AUXILIARY_WIDTH = (
    2 * len(MARKET_TRANSITION_AUXILIARY_FAMILIES) + 2
)


HIDDEN_COMPLETION_CHANNELS = (
    "investor_foreign_flow_ratio_1d",
    "investor_institution_flow_ratio_1d",
)
HIDDEN_COMPLETION_WIDTH = len(HIDDEN_COMPLETION_CHANNELS)


@dataclass
class GraphBatch:
    """One graph snapshot with feature-level observation masks.

    node_features are expected to be normalized before training. feature_mask is
    1 for observed features and 0 for features hidden from the context encoder.
    """

    node_features: Tensor
    feature_mask: Tensor
    edge_index: Tensor
    edge_weight: Optional[Tensor] = None
    available_mask: Optional[Tensor] = None
    supervision_node_mask: Optional[Tensor] = None
    graph_index: Optional[Tensor] = None
    target_entry_path: Optional[Tensor] = None
    target_downstream: Optional[Tensor] = None
    target_downstream_scale: Optional[Tensor] = None
    target_downstream_causal_scale: Optional[Tensor] = None
    target_market_transition: Optional[Tensor] = None
    # t's structurally-hidden flow, disclosed at t+1. Attached to the CONTEXT
    # batch only; None everywhere else. NaN marks a node with no valid target.
    hidden_target: Optional[Tensor] = None
    # (nodes, window, features) trailing window; None keeps the single-step path.
    node_sequence: Optional[Tensor] = None

    def to(self, device: torch.device | str) -> "GraphBatch":
        return GraphBatch(
            node_features=self.node_features.to(device),
            feature_mask=self.feature_mask.to(device),
            edge_index=self.edge_index.to(device),
            edge_weight=None if self.edge_weight is None else self.edge_weight.to(device),
            available_mask=None if self.available_mask is None else self.available_mask.to(device),
            supervision_node_mask=(
                None if self.supervision_node_mask is None else self.supervision_node_mask.to(device)
            ),
            graph_index=None if self.graph_index is None else self.graph_index.to(device),
            node_sequence=None if self.node_sequence is None else self.node_sequence.to(device),
            target_entry_path=(
                None
                if self.target_entry_path is None
                else self.target_entry_path.to(device)
            ),
            target_downstream=(
                None
                if self.target_downstream is None
                else self.target_downstream.to(device)
            ),
            target_downstream_scale=(
                None
                if self.target_downstream_scale is None
                else self.target_downstream_scale.to(device)
            ),
            target_downstream_causal_scale=(
                None
                if self.target_downstream_causal_scale is None
                else self.target_downstream_causal_scale.to(device)
            ),
            target_market_transition=(
                None
                if self.target_market_transition is None
                else self.target_market_transition.to(device)
            ),
            hidden_target=(
                None if self.hidden_target is None else self.hidden_target.to(device)
            ),
        )


def merge_graph_batches(batches: Sequence[GraphBatch]) -> GraphBatch:
    """Combine disjoint graph snapshots so they can share one model forward pass."""
    if not batches:
        raise ValueError("batches must not be empty")

    first = batches[0]
    feature_dim = first.node_features.shape[1]
    if any(batch.node_features.shape[1] != feature_dim for batch in batches):
        raise ValueError("all graph batches must have the same feature width")

    node_features = []
    feature_masks = []
    available_masks = []
    supervision_masks = []
    graph_indices = []
    edge_indices = []
    edge_weights = []
    use_available_mask = any(batch.available_mask is not None for batch in batches)
    use_supervision_mask = any(batch.supervision_node_mask is not None for batch in batches)
    use_edge_weight = any(batch.edge_weight is not None for batch in batches)
    # Nodes concatenate along dim 0, so the trailing window rides along with them.
    # All-or-nothing: a mix means the snapshots were built with different
    # sequence_window settings, which would silently train on ragged input.
    seq_present = [batch.node_sequence is not None for batch in batches]
    if any(seq_present) and not all(seq_present):
        raise ValueError("graph batches disagree on node_sequence; check sequence_window")
    node_sequences: list[Tensor] = []
    use_target_entry_path = any(
        batch.target_entry_path is not None for batch in batches
    )
    use_target_downstream = any(
        batch.target_downstream is not None for batch in batches
    )
    use_target_market_transition = any(
        batch.target_market_transition is not None for batch in batches
    )
    target_entry_paths = []
    target_downstream = []
    target_market_transition = []
    node_offset = 0
    graph_offset = 0

    for batch in batches:
        node_count = batch.node_features.shape[0]
        node_features.append(batch.node_features)
        feature_masks.append(batch.feature_mask)
        if use_available_mask:
            available_masks.append(
                batch.available_mask
                if batch.available_mask is not None
                else torch.ones_like(batch.node_features)
            )
        if use_supervision_mask:
            supervision_masks.append(
                batch.supervision_node_mask
                if batch.supervision_node_mask is not None
                else torch.ones(node_count, dtype=batch.node_features.dtype, device=batch.node_features.device)
            )
        if use_target_entry_path:
            target_entry_paths.append(
                batch.target_entry_path
                if batch.target_entry_path is not None
                else torch.full(
                    (node_count,),
                    float("nan"),
                    dtype=batch.node_features.dtype,
                    device=batch.node_features.device,
                )
            )
        if use_target_downstream:
            values = batch.target_downstream
            if values is not None and values.shape != (
                node_count,
                len(DOWNSTREAM_AUXILIARY_TASKS),
            ):
                raise ValueError(
                    "target_downstream must contain one row per node and one "
                    "column per auxiliary task"
                )
            target_downstream.append(
                values
                if values is not None
                else torch.full(
                    (node_count, len(DOWNSTREAM_AUXILIARY_TASKS)),
                    float("nan"),
                    dtype=batch.node_features.dtype,
                    device=batch.node_features.device,
                )
            )
        local_graph_index = (
            torch.zeros(node_count, dtype=torch.long, device=batch.node_features.device)
            if batch.graph_index is None
            else batch.graph_index.to(device=batch.node_features.device, dtype=torch.long)
        )
        local_graph_count = (
            int(local_graph_index.max().item()) + 1 if node_count else 0
        )
        if use_target_market_transition:
            values = batch.target_market_transition
            if values is not None and values.shape != (
                local_graph_count,
                MARKET_TRANSITION_AUXILIARY_WIDTH,
            ):
                raise ValueError(
                    "target_market_transition must contain one row per graph "
                    "and the configured auxiliary target width"
                )
            target_market_transition.append(
                values
                if values is not None
                else torch.full(
                    (local_graph_count, MARKET_TRANSITION_AUXILIARY_WIDTH),
                    float("nan"),
                    dtype=batch.node_features.dtype,
                    device=batch.node_features.device,
                )
            )
        graph_indices.append(local_graph_index + graph_offset)
        graph_offset += local_graph_count
        if batch.edge_index.numel():
            edge_indices.append(batch.edge_index + node_offset)
            if use_edge_weight:
                edge_weights.append(
                    batch.edge_weight
                    if batch.edge_weight is not None
                    else torch.ones(
                        batch.edge_index.shape[1],
                        dtype=batch.node_features.dtype,
                        device=batch.node_features.device,
                    )
                )
        if node_sequences is not None and batch.node_sequence is not None:
            node_sequences.append(batch.node_sequence)
        node_offset += node_count

    if edge_indices:
        edge_index = torch.cat(edge_indices, dim=1)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=first.node_features.device)

    return GraphBatch(
        node_features=torch.cat(node_features, dim=0),
        feature_mask=torch.cat(feature_masks, dim=0),
        edge_index=edge_index,
        edge_weight=torch.cat(edge_weights, dim=0) if edge_weights else None,
        available_mask=torch.cat(available_masks, dim=0) if available_masks else None,
        supervision_node_mask=torch.cat(supervision_masks, dim=0) if supervision_masks else None,
        graph_index=torch.cat(graph_indices, dim=0),
        node_sequence=(
            torch.cat(node_sequences, dim=0) if node_sequences else None
        ),
        target_entry_path=(
            torch.cat(target_entry_paths, dim=0) if target_entry_paths else None
        ),
        target_downstream=(
            torch.cat(target_downstream, dim=0) if target_downstream else None
        ),
        target_market_transition=(
            torch.cat(target_market_transition, dim=0)
            if target_market_transition
            else None
        ),
    )


class GraphConvBlock(nn.Module):
    """Small weighted message passing block without torch-geometric."""

    def __init__(
        self,
        hidden_dim: int,
        dropout: float = 0.05,
        neighbor_scale: float = 1.0,
    ):
        super().__init__()
        if not math.isfinite(neighbor_scale) or neighbor_scale < 0.0:
            raise ValueError("neighbor_scale must be finite and non-negative")
        self.self_proj = nn.Linear(hidden_dim, hidden_dim)
        self.neighbor_proj = nn.Linear(hidden_dim, hidden_dim)
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.neighbor_scale = float(neighbor_scale)

    def forward(
        self,
        h: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        neighbor_scale: Optional[float] = None,
    ) -> Tensor:
        scale = self.neighbor_scale if neighbor_scale is None else float(neighbor_scale)
        if not math.isfinite(scale) or scale < 0.0:
            raise ValueError("neighbor_scale must be finite and non-negative")
        if edge_index.numel() == 0:
            mixed = self.self_proj(h)
        else:
            src, dst = edge_index
            if edge_weight is None:
                weight = torch.ones(src.shape[0], device=h.device, dtype=h.dtype)
            else:
                weight = edge_weight.to(device=h.device, dtype=h.dtype)

            messages = h[src] * weight.unsqueeze(-1)
            agg = torch.zeros_like(h)
            agg.index_add_(0, dst, messages)

            degree = torch.zeros(h.shape[0], 1, device=h.device, dtype=h.dtype)
            degree.index_add_(0, dst, weight.abs().unsqueeze(-1))
            agg = agg / degree.clamp_min(1.0)

            mixed = self.self_proj(h) + scale * self.neighbor_proj(agg)

        return self.norm(h + self.dropout(F.gelu(mixed)))


class TemporalSequenceEncoder(nn.Module):
    """Attention over the trailing window, replacing the single-timestep projection.

    The production encoder saw one row of 149 hand-built features, so history
    entered only through whatever windows a human had pre-computed (return_20d,
    volatility_60d, ...). A latent is a deterministic function of its input, so
    that ceiling was absolute: no objective -- JEPA alignment included -- could
    recover structure the input never carried. Measured directly, JEPA weight
    0.00/0.25/1.00 all landed within 0.6 sigma of each other.

    Every element of the window is observable at decision time, so attention here
    is unmasked; the leakage boundary is the window's right edge, not causality
    inside it. Quarterly events sit ~90 days apart, which no fixed window can
    isolate -- attention can put weight on the report date itself.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        window: int,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.05,
        residual: bool = False,
        node_chunk: int = 1024,
    ):
        super().__init__()
        if window < 1:
            raise ValueError("window must be >= 1")
        # Nodes never attend to each other, so this bounds peak memory without
        # touching the result. Measured peak for 8208 nodes: 12.06 GiB whole,
        # 3.69 GiB at 1024.
        self.node_chunk = int(node_chunk) if node_chunk and node_chunk > 0 else 0
        self.residual = bool(residual)
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.window = int(window)
        self.value_proj = nn.Linear(input_dim, hidden_dim)
        self.position = nn.Parameter(torch.randn(window, hidden_dim) * 0.02)
        self.layers = nn.ModuleList(
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 2,
                dropout=dropout,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(self, sequence: Tensor) -> Tensor:
        """(nodes, window, input_dim) -> (nodes, hidden_dim) at the newest step."""
        if sequence.dim() != 3:
            raise ValueError("sequence must be (nodes, window, input_dim)")
        w = sequence.shape[1]
        if w > self.window:
            sequence = sequence[:, -self.window:]
            w = self.window
        chunk = self.node_chunk or sequence.shape[0]
        parts = []
        for start in range(0, sequence.shape[0], chunk):
            part = sequence[start:start + chunk]
            h = self.value_proj(part) + self.position[-w:].unsqueeze(0)
            recompute = self.training and torch.is_grad_enabled() and h.requires_grad
            for layer in self.layers:
                if recompute:
                    h = checkpoint(layer, h, use_reentrant=False)
                else:
                    h = layer(h)
            piece = self.output_norm(h[:, -1])
            if self.residual:
                piece = piece - self.output_norm(self.value_proj(part[:, -1]))
            parts.append(piece)
        out = parts[0] if len(parts) == 1 else torch.cat(parts, dim=0)
        return out


class GraphEncoder(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        dropout: float = 0.05,
        neighbor_scale: float = 1.0,
        sequence_window: int = 0,
        sequence_layers: int = 2,
        sequence_heads: int = 8,
        sequence_residual: bool = False,
    ):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.temporal = (
            TemporalSequenceEncoder(
                input_dim=input_dim,
                hidden_dim=hidden_dim,
                window=int(sequence_window),
                num_layers=int(sequence_layers),
                num_heads=int(sequence_heads),
                dropout=dropout,
                residual=bool(sequence_residual),
            )
            if int(sequence_window) > 0
            else None
        )
        self.blocks = nn.ModuleList(
            GraphConvBlock(
                hidden_dim=hidden_dim,
                dropout=dropout,
                neighbor_scale=neighbor_scale,
            )
            for _ in range(num_layers)
        )
        self.output_norm = nn.LayerNorm(hidden_dim)

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        neighbor_scale: Optional[float] = None,
        sequence: Optional[Tensor] = None,
    ) -> Tensor:
        if self.temporal is not None:
            if sequence is None:
                # Falling back to the MLP here would train a different model than
                # the one requested and leave the sequence encoder at init.
                raise ValueError(
                    "encoder was built with a sequence module but received no "
                    "node_sequence; the batch lost it (see merge_graph_batches)"
                )
            h = self.temporal(sequence)
            if getattr(self.temporal, "residual", False):
                h = F.gelu(self.input_proj(x)) + h
        else:
            h = F.gelu(self.input_proj(x))
        for block in self.blocks:
            h = block(
                h,
                edge_index=edge_index,
                edge_weight=edge_weight,
                neighbor_scale=neighbor_scale,
            )
        return self.output_norm(h)


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        output_norm: bool = False,
    ):
        super().__init__()
        inner = hidden_dim or max(input_dim, output_dim)
        layers: list[nn.Module] = [
            nn.Linear(input_dim, inner),
            nn.GELU(),
            nn.LayerNorm(inner),
            nn.Linear(inner, output_dim),
        ]
        if output_norm:
            layers.append(nn.LayerNorm(output_dim, elementwise_affine=False))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class StockGraphJEPA(nn.Module):
    """JEPA-style model for dynamic stock graph state imputation.

    The context encoder receives partial node state and an observation mask. The
    target encoder receives the complete state and is updated with EMA. The
    predictor learns to match target latent states on masked nodes/features.

    A small state head is included so the latent prediction can be inspected as
    an estimate of currently unobserved normalized features.
    """

    def __init__(
        self,
        num_features: int,
        hidden_dim: int = 128,
        num_layers: int = 3,
        ema_decay: float = 0.99,
        latent_loss_weight: float = 1.0,
        state_loss_weight: float = 0.25,
        temporal_state_mode: str = "direct",
        feature_names: Optional[Sequence[str]] = None,
        temporal_residual_short_steps: int = 2,
        temporal_head_steps: Optional[Sequence[int]] = None,
        state_feature_weights: Optional[Sequence[float]] = None,
        temporal_state_feature_weights: Optional[Sequence[float]] = None,
        temporal_state_context_skip: bool = False,
        hybrid_fast_direct: bool = False,
        return_correlation_loss_weight: float = 0.0,
        entry_path_correlation_loss_weight: float = 0.0,
        feature_means: Optional[Sequence[float]] = None,
        feature_stds: Optional[Sequence[float]] = None,
        normalize_predictor_output: bool = False,
        current_imputation_loss_weight: float = 0.0,
        imputation_standalone: bool = False,
        sequence_window: int = 0,
        sequence_layers: int = 2,
        sequence_heads: int = 8,
        sequence_residual: bool = False,
        latent_variance_weight: float = 0.0,
        latent_covariance_weight: float = 0.0,
        latent_variance_target: float = 1.0,
        hidden_completion_loss_weight: float = 0.0,
        hidden_completion_width: Optional[int] = None,
        graph_neighbor_scale: float = 1.0,
        temporal_graph_neighbor_scale: Optional[float] = None,
        temporal_stock_edge_scale: float = 1.0,
        global_stock_context: bool = False,
        downstream_auxiliary_loss_weight: float = 0.0,
        downstream_plan_loss_weight: float = 0.0,
        plan_temperature: float = 0.01,
        plan_buy_sell: bool = False,
        plan_permute_seed: int = 0,
        downstream_auxiliary_task_weights: Optional[Sequence[float]] = None,
        downstream_market_loss_weight: float = 0.0,
        downstream_market_cost_bps: float = 50.0,
        downstream_transition_loss_weight: float = 0.0,
        downstream_transition_pooling: str = "mean",
        temporal_impact_loss_mix: float = 0.0,
    ):
        super().__init__()
        if temporal_state_mode not in {
            "direct",
            "residual_mixed",
            "horizon_hybrid",
            "horizon_residual_heads",
        }:
            raise ValueError(
                "temporal_state_mode must be 'direct', 'residual_mixed', "
                "'horizon_hybrid', or 'horizon_residual_heads'"
            )
        if temporal_residual_short_steps < 1:
            raise ValueError("temporal_residual_short_steps must be >= 1")
        if temporal_state_mode == "horizon_hybrid" and (
            feature_names is None or len(feature_names) != num_features
        ):
            raise ValueError("horizon_hybrid temporal state mode requires every feature name")
        normalized_head_steps = tuple(
            sorted({int(step) for step in (temporal_head_steps or ())})
        )
        if any(step < 1 for step in normalized_head_steps):
            raise ValueError("temporal_head_steps must contain positive integers")
        if temporal_state_mode == "horizon_residual_heads" and not normalized_head_steps:
            raise ValueError(
                "horizon_residual_heads temporal state mode requires temporal_head_steps"
            )
        self.num_features = num_features
        self.ema_decay = ema_decay
        if not math.isfinite(latent_loss_weight) or latent_loss_weight < 0.0:
            raise ValueError("latent_loss_weight must be finite and non-negative")
        self.latent_loss_weight = float(latent_loss_weight)
        self.state_loss_weight = state_loss_weight
        self.temporal_state_mode = temporal_state_mode
        self.temporal_state_context_skip = bool(temporal_state_context_skip)
        if self.temporal_state_context_skip and temporal_state_mode != "horizon_residual_heads":
            raise ValueError(
                "temporal_state_context_skip requires horizon_residual_heads mode"
            )
        self.temporal_residual_short_steps = int(temporal_residual_short_steps)
        self.temporal_head_steps = normalized_head_steps
        self.hybrid_fast_direct = bool(hybrid_fast_direct)
        if return_correlation_loss_weight < 0.0:
            raise ValueError("return_correlation_loss_weight must be non-negative")
        self.return_correlation_loss_weight = float(return_correlation_loss_weight)
        if (
            not math.isfinite(entry_path_correlation_loss_weight)
            or entry_path_correlation_loss_weight < 0.0
        ):
            raise ValueError(
                "entry_path_correlation_loss_weight must be finite and non-negative"
            )
        self.entry_path_correlation_loss_weight = float(
            entry_path_correlation_loss_weight
        )
        self.downstream_plan_loss_weight = float(downstream_plan_loss_weight)
        self.plan_temperature = float(plan_temperature)
        self.plan_buy_sell = bool(plan_buy_sell)
        self.plan_permute_seed = int(plan_permute_seed)
        if (
            not math.isfinite(self.downstream_plan_loss_weight)
            or self.downstream_plan_loss_weight < 0.0
        ):
            raise ValueError(
                "downstream_plan_loss_weight must be finite and non-negative"
            )
        if self.downstream_plan_loss_weight > 0.0 and self.plan_temperature <= 0.0:
            raise ValueError("plan_temperature must be positive")
        if (
            not math.isfinite(downstream_auxiliary_loss_weight)
            or downstream_auxiliary_loss_weight < 0.0
        ):
            raise ValueError(
                "downstream_auxiliary_loss_weight must be finite and non-negative"
            )
        self.downstream_auxiliary_loss_weight = float(
            downstream_auxiliary_loss_weight
        )
        auxiliary_task_weights = torch.as_tensor(
            downstream_auxiliary_task_weights
            if downstream_auxiliary_task_weights is not None
            else [1.0] * len(DOWNSTREAM_AUXILIARY_TASKS),
            dtype=torch.float32,
        )
        if auxiliary_task_weights.shape != (len(DOWNSTREAM_AUXILIARY_TASKS),):
            raise ValueError(
                "downstream_auxiliary_task_weights must match auxiliary tasks"
            )
        if (
            not torch.isfinite(auxiliary_task_weights).all()
            or (auxiliary_task_weights < 0.0).any()
            or float(auxiliary_task_weights.sum()) <= 0.0
        ):
            raise ValueError(
                "downstream_auxiliary_task_weights must be finite, non-negative, "
                "and contain a positive weight"
            )
        if self.downstream_auxiliary_loss_weight > 0.0 and not normalized_head_steps:
            raise ValueError(
                "downstream auxiliary loss requires temporal_head_steps"
            )
        self.register_buffer(
            "downstream_auxiliary_task_weights",
            auxiliary_task_weights,
            persistent=False,
        )
        if (
            not math.isfinite(downstream_market_loss_weight)
            or downstream_market_loss_weight < 0.0
        ):
            raise ValueError(
                "downstream_market_loss_weight must be finite and non-negative"
            )
        if (
            not math.isfinite(downstream_market_cost_bps)
            or downstream_market_cost_bps < 0.0
        ):
            raise ValueError(
                "downstream_market_cost_bps must be finite and non-negative"
            )
        self.downstream_market_loss_weight = float(downstream_market_loss_weight)
        self.downstream_market_cost_bps = float(downstream_market_cost_bps)
        if self.downstream_market_loss_weight > 0.0 and not normalized_head_steps:
            raise ValueError("downstream market loss requires temporal_head_steps")
        if (
            not math.isfinite(downstream_transition_loss_weight)
            or downstream_transition_loss_weight < 0.0
        ):
            raise ValueError(
                "downstream_transition_loss_weight must be finite and non-negative"
            )
        self.downstream_transition_loss_weight = float(
            downstream_transition_loss_weight
        )
        if downstream_transition_pooling not in {
            "mean",
            "robust",
            "robust_projected",
        }:
            raise ValueError(
                "downstream_transition_pooling must be 'mean', 'robust', "
                "or 'robust_projected'"
            )
        self.downstream_transition_pooling = str(
            downstream_transition_pooling
        )
        if self.downstream_transition_loss_weight > 0.0 and not normalized_head_steps:
            raise ValueError(
                "downstream transition loss requires temporal_head_steps"
            )
        if (
            not math.isfinite(temporal_impact_loss_mix)
            or not 0.0 <= temporal_impact_loss_mix <= 1.0
        ):
            raise ValueError("temporal_impact_loss_mix must be between 0 and 1")
        self.temporal_impact_loss_mix = float(temporal_impact_loss_mix)
        self.normalize_predictor_output = bool(normalize_predictor_output)
        if current_imputation_loss_weight < 0.0:
            raise ValueError("current_imputation_loss_weight must be non-negative")
        self.current_imputation_loss_weight = float(current_imputation_loss_weight)
        self.imputation_standalone = bool(imputation_standalone)
        self.latent_variance_weight = float(latent_variance_weight)
        self.latent_covariance_weight = float(latent_covariance_weight)
        self.latent_variance_target = float(latent_variance_target)
        if not math.isfinite(graph_neighbor_scale) or graph_neighbor_scale < 0.0:
            raise ValueError("graph_neighbor_scale must be finite and non-negative")
        self.graph_neighbor_scale = float(graph_neighbor_scale)
        temporal_scale = (
            self.graph_neighbor_scale
            if temporal_graph_neighbor_scale is None
            else float(temporal_graph_neighbor_scale)
        )
        if not math.isfinite(temporal_scale) or temporal_scale < 0.0:
            raise ValueError(
                "temporal_graph_neighbor_scale must be finite and non-negative"
            )
        self.temporal_graph_neighbor_scale = temporal_scale
        if not math.isfinite(temporal_stock_edge_scale) or temporal_stock_edge_scale < 0.0:
            raise ValueError("temporal_stock_edge_scale must be finite and non-negative")
        self.temporal_stock_edge_scale = float(temporal_stock_edge_scale)
        self.global_stock_context = bool(global_stock_context)
        self.return_feature_indices: Dict[int, int] = {}
        for index, name in enumerate(feature_names or ()):
            if name.startswith("return_") and name.endswith("d"):
                raw_horizon = name[len("return_") : -1]
                if raw_horizon.isdigit():
                    self.return_feature_indices[int(raw_horizon)] = index
        self.return_feature_index = self.return_feature_indices.get(1)
        names = tuple(feature_names or ())
        self.gap_open_feature_index = (
            names.index("gap_open") if "gap_open" in names else None
        )
        self.intraday_return_feature_index = (
            names.index("intraday_return") if "intraday_return" in names else None
        )
        if self.return_correlation_loss_weight > 0.0 and not self.return_feature_indices:
            raise ValueError("return correlation loss requires return_<horizon>d features")
        if self.entry_path_correlation_loss_weight > 0.0 and (
            self.gap_open_feature_index is None
            or self.intraday_return_feature_index is None
            or not self.return_feature_indices
        ):
            raise ValueError(
                "entry path correlation loss requires gap_open, intraday_return, "
                "and return_<horizon>d features"
            )
        means = (
            torch.zeros(num_features, dtype=torch.float32)
            if feature_means is None
            else torch.as_tensor(feature_means, dtype=torch.float32)
        )
        stds = (
            torch.ones(num_features, dtype=torch.float32)
            if feature_stds is None
            else torch.as_tensor(feature_stds, dtype=torch.float32)
        )
        if means.shape != (num_features,) or stds.shape != (num_features,):
            raise ValueError("feature normalization statistics must match num_features")
        if (
            not torch.isfinite(means).all()
            or not torch.isfinite(stds).all()
            or (stds <= 0.0).any()
        ):
            raise ValueError("feature normalization statistics must be finite with positive std")
        if self.entry_path_correlation_loss_weight > 0.0 and (
            feature_means is None or feature_stds is None
        ):
            raise ValueError(
                "entry path correlation loss requires feature normalization statistics"
            )
        self.register_buffer("feature_means", means, persistent=False)
        self.register_buffer("feature_stds", stds, persistent=False)
        feature_weights = (
            torch.ones(num_features, dtype=torch.float32)
            if state_feature_weights is None
            else torch.as_tensor(state_feature_weights, dtype=torch.float32)
        )
        if feature_weights.shape != (num_features,):
            raise ValueError("state_feature_weights must match num_features")
        if not torch.isfinite(feature_weights).all() or (feature_weights < 0.0).any():
            raise ValueError("state_feature_weights must be finite and non-negative")
        if float(feature_weights.sum()) <= 0.0:
            raise ValueError("state_feature_weights must contain a positive weight")
        self.register_buffer("state_feature_weights", feature_weights, persistent=False)
        temporal_feature_weights = (
            feature_weights.clone()
            if temporal_state_feature_weights is None
            else torch.as_tensor(temporal_state_feature_weights, dtype=torch.float32)
        )
        if temporal_feature_weights.shape != (num_features,):
            raise ValueError("temporal_state_feature_weights must match num_features")
        if (
            not torch.isfinite(temporal_feature_weights).all()
            or (temporal_feature_weights < 0.0).any()
        ):
            raise ValueError(
                "temporal_state_feature_weights must be finite and non-negative"
            )
        if float(temporal_feature_weights.sum()) <= 0.0:
            raise ValueError(
                "temporal_state_feature_weights must contain a positive weight"
            )
        self.register_buffer(
            "temporal_state_feature_weights",
            temporal_feature_weights,
            persistent=False,
        )

        self.sequence_window = int(sequence_window)
        self.sequence_layers = int(sequence_layers)
        self.sequence_heads = int(sequence_heads)
        self.sequence_residual = bool(sequence_residual)
        encoder_input_dim = num_features * (4 if self.global_stock_context else 2)
        self.context_encoder = GraphEncoder(
            input_dim=encoder_input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            neighbor_scale=self.graph_neighbor_scale,
            sequence_window=self.sequence_window,
            sequence_layers=self.sequence_layers,
            sequence_heads=self.sequence_heads,
            sequence_residual=self.sequence_residual,
        )
        self.target_encoder = deepcopy(self.context_encoder)
        for param in self.target_encoder.parameters():
            param.requires_grad = False
        # The EMA teacher must not inject dropout noise into the target.
        self.target_encoder.eval()

        self.predictor = MLP(
            hidden_dim,
            hidden_dim,
            hidden_dim=hidden_dim * 2,
            output_norm=self.normalize_predictor_output,
        )
        self.state_head = MLP(hidden_dim, num_features, hidden_dim=hidden_dim)
        # Hidden-state completion. Reads the graph-on completion context and
        # estimates the flow that is true at t but not disclosed until t+1. Built
        # only when active, so checkpoints without it stay loadable under strict.
        self.hidden_completion_loss_weight = float(hidden_completion_loss_weight)
        self.hidden_completion_head = (
            MLP(hidden_dim, int(hidden_completion_width or HIDDEN_COMPLETION_WIDTH), hidden_dim=hidden_dim)
            if hidden_completion_loss_weight > 0.0
            else None
        )
        self.delta_head = (
            MLP(hidden_dim, num_features, hidden_dim=hidden_dim)
            if temporal_state_mode in {"residual_mixed", "horizon_hybrid"}
            else None
        )
        if self.delta_head is not None:
            # Start residual forecasts at the causal persistence baseline.
            nn.init.zeros_(self.delta_head.net[-1].weight)
            nn.init.zeros_(self.delta_head.net[-1].bias)
        self.temporal_heads = nn.ModuleDict(
            {
                str(step): MLP(
                    hidden_dim * (2 if self.temporal_state_context_skip else 1),
                    num_features * 2,
                    hidden_dim=hidden_dim,
                )
                for step in normalized_head_steps
            }
            if temporal_state_mode == "horizon_residual_heads"
            else {}
        )
        auxiliary_input_dim = hidden_dim * (
            2 if self.temporal_state_context_skip else 1
        )
        auxiliary_hidden_dim = min(256, max(32, hidden_dim // 4))
        self.downstream_auxiliary_heads = nn.ModuleDict(
            {
                f"{step}:{task}": MLP(
                    auxiliary_input_dim,
                    1,
                    hidden_dim=auxiliary_hidden_dim,
                )
                for step in normalized_head_steps
                for task in DOWNSTREAM_AUXILIARY_TASKS
            }
            if self.downstream_auxiliary_loss_weight > 0.0
            else {}
        )
        self.downstream_market_heads = nn.ModuleDict(
            {
                str(step): MLP(
                    auxiliary_input_dim * 2,
                    2,
                    hidden_dim=auxiliary_hidden_dim,
                )
                for step in normalized_head_steps
            }
            if self.downstream_market_loss_weight > 0.0
            else {}
        )
        self.downstream_transition_projector = (
            nn.Sequential(
                nn.LayerNorm(auxiliary_input_dim),
                nn.Linear(auxiliary_input_dim, auxiliary_hidden_dim),
                nn.SiLU(),
                nn.LayerNorm(auxiliary_hidden_dim),
            )
            if self.downstream_transition_loss_weight > 0.0
            and self.downstream_transition_pooling == "robust_projected"
            else nn.Identity()
        )
        transition_pool_dim = (
            auxiliary_hidden_dim
            if self.downstream_transition_pooling == "robust_projected"
            else auxiliary_input_dim
        )
        self.downstream_transition_heads = nn.ModuleDict(
            {
                str(step): MLP(
                    transition_pool_dim
                    * (
                        5
                        if self.downstream_transition_pooling
                        in {"robust", "robust_projected"}
                        else 2
                    ),
                    MARKET_TRANSITION_AUXILIARY_WIDTH,
                    hidden_dim=auxiliary_hidden_dim,
                )
                for step in normalized_head_steps
            }
            if self.downstream_transition_loss_weight > 0.0
            else {}
        )
        for head in self.temporal_heads.values():
            output = head.net[-1]
            if not isinstance(output, nn.Linear):
                raise TypeError("temporal state head must end with a linear layer")
            nn.init.zeros_(output.weight[num_features:])
            nn.init.zeros_(output.bias[num_features:])
        hybrid_residual = torch.zeros(num_features, dtype=torch.bool)
        hybrid_short_residual = torch.ones(num_features, dtype=torch.bool)
        if temporal_state_mode == "horizon_hybrid":
            for index, name in enumerate(feature_names or []):
                if (
                    name.startswith("fund_")
                    or name.startswith("ma")
                    or "drawdown" in name
                    or "breakout" in name
                    or "range_position" in name
                    or name.startswith("market_")
                ):
                    hybrid_residual[index] = True
                if self.hybrid_fast_direct and (
                    name in {"return_1d", "gap_open", "intraday_return", "market_return_1d"}
                    or name.endswith("_flow_ratio_1d")
                    or name == "news_score_1d"
                ):
                    hybrid_short_residual[index] = False
        self.register_buffer("hybrid_residual_feature_mask", hybrid_residual, persistent=False)
        self.register_buffer(
            "hybrid_short_residual_feature_mask",
            hybrid_short_residual,
            persistent=False,
        )

    def train(self, mode: bool = True) -> "StockGraphJEPA":
        """Keep the EMA target deterministic while training the online network."""

        super().train(mode)
        self.target_encoder.eval()
        return self

    def _masked_encoder_input(
        self,
        x: Tensor,
        mask: Tensor,
        supervision_node_mask: Optional[Tensor] = None,
        graph_index: Optional[Tensor] = None,
    ) -> Tensor:
        parts = [x * mask, mask]
        if self.global_stock_context:
            mean, std = self._global_stock_moments(
                x,
                mask,
                supervision_node_mask=supervision_node_mask,
                graph_index=graph_index,
            )
            parts.extend([mean, std])
        return torch.cat(parts, dim=-1)

    def _context_input(
        self,
        x: Tensor,
        mask: Tensor,
        supervision_node_mask: Optional[Tensor] = None,
        graph_index: Optional[Tensor] = None,
    ) -> Tensor:
        return self._masked_encoder_input(
            x,
            mask,
            supervision_node_mask=supervision_node_mask,
            graph_index=graph_index,
        )

    def _target_input(
        self,
        x: Tensor,
        available_mask: Optional[Tensor] = None,
        supervision_node_mask: Optional[Tensor] = None,
        graph_index: Optional[Tensor] = None,
    ) -> Tensor:
        mask = torch.ones_like(x) if available_mask is None else available_mask.to(dtype=x.dtype)
        return self._masked_encoder_input(
            x,
            mask,
            supervision_node_mask=supervision_node_mask,
            graph_index=graph_index,
        )

    @staticmethod
    def _global_stock_moments(
        x: Tensor,
        mask: Tensor,
        supervision_node_mask: Optional[Tensor] = None,
        graph_index: Optional[Tensor] = None,
    ) -> tuple[Tensor, Tensor]:
        if x.ndim != 2 or mask.shape != x.shape:
            raise ValueError("global stock moments require aligned 2D state and mask tensors")
        node_count, feature_count = x.shape
        stock_nodes = (
            torch.ones(node_count, dtype=torch.bool, device=x.device)
            if supervision_node_mask is None
            else supervision_node_mask.to(device=x.device) > 0.5
        )
        if stock_nodes.shape != (node_count,):
            raise ValueError("supervision_node_mask must contain one value per node")
        groups = (
            torch.zeros(node_count, dtype=torch.long, device=x.device)
            if graph_index is None
            else graph_index.to(device=x.device, dtype=torch.long)
        )
        if groups.shape != (node_count,):
            raise ValueError("graph_index must contain one value per node")
        if node_count and int(groups.min().item()) < 0:
            raise ValueError("graph_index values must be non-negative")
        graph_count = int(groups.max().item()) + 1 if node_count else 0
        valid = stock_nodes.unsqueeze(-1) & (mask > 0.5) & torch.isfinite(x)
        weights = valid.to(dtype=x.dtype)
        values = torch.where(valid, x, torch.zeros_like(x))
        counts = x.new_zeros((graph_count, feature_count))
        sums = x.new_zeros((graph_count, feature_count))
        square_sums = x.new_zeros((graph_count, feature_count))
        if node_count:
            counts.index_add_(0, groups, weights)
            sums.index_add_(0, groups, values)
            square_sums.index_add_(0, groups, values.square())
        denominator = counts.clamp_min(1.0)
        means = sums / denominator
        variances = (square_sums / denominator - means.square()).clamp_min(0.0)
        observed = counts > 0.0
        means = torch.where(observed, means, torch.zeros_like(means))
        stds = torch.where(observed, variances.sqrt(), torch.zeros_like(variances))
        return means[groups], stds[groups]

    def _supervision_node_mask(self, batch: GraphBatch) -> Tensor:
        if batch.supervision_node_mask is None:
            return torch.ones(batch.node_features.shape[0], device=batch.node_features.device, dtype=torch.bool)
        return batch.supervision_node_mask.to(device=batch.node_features.device) > 0.5

    def _state_reconstruction_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
        feature_weights: Optional[Tensor] = None,
        node_weights: Optional[Tensor] = None,
    ) -> Tensor:
        element_loss = F.smooth_l1_loss(prediction, target, reduction="none")
        selected_feature_weights = (
            self.state_feature_weights
            if feature_weights is None
            else feature_weights.to(device=prediction.device, dtype=prediction.dtype)
        )
        weights = selected_feature_weights.unsqueeze(0).expand_as(element_loss)
        if node_weights is not None:
            node_weights = node_weights.to(
                device=prediction.device,
                dtype=prediction.dtype,
            )
            if node_weights.shape != prediction.shape[:1]:
                raise ValueError("node_weights must contain one value per node")
            if (
                not torch.isfinite(node_weights).all()
                or (node_weights < 0.0).any()
            ):
                raise ValueError("node_weights must be finite and non-negative")
            weights = weights * node_weights.unsqueeze(-1)
        selected_weights = weights[mask]
        denominator = selected_weights.sum()
        return (element_loss[mask] * selected_weights).sum() / denominator.clamp_min(1e-12)

    def _return_correlation_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        mask: Tensor,
        graph_index: Optional[Tensor],
        target_horizon: Optional[int] = None,
    ) -> Tensor:
        if self.return_correlation_loss_weight <= 0.0:
            return prediction.new_tensor(0.0)
        horizon = 1 if target_horizon is None else int(target_horizon)
        feature_index = self.return_feature_indices.get(horizon)
        if feature_index is None:
            raise ValueError(
                "return correlation loss has no matching state feature for "
                f"horizon={horizon}"
            )
        valid = mask[:, feature_index]
        return self._grouped_correlation_loss(
            prediction[:, feature_index],
            target[:, feature_index],
            valid,
            graph_index,
        )

    def _grouped_correlation_loss(
        self,
        prediction: Tensor,
        target: Tensor,
        valid: Tensor,
        graph_index: Optional[Tensor],
    ) -> Tensor:
        if not valid.any():
            return prediction.new_tensor(0.0)
        compute_dtype = torch.promote_types(prediction.dtype, target.dtype)
        if compute_dtype in {torch.float16, torch.bfloat16}:
            compute_dtype = torch.float32
        pred = prediction[valid].to(dtype=compute_dtype)
        observed = target[valid].to(dtype=compute_dtype)
        groups = (
            torch.zeros(pred.shape[0], dtype=torch.long, device=prediction.device)
            if graph_index is None
            else graph_index[valid].to(device=prediction.device, dtype=torch.long)
        )
        counts = torch.bincount(groups).to(dtype=compute_dtype)
        pred_sum = torch.zeros_like(counts).index_add_(0, groups, pred)
        target_sum = torch.zeros_like(counts).index_add_(0, groups, observed)
        pred_sq_sum = torch.zeros_like(counts).index_add_(0, groups, pred.square())
        target_sq_sum = torch.zeros_like(counts).index_add_(0, groups, observed.square())
        cross_sum = torch.zeros_like(counts).index_add_(0, groups, pred * observed)
        safe_counts = counts.clamp_min(1.0)
        covariance = cross_sum - pred_sum * target_sum / safe_counts
        pred_variance = pred_sq_sum - pred_sum.square() / safe_counts
        target_variance = target_sq_sum - target_sum.square() / safe_counts
        usable = (counts >= 3.0) & (pred_variance > 1e-8) & (target_variance > 1e-8)
        if not usable.any():
            return prediction.new_tensor(0.0)
        correlation = covariance[usable] / torch.sqrt(
            pred_variance[usable] * target_variance[usable]
        ).clamp_min(1e-8)
        return 1.0 - correlation.clamp(-1.0, 1.0).mean()

    def _entry_path_correlation_loss(
        self,
        prediction: Tensor,
        next_open_prediction: Tensor,
        target_batch: GraphBatch,
        mask: Tensor,
        target_horizon: int,
    ) -> Tensor:
        if self.entry_path_correlation_loss_weight <= 0.0:
            return prediction.new_tensor(0.0)
        if target_batch.target_entry_path is None:
            raise ValueError("entry path correlation loss requires target_entry_path")
        horizon = int(target_horizon)
        if horizon == 1:
            feature_index = self.intraday_return_feature_index
            if feature_index is None:
                raise ValueError("entry path h1 requires intraday_return")
            path_prediction = (
                prediction[:, feature_index] * self.feature_stds[feature_index]
                + self.feature_means[feature_index]
            )
            valid = mask[:, feature_index]
        else:
            feature_index = self.return_feature_indices.get(horizon)
            gap_index = self.gap_open_feature_index
            if feature_index is None or gap_index is None:
                raise ValueError(
                    f"entry path loss has no state features for horizon={horizon}"
                )
            close_return = (
                prediction[:, feature_index] * self.feature_stds[feature_index]
                + self.feature_means[feature_index]
            )
            next_open_gap = (
                next_open_prediction[:, gap_index] * self.feature_stds[gap_index]
                + self.feature_means[gap_index]
            )
            denominator = 1.0 + next_open_gap
            safe_denominator = torch.where(
                denominator > 1e-6,
                denominator,
                torch.ones_like(denominator),
            )
            path_prediction = (1.0 + close_return) / safe_denominator - 1.0
            valid = mask[:, feature_index] & (denominator > 1e-6)
        target_path = target_batch.target_entry_path.to(
            device=prediction.device,
            dtype=prediction.dtype,
        )
        if target_path.shape != prediction.shape[:1]:
            raise ValueError("target_entry_path must contain one value per node")
        valid = valid & torch.isfinite(path_prediction) & torch.isfinite(target_path)
        return self._grouped_correlation_loss(
            path_prediction,
            target_path,
            valid,
            target_batch.graph_index,
        )

    def predict_downstream_auxiliary(
        self,
        z_context: Tensor,
        z_pred: Tensor,
        rollout_steps: int,
    ) -> Tensor:
        """Predict the standardized specialist targets for one rollout horizon."""

        if z_context.shape != z_pred.shape:
            raise ValueError("z_context and z_pred must have the same shape")
        head_input = (
            torch.cat([z_context, z_pred - z_context], dim=-1)
            if self.temporal_state_context_skip
            else z_pred
        )
        predictions = []
        for task_name in DOWNSTREAM_AUXILIARY_TASKS:
            key = f"{int(rollout_steps)}:{task_name}"
            if key not in self.downstream_auxiliary_heads:
                raise ValueError(f"missing downstream auxiliary head for {key}")
            predictions.append(
                self.downstream_auxiliary_heads[key](head_input).squeeze(-1)
            )
        return torch.stack(predictions, dim=-1)



    def _collect_plan_horizon(
        self,
        predicted: Dict[int, Tensor],
        realized: Dict[int, Tensor],
        valid: Dict[int, Tensor],
        *,
        z_context: Tensor,
        z_pred: Tensor,
        target_batch: GraphBatch,
        rollout_steps: int,
    ) -> None:
        """Stash one horizon's raw path prediction and realized return.

        `attach_downstream_targets` standardizes each date's targets as
        (x - mean) / std, so both the stored target and the head's prediction
        live in standardized space. Multiplying by std and adding mean recovers
        the raw return exactly, and reusing the same scale for both keeps the
        prediction and the label on one scale by construction -- safer than
        recomputing raw returns from prices in a second place.
        """

        if target_batch.target_downstream is None:
            raise ValueError("plan loss requires target_downstream")
        if target_batch.target_downstream_scale is None:
            raise ValueError(
                "plan loss requires target_downstream_scale; run "
                "attach_downstream_targets with the plan flag enabled"
            )
        # The plan ranks horizons, so its scale must be knowable at the decision
        # date. target_downstream_scale is this date's realized cross-section and
        # would make the ranking hindsight; the causal scale is fitted on paths
        # that finished before the decision. Falling back to the realized scale
        # if the causal one is absent would silently restore the leak, so a tree
        # without it is an error.
        causal = getattr(target_batch, "target_downstream_causal_scale", None)
        if causal is None:
            raise ValueError(
                "plan loss requires target_downstream_causal_scale; the realized "
                "scale leaks the decision date's own cross-sectional mean"
            )
        scale = causal.to(device=z_pred.device, dtype=z_pred.dtype)
        mean = scale[:, 0]
        deviation = scale[:, 1]
        head = self.predict_downstream_auxiliary(z_context, z_pred, rollout_steps)
        standardized_prediction = head[:, PATH_RETURN_TASK_INDEX]
        standardized_target = target_batch.target_downstream.to(
            device=z_pred.device, dtype=z_pred.dtype
        )[:, PATH_RETURN_TASK_INDEX]
        supervised = self._supervision_node_mask(target_batch)

        # attach_downstream_targets leaves NaN in the scale for any date whose
        # cross-sectional std was degenerate. Multiplying the head's output by a
        # NaN puts NaN INSIDE the autograd graph: masking it afterwards hides it
        # in the forward but the backward still returns a NaN gradient, which
        # poisons every weight on the first step. Sanitize before the multiply
        # and let the mask -- built from the ORIGINAL scale -- do the excluding.
        usable = (
            torch.isfinite(mean)
            & torch.isfinite(deviation)
            & (deviation > 0)
        )
        safe_mean = torch.where(usable, mean, torch.zeros_like(mean))
        safe_deviation = torch.where(
            usable, deviation, torch.ones_like(deviation)
        )
        safe_target = torch.where(
            torch.isfinite(standardized_target),
            standardized_target,
            torch.zeros_like(standardized_target),
        )
        predicted[int(rollout_steps)] = (
            standardized_prediction * safe_deviation + safe_mean
        )
        realized[int(rollout_steps)] = safe_target * safe_deviation + safe_mean
        valid[int(rollout_steps)] = (
            supervised & usable & torch.isfinite(standardized_target)
        )

    def _downstream_auxiliary_loss(
        self,
        z_context: Tensor,
        z_pred: Tensor,
        target_batch: GraphBatch,
        rollout_steps: int,
    ) -> Tensor:
        if self.downstream_auxiliary_loss_weight <= 0.0:
            return z_pred.new_tensor(0.0)
        if target_batch.target_downstream is None:
            raise ValueError(
                "downstream auxiliary loss requires target_downstream"
            )
        targets = target_batch.target_downstream.to(
            device=z_pred.device,
            dtype=z_pred.dtype,
        )
        expected = (z_pred.shape[0], len(DOWNSTREAM_AUXILIARY_TASKS))
        if targets.shape != expected:
            raise ValueError(
                f"target_downstream shape must be {expected}, got {tuple(targets.shape)}"
            )
        predictions = self.predict_downstream_auxiliary(
            z_context,
            z_pred,
            rollout_steps,
        )
        supervised_nodes = self._supervision_node_mask(target_batch)
        weighted_losses = []
        selected_weights = []
        for task_index, task_name in enumerate(DOWNSTREAM_AUXILIARY_TASKS):
            task_weight = self.downstream_auxiliary_task_weights[task_index]
            if float(task_weight) <= 0.0:
                continue
            prediction = predictions[:, task_index]
            target = targets[:, task_index]
            valid = supervised_nodes & torch.isfinite(target)
            if not valid.any():
                continue
            regression = F.smooth_l1_loss(
                prediction[valid],
                target[valid],
            )
            correlation = self._grouped_correlation_loss(
                prediction,
                target,
                valid,
                target_batch.graph_index,
            )
            weighted_losses.append(
                task_weight * (0.5 * regression + 0.5 * correlation)
            )
            selected_weights.append(task_weight)
        if not weighted_losses:
            return z_pred.new_tensor(0.0)
        return torch.stack(weighted_losses).sum() / torch.stack(
            selected_weights
        ).sum().clamp_min(1e-12)

    @staticmethod
    def _pool_nodes_by_graph(
        values: Tensor,
        selected: Tensor,
        graph_index: Optional[Tensor],
        graph_count: Optional[int] = None,
    ) -> tuple[Tensor, Tensor]:
        if values.ndim != 2 or selected.shape != values.shape[:1]:
            raise ValueError("graph pooling inputs are not aligned")
        groups = (
            torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
            if graph_index is None
            else graph_index.to(device=values.device, dtype=torch.long)
        )
        if groups.shape != values.shape[:1]:
            raise ValueError("graph_index must contain one value per node")
        inferred_count = int(groups.max().item()) + 1 if len(groups) else 0
        count = inferred_count if graph_count is None else int(graph_count)
        if count < inferred_count:
            raise ValueError("graph_count cannot omit observed graph indices")
        selected_groups = groups[selected]
        counts = torch.zeros(count, dtype=values.dtype, device=values.device)
        pooled = torch.zeros(
            (count, values.shape[1]), dtype=values.dtype, device=values.device
        )
        if selected_groups.numel():
            counts.index_add_(
                0,
                selected_groups,
                torch.ones_like(selected_groups, dtype=values.dtype),
            )
            pooled.index_add_(0, selected_groups, values[selected])
        pooled = pooled / counts.clamp_min(1.0).unsqueeze(-1)
        return pooled, counts

    @classmethod
    def _pool_distribution_by_graph(
        cls,
        values: Tensor,
        selected: Tensor,
        graph_index: Optional[Tensor],
        graph_count: Optional[int] = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        mean, counts = cls._pool_nodes_by_graph(
            values, selected, graph_index, graph_count
        )
        second_moment, _ = cls._pool_nodes_by_graph(
            values.square(), selected, graph_index, mean.shape[0]
        )
        std = (second_moment - mean.square()).clamp_min(0.0).sqrt()
        groups = (
            torch.zeros(values.shape[0], dtype=torch.long, device=values.device)
            if graph_index is None
            else graph_index.to(device=values.device, dtype=torch.long)
        )
        median_rows = []
        for graph in range(mean.shape[0]):
            graph_values = values[selected & (groups == graph)]
            median_rows.append(
                graph_values.median(dim=0).values
                if graph_values.shape[0]
                else torch.zeros(
                    values.shape[1], dtype=values.dtype, device=values.device
                )
            )
        median = (
            torch.stack(median_rows, dim=0)
            if median_rows
            else values.new_zeros((0, values.shape[1]))
        )
        return mean, std, median, counts

    def predict_downstream_market(
        self,
        z_context: Tensor,
        z_pred: Tensor,
        supervision_node_mask: Tensor,
        graph_index: Optional[Tensor],
        rollout_steps: int,
    ) -> Tensor:
        """Predict market return in percent and cost-exceedance logit per date."""

        if z_context.shape != z_pred.shape:
            raise ValueError("z_context and z_pred must have the same shape")
        if supervision_node_mask.shape != z_pred.shape[:1]:
            raise ValueError("supervision_node_mask must contain one value per node")
        head_input = (
            torch.cat([z_context, z_pred - z_context], dim=-1)
            if self.temporal_state_context_skip
            else z_pred
        )
        stock_mask = supervision_node_mask.to(device=z_pred.device) > 0.5
        groups = (
            torch.zeros(z_pred.shape[0], dtype=torch.long, device=z_pred.device)
            if graph_index is None
            else graph_index.to(device=z_pred.device, dtype=torch.long)
        )
        graph_count = int(groups.max().item()) + 1 if len(groups) else 0
        stock_state, stock_counts = self._pool_nodes_by_graph(
            head_input, stock_mask, groups, graph_count
        )
        external_state, _external_counts = self._pool_nodes_by_graph(
            head_input, ~stock_mask, groups, graph_count
        )
        if (stock_counts <= 0.0).any():
            raise ValueError("each graph requires at least one supervised stock node")
        key = str(int(rollout_steps))
        if key not in self.downstream_market_heads:
            raise ValueError(f"missing downstream market head for {key}")
        return self.downstream_market_heads[key](
            torch.cat([stock_state, external_state], dim=-1)
        )

    def _downstream_market_loss(
        self,
        z_context: Tensor,
        z_pred: Tensor,
        target_batch: GraphBatch,
        rollout_steps: int,
    ) -> Tensor:
        if self.downstream_market_loss_weight <= 0.0:
            return z_pred.new_tensor(0.0)
        if target_batch.target_entry_path is None:
            raise ValueError("downstream market loss requires target_entry_path")
        supervised_nodes = self._supervision_node_mask(target_batch)
        outputs = self.predict_downstream_market(
            z_context,
            z_pred,
            supervised_nodes,
            target_batch.graph_index,
            rollout_steps,
        )
        target_path = target_batch.target_entry_path.to(
            device=z_pred.device, dtype=z_pred.dtype
        )
        valid = supervised_nodes & torch.isfinite(target_path)
        target_mean, target_counts = self._pool_nodes_by_graph(
            target_path.unsqueeze(-1),
            valid,
            target_batch.graph_index,
            outputs.shape[0],
        )
        usable = target_counts >= 20.0
        if not usable.any():
            return z_pred.new_tensor(0.0)
        target_percent = 100.0 * target_mean[usable, 0]
        return_loss = F.smooth_l1_loss(outputs[usable, 0], target_percent)
        threshold_percent = self.downstream_market_cost_bps / 100.0
        cost_exceedance = (target_percent > threshold_percent).to(outputs.dtype)
        classification_loss = F.binary_cross_entropy_with_logits(
            outputs[usable, 1], cost_exceedance
        )
        return 0.5 * return_loss + 0.5 * classification_loss

    def predict_downstream_transition(
        self,
        z_context: Tensor,
        z_pred: Tensor,
        supervision_node_mask: Tensor,
        graph_index: Optional[Tensor],
        rollout_steps: int,
        stock_pool_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict broad stock-market transition intensity and event logits."""

        if z_context.shape != z_pred.shape:
            raise ValueError("z_context and z_pred must have the same shape")
        if supervision_node_mask.shape != z_pred.shape[:1]:
            raise ValueError("supervision_node_mask must contain one value per node")
        head_input = (
            torch.cat([z_context, z_pred - z_context], dim=-1)
            if self.temporal_state_context_skip
            else z_pred
        )
        head_input = self.downstream_transition_projector(head_input)
        stock_mask = supervision_node_mask.to(device=z_pred.device) > 0.5
        pooled_stock_mask = stock_mask
        if stock_pool_mask is not None:
            if stock_pool_mask.shape != z_pred.shape[:1]:
                raise ValueError("stock_pool_mask must contain one value per node")
            pooled_stock_mask = stock_mask & stock_pool_mask.to(
                device=z_pred.device, dtype=torch.bool
            )
        groups = (
            torch.zeros(z_pred.shape[0], dtype=torch.long, device=z_pred.device)
            if graph_index is None
            else graph_index.to(device=z_pred.device, dtype=torch.long)
        )
        graph_count = int(groups.max().item()) + 1 if len(groups) else 0
        if self.downstream_transition_pooling in {"robust", "robust_projected"}:
            stock_mean, stock_std, stock_median, stock_counts = (
                self._pool_distribution_by_graph(
                    head_input, pooled_stock_mask, groups, graph_count
                )
            )
            external_mean, external_std, _external_median, _external_counts = (
                self._pool_distribution_by_graph(
                    head_input, ~stock_mask, groups, graph_count
                )
            )
            pooled = torch.cat(
                (
                    stock_mean,
                    stock_std,
                    stock_median,
                    external_mean,
                    external_std,
                ),
                dim=-1,
            )
        else:
            stock_mean, stock_counts = self._pool_nodes_by_graph(
                head_input, pooled_stock_mask, groups, graph_count
            )
            external_mean, _external_counts = self._pool_nodes_by_graph(
                head_input, ~stock_mask, groups, graph_count
            )
            pooled = torch.cat((stock_mean, external_mean), dim=-1)
        if (stock_counts <= 0.0).any():
            raise ValueError("each graph requires at least one supervised stock node")
        key = str(int(rollout_steps))
        if key not in self.downstream_transition_heads:
            raise ValueError(f"missing downstream transition head for {key}")
        return self.downstream_transition_heads[key](pooled)

    @staticmethod
    def _binary_focal_loss(
        logits: Tensor,
        target: Tensor,
        sample_weight: Optional[Tensor] = None,
    ) -> Tensor:
        target = target.to(dtype=logits.dtype)
        cross_entropy = F.binary_cross_entropy_with_logits(
            logits, target, reduction="none"
        )
        probability = torch.sigmoid(logits)
        probability_target = probability * target + (1.0 - probability) * (
            1.0 - target
        )
        alpha = 0.75 * target + 0.25 * (1.0 - target)
        loss = alpha * (1.0 - probability_target).square() * cross_entropy
        if sample_weight is None:
            return loss.mean()
        weight = sample_weight.to(device=loss.device, dtype=loss.dtype)
        if weight.ndim != 1 or weight.shape[0] != loss.shape[0]:
            raise ValueError("sample_weight must contain one value per graph")
        while weight.ndim < loss.ndim:
            weight = weight.unsqueeze(-1)
        weight = weight.expand_as(loss)
        return (loss * weight).sum() / weight.sum().clamp_min(1e-12)

    @staticmethod
    def _market_transition_row_weight(targets: Tensor) -> Tensor:
        """Prioritize broad transition mass without using any individual node."""

        family_count = len(MARKET_TRANSITION_AUXILIARY_FAMILIES)
        if targets.ndim != 2 or targets.shape[1] != MARKET_TRANSITION_AUXILIARY_WIDTH:
            raise ValueError("invalid market transition target layout")
        family_salience = torch.expm1(
            targets[:, :family_count].clamp_min(0.0)
        )
        broad_selloff = targets[:, 2 * family_count].clamp(0.0, 1.0)
        systemic_impact = torch.maximum(
            family_salience.amax(dim=1), broad_selloff
        )
        return 1.0 + 3.0 * systemic_impact.clamp(max=3.0)

    def _temporal_impact_node_weights(
        self,
        target_batch: GraphBatch,
    ) -> Optional[Tensor]:
        if self.temporal_impact_loss_mix <= 0.0:
            return None
        if target_batch.target_market_transition is None:
            raise ValueError(
                "temporal impact loss requires target_market_transition"
            )
        targets = target_batch.target_market_transition.to(
            device=target_batch.node_features.device,
            dtype=target_batch.node_features.dtype,
        )
        if (
            targets.ndim != 2
            or targets.shape[1] != MARKET_TRANSITION_AUXILIARY_WIDTH
            or not torch.isfinite(targets).all()
        ):
            raise ValueError("invalid temporal impact target layout")
        row_weights = self._market_transition_row_weight(targets)
        node_count = target_batch.node_features.shape[0]
        if target_batch.graph_index is None:
            if row_weights.shape[0] != 1:
                raise ValueError(
                    "graph_index is required for multiple temporal impact rows"
                )
            groups = torch.zeros(
                node_count,
                dtype=torch.long,
                device=target_batch.node_features.device,
            )
        else:
            groups = target_batch.graph_index.to(
                device=target_batch.node_features.device,
                dtype=torch.long,
            )
            if groups.shape != (node_count,):
                raise ValueError("graph_index must contain one value per node")
            if node_count and (
                int(groups.min().item()) < 0
                or int(groups.max().item()) >= row_weights.shape[0]
            ):
                raise ValueError(
                    "graph_index does not align with temporal impact rows"
                )
        return row_weights[groups]

    def _blend_temporal_impact_loss(
        self,
        unweighted_loss: Tensor,
        weighted_loss: Tensor,
    ) -> Tensor:
        mix = self.temporal_impact_loss_mix
        return unweighted_loss + mix * (weighted_loss - unweighted_loss)

    @staticmethod
    def _batch_correlation_loss(prediction: Tensor, target: Tensor) -> Tensor:
        if prediction.numel() < 3:
            return prediction.new_tensor(0.0)
        prediction = prediction - prediction.mean()
        target = target - target.mean()
        denominator = prediction.square().sum().sqrt() * target.square().sum().sqrt()
        if float(denominator.detach()) <= 1e-12:
            return prediction.new_tensor(0.0)
        return 1.0 - (prediction * target).sum() / denominator.clamp_min(1e-12)

    def _downstream_transition_loss(
        self,
        z_context: Tensor,
        z_pred: Tensor,
        context_batch: GraphBatch,
        target_batch: GraphBatch,
        rollout_steps: int,
    ) -> Tensor:
        if self.downstream_transition_loss_weight <= 0.0:
            return z_pred.new_tensor(0.0)
        if target_batch.target_market_transition is None:
            raise ValueError(
                "downstream transition loss requires target_market_transition"
            )
        supervised_nodes = self._supervision_node_mask(context_batch)
        context_available = (
            torch.ones_like(context_batch.node_features, dtype=torch.bool)
            if context_batch.available_mask is None
            else context_batch.available_mask > 0.5
        )
        stock_pool_mask = supervised_nodes & context_available.any(dim=-1)
        outputs = self.predict_downstream_transition(
            z_context,
            z_pred,
            supervised_nodes,
            context_batch.graph_index,
            rollout_steps,
            stock_pool_mask=stock_pool_mask,
        )
        targets = target_batch.target_market_transition.to(
            device=z_pred.device, dtype=z_pred.dtype
        )
        if targets.shape != outputs.shape or targets.shape[1] != (
            MARKET_TRANSITION_AUXILIARY_WIDTH
        ):
            raise ValueError(
                "target_market_transition must align with transition outputs"
            )
        if not torch.isfinite(targets).all():
            raise ValueError("target_market_transition must be finite")

        family_count = len(MARKET_TRANSITION_AUXILIARY_FAMILIES)
        predicted_family = F.softplus(outputs[:, :family_count])
        target_family = targets[:, :family_count]
        row_weight = self._market_transition_row_weight(targets)
        regression_rows = F.smooth_l1_loss(
            predicted_family,
            target_family,
            reduction="none",
        ).mean(dim=1)
        regression = (regression_rows * row_weight).sum() / row_weight.sum().clamp_min(
            1e-12
        )
        rank_losses = [
            self._batch_correlation_loss(
                predicted_family[:, index], target_family[:, index]
            )
            for index in range(family_count)
        ]
        rank = torch.stack(rank_losses).mean()
        family_labels = targets[:, family_count : 2 * family_count]
        family_event = self._binary_focal_loss(
            outputs[:, family_count : 2 * family_count],
            family_labels,
            row_weight,
        )
        broad_selloff = self._binary_focal_loss(
            outputs[:, 2 * family_count],
            targets[:, 2 * family_count],
            row_weight,
        )
        systemic_event = self._binary_focal_loss(
            outputs[:, -1], targets[:, -1], row_weight
        )
        return (
            0.35 * regression
            + 0.20 * rank
            + 0.25 * family_event
            + 0.10 * broad_selloff
            + 0.10 * systemic_event
        )


    def _latent_regularizers(self, z: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        """VICReg-style anti-collapse terms + collapse diagnostics on a latent batch.

        variance: hinge(gamma - std_d) averaged over dims -- keeps every dimension
        expressive, which is exactly what a cosine-alignment objective can destroy.
        covariance: mean squared off-diagonal covariance -- decorrelates dimensions.
        Diagnostics (always returned, no grad cost worth avoiding): mean per-dim std
        and the variance participation ratio (effective dimensionality, 1.0 = collapsed
        to one direction, D = isotropic).
        """
        if z.dim() != 2 or z.shape[0] < 2:
            zero = z.new_tensor(0.0)
            return zero, zero, zero, zero
        zc = z - z.mean(dim=0, keepdim=True)
        n, d = zc.shape
        var = zc.var(dim=0, unbiased=False)
        std = torch.sqrt(var + 1e-6)
        var_loss = F.relu(self.latent_variance_target - std).mean()
        cov = (zc.T @ zc) / max(n - 1, 1)
        off = cov - torch.diag_embed(torch.diagonal(cov))
        cov_loss = off.pow(2).sum() / d
        # scale-invariant participation ratio: normalize the variance spectrum first,
        # otherwise a tiny-variance (collapsed) batch hits the epsilon floor and reports
        # a misleadingly small effective dimension.
        share = var / var.sum().clamp_min(1e-12)
        part_ratio = 1.0 / share.pow(2).sum().clamp_min(1e-12)
        return var_loss, cov_loss, std.mean().detach(), part_ratio.detach()

    def _current_imputation_loss(
        self,
        context_batch: GraphBatch,
        z_context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        if self.current_imputation_loss_weight <= 0.0:
            zero = z_context.new_tensor(0.0)
            return zero, zero
        available_mask = (
            torch.ones_like(context_batch.node_features, dtype=torch.bool)
            if context_batch.available_mask is None
            else context_batch.available_mask > 0.5
        )
        supervised_nodes = self._supervision_node_mask(context_batch)
        hidden_mask = (
            (context_batch.feature_mask < 0.5)
            & available_mask
            & supervised_nodes.unsqueeze(-1)
        )
        if not hidden_mask.any():
            zero = z_context.new_tensor(0.0)
            return zero, zero
        prediction = self.state_head(z_context)
        loss = self._state_reconstruction_loss(
            prediction,
            context_batch.node_features,
            hidden_mask,
        )
        mae = (
            prediction[hidden_mask] - context_batch.node_features[hidden_mask]
        ).abs().mean()
        return loss, mae

    def _hidden_completion_loss(
        self,
        context_batch: GraphBatch,
        current_context: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Estimate structurally-hidden state (t's flow) from the completion rep.

        Reads `current_context` -- the graph-on encoding the imputation path uses
        -- not the rollout, because the hidden is a property of NOW. Scored
        against the next-disclosed value carried on the context batch. Masked to
        supervised nodes with a finite target, so a node without a valid flow
        disclosure contributes nothing.
        """

        if self.hidden_completion_loss_weight <= 0.0 or self.hidden_completion_head is None:
            zero = current_context.new_tensor(0.0)
            return zero, zero
        target = context_batch.hidden_target
        if target is None:
            zero = current_context.new_tensor(0.0)
            return zero, zero
        target = target.to(device=current_context.device, dtype=current_context.dtype)
        prediction = self.hidden_completion_head(current_context)
        supervised = self._supervision_node_mask(context_batch).unsqueeze(-1)
        valid = torch.isfinite(target) & supervised
        if not valid.any():
            zero = current_context.new_tensor(0.0)
            return zero, zero
        safe_target = torch.where(valid, target, torch.zeros_like(target))
        errors = F.smooth_l1_loss(prediction, safe_target, reduction="none")
        loss = errors[valid].mean()
        mae = (prediction[valid] - target[valid]).abs().mean()
        return loss, mae

    def _current_imputation_context(
        self,
        context_batch: GraphBatch,
        temporal_context: Tensor,
    ) -> Tensor:
        if (
            self.temporal_graph_neighbor_scale == self.graph_neighbor_scale
            and self.temporal_stock_edge_scale == 1.0
        ):
            return temporal_context
        return self.encode_context(context_batch)

    def encode_context(
        self,
        batch: GraphBatch,
        neighbor_scale: Optional[float] = None,
    ) -> Tensor:
        return self.context_encoder(
            self._context_input(
                batch.node_features,
                batch.feature_mask,
                supervision_node_mask=batch.supervision_node_mask,
                graph_index=batch.graph_index,
            ),
            edge_index=batch.edge_index,
            edge_weight=batch.edge_weight,
            neighbor_scale=neighbor_scale,
            sequence=batch.node_sequence,
        )

    def encode_temporal_context(self, batch: GraphBatch) -> Tensor:
        edge_index, edge_weight = self._temporal_graph_edges(batch)
        return self.context_encoder(
            self._context_input(
                batch.node_features,
                batch.feature_mask,
                supervision_node_mask=batch.supervision_node_mask,
                graph_index=batch.graph_index,
            ),
            edge_index=edge_index,
            edge_weight=edge_weight,
            neighbor_scale=self.temporal_graph_neighbor_scale,
            sequence=batch.node_sequence,
        )

    def _temporal_graph_edges(
        self,
        batch: GraphBatch,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Optionally remove stock-to-stock edges while retaining external links."""

        if (
            self.temporal_stock_edge_scale == 1.0
            or batch.supervision_node_mask is None
            or batch.edge_index.numel() == 0
        ):
            return batch.edge_index, batch.edge_weight
        stock_nodes = batch.supervision_node_mask.to(batch.edge_index.device) > 0.5
        source, destination = batch.edge_index
        stock_to_stock = stock_nodes[source] & stock_nodes[destination]
        if self.temporal_stock_edge_scale == 0.0:
            keep = ~stock_to_stock
            return (
                batch.edge_index[:, keep],
                None if batch.edge_weight is None else batch.edge_weight[keep],
            )
        edge_weight = (
            torch.ones(
                batch.edge_index.shape[1],
                dtype=batch.node_features.dtype,
                device=batch.edge_index.device,
            )
            if batch.edge_weight is None
            else batch.edge_weight.to(
                device=batch.edge_index.device,
                dtype=batch.node_features.dtype,
            )
        )
        scale = torch.where(
            stock_to_stock,
            edge_weight.new_tensor(self.temporal_stock_edge_scale),
            edge_weight.new_tensor(1.0),
        )
        return batch.edge_index, edge_weight * scale

    def _predict_latent(self, latent: Tensor) -> Tensor:
        predicted = self.predictor(latent)
        if self.normalize_predictor_output:
            predicted = F.normalize(predicted, p=2, dim=-1) * (
                float(predicted.shape[-1]) ** 0.5
            )
        return predicted

    @torch.no_grad()
    def encode_target(
        self,
        batch: GraphBatch,
        neighbor_scale: Optional[float] = None,
    ) -> Tensor:
        return self.target_encoder(
            self._target_input(
                batch.node_features,
                batch.available_mask,
                supervision_node_mask=batch.supervision_node_mask,
                graph_index=batch.graph_index,
            ),
            edge_index=batch.edge_index,
            edge_weight=batch.edge_weight,
            neighbor_scale=neighbor_scale,
            sequence=batch.node_sequence,
        )

    @torch.no_grad()
    def encode_temporal_target(self, batch: GraphBatch) -> Tensor:
        available = (
            torch.ones_like(batch.node_features)
            if batch.available_mask is None
            else batch.available_mask.to(dtype=batch.node_features.dtype)
        )
        target_mask = available * (
            self.temporal_state_feature_weights > 0.0
        ).to(dtype=batch.node_features.dtype).unsqueeze(0)
        edge_index, edge_weight = self._temporal_graph_edges(batch)
        return self.target_encoder(
            self._target_input(
                batch.node_features,
                target_mask,
                supervision_node_mask=batch.supervision_node_mask,
                graph_index=batch.graph_index,
            ),
            edge_index=edge_index,
            edge_weight=edge_weight,
            neighbor_scale=self.temporal_graph_neighbor_scale,
            sequence=batch.node_sequence,
        )

    def forward(self, batch: GraphBatch) -> Dict[str, Tensor]:
        z_context = self.encode_context(batch)
        z_pred = self._predict_latent(z_context)
        x_pred = self.state_head(z_pred)
        z_target = self.encode_target(batch)

        return {
            "z_context": z_context,
            "z_pred": z_pred,
            "z_target": z_target,
            "state_pred": x_pred,
        }

    def loss(self, batch: GraphBatch) -> Tuple[Tensor, Dict[str, float]]:
        out = self.forward(batch)
        available_mask = (
            torch.ones_like(batch.node_features, dtype=torch.bool)
            if batch.available_mask is None
            else batch.available_mask > 0.5
        )
        supervised_nodes = self._supervision_node_mask(batch)
        hidden_feature_mask = (batch.feature_mask < 0.5) & available_mask
        supervised_hidden_feature_mask = hidden_feature_mask & supervised_nodes.unsqueeze(-1)
        target_node_mask = supervised_hidden_feature_mask.any(dim=-1)

        if target_node_mask.any():
            latent_loss = 1.0 - F.cosine_similarity(
                F.normalize(out["z_pred"][target_node_mask], dim=-1),
                F.normalize(out["z_target"][target_node_mask], dim=-1),
                dim=-1,
            ).mean()
        else:
            latent_loss = out["z_pred"].new_tensor(0.0)

        if supervised_nodes.any():
            lat_var, lat_cov, lat_std, lat_pr = self._latent_regularizers(
                out["z_pred"][supervised_nodes]
            )
        else:
            _z0 = out["z_pred"].new_tensor(0.0)
            lat_var = lat_cov = lat_std = lat_pr = _z0
        latent_loss = (
            latent_loss
            + self.latent_variance_weight * lat_var
            + self.latent_covariance_weight * lat_cov
        )

        if supervised_hidden_feature_mask.any():
            state_loss = self._state_reconstruction_loss(
                out["state_pred"],
                batch.node_features,
                supervised_hidden_feature_mask,
            )
            mae = (
                out["state_pred"][supervised_hidden_feature_mask]
                - batch.node_features[supervised_hidden_feature_mask]
            ).abs().mean()
        else:
            state_loss = out["z_pred"].new_tensor(0.0)
            mae = out["z_pred"].new_tensor(0.0)

        total = self.latent_loss_weight * latent_loss + self.state_loss_weight * state_loss
        metrics = {
            "loss": float(total.detach().cpu()),
            "latent_loss": float(latent_loss.detach().cpu()),
            "state_loss": float(state_loss.detach().cpu()),
            "masked_mae": float(mae.detach().cpu()),
            "latent_std": float(lat_std.detach().cpu()),
            "latent_participation": float(lat_pr.detach().cpu()),
        }
        return total, metrics

    def rollout_latent(self, z_context: Tensor, steps: int = 1) -> Tensor:
        """Roll a context latent forward with the predictor used as transition."""

        if steps < 1:
            raise ValueError("rollout steps must be >= 1")
        z = z_context
        for _ in range(steps):
            z = self._predict_latent(z)
        return z

    def predict_temporal_state(
        self,
        context_batch: GraphBatch,
        z_pred: Tensor,
        rollout_steps: int = 1,
        z_context: Optional[Tensor] = None,
    ) -> Tensor:
        """Predict future state with persistence for observed context cells.

        In residual mode, observed input cells use ``x_t + delta`` while
        hidden or unavailable cells use the direct imputation decoder. This is
        causal because the residual branch reads only the feature mask.
        """

        if self.temporal_state_mode == "horizon_residual_heads":
            key = str(int(rollout_steps))
            if key not in self.temporal_heads:
                raise ValueError(
                    "no horizon-specific state head for rollout_steps="
                    f"{rollout_steps}; configured={list(self.temporal_head_steps)}"
                )
            if self.temporal_state_context_skip:
                if z_context is None:
                    raise ValueError(
                        "temporal state context skip requires z_context"
                    )
                head_input = torch.cat([z_context, z_pred - z_context], dim=-1)
            else:
                head_input = z_pred
            direct, delta = self.temporal_heads[key](head_input).chunk(2, dim=-1)
            observed = context_batch.feature_mask > 0.5
            residual = context_batch.node_features + delta
            forecast = torch.where(observed, residual, direct)
            temporal_features = self.temporal_state_feature_weights > 0.0
            return torch.where(
                temporal_features.unsqueeze(0),
                forecast,
                context_batch.node_features,
            )

        direct = self.state_head(z_pred)
        if self.temporal_state_mode == "direct":
            temporal_features = self.temporal_state_feature_weights > 0.0
            return torch.where(
                temporal_features.unsqueeze(0),
                direct,
                context_batch.node_features,
            )
        if self.delta_head is None:
            raise RuntimeError("residual temporal state mode requires delta_head")
        observed = context_batch.feature_mask > 0.5
        if self.temporal_state_mode == "residual_mixed":
            residual_features = torch.ones(
                self.num_features,
                dtype=torch.bool,
                device=z_pred.device,
            )
        elif rollout_steps <= self.temporal_residual_short_steps:
            residual_features = self.hybrid_short_residual_feature_mask
        else:
            residual_features = self.hybrid_residual_feature_mask
        residual = context_batch.node_features + self.delta_head(z_pred)
        forecast = torch.where(observed & residual_features.unsqueeze(0), residual, direct)
        temporal_features = self.temporal_state_feature_weights > 0.0
        return torch.where(
            temporal_features.unsqueeze(0),
            forecast,
            context_batch.node_features,
        )

    def temporal_loss(
        self,
        context_batch: GraphBatch,
        target_batch: GraphBatch,
        rollout_steps: int = 1,
        target_horizon: Optional[int] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Predict a future graph latent/state from the current graph context.

        `rollout_steps=1` preserves the original single-step objective. Larger
        values train an explicit latent rollout: z_t -> z_{t+1} -> ... -> z_{t+T}.
        The target remains the EMA target encoder applied to the observed graph
        at the future timestamp.
        """

        z_context = self.encode_temporal_context(context_batch)
        z_target = self.encode_temporal_target(target_batch)
        z_pred = self.rollout_latent(z_context, steps=rollout_steps)
        state_pred = self.predict_temporal_state(
            context_batch,
            z_pred,
            rollout_steps=rollout_steps,
            z_context=z_context,
        )
        target_available = (
            torch.ones_like(target_batch.node_features, dtype=torch.bool)
            if target_batch.available_mask is None
            else target_batch.available_mask > 0.5
        )
        temporal_available = target_available & (
            self.temporal_state_feature_weights > 0.0
        ).unsqueeze(0)
        supervised_nodes = self._supervision_node_mask(target_batch) & temporal_available.any(dim=-1)
        impact_node_weights = self._temporal_impact_node_weights(target_batch)
        impact_weight_mean = (
            z_pred.new_tensor(1.0)
            if impact_node_weights is None
            else impact_node_weights.mean()
        )
        if supervised_nodes.any():
            latent_errors = 1.0 - F.cosine_similarity(
                F.normalize(z_pred[supervised_nodes], dim=-1),
                F.normalize(z_target[supervised_nodes], dim=-1),
                dim=-1,
            )
            latent_loss = latent_errors.mean()
            if impact_node_weights is not None:
                selected_impact_weights = impact_node_weights[supervised_nodes]
                impact_latent_loss = (
                    latent_errors * selected_impact_weights
                ).sum() / selected_impact_weights.sum().clamp_min(1e-12)
                latent_loss = self._blend_temporal_impact_loss(
                    latent_loss,
                    impact_latent_loss,
                )
        else:
            latent_loss = z_pred.new_tensor(0.0)
        if supervised_nodes.any():
            lat_var, lat_cov, lat_std, lat_pr = self._latent_regularizers(z_pred[supervised_nodes])
        else:
            _z0 = z_pred.new_tensor(0.0); lat_var = lat_cov = lat_std = lat_pr = _z0
        latent_loss = (
            latent_loss
            + self.latent_variance_weight * lat_var
            + self.latent_covariance_weight * lat_cov
        )
        supervised_target_available = (
            temporal_available & supervised_nodes.unsqueeze(-1)
        )
        if supervised_target_available.any():
            state_loss = self._state_reconstruction_loss(
                state_pred,
                target_batch.node_features,
                supervised_target_available,
                feature_weights=self.temporal_state_feature_weights,
            )
            if impact_node_weights is not None:
                impact_state_loss = self._state_reconstruction_loss(
                    state_pred,
                    target_batch.node_features,
                    supervised_target_available,
                    feature_weights=self.temporal_state_feature_weights,
                    node_weights=impact_node_weights,
                )
                state_loss = self._blend_temporal_impact_loss(
                    state_loss,
                    impact_state_loss,
                )
            mae = (
                state_pred[supervised_target_available]
                - target_batch.node_features[supervised_target_available]
            ).abs().mean()
        else:
            state_loss = z_pred.new_tensor(0.0)
            mae = z_pred.new_tensor(0.0)
        return_corr_loss = self._return_correlation_loss(
            state_pred,
            target_batch.node_features,
            supervised_target_available,
            target_batch.graph_index,
            target_horizon=target_horizon,
        )
        horizon = 1 if target_horizon is None else int(target_horizon)
        if self.entry_path_correlation_loss_weight > 0.0 and horizon != 1:
            raise ValueError(
                "multi-horizon entry path loss requires temporal_multi_loss with h1"
            )
        entry_path_corr_loss = self._entry_path_correlation_loss(
            state_pred,
            state_pred,
            target_batch,
            supervised_target_available,
            target_horizon=horizon,
        )
        downstream_auxiliary_loss = self._downstream_auxiliary_loss(
            z_context,
            z_pred,
            target_batch,
            rollout_steps=rollout_steps,
        )
        downstream_market_loss = self._downstream_market_loss(
            z_context,
            z_pred,
            target_batch,
            rollout_steps=rollout_steps,
        )
        downstream_transition_loss = self._downstream_transition_loss(
            z_context,
            z_pred,
            context_batch,
            target_batch,
            rollout_steps=rollout_steps,
        )
        current_context = self._current_imputation_context(
            context_batch,
            z_context,
        )
        current_imputation_loss, current_imputation_mae = self._current_imputation_loss(
            context_batch,
            current_context,
        )
        hidden_completion_loss, hidden_completion_mae = self._hidden_completion_loss(
            context_batch,
            current_context,
        )
        total = (
            self.latent_loss_weight * latent_loss
            + self.state_loss_weight * state_loss
            + self.return_correlation_loss_weight * return_corr_loss
            + self.entry_path_correlation_loss_weight * entry_path_corr_loss
            + self.downstream_auxiliary_loss_weight
            * downstream_auxiliary_loss
            + self.downstream_market_loss_weight * downstream_market_loss
            + self.downstream_transition_loss_weight
            * downstream_transition_loss
            + self.current_imputation_loss_weight
            * (1.0 if self.imputation_standalone else self.state_loss_weight)
            * current_imputation_loss
            + self.hidden_completion_loss_weight * hidden_completion_loss
        )
        metrics = {
            "loss": float(total.detach().cpu()),
            "latent_loss": float(latent_loss.detach().cpu()),
            "latent_std": float(lat_std.detach().cpu()),
            "latent_participation": float(lat_pr.detach().cpu()),
            "state_loss": float(state_loss.detach().cpu()),
            "masked_mae": float(mae.detach().cpu()),
            "return_corr_loss": float(return_corr_loss.detach().cpu()),
            "entry_path_corr_loss": float(entry_path_corr_loss.detach().cpu()),
            "downstream_auxiliary_loss": float(
                downstream_auxiliary_loss.detach().cpu()
            ),
            "downstream_market_loss": float(
                downstream_market_loss.detach().cpu()
            ),
            "downstream_transition_loss": float(
                downstream_transition_loss.detach().cpu()
            ),
            "current_imputation_loss": float(current_imputation_loss.detach().cpu()),
            "current_imputation_mae": float(current_imputation_mae.detach().cpu()),
            "hidden_completion_loss": float(hidden_completion_loss.detach().cpu()),
            "hidden_completion_mae": float(hidden_completion_mae.detach().cpu()),
            "temporal_impact_weight_mean": float(
                impact_weight_mean.detach().cpu()
            ),
            "rollout_steps": float(rollout_steps),
        }
        return total, metrics

    def temporal_multi_loss(
        self,
        context_batch: GraphBatch,
        target_batches: Sequence[GraphBatch],
        rollout_steps: Sequence[int],
        rollout_loss_weights: Optional[Sequence[float]] = None,
        target_horizons: Optional[Sequence[int]] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Aggregate temporal losses for several future offsets efficiently.

        Every target offset starts from the same observed graph at time ``t``.
        Encoding that graph separately for each horizon wastes most of the
        temporal pretraining compute and samples different dropout states for
        what should be one context. This method encodes it once, rolls the
        shared latent transition to the requested steps, and batches the
        EMA-teacher target encodings across future graph snapshots.
        """

        if len(target_batches) != len(rollout_steps):
            raise ValueError("target_batches and rollout_steps must have the same length")
        if not target_batches:
            raise ValueError("at least one target batch is required")
        if any(int(steps) < 1 for steps in rollout_steps):
            raise ValueError("rollout steps must be >= 1")
        horizons = (
            [int(steps) for steps in rollout_steps]
            if target_horizons is None
            else [int(horizon) for horizon in target_horizons]
        )
        if len(horizons) != len(target_batches):
            raise ValueError("target_horizons must match target_batches")
        if any(horizon < 1 for horizon in horizons):
            raise ValueError("target_horizons must be positive")
        if rollout_loss_weights is None:
            weights = [1.0] * len(target_batches)
        else:
            weights = [float(weight) for weight in rollout_loss_weights]
            if len(weights) != len(target_batches):
                raise ValueError("rollout_loss_weights must match target_batches")
            if any(weight < 0.0 for weight in weights) or sum(weights) <= 0.0:
                raise ValueError("rollout_loss_weights must be non-negative with a positive sum")

        context_nodes = context_batch.node_features.shape[0]
        if any(batch.node_features.shape[0] != context_nodes for batch in target_batches):
            raise ValueError("each target batch must contain the same graph count as the context batch")

        z_context = self.encode_temporal_context(context_batch)
        current_context = self._current_imputation_context(
            context_batch,
            z_context,
        )
        current_imputation_loss, current_imputation_mae = self._current_imputation_loss(
            context_batch,
            current_context,
        )
        hidden_completion_loss, hidden_completion_mae = self._hidden_completion_loss(
            context_batch,
            current_context,
        )
        requested_steps = {int(steps) for steps in rollout_steps}
        predicted_by_step: Dict[int, Tensor] = {}
        z_pred = z_context
        for step in range(1, max(requested_steps) + 1):
            z_pred = self._predict_latent(z_pred)
            if step in requested_steps:
                predicted_by_step[step] = z_pred

        merged_targets = merge_graph_batches(target_batches)
        z_targets = self.encode_temporal_target(merged_targets)
        state_predictions = [
            self.predict_temporal_state(
                context_batch,
                predicted_by_step[int(steps)],
                rollout_steps=int(steps),
                z_context=z_context,
            )
            for steps in rollout_steps
        ]
        if self.entry_path_correlation_loss_weight > 0.0:
            if 1 not in horizons:
                raise ValueError(
                    "entry path correlation loss requires horizon 1 in target_horizons"
                )
            next_open_prediction = state_predictions[horizons.index(1)]
        else:
            next_open_prediction = state_predictions[0]
        plan_predicted: Dict[int, Tensor] = {}
        plan_realized: Dict[int, Tensor] = {}
        plan_valid: Dict[int, Tensor] = {}
        losses = []
        metric_rows = []
        latent_diag_rows: list = []
        for index, (target_batch, steps, target_horizon, state_pred) in enumerate(
            zip(target_batches, rollout_steps, horizons, state_predictions)
        ):
            start = index * context_nodes
            stop = start + context_nodes
            target_available = (
                torch.ones_like(target_batch.node_features, dtype=torch.bool)
                if target_batch.available_mask is None
                else target_batch.available_mask > 0.5
            )
            temporal_available = target_available & (
                self.temporal_state_feature_weights > 0.0
            ).unsqueeze(0)
            supervised_nodes = (
                self._supervision_node_mask(target_batch)
                & temporal_available.any(dim=-1)
            )
            z_pred_for_step = predicted_by_step[int(steps)]
            z_target = z_targets[start:stop]
            impact_node_weights = self._temporal_impact_node_weights(
                target_batch
            )
            impact_weight_mean = (
                z_pred_for_step.new_tensor(1.0)
                if impact_node_weights is None
                else impact_node_weights.mean()
            )
            if supervised_nodes.any():
                latent_errors = 1.0 - F.cosine_similarity(
                    F.normalize(z_pred_for_step[supervised_nodes], dim=-1),
                    F.normalize(z_target[supervised_nodes], dim=-1),
                    dim=-1,
                )
                latent_loss = latent_errors.mean()
                if impact_node_weights is not None:
                    selected_impact_weights = impact_node_weights[
                        supervised_nodes
                    ]
                    impact_latent_loss = (
                        latent_errors * selected_impact_weights
                    ).sum() / selected_impact_weights.sum().clamp_min(1e-12)
                    latent_loss = self._blend_temporal_impact_loss(
                        latent_loss,
                        impact_latent_loss,
                    )
            else:
                latent_loss = z_pred_for_step.new_tensor(0.0)
            if supervised_nodes.any():
                lat_var, lat_cov, lat_std, lat_pr = self._latent_regularizers(
                    z_pred_for_step[supervised_nodes]
                )
            else:
                _z0 = z_pred_for_step.new_tensor(0.0)
                lat_var = lat_cov = lat_std = lat_pr = _z0
            latent_loss = (
                latent_loss
                + self.latent_variance_weight * lat_var
                + self.latent_covariance_weight * lat_cov
            )
            latent_diag_rows.append((lat_std, lat_pr))

            supervised_target_available = (
                temporal_available & supervised_nodes.unsqueeze(-1)
            )
            if supervised_target_available.any():
                state_loss = self._state_reconstruction_loss(
                    state_pred,
                    target_batch.node_features,
                    supervised_target_available,
                    feature_weights=self.temporal_state_feature_weights,
                )
                if impact_node_weights is not None:
                    impact_state_loss = self._state_reconstruction_loss(
                        state_pred,
                        target_batch.node_features,
                        supervised_target_available,
                        feature_weights=self.temporal_state_feature_weights,
                        node_weights=impact_node_weights,
                    )
                    state_loss = self._blend_temporal_impact_loss(
                        state_loss,
                        impact_state_loss,
                    )
                mae = (
                    state_pred[supervised_target_available]
                    - target_batch.node_features[supervised_target_available]
                ).abs().mean()
            else:
                state_loss = z_pred_for_step.new_tensor(0.0)
                mae = z_pred_for_step.new_tensor(0.0)

            return_corr_loss = self._return_correlation_loss(
                state_pred,
                target_batch.node_features,
                supervised_target_available,
                target_batch.graph_index,
                target_horizon=target_horizon,
            )
            entry_path_corr_loss = self._entry_path_correlation_loss(
                state_pred,
                next_open_prediction,
                target_batch,
                supervised_target_available,
                target_horizon=target_horizon,
            )
            downstream_auxiliary_loss = self._downstream_auxiliary_loss(
                z_context,
                z_pred_for_step,
                target_batch,
                rollout_steps=int(steps),
            )
            if self.downstream_plan_loss_weight > 0.0:
                self._collect_plan_horizon(
                    plan_predicted,
                    plan_realized,
                    plan_valid,
                    z_context=z_context,
                    z_pred=z_pred_for_step,
                    target_batch=target_batch,
                    rollout_steps=int(steps),
                )
            downstream_market_loss = self._downstream_market_loss(
                z_context,
                z_pred_for_step,
                target_batch,
                rollout_steps=int(steps),
            )
            downstream_transition_loss = self._downstream_transition_loss(
                z_context,
                z_pred_for_step,
                context_batch,
                target_batch,
                rollout_steps=int(steps),
            )
            total = (
                self.latent_loss_weight * latent_loss
                + self.state_loss_weight * state_loss
                + self.return_correlation_loss_weight * return_corr_loss
                + self.entry_path_correlation_loss_weight * entry_path_corr_loss
                + self.downstream_auxiliary_loss_weight
                * downstream_auxiliary_loss
                + self.downstream_market_loss_weight * downstream_market_loss
                + self.downstream_transition_loss_weight
                * downstream_transition_loss
            )
            losses.append(total)
            metric_rows.append(
                (
                    latent_loss,
                    state_loss,
                    mae,
                    return_corr_loss,
                    entry_path_corr_loss,
                    downstream_auxiliary_loss,
                    downstream_market_loss,
                    downstream_transition_loss,
                    impact_weight_mean,
                )
            )

        weight_tensor = torch.tensor(weights, dtype=losses[0].dtype, device=losses[0].device)
        weight_tensor = weight_tensor / weight_tensor.sum()
        temporal_loss = (torch.stack(losses) * weight_tensor).sum()
        loss = (
            temporal_loss
            + self.current_imputation_loss_weight
            * (1.0 if self.imputation_standalone else self.state_loss_weight)
            * current_imputation_loss
            + self.hidden_completion_loss_weight * hidden_completion_loss
        )
        plan_diagnostics: Dict[str, float] = {}
        if self.downstream_plan_loss_weight > 0.0 and plan_predicted:
            plan_loss, plan_diagnostics = plan_timing_loss(
                plan_predicted,
                plan_realized,
                plan_valid,
                temperature=self.plan_temperature,
                permute_seed=self.plan_permute_seed,
                buy_sell=self.plan_buy_sell,
            )
            loss = loss + self.downstream_plan_loss_weight * plan_loss
        metrics = {
            "plan_loss_weight": self.downstream_plan_loss_weight,
            "loss": float(loss.detach().cpu()),
            "latent_loss": float((torch.stack([row[0] for row in metric_rows]) * weight_tensor).sum().detach().cpu()),
            "latent_std": float(
                torch.stack([r[0] for r in latent_diag_rows]).mean().detach().cpu()
            ) if latent_diag_rows else 0.0,
            "latent_participation": float(
                torch.stack([r[1] for r in latent_diag_rows]).mean().detach().cpu()
            ) if latent_diag_rows else 0.0,
            "state_loss": float((torch.stack([row[1] for row in metric_rows]) * weight_tensor).sum().detach().cpu()),
            "masked_mae": float((torch.stack([row[2] for row in metric_rows]) * weight_tensor).sum().detach().cpu()),
            "return_corr_loss": float(
                (torch.stack([row[3] for row in metric_rows]) * weight_tensor).sum().detach().cpu()
            ),
            "entry_path_corr_loss": float(
                (torch.stack([row[4] for row in metric_rows]) * weight_tensor)
                .sum()
                .detach()
                .cpu()
            ),
            "downstream_auxiliary_loss": float(
                (torch.stack([row[5] for row in metric_rows]) * weight_tensor)
                .sum()
                .detach()
                .cpu()
            ),
            "downstream_market_loss": float(
                (torch.stack([row[6] for row in metric_rows]) * weight_tensor)
                .sum()
                .detach()
                .cpu()
            ),
            "downstream_transition_loss": float(
                (torch.stack([row[7] for row in metric_rows]) * weight_tensor)
                .sum()
                .detach()
                .cpu()
            ),
            "temporal_impact_weight_mean": float(
                (torch.stack([row[8] for row in metric_rows]) * weight_tensor)
                .sum()
                .detach()
                .cpu()
            ),
            "current_imputation_loss": float(current_imputation_loss.detach().cpu()),
            "current_imputation_mae": float(current_imputation_mae.detach().cpu()),
            "hidden_completion_loss": float(hidden_completion_loss.detach().cpu()),
            "hidden_completion_mae": float(hidden_completion_mae.detach().cpu()),
        }
        # plan_timing_loss computes these under no_grad every step. Without this
        # merge they are discarded, and an inert plan loss is indistinguishable
        # from a working one in the artifacts -- which is precisely what the
        # H6-1 contract's decision rule 2 exists to detect. Empty when the plan
        # loss is off, so the aux-only arm's metrics are unchanged.
        metrics.update(plan_diagnostics)
        return loss, metrics

    @torch.no_grad()
    def update_target_encoder(self, decay: Optional[float] = None) -> None:
        decay = self.ema_decay if decay is None else float(decay)
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1)")
        for target_param, context_param in zip(
            self.target_encoder.parameters(),
            self.context_encoder.parameters(),
        ):
            target_param.data.mul_(decay).add_(context_param.data, alpha=1.0 - decay)
        self.target_encoder.eval()

    @torch.no_grad()
    def infer_unobserved_state(self, batch: GraphBatch) -> Tensor:
        """Return estimated normalized feature values for every node/feature."""

        self.eval()
        z_context = self.encode_context(batch)
        if self.current_imputation_loss_weight > 0.0:
            return self.state_head(z_context)
        return self.state_head(self._predict_latent(z_context))


def make_feature_mask(
    node_features: Tensor,
    hide_ratio: float = 0.25,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Randomly hide feature cells while keeping at least one feature per node visible."""

    if not 0.0 < hide_ratio < 1.0:
        raise ValueError("hide_ratio must be between 0 and 1")

    mask = (torch.rand(node_features.shape, generator=generator, device=node_features.device) > hide_ratio).float()
    num_nodes, num_features = mask.shape
    for node_idx in range(num_nodes):
        if mask[node_idx].sum() == 0:
            keep_idx = torch.randint(num_features, (1,), generator=generator, device=node_features.device)
            mask[node_idx, keep_idx] = 1.0
    return mask



def _feature_group_map(feature_names: Sequence[str]) -> dict[str, list[int]]:
    groups: dict[str, list[int]] = {
        "returns": [],
        "trend": [],
        "risk": [],
        "liquidity": [],
        "market": [],
        "intraday": [],
        "news": [],
        "investor": [],
        "external": [],
    }
    for idx, name in enumerate(feature_names):
        if name.startswith("ext_"):
            groups["external"].append(idx)
        elif name.startswith("news_"):
            groups["news"].append(idx)
        elif name.startswith("investor_"):
            groups["investor"].append(idx)
        elif name.startswith("return_") or "relative_return" in name or "cs_rank_return" in name:
            groups["returns"].append(idx)
        elif name.startswith("ma") or "drawdown" in name or "breakout" in name:
            groups["trend"].append(idx)
        elif "volatility" in name or "range" in name:
            groups["risk"].append(idx)
        elif "volume" in name or "value" in name or "cs_rank_value" in name or "amihud" in name:
            groups["liquidity"].append(idx)
        elif name.startswith("market_"):
            groups["market"].append(idx)
        elif "gap_open" in name or "intraday" in name:
            groups["intraday"].append(idx)
    used = {idx for group in groups.values() for idx in group}
    other = [idx for idx in range(len(feature_names)) if idx not in used]
    if other:
        groups["other"] = other
    return {name: group for name, group in groups.items() if group}


def _feature_groups(feature_names: Sequence[str]) -> list[list[int]]:
    return list(_feature_group_map(feature_names).values())


def _randperm(n: int, generator: Optional[torch.Generator], device: torch.device) -> Tensor:
    return torch.randperm(n, generator=generator, device=device)


def make_structured_feature_mask(
    node_features: Tensor,
    feature_names: Sequence[str],
    hide_ratio: float = 0.30,
    strategy: str = "mixed",
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Hide economically meaningful state blocks instead of isolated cells.

    Strategies:
    - `feature_group`: hide whole modalities, such as returns, trend, risk, or
      liquidity, for every stock node.
    - `node_block`: hide most or all state for a subset of stock nodes so the
      model must infer them from graph neighbours.
    - `mixed`: combine feature-group and node-block masking.
    - `operational_mixed`: sample structured market-data failure and hard
      downstream scenarios while retaining some generic mixed masks.
    - `random_cell`: keep the legacy independent cell mask.
    """

    if strategy == "random_cell":
        return make_feature_mask(node_features, hide_ratio=hide_ratio, generator=generator)
    if strategy == "operational_mixed":
        return make_operational_feature_mask(
            node_features,
            feature_names=feature_names,
            hide_ratio=hide_ratio,
            generator=generator,
        )
    if strategy not in {"feature_group", "node_block", "mixed"}:
        raise ValueError(f"unknown mask strategy: {strategy}")
    if not 0.0 < hide_ratio < 1.0:
        raise ValueError("hide_ratio must be between 0 and 1")

    device = node_features.device
    num_nodes, num_features = node_features.shape
    mask = torch.ones_like(node_features)

    if strategy in {"feature_group", "mixed"}:
        groups = _feature_groups(feature_names)
        order = _randperm(len(groups), generator=generator, device=device).tolist()
        target_hidden_features = max(1, int(round(num_features * hide_ratio)))
        hidden: set[int] = set()
        for group_idx in order:
            hidden.update(groups[group_idx])
            if len(hidden) >= target_hidden_features:
                break
        if hidden:
            feature_idx = torch.tensor(sorted(hidden), dtype=torch.long, device=device)
            mask[:, feature_idx] = 0.0

    if strategy in {"node_block", "mixed"}:
        block_ratio = hide_ratio if strategy == "node_block" else min(0.50, hide_ratio * 0.75)
        num_hidden_nodes = max(1, int(round(num_nodes * block_ratio)))
        node_idx = _randperm(num_nodes, generator=generator, device=device)[:num_hidden_nodes]
        if strategy == "node_block":
            mask[node_idx, :] = 0.0
        else:
            # Mixed mode leaves a small anchor feature visible when possible;
            # the model still has to reconstruct most node state from neighbours.
            mask[node_idx, :] = 0.0
            anchor = 0
            for idx, name in enumerate(feature_names):
                if name.startswith("market_"):
                    anchor = idx
                    break
            mask[node_idx, anchor] = 1.0

    if bool((mask < 0.5).any()):
        return mask
    return make_feature_mask(node_features, hide_ratio=hide_ratio, generator=generator)


def make_operational_feature_mask(
    node_features: Tensor,
    feature_names: Sequence[str],
    hide_ratio: float = 0.30,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    """Sample masks that resemble decision-time sensor failures.

    The policy keeps 20% generic structured masks for representation coverage.
    The remainder covers whole-node outages, partial modality failures, stale
    price/liquidity sensors, and hard return/risk reconstruction. External
    nodes are restored by ``make_real_snapshot`` and are never supervision
    targets.
    """

    if not 0.0 < hide_ratio < 1.0:
        raise ValueError("hide_ratio must be between 0 and 1")
    if len(feature_names) != node_features.shape[1]:
        raise ValueError("feature_names must match node feature width")
    device = node_features.device
    num_nodes, _ = node_features.shape
    if num_nodes < 1:
        raise ValueError("node_features must contain at least one node")
    groups = _feature_group_map(feature_names)
    scenario = int(
        torch.randint(100, (1,), generator=generator, device=device).item()
    )
    if scenario < 20:
        return make_structured_feature_mask(
            node_features,
            feature_names=feature_names,
            hide_ratio=hide_ratio,
            strategy="mixed",
            generator=generator,
        )

    mask = torch.ones_like(node_features)
    anchor = next(
        (
            index
            for index, name in enumerate(feature_names)
            if name.startswith("market_")
        ),
        0,
    )

    def selected_nodes(ratio: float) -> Tensor:
        count = min(num_nodes, max(1, int(round(num_nodes * ratio))))
        return _randperm(num_nodes, generator=generator, device=device)[:count]

    if scenario < 40:
        nodes = selected_nodes(hide_ratio)
        mask[nodes, :] = 0.0
        mask[nodes, anchor] = 1.0
    elif scenario < 60:
        candidate_groups = [
            group
            for name, group in groups.items()
            if name not in {"market", "external", "other"}
        ]
        if not candidate_groups:
            return make_structured_feature_mask(
                node_features,
                feature_names=feature_names,
                hide_ratio=hide_ratio,
                strategy="node_block",
                generator=generator,
            )
        nodes = selected_nodes(min(0.85, max(0.35, hide_ratio * 2.0)))
        group_order = _randperm(
            len(candidate_groups), generator=generator, device=device
        ).tolist()
        hidden_width = 0
        target_width = max(1, int(round(node_features.shape[1] * hide_ratio)))
        for group_index in group_order:
            feature_index = torch.tensor(
                candidate_groups[group_index], dtype=torch.long, device=device
            )
            mask[nodes[:, None], feature_index[None, :]] = 0.0
            hidden_width += len(candidate_groups[group_index])
            if hidden_width >= target_width:
                break
    elif scenario < 80:
        stale_features = sorted(
            {
                feature
                for name in ("returns", "trend", "risk", "liquidity", "intraday")
                for feature in groups.get(name, [])
            }
        )
        if stale_features:
            nodes = selected_nodes(min(0.80, max(0.40, hide_ratio * 2.0)))
            features = torch.tensor(stale_features, dtype=torch.long, device=device)
            mask[nodes[:, None], features[None, :]] = 0.0
    else:
        hard_features = sorted(
            {
                feature
                for name in ("returns", "risk", "intraday")
                for feature in groups.get(name, [])
            }
        )
        if hard_features:
            nodes = selected_nodes(min(1.0, max(0.65, hide_ratio * 2.5)))
            features = torch.tensor(hard_features, dtype=torch.long, device=device)
            mask[nodes[:, None], features[None, :]] = 0.0

    fully_hidden = mask.sum(dim=1) == 0
    if fully_hidden.any():
        mask[fully_hidden, anchor] = 1.0
    if bool((mask < 0.5).any()):
        return mask
    return make_feature_mask(
        node_features,
        hide_ratio=hide_ratio,
        generator=generator,
    )
