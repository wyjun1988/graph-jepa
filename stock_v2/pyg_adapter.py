from __future__ import annotations

from collections.abc import Sequence

import torch

from stock_v2.graph_jepa import GraphBatch


def _pyg_types():
    try:
        from torch_geometric.data import Batch, Data
    except ImportError as exc:
        raise RuntimeError(
            "PyG support requires torch-geometric; install requirements-pyg.txt"
        ) from exc
    return Batch, Data


def graph_batch_to_pyg(batch: GraphBatch):
    """Convert one stock graph snapshot to a PyG Data object without copying tensors."""

    _Batch, Data = _pyg_types()
    if batch.node_features.ndim != 2 or batch.feature_mask.shape != batch.node_features.shape:
        raise ValueError("node features and feature mask must be aligned 2D tensors")
    if batch.edge_index.ndim != 2 or batch.edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, edge_count]")
    if batch.graph_index is not None:
        graph_ids = torch.unique(batch.graph_index)
        if len(graph_ids) > 1:
            raise ValueError("graph_batch_to_pyg accepts one snapshot, not a merged graph batch")
    data = Data(
        x=batch.node_features,
        edge_index=batch.edge_index,
        feature_mask=batch.feature_mask,
        num_nodes=int(batch.node_features.shape[0]),
    )
    optional = {
        "edge_weight": batch.edge_weight,
        "available_mask": batch.available_mask,
        "supervision_node_mask": batch.supervision_node_mask,
        "target_entry_path": batch.target_entry_path,
        "target_downstream": batch.target_downstream,
        "target_market_transition": batch.target_market_transition,
    }
    for name, value in optional.items():
        if value is not None:
            setattr(data, name, value)
    return data


def pyg_to_graph_batch(data) -> GraphBatch:
    """Convert PyG Data or Batch back to the model's GraphBatch contract."""

    Batch, _Data = _pyg_types()
    if not hasattr(data, "x") or not hasattr(data, "feature_mask"):
        raise ValueError("PyG data must contain x and feature_mask")
    graph_index = data.batch if isinstance(data, Batch) and hasattr(data, "batch") else None

    def optional(name: str):
        return getattr(data, name) if hasattr(data, name) else None

    return GraphBatch(
        node_features=data.x,
        feature_mask=data.feature_mask,
        edge_index=data.edge_index,
        edge_weight=optional("edge_weight"),
        available_mask=optional("available_mask"),
        supervision_node_mask=optional("supervision_node_mask"),
        graph_index=graph_index,
        target_entry_path=optional("target_entry_path"),
        target_downstream=optional("target_downstream"),
        target_market_transition=optional("target_market_transition"),
    )


def graph_batches_to_pyg(batches: Sequence[GraphBatch]):
    """Use PyG's standard disjoint batching for a sequence of graph snapshots."""

    Batch, _Data = _pyg_types()
    if not batches:
        raise ValueError("batches must not be empty")
    return Batch.from_data_list([graph_batch_to_pyg(batch) for batch in batches])
