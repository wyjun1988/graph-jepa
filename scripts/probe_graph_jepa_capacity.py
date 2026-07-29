from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch

from stock_v2.graph_jepa import GraphBatch, StockGraphJEPA


def parse_int_list(value: str) -> list[int]:
    values = [int(item.strip()) for item in value.split(",") if item.strip()]
    if not values or any(item < 1 for item in values):
        raise ValueError("expected one or more positive integers")
    return values


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure Graph-JEPA training-step capacity for a node universe."
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--features", type=int, default=129)
    parser.add_argument("--stock-nodes", type=int, default=500)
    parser.add_argument("--external-nodes", type=int, default=11)
    parser.add_argument("--hidden-dim", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=10)
    parser.add_argument("--edge-top-k", type=int, default=6)
    parser.add_argument("--batches", default="8,16,24,32")
    parser.add_argument("--rollout-offsets", default="1,2,3,5,10")
    parser.add_argument("--mask-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=7)
    return parser.parse_args()


def batched_ring_edges(
    batch_size: int,
    node_count: int,
    edge_top_k: int,
    device: torch.device,
) -> torch.Tensor:
    offsets = torch.arange(1, edge_top_k + 1, dtype=torch.long)
    base = torch.arange(node_count, dtype=torch.long)
    src = base.repeat_interleave(edge_top_k)
    dst = (base[:, None] + offsets[None, :]).remainder(node_count).reshape(-1)
    single = torch.stack([src, dst])
    return torch.cat(
        [single + batch_index * node_count for batch_index in range(batch_size)],
        dim=1,
    ).to(device)


def memory_gib(device: torch.device) -> tuple[float | None, float | None]:
    if device.type != "cuda":
        return None, None
    return (
        torch.cuda.max_memory_allocated(device) / 1024**3,
        torch.cuda.max_memory_reserved(device) / 1024**3,
    )


def main() -> None:
    args = parse_args()
    if args.features < 1 or args.stock_nodes < 1 or args.hidden_dim < 1 or args.layers < 1:
        raise ValueError("features, stock-nodes, hidden-dim, and layers must be positive")
    if args.external_nodes < 0 or args.edge_top_k < 1:
        raise ValueError("external-nodes must be nonnegative and edge-top-k must be positive")
    if not 0.0 <= args.mask_ratio < 1.0:
        raise ValueError("mask-ratio must be in [0, 1)")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA device requested but unavailable")
    batches = parse_int_list(args.batches)
    offsets = parse_int_list(args.rollout_offsets)
    node_count = args.stock_nodes + args.external_nodes
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    feature_names = ["return_1d"] + [
        f"feature_{index}" for index in range(args.features - 1)
    ]
    for batch_size in batches:
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
        gc.collect()
        edge_index = None
        context_features = None
        feature_mask = None
        available_mask = None
        context = None
        targets = None
        model = None
        optimizer = None
        loss = None
        total_nodes = batch_size * node_count
        row: dict[str, object] = {
            "batch_size": batch_size,
            "nodes_per_graph": node_count,
            "total_nodes": total_nodes,
            "rollout_offsets": offsets,
        }
        try:
            edge_index = batched_ring_edges(
                batch_size,
                node_count,
                args.edge_top_k,
                device,
            )
            context_features = torch.randn(total_nodes, args.features, device=device)
            feature_mask = (
                torch.rand(total_nodes, args.features, device=device) >= args.mask_ratio
            ).to(dtype=context_features.dtype)
            available_mask = torch.ones_like(context_features)
            context = GraphBatch(
                node_features=context_features,
                feature_mask=feature_mask,
                edge_index=edge_index,
                available_mask=available_mask,
            )
            targets = [
                GraphBatch(
                    node_features=torch.randn(total_nodes, args.features, device=device),
                    feature_mask=torch.ones_like(context_features),
                    edge_index=edge_index,
                    available_mask=available_mask,
                )
                for _ in offsets
            ]
            model = StockGraphJEPA(
                num_features=args.features,
                hidden_dim=args.hidden_dim,
                num_layers=args.layers,
                ema_decay=0.9995,
                state_loss_weight=0.35,
                temporal_state_mode="horizon_hybrid",
                temporal_residual_short_steps=2,
                feature_names=feature_names,
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            started = time.perf_counter()
            loss, metrics = model.temporal_multi_loss(context, targets, offsets)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            peak_allocated, peak_reserved = memory_gib(device)
            row.update(
                {
                    "status": "ok",
                    "step_sec": round(time.perf_counter() - started, 4),
                    "loss": round(float(metrics["loss"]), 6),
                    "peak_allocated_gib": (
                        round(peak_allocated, 3) if peak_allocated is not None else None
                    ),
                    "peak_reserved_gib": (
                        round(peak_reserved, 3) if peak_reserved is not None else None
                    ),
                }
            )
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            row.update({"status": "oom", "error": str(exc).splitlines()[0]})
        print(json.dumps(row), flush=True)
        edge_index = context_features = feature_mask = available_mask = None
        context = targets = model = optimizer = loss = None
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
