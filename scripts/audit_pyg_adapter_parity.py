from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
import torch_geometric

from scripts.benchmark_direct_baselines import _edge_settings, evaluator_namespace
from scripts.evaluate_node_prediction import build_features_from_ckpt
from stock_v2.graph_jepa import merge_graph_batches
from stock_v2.pyg_adapter import graph_batches_to_pyg, pyg_to_graph_batch
from stock_v2.real_features import make_real_snapshot


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit real GraphBatch/PyG Data batching parity for a checkpoint."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--offsets-from-end", default="12,11")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "graph_jepa_real.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    features, checkpoint_args = build_features_from_ckpt(
        checkpoint, evaluator_namespace(args)
    )
    offsets = [int(value.strip()) for value in args.offsets_from_end.split(",")]
    if len(offsets) < 2 or any(value <= 0 for value in offsets):
        raise ValueError("offsets-from-end requires at least two positive integers")
    steps = [len(features.dates) - value for value in offsets]
    if min(steps) < int(checkpoint_args.get("edge_window", 60)):
        raise ValueError("selected step does not have enough edge history")
    snapshots = [
        make_real_snapshot(
            features,
            step,
            **_edge_settings(checkpoint_args),
            full_observation=True,
        )
        for step in steps
    ]
    native = merge_graph_batches(snapshots)
    pyg = pyg_to_graph_batch(graph_batches_to_pyg(snapshots))
    fields = (
        "node_features",
        "feature_mask",
        "edge_index",
        "edge_weight",
        "available_mask",
        "supervision_node_mask",
        "graph_index",
    )
    checks = {
        name: bool(torch.equal(getattr(native, name), getattr(pyg, name)))
        for name in fields
    }
    payload = {
        "status": "pass" if all(checks.values()) else "fail",
        "role": "research_only_pyg_adapter_real_snapshot_parity",
        "checkpoint": str(checkpoint_path),
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "steps": steps,
        "dates": [str(features.dates[step].date()) for step in steps],
        "nodes_per_snapshot": int(features.node_count),
        "stocks": int(features.tradable_count),
        "features": int(len(features.feature_names)),
        "edges_per_snapshot": [int(value.edge_index.shape[1]) for value in snapshots],
        "checks": checks,
        "live_orders_allowed": False,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload))
    if payload["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
