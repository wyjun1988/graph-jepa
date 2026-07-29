from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence


SOURCE_PATHS = (
    "stock_v2/graph_jepa.py",
    "stock_v2/ops/signals.py",
    "scripts/run_real_backtest.py",
    "scripts/run_walk_forward_node_eval.py",
    "scripts/evaluate_node_prediction.py",
    "scripts/benchmark_direct_state_mlp.py",
    "scripts/compare_direct_state_mlp.py",
    "scripts/combine_direct_state_challenges.py",
    "scripts/gate_shadow_candidate.py",
    "scripts/select_path_objective_candidates.py",
    "scripts/summarize_checkpoint_epochs.py",
    "scripts/ensure_path_run_contract.py",
    "scripts/run_path_objective_pipeline_gpu.sh",
)

SCREEN_CANDIDATES = (
    "control_w1_c0_l1_noskip",
    "control_w1_c0_l1_skip",
    "control_w1_c0_l025_skip",
    "path_w4_p001_l025_skip",
    "path_w8_p0025_l025_noskip",
    "path_w8_p0025_l025_skip",
    "path_w12_p005_l025_skip",
)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_contract(
    root: Path,
    *,
    temporal_graph_neighbor_scale: float,
    temporal_stock_edge_scale: float,
    fold_panel_sha256: Sequence[str],
    fold_edge_sha256: Sequence[str],
    source_paths: Sequence[str] = SOURCE_PATHS,
) -> dict[str, Any]:
    if len(fold_panel_sha256) != 2 or len(fold_edge_sha256) != 2:
        raise ValueError("exactly two panel hashes and two edge hashes are required")
    return {
        "schema_version": 1,
        "immutable_inputs": {
            "fold_panel_sha256": list(fold_panel_sha256),
            "fold_edge_sha256": list(fold_edge_sha256),
        },
        "temporal_graph_neighbor_scale": float(temporal_graph_neighbor_scale),
        "temporal_stock_edge_scale": float(temporal_stock_edge_scale),
        "screen_shape": {"hidden_dim": 512, "layers": 6, "epochs": 8},
        "confirmation_shape": {"hidden_dim": 1024, "layers": 10, "epochs": 24},
        "screen_candidates": list(SCREEN_CANDIDATES),
        "source_sha256": {
            path: sha256_file(root / path)
            for path in source_paths
        },
    }


def ensure_contract(path: Path, contract: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(f"run contract mismatch; refusing stale resume: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze source, data, and architecture identity for a path run."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", action="append", required=True)
    parser.add_argument("--temporal-graph-neighbor-scale", type=float, required=True)
    parser.add_argument("--temporal-stock-edge-scale", type=float, required=True)
    parser.add_argument("--fold-panel-sha256", action="append", required=True)
    parser.add_argument("--fold-edge-sha256", action="append", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    contract = build_contract(
        root,
        temporal_graph_neighbor_scale=args.temporal_graph_neighbor_scale,
        temporal_stock_edge_scale=args.temporal_stock_edge_scale,
        fold_panel_sha256=args.fold_panel_sha256,
        fold_edge_sha256=args.fold_edge_sha256,
    )
    for output in args.output:
        ensure_contract(root / output, contract)
    print(hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest())


if __name__ == "__main__":
    main()
