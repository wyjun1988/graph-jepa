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
    "scripts/summarize_checkpoint_epochs.py",
    "scripts/ensure_path_confirmation_contract.py",
    "scripts/run_path_confirmation_gpu.sh",
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

OBJECTIVES: dict[str, dict[str, Any]] = {
    "control_w1_c0_l1_noskip": {
        "return_state_weight": 1.0,
        "entry_path_correlation_weight": 0.0,
        "latent_weight": 1.0,
        "context_skip": False,
    },
    "control_w1_c0_l1_skip": {
        "return_state_weight": 1.0,
        "entry_path_correlation_weight": 0.0,
        "latent_weight": 1.0,
        "context_skip": True,
    },
    "control_w1_c0_l025_skip": {
        "return_state_weight": 1.0,
        "entry_path_correlation_weight": 0.0,
        "latent_weight": 0.25,
        "context_skip": True,
    },
    "path_w4_p001_l025_skip": {
        "return_state_weight": 4.0,
        "entry_path_correlation_weight": 0.01,
        "latent_weight": 0.25,
        "context_skip": True,
    },
    "path_w8_p0025_l025_noskip": {
        "return_state_weight": 8.0,
        "entry_path_correlation_weight": 0.025,
        "latent_weight": 0.25,
        "context_skip": False,
    },
    "path_w8_p0025_l025_skip": {
        "return_state_weight": 8.0,
        "entry_path_correlation_weight": 0.025,
        "latent_weight": 0.25,
        "context_skip": True,
    },
    "path_w12_p005_l025_skip": {
        "return_state_weight": 12.0,
        "entry_path_correlation_weight": 0.05,
        "latent_weight": 0.25,
        "context_skip": True,
    },
}


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_contract(
    root: Path,
    *,
    screen_selection: Path,
    screen_contract: Path,
    expected_selected_label: str,
    fold_panel_sha256: Sequence[str],
    fold_edge_sha256: Sequence[str],
    temporal_graph_neighbor_scale: float,
    temporal_stock_edge_scale: float,
    source_paths: Sequence[str] = SOURCE_PATHS,
) -> dict[str, Any]:
    if len(fold_panel_sha256) != 2 or len(fold_edge_sha256) != 2:
        raise ValueError("exactly two panel hashes and two edge hashes are required")
    selection = json.loads(screen_selection.read_text(encoding="utf-8"))
    selected_label = str(selection.get("selected_label") or "")
    if selected_label != expected_selected_label:
        raise ValueError(
            f"screen selected {selected_label!r}, expected {expected_selected_label!r}"
        )
    candidate_labels = tuple(
        str(row.get("label") or "") for row in selection.get("candidates", [])
    )
    if candidate_labels != SCREEN_CANDIDATES:
        raise ValueError("screen selection candidate order does not match the frozen screen")
    if selected_label not in OBJECTIVES:
        raise ValueError(f"unknown selected objective: {selected_label}")
    return {
        "schema_version": 1,
        "scope": "confirmation_only_after_frozen_fold1_screen",
        "screen": {
            "selection_path": str(screen_selection.relative_to(root)),
            "selection_sha256": sha256_file(screen_selection),
            "run_contract_path": str(screen_contract.relative_to(root)),
            "run_contract_sha256": sha256_file(screen_contract),
            "selected_label": selected_label,
            "candidate_labels": list(candidate_labels),
        },
        "immutable_inputs": {
            "fold_panel_sha256": list(fold_panel_sha256),
            "fold_edge_sha256": list(fold_edge_sha256),
        },
        "temporal_graph_neighbor_scale": float(temporal_graph_neighbor_scale),
        "temporal_stock_edge_scale": float(temporal_stock_edge_scale),
        "selected_objective": dict(OBJECTIVES[selected_label]),
        "confirmation": {
            "folds": ["2023-12-29:2024-12-30", "2024-12-30:2026-07-10"],
            "hidden_dim": 1024,
            "layers": 10,
            "epochs": 24,
            "checkpoint_epochs": [8, 16, 24],
            "train_batch_size": 8,
            "training_seed": 17,
            "evaluation_seed": 17,
            "evaluation_max_steps": 0,
        },
        "safety": {
            "maximum_scope": "read_only_shadow",
            "live_orders_allowed": False,
        },
        "source_sha256": {
            path: sha256_file(root / path)
            for path in source_paths
        },
    }


def ensure_contract(path: Path, contract: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError(f"confirmation contract mismatch; refusing stale resume: {path}")
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
        description="Bind a confirmation run to a frozen Fold 1 path screen."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--output", required=True)
    parser.add_argument("--screen-selection", required=True)
    parser.add_argument("--screen-contract", required=True)
    parser.add_argument("--expected-selected-label", required=True)
    parser.add_argument("--fold-panel-sha256", action="append", required=True)
    parser.add_argument("--fold-edge-sha256", action="append", required=True)
    parser.add_argument("--temporal-graph-neighbor-scale", type=float, required=True)
    parser.add_argument("--temporal-stock-edge-scale", type=float, required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    contract = build_contract(
        root,
        screen_selection=(root / args.screen_selection).resolve(),
        screen_contract=(root / args.screen_contract).resolve(),
        expected_selected_label=args.expected_selected_label,
        fold_panel_sha256=args.fold_panel_sha256,
        fold_edge_sha256=args.fold_edge_sha256,
        temporal_graph_neighbor_scale=args.temporal_graph_neighbor_scale,
        temporal_stock_edge_scale=args.temporal_stock_edge_scale,
    )
    output = (root / args.output).resolve()
    ensure_contract(output, contract)
    print(hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest())


if __name__ == "__main__":
    main()
