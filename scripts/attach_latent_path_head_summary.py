from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from typing import Any


def significance_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "rows": int(row["rows"]),
        "mean": float(row["mean"]),
        "mean_target_corr": float(row["mean"]),
        "newey_west_lag": int(row["newey_west_lag"]),
        "newey_west_standard_error": float(row["newey_west_standard_error"]),
        "newey_west_t_stat": float(row["newey_west_t"]),
        "positive_day_fraction": float(row["positive_fraction"]),
    }


def attach_summary(
    node_summary: dict[str, Any],
    head_summary: dict[str, Any],
) -> dict[str, Any]:
    if head_summary.get("status") != "complete":
        raise ValueError("latent trajectory path head is not complete")
    if int(node_summary.get("eval_steps", 0)) != int(head_summary.get("test_dates", -1)):
        raise ValueError("node and path-head evaluation windows have different lengths")
    node_horizons = set((node_summary.get("future_rollout_by_horizon") or {}).keys())
    head_horizons = set((head_summary.get("horizons") or {}).keys())
    if node_horizons != head_horizons:
        raise ValueError("node and path-head summaries have different horizons")
    result = deepcopy(node_summary)
    result["realized_entry_path_correlation_significance"] = {
        horizon: significance_row(head_summary["horizons"][horizon]["all_stock"])
        for horizon in sorted(head_horizons, key=int)
    }
    liquidity = deepcopy(result.get("realized_entry_path_liquidity_significance") or {})
    liquidity["top300"] = {
        horizon: significance_row(head_summary["horizons"][horizon]["top300"])
        for horizon in sorted(head_horizons, key=int)
    }
    result["realized_entry_path_liquidity_significance"] = liquidity
    result["latent_trajectory_path_head"] = {
        "role": head_summary.get("role"),
        "parent_model_sha256": head_summary.get("parent_model_sha256"),
        "train_data_manifest_sha256": head_summary.get("train_data_manifest_sha256"),
        "train_edge_manifest_sha256": head_summary.get("train_edge_manifest_sha256"),
        "evaluation_seed": head_summary.get("evaluation_seed"),
        "latent_blend_weight": head_summary.get("latent_blend_weight"),
        "weighted_path_ic": head_summary.get("weighted_path_ic"),
        "fold2_used_for_selection": head_summary.get("fold2_used_for_selection"),
        "live_orders_allowed": False,
    }
    result["entry_path_source"] = "latent_trajectory_residual_head"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Attach a contract-bound latent path-head result to a node summary."
    )
    parser.add_argument("--node-summary", required=True)
    parser.add_argument("--head-summary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    node = json.loads(Path(args.node_summary).read_text(encoding="utf-8"))
    head = json.loads(Path(args.head_summary).read_text(encoding="utf-8"))
    result = attach_summary(node, head)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"wrote {output}")


if __name__ == "__main__":
    main()
