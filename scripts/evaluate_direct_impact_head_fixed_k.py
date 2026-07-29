from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_direct_baselines import (
    build_context_layout,
    evaluator_namespace,
    load_or_build_context_matrix,
    rows_for_steps,
)
from scripts.benchmark_direct_impact_head import DirectImpactHead
from scripts.benchmark_direct_state_mlp import amp_context, prepare_context_tensor, resolve_device
from scripts.benchmark_latent_trajectory_path_head import top_liquidity_mask
from scripts.evaluate_impact_head_fixed_k import (
    FIXED_K_METRICS,
    STRATEGIES,
    aggregate_horizon_selection_scores,
    cross_horizon_selection_contract,
    fixed_k_metric_row,
    fixed_k_metrics_for_selected,
    select_fixed_k,
    summarize_rows,
)
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.run_real_backtest import date_indices, parse_int_list, temporal_training_indices


def score_steps(
    head: DirectImpactHead,
    context: torch.Tensor,
    features,
    steps: np.ndarray,
    horizons: Sequence[int],
    counts: Sequence[int],
    liquidity_top_k: int,
    batch_dates: int,
    amp_enabled: bool,
) -> list[dict[str, Any]]:
    head.eval()
    date_count, stock_count, _ = context.shape
    signed_chunks: dict[int, list[np.ndarray]] = {int(h): [] for h in horizons}
    impact_chunks: dict[int, list[np.ndarray]] = {int(h): [] for h in horizons}
    with torch.inference_mode():
        for start in range(0, date_count, int(batch_dates)):
            selected = context[start : start + int(batch_dates)]
            count = len(selected)
            flat = selected.reshape(count * stock_count, -1)
            with amp_context(amp_enabled):
                for horizon in horizons:
                    signed, impact = head(flat, int(horizon))
                    signed_chunks[int(horizon)].append(
                        signed.float().cpu().numpy().reshape(count, stock_count)
                    )
                    impact_chunks[int(horizon)].append(
                        impact.float().cpu().numpy().reshape(count, stock_count)
                    )
    predictions = {
        int(horizon): (
            np.concatenate(signed_chunks[int(horizon)], axis=0),
            np.concatenate(impact_chunks[int(horizon)], axis=0),
        )
        for horizon in horizons
    }
    liquidity_index = features.feature_names.index("value_ma20_log")
    rows: list[dict[str, Any]] = []
    for position, step in enumerate(steps):
        scopes = {
            "all": np.ones(stock_count, dtype=bool),
            "top300": top_liquidity_mask(
                features.raw_features[int(step), :stock_count, liquidity_index],
                liquidity_top_k,
            ),
        }
        date_targets = {
            int(horizon): np.asarray(
                features.target_return_paths[int(horizon)][int(step), :stock_count],
                dtype=np.float64,
            )
            for horizon in horizons
        }
        signed_by_horizon = {
            int(horizon): predictions[int(horizon)][0][position]
            for horizon in horizons
        }
        impact_by_horizon = {
            int(horizon): predictions[int(horizon)][1][position]
            for horizon in horizons
        }
        for horizon in horizons:
            signed, impact = predictions[int(horizon)]
            target = date_targets[int(horizon)]
            for scope, valid in scopes.items():
                for count in counts:
                    for strategy in STRATEGIES:
                        rows.append(
                            {
                                "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                                "horizon": int(horizon),
                                "scope": scope,
                                "k": int(count),
                                "strategy": strategy,
                                "selection_mode": "per_horizon",
                                "selected_tickers": "",
                                **fixed_k_metric_row(
                                    signed[position],
                                    impact[position],
                                    target,
                                    valid,
                                    int(count),
                                    strategy,
                                ),
                            }
                        )
        for scope, scope_valid in scopes.items():
            aggregate_confidence, aggregate_impact, selection_valid = (
                aggregate_horizon_selection_scores(
                    signed_by_horizon,
                    impact_by_horizon,
                    scope_valid,
                    horizons,
                )
            )
            for count in counts:
                for strategy in STRATEGIES:
                    selected = select_fixed_k(
                        aggregate_confidence,
                        aggregate_impact,
                        selection_valid,
                        int(count),
                        strategy,
                    )
                    selected_tickers = "|".join(
                        str(features.tickers[int(index)]) for index in selected
                    )
                    for horizon in horizons:
                        target = date_targets[int(horizon)]
                        eligible = (
                            scope_valid
                            & np.isfinite(signed_by_horizon[int(horizon)])
                            & np.isfinite(impact_by_horizon[int(horizon)])
                            & np.isfinite(target)
                        )
                        rows.append(
                            {
                                "date": str(
                                    pd.Timestamp(features.dates[int(step)]).date()
                                ),
                                "horizon": int(horizon),
                                "scope": scope,
                                "k": int(count),
                                "strategy": strategy,
                                "selection_mode": "cross_horizon",
                                "selected_tickers": selected_tickers,
                                **fixed_k_metrics_for_selected(
                                    signed_by_horizon[int(horizon)],
                                    target,
                                    eligible,
                                    selected,
                                ),
                            }
                        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate an equal-objective direct impact head at fixed counts."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--head-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--counts", default="1,3,5")
    parser.add_argument("--liquidity-top-k", type=int, default=300)
    parser.add_argument("--batch-dates", type=int, default=32)
    parser.add_argument("--feature-workers", type=int, default=8)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--context-cache", default=None)
    args = parser.parse_args()

    device = resolve_device(str(args.device))
    horizons = parse_int_list(args.horizons)
    counts = parse_int_list(args.counts)
    model_dir = Path(args.model_dir)
    ckpt = torch.load(
        model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    artifact = torch.load(args.head_path, map_location="cpu", weights_only=False)
    if artifact.get("train_data_manifest_sha256") != ckpt.get(
        "train_data_manifest", {}
    ).get("sha256"):
        raise ValueError("direct impact head data manifest does not match parent")
    if [int(value) for value in artifact["horizons"]] != [int(value) for value in horizons]:
        raise ValueError("direct impact head horizon contract does not match request")
    features, ckpt_args = build_features_from_ckpt(ckpt, evaluator_namespace(args))
    train_end = str(ckpt_args["train_end"])
    max_horizon = max(horizons)
    edge_window = int(ckpt_args.get("edge_window", 60))
    train_indices = date_indices(features.dates, end=train_end)
    train_steps = temporal_training_indices(
        train_indices,
        edge_window=edge_window,
        max_rollout_offset=max_horizon,
        total_steps=len(features.dates),
    )
    test_steps = date_indices(
        features.dates,
        start=(pd.Timestamp(train_end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    test_steps = test_steps[
        (test_steps >= edge_window)
        & (test_steps <= len(features.dates) - 1 - max_horizon)
    ]
    if args.max_test_steps and len(test_steps) > args.max_test_steps:
        positions = np.linspace(0, len(test_steps) - 1, args.max_test_steps).round().astype(int)
        test_steps = test_steps[positions]
    layout = build_context_layout(features, train_steps, include_calendar=False)
    matrix = load_or_build_context_matrix(
        features,
        test_steps,
        layout,
        ckpt,
        ckpt_args,
        workers=max(1, int(args.feature_workers)),
        cache_path=Path(args.context_cache) if args.context_cache else None,
    )
    input_dim = int(artifact["input_dim"])
    expected_dim = int(
        layout.total_feature_count
        if artifact["uses_graph_neighbor_state"]
        else layout.base_feature_count
    )
    if input_dim != expected_dim:
        raise ValueError(f"direct impact input mismatch: artifact={input_dim} expected={expected_dim}")
    matrix_rows = rows_for_steps(
        test_steps,
        {int(step): index for index, step in enumerate(test_steps)},
        features.tradable_count,
    )
    amp_enabled = device.type == "cuda" and bool(args.amp)
    context = prepare_context_tensor(
        matrix, matrix_rows, input_dim, device, amp_enabled
    ).reshape(len(test_steps), features.tradable_count, input_dim)
    head = DirectImpactHead(
        input_dim,
        horizons,
        hidden_dim=int(artifact["hidden_dim"]),
        dropout=float(artifact["dropout"]),
    ).to(device)
    head.load_state_dict(artifact["state_dict"])
    rows = score_steps(
        head,
        context,
        features,
        test_steps,
        horizons,
        counts,
        int(args.liquidity_top_k),
        int(args.batch_dates),
        amp_enabled,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "daily_fixed_k_metrics.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "date",
                "horizon",
                "scope",
                "k",
                "strategy",
                "selection_mode",
                "selected_tickers",
                *FIXED_K_METRICS,
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "complete",
        "role": "equal_objective_direct_impact_fixed_count_challenger",
        "model_dir": str(model_dir),
        "head_path": str(args.head_path),
        "train_data_manifest_sha256": artifact["train_data_manifest_sha256"],
        "train_end": train_end,
        "test_dates": int(len(test_steps)),
        "uses_graph_neighbor_state": bool(artifact["uses_graph_neighbor_state"]),
        "counts": counts,
        "strategies": list(STRATEGIES),
        "metrics": summarize_rows(rows, horizons, counts),
        "cross_horizon_metrics": summarize_rows(
            rows, horizons, counts, selection_mode="cross_horizon"
        ),
        "selection_modes": ["per_horizon", "cross_horizon"],
        "cross_horizon_selection_contract": cross_horizon_selection_contract(rows),
        "posthoc_notice": (
            "Cross-horizon daily selection was added after the per-horizon "
            "fixed-count results and remains a secondary diagnostic."
        ),
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"test_dates": len(test_steps), "rows": len(rows)}))


if __name__ == "__main__":
    main()
