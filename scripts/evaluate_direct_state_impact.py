from __future__ import annotations

import argparse
import csv
import json
import math
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
    newey_west_mean,
    rows_for_steps,
)
from scripts.benchmark_direct_state_mlp import (
    ResidualStateMLP,
    prepare_context_tensor,
    predict_batches,
    resolve_device,
)
from scripts.benchmark_impact_trajectory_head import (
    DEFAULT_IMPACT_FRACTIONS,
    METRIC_NAMES,
    impact_metric_row,
)
from scripts.benchmark_latent_trajectory_path_head import (
    HORIZON_WEIGHTS,
    top_liquidity_mask,
)
from scripts.evaluate_node_prediction import (
    build_features_from_ckpt,
    derive_entry_path_return,
)
from scripts.run_real_backtest import date_indices, parse_int_list, temporal_training_indices


DIRECT_VARIANTS = ("direct_graph", "direct_nograph")


def load_direct_model(path: Path, device: torch.device) -> tuple[ResidualStateMLP, dict[str, Any]]:
    artifact = torch.load(path, map_location="cpu", weights_only=False)
    state = artifact["model_state_dict"]
    linear_weights = [
        value
        for name, value in state.items()
        if name.startswith("trunk.") and name.endswith(".weight") and value.ndim == 2
    ]
    if not linear_weights:
        raise ValueError(f"cannot infer direct MLP shape from {path}")
    model = ResidualStateMLP(
        input_dim=int(artifact["input_dim"]),
        output_dim=int(artifact["output_dim"]),
        hidden_dim=int(linear_weights[0].shape[0]),
        layers=len(linear_weights),
        dropout=0.0,
    ).to(device)
    model.load_state_dict(state)
    model.eval()
    return model, artifact


def path_score_from_state(
    prediction: np.ndarray,
    horizon: int,
    feature_names: Sequence[str],
    train_mean: np.ndarray,
    train_std: np.ndarray,
    next_open_gap: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if int(horizon) == 1:
        gap_index = feature_names.index("gap_open")
        intraday_index = feature_names.index("intraday_return")
        gap = (
            prediction[:, gap_index] * float(train_std[gap_index])
            + float(train_mean[gap_index])
        )
        path = (
            prediction[:, intraday_index] * float(train_std[intraday_index])
            + float(train_mean[intraday_index])
        )
        return path, gap
    if next_open_gap is None:
        raise ValueError("horizon-1 open-gap forecast is required before longer horizons")
    return_index = feature_names.index(f"return_{int(horizon)}d")
    close_return = (
        prediction[:, return_index] * float(train_std[return_index])
        + float(train_mean[return_index])
    )
    return derive_entry_path_return(close_return, next_open_gap), next_open_gap


def evaluate_variant(
    variant: str,
    direct_dir: Path,
    context: torch.Tensor,
    features,
    test_steps: np.ndarray,
    horizons: Sequence[int],
    fractions: Sequence[float],
    liquidity_top_k: int,
    batch_size: int,
    device: torch.device,
) -> tuple[list[dict[str, Any]], dict[str, str]]:
    stock_count = int(features.tradable_count)
    liquidity_index = features.feature_names.index("value_ma20_log")
    rows: list[dict[str, Any]] = []
    checkpoint_hashes: dict[str, str] = {}
    next_open_gap: np.ndarray | None = None
    for horizon in horizons:
        path = direct_dir / f"direct_state_mlp_h{int(horizon)}.pt"
        model, artifact = load_direct_model(path, device)
        if int(artifact["input_dim"]) != int(context.shape[1]):
            raise ValueError(
                f"{variant} h{horizon} expects {artifact['input_dim']} inputs, "
                f"received {context.shape[1]}"
            )
        prediction = predict_batches(
            model,
            context,
            int(batch_size),
            device.type == "cuda",
        )
        signed_flat, next_open_gap = path_score_from_state(
            prediction,
            int(horizon),
            list(features.feature_names),
            features.train_mean,
            features.train_std,
            next_open_gap,
        )
        signed = signed_flat.reshape(len(test_steps), stock_count)
        target_matrix = np.asarray(
            features.target_return_paths[int(horizon)][test_steps, :stock_count],
            dtype=np.float64,
        )
        for position, step in enumerate(test_steps):
            liquidity = features.raw_features[int(step), :stock_count, liquidity_index]
            scope_masks = {
                "all": np.ones(stock_count, dtype=bool),
                "top300": top_liquidity_mask(liquidity, liquidity_top_k),
            }
            target = target_matrix[position]
            score = signed[position]
            for scope, valid in scope_masks.items():
                for fraction in fractions:
                    metrics = impact_metric_row(
                        score,
                        np.abs(score),
                        target,
                        valid,
                        float(fraction),
                    )
                    rows.append(
                        {
                            "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                            "horizon": int(horizon),
                            "scope": scope,
                            "fraction": float(fraction),
                            "variant": variant,
                            **metrics,
                        }
                    )
        import hashlib

        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        checkpoint_hashes[str(int(horizon))] = digest.hexdigest()
        del model, prediction
        if device.type == "cuda":
            torch.cuda.empty_cache()
    return rows, checkpoint_hashes


def metric_summary(values: list[float], lag: int) -> dict[str, Any]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {
            "rows": 0,
            "mean": float("nan"),
            "newey_west_lag": int(lag),
            "newey_west_standard_error": float("nan"),
            "newey_west_t": float("nan"),
            "positive_fraction": float("nan"),
        }
    return newey_west_mean(finite, lag=int(lag))


def summarize_rows(
    rows: list[dict[str, Any]],
    horizons: Sequence[int],
    fractions: Sequence[float],
) -> tuple[dict[str, Any], dict[str, float]]:
    summary: dict[str, Any] = {}
    for horizon in horizons:
        horizon_result: dict[str, Any] = {}
        for scope in ("all", "top300"):
            scope_result: dict[str, Any] = {}
            for fraction in fractions:
                fraction_key = f"{float(fraction):.2f}"
                variant_result: dict[str, Any] = {}
                for variant in DIRECT_VARIANTS:
                    selected = [
                        row
                        for row in rows
                        if int(row["horizon"]) == int(horizon)
                        and row["scope"] == scope
                        and abs(float(row["fraction"]) - float(fraction)) < 1e-9
                        and row["variant"] == variant
                    ]
                    variant_result[variant] = {
                        metric: metric_summary(
                            [float(row[metric]) for row in selected], int(horizon)
                        )
                        for metric in METRIC_NAMES
                    }
                scope_result[fraction_key] = variant_result
            horizon_result[scope] = scope_result
        summary[str(int(horizon))] = horizon_result

    primary_fraction = min(fractions, key=lambda value: abs(float(value) - 0.10))
    primary_key = f"{float(primary_fraction):.2f}"
    primary_scores: dict[str, float] = {}
    for variant in DIRECT_VARIANTS:
        weighted = 0.0
        weight_sum = 0.0
        for horizon in horizons:
            metrics = summary[str(int(horizon))]["top300"][primary_key][variant]
            precision = float(metrics["precision"]["mean"])
            direction = float(metrics["captured_direction_accuracy"]["mean"])
            tail_ic = float(metrics["tail_ic"]["mean"])
            if not all(math.isfinite(value) for value in (precision, direction, tail_ic)):
                continue
            impact_skill = (precision - float(primary_fraction)) / (
                1.0 - float(primary_fraction)
            )
            direction_skill = 2.0 * (direction - 0.5)
            score = 0.50 * impact_skill + 0.30 * direction_skill + 0.20 * tail_ic
            weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
            weighted += weight * score
            weight_sum += weight
        primary_scores[variant] = weighted / weight_sum if weight_sum else float("nan")
    return summary, primary_scores


def write_daily(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = ["date", "horizon", "scope", "fraction", "variant", *METRIC_NAMES]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate frozen direct state MLP checkpoints on impact-tail metrics."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--direct-graph-dir", required=True)
    parser.add_argument("--direct-nograph-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--impact-fractions", default="0.05,0.10,0.20")
    parser.add_argument("--liquidity-top-k", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--feature-workers", type=int, default=8)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--context-cache", default=None)
    args = parser.parse_args()

    horizons = parse_int_list(args.horizons)
    fractions = tuple(float(value) for value in args.impact_fractions.split(","))
    if any(not 0.0 < value < 1.0 for value in fractions):
        raise ValueError("all impact fractions must be between zero and one")
    device = resolve_device(str(args.device))
    model_dir = Path(args.model_dir)
    ckpt = torch.load(
        model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    feature_args = argparse.Namespace(**vars(args))
    feature_args.horizons = args.horizons
    features, ckpt_args = build_features_from_ckpt(
        ckpt, evaluator_namespace(feature_args)
    )
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
    context_matrix = load_or_build_context_matrix(
        features,
        test_steps,
        layout,
        ckpt,
        ckpt_args,
        workers=max(1, int(args.feature_workers)),
        cache_path=Path(args.context_cache) if args.context_cache else None,
    )
    graph_input_dim = int(
        torch.load(
            Path(args.direct_graph_dir) / "direct_state_mlp_h1.pt",
            map_location="cpu",
            weights_only=False,
        )["input_dim"]
    )
    nograph_input_dim = int(
        torch.load(
            Path(args.direct_nograph_dir) / "direct_state_mlp_h1.pt",
            map_location="cpu",
            weights_only=False,
        )["input_dim"]
    )
    matrix_rows = rows_for_steps(
        test_steps,
        {int(step): index for index, step in enumerate(test_steps)},
        features.tradable_count,
    )
    contexts = {
        "direct_graph": prepare_context_tensor(
            context_matrix, matrix_rows, graph_input_dim, device, device.type == "cuda"
        ),
        "direct_nograph": prepare_context_tensor(
            context_matrix, matrix_rows, nograph_input_dim, device, device.type == "cuda"
        ),
    }
    all_rows: list[dict[str, Any]] = []
    hashes: dict[str, dict[str, str]] = {}
    for variant, direct_dir in (
        ("direct_graph", Path(args.direct_graph_dir)),
        ("direct_nograph", Path(args.direct_nograph_dir)),
    ):
        variant_rows, variant_hashes = evaluate_variant(
            variant,
            direct_dir,
            contexts[variant],
            features,
            test_steps,
            horizons,
            fractions,
            int(args.liquidity_top_k),
            int(args.batch_size),
            device,
        )
        all_rows.extend(variant_rows)
        hashes[variant] = variant_hashes

    metrics, primary_scores = summarize_rows(all_rows, horizons, fractions)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_daily(output_dir / "daily_impact_metrics.csv", all_rows)
    summary = {
        "status": "complete",
        "role": "frozen_direct_state_mlp_impact_challenger",
        "model_dir": str(model_dir),
        "direct_graph_dir": str(args.direct_graph_dir),
        "direct_nograph_dir": str(args.direct_nograph_dir),
        "checkpoint_sha256": hashes,
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get("sha256"),
        "train_edge_manifest_sha256": ckpt.get("train_edge_manifest", {}).get("sha256"),
        "train_end": train_end,
        "test_dates": int(len(test_steps)),
        "horizons": horizons,
        "impact_fractions": list(fractions),
        "metrics": metrics,
        "primary_scores": primary_scores,
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"primary_scores": primary_scores, "test_dates": len(test_steps)}))


if __name__ == "__main__":
    main()
