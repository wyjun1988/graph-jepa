from __future__ import annotations

import argparse
from copy import deepcopy
import csv
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_frozen_downstream import load_or_build_latent_cache
from scripts.evaluate_node_prediction import build_features_from_ckpt, load_model
from stock_v2.downstream_probes import causal_probe_splits
from stock_v2.latent_path_head import sha256_file


def parse_ints(value: str) -> list[int]:
    parsed = [int(item.strip()) for item in str(value).split(",") if item.strip()]
    if not parsed or len(parsed) != len(set(parsed)) or any(item < 1 for item in parsed):
        raise ValueError("values must be unique positive integers")
    return parsed


def mean_daily_feature_ic(
    context: np.ndarray,
    delta: np.ndarray,
    labels: np.ndarray,
    available: np.ndarray,
    fit_positions: np.ndarray,
    stock_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    hidden_dim = int(context.shape[1])
    if delta.shape != context.shape:
        raise ValueError("context and delta shapes do not match")
    if len(context) % int(stock_count):
        raise ValueError("latent rows do not contain complete date blocks")
    date_count = len(context) // int(stock_count)
    if labels.shape != (date_count, int(stock_count)):
        raise ValueError("label shape does not match latent date blocks")
    if available.shape != labels.shape:
        raise ValueError("availability shape does not match labels")

    sums = np.zeros(2 * hidden_dim, dtype=np.float64)
    counts = np.zeros(2 * hidden_dim, dtype=np.int64)
    for position in np.asarray(fit_positions, dtype=np.int64):
        if not 0 <= int(position) < date_count:
            raise ValueError("fit position is outside the latent panel")
        start = int(position) * int(stock_count)
        end = start + int(stock_count)
        target = np.asarray(labels[int(position)], dtype=np.float64)
        valid = np.asarray(available[int(position)], dtype=bool) & np.isfinite(target)
        if int(valid.sum()) < 3:
            continue
        target = target[valid]
        target -= target.mean()
        target_norm = float(np.sqrt(target @ target))
        if target_norm < 1e-12:
            continue
        values = np.concatenate(
            (
                np.asarray(context[start:end][valid], dtype=np.float32),
                np.asarray(delta[start:end][valid], dtype=np.float32),
            ),
            axis=1,
        ).astype(np.float64, copy=False)
        finite = np.isfinite(values).all(axis=0)
        values -= values.mean(axis=0, keepdims=True)
        denominator = np.sqrt(np.square(values).sum(axis=0)) * target_norm
        usable = finite & (denominator > 1e-12)
        correlation = np.full(values.shape[1], np.nan, dtype=np.float64)
        correlation[usable] = (values[:, usable].T @ target) / denominator[usable]
        observed = np.isfinite(correlation)
        sums[observed] += correlation[observed]
        counts[observed] += 1
    means = np.divide(
        sums,
        counts,
        out=np.full_like(sums, np.nan),
        where=counts > 0,
    )
    return means, counts


def select_indices(mean_ic: np.ndarray, counts: np.ndarray, feature_count: int) -> np.ndarray:
    if mean_ic.shape != counts.shape:
        raise ValueError("feature score and count arrays do not match")
    valid = np.isfinite(mean_ic) & (counts > 0)
    candidates = np.flatnonzero(valid)
    if len(candidates) < int(feature_count):
        raise ValueError("not enough observed latent dimensions for selection")
    order = np.argsort(np.abs(mean_ic[candidates]), kind="stable")
    return candidates[order[-int(feature_count) :][::-1]].astype(np.int64)


def write_selected_matrix(
    output_path: Path,
    context: np.ndarray,
    delta: np.ndarray,
    selected: np.ndarray,
    chunk_rows: int,
) -> None:
    rows = len(context)
    hidden_dim = int(context.shape[1])
    matrix = np.lib.format.open_memmap(
        output_path,
        mode="w+",
        dtype=np.float16,
        shape=(rows, len(selected)),
    )
    context_indices = np.flatnonzero(selected < hidden_dim)
    delta_indices = np.flatnonzero(selected >= hidden_dim)
    for start in range(0, rows, int(chunk_rows)):
        end = min(start + int(chunk_rows), rows)
        if len(context_indices):
            matrix[start:end, context_indices] = context[
                start:end, selected[context_indices]
            ]
        if len(delta_indices):
            matrix[start:end, delta_indices] = delta[
                start:end, selected[delta_indices] - hidden_dim
            ]
    matrix.flush()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select compact JEPA latent features using fit-only daily IC."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--split-horizons", default="5,10")
    parser.add_argument("--feature-count", type=int, default=64)
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--test-end", required=True)
    parser.add_argument("--chunk-rows", type=int, default=32768)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    horizon = int(args.horizon)
    split_horizons = parse_ints(args.split_horizons)
    if horizon not in split_horizons:
        raise ValueError("selection horizon must be included in split horizons")
    if int(args.feature_count) < 1:
        raise ValueError("feature count must be positive")
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        raise ValueError(f"refusing to overwrite output directory: {output_dir}")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_dir = Path(args.model_dir)
    model, checkpoint = load_model(model_dir, device)
    model._checkpoint_path = str(model_dir / "graph_jepa_real.pt")  # type: ignore[attr-defined]
    checkpoint_args = dict(checkpoint.get("args", {}))
    feature_args = deepcopy(args)
    configured_horizons = checkpoint_args.get("rollout_offsets", split_horizons)
    if isinstance(configured_horizons, str):
        feature_args.horizons = configured_horizons
    else:
        feature_args.horizons = ",".join(
            str(int(value)) for value in configured_horizons
        )
    features, checkpoint_args = build_features_from_ckpt(
        checkpoint,
        evaluator_namespace(feature_args),
    )
    splits = causal_probe_splits(
        features.dates,
        train_end=str(checkpoint_args["train_end"]),
        edge_window=int(checkpoint_args.get("edge_window", 60)),
        max_horizon=max(split_horizons),
        validation_days=int(args.validation_days),
        max_test_steps=int(args.max_test_steps),
        test_end=str(args.test_end),
    )
    all_steps = np.unique(
        np.concatenate([splits.fit_steps, splits.validation_steps, splits.test_steps])
    ).astype(np.int64)
    step_positions = {int(step): position for position, step in enumerate(all_steps)}
    context, deltas, latent_contract = load_or_build_latent_cache(
        model,
        features,
        checkpoint,
        checkpoint_args,
        all_steps,
        split_horizons,
        Path(args.latent_cache_dir),
        device,
    )
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    stock_count = int(features.tradable_count)
    labels = np.asarray(
        features.target_return_paths[horizon][all_steps, :stock_count],
        dtype=np.float64,
    )
    available = np.asarray(features.available_mask[all_steps, :stock_count])
    if available.ndim == 3:
        available = available.any(axis=2)
    elif available.ndim != 2:
        raise ValueError("availability mask has an unsupported shape")
    fit_positions = np.asarray(
        [step_positions[int(step)] for step in splits.fit_steps], dtype=np.int64
    )
    scores, counts = mean_daily_feature_ic(
        context,
        deltas[horizon],
        labels,
        available,
        fit_positions,
        stock_count,
    )
    selected = select_indices(scores, counts, int(args.feature_count))

    output_dir.mkdir(parents=True)
    matrix_path = output_dir / "selected_latent.npy"
    write_selected_matrix(
        matrix_path,
        context,
        deltas[horizon],
        selected,
        int(args.chunk_rows),
    )
    score_path = output_dir / "feature_scores.csv"
    hidden_dim = int(context.shape[1])
    with score_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["rank", "combined_index", "source", "source_index", "mean_fit_daily_ic", "observed_dates"],
        )
        writer.writeheader()
        for rank, index in enumerate(selected, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "combined_index": int(index),
                    "source": "context" if int(index) < hidden_dim else "delta",
                    "source_index": int(index) if int(index) < hidden_dim else int(index) - hidden_dim,
                    "mean_fit_daily_ic": float(scores[int(index)]),
                    "observed_dates": int(counts[int(index)]),
                }
            )

    dates = [str(features.dates[int(step)].date()) for step in all_steps]
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "role": "fit_only_compact_jepa_latent_features",
        "live_orders_allowed": False,
        "checkpoint_sha256": sha256_file(model_dir / "graph_jepa_real.pt"),
        "train_data_manifest_sha256": checkpoint.get("train_data_manifest", {}).get("sha256"),
        "train_edge_manifest_sha256": checkpoint.get("train_edge_manifest", {}).get("sha256"),
        "latent_cache_contract": latent_contract,
        "horizon": horizon,
        "selection_metric": "absolute mean fit-period daily cross-sectional Pearson IC",
        "feature_count": int(len(selected)),
        "selected_indices": [int(value) for value in selected],
        "hidden_dim": hidden_dim,
        "rows": int(len(context)),
        "date_count": int(len(all_steps)),
        "stock_count": stock_count,
        "dates": dates,
        "fit_start": str(features.dates[int(splits.fit_steps[0])].date()),
        "fit_end": str(features.dates[int(splits.fit_steps[-1])].date()),
        "validation_start": str(features.dates[int(splits.validation_steps[0])].date()),
        "validation_end": str(features.dates[int(splits.validation_steps[-1])].date()),
        "test_start": str(features.dates[int(splits.test_steps[0])].date()),
        "test_end": str(features.dates[int(splits.test_steps[-1])].date()),
        "selected_matrix": matrix_path.name,
        "selected_matrix_sha256": sha256_file(matrix_path),
        "feature_scores": score_path.name,
        "feature_scores_sha256": sha256_file(score_path),
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "features": int(len(selected)),
                "context_features": int((selected < hidden_dim).sum()),
                "delta_features": int((selected >= hidden_dim).sum()),
                "matrix_sha256": metadata["selected_matrix_sha256"],
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
