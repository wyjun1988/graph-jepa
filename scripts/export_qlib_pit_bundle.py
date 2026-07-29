from __future__ import annotations

import argparse
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.benchmark_direct_baselines import (
    build_context_layout,
    evaluator_namespace,
    load_or_build_context_matrix,
)
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.run_real_backtest import date_indices, parse_int_list, temporal_training_indices


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _context_artifacts(cache_path: Path) -> list[Path]:
    metadata_path = cache_path.with_suffix(cache_path.suffix + ".json")
    if not metadata_path.exists():
        raise FileNotFoundError(f"context metadata is missing: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts = [metadata_path]
    cache_format = metadata.get("format")
    if cache_format == "single_npy":
        artifacts.append(cache_path)
    elif cache_format == "chunked_npy":
        parts_dir = cache_path.with_suffix(cache_path.suffix + ".parts")
        artifacts.extend(parts_dir / str(part["file"]) for part in metadata.get("parts", []))
    else:
        raise ValueError(f"unsupported context cache format: {cache_format}")
    missing = [str(path) for path in artifacts if not path.exists()]
    if missing:
        raise FileNotFoundError(f"context artifacts are missing: {missing}")
    return artifacts


def _date(value: Any) -> str:
    return str(pd.Timestamp(value).date())


def split_steps(features, ckpt_args: dict[str, Any], horizons: list[int], validation_days: int):
    train_end = str(ckpt_args.get("train_end", "2023-12-29"))
    edge_window = int(ckpt_args.get("edge_window", 60))
    max_horizon = max(horizons)
    train_indices = date_indices(features.dates, end=train_end)
    train_steps = temporal_training_indices(
        train_indices,
        edge_window=edge_window,
        max_rollout_offset=max_horizon,
        total_steps=len(features.dates),
    )
    if len(train_steps) <= validation_days:
        raise ValueError("training range is too short for the requested validation window")
    validation_steps = train_steps[-int(validation_days) :]
    fit_steps = train_steps[train_steps < int(validation_steps[0]) - max_horizon]
    test_steps = date_indices(
        features.dates,
        start=(pd.Timestamp(train_end) + pd.DateOffset(days=1)).strftime("%Y-%m-%d"),
    )
    test_steps = test_steps[
        (test_steps >= edge_window)
        & (test_steps <= len(features.dates) - 1 - max_horizon)
    ]
    if len(fit_steps) < 260 or len(validation_steps) < 20 or len(test_steps) < 20:
        raise ValueError("fit, validation, or test split is too short")
    all_steps = np.unique(np.concatenate([fit_steps, validation_steps, test_steps])).astype(
        np.int64
    )
    return fit_steps, validation_steps, test_steps, all_steps


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the exact Graph-JEPA point-in-time panel for a Qlib baseline."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--without-graph", action="store_true")
    parser.add_argument("--feature-workers", type=int, default=8)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    horizons = parse_int_list(args.horizons)
    if not horizons or any(horizon <= 0 for horizon in horizons):
        raise ValueError("horizons must be positive")
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir = Path(args.model_dir).resolve()
    checkpoint_path = model_dir / "graph_jepa_real.pt"
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    feature_args = deepcopy(args)
    feature_args.horizons = args.horizons
    features, ckpt_args = build_features_from_ckpt(ckpt, evaluator_namespace(feature_args))
    fit_steps, validation_steps, test_steps, all_steps = split_steps(
        features, ckpt_args, horizons, int(args.validation_days)
    )

    layout = build_context_layout(features, fit_steps, include_calendar=False)
    context_cache = output_dir / "context.npy"
    context = load_or_build_context_matrix(
        features,
        all_steps,
        layout,
        ckpt,
        ckpt_args,
        workers=max(1, int(args.feature_workers)),
        cache_path=context_cache,
    )
    feature_names = list(layout.base_feature_names)
    if not args.without_graph:
        feature_names.extend(layout.graph_feature_names)
    feature_count = len(feature_names)
    expected_rows = len(all_steps) * features.tradable_count
    if context.shape[0] != expected_rows or context.shape[1] < feature_count:
        raise ValueError("context matrix does not match the exported panel contract")
    if len(feature_names) != len(set(feature_names)):
        raise ValueError("Qlib feature names must be unique")

    stock_count = int(features.tradable_count)
    labels = np.stack(
        [
            features.target_return_paths[int(horizon)][all_steps, :stock_count]
            for horizon in horizons
        ],
        axis=0,
    ).astype(np.float32, copy=False)
    liquidity_index = features.feature_names.index("value_ma20_log")
    liquidity = features.raw_features[
        all_steps, :stock_count, liquidity_index
    ].astype(np.float32, copy=False)
    current_available = features.available_mask[
        all_steps, :stock_count
    ].any(axis=2).astype(np.uint8, copy=False)
    dates = np.asarray(features.dates[all_steps], dtype="datetime64[ns]")
    tickers = [str(value) for value in features.node_tickers[:stock_count]]
    if len(tickers) != stock_count or len(set(tickers)) != stock_count:
        raise ValueError("stock ticker mapping is missing or non-unique")

    arrays_path = output_dir / "targets_and_metadata.npz"
    np.savez_compressed(
        arrays_path,
        dates=dates,
        labels=labels,
        liquidity=liquidity,
        current_available=current_available,
    )
    tickers_path = output_dir / "tickers.json"
    tickers_path.write_text(
        json.dumps(tickers, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    context_artifacts = _context_artifacts(context_cache)
    artifact_paths = [arrays_path, tickers_path, *context_artifacts]
    artifact_hashes = {
        str(path.relative_to(output_dir)): sha256_file(path) for path in artifact_paths
    }
    split_contract = {
        "fit": {
            "start": _date(features.dates[int(fit_steps[0])]),
            "end": _date(features.dates[int(fit_steps[-1])]),
            "dates": int(len(fit_steps)),
        },
        "validation": {
            "start": _date(features.dates[int(validation_steps[0])]),
            "end": _date(features.dates[int(validation_steps[-1])]),
            "dates": int(len(validation_steps)),
        },
        "test": {
            "start": _date(features.dates[int(test_steps[0])]),
            "end": _date(features.dates[int(test_steps[-1])]),
            "dates": int(len(test_steps)),
        },
    }
    contract: dict[str, Any] = {
        "schema_version": 1,
        "role": "research_only_qlib_pit_adapter",
        "test_used_for_selection": False,
        "live_orders_allowed": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "train_data_manifest_sha256": ckpt.get("train_data_manifest", {}).get("sha256"),
        "train_edge_manifest_sha256": ckpt.get("train_edge_manifest", {}).get("sha256"),
        "train_end": str(ckpt_args.get("train_end")),
        "splits": split_contract,
        "horizons": horizons,
        "label_source": "RealFeatureBundle.target_return_paths",
        "label_timing": "future path label; never included in feature columns",
        "dates": int(len(all_steps)),
        "stocks": stock_count,
        "rows": int(expected_rows),
        "feature_count": feature_count,
        "feature_names": feature_names,
        "uses_graph_neighbor_state": not bool(args.without_graph),
        "arrays_file": arrays_path.name,
        "tickers_file": tickers_path.name,
        "context_cache": context_cache.name,
        "artifact_sha256": artifact_hashes,
    }
    canonical = json.dumps(contract, ensure_ascii=False, sort_keys=True).encode("utf-8")
    contract["bundle_contract_sha256"] = hashlib.sha256(canonical).hexdigest()
    contract_path = output_dir / "bundle_contract.json"
    contract_path.write_text(
        json.dumps(contract, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        f"Qlib PIT bundle: rows={expected_rows} features={feature_count} "
        f"stocks={stock_count} dates={len(all_steps)} output={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
