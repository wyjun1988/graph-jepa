from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_latent_trajectory_path_head import (
    latent_trajectories,
    snapshot_batch,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
)
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.ops.signals import materialize_quote_overlay_session


CACHE_CONTRACT = "strict_oos_stale_daily_jepa_h1_v2"
DEFAULT_STATE_FEATURES = (
    "return_1d",
    "return_2d",
    "return_3d",
    "return_5d",
    "return_10d",
    "gap_open",
    "intraday_return",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cache one-day stale JEPA latents for intraday reforecasting."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--target-start", required=True)
    parser.add_argument("--target-end", required=True)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--edge-cache-workers", type=int, default=8)
    parser.add_argument("--state-features", default=",".join(DEFAULT_STATE_FEATURES))
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument("--allow-in-sample-checkpoint", action="store_true")
    parser.add_argument(
        "--prospective-target-date",
        help=(
            "Append this current/future session as a label-free placeholder so its "
            "stale context can be rolled from the latest complete panel row."
        ),
    )
    parser.add_argument(
        "--prospective-context-date",
        help="Require this exact latest complete panel session for prospective rollout.",
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device(value: str) -> torch.device:
    requested = str(value).lower()
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return torch.device(requested)


def _atomic_replace_directory(temporary: Path, output: Path) -> None:
    if output.exists():
        backup = Path(str(output) + ".previous")
        if backup.exists():
            shutil.rmtree(backup)
        output.replace(backup)
        temporary.replace(output)
        shutil.rmtree(backup)
    else:
        temporary.replace(output)


def _assert_extension_prefix_unchanged(original: object, extended: object) -> None:
    original_dates = pd.DatetimeIndex(original.dates)
    extended_dates = pd.DatetimeIndex(extended.dates)
    if not original_dates.equals(extended_dates[: len(original_dates)]):
        raise ValueError("extended feature panel changed the checkpoint date prefix")
    if list(original.tickers) != list(extended.tickers):
        raise ValueError("extended feature panel changed ticker order")
    if list(original.node_tickers or []) != list(extended.node_tickers or []):
        raise ValueError("extended feature panel changed node order")
    if list(original.feature_names) != list(extended.feature_names):
        raise ValueError("extended feature panel changed feature order")
    arrays = (
        ("features", original.features, extended.features),
        ("raw_features", original.raw_features, extended.raw_features),
        ("available_mask", original.available_mask, extended.available_mask),
        ("returns_1d", original.returns_1d, extended.returns_1d),
        ("open", original.open, extended.open),
        ("close", original.close, extended.close),
    )
    for label, expected, candidate in arrays:
        expected = np.asarray(expected)
        candidate = np.asarray(candidate)[: len(expected)]
        if expected.shape != candidate.shape or not np.array_equal(
            expected, candidate, equal_nan=True
        ):
            raise ValueError(f"extended feature panel changed checkpoint prefix: {label}")


def _pack_stock_graph_edges(
    edge_cache: dict[int, tuple[torch.Tensor, torch.Tensor]] | None,
    context_steps: np.ndarray,
    stock_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pack date-aligned, finite inter-stock edges into CSR-style arrays."""

    if edge_cache is None:
        raise ValueError("causal stock graph persistence requires an edge cache")
    offsets = np.zeros(len(context_steps) + 1, dtype=np.int64)
    index_blocks: list[np.ndarray] = []
    weight_blocks: list[np.ndarray] = []
    for row, step in enumerate(np.asarray(context_steps, dtype=np.int64)):
        cached = edge_cache.get(int(step))
        if cached is None:
            raise ValueError(f"edge cache is missing context step {int(step)}")
        edge_index = cached[0].detach().cpu().numpy().astype(np.int64, copy=False)
        edge_weight = cached[1].detach().cpu().numpy().astype(np.float32, copy=False)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge cache index must be shaped [2, edges]")
        if edge_weight.shape != (edge_index.shape[1],):
            raise ValueError("edge cache weights must align with edge indices")
        selected = (
            (edge_index[0] >= 0)
            & (edge_index[1] >= 0)
            & (edge_index[0] < int(stock_count))
            & (edge_index[1] < int(stock_count))
            & (edge_index[0] != edge_index[1])
            & np.isfinite(edge_weight)
            & (edge_weight != 0.0)
        )
        selected_index = edge_index[:, selected].astype(np.int32, copy=False)
        selected_weight = edge_weight[selected].astype(np.float32, copy=False)
        index_blocks.append(selected_index)
        weight_blocks.append(selected_weight)
        offsets[row + 1] = offsets[row] + selected_weight.size
    total_edges = int(offsets[-1])
    if total_edges == 0:
        raise ValueError("causal stock graph contains no finite inter-stock edges")
    return (
        offsets,
        np.concatenate(index_blocks, axis=1),
        np.concatenate(weight_blocks),
    )


def materialize_prospective_target(
    features: object,
    *,
    target_date: str,
    context_date: str,
) -> tuple[pd.Timestamp, pd.Timestamp]:
    dates = pd.DatetimeIndex(features.dates).normalize()
    if not len(dates) or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("prospective feature dates must be non-empty, unique, and sorted")
    target = pd.Timestamp(target_date).normalize()
    context = pd.Timestamp(context_date).normalize()
    latest = dates[-1]
    if latest != context:
        raise ValueError(
            "prospective context date does not match the latest complete panel session"
        )
    if target <= context or target in dates:
        raise ValueError("prospective target must follow and be absent from the panel")
    if target.dayofweek >= 5:
        raise ValueError("prospective target cannot be a weekend")
    target_step = materialize_quote_overlay_session(
        features,
        len(dates) - 1,
        target,
    )
    updated_dates = pd.DatetimeIndex(features.dates).normalize()
    if target_step != len(updated_dates) - 1 or updated_dates[-1] != target:
        raise RuntimeError("prospective target was not appended at the panel boundary")
    if np.isfinite(np.asarray(features.target_returns[target_step])).any():
        raise RuntimeError("prospective target placeholder contains target returns")
    if any(
        np.isfinite(np.asarray(values[target_step])).any()
        for values in features.target_return_paths.values()
    ):
        raise RuntimeError("prospective target placeholder contains target return paths")
    return context, target


def main() -> int:
    args = parse_args()
    if int(args.batch_size) <= 0 or int(args.edge_cache_workers) <= 0:
        raise ValueError("batch size and edge-cache workers must be positive")
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    temporary_dir = Path(str(output_dir) + ".tmp")
    if temporary_dir.exists():
        shutil.rmtree(temporary_dir)
    temporary_dir.mkdir(parents=True)
    device = _device(args.device)
    prospective_enabled = bool(args.prospective_target_date)
    if prospective_enabled != bool(args.prospective_context_date):
        raise ValueError(
            "prospective target and context dates must be supplied together"
        )
    if prospective_enabled and not (
        str(args.target_start) == str(args.prospective_target_date)
        and str(args.target_end) == str(args.prospective_target_date)
    ):
        raise ValueError(
            "prospective cache must request exactly its one target session"
        )
    model, ckpt = load_model(model_dir, device)
    ckpt_args = dict(ckpt.get("args", {}))

    trained_horizons = ckpt_args.get("path_horizons") or ckpt_args.get(
        "rollout_offsets", [1]
    )
    if isinstance(trained_horizons, (list, tuple)):
        trained_horizons = ",".join(str(value) for value in trained_horizons)
    base_args = argparse.Namespace(
        cache_dir=args.cache_dir,
        external_cache_dir=args.external_cache_dir,
        horizons=str(trained_horizons),
    )
    cli = evaluator_namespace(base_args)
    cli.horizons = str(trained_horizons)
    cli.edge_cache_workers = int(args.edge_cache_workers)
    original_features, ckpt_args = build_features_from_ckpt(ckpt, cli)
    checkpoint_panel_end = pd.DatetimeIndex(original_features.dates).max().normalize()
    requested_end = pd.Timestamp(args.target_end).normalize()
    extension_verified = False
    if requested_end > checkpoint_panel_end:
        extended_checkpoint = dict(ckpt)
        extended_checkpoint.pop("train_data_manifest", None)
        extended_cli = evaluator_namespace(base_args)
        extended_cli.horizons = str(trained_horizons)
        extended_cli.end = (
            args.prospective_context_date
            if prospective_enabled
            else args.target_end
        )
        extended_cli.allow_unverified_legacy = True
        extended_cli.edge_cache_workers = int(args.edge_cache_workers)
        features, _extended_args = build_features_from_ckpt(
            extended_checkpoint, extended_cli
        )
        _assert_extension_prefix_unchanged(original_features, features)
        extension_verified = True
        del original_features
        cli = extended_cli
    else:
        features = original_features
    prospective_context: pd.Timestamp | None = None
    prospective_target: pd.Timestamp | None = None
    if prospective_enabled:
        prospective_context, prospective_target = materialize_prospective_target(
            features,
            target_date=str(args.prospective_target_date),
            context_date=str(args.prospective_context_date),
        )
    feature_dates = pd.DatetimeIndex(features.dates).normalize()
    start = pd.Timestamp(args.target_start).normalize()
    end = pd.Timestamp(args.target_end).normalize()
    target_positions = np.flatnonzero((feature_dates >= start) & (feature_dates <= end))
    target_positions = target_positions[target_positions > 0]
    if not len(target_positions):
        raise ValueError("no target dates overlap the rebuilt JEPA feature panel")
    target_dates = feature_dates[target_positions]
    context_steps = target_positions - 1
    context_dates = feature_dates[context_steps]
    if not np.all(context_dates < target_dates):
        raise RuntimeError("stale rollout context must strictly precede its target session")
    train_end = pd.Timestamp(ckpt_args.get("train_end", "2099-12-31")).normalize()
    if not args.allow_in_sample_checkpoint and target_dates.min() <= train_end:
        raise ValueError(
            f"checkpoint train_end={train_end.date()} overlaps target cache; "
            "use an earlier walk-forward checkpoint"
        )

    checkpoint_tickers = tuple(str(value) for value in ckpt.get("tickers", []))
    feature_tickers = tuple(str(value) for value in features.tickers)
    if checkpoint_tickers and feature_tickers != checkpoint_tickers:
        raise ValueError("rebuilt ticker order differs from the checkpoint")
    stock_count = int(features.tradable_count)
    if stock_count != len(feature_tickers):
        raise ValueError("stale cache requires one output row per checkpoint stock")
    state_features = tuple(
        value.strip() for value in args.state_features.split(",") if value.strip()
    )
    missing_state = [name for name in state_features if name not in features.feature_names]
    if missing_state:
        raise ValueError(f"state features absent from checkpoint: {missing_state}")
    state_indices = np.asarray(
        [features.feature_names.index(name) for name in state_features], dtype=np.int64
    )
    latent_dim = int(ckpt_args.get("hidden_dim", 0))
    if latent_dim <= 0:
        raise ValueError("checkpoint hidden dimension is missing")

    edge_cache = build_evaluation_edge_cache(
        features, context_steps, ckpt_args, cli
    )
    edge_offsets, stock_edge_index, stock_edge_weight = _pack_stock_graph_edges(
        edge_cache,
        context_steps,
        stock_count,
    )
    context_path = temporary_dir / "context_latent_f16.npy"
    delta_path = temporary_dir / "predicted_delta_f16.npy"
    state_path = temporary_dir / "predicted_state_f32.npy"
    context_output = np.lib.format.open_memmap(
        context_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(target_dates), stock_count, latent_dim),
    )
    delta_output = np.lib.format.open_memmap(
        delta_path,
        mode="w+",
        dtype=np.float16,
        shape=(len(target_dates), stock_count, latent_dim),
    )
    state_output = np.lib.format.open_memmap(
        state_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(target_dates), stock_count, len(state_features)),
    )
    rollout_namespace = argparse.Namespace(**ckpt_args)
    rollout_steps = rollout_steps_for_offset(rollout_namespace, 1)
    feature_mean = model.feature_means.detach().float()
    feature_std = model.feature_stds.detach().float()
    state_index_tensor = torch.as_tensor(
        state_indices, dtype=torch.long, device=device
    )

    for start_row in range(0, len(context_steps), int(args.batch_size)):
        stop_row = min(start_row + int(args.batch_size), len(context_steps))
        selected_steps = context_steps[start_row:stop_row]
        batch = snapshot_batch(
            features, selected_steps, ckpt_args, cli, edge_cache, device
        )
        context, predicted = latent_trajectories(model, batch, [1], ckpt_args)
        with torch.no_grad():
            predicted_state = model.predict_temporal_state(
                batch,
                predicted[1],
                rollout_steps=rollout_steps,
                z_context=context,
            )
        batch_count = len(selected_steps)
        context = context.reshape(batch_count, features.node_count, latent_dim)[
            :, :stock_count
        ]
        predicted_latent = predicted[1].reshape(
            batch_count, features.node_count, latent_dim
        )[:, :stock_count]
        predicted_state = predicted_state.reshape(
            batch_count, features.node_count, len(features.feature_names)
        )[:, :stock_count]
        selected_state = predicted_state.index_select(-1, state_index_tensor)
        selected_state = (
            selected_state
            * feature_std.index_select(0, state_index_tensor)[None, None, :]
            + feature_mean.index_select(0, state_index_tensor)[None, None, :]
        )
        context_output[start_row:stop_row] = context.float().cpu().numpy().astype(
            np.float16
        )
        delta_output[start_row:stop_row] = (
            (predicted_latent - context).float().cpu().numpy().astype(np.float16)
        )
        state_output[start_row:stop_row] = selected_state.float().cpu().numpy()
        print(
            f"cached={stop_row}/{len(context_steps)} "
            f"target={target_dates[stop_row - 1].date()}",
            flush=True,
        )
    context_output.flush()
    delta_output.flush()
    state_output.flush()
    del context_output, delta_output, state_output

    dates_path = temporary_dir / "dates_and_tickers.npz"
    with dates_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            target_dates=np.asarray(
                [str(value.date()) for value in target_dates], dtype="U10"
            ),
            context_dates=np.asarray(
                [str(value.date()) for value in context_dates], dtype="U10"
            ),
            context_steps=context_steps.astype(np.int32),
            tickers=np.asarray(feature_tickers, dtype="U6"),
            state_feature_names=np.asarray(state_features, dtype="U32"),
        )

    graph_path = temporary_dir / "causal_stock_graph.npz"
    graph_target_dates = np.asarray(
        [str(value.date()) for value in target_dates], dtype="U10"
    )
    graph_context_dates = np.asarray(
        [str(value.date()) for value in context_dates], dtype="U10"
    )
    with graph_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            target_dates=graph_target_dates,
            context_dates=graph_context_dates,
            edge_offsets=edge_offsets,
            edge_index=stock_edge_index,
            edge_weight=stock_edge_weight,
        )

    files = {
        "context_latent_f16": context_path,
        "predicted_delta_f16": delta_path,
        "predicted_state_f32": state_path,
        "dates_and_tickers": dates_path,
        "causal_stock_graph": graph_path,
    }
    per_date_edge_counts = np.diff(edge_offsets)
    manifest = {
        "schema_version": 2,
        "cache_contract": CACHE_CONTRACT,
        "checkpoint": str(model_dir / "graph_jepa_real.pt"),
        "checkpoint_sha256": file_sha256(model_dir / "graph_jepa_real.pt"),
        "checkpoint_train_end": str(train_end.date()),
        "checkpoint_panel_end": str(checkpoint_panel_end.date()),
        "evaluation_extension_prefix_verified": extension_verified,
        "prospective_target": {
            "enabled": prospective_enabled,
            "target_date": (
                str(prospective_target.date()) if prospective_target is not None else None
            ),
            "context_date": (
                str(prospective_context.date())
                if prospective_context is not None
                else None
            ),
            "target_observations_injected": False,
        },
        "strict_out_of_sample": bool(target_dates.min() > train_end),
        "target_start": str(target_dates.min().date()),
        "target_end": str(target_dates.max().date()),
        "dates": len(target_dates),
        "stocks": stock_count,
        "latent_dim": latent_dim,
        "state_features": list(state_features),
        "device": str(device),
        "stock_graph": {
            "date_aligned": True,
            "directed": True,
            "signed_weights": bool((stock_edge_weight < 0.0).any()),
            "self_loops_excluded": True,
            "external_nodes_excluded": True,
            "total_edges": int(stock_edge_weight.size),
            "minimum_edges_per_date": int(per_date_edge_counts.min()),
            "median_edges_per_date": float(np.median(per_date_edge_counts)),
            "maximum_edges_per_date": int(per_date_edge_counts.max()),
        },
        "causality": {
            "context_strictly_precedes_target_session": True,
            "one_day_rollout_only": True,
            "full_target_session_absent_from_input": True,
            "checkpoint_training_precedes_all_targets": bool(
                target_dates.min() > train_end
            ),
            "extended_panel_prefix_matches_checkpoint_panel": bool(
                extension_verified or target_dates.max() <= checkpoint_panel_end
            ),
            "stock_graph_aligned_to_context_rows": True,
            "stock_graph_uses_context_session_or_earlier_only": True,
            "prospective_target_is_label_free": bool(prospective_enabled),
            "prospective_context_is_latest_complete_panel_session": bool(
                prospective_enabled
            ),
        },
        "files": {
            name: {
                "path": str(path.name),
                "sha256": file_sha256(path),
                "bytes": path.stat().st_size,
            }
            for name, path in files.items()
        },
        "promotion_eligible": False,
        "live_orders_allowed": False,
    }
    manifest_path = temporary_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    _atomic_replace_directory(temporary_dir, output_dir)
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
