from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.audit_systemic_transition_targets import _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_latent_trajectory_path_head import (
    checkpoint_sha256,
    latent_trajectories,
    snapshot_batch,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import parse_int_list, rollout_steps_for_offset


def sha256_array(values: np.ndarray) -> str:
    digest = hashlib.sha256()
    data = np.ascontiguousarray(values).view(np.uint8).ravel()
    for start in range(0, len(data), 1024 * 1024):
        digest.update(data[start : start + 1024 * 1024])
    return digest.hexdigest()


def cache_contract(
    model_dir: Path,
    steps: np.ndarray,
    split_horizons: Sequence[int],
    features,
    eligible_indices: np.ndarray,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "cache_kind": "h1_temporal_state_only",
        "checkpoint_sha256": checkpoint_sha256(model_dir),
        "steps_sha256": sha256_array(steps),
        "step_start": int(steps[0]),
        "step_end": int(steps[-1]),
        "rows": int(len(steps)),
        "horizons": [1],
        "split_horizons": [int(value) for value in split_horizons],
        "node_count": int(features.node_count),
        "eligible_indices": eligible_indices.tolist(),
        "dtype": "float16",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a resumable h1 JEPA state cache for open-nowcast folds."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--split-horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--edge-cache-workers", type=int, default=16)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    split_horizons = tuple(parse_int_list(args.split_horizons))
    device = torch.device(args.device)
    model, checkpoint = load_model(model_dir, device)
    validate_future_rollout_contract(dict(checkpoint.get("args", {})), [1], False)
    checkpoint_args = dict(checkpoint.get("args", {}))
    args.horizons = args.split_horizons
    feature_args = evaluator_namespace(args)
    feature_args.horizons = args.split_horizons
    feature_args.edge_cache_workers = int(args.edge_cache_workers)
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(
        features, checkpoint_args, split_horizons, int(args.validation_days)
    )
    minimum_step = max(0, int(splits["fit"].min()) - max(split_horizons))
    maximum_step = int(splits["test"].max())
    steps = np.arange(minimum_step, maximum_step + 1, dtype=np.int64)
    temporal_weights = checkpoint.get("temporal_state_feature_weights")
    if torch.is_tensor(temporal_weights):
        temporal_weights = temporal_weights.detach().cpu().numpy()
    eligible_indices = np.flatnonzero(
        np.asarray(temporal_weights, dtype=np.float32) > 0.0
    ).astype(np.int64)
    contract = cache_contract(
        model_dir, steps, split_horizons, features, eligible_indices
    )
    contract_path = output_dir / "contract.json"
    progress_path = output_dir / "progress.json"
    complete_path = output_dir / "CACHE_COMPLETE"
    state_path = output_dir / "state_h1.npy"
    if complete_path.exists() and contract_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("completed h1 cache contract differs from this run")
        state = np.load(state_path, mmap_mode="r")
        expected = (len(steps), features.node_count, len(eligible_indices))
        if state.shape != expected or state.dtype != np.float16:
            raise ValueError("completed h1 state cache has the wrong shape or dtype")
        print(json.dumps({"status": "already_complete", **contract}), flush=True)
        return

    expected_shape = (len(steps), features.node_count, len(eligible_indices))
    if contract_path.exists() and state_path.exists():
        existing = json.loads(contract_path.read_text(encoding="utf-8"))
        if existing != contract:
            raise ValueError("partial h1 cache contract differs from this run")
        start_row = (
            int(json.loads(progress_path.read_text(encoding="utf-8")).get("rows", 0))
            if progress_path.exists()
            else 0
        )
        state = np.lib.format.open_memmap(state_path, mode="r+")
        if state.shape != expected_shape or state.dtype != np.float16:
            raise ValueError("partial h1 state cache has the wrong shape or dtype")
    else:
        contract_path.write_text(
            json.dumps(contract, indent=2) + "\n", encoding="utf-8"
        )
        start_row = 0
        state = np.lib.format.open_memmap(
            state_path,
            mode="w+",
            dtype=np.float16,
            shape=expected_shape,
        )

    edge_cache = build_evaluation_edge_cache(
        features, steps, checkpoint_args, feature_args
    )
    rollout_namespace = argparse.Namespace(**checkpoint_args)
    model.eval()
    for start in range(start_row, len(steps), int(args.batch_size)):
        end = min(start + int(args.batch_size), len(steps))
        selected_steps = steps[start:end]
        batch = snapshot_batch(
            features,
            selected_steps,
            checkpoint_args,
            feature_args,
            edge_cache,
            device,
        )
        with torch.inference_mode():
            context, predictions = latent_trajectories(
                model, batch, [1], checkpoint_args
            )
            rollout_steps = rollout_steps_for_offset(rollout_namespace, 1)
            prediction = model.predict_temporal_state(
                batch,
                predictions[1],
                rollout_steps=rollout_steps,
                z_context=context,
            )
        state[start:end] = (
            prediction[:, eligible_indices]
            .reshape(len(selected_steps), features.node_count, -1)
            .to(dtype=torch.float16)
            .cpu()
            .numpy()
        )
        state.flush()
        progress_path.write_text(
            json.dumps({"rows": end}) + "\n", encoding="utf-8"
        )
        if end % 40 == 0 or end == len(steps):
            print(f"h1_state_cache={end}/{len(steps)}", flush=True)
    complete_path.touch()
    summary = {
        "status": "complete",
        "role": "research_only_h1_state_forecast_cache",
        "contract": contract,
        "split_dates": {name: int(len(values)) for name, values in splits.items()},
        "state_bytes": int(state_path.stat().st_size),
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
