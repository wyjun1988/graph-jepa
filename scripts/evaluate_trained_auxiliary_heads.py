from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.benchmark_direct_baselines import evaluator_namespace, rows_for_steps
from scripts.benchmark_frozen_downstream import as_rollout_namespace
from scripts.evaluate_node_prediction import build_features_from_ckpt, load_model
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.downstream_probes import (
    build_downstream_targets,
    causal_probe_splits,
    newey_west_mean,
    pearson,
)
from stock_v2.graph_jepa import DOWNSTREAM_AUXILIARY_TASKS
from stock_v2.latent_path_head import sha256_file


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the jointly trained JEPA specialist heads out of sample."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--latent-cache-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--test-end", default=None)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    return parser.parse_args()


def daily_task_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    valid: np.ndarray,
    *,
    horizon: int,
) -> dict[str, object]:
    daily_ic = []
    for date_index in range(prediction.shape[0]):
        selected = valid[date_index]
        daily_ic.append(
            pearson(
                prediction[date_index, selected],
                target[date_index, selected],
            )
        )
    selected_prediction = prediction[valid].astype(np.float64)
    selected_target = target[valid].astype(np.float64)
    model_sse = float(np.square(selected_prediction - selected_target).sum())
    zero_sse = float(np.square(selected_target).sum())
    return {
        "daily_ic": newey_west_mean(daily_ic, lag=int(horizon)),
        "pooled_correlation": pearson(selected_prediction, selected_target),
        "mse": model_sse / max(len(selected_target), 1),
        "mse_skill_vs_cross_sectional_zero": (
            1.0 - model_sse / zero_sse if zero_sse > 0.0 else float("nan")
        ),
        "observed": int(len(selected_target)),
    }


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    horizons = sorted({int(value) for value in args.horizons.split(",") if value})
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")

    model_dir = Path(args.model_dir)
    checkpoint_path = model_dir / "graph_jepa_real.pt"
    model, checkpoint = load_model(model_dir, device)
    checkpoint_args = dict(checkpoint.get("args", {}))
    if not model.downstream_auxiliary_heads:
        raise ValueError("checkpoint does not contain trained downstream auxiliary heads")

    feature_args = deepcopy(args)
    configured_horizons = checkpoint_args.get("rollout_offsets", horizons)
    if isinstance(configured_horizons, str):
        feature_args.horizons = configured_horizons
    else:
        feature_args.horizons = ",".join(str(int(value)) for value in configured_horizons)
    features, checkpoint_args = build_features_from_ckpt(
        checkpoint,
        evaluator_namespace(feature_args),
    )
    splits = causal_probe_splits(
        features.dates,
        train_end=str(checkpoint_args["train_end"]),
        edge_window=int(checkpoint_args.get("edge_window", 60)),
        max_horizon=max(horizons),
        validation_days=int(args.validation_days),
        test_end=args.test_end,
    )
    all_steps = np.unique(
        np.concatenate([splits.fit_steps, splits.validation_steps, splits.test_steps])
    ).astype(np.int64)
    step_positions = {int(step): position for position, step in enumerate(all_steps)}
    stock_count = int(features.tradable_count)
    test_rows = rows_for_steps(splits.test_steps, step_positions, stock_count)

    latent_cache_dir = Path(args.latent_cache_dir)
    metadata = json.loads((latent_cache_dir / "metadata.json").read_text(encoding="utf-8"))
    expected_checkpoint_sha = sha256_file(checkpoint_path)
    if metadata.get("checkpoint_sha256") != expected_checkpoint_sha:
        raise ValueError("latent cache checkpoint does not match the evaluated model")
    if metadata.get("steps") != [int(value) for value in all_steps]:
        raise ValueError("latent cache steps do not match the causal evaluation split")
    if int(metadata.get("stock_count", -1)) != stock_count:
        raise ValueError("latent cache stock count does not match the feature panel")

    context = np.load(latent_cache_dir / "context.npy", mmap_mode="r")
    rollout_args = as_rollout_namespace(checkpoint_args)
    results: dict[str, object] = {}
    model.eval()
    for horizon in horizons:
        delta = np.load(latent_cache_dir / f"delta_h{horizon}.npy", mmap_mode="r")
        predictions = np.empty(
            (len(test_rows), len(DOWNSTREAM_AUXILIARY_TASKS)),
            dtype=np.float32,
        )
        rollout_steps = rollout_steps_for_offset(rollout_args, int(horizon))
        for start in range(0, len(test_rows), int(args.batch_size)):
            end = min(start + int(args.batch_size), len(test_rows))
            rows = test_rows[start:end]
            context_batch = torch.from_numpy(
                np.asarray(context[rows], dtype=np.float32)
            ).to(device)
            predicted_batch = context_batch + torch.from_numpy(
                np.asarray(delta[rows], dtype=np.float32)
            ).to(device)
            with torch.inference_mode():
                values = model.predict_downstream_auxiliary(
                    context_batch,
                    predicted_batch,
                    rollout_steps,
                )
            predictions[start:end] = values.float().cpu().numpy()

        targets = build_downstream_targets(features, splits.test_steps, int(horizon))
        target_values = targets.continuous[:, : len(DOWNSTREAM_AUXILIARY_TASKS)].reshape(
            len(splits.test_steps), stock_count, -1
        )
        target_valid = targets.continuous_valid[
            :, : len(DOWNSTREAM_AUXILIARY_TASKS)
        ].reshape(len(splits.test_steps), stock_count, -1)
        predictions_by_date = predictions.reshape(len(splits.test_steps), stock_count, -1)
        task_rows: dict[str, object] = {}
        for task_index, task_name in enumerate(DOWNSTREAM_AUXILIARY_TASKS):
            if int(horizon) == 1 and task_name in {
                "max_favorable_excursion",
                "max_adverse_excursion",
            }:
                task_rows[task_name] = {"status": "not_trained_for_horizon_1"}
                continue
            task_rows[task_name] = daily_task_metrics(
                predictions_by_date[:, :, task_index],
                target_values[:, :, task_index],
                target_valid[:, :, task_index],
                horizon=int(horizon),
            )
        results[str(horizon)] = {
            "rollout_steps": int(rollout_steps),
            "tasks": task_rows,
        }

    output = {
        "status": "complete",
        "approval_scope": "research_only",
        "live_orders_allowed": False,
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": expected_checkpoint_sha,
        "train_data_manifest_sha256": checkpoint.get("train_data_manifest", {}).get(
            "sha256"
        ),
        "train_edge_manifest_sha256": checkpoint.get("train_edge_manifest", {}).get(
            "sha256"
        ),
        "test_start": str(features.dates[int(splits.test_steps[0])].date()),
        "test_end": str(features.dates[int(splits.test_steps[-1])].date()),
        "test_dates": int(len(splits.test_steps)),
        "stocks": stock_count,
        "results": results,
    }
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
