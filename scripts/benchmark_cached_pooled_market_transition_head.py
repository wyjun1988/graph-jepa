from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.audit_market_transition_targets import _actual_rows
from scripts.audit_systemic_transition_targets import _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_direct_market_transition_head import predict_steps, train_epoch
from scripts.benchmark_latent_trajectory_path_head import (
    checkpoint_sha256,
    latent_trajectories,
    snapshot_batch,
)
from scripts.benchmark_market_transition_head import (
    LOSS_WEIGHTS,
    _daily_rows,
    _subsample,
    _write_csv,
    build_target_arrays,
    build_target_contracts,
    configured_horizon_text,
    fit_trajectory_event_rate,
    summarize,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import parse_int_list
from stock_v2.market_transition import (
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
)
from stock_v2.market_transition_head import PooledMarketTrajectoryHead


CACHE_VERSION = "frozen_robust_transition_pool_v1_20260714"


def robust_transition_pool(
    model,
    batch,
    context: torch.Tensor,
    predicted: dict[int, torch.Tensor],
    horizons: list[int],
    pooling_mode: str,
) -> torch.Tensor:
    stock_mask = model._supervision_node_mask(batch)
    groups = (
        torch.zeros(context.shape[0], dtype=torch.long, device=context.device)
        if batch.graph_index is None
        else batch.graph_index.to(device=context.device, dtype=torch.long)
    )
    graph_count = int(groups.max().item()) + 1 if len(groups) else 0
    sequences = []
    for horizon in horizons:
        future = predicted[int(horizon)]
        head_input = (
            torch.cat((context, future - context), dim=-1)
            if model.temporal_state_context_skip
            else future
        )
        if pooling_mode == "projected":
            pooled_input = model.downstream_transition_projector(head_input)
        elif pooling_mode == "raw":
            pooled_input = head_input
        else:
            raise ValueError(f"unsupported transition pooling mode: {pooling_mode}")
        stock_mean, stock_std, stock_median, stock_counts = (
            model._pool_distribution_by_graph(
                pooled_input, stock_mask, groups, graph_count
            )
        )
        external_mean, external_std, _external_median, _external_counts = (
            model._pool_distribution_by_graph(
                pooled_input, ~stock_mask, groups, graph_count
            )
        )
        if (stock_counts <= 0.0).any():
            raise ValueError("each graph requires supervised stock nodes")
        sequences.append(
            torch.cat(
                (
                    stock_mean,
                    stock_std,
                    stock_median,
                    external_mean,
                    external_std,
                ),
                dim=-1,
            )
        )
    return torch.stack(sequences, dim=1)


def cache_sequences(
    model,
    features,
    steps,
    horizons,
    checkpoint_args,
    feature_args,
    edge_cache,
    device,
    batch_size,
    label,
    pooling_mode,
) -> np.ndarray:
    model.eval()
    rows = []
    for start in range(0, len(steps), int(batch_size)):
        selected = np.asarray(steps[start : start + int(batch_size)], dtype=np.int64)
        batch = snapshot_batch(
            features, selected, checkpoint_args, feature_args, edge_cache, device
        )
        with torch.no_grad():
            context, predicted = latent_trajectories(
                model, batch, horizons, checkpoint_args
            )
            pooled = robust_transition_pool(
                model, batch, context, predicted, horizons, pooling_mode
            )
        rows.append(pooled.float().cpu().numpy())
        print(
            f"cache={label} rows={min(start + int(batch_size), len(steps))}/{len(steps)}",
            flush=True,
        )
    return np.concatenate(rows, axis=0).astype(np.float32)


def normalize_sequences(fit, validation, test):
    mean = fit.mean(axis=0, dtype=np.float64)
    std = fit.std(axis=0, dtype=np.float64)
    std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0)

    def apply(values):
        return ((values - mean[None, :, :]) / std[None, :, :]).astype(np.float32)

    return apply(fit), apply(validation), apply(test), mean, std


def load_or_build_cache(
    path,
    *,
    model,
    features,
    splits,
    horizons,
    checkpoint_args,
    feature_args,
    edge_cache,
    device,
    batch_size,
    parent_sha,
    pooling_mode,
):
    if path.exists():
        cached = np.load(path, allow_pickle=False)
        if str(cached["cache_version"].item()) != CACHE_VERSION:
            raise ValueError("pooled cache version differs")
        if str(cached["parent_sha256"].item()) != parent_sha:
            raise ValueError("pooled cache checkpoint differs")
        cached_pooling_mode = (
            str(cached["pooling_mode"].item())
            if "pooling_mode" in cached.files
            else "projected"
        )
        if cached_pooling_mode != pooling_mode:
            raise ValueError("pooled cache representation mode differs")
        if not np.array_equal(cached["horizons"], np.asarray(horizons)):
            raise ValueError("pooled cache horizons differ")
        for split in splits:
            if not np.array_equal(cached[f"steps_{split}"], splits[split]):
                raise ValueError(f"pooled cache {split} steps differ")
        return {split: cached[f"sequence_{split}"] for split in splits}

    if model is None or edge_cache is None:
        raise ValueError("model and edge cache are required to build pooled cache")

    sequences = {
        split: cache_sequences(
            model,
            features,
            steps,
            horizons,
            checkpoint_args,
            feature_args,
            edge_cache,
            device,
            batch_size,
            split,
            pooling_mode,
        )
        for split, steps in splits.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        cache_version=np.asarray(CACHE_VERSION),
        parent_sha256=np.asarray(parent_sha),
        pooling_mode=np.asarray(pooling_mode),
        horizons=np.asarray(horizons, dtype=np.int64),
        **{f"steps_{split}": steps for split, steps in splits.items()},
        **{
            f"sequence_{split}": sequence
            for split, sequence in sequences.items()
        },
    )
    return sequences


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a joint transition head on cached frozen JEPA graph pools."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--pooled-cache", default=None)
    parser.add_argument(
        "--pooling-mode", choices=("projected", "raw"), default="projected"
    )
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--cache-batch-size", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--edge-cache-workers", type=int, default=8)
    parser.add_argument("--max-fit-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    horizons = parse_int_list(args.horizons)
    model_dir = Path(args.model_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_sha = checkpoint_sha256(model_dir)
    pooled_cache = (
        Path(args.pooled_cache)
        if args.pooled_cache
        else output_dir / "frozen_transition_pool.npz"
    )

    checkpoint = torch.load(
        model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    checkpoint_args = dict(checkpoint.get("args", {}))
    if (
        args.pooling_mode == "projected"
        and checkpoint_args.get("downstream_transition_pooling") != "robust_projected"
    ):
        raise ValueError("cached diagnostic requires robust_projected transition pooling")
    validate_future_rollout_contract(checkpoint_args, horizons, False)
    feature_args = evaluator_namespace(args)
    feature_args.horizons = configured_horizon_text(checkpoint_args, horizons)
    feature_args.edge_cache_workers = int(args.edge_cache_workers)
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(features, checkpoint_args, horizons, int(args.validation_days))
    splits["fit"] = _subsample(splits["fit"], int(args.max_fit_steps))
    splits["validation"] = _subsample(
        splits["validation"], int(args.max_validation_steps)
    )
    splits["test"] = _subsample(splits["test"], int(args.max_test_steps))

    raw_rows = {
        split: _actual_rows(features, steps, horizons, split)
        for split, steps in splits.items()
    }
    contracts = build_target_contracts(raw_rows["fit"], horizons)
    targets = {
        split: build_target_arrays(raw_rows[split], steps, horizons, contracts)
        for split, steps in splits.items()
    }
    fit_event_rate = fit_trajectory_event_rate(targets["fit"])
    model = None
    edge_cache = None
    if not pooled_cache.exists():
        model, loaded_checkpoint = load_model(model_dir, device)
        if loaded_checkpoint.get("checkpoint_epoch") != checkpoint.get(
            "checkpoint_epoch"
        ):
            raise ValueError("loaded checkpoint metadata differs")
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        all_steps = np.unique(np.concatenate(list(splits.values())))
        edge_cache = build_evaluation_edge_cache(
            features, all_steps, checkpoint_args, feature_args
        )
        del loaded_checkpoint
    del checkpoint
    sequences = load_or_build_cache(
        pooled_cache,
        model=model,
        features=features,
        splits=splits,
        horizons=horizons,
        checkpoint_args=checkpoint_args,
        feature_args=feature_args,
        edge_cache=edge_cache,
        device=device,
        batch_size=int(args.cache_batch_size),
        parent_sha=parent_sha,
        pooling_mode=str(args.pooling_mode),
    )
    del model, edge_cache
    if device.type == "mps":
        torch.mps.empty_cache()
    fit_design, validation_design, test_design, design_mean, design_std = (
        normalize_sequences(
            sequences["fit"], sequences["validation"], sequences["test"]
        )
    )

    head = PooledMarketTrajectoryHead(
        fit_design.shape[-1],
        horizons,
        hidden_dim=int(args.hidden_dim),
        layers=int(args.layers),
        heads=int(args.heads),
        dropout=float(args.dropout),
    ).to(device)
    optimizer = torch.optim.AdamW(
        head.parameters(),
        lr=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
    )
    history = []
    best_score = -math.inf
    best_state = None
    stale = 0
    for epoch in range(1, int(args.epochs) + 1):
        train_loss, terms = train_epoch(
            head,
            fit_design,
            targets["fit"],
            contracts,
            horizons,
            optimizer,
            device,
            int(args.batch_size),
            int(args.seed) + epoch,
        )
        validation_predictions = predict_steps(
            head,
            validation_design,
            targets["validation"],
            contracts,
            horizons,
            device,
            int(args.eval_batch_size),
        )
        validation = summarize(
            validation_predictions, contracts, horizons, fit_event_rate
        )
        score = float(validation["validation_formula_score"])
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_terms": terms,
                "validation_score": score,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"validation_market_score={score:+.6f}",
            flush=True,
        )
        if math.isfinite(score) and score > best_score + 1e-4:
            best_score = score
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("cached pooled head produced no valid checkpoint")
    head.load_state_dict(best_state)
    predictions = {
        "validation": predict_steps(
            head,
            validation_design,
            targets["validation"],
            contracts,
            horizons,
            device,
            int(args.eval_batch_size),
        ),
        "test": predict_steps(
            head,
            test_design,
            targets["test"],
            contracts,
            horizons,
            device,
            int(args.eval_batch_size),
        ),
    }
    metrics = {
        split: summarize(values, contracts, horizons, fit_event_rate)
        for split, values in predictions.items()
    }
    for split in predictions:
        _write_csv(
            output_dir / f"daily_{split}.csv",
            _daily_rows(predictions[split], contracts, horizons, split),
        )

    architecture = {
        "input_dim": int(fit_design.shape[-1]),
        "hidden_dim": int(args.hidden_dim),
        "layers": int(args.layers),
        "heads": int(args.heads),
        "dropout": float(args.dropout),
        "joint_horizon_encoder": True,
        "pooling_mode": str(args.pooling_mode),
        "frozen_checkpoint_transition_projector": args.pooling_mode == "projected",
        "pooling": [
            "stock_mean",
            "stock_std",
            "stock_median",
            "external_mean",
            "external_std",
        ],
    }
    summary = {
        "status": "complete",
        "role": "cached_frozen_jepa_robust_transition_head",
        "cache_version": CACHE_VERSION,
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
        "model_dir": str(model_dir),
        "parent_model_sha256": parent_sha,
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "split_dates": {split: len(steps) for split, steps in splits.items()},
        "architecture": architecture,
        "design_mean": design_mean.tolist(),
        "design_std": design_std.tolist(),
        "loss_weights": LOSS_WEIGHTS,
        "sample_weight": "1 + 3 * min(max_family_threshold_ratio, 3)",
        "impact_weighted_event_loss": True,
        "target_contracts": {
            str(horizon): contracts[int(horizon)].to_dict()
            for horizon in horizons
        },
        "fit_cross_horizon_event_rate": fit_event_rate,
        "best_validation_score": best_score,
        "history": history,
        "metrics": metrics,
        "test_used_for_selection": False,
        "selection_status": "research_only_requires_future_confirmation",
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "target_version": MARKET_TRANSITION_TARGET_VERSION,
            "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
            "parent_model_sha256": parent_sha,
            "horizons": horizons,
            "architecture": architecture,
            "design_mean": design_mean,
            "design_std": design_std,
            "target_contracts": summary["target_contracts"],
            "live_orders_allowed": False,
        },
        output_dir / "cached_pooled_market_transition_head.pt",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "best_validation_score": best_score,
                "test_trajectory": metrics["test"]["trajectory"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
