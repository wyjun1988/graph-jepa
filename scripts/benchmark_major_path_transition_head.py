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
from scripts.benchmark_latent_trajectory_path_head import (
    checkpoint_sha256,
    latent_trajectories,
    snapshot_batch,
)
from scripts.benchmark_market_transition_head import (
    _daily_rows,
    _subsample,
    _target_batch,
    _write_csv,
    build_target_arrays,
    build_target_contracts,
    configured_horizon_text,
    loss_terms,
    predict_steps,
    summarize,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import parse_int_list
from stock_v2.major_path_objective import (
    MAJOR_PATH_LOSS_WEIGHTS,
    MajorPathContract,
    add_major_path_targets,
    family_threshold_matrix,
    fit_major_path_contract,
    major_path_loss_terms,
    major_target_batch,
)
from stock_v2.market_transition import (
    MARKET_TRANSITION_TARGET_VERSION,
    binary_ranking_metrics,
)
from stock_v2.market_transition_head import MarketTrajectoryHead


def combined_loss_terms(
    predictions,
    target,
    contracts,
    horizons,
    path_contract: MajorPathContract,
):
    _unused, base = loss_terms(predictions, target, contracts, horizons)
    major = major_path_loss_terms(
        predictions[1], target, contracts, horizons, path_contract
    )
    terms = {**base, **major}
    loss = sum(
        MAJOR_PATH_LOSS_WEIGHTS[name] * terms[name]
        for name in MAJOR_PATH_LOSS_WEIGHTS
    )
    return loss, terms


def major_metrics(predictions, targets, contracts, horizons, path_contract):
    predicted_family = np.stack(
        [
            np.asarray(
                [row["predicted_families"] for row in predictions[int(horizon)]],
                dtype=np.float64,
            )
            for horizon in horizons
        ],
        axis=1,
    )
    thresholds = family_threshold_matrix(contracts, horizons)
    horizon_salience = np.max(
        predicted_family / np.maximum(thresholds[None, :, :], 1e-8), axis=2
    )
    path_salience = np.max(horizon_salience, axis=1)
    labels = np.asarray(targets["major_label"], dtype=bool)
    ranking = binary_ranking_metrics(
        labels,
        path_salience,
        selection_rate=float(path_contract.fit_event_rate),
    )
    event_rate = float(ranking["event_rate"])
    ranking.update(
        {
            "average_precision_lift": (
                float(ranking["average_precision"]) / event_rate
                if event_rate > 0.0
                else float("nan")
            ),
            "fit_major_event_rate": float(path_contract.fit_event_rate),
            "fit_major_event_threshold": float(path_contract.event_threshold),
            "peak_horizon_accuracy_on_major_events": (
                float(
                    np.mean(
                        np.argmax(horizon_salience[labels], axis=1)
                        == np.asarray(targets["peak_horizon_index"])[labels]
                    )
                )
                if labels.any()
                else float("nan")
            ),
        }
    )
    return ranking


def combined_validation_score(base_summary, major):
    def finite(value, fallback):
        value = float(value)
        return value if math.isfinite(value) else float(fallback)

    auc_skill = np.clip(2.0 * (finite(major["roc_auc"], 0.5) - 0.5), -1.0, 1.0)
    ap_skill = np.clip(finite(major["average_precision_lift"], 1.0) - 1.0, -1.0, 1.0)
    peak_skill = np.clip(
        (finite(major["peak_horizon_accuracy_on_major_events"], 0.20) - 0.20)
        / 0.80,
        -1.0,
        1.0,
    )
    major_score = 0.40 * auc_skill + 0.30 * ap_skill + 0.30 * peak_skill
    return 0.60 * float(base_summary["validation_formula_score"]) + 0.40 * float(
        major_score
    )


def train_epoch(
    model,
    head,
    features,
    steps,
    targets,
    contracts,
    horizons,
    path_contract,
    checkpoint_args,
    feature_args,
    edge_cache,
    optimizer,
    device,
    batch_size,
    seed,
):
    head.train()
    step_to_position = {int(step): index for index, step in enumerate(steps)}
    shuffled = np.random.default_rng(seed).permutation(steps)
    losses = []
    history = {name: [] for name in MAJOR_PATH_LOSS_WEIGHTS}
    for start in range(0, len(shuffled), int(batch_size)):
        selected_steps = np.asarray(
            shuffled[start : start + int(batch_size)], dtype=np.int64
        )
        positions = [step_to_position[int(step)] for step in selected_steps]
        batch = snapshot_batch(
            features, selected_steps, checkpoint_args, feature_args, edge_cache, device
        )
        with torch.no_grad():
            context, predicted = latent_trajectories(
                model, batch, horizons, checkpoint_args
            )
        output = head(
            context,
            predicted,
            batch_size=len(selected_steps),
            node_count=features.node_count,
            stock_count=features.tradable_count,
        )
        target = {
            **_target_batch(targets, positions, device),
            **major_target_batch(targets, positions, device),
        }
        loss, terms = combined_loss_terms(
            output, target, contracts, horizons, path_contract
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for name, value in terms.items():
            history[name].append(float(value.detach().cpu()))
    return float(np.mean(losses)), {
        name: float(np.mean(values)) for name, values in history.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a major-path-aware joint transition head on frozen JEPA latents."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--major-event-quantile", type=float, default=0.90)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--projection-dim", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--edge-cache-workers", type=int, default=16)
    parser.add_argument("--max-fit-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="cuda")
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
    model, checkpoint = load_model(model_dir, device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    checkpoint_args = dict(checkpoint.get("args", {}))
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
        name: _actual_rows(features, steps, horizons, name)
        for name, steps in splits.items()
    }
    contracts = build_target_contracts(raw_rows["fit"], horizons)
    base_targets = {
        name: build_target_arrays(raw_rows[name], steps, horizons, contracts)
        for name, steps in splits.items()
    }
    path_contract = fit_major_path_contract(
        base_targets["fit"],
        contracts,
        horizons,
        event_quantile=float(args.major_event_quantile),
    )
    targets = {
        name: add_major_path_targets(values, contracts, horizons, path_contract)
        for name, values in base_targets.items()
    }
    all_steps = np.unique(np.concatenate(list(splits.values())))
    edge_cache = build_evaluation_edge_cache(
        features, all_steps, checkpoint_args, feature_args
    )

    head = MarketTrajectoryHead(
        int(checkpoint_args["hidden_dim"]),
        horizons,
        projection_dim=int(args.projection_dim),
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
            model,
            head,
            features,
            splits["fit"],
            targets["fit"],
            contracts,
            horizons,
            path_contract,
            checkpoint_args,
            feature_args,
            edge_cache,
            optimizer,
            device,
            int(args.batch_size),
            int(args.seed) + epoch,
        )
        validation_predictions = predict_steps(
            model,
            head,
            features,
            splits["validation"],
            targets["validation"],
            contracts,
            horizons,
            checkpoint_args,
            feature_args,
            edge_cache,
            device,
            int(args.eval_batch_size),
        )
        validation_base = summarize(
            validation_predictions,
            contracts,
            horizons,
            float(np.mean(targets["fit"]["labels"][..., 0].any(axis=1))),
        )
        validation_major = major_metrics(
            validation_predictions,
            targets["validation"],
            contracts,
            horizons,
            path_contract,
        )
        score = combined_validation_score(validation_base, validation_major)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_terms": terms,
                "validation_score": score,
                "validation_major": validation_major,
            }
        )
        print(
            f"epoch={epoch:02d} train_loss={train_loss:.6f} "
            f"validation_major_path_score={score:+.6f}",
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
        raise RuntimeError("major-path head produced no valid checkpoint")
    head.load_state_dict(best_state)
    predictions = {
        split: predict_steps(
            model,
            head,
            features,
            splits[split],
            targets[split],
            contracts,
            horizons,
            checkpoint_args,
            feature_args,
            edge_cache,
            device,
            int(args.eval_batch_size),
        )
        for split in ("validation", "test")
    }
    fit_any_rate = float(np.mean(targets["fit"]["labels"][..., 0].any(axis=1)))
    metrics = {}
    for split, values in predictions.items():
        metrics[split] = {
            "base": summarize(values, contracts, horizons, fit_any_rate),
            "major_path": major_metrics(
                values, targets[split], contracts, horizons, path_contract
            ),
        }
        _write_csv(
            output_dir / f"daily_{split}.csv",
            _daily_rows(values, contracts, horizons, split),
        )

    parent_sha = checkpoint_sha256(model_dir)
    summary = {
        "status": "complete",
        "role": "posthoc_frozen_jepa_major_path_transition_head_v31",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "objective_version": "major_path_v31_20260714",
        "model_dir": str(model_dir),
        "parent_model_sha256": parent_sha,
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "split_dates": {name: len(steps) for name, steps in splits.items()},
        "architecture": {
            "projection_dim": int(args.projection_dim),
            "hidden_dim": int(args.hidden_dim),
            "layers": int(args.layers),
            "heads": int(args.heads),
            "dropout": float(args.dropout),
            "joint_horizon_encoder": True,
            "individual_node_max_pooling": False,
        },
        "loss_weights": MAJOR_PATH_LOSS_WEIGHTS,
        "major_path_contract": path_contract.to_dict(),
        "target_contracts": {
            str(horizon): contracts[int(horizon)].to_dict() for horizon in horizons
        },
        "best_validation_score": best_score,
        "history": history,
        "metrics": metrics,
        "test_used_for_selection": False,
        "selection_status": "exploratory_only_requires_future_confirmation",
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "target_version": MARKET_TRANSITION_TARGET_VERSION,
            "objective_version": summary["objective_version"],
            "parent_model_sha256": parent_sha,
            "horizons": horizons,
            "architecture": summary["architecture"],
            "loss_weights": MAJOR_PATH_LOSS_WEIGHTS,
            "major_path_contract": path_contract.to_dict(),
            "target_contracts": summary["target_contracts"],
            "live_orders_allowed": False,
        },
        output_dir / "major_path_transition_head.pt",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "best_validation_score": best_score,
                "test_major_path": metrics["test"]["major_path"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
