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
import pandas as pd
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
    summarize,
)
from scripts.evaluate_node_prediction import (
    build_evaluation_edge_cache,
    build_features_from_ckpt,
    load_model,
    pearson,
    validate_future_rollout_contract,
)
from scripts.run_real_backtest import parse_int_list
from stock_v2.major_path_objective import (
    add_major_path_targets,
    fit_major_path_contract,
    major_target_batch,
)
from stock_v2.market_transition import (
    MARKET_TRANSITION_TARGET_VERSION,
    binary_ranking_metrics,
)
from stock_v2.market_transition_head import MARKET_COMPONENT_TARGETS
from stock_v2.separated_major_path import (
    SEPARATED_MAJOR_LOSS_WEIGHTS,
    SEPARATED_MAJOR_OBJECTIVE_VERSION,
    SeparatedMajorMarketTrajectoryHead,
    separated_major_loss_terms,
)


def combined_loss_terms(output, target, contracts, horizons):
    base_output, major_output = output
    _unused, base = loss_terms(base_output, target, contracts, horizons)
    major = separated_major_loss_terms(major_output, target)
    terms = {**base, **major}
    loss = sum(
        SEPARATED_MAJOR_LOSS_WEIGHTS[name] * terms[name]
        for name in SEPARATED_MAJOR_LOSS_WEIGHTS
    )
    return loss, terms


def separated_major_metrics(rows, targets, path_contract):
    labels = np.asarray(targets["major_label"], dtype=bool)
    event_scores = np.asarray([row["major_logit"] for row in rows], dtype=np.float64)
    horizon_log = np.asarray(
        [row["horizon_log_salience"] for row in rows], dtype=np.float64
    )
    peak_logits = np.asarray([row["peak_logits"] for row in rows], dtype=np.float64)
    predicted_path_log = np.max(horizon_log, axis=1)
    actual_path_log = np.log1p(np.asarray(targets["path_salience"], dtype=np.float64))
    ranking = binary_ranking_metrics(
        labels,
        event_scores,
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
            "path_log_mae": float(np.mean(np.abs(predicted_path_log - actual_path_log))),
            "path_log_correlation": pearson(predicted_path_log, actual_path_log),
            "horizon_log_salience_correlation": pearson(
                horizon_log.reshape(-1),
                np.log1p(
                    np.asarray(targets["horizon_salience"], dtype=np.float64)
                ).reshape(-1),
            ),
            "peak_horizon_accuracy_on_major_events": (
                float(
                    np.mean(
                        np.argmax(peak_logits[labels], axis=1)
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
    ap_skill = np.clip(
        finite(major["average_precision_lift"], 1.0) - 1.0, -1.0, 1.0
    )
    peak_skill = np.clip(
        (finite(major["peak_horizon_accuracy_on_major_events"], 0.20) - 0.20)
        / 0.80,
        -1.0,
        1.0,
    )
    path_skill = np.clip(finite(major["path_log_correlation"], 0.0), -1.0, 1.0)
    major_score = (
        0.30 * auc_skill + 0.25 * ap_skill + 0.25 * peak_skill + 0.20 * path_skill
    )
    return 0.70 * float(base_summary["validation_formula_score"]) + 0.30 * float(
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
    history = {name: [] for name in SEPARATED_MAJOR_LOSS_WEIGHTS}
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
        loss, terms = combined_loss_terms(output, target, contracts, horizons)
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


def predict_steps(
    model,
    head,
    features,
    steps,
    targets,
    contracts,
    horizons,
    checkpoint_args,
    feature_args,
    edge_cache,
    device,
    batch_size,
):
    head.eval()
    base_records = {int(horizon): [] for horizon in horizons}
    major_rows = []
    for start in range(0, len(steps), int(batch_size)):
        selected_steps = np.asarray(
            steps[start : start + int(batch_size)], dtype=np.int64
        )
        batch = snapshot_batch(
            features, selected_steps, checkpoint_args, feature_args, edge_cache, device
        )
        with torch.no_grad():
            context, predicted = latent_trajectories(
                model, batch, horizons, checkpoint_args
            )
            base, major = head(
                context,
                predicted,
                batch_size=len(selected_steps),
                node_count=features.node_count,
                stock_count=features.tradable_count,
            )
        normalized_component, family_log, event_logits = base
        component_numpy = normalized_component.float().cpu().numpy()
        family_numpy = np.maximum(
            np.expm1(np.clip(family_log.float().cpu().numpy(), -5.0, 5.0)), 0.0
        )
        event_numpy = event_logits.float().cpu().numpy()
        horizon_log = major["horizon_log_salience"].float().cpu().numpy()
        major_logit = major["major_logit"].float().cpu().numpy()
        peak_logits = major["peak_logits"].float().cpu().numpy()
        for position, step in enumerate(selected_steps):
            major_rows.append(
                {
                    "step": int(step),
                    "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                    "horizon_log_salience": horizon_log[position].tolist(),
                    "major_logit": float(major_logit[position]),
                    "peak_logits": peak_logits[position].tolist(),
                }
            )
        for horizon_index, horizon in enumerate(horizons):
            contract = contracts[int(horizon)]
            raw_component = (
                component_numpy[:, horizon_index] * contract.component_std[None, :]
                + contract.component_mean[None, :]
            )
            for position, step in enumerate(selected_steps):
                base_records[int(horizon)].append(
                    {
                        "step": int(step),
                        "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                        "horizon": int(horizon),
                        "actual": targets["rows"][start + position][horizon_index],
                        "predicted": {
                            name: float(raw_component[position, component_index])
                            for component_index, name in enumerate(
                                MARKET_COMPONENT_TARGETS
                            )
                        },
                        "predicted_families": family_numpy[
                            position, horizon_index
                        ].tolist(),
                        "event_logits": event_numpy[position, horizon_index].tolist(),
                    }
                )
    return base_records, major_rows


def _major_daily(rows, targets, horizons, split):
    output = []
    for index, row in enumerate(rows):
        output.append(
            {
                "split": split,
                "date": row["date"],
                "actual_major_event": bool(targets["major_label"][index]),
                "actual_path_salience": float(targets["path_salience"][index]),
                "predicted_path_salience": float(
                    np.expm1(np.max(row["horizon_log_salience"]))
                ),
                "major_logit": float(row["major_logit"]),
                "actual_peak_horizon": int(
                    horizons[int(targets["peak_horizon_index"][index])]
                ),
                "predicted_peak_horizon": int(
                    horizons[int(np.argmax(row["peak_logits"]))]
                ),
                **{
                    f"actual_horizon_salience_{horizon}": float(
                        targets["horizon_salience"][index, horizon_index]
                    )
                    for horizon_index, horizon in enumerate(horizons)
                },
                **{
                    f"predicted_horizon_salience_{horizon}": float(
                        np.expm1(row["horizon_log_salience"][horizon_index])
                    )
                    for horizon_index, horizon in enumerate(horizons)
                },
            }
        )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train independent major-event/timing heads beside family forecasts."
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
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
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
    for split, argument in (
        ("fit", args.max_fit_steps),
        ("validation", args.max_validation_steps),
        ("test", args.max_test_steps),
    ):
        splits[split] = _subsample(splits[split], int(argument))
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
    head = SeparatedMajorMarketTrajectoryHead(
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
    fit_any_rate = float(np.mean(targets["fit"]["labels"][..., 0].any(axis=1)))
    for epoch in range(1, int(args.epochs) + 1):
        train_loss, terms = train_epoch(
            model,
            head,
            features,
            splits["fit"],
            targets["fit"],
            contracts,
            horizons,
            checkpoint_args,
            feature_args,
            edge_cache,
            optimizer,
            device,
            int(args.batch_size),
            int(args.seed) + epoch,
        )
        validation_base_rows, validation_major_rows = predict_steps(
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
            validation_base_rows, contracts, horizons, fit_any_rate
        )
        validation_major = separated_major_metrics(
            validation_major_rows, targets["validation"], path_contract
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
            f"validation_separated_major_score={score:+.6f}",
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
        raise RuntimeError("separated major-path head produced no valid checkpoint")
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
    metrics = {}
    for split, (base_rows, major_rows) in predictions.items():
        metrics[split] = {
            "base": summarize(base_rows, contracts, horizons, fit_any_rate),
            "major_path": separated_major_metrics(
                major_rows, targets[split], path_contract
            ),
        }
        _write_csv(
            output_dir / f"daily_{split}.csv",
            _daily_rows(base_rows, contracts, horizons, split),
        )
        _write_csv(
            output_dir / f"daily_major_{split}.csv",
            _major_daily(major_rows, targets[split], horizons, split),
        )
    parent_sha = checkpoint_sha256(model_dir)
    summary = {
        "status": "complete",
        "role": "posthoc_frozen_jepa_separated_major_path_head_v32",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "objective_version": SEPARATED_MAJOR_OBJECTIVE_VERSION,
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
            "separate_family_event_peak_outputs": True,
            "individual_node_max_pooling": False,
        },
        "loss_weights": SEPARATED_MAJOR_LOSS_WEIGHTS,
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
            "objective_version": SEPARATED_MAJOR_OBJECTIVE_VERSION,
            "parent_model_sha256": parent_sha,
            "horizons": horizons,
            "architecture": summary["architecture"],
            "loss_weights": SEPARATED_MAJOR_LOSS_WEIGHTS,
            "major_path_contract": path_contract.to_dict(),
            "target_contracts": summary["target_contracts"],
            "live_orders_allowed": False,
        },
        output_dir / "separated_major_path_head.pt",
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
