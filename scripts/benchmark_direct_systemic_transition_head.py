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
import torch.nn.functional as F

from scripts.audit_systemic_transition_targets import _actual_rows, _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_systemic_transition_head import (
    LOSS_WEIGHTS,
    _batch_target,
    _daily_rows,
    _masked_component_loss,
    _subsample,
    _target_arrays,
    _validation_score,
    _write_csv,
    build_target_contracts,
    configured_horizon_text,
    fit_trajectory_event_rate,
    summarize_predictions,
    trajectory_metrics,
)
from scripts.benchmark_latent_trajectory_path_head import HORIZON_WEIGHTS, checkpoint_sha256
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.evaluate_auxiliary_trading_policy import (
    _external_state_features,
    _masked_stock_moments,
)
from scripts.run_real_backtest import parse_int_list
from stock_v2.systemic_head import (
    DirectSystemicTransitionHead,
    correlation_rank_loss,
    focal_binary_loss,
    weighted_smooth_l1_loss,
)
from stock_v2.systemic_transition import SYSTEMIC_TARGET_VERSION


def build_design(features, steps):
    stock_values, stock_names = _masked_stock_moments(features, steps)
    external_values, external_names = _external_state_features(features, steps)
    return (
        np.concatenate([stock_values, external_values], axis=1).astype(np.float32),
        np.asarray(stock_names + external_names),
    )


def normalize_design(
    fit: np.ndarray,
    *others: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray]]:
    fit = np.asarray(fit, dtype=np.float64)
    finite = np.isfinite(fit)
    count = finite.sum(axis=0)
    total = np.where(finite, fit, 0.0).sum(axis=0)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0)
    centered = np.where(finite, fit - mean[None, :], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=0),
        count,
        out=np.zeros_like(total),
        where=count > 0,
    )
    std = np.sqrt(np.maximum(variance, 0.0))
    std = np.where(np.isfinite(std) & (std > 1e-8), std, 1.0)

    def transform(values):
        values = np.asarray(values, dtype=np.float64)
        normalized = (values - mean[None, :]) / std[None, :]
        return np.where(np.isfinite(normalized), normalized, 0.0).astype(np.float32)

    return transform(fit), mean.astype(np.float32), std.astype(np.float32), [
        transform(values) for values in others
    ]


def train_epoch(
    head,
    design,
    targets,
    contracts,
    horizons,
    optimizer,
    device,
    batch_size,
    seed,
):
    head.train()
    order = np.random.default_rng(seed).permutation(len(design))
    losses = []
    component_history = {name: [] for name in LOSS_WEIGHTS}
    for start in range(0, len(order), int(batch_size)):
        positions = order[start : start + int(batch_size)]
        values = torch.as_tensor(design[positions], device=device)
        horizon_losses = []
        horizon_weight_sum = 0.0
        batch_components = {name: [] for name in LOSS_WEIGHTS}
        for horizon in horizons:
            component_prediction, energy_prediction, event_logits = head(
                values, int(horizon)
            )
            target = _batch_target(targets, horizon, positions, device)
            component_loss = _masked_component_loss(
                component_prediction,
                target["components"],
                target["component_valid"],
                target["sample_weight"],
            )
            energy_loss = weighted_smooth_l1_loss(
                energy_prediction, target["log_energy"], target["sample_weight"]
            )
            rank_loss = correlation_rank_loss(
                energy_prediction, target["log_energy"]
            )
            event_loss = focal_binary_loss(event_logits[:, 0], target["labels"][:, 0])
            subtype_pos_weight = torch.as_tensor(
                contracts[int(horizon)].subtype_pos_weight,
                dtype=event_logits.dtype,
                device=device,
            )
            subtype_loss = F.binary_cross_entropy_with_logits(
                event_logits[:, 1:],
                target["labels"][:, 1:],
                pos_weight=subtype_pos_weight,
            )
            parts = {
                "components": component_loss,
                "energy": energy_loss,
                "energy_rank": rank_loss,
                "event": event_loss,
                "subtypes": subtype_loss,
            }
            loss = sum(LOSS_WEIGHTS[name] * parts[name] for name in LOSS_WEIGHTS)
            weight = float(HORIZON_WEIGHTS.get(int(horizon), 1.0))
            horizon_losses.append(weight * loss)
            horizon_weight_sum += weight
            for name, part in parts.items():
                batch_components[name].append(part)
        loss = torch.stack(horizon_losses).sum() / horizon_weight_sum
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
        for name, parts in batch_components.items():
            component_history[name].append(
                float(torch.stack(parts).mean().detach().cpu())
            )
    return float(np.mean(losses)), {
        name: float(np.mean(values)) for name, values in component_history.items()
    }


def predict_steps(
    head,
    design,
    targets,
    contracts,
    horizons,
    device,
    batch_size,
):
    head.eval()
    output = {int(horizon): [] for horizon in horizons}
    for start in range(0, len(design), int(batch_size)):
        end = min(start + int(batch_size), len(design))
        values = torch.as_tensor(design[start:end], device=device)
        with torch.no_grad():
            for horizon in horizons:
                normalized_components, log_energy, logits = head(values, int(horizon))
                contract = contracts[int(horizon)]
                raw_components = (
                    normalized_components.float().cpu().numpy()
                    * contract.component_std[None, :]
                    + contract.component_mean[None, :]
                )
                energy = np.maximum(
                    np.expm1(np.clip(log_energy.float().cpu().numpy(), -5.0, 5.0)),
                    0.0,
                )
                event_scores = logits.float().cpu().numpy()
                for position in range(end - start):
                    row_index = start + position
                    predicted_row = {
                        name: float(raw_components[position, index])
                        for index, name in enumerate(
                            contracts[int(horizon)].to_dict()["component_names"]
                        )
                    }
                    actual = targets[int(horizon)]["rows"][row_index]
                    output[int(horizon)].append(
                        {
                            "step": int(actual["step"]),
                            "date": str(actual["date"]),
                            "horizon": int(horizon),
                            "actual": actual,
                            "predicted": predicted_row,
                            "predicted_energy": float(energy[position]),
                            "event_logits": event_scores[position].tolist(),
                        }
                    )
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the same systemic objective on direct causal graph summaries."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--horizon-dim", type=int, default=16)
    parser.add_argument("--dropout", type=float, default=0.10)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-fit-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="cpu")
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
    checkpoint = torch.load(
        model_dir / "graph_jepa_real.pt", map_location="cpu", weights_only=False
    )
    checkpoint_args = dict(checkpoint.get("args", {}))
    feature_args = evaluator_namespace(args)
    feature_args.horizons = configured_horizon_text(checkpoint_args, horizons)
    features, checkpoint_args = build_features_from_ckpt(checkpoint, feature_args)
    splits = _split_steps(
        features, checkpoint_args, horizons, int(args.validation_days)
    )
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
    targets = {
        name: _target_arrays(raw_rows[name], steps, horizons, contracts)
        for name, steps in splits.items()
    }
    fit_trajectory_rate = fit_trajectory_event_rate(
        targets["fit"], contracts, horizons
    )

    fit_design, feature_names = build_design(features, splits["fit"])
    validation_design, validation_names = build_design(
        features, splits["validation"]
    )
    test_design, test_names = build_design(features, splits["test"])
    if not np.array_equal(feature_names, validation_names) or not np.array_equal(
        feature_names, test_names
    ):
        raise ValueError("direct systemic feature contracts do not align")
    fit_design, design_mean, design_std, normalized = normalize_design(
        fit_design, validation_design, test_design
    )
    validation_design, test_design = normalized

    head = DirectSystemicTransitionHead(
        fit_design.shape[1],
        horizons,
        hidden_dim=int(args.hidden_dim),
        horizon_dim=int(args.horizon_dim),
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
        train_loss, train_components = train_epoch(
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
        validation_metrics, validation_score = summarize_predictions(
            validation_predictions, contracts, horizons
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_components": train_components,
                "validation_score": validation_score,
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
            f"validation_systemic_score={validation_score:+.6f}",
            flush=True,
        )
        if math.isfinite(validation_score) and validation_score > best_score + 1e-4:
            best_score = validation_score
            best_state = copy.deepcopy(head.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= int(args.patience):
                break
    if best_state is None:
        raise RuntimeError("direct systemic head produced no valid validation checkpoint")
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
    summaries = {}
    for split in predictions:
        horizon_metrics, score = summarize_predictions(
            predictions[split], contracts, horizons
        )
        summaries[split] = {
            "horizons": horizon_metrics,
            "weighted_validation_formula_score": score,
            "trajectory": trajectory_metrics(
                predictions[split], contracts, horizons, fit_trajectory_rate
            ),
        }
        _write_csv(
            output_dir / f"daily_{split}.csv",
            _daily_rows(predictions[split], contracts, horizons, split),
        )

    parent_sha = checkpoint_sha256(model_dir)
    summary = {
        "status": "complete",
        "role": "same_objective_direct_causal_systemic_transition_head",
        "target_version": SYSTEMIC_TARGET_VERSION,
        "model_dir": str(model_dir),
        "parent_model_sha256": parent_sha,
        "train_end": str(checkpoint_args["train_end"]),
        "horizons": horizons,
        "split_dates": {name: len(steps) for name, steps in splits.items()},
        "input_features": int(fit_design.shape[1]),
        "input_feature_names": feature_names.tolist(),
        "design_mean": design_mean.tolist(),
        "design_std": design_std.tolist(),
        "architecture": {
            "hidden_dim": int(args.hidden_dim),
            "horizon_dim": int(args.horizon_dim),
            "dropout": float(args.dropout),
        },
        "loss_weights": LOSS_WEIGHTS,
        "target_contracts": {
            str(horizon): contracts[int(horizon)].to_dict() for horizon in horizons
        },
        "fit_cross_horizon_event_rate": fit_trajectory_rate,
        "best_validation_score": best_score,
        "history": history,
        "metrics": summaries,
        "fold2_used_for_selection": False,
        "selection_status": "exploratory_only_requires_future_confirmation",
        "live_orders_allowed": False,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    torch.save(
        {
            "state_dict": head.state_dict(),
            "target_version": SYSTEMIC_TARGET_VERSION,
            "parent_model_sha256": parent_sha,
            "horizons": horizons,
            "input_dim": int(fit_design.shape[1]),
            "hidden_dim": int(args.hidden_dim),
            "horizon_dim": int(args.horizon_dim),
            "dropout": float(args.dropout),
            "feature_names": feature_names.tolist(),
            "design_mean": design_mean,
            "design_std": design_std,
            "target_contracts": summary["target_contracts"],
            "live_orders_allowed": False,
        },
        output_dir / "direct_systemic_transition_head.pt",
    )
    print(
        json.dumps(
            {
                "status": "complete",
                "best_validation_score": best_score,
                "test_trajectory": summaries["test"]["trajectory"],
                "live_orders_allowed": False,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
