from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from scripts.audit_market_transition_targets import _actual_rows
from scripts.audit_systemic_transition_targets import _split_steps
from scripts.benchmark_direct_baselines import evaluator_namespace
from scripts.benchmark_direct_systemic_transition_head import normalize_design
from scripts.benchmark_latent_trajectory_path_head import checkpoint_sha256
from scripts.benchmark_market_transition_head import (
    LOSS_WEIGHTS,
    _daily_rows,
    _subsample,
    _target_batch,
    _write_csv,
    build_target_arrays,
    build_target_contracts,
    configured_horizon_text,
    fit_trajectory_event_rate,
    loss_terms,
    summarize,
)
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.run_real_backtest import parse_int_list
from stock_v2.market_transition import (
    MARKET_TRANSITION_IMPACT_METRIC_VERSION,
    MARKET_TRANSITION_TARGET_VERSION,
)
from stock_v2.market_transition_head import (
    MARKET_COMPONENT_TARGETS,
    MARKET_EVENT_TARGETS,
    MARKET_FAMILY_TARGETS,
    DirectMarketTrajectoryHead,
)


def build_design(features, steps):
    """Causal robust stock distributions plus observed external-node values."""

    from scripts.evaluate_auxiliary_trading_policy import _external_state_features

    stock_count = int(features.tradable_count)
    raw = features.raw_features[steps, :stock_count].astype(np.float64)
    available = features.available_mask[steps, :stock_count] > 0.5
    count = available.sum(axis=1).astype(np.float64)
    total = np.where(available, raw, 0.0).sum(axis=1)
    mean = np.divide(total, count, out=np.zeros_like(total), where=count > 0.0)
    centered = np.where(available, raw - mean[:, None, :], 0.0)
    variance = np.divide(
        np.square(centered).sum(axis=1),
        count,
        out=np.zeros_like(total),
        where=count > 0.0,
    )
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    masked = np.where(available, raw, np.nan)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        quartiles = np.nanquantile(masked, (0.25, 0.50, 0.75), axis=1)
    quartiles = np.where(np.isfinite(quartiles), quartiles, 0.0)
    availability = count / float(stock_count)
    stock_values = np.concatenate(
        [
            mean,
            standard_deviation,
            quartiles[0],
            quartiles[1],
            quartiles[2],
            availability,
        ],
        axis=1,
    ).astype(np.float32)
    stock_names = []
    for prefix in (
        "stock_mean",
        "stock_std",
        "stock_q25",
        "stock_median",
        "stock_q75",
        "stock_available",
    ):
        stock_names.extend(f"{prefix}:{name}" for name in features.feature_names)
    external_values, external_names = _external_state_features(features, steps)
    return (
        np.concatenate([stock_values, external_values], axis=1).astype(np.float32),
        np.asarray(stock_names + external_names),
    )


def parse_transition_history_lags(value: str) -> tuple[int, ...]:
    if not str(value).strip():
        return ()
    lags = tuple(sorted({int(item.strip()) for item in str(value).split(",")}))
    if not lags or any(lag < 1 for lag in lags):
        raise ValueError("transition history lags must be positive integers")
    return lags


def assemble_transition_history_design(
    steps,
    history_steps,
    history_targets,
    lags,
):
    """Map completed one-step market transitions to each causal context."""

    steps = np.asarray(steps, dtype=np.int64)
    history_steps = np.asarray(history_steps, dtype=np.int64)
    normalized_lags = tuple(int(lag) for lag in lags)
    if not normalized_lags:
        return np.zeros((len(steps), 0), dtype=np.float32), np.asarray([], dtype=str)
    lookup = {int(step): index for index, step in enumerate(history_steps)}
    component = np.asarray(history_targets["components"], dtype=np.float32)[:, 0]
    family = np.asarray(history_targets["family_log"], dtype=np.float32)[:, 0]
    labels = np.asarray(history_targets["labels"], dtype=np.float32)[:, 0]
    transition = np.concatenate((component, family, labels), axis=1)
    expected_width = (
        len(MARKET_COMPONENT_TARGETS)
        + len(MARKET_FAMILY_TARGETS)
        + len(MARKET_EVENT_TARGETS)
    )
    if transition.shape != (len(history_steps), expected_width):
        raise ValueError("invalid one-step transition history target layout")

    columns = []
    for lag in normalized_lags:
        try:
            positions = np.asarray(
                [lookup[int(step) - lag] for step in steps], dtype=np.int64
            )
        except KeyError as error:
            raise ValueError(
                f"missing completed transition history for step {error.args[0]}"
            ) from error
        columns.append(transition[positions])
    names = []
    base_names = [
        *[f"component:{name}" for name in MARKET_COMPONENT_TARGETS],
        *[f"family_log:{name}" for name in MARKET_FAMILY_TARGETS],
        *[f"event:{name}" for name in MARKET_EVENT_TARGETS],
    ]
    for lag in normalized_lags:
        names.extend(f"transition_lag{lag}:{name}" for name in base_names)
    return np.concatenate(columns, axis=1), np.asarray(names)


def build_transition_history_design(features, steps, contracts, lags):
    normalized_lags = tuple(int(lag) for lag in lags)
    if not normalized_lags:
        return np.zeros((len(steps), 0), dtype=np.float32), np.asarray([], dtype=str)
    if 1 not in contracts:
        raise ValueError("transition history requires the one-step target contract")
    steps = np.asarray(steps, dtype=np.int64)
    history_steps = np.asarray(
        sorted(
            {
                int(step) - int(lag)
                for step in steps
                for lag in normalized_lags
            }
        ),
        dtype=np.int64,
    )
    if history_steps.size == 0 or int(history_steps.min()) < 0:
        raise ValueError("transition history precedes the available feature panel")
    history_rows = _actual_rows(features, history_steps, [1], "causal_history")
    history_targets = build_target_arrays(
        history_rows,
        history_steps,
        [1],
        {1: contracts[1]},
    )
    return assemble_transition_history_design(
        steps,
        history_steps,
        history_targets,
        normalized_lags,
    )


def build_design_with_transition_history(features, steps, contracts, lags):
    current, current_names = build_design(features, steps)
    history, history_names = build_transition_history_design(
        features, steps, contracts, lags
    )
    return (
        np.concatenate((current, history), axis=1).astype(np.float32),
        np.concatenate((current_names, history_names)),
    )


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
    history = {name: [] for name in LOSS_WEIGHTS}
    for start in range(0, len(order), int(batch_size)):
        positions = order[start : start + int(batch_size)]
        values = torch.as_tensor(design[positions], device=device)
        output = head(values)
        target = _target_batch(targets, positions, device)
        loss, terms = loss_terms(output, target, contracts, horizons)
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
            normalized_component, family_log, event_logits = head(values)
        component_numpy = normalized_component.float().cpu().numpy()
        family_numpy = np.maximum(
            np.expm1(np.clip(family_log.float().cpu().numpy(), -5.0, 5.0)), 0.0
        )
        event_numpy = event_logits.float().cpu().numpy()
        for horizon_index, horizon in enumerate(horizons):
            contract = contracts[int(horizon)]
            raw_component = (
                component_numpy[:, horizon_index] * contract.component_std[None, :]
                + contract.component_mean[None, :]
            )
            for position in range(end - start):
                row_index = start + position
                actual = targets["rows"][row_index][horizon_index]
                output[int(horizon)].append(
                    {
                        "step": int(actual["step"]),
                        "date": str(actual["date"]),
                        "horizon": int(horizon),
                        "actual": actual,
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
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the joint market-transition objective on direct causal summaries."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=512)
    parser.add_argument("--max-fit-steps", type=int, default=0)
    parser.add_argument("--max-validation-steps", type=int, default=0)
    parser.add_argument("--max-test-steps", type=int, default=0)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--seed", type=int, default=2701)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--external-cache-dir", default=None)
    parser.add_argument(
        "--transition-history-lags",
        default="",
        help="Completed one-step market-transition lags, for example 1,2,5.",
    )
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
    targets = {
        name: build_target_arrays(raw_rows[name], steps, horizons, contracts)
        for name, steps in splits.items()
    }
    fit_event_rate = fit_trajectory_event_rate(targets["fit"])
    transition_history_lags = parse_transition_history_lags(
        args.transition_history_lags
    )

    fit_design, feature_names = build_design_with_transition_history(
        features, splits["fit"], contracts, transition_history_lags
    )
    validation_design, validation_names = build_design_with_transition_history(
        features, splits["validation"], contracts, transition_history_lags
    )
    test_design, test_names = build_design_with_transition_history(
        features, splits["test"], contracts, transition_history_lags
    )
    if not np.array_equal(feature_names, validation_names) or not np.array_equal(
        feature_names, test_names
    ):
        raise ValueError("direct market-transition feature contracts do not align")
    fit_design, design_mean, design_std, normalized = normalize_design(
        fit_design, validation_design, test_design
    )
    validation_design, test_design = normalized

    head = DirectMarketTrajectoryHead(
        fit_design.shape[1],
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
        raise RuntimeError("direct market trajectory head produced no valid checkpoint")
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

    parent_sha = checkpoint_sha256(model_dir)
    summary = {
        "status": "complete",
        "role": "same_objective_direct_joint_market_transition_head",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
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
            "layers": int(args.layers),
            "heads": int(args.heads),
            "dropout": float(args.dropout),
            "joint_horizon_encoder": True,
            "causal_stock_statistics": [
                "mean",
                "std",
                "q25",
                "median",
                "q75",
                "availability",
            ],
            "transition_history_lags": list(transition_history_lags),
            "transition_history_semantics": (
                "lag L is the completed one-step transition from t-L to t-L+1"
            ),
        },
        "loss_weights": LOSS_WEIGHTS,
        "sample_weight": "1 + 3 * min(max_family_threshold_ratio, 3)",
        "impact_weighted_event_loss": True,
        "target_contracts": {
            str(horizon): contracts[int(horizon)].to_dict() for horizon in horizons
        },
        "fit_cross_horizon_event_rate": fit_event_rate,
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
            "impact_metric_version": MARKET_TRANSITION_IMPACT_METRIC_VERSION,
            "parent_model_sha256": parent_sha,
            "horizons": horizons,
            "input_dim": int(fit_design.shape[1]),
            "architecture": summary["architecture"],
            "feature_names": feature_names.tolist(),
            "design_mean": design_mean,
            "design_std": design_std,
            "target_contracts": summary["target_contracts"],
            "live_orders_allowed": False,
        },
        output_dir / "direct_market_transition_head.pt",
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
