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
from scripts.benchmark_direct_market_transition_head import (
    build_design,
    normalize_design,
    predict_steps as predict_base_steps,
)
from scripts.benchmark_latent_trajectory_path_head import checkpoint_sha256
from scripts.benchmark_market_transition_head import (
    _daily_rows,
    _subsample,
    _target_batch,
    _write_csv,
    build_target_arrays,
    build_target_contracts,
    configured_horizon_text,
    summarize,
)
from scripts.benchmark_separated_major_path_head import (
    _major_daily,
    combined_loss_terms,
    combined_validation_score,
    separated_major_metrics,
)
from scripts.evaluate_node_prediction import build_features_from_ckpt
from scripts.run_real_backtest import parse_int_list
from stock_v2.major_path_objective import (
    add_major_path_targets,
    fit_major_path_contract,
    major_target_batch,
)
from stock_v2.market_transition import MARKET_TRANSITION_TARGET_VERSION
from stock_v2.separated_major_path import (
    BaseOutputView,
    SEPARATED_MAJOR_LOSS_WEIGHTS,
    SEPARATED_MAJOR_OBJECTIVE_VERSION,
    SeparatedMajorDirectTrajectoryHead,
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
    history = {name: [] for name in SEPARATED_MAJOR_LOSS_WEIGHTS}
    for start in range(0, len(order), int(batch_size)):
        positions = order[start : start + int(batch_size)]
        output = head(torch.as_tensor(design[positions], device=device))
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


def predict_major_steps(head, design, targets, device, batch_size):
    head.eval()
    rows = []
    for start in range(0, len(design), int(batch_size)):
        end = min(start + int(batch_size), len(design))
        with torch.no_grad():
            _base, major = head(torch.as_tensor(design[start:end], device=device))
        horizon_log = major["horizon_log_salience"].float().cpu().numpy()
        major_logit = major["major_logit"].float().cpu().numpy()
        peak_logits = major["peak_logits"].float().cpu().numpy()
        for position in range(end - start):
            index = start + position
            actual = targets["rows"][index][0]
            rows.append(
                {
                    "step": int(actual["step"]),
                    "date": str(actual["date"]),
                    "horizon_log_salience": horizon_log[position].tolist(),
                    "major_logit": float(major_logit[position]),
                    "peak_logits": peak_logits[position].tolist(),
                }
            )
    return rows


def evaluate(
    head,
    design,
    targets,
    contracts,
    horizons,
    path_contract,
    fit_any_rate,
    device,
    batch_size,
):
    base = predict_base_steps(
        BaseOutputView(head),
        design,
        targets,
        contracts,
        horizons,
        device,
        int(batch_size),
    )
    major = predict_major_steps(head, design, targets, device, int(batch_size))
    return base, major, {
        "base": summarize(base, contracts, horizons, fit_any_rate),
        "major_path": separated_major_metrics(major, targets, path_contract),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Direct comparator for separated major-event/timing heads."
    )
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--validation-days", type=int, default=126)
    parser.add_argument("--major-event-quantile", type=float, default=0.90)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
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

    fit_design, feature_names = build_design(features, splits["fit"])
    validation_design, validation_names = build_design(features, splits["validation"])
    test_design, test_names = build_design(features, splits["test"])
    if not np.array_equal(feature_names, validation_names) or not np.array_equal(
        feature_names, test_names
    ):
        raise ValueError("direct separated-major feature contracts do not align")
    fit_design, design_mean, design_std, normalized = normalize_design(
        fit_design, validation_design, test_design
    )
    validation_design, test_design = normalized
    head = SeparatedMajorDirectTrajectoryHead(
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
    fit_any_rate = float(np.mean(targets["fit"]["labels"][..., 0].any(axis=1)))
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
        _base, _major, validation = evaluate(
            head,
            validation_design,
            targets["validation"],
            contracts,
            horizons,
            path_contract,
            fit_any_rate,
            device,
            int(args.eval_batch_size),
        )
        score = combined_validation_score(
            validation["base"], validation["major_path"]
        )
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_terms": terms,
                "validation_score": score,
                "validation_major": validation["major_path"],
            }
        )
        print(
            f"epoch={epoch:03d} train_loss={train_loss:.6f} "
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
        raise RuntimeError("direct separated-major head produced no valid checkpoint")
    head.load_state_dict(best_state)
    split_design = {"validation": validation_design, "test": test_design}
    metrics = {}
    for split, design in split_design.items():
        base, major, metrics[split] = evaluate(
            head,
            design,
            targets[split],
            contracts,
            horizons,
            path_contract,
            fit_any_rate,
            device,
            int(args.eval_batch_size),
        )
        _write_csv(
            output_dir / f"daily_{split}.csv",
            _daily_rows(base, contracts, horizons, split),
        )
        _write_csv(
            output_dir / f"daily_major_{split}.csv",
            _major_daily(major, targets[split], horizons, split),
        )

    parent_sha = checkpoint_sha256(model_dir)
    summary = {
        "status": "complete",
        "role": "same_objective_robust_direct_separated_major_path_head_v32",
        "target_version": MARKET_TRANSITION_TARGET_VERSION,
        "objective_version": SEPARATED_MAJOR_OBJECTIVE_VERSION,
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
            "separate_family_event_peak_outputs": True,
            "causal_stock_statistics": [
                "mean",
                "std",
                "q25",
                "median",
                "q75",
                "availability",
            ],
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
            "input_dim": int(fit_design.shape[1]),
            "architecture": summary["architecture"],
            "feature_names": feature_names.tolist(),
            "design_mean": design_mean,
            "design_std": design_std,
            "loss_weights": SEPARATED_MAJOR_LOSS_WEIGHTS,
            "major_path_contract": path_contract.to_dict(),
            "target_contracts": summary["target_contracts"],
            "live_orders_allowed": False,
        },
        output_dir / "direct_separated_major_path_head.pt",
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
