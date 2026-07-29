"""Do the four downstream heads predict anything? Intent 3's actual question.

The rolling gate scores STATE. It says nothing about whether the heads bolted on
top of the shared state are any good, so a model can pass 169/169 with four
heads that emit noise. Intent 3 -- "several downstream heads extract
high-dimensional data for return strategies" -- is only served if they predict.

WHY THIS IS LEAK-FREE, unlike the plan evaluation. The heads emit values
standardized across the cross-section within each (date, horizon), and the
targets are standardized by the same per-date statistics. A correlation between
the two is invariant to that shared affine transform, so nothing has to be
de-standardized and the date's realized mean never enters. The plan loss needed
de-standardization only because it had to RANK horizons against each other;
scoring a head against its own target does not.

Each task is scored per date as a cross-sectional Pearson IC against its own
realized target, then averaged over sessions with a moving-block bootstrap. The
placebo shuffles predictions across stocks within a date: it preserves the
prediction distribution exactly and breaks only the stock-to-prediction pairing,
so it isolates whether the head knows which stock is which.

Evidence class: research. Nothing here qualifies or promotes a model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.evaluate_node_prediction import (
    build_features_from_ckpt,
    graph_edge_kwargs,
    load_model,
    select_steps,
)
from scripts.evaluate_plan_timing import evaluator_contract_defaults, moving_block_bootstrap
from scripts.run_real_backtest import attach_downstream_targets, rollout_steps_for_offset
from stock_v2.graph_jepa import DOWNSTREAM_AUXILIARY_TASKS
from stock_v2.real_features import make_real_snapshot

PLACEBO_SEED = 20260717


def cross_sectional_ic(prediction: np.ndarray, target: np.ndarray) -> float:
    finite = np.isfinite(prediction) & np.isfinite(target)
    if finite.sum() < 30:
        return float("nan")
    a = prediction[finite]
    b = target[finite]
    if a.std() < 1e-12 or b.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--max-steps", type=int, default=194)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--bootstrap-block", type=int, default=5)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = evaluator_contract_defaults(parser.parse_args())

    horizons = [int(v) for v in args.horizons.split(",") if v.strip()]
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    model, ckpt = load_model(model_dir, device)
    features, ckpt_args = build_features_from_ckpt(ckpt, args)
    namespace = argparse.Namespace(**ckpt_args)
    steps = select_steps(features, ckpt_args, args)
    stock_count = int(features.tradable_count)
    node_count = int(features.node_count)

    edge_window = int(ckpt_args.get("edge_window", 60))
    edge_top_k = int(ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(ckpt_args.get("min_abs_corr", 0.2))
    generator = np.random.default_rng(PLACEBO_SEED)

    rows: list[dict[str, Any]] = []
    for step in steps:
        batch = make_real_snapshot(
            features,
            step=int(step),
            full_observation=True,
            edge_window=edge_window,
            top_k=edge_top_k,
            min_abs_corr=min_abs_corr,
            **graph_edge_kwargs(ckpt_args, args),
        ).to(device)
        with torch.no_grad():
            context = model.encode_temporal_context(batch)

        for horizon in horizons:
            steps_forward = max(1, int(rollout_steps_for_offset(namespace, int(horizon))))
            with torch.no_grad():
                z_pred = model.rollout_latent(context, steps=steps_forward)
                head = (
                    model.predict_downstream_auxiliary(
                        context, z_pred, rollout_steps=steps_forward
                    )
                    .detach()
                    .cpu()
                    .numpy()
                )

            # attach_downstream_targets is imported, not reimplemented: the
            # targets must be the exact quantity the heads were trained against,
            # standardization and validity rules included.
            target_batch = make_real_snapshot(
                features,
                step=int(step),
                full_observation=True,
                edge_window=edge_window,
                top_k=edge_top_k,
                min_abs_corr=min_abs_corr,
                **graph_edge_kwargs(ckpt_args, args),
            )
            target_batch = attach_downstream_targets(
                target_batch, features, np.array([int(step)]), int(horizon)
            )
            targets = target_batch.target_downstream.numpy().reshape(node_count, -1)

            row: dict[str, Any] = {
                "date": str(pd.Timestamp(features.dates[int(step)]).date()),
                "horizon": int(horizon),
            }
            order = generator.permutation(stock_count)
            for index, task in enumerate(DOWNSTREAM_AUXILIARY_TASKS):
                prediction = head[:stock_count, index]
                target = targets[:stock_count, index]
                row[f"ic_{task}"] = cross_sectional_ic(prediction, target)
                row[f"placebo_ic_{task}"] = cross_sectional_ic(prediction[order], target)
            rows.append(row)

    if not rows:
        raise SystemExit("no session scored")
    frame = pd.DataFrame(rows)

    boot = dict(block=args.bootstrap_block, samples=args.bootstrap_samples, seed=args.seed)
    results: dict[str, Any] = {}
    for task in DOWNSTREAM_AUXILIARY_TASKS:
        per_horizon: dict[str, Any] = {}
        for horizon in horizons:
            subset = frame[frame["horizon"] == horizon]
            values = subset[f"ic_{task}"].to_numpy()
            values = values[np.isfinite(values)]
            placebo = subset[f"placebo_ic_{task}"].to_numpy()
            placebo = placebo[np.isfinite(placebo)]
            if not len(values):
                continue
            lower, upper = moving_block_bootstrap(values, **boot)
            per_horizon[str(horizon)] = {
                "sessions": int(len(values)),
                "mean_ic": float(values.mean()),
                "boot_lower_95": lower,
                "boot_upper_95": upper,
                "positive_session_fraction": float((values > 0).mean()),
                "placebo_mean_ic": float(placebo.mean()) if len(placebo) else float("nan"),
                "beats_zero": bool(lower > 0),
            }
        results[task] = per_horizon

    payload = {
        "role": "research_only_downstream_head_quality",
        "live_orders_allowed": False,
        "test_used_for_selection": False,
        "promotion_eligible": False,
        "model_dir": str(model_dir),
        "checkpoint_sha256": hashlib.sha256(
            (model_dir / "graph_jepa_real.pt").read_bytes()
        ).hexdigest(),
        "train_data_manifest_sha256": ckpt.get("train_data_manifest_sha256"),
        "sessions": int(frame["date"].nunique()),
        "tasks": list(DOWNSTREAM_AUXILIARY_TASKS),
        "metric": "per-date cross-sectional Pearson IC of the head output against its own standardized target",
        "leak_note": "both sides carry the same per-date standardization, so the correlation is invariant to it and the realized date-level mean never enters the score",
        "placebo": f"predictions shuffled across stocks within a date, seed {PLACEBO_SEED}",
        "results": results,
    }
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output / "daily_head_ic.csv", index=False)
    (output / "summary.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    print(f"sessions: {payload['sessions']}\n")
    print(f"{'task':26}{'h':>4}{'mean IC':>10}{'95% low':>10}{'placebo':>10}  판정")
    for task, per_horizon in results.items():
        for horizon, stats in per_horizon.items():
            print(
                f"{task:26}{horizon:>4}{stats['mean_ic']:>10.4f}"
                f"{stats['boot_lower_95']:>10.4f}{stats['placebo_mean_ic']:>10.4f}"
                f"  {'PASS' if stats['beats_zero'] else '0 포함'}"
            )
    print(f"\n-> {output}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
