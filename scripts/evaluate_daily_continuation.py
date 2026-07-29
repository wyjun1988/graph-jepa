"""Score the continuation head on intent 2. The gate's missing input.

The head emits a CROSS-SECTIONALLY STANDARDIZED continuation rate, and the
frozen contract's model score is a ratio -- predicted_future_rate divided by the
node's own observed rate. Forming that ratio needs the prediction in raw units,
which means de-standardising, which is precisely the operation that leaked in the
plan loss: multiplying by the decision date's REALIZED cross-sectional mean hands
the model the answer, and a rule that never looked at the model scored +0.01404
where the leaky arm scored +0.01416.

So the scale here is causal by construction: a trailing window over horizons that
had already completed before the decision date. Same fix, same reasoning, applied
before the number exists rather than after it was believed.

TWO CONDITIONS, and the second is the one that matters:

  vs the FRONTIER   the best decision-time feature reaches effective AUC 0.622 /
                    0.642 / 0.651 at h1/h2/h3. A head under that is worse than a
                    statistic already in its own input.
  vs the PLACEBO    predictions shuffled across stocks within a date. This is not
                    optional book-keeping. The ratio has the observed rate in its
                    DENOMINATOR, and the frontier's own finding is that a bigger
                    shock fades -- so a head that predicts a constant still ranks
                    correctly, purely through the denominator, and would clear the
                    frontier while contributing nothing. The placebo is the only
                    thing that separates the head from its denominator.

Effective AUC is 0.5 + |auc - 0.5| throughout, because every leading frontier
feature scores BELOW 0.5: bigger and more volatile shocks fade, so a raw 0.378 is
exactly as informative as 0.622 once sign-aligned.

Evidence class: research. Nothing here promotes a model.
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
from scripts.evaluate_plan_timing import causal_path_scale, evaluator_contract_defaults
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.daily_continuation import (
    build_cells,
    continuation_ratio_score,
    continuation_threshold,
    shock_statistic,
)
from stock_v2.graph_jepa import DOWNSTREAM_AUXILIARY_TASKS
from stock_v2.prospective_recompute import development_auc
from stock_v2.real_features import make_real_snapshot

PLACEBO_SEED = 20260717


def causal_rate_scale(features, step: int, horizon: int, lookback: int) -> tuple[float, float]:
    """Trailing mean/std of the continuation rate, from horizons already finished.

    causal_path_scale does this for target_return_paths; the continuation rate is
    a different quantity (net displacement per day, not the path return), so it
    needs its own window over the same causal boundary: an entry at s for horizon
    h is unknown until s+h, so at t the last usable entry is t-h (t-h-1 here, one
    session of margin).
    """

    stock_count = int(features.tradable_count)
    last_known = int(step) - int(horizon) - 1
    first = max(0, last_known - int(lookback) + 1)
    if last_known < first:
        return float("nan"), float("nan")
    values: list[np.ndarray] = []
    for entry in range(first, last_known + 1):
        if entry + horizon >= features.close.shape[0]:
            continue
        start = np.asarray(features.close[entry, :stock_count], dtype=np.float64)
        end = np.asarray(features.close[entry + horizon, :stock_count], dtype=np.float64)
        usable = np.isfinite(start) & np.isfinite(end) & (start > 0.0)
        rate = np.full(stock_count, np.nan)
        rate[usable] = np.abs(end[usable] / start[usable] - 1.0) / float(horizon)
        values.append(rate)
    if not values:
        return float("nan"), float("nan")
    window = np.concatenate(values)
    finite = window[np.isfinite(window)]
    if finite.size < 30:
        return float("nan"), float("nan")
    mean = float(finite.mean())
    std = float(finite.std())
    if not np.isfinite(std) or std < 1e-12:
        return float("nan"), float("nan")
    return mean, std


def effective(auc: float) -> float:
    return 0.5 + abs(auc - 0.5)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--max-steps", type=int, default=194)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--device", default="mps")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--scale-lookback", type=int, default=60)
    parser.add_argument(
        "--frontier",
        default="reports/daily_continuation_frontier_r3_20260717/summary.json",
        help="supplies the shock threshold and the bar; both were fitted before train_end",
    )
    args = evaluator_contract_defaults(parser.parse_args())

    horizons = [int(v) for v in args.horizons.split(",") if v.strip()]
    device = torch.device(args.device)
    model_dir = Path(args.model_dir)
    model, ckpt = load_model(model_dir, device)
    if "continuation_rate" not in DOWNSTREAM_AUXILIARY_TASKS:
        raise SystemExit("this tree has no continuation task; the checkpoint cannot be scored for intent 2")
    task_index = DOWNSTREAM_AUXILIARY_TASKS.index("continuation_rate")

    features, ckpt_args = build_features_from_ckpt(ckpt, args)
    namespace = argparse.Namespace(**ckpt_args)
    steps = select_steps(features, ckpt_args, args)
    stock_count = int(features.tradable_count)
    names = list(features.feature_names)
    volatility_index = names.index("volatility_20d")
    volatility = features.raw_features[:, :, volatility_index]

    frontier = json.loads((ROOT / args.frontier).read_text(encoding="utf-8"))
    threshold = float(frontier["shock_threshold"])
    bars = {
        h: frontier["untouched_test"][h]["features"]
        for h in frontier["untouched_test"]
    }

    generator = np.random.default_rng(PLACEBO_SEED)
    pooled: dict[int, dict[str, list[np.ndarray]]] = {
        h: {"label": [], "score": [], "placebo": []} for h in horizons
    }
    sessions: dict[int, int] = {h: 0 for h in horizons}

    for step in steps:
        batch = make_real_snapshot(
            features,
            step=int(step),
            full_observation=True,
            edge_window=int(ckpt_args.get("edge_window", 60)),
            top_k=int(ckpt_args.get("edge_top_k", 6)),
            min_abs_corr=float(ckpt_args.get("min_abs_corr", 0.2)),
            **graph_edge_kwargs(ckpt_args, args),
        ).to(device)
        with torch.no_grad():
            context = model.encode_temporal_context(batch)

        for horizon in horizons:
            if int(step) + horizon >= len(features.dates):
                continue
            cells = build_cells(
                close=features.close,
                returns_1d=features.returns_1d,
                volatility_20d=volatility,
                step=int(step),
                horizon=int(horizon),
                stock_count=stock_count,
                threshold=threshold,
            )
            if cells.in_scope.sum() < 5:
                continue
            mean, std = causal_rate_scale(features, int(step), int(horizon), args.scale_lookback)
            if not (np.isfinite(mean) and np.isfinite(std)):
                continue

            forward = max(1, int(rollout_steps_for_offset(namespace, int(horizon))))
            with torch.no_grad():
                z_pred = model.rollout_latent(context, steps=forward)
                head = model.predict_downstream_auxiliary(context, z_pred, rollout_steps=forward)
                standardized = head[:stock_count, task_index].detach().cpu().numpy().astype(np.float64)

            predicted_rate = standardized * std + mean
            score = continuation_ratio_score(predicted_rate, cells.observed_rate)
            order = generator.permutation(stock_count)
            placebo = continuation_ratio_score(predicted_rate[order], cells.observed_rate)

            scope = cells.in_scope & np.isfinite(score) & np.isfinite(placebo)
            if scope.sum() < 5:
                continue
            pooled[horizon]["label"].append(cells.label[scope])
            pooled[horizon]["score"].append(score[scope])
            pooled[horizon]["placebo"].append(placebo[scope])
            sessions[horizon] += 1

    results: dict[str, Any] = {}
    for horizon in horizons:
        if not pooled[horizon]["label"]:
            continue
        label = np.concatenate(pooled[horizon]["label"])
        score = np.concatenate(pooled[horizon]["score"])
        placebo = np.concatenate(pooled[horizon]["placebo"])
        if label.size < 100 or len(np.unique(label)) < 2:
            continue
        auc = development_auc(score, label, big_threshold=0.5)
        placebo_auc = development_auc(placebo, label, big_threshold=0.5)
        if auc is None:
            continue
        key = str(horizon)
        bar_features = bars.get(key, {})
        bar = max((effective(v) for v in bar_features.values()), default=None)
        results[key] = {
            "sessions": sessions[horizon],
            "scored_nodes": int(label.size),
            "base_rate": float(label.mean()),
            "auc": float(auc),
            "effective_auc": effective(float(auc)),
            "placebo_auc": float(placebo_auc) if placebo_auc is not None else None,
            "placebo_effective_auc": effective(float(placebo_auc)) if placebo_auc is not None else None,
            "frontier_effective_auc": bar,
            "beats_frontier": bool(bar is not None and effective(float(auc)) > bar),
        }

    payload = {
        "role": "research_only_daily_continuation_head",
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "test_used_for_selection": False,
        "model_dir": str(model_dir),
        "checkpoint_sha256": hashlib.sha256((model_dir / "graph_jepa_real.pt").read_bytes()).hexdigest(),
        "question": "a large daily move was observed. Does it keep going at that intensity?",
        "score": "predicted continuation_rate / observed_rate, de-standardized with a CAUSAL trailing scale",
        "scale": f"causal, {args.scale_lookback}-session trailing window over horizons finished before the decision date",
        "shock_threshold": threshold,
        "frontier": args.frontier,
        "placebo": f"predictions shuffled across stocks within a date, seed {PLACEBO_SEED}",
        "effective_auc_note": "0.5 + |auc - 0.5|; the frontier's leading features score below 0.5 because bigger shocks fade",
        "horizons": results,
    }
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"{'h':>4}{'nodes':>8}{'base':>7}{'AUC':>8}{'유효':>8}{'플라시보':>9}{'프론티어':>9}   판정")
    for horizon, cell in results.items():
        mark = "PASS" if cell["beats_frontier"] else "미달"
        placebo_shown = f"{cell['placebo_effective_auc']:.3f}" if cell["placebo_effective_auc"] else "-"
        bar_shown = f"{cell['frontier_effective_auc']:.3f}" if cell["frontier_effective_auc"] else "-"
        print(
            f"{horizon:>4}{cell['scored_nodes']:>8}{cell['base_rate']:>7.3f}"
            f"{cell['auc']:>8.3f}{cell['effective_auc']:>8.3f}{placebo_shown:>9}{bar_shown:>9}   {mark}"
        )
    print(f"\n-> {out}/summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
