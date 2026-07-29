"""What bar must a model clear on daily shock continuation? Establish it first.

The intraday study asked this and got a decisive answer: no feature separates
developing shocks from fading ones by much. The best single decision-time
feature, cumulative_value_shock_20, reached AUC 0.6396 on its selection period
and 0.6169 on an untouched test period, while the shock's own SIZE scored 0.423
and 0.395 -- below 0.5, meaning big shocks predict NON-continuation.

The daily question needs its own bar, because the intraday numbers do not
transfer: different sampling, different horizons, different base rate. Without
one, a continuation head's AUC would be uninterpretable -- 0.55 could be
excellent or worthless.

This scores decision-time FEATURES as predictors, exactly as the intraday
frontier study did. If nothing separates, the question is unanswerable from this
input set and no head will rescue it. If something does, its AUC is the bar.

Split discipline: the shock threshold and the feature ranking are fitted on the
selection window only; the test window is scored once and never selected on.
Both windows end before the fold's evaluation period, so this reads no evaluation
data at all.

Evidence class: retrospective selection. Nothing here qualifies a model.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.evaluate_node_prediction import build_features_from_ckpt, load_model
from scripts.evaluate_plan_timing import evaluator_contract_defaults, moving_block_bootstrap
from stock_v2.daily_continuation import (
    build_cells,
    continuation_ratio_score,
    continuation_threshold,
    shock_statistic,
)
from stock_v2.prospective_recompute import development_auc


def score_window(
    features,
    dates: range,
    horizons: list[int],
    threshold: float,
    candidates: dict[str, int],
    stock_count: int,
    volatility_index: int,
) -> dict[str, Any]:
    """AUC per (feature, horizon) pooled over the window, plus the base rate."""

    per_cell: dict[str, dict[str, Any]] = {}
    for horizon in horizons:
        pooled_label: list[np.ndarray] = []
        pooled_scores: dict[str, list[np.ndarray]] = {name: [] for name in candidates}
        pooled_scores["__shock_size__"] = []
        daily_counts = []
        for step in dates:
            if step + horizon >= len(features.dates):
                continue
            cells = build_cells(
                close=features.close,
                returns_1d=features.returns_1d,
                volatility_20d=features.raw_features[:, :, volatility_index],
                step=int(step),
                horizon=int(horizon),
                stock_count=stock_count,
                threshold=threshold,
            )
            if cells.in_scope.sum() < 5:
                continue
            scope = cells.in_scope
            pooled_label.append(cells.label[scope])
            daily_counts.append(int(scope.sum()))
            for name, index in candidates.items():
                raw = features.raw_features[int(step), :stock_count, index]
                pooled_scores[name].append(np.asarray(raw, dtype=np.float64)[scope])
            # The shock's own size, as its own candidate. The intraday study
            # found this scores BELOW 0.5; if the daily version does too, it is
            # the same economics and worth knowing before any model is built.
            pooled_scores["__shock_size__"].append(cells.shock_statistic[scope])

        if not pooled_label:
            continue
        label = np.concatenate(pooled_label)
        row: dict[str, Any] = {
            "scored_nodes": int(label.size),
            "sessions": len(daily_counts),
            "base_rate": float(np.nanmean(label)),
            "features": {},
        }
        for name, chunks in pooled_scores.items():
            score = np.concatenate(chunks)
            finite = np.isfinite(score) & np.isfinite(label)
            if finite.sum() < 100:
                continue
            auc = development_auc(score[finite], label[finite], big_threshold=0.5)
            if auc is None:
                continue
            row["features"][name] = float(auc)
        per_cell[str(horizon)] = row
    return per_cell


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True, help="only to rebuild the same feature panel")
    parser.add_argument("--output", required=True)
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--selection-fraction", type=float, default=0.6)
    parser.add_argument("--cache-dir", default=None)
    args = evaluator_contract_defaults(parser.parse_args())

    horizons = [int(v) for v in args.horizons.split(",") if v.strip()]
    _, ckpt = load_model(Path(args.model_dir), torch.device("cpu"))
    features, ckpt_args = build_features_from_ckpt(ckpt, args)
    stock_count = int(features.tradable_count)
    names = list(features.feature_names)
    if "volatility_20d" not in names:
        raise SystemExit("the shock statistic needs volatility_20d in the panel")
    volatility_index = names.index("volatility_20d")

    # Only dates strictly before the fold's training end are used, so neither
    # window can see the evaluation regime.
    train_end = pd.Timestamp(ckpt_args["train_end"])
    usable = [i for i, d in enumerate(features.dates) if pd.Timestamp(d) < train_end]
    usable = [i for i in usable if i >= 60]  # volatility_20d needs history
    if len(usable) < 200:
        raise SystemExit("not enough pre-train-end sessions to fit a frontier")
    cut = int(len(usable) * args.selection_fraction)
    selection, test = usable[:cut], usable[cut:]

    # The threshold is the 80th percentile of the shock statistic on the
    # SELECTION window only. Fitting it on everything would let the test window
    # inform which nodes are in scope.
    stats = np.concatenate(
        [
            shock_statistic(features.returns_1d, features.raw_features[:, :, volatility_index], s, stock_count)
            for s in selection
        ]
    )
    threshold = continuation_threshold(stats)

    # Decision-time candidates: what a person could look at on the shock day.
    # The intraday finding was that VALUE and VOLUME shocks predict continuation
    # while the move's own size predicts the opposite, so those go in first.
    wanted = [
        "value_z20", "value_z60", "volume_z20", "volume_z60", "cs_rank_value_20d",
        "range_z20", "range_pct", "volatility_ratio_20_60", "amihud_20d",
        "volatility_5d", "volatility_20d", "downside_volatility_20d",
        "return_1d", "return_5d", "market_return_1d", "range_position_120d",
    ]
    candidates = {n: names.index(n) for n in wanted if n in names}

    selection_result = score_window(
        features, range(min(selection), max(selection) + 1), horizons, threshold, candidates, stock_count, volatility_index
    )
    test_result = score_window(
        features, range(min(test), max(test) + 1), horizons, threshold, candidates, stock_count, volatility_index
    )

    # Winner chosen on the SELECTION window only, then read once on test.
    ranked: dict[str, float] = {}
    for name in list(candidates) + ["__shock_size__"]:
        aucs = [
            abs(cell["features"][name] - 0.5)
            for cell in selection_result.values()
            if name in cell["features"]
        ]
        if aucs:
            ranked[name] = float(np.mean(aucs))
    winner = max(ranked, key=lambda k: ranked[k]) if ranked else None

    payload = {
        "role": "research_only_daily_continuation_frontier",
        "evidence_class": "retrospective_selection",
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "question": "a large DAILY move has been observed on a node. Does it keep going at that intensity, or fade?",
        "label": "future_rate >= observed_rate; observed_rate=|return_1d(t)|, future_rate=|close(t+h)/close(t)-1|/h",
        "derived_from": "configs/post_impact_continuation_gate_v1.json -- the same rate-matched, own-baseline construction, which that contract states is scale-free",
        "model_dir": str(args.model_dir),
        "train_end": str(train_end.date()),
        "shock_threshold": threshold,
        "shock_threshold_rule": f"{80.0}th percentile of |return_1d|/volatility_20d on the selection window only",
        "selection_sessions": [str(pd.Timestamp(features.dates[selection[0]]).date()), str(pd.Timestamp(features.dates[selection[-1]]).date())],
        "test_sessions": [str(pd.Timestamp(features.dates[test[0]]).date()), str(pd.Timestamp(features.dates[test[-1]]).date())],
        "selection": selection_result,
        "untouched_test": test_result,
        "winner_on_selection": winner,
        "winner_mean_abs_auc_minus_half": ranked.get(winner) if winner else None,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"shock threshold (|return_1d|/vol20, 80th pct on selection): {threshold:.4f}")
    print(f"selection {payload['selection_sessions']}   test {payload['test_sessions']}\n")
    for label, result in (("선택기간", selection_result), ("미접촉 테스트", test_result)):
        print(f"=== {label} ===")
        for horizon, cell in result.items():
            best = sorted(cell["features"].items(), key=lambda kv: -abs(kv[1] - 0.5))[:4]
            shown = "  ".join(f"{n}={v:.3f}" for n, v in best)
            print(f"  h{horizon:<3} n={cell['scored_nodes']:<6} base={cell['base_rate']:.3f}  {shown}")
        print()
    print(f"선택기간 승자: {winner}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
