"""Can a trivial feature COMBINATION eat the continuation head's margin?

WHY THIS EXISTS. The intent-2 bar the v9 contract pins is the best SINGLE
decision-time feature, scored one at a time by daily_continuation_frontier.py.
An audit on 2026-07-17 named that the strongest concrete challenge to the pass:
the head's own increment over the frontier is +0.022..+0.070 effective AUC, and
every leading feature is from the same volatility family (__shock_size__ 0.3802,
volatility_5d 0.3880, volatility_20d 0.3958, volatility_ratio_20_60 0.4082 --
all inverted, all correlated). A logistic over two or three of them could
plausibly close that gap, and nobody had measured it.

A head that cannot beat a three-feature logistic over its own inputs has not
demonstrated intent 2. That is the question this answers.

IT ALSO FIXES A PERIOD MISMATCH. The pinned bars are measured before train_end
while the model is scored after it, and the bar itself drifts +/-0.012 across
fold windows -- the same order as the smallest model margin. Here the frozen
features are ALSO scored on the model's own evaluation window, so the comparison
is like for like.

WHAT KEEPS THIS LEAK-FREE. Coefficients, the standardiser and the shock
threshold are fitted strictly on sessions before train_end. Scoring then happens
on the evaluation window, which the fit never saw. This is the same discipline
the frontier uses, extended to a model with more than one coefficient.

WHAT THIS DOES NOT DO. It does not move v9's bar. That contract is frozen and it
decided v9; changing a bar after seeing a result is the thing the gate exists to
prevent. This measures a NEW bar, reports it beside the old one, and if the
logistic wins, that is a finding to report and a bar for the NEXT contract to
pin -- not a retroactive re-judgement.

Evidence class: research. Nothing here qualifies or disqualifies a model.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from sklearn.linear_model import LogisticRegression

from scripts.evaluate_node_prediction import build_features_from_ckpt, select_steps
from scripts.evaluate_plan_timing import evaluator_contract_defaults
from stock_v2.daily_continuation import build_cells, continuation_threshold, shock_statistic
from stock_v2.prospective_recompute import development_auc

CANDIDATES = [
    "value_z20", "value_z60", "volume_z20", "volume_z60", "cs_rank_value_20d",
    "range_z20", "range_pct", "volatility_ratio_20_60", "amihud_20d",
    "volatility_5d", "volatility_20d", "downside_volatility_20d",
    "return_1d", "return_5d", "market_return_1d", "range_position_120d",
]


def effective(auc: float) -> float:
    """0.5 + |auc - 0.5|. The frontier's leading features sit BELOW 0.5 (bigger
    shocks fade), so a raw comparison would call the most informative feature the
    worst one. Applied identically to every arm here."""

    return 0.5 + abs(float(auc) - 0.5)


def collect(
    features,
    steps,
    horizon: int,
    threshold: float,
    indices: list[int],
    stock_count: int,
    volatility_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design matrix, label and the shock statistic, pooled over sessions."""

    design: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    shocks: list[np.ndarray] = []
    for step in steps:
        if int(step) + horizon >= len(features.dates):
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
        scope = cells.in_scope
        if scope.sum() < 5:
            continue
        block = np.asarray(
            features.raw_features[int(step), :stock_count, :][:, indices], dtype=np.float64
        )
        design.append(block[scope])
        labels.append(cells.label[scope])
        shocks.append(cells.shock_statistic[scope])
    if not labels:
        return np.empty((0, len(indices))), np.empty(0), np.empty(0)
    return np.concatenate(design), np.concatenate(labels), np.concatenate(shocks)


def fit_and_score(
    train_design: np.ndarray,
    train_label: np.ndarray,
    eval_design: np.ndarray,
    eval_label: np.ndarray,
    columns: list[int],
) -> float | None:
    """Logistic on `columns`, fitted on train only, scored on eval."""

    train_block = train_design[:, columns]
    eval_block = eval_design[:, columns]
    train_ok = np.isfinite(train_block).all(axis=1) & np.isfinite(train_label)
    eval_ok = np.isfinite(eval_block).all(axis=1) & np.isfinite(eval_label)
    if train_ok.sum() < 200 or eval_ok.sum() < 100:
        return None
    x_train, y_train = train_block[train_ok], train_label[train_ok]
    if len(np.unique(y_train)) < 2:
        return None

    # The standardiser is part of the model, so it is fitted on train only too.
    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0)
    std[std < 1e-12] = 1.0
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit((x_train - mean) / std, y_train)

    score = model.predict_proba((eval_block[eval_ok] - mean) / std)[:, 1]
    auc = development_auc(score, eval_label[eval_ok], big_threshold=0.5)
    return None if auc is None else effective(auc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--frontier", required=True, help="supplies the frozen shock threshold and train_end")
    parser.add_argument("--continuation", default=None, help="the model's own summary.json, to compare against")
    parser.add_argument("--output", required=True)
    # build_features_from_ckpt reads --horizons and passes it as the panel's
    # path_horizons (evaluate_node_prediction.py:512,679), so this flag decides
    # the PANEL, not just what gets scored. Narrowing it to the horizons of
    # interest built a different panel and the data contract correctly refused
    # to pair the checkpoint with it. The panel must match what the model was
    # trained on; the scoring subset is a separate choice.
    parser.add_argument("--horizons", default="1,2,3,5,10", help="panel horizons -- must match training")
    parser.add_argument("--score-horizons", default="1,2,3", help="which horizons to actually report")
    parser.add_argument("--selection-fraction", type=float, default=0.6)
    parser.add_argument("--max-subset", type=int, default=3, help="largest exhaustive feature combination")
    parser.add_argument("--max-steps", type=int, default=194)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", default=None)
    args = evaluator_contract_defaults(parser.parse_args())

    horizons = [int(v) for v in args.score_horizons.split(",") if v.strip()]
    frontier = json.loads(Path(args.frontier).read_text(encoding="utf-8"))
    threshold = float(frontier["shock_threshold"])

    # The checkpoint is read for its feature panel definition only -- universe,
    # feature names, train_end. Nothing here runs the network, so the weights are
    # never instantiated; that also lets this run on a host whose graph_jepa.py
    # has fewer auxiliary tasks than the checkpoint was trained with.
    ckpt = torch.load(Path(args.model_dir) / "graph_jepa_real.pt", map_location="cpu", weights_only=False)
    features, ckpt_args = build_features_from_ckpt(ckpt, args)
    names = list(features.feature_names)
    stock_count = int(features.tradable_count)
    volatility_index = names.index("volatility_20d")

    present = [n for n in CANDIDATES if n in names]
    indices = [names.index(n) for n in present]

    # FIT WINDOW: sessions strictly before the fold's train_end, so nothing the
    # coefficients see overlaps what they are scored on.
    train_end = pd.Timestamp(ckpt_args["train_end"])
    before = [i for i, d in enumerate(features.dates) if pd.Timestamp(d) < train_end and i >= 60]
    if len(before) < 120:
        raise SystemExit("not enough pre-train-end sessions to fit a baseline")
    fit_steps = before[: int(len(before) * args.selection_fraction)]

    # EVAL WINDOW: the model's own, chosen by the evaluator's own selector.
    eval_steps = select_steps(features, ckpt_args, args)

    model_scores: dict[str, float] = {}
    if args.continuation:
        payload = json.loads(Path(args.continuation).read_text(encoding="utf-8"))
        for horizon, cell in (payload.get("horizons") or {}).items():
            if isinstance(cell, dict) and cell.get("effective_auc") is not None:
                model_scores[str(horizon)] = float(cell["effective_auc"])

    results: dict[str, Any] = {}
    for horizon in horizons:
        train_design, train_label, _ = collect(
            features, fit_steps, horizon, threshold, indices, stock_count, volatility_index
        )
        eval_design, eval_label, eval_shock = collect(
            features, eval_steps, horizon, threshold, indices, stock_count, volatility_index
        )
        if train_label.size < 200 or eval_label.size < 100:
            continue

        row: dict[str, Any] = {
            "fit_nodes": int(train_label.size),
            "eval_nodes": int(eval_label.size),
            "eval_base_rate": float(np.nanmean(eval_label)),
        }

        # Arm 1: the frozen single features, scored on the MODEL's window. This is
        # the pinned bar's own comparison, moved onto like-for-like dates.
        singles: dict[str, float] = {}
        for name, index in zip(present, range(len(indices))):
            column = eval_design[:, index]
            ok = np.isfinite(column) & np.isfinite(eval_label)
            if ok.sum() < 100:
                continue
            auc = development_auc(column[ok], eval_label[ok], big_threshold=0.5)
            if auc is not None:
                singles[name] = effective(auc)
        shock_ok = np.isfinite(eval_shock) & np.isfinite(eval_label)
        if shock_ok.sum() >= 100:
            auc = development_auc(eval_shock[shock_ok], eval_label[shock_ok], big_threshold=0.5)
            if auc is not None:
                singles["__shock_size__"] = effective(auc)
        row["single_features_on_eval_window"] = singles
        row["best_single_on_eval_window"] = max(singles.values()) if singles else None

        # Arm 2: exhaustive small logistics. The audit's question is whether TWO
        # OR THREE correlated volatility features already carry what the head is
        # credited with, so every such combination is tried rather than a chosen one.
        subsets: dict[str, float] = {}
        for size in range(1, min(args.max_subset, len(indices)) + 1):
            best_value, best_names = None, None
            for combination in itertools.combinations(range(len(indices)), size):
                value = fit_and_score(
                    train_design, train_label, eval_design, eval_label, list(combination)
                )
                if value is not None and (best_value is None or value > best_value):
                    best_value, best_names = value, [present[i] for i in combination]
            if best_value is not None:
                subsets[str(size)] = {"effective_auc": best_value, "features": best_names}
        row["best_logistic_by_size"] = subsets

        # Arm 3: everything at once. Not the tightest bar -- it overfits -- but it
        # bounds what this input set carries under a linear read.
        everything = fit_and_score(
            train_design, train_label, eval_design, eval_label, list(range(len(indices)))
        )
        row["logistic_all_features"] = everything

        bars = [v for v in [row["best_single_on_eval_window"], everything] if v is not None]
        bars += [c["effective_auc"] for c in subsets.values()]
        row["strongest_baseline"] = max(bars) if bars else None

        model_value = model_scores.get(str(horizon))
        if model_value is not None and row["strongest_baseline"] is not None:
            row["model_effective_auc"] = model_value
            row["model_minus_strongest_baseline"] = model_value - row["strongest_baseline"]
        results[str(horizon)] = row

    payload = {
        "role": "research_only_multivariate_continuation_baseline",
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "changes_a_frozen_bar": False,
        "question": "does the continuation head beat a logistic over two or three of its own input features?",
        "why": "the v9 contract's bar is the best SINGLE feature, and its leaders are all correlated volatility statistics. The head's increment over that bar is +0.022..+0.070, which a small combination could plausibly close.",
        "leak_control": "coefficients, standardiser and shock threshold are fitted only on sessions before train_end; scoring happens on the model's evaluation window, which the fit never saw",
        "period_mismatch_fixed": "the frozen single features are scored on the model's evaluation window too, so the comparison is like for like",
        "model_dir": str(args.model_dir),
        "frontier": str(args.frontier),
        "train_end": str(train_end.date()),
        "shock_threshold": threshold,
        "fit_sessions": len(fit_steps),
        "eval_sessions": int(len(eval_steps)),
        "candidate_features": present,
        "results": results,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"train_end {train_end.date()}  적합 {len(fit_steps)}세션(이전) -> 채점 {len(eval_steps)}세션(모델 평가창)")
    print(f"후보 피처 {len(present)}개, 부분집합 크기 1~{args.max_subset} 전수\n")
    print(f"{'h':>3}{'최강 단일':>11}{'로지스틱2':>11}{'로지스틱3':>11}{'전체':>9}{'모델':>9}{'모델-최강':>11}")
    for horizon, row in results.items():
        def cell(key: str) -> str:
            value = (row.get("best_logistic_by_size") or {}).get(key)
            return f"{value['effective_auc']:.4f}" if value else "  --  "
        model_value = row.get("model_effective_auc")
        delta = row.get("model_minus_strongest_baseline")
        print(
            f"{horizon:>3}{row['best_single_on_eval_window'] or 0:11.4f}{cell('2'):>11}{cell('3'):>11}"
            f"{row['logistic_all_features'] or 0:9.4f}"
            f"{(model_value if model_value is not None else 0):9.4f}"
            f"{(delta if delta is not None else 0):+11.4f}"
        )
    print()
    for horizon, row in results.items():
        winner = (row.get("best_logistic_by_size") or {}).get("3")
        if winner:
            print(f"  h{horizon} 3피처 승자: {', '.join(winner['features'])}")
    losses = [h for h, r in results.items() if (r.get("model_minus_strongest_baseline") or 0) <= 0]
    print()
    if losses:
        print(f"  -> 모델이 h{','.join(losses)}에서 단순 조합에 진다. 의도2의 통과는 이 바 앞에서 재검토되어야 함.")
    elif results:
        print("  -> 모델이 모든 지평에서 최강 단순 조합을 이긴다. 헤드의 증분은 단일피처 바의 약함 때문이 아님.")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
