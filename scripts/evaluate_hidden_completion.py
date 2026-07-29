"""Does the world model's state carry the PRESENT hidden flow? (v14, rule 2 & 3)

The v14 head reads current_context -- the graph-on completion representation --
and estimates t's investor flow, which is disclosed only at t+1. This scores it
the way intent 3 scores its heads: per-date cross-sectional IC against the true
next-disclosed flow, on the model's own evaluation window.

Two questions, from the frozen contract:
  rule 2  the head's IC beats the observable-feature ridge frontier (0.346 for
          foreign flow). Clearing it means the JEPA state carries hidden flow a
          linear read of the same observables does not extract.
  rule 3  severing the graph (neighbour_scale 0) drops the IC. The user's claim
          is that neighbours -- stocks moving like the names foreigners are
          buying -- carry the hidden. If severing does not drop it, the head is
          using own-node dynamics, not the graph.

Leak-free: current_context at t is built only from row t (which holds t-1's
disclosed flow); the target is the true flow at t, read from panel row t+1. The
persistence signal (yesterday's flow) is in the input on purpose -- beating the
frontier, which already includes it, is the point.

Evidence class: research. Nothing here qualifies a model.
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

from scripts.evaluate_node_prediction import (
    build_features_from_ckpt,
    graph_edge_kwargs,
    load_model,
    select_steps,
)
from scripts.evaluate_plan_timing import evaluator_contract_defaults
from stock_v2.graph_jepa import HIDDEN_COMPLETION_CHANNELS
from stock_v2.real_features import make_real_snapshot

PLACEBO_SEED = 20260717


def per_date_ic(pred: np.ndarray, truth: np.ndarray, dates: list) -> float:
    frame = pd.DataFrame({"p": pred, "y": truth, "d": dates}).dropna()
    ics = []
    for _, g in frame.groupby("d"):
        if len(g) >= 5 and g["p"].std() > 1e-9 and g["y"].std() > 1e-9:
            ics.append(np.corrcoef(g["p"], g["y"])[0, 1])
    return float(np.nanmean(ics)) if ics else float("nan")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frontier-bar", type=float, default=0.346,
                        help="the observable-feature ridge IC for foreign flow (rule 2 bar)")
    parser.add_argument("--horizons", default="1,2,3,5,10")
    parser.add_argument("--device", default="mps")
    parser.add_argument("--max-steps", type=int, default=194)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--cache-dir", default=None)
    args = evaluator_contract_defaults(parser.parse_args())

    model, ckpt = load_model(Path(args.model_dir), torch.device(args.device))
    device = next(model.parameters()).device
    model.eval()
    if getattr(model, "hidden_completion_head", None) is None:
        raise SystemExit("this checkpoint has no hidden_completion_head (not a v14/v15 model)")

    features, ckpt_args = build_features_from_ckpt(ckpt, args)
    names = list(features.feature_names)
    channels = [names.index(n) for n in HIDDEN_COMPLETION_CHANNELS if n in names]
    stock_count = int(features.tradable_count)
    steps = select_steps(features, ckpt_args, args)
    norm = features.features
    n_dates = norm.shape[0]

    edge_window = int(ckpt_args.get("edge_window", 60))
    edge_top_k = int(ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(ckpt_args.get("min_abs_corr", 0.2))

    # accumulate predictions (intact and severed) and the true next-disclosed flow
    collected: dict[str, list] = {
        f"{arm}_{ci}": [] for arm in ("intact", "severed") for ci in range(len(channels))
    }
    truth_by: dict[int, list] = {ci: [] for ci in range(len(channels))}
    date_by: list = []

    for step in steps:
        nxt = int(step) + 1
        if nxt >= n_dates:
            continue
        batch = make_real_snapshot(
            features, step=int(step), full_observation=True,
            edge_window=edge_window, top_k=edge_top_k, min_abs_corr=min_abs_corr,
            **graph_edge_kwargs(ckpt_args, args),
        ).to(device)
        with torch.no_grad():
            # graph-on completion rep (the imputation path uses exactly this)
            ctx_intact = model.encode_context(batch, neighbor_scale=None)
            ctx_severed = model.encode_context(batch, neighbor_scale=0.0)
            pred_intact = model.hidden_completion_head(ctx_intact)[:stock_count].cpu().numpy()
            pred_severed = model.hidden_completion_head(ctx_severed)[:stock_count].cpu().numpy()

        day = pd.Timestamp(features.dates[int(step)])
        for ci, feat_idx in enumerate(channels):
            true_flow = np.asarray(norm[nxt, :stock_count, feat_idx], dtype=np.float64)
            collected[f"intact_{ci}"].append(pred_intact[:, ci])
            collected[f"severed_{ci}"].append(pred_severed[:, ci])
            truth_by[ci].append(true_flow)
        date_by.append(np.full(stock_count, day, dtype=object))

    dates_flat = np.concatenate(date_by).tolist()
    results: dict[str, Any] = {}
    for ci, name in enumerate(HIDDEN_COMPLETION_CHANNELS[: len(channels)]):
        truth = np.concatenate(truth_by[ci])
        intact = np.concatenate(collected[f"intact_{ci}"])
        severed = np.concatenate(collected[f"severed_{ci}"])
        ic_intact = per_date_ic(intact, truth, dates_flat)
        ic_severed = per_date_ic(severed, truth, dates_flat)
        results[name] = {
            "ic_intact": ic_intact,
            "ic_severed": ic_severed,
            "graph_contribution": ic_intact - ic_severed,
            "beats_frontier": bool(np.isfinite(ic_intact) and ic_intact > args.frontier_bar),
        }

    payload = {
        "role": "research_only_hidden_completion_eval",
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "question": "does the JEPA state carry the present hidden flow, and is the graph the source?",
        "frontier_bar": args.frontier_bar,
        "model_dir": str(args.model_dir),
        "eval_sessions": int(len(steps)),
        "results": results,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"평가 {len(steps)}세션\n")
    print(f"{'채널':38}{'IC(intact)':>12}{'IC(severed)':>13}{'그래프기여':>11}{'프론티어돌파':>13}")
    for name, r in results.items():
        print(f"  {name:36}{r['ic_intact']:12.4f}{r['ic_severed']:13.4f}"
              f"{r['graph_contribution']:+11.4f}{'예' if r['beats_frontier'] else '아니오':>13}")
    print()
    foreign = results.get("investor_foreign_flow_ratio_1d", {})
    if foreign.get("beats_frontier"):
        print(f"  rule 2 통과: 상태가 프론티어({args.frontier_bar})를 넘는 히든 수급을 담음")
        if foreign.get("graph_contribution", 0) > 0.02:
            print(f"  rule 3 통과: 엣지 절단 시 {foreign['graph_contribution']:.4f} 하락 -> 그래프가 히든의 원천")
        else:
            print("  rule 3 미통과: 엣지 절단해도 IC 유지 -> 그래프가 아니라 자기노드 동학 사용")
    else:
        print("  rule 2 미통과: 상태가 선형 프론티어를 못 넘음 -- 수급이 곧 당일 가격일 가능성")
    print(f"\n-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
