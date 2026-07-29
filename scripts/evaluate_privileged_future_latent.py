"""Does the privileged teacher put the FUTURE hidden into the rolled-out latent?

v15's teacher encodes the complete future state (true flow(t+h) filled in), so
the student's predicted future latent z_pred(t -> t+h) should learn to carry the
hidden flow that will be disclosed at t+h+1. This measures exactly that, v15
against v14 (whose teacher never saw the future hidden).

METHOD. At each decision date t and horizon h, roll the latent forward h steps,
then read the hidden-completion head on the ROLLED-OUT latent (not the present
context). Score its per-date cross-sectional IC against the true future flow
flow(t+h) = panel row (t+h)+1. If v15 > v14, the privileged teacher pushed the
future hidden into the prediction -- the contract's rule 2.

Leak-free: the rollout starts from z_context(t), built only from row t. The
target flow(t+h) is read only as a label. No future price or flow enters the
student's input.

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
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.graph_jepa import HIDDEN_COMPLETION_CHANNELS
from stock_v2.real_features import make_real_snapshot


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
        raise SystemExit("this checkpoint has no hidden_completion_head")

    features, ckpt_args = build_features_from_ckpt(ckpt, args)
    namespace = argparse.Namespace(**ckpt_args)
    names = list(features.feature_names)
    channels = [names.index(n) for n in HIDDEN_COMPLETION_CHANNELS if n in names]
    stock_count = int(features.tradable_count)
    steps = select_steps(features, ckpt_args, args)
    horizons = [int(v) for v in args.horizons.split(",") if v.strip()]
    norm = features.features
    n_dates = norm.shape[0]

    edge_window = int(ckpt_args.get("edge_window", 60))
    edge_top_k = int(ckpt_args.get("edge_top_k", 6))
    min_abs_corr = float(ckpt_args.get("min_abs_corr", 0.2))

    acc: dict[tuple, list] = {}
    results: dict[str, Any] = {}

    for horizon in horizons:
        forward = max(1, int(rollout_steps_for_offset(namespace, int(horizon))))
        preds = {ci: [] for ci in range(len(channels))}
        truths = {ci: [] for ci in range(len(channels))}
        dates: list = []
        for step in steps:
            future_row = int(step) + int(horizon) + 1  # flow(t+h) disclosed at (t+h)+1
            if future_row >= n_dates:
                continue
            batch = make_real_snapshot(
                features, step=int(step), full_observation=True,
                edge_window=edge_window, top_k=edge_top_k, min_abs_corr=min_abs_corr,
                **graph_edge_kwargs(ckpt_args, args),
            ).to(device)
            with torch.no_grad():
                context = model.encode_temporal_context(batch)
                z_pred = model.rollout_latent(context, steps=forward)
                head = model.hidden_completion_head(z_pred)[:stock_count].cpu().numpy()
            day = pd.Timestamp(features.dates[int(step)])
            for ci, feat_idx in enumerate(channels):
                preds[ci].append(head[:, ci])
                truths[ci].append(np.asarray(norm[future_row, :stock_count, feat_idx], dtype=np.float64))
            dates.append(np.full(stock_count, day, dtype=object))
        if not dates:
            continue
        dflat = np.concatenate(dates).tolist()
        for ci, name in enumerate(HIDDEN_COMPLETION_CHANNELS[: len(channels)]):
            ic = per_date_ic(np.concatenate(preds[ci]), np.concatenate(truths[ci]), dflat)
            results[f"{name}@h{horizon}"] = ic

    payload = {
        "role": "research_only_privileged_future_latent_eval",
        "live_orders_allowed": False,
        "promotion_eligible": False,
        "question": "does the rolled-out future latent encode the FUTURE hidden flow (flow(t+h))?",
        "note": "compare to the v14 run of this same script; v15 > v14 means the privileged teacher pushed the hidden into future prediction (contract rule 2)",
        "model_dir": str(args.model_dir),
        "eval_sessions": int(len(steps)),
        "results": results,
    }
    out = ROOT / args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"평가 {len(steps)}세션  롤아웃 latent -> 미래 flow IC\n")
    print(f"{'채널@지평':40}{'IC':>10}")
    for k, v in results.items():
        print(f"  {k:38}{v:10.4f}")
    print(f"\n  (v14 대비 상승이면 특권 teacher가 미래 예측에 히든을 심음)")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
