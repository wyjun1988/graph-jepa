"""FINAL operational strategy, validated end-to-end across all 5 folds, NET of costs.

Strategy = top-20% by predicted return, confidence-weighted sizing (exp(-epi_z),
MC-dropout), beta-hedged (index futures proxy = minus universe mean), h10 hold,
non-overlapping. Turnover measured on the actual weighted book; Korea costs
applied (sell tax 0.20% dominates the long leg; futures leg ~free).

Prints per-fold and aggregate GROSS and NET Sharpe -- the operational evidence.
"""

import sys, argparse, json
from pathlib import Path
SC = Path("/Users/wooyeol/work/stock-v2-candidate-v17")
sys.path.insert(0, str(SC))
import numpy as np
import torch
import torch.nn as nn
from scripts.evaluate_node_prediction import build_features_from_ckpt, graph_edge_kwargs, load_model, select_steps
from scripts.evaluate_plan_timing import evaluator_contract_defaults
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.graph_jepa import DOWNSTREAM_AUXILIARY_TASKS
from stock_v2.real_features import make_real_snapshot

p = argparse.ArgumentParser()
p.add_argument("--model-root", default="models/v16_buysell_5fold_seed17_20260718")
p.add_argument("--folds", default="r1,r2,r3,r4,r5")
p.add_argument("--horizon", type=int, default=10)
p.add_argument("--top-frac", type=float, default=0.2)
p.add_argument("--passes", type=int, default=12)
p.add_argument("--sizing", default="confidence", choices=["confidence", "equal"])
p.add_argument("--roundtrip-cost", type=float, default=0.0033, help="long-leg round-trip (sell tax+comm+slippage)")
p.add_argument("--futures-cost", type=float, default=0.0005, help="index-futures round-trip")
p.add_argument("--horizons", default="1,2,3,5,10")
p.add_argument("--device", default="mps")
p.add_argument("--max-steps", type=int, default=194)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--cache-dir", default="data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv")
p.add_argument("--output", default="")
args = evaluator_contract_defaults(p.parse_args())
H = int(args.horizon); N = int(args.passes)


def fold_series(model_dir):
    torch.manual_seed(args.seed)
    model, ckpt = load_model(Path(model_dir), torch.device(args.device))
    model.eval()
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.train()
    ns = argparse.Namespace(**dict(ckpt.get("args", {})))
    feats, ckpt_args = build_features_from_ckpt(ckpt, args)
    steps = select_steps(feats, ckpt_args, args)
    stock = int(feats.tradable_count)
    ti = {t: i for i, t in enumerate(DOWNSTREAM_AUXILIARY_TASKS)}
    ew = int(ckpt_args.get("edge_window", 60)); tk = int(ckpt_args.get("edge_top_k", 6)); mcr = float(ckpt_args.get("min_abs_corr", 0.2))
    fwd = max(1, int(rollout_steps_for_offset(ns, H)))
    gross, prev_w = [], None
    turnovers = []
    for step in list(steps)[::H]:                          # non-overlapping
        s = int(step)
        if s + H >= len(feats.dates) or s + 1 >= len(feats.dates):
            continue
        batch = make_real_snapshot(feats, step=s, full_observation=True, edge_window=ew, top_k=tk, min_abs_corr=mcr, **graph_edge_kwargs(ckpt_args, args)).to(next(model.parameters()).device)
        samp = []
        with torch.no_grad():
            for _ in range(N):
                ctx = model.encode_temporal_context(batch)
                z = model.rollout_latent(ctx, steps=fwd)
                samp.append(model.predict_downstream_auxiliary(ctx, z, rollout_steps=fwd)[:stock, ti["path_return"]].cpu().numpy())
        samp = np.stack(samp); point = samp.mean(0); epi = samp.std(0)
        entry = np.asarray(feats.open[s + 1, :stock]); exitp = np.asarray(feats.close[s + H, :stock])
        ok = np.isfinite(entry) & (entry > 0) & np.isfinite(exitp) & (exitp > 0) & np.isfinite(point)
        idx = np.where(ok)[0]
        if len(idx) < 10:
            continue
        r = exitp[idx] / entry[idx] - 1.0
        thr = np.quantile(point[idx], 1 - args.top_frac)
        selmask = point[idx] >= thr
        ez = (epi[idx] - epi[idx].mean()) / (epi[idx].std() + 1e-9)
        if args.sizing == "equal":
            w = selmask.astype(float)
        else:
            w = np.where(selmask, np.exp(-ez), 0.0)
        w = w / w.sum()
        gross.append(float((w * r).sum() - r.mean()))       # beta-hedged
        # turnover vs previous book (aligned by stock index)
        full = np.zeros(stock); full[idx] = w
        if prev_w is not None:
            turnovers.append(0.5 * np.abs(full - prev_w).sum())
        prev_w = full
    g = np.array(gross)
    to = float(np.mean(turnovers)) if turnovers else 1.0
    return g, to


ppy = 252 / H
rows = {}
for tag in args.folds.split(","):
    g, to = fold_series(f"{args.model_root}/{tag}")
    if len(g) < 3:
        continue
    cost = to * args.roundtrip_cost + args.futures_cost      # per-period cost (turnover-scaled long leg + futures)
    net = g - cost
    def sh(x): return float(x.mean() / x.std() * np.sqrt(ppy)) if x.std() > 0 else 0.0
    rows[tag] = {"gross_sharpe": round(sh(g), 2), "net_sharpe": round(sh(net), 2),
                 "turnover": round(to, 2), "cost_per_period_pct": round(cost * 100, 3),
                 "gross_mean_pct": round(float(g.mean() * 100), 3), "net_mean_pct": round(float(net.mean() * 100), 3), "n": len(g)}
    print(f"{tag}: gross Sharpe {rows[tag]['gross_sharpe']:.2f}  turnover {to:.0%}  cost {cost*100:.2f}%/기간  -> NET Sharpe {rows[tag]['net_sharpe']:.2f} (순수익 {rows[tag]['net_mean_pct']:.3f}%/기간)")

gs = np.array([v["gross_sharpe"] for v in rows.values()]); nsr = np.array([v["net_sharpe"] for v in rows.values()])
print(f"\n집계 ({len(rows)}폴드): gross 평균 {gs.mean():.2f} / NET 평균 {nsr.mean():.2f}, NET 최악 {nsr.min():.2f}, NET std {nsr.std():.2f}")
if args.output:
    Path("/Users/wooyeol/work/stock-v2", args.output).write_text(json.dumps({
        "strategy": "top-20% return, confidence-weighted (MC-dropout), beta-hedged (futures), h10, non-overlap",
        "roundtrip_cost_long": args.roundtrip_cost, "futures_cost": args.futures_cost,
        "per_fold": rows, "gross_mean": round(float(gs.mean()), 2), "net_mean": round(float(nsr.mean()), 2),
        "net_worst": round(float(nsr.min()), 2), "net_std": round(float(nsr.std()), 2)}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"-> {args.output}")
