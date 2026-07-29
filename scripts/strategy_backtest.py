"""Strategy exploration: use the model's STRONG risk heads, not the weak return head.

Head IC (v16 h10): volatility 0.53, MAE 0.36, continuation 0.30, MFE 0.16,
path_return 0.06. The model predicts RISK 6-9x better than RETURN. So this tests
whether risk-aware portfolios beat a return-ranked baseline on realized outcomes,
especially risk-adjusted (Sharpe, drawdown).

Leak-free: predictions come from the model at t (rollout + heads on row t's
context); realized return is close(t+h)/open(t+1)-1 from actual prices. Ranking
uses the standardized head outputs (cross-sectional order is preserved).

Evidence class: research. No promotion, returns never gate.
"""

import sys, json
from pathlib import Path

SC = Path("/Users/wooyeol/work/stock-v2-candidate-v17")
sys.path.insert(0, str(SC))

import argparse
import numpy as np
import pandas as pd
import torch

from scripts.evaluate_node_prediction import build_features_from_ckpt, graph_edge_kwargs, load_model, select_steps
from scripts.evaluate_plan_timing import evaluator_contract_defaults
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.graph_jepa import DOWNSTREAM_AUXILIARY_TASKS
from stock_v2.real_features import make_real_snapshot

p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--output", required=True)
p.add_argument("--horizon", type=int, default=10)
p.add_argument("--top-frac", type=float, default=0.2, help="long the top fraction by score")
p.add_argument("--horizons", default="1,2,3,5,10")
p.add_argument("--device", default="mps")
p.add_argument("--max-steps", type=int, default=194)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--cache-dir", default="data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv")
p.add_argument("--pred-cache", default="", help="parquet of per-(date,stock) preds; load if present, else write")
args = evaluator_contract_defaults(p.parse_args())

H = int(args.horizon)
cache = Path(args.pred_cache) if args.pred_cache else None
if cache and cache.is_file():
    df = pd.read_parquet(cache)
    print(f"predictions from cache: {cache} ({len(df)} rows)")
else:
    model, ckpt = load_model(Path(args.model_dir), torch.device(args.device))
    model.eval()
    ns = argparse.Namespace(**dict(ckpt.get("args", {})))
    feats, ckpt_args = build_features_from_ckpt(ckpt, args)
    steps = select_steps(feats, ckpt_args, args)
    stock = int(feats.tradable_count)
    ti = {t: i for i, t in enumerate(DOWNSTREAM_AUXILIARY_TASKS)}
    ew = int(ckpt_args.get("edge_window", 60)); tk = int(ckpt_args.get("edge_top_k", 6)); mc = float(ckpt_args.get("min_abs_corr", 0.2))
    fwd = max(1, int(rollout_steps_for_offset(ns, H)))
    # per (date, stock): predicted heads + realized forward return
    rows = []
    for step in steps:
        if int(step) + H >= len(feats.dates) or int(step) + 1 >= len(feats.dates):
            continue
        batch = make_real_snapshot(feats, step=int(step), full_observation=True, edge_window=ew, top_k=tk, min_abs_corr=mc, **graph_edge_kwargs(ckpt_args, args)).to(next(model.parameters()).device)
        with torch.no_grad():
            ctx = model.encode_temporal_context(batch)
            z = model.rollout_latent(ctx, steps=fwd)
            head = model.predict_downstream_auxiliary(ctx, z, rollout_steps=fwd)[:stock].cpu().numpy()
        entry = np.asarray(feats.open[int(step) + 1, :stock], dtype=np.float64)
        exitp = np.asarray(feats.close[int(step) + H, :stock], dtype=np.float64)
        ok = np.isfinite(entry) & (entry > 0) & np.isfinite(exitp) & (exitp > 0)
        realized = np.full(stock, np.nan); realized[ok] = exitp[ok] / entry[ok] - 1.0
        day = pd.Timestamp(feats.dates[int(step)])
        for s in range(stock):
            if not ok[s]:
                continue
            rows.append({
                "date": day, "ret": realized[s],
                "pred_ret": head[s, ti["path_return"]],
                "pred_mfe": head[s, ti["max_favorable_excursion"]],
                "pred_mae": head[s, ti["max_adverse_excursion"]],
                "pred_vol": head[s, ti["realized_volatility"]],
            })
    df = pd.DataFrame(rows).dropna()
    if cache:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache)
        print(f"predictions cached -> {cache} ({len(df)} rows)")

def portfolio(score_col, name, avoid_high=None):
    """Daily: long top `top_frac` by score_col; if avoid_high, drop names in the
    top `top_frac` of avoid_high first (risk cut). Equal-weight, forward return."""
    daily = []
    for _, g in df.groupby("date"):
        g = g.copy()
        if avoid_high is not None:
            keep = g[avoid_high] <= g[avoid_high].quantile(1 - args.top_frac)
            g = g[keep]
        if len(g) < 5:
            continue
        thr = g[score_col].quantile(1 - args.top_frac)
        picks = g[g[score_col] >= thr]
        daily.append(picks["ret"].mean())
    d = np.array(daily)
    if len(d) < 5:
        return None
    ann = np.sqrt(252 / H)  # h-day holding, non-overlapping approx
    sharpe = (d.mean() / d.std() * ann) if d.std() > 0 else 0.0
    cum = np.cumprod(1 + d)
    dd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return {"name": name, "n_days": len(d), "mean_per_period": float(d.mean()), "std": float(d.std()),
            "sharpe_ann": float(sharpe), "max_drawdown": dd, "hit_rate": float((d > 0).mean())}

# universe baseline (equal-weight all, per date)
uni = df.groupby("date")["ret"].mean().values
ann = np.sqrt(252 / H)
strategies = [
    {"name": "0_universe_EW", "n_days": len(uni), "mean_per_period": float(uni.mean()), "std": float(uni.std()),
     "sharpe_ann": float(uni.mean()/uni.std()*ann) if uni.std()>0 else 0, "max_drawdown": float((np.cumprod(1+uni)/np.maximum.accumulate(np.cumprod(1+uni))-1).min()), "hit_rate": float((uni>0).mean())},
]
for col, name, avoid in [
    ("pred_ret", "1_return_only (weak head)", None),
    ("pred_mfe", "2_upside_MFE", None),
    ("pred_ret", "3_return_cut_highMAE", "pred_mae"),
    ("pred_ret", "4_return_cut_highVOL", "pred_vol"),
]:
    r = portfolio(col, name, avoid_high=avoid)
    if r: strategies.append(r)

# 5: risk-adjusted asymmetry (MFE - MAE), and 6: return/vol
df["asym"] = df["pred_mfe"] - df["pred_mae"]
df["ret_over_vol"] = df["pred_ret"] - df["pred_vol"]  # standardized proxy for return per risk
for col, name in [("asym", "5_asymmetry_MFEminusMAE"), ("ret_over_vol", "6_return_minus_vol")]:
    r = portfolio(col, name);
    if r: strategies.append(r)


def _stats(d, name):
    d = np.asarray(d, dtype=np.float64)
    if len(d) < 5:
        return None
    ann = np.sqrt(252 / H)
    sharpe = (d.mean() / d.std() * ann) if d.std() > 0 else 0.0
    cum = np.cumprod(1 + d)
    dd = float((cum / np.maximum.accumulate(cum) - 1).min())
    return {"name": name, "n_days": len(d), "mean_per_period": float(d.mean()), "std": float(d.std()),
            "sharpe_ann": float(sharpe), "max_drawdown": dd, "hit_rate": float((d > 0).mean())}


def long_short(long_col, short_col, name, long_cut=None):
    """Market-neutral: long top `top_frac` by long_col (optionally cutting names
    in the top `top_frac` of long_cut first), short top `top_frac` by short_col.
    Per-period spread = mean(long ret) - mean(short ret). Removes market beta."""
    daily = []
    for _, g in df.groupby("date"):
        gl = g
        if long_cut is not None:
            gl = g[g[long_cut] <= g[long_cut].quantile(1 - args.top_frac)]
        if len(gl) < 5 or len(g) < 5:
            continue
        lo = gl[gl[long_col] >= gl[long_col].quantile(1 - args.top_frac)]["ret"].mean()
        sh = g[g[short_col] >= g[short_col].quantile(1 - args.top_frac)]["ret"].mean()
        daily.append(lo - sh)
    return _stats(daily, name)


ls = []
for lc, sc, name, cut in [
    ("pred_ret", "pred_ret_neg", "LS1_return (long top, short bottom)", None),
    ("pred_ret", "pred_mae", "LS2_long_return_cutMAE / short_highMAE", "pred_mae"),
    ("pred_ret", "pred_vol", "LS3_long_return_cutVOL / short_highVOL", "pred_vol"),
    ("asym", "pred_mae", "LS4_long_asym / short_highMAE", None),
]:
    if sc == "pred_ret_neg":
        df["pred_ret_neg"] = -df["pred_ret"]
    r = long_short(lc, sc, name, long_cut=cut)
    if r: ls.append(r)

payload = {"role": "research_strategy_backtest", "live_orders_allowed": False, "promotion_eligible": False,
           "model_dir": str(args.model_dir), "horizon": H, "top_frac": args.top_frac,
           "note": "leak-free: preds at t, realized close(t+H)/open(t+1)-1. Ranking on standardized heads.",
           "long_only": strategies, "long_short_market_neutral": ls}
out = Path("/Users/wooyeol/work/stock-v2") / args.output
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def show(title, items):
    print(f"\n{title}")
    print(f"{'전략':40}{'기간수익%':>10}{'Sharpe(연)':>11}{'MDD%':>9}{'적중':>7}")
    for s in items:
        print(f"  {s['name']:38}{s['mean_per_period']*100:10.3f}{s['sharpe_ann']:11.2f}{s['max_drawdown']*100:9.1f}{s['hit_rate']*100:7.0f}")

print(f"전략 백테스트 (h{H}, 상위 {args.top_frac:.0%}, {len(df.groupby('date'))} 세션)")
show("[롱-only — 시장 베타 포함]", strategies)
show("[롱숏 마켓뉴트럴 — 베타 제거]", ls)
print(f"\n-> {out}")
