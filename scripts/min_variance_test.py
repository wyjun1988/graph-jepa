"""Fair test of the user's theory idea: does MIN-VARIANCE sizing, using the world
model's covariance, beat equal-weight on past data?

Selection is held fixed (top-20% by predicted return). Only SIZING varies:
  1. equal-weight (current champion)
  2. inverse trailing-variance (diagonal min-var, sample)
  3. min-var, full sample covariance (trailing 60d, shrunk) -- classic Markowitz
  4. min-var, MODEL-vol diagonal + sample correlation -- model's forward risk view
  5. min-var, MODEL-GRAPH correlation + model-vol diagonal -- the world-model's own
     relational structure (the unique asset)

All beta-hedged (minus universe mean), non-overlapping, OOS. If a covariance-based
method significantly beats equal-weight, the theory + world model earns its place.

Research only. Returns diagnostic.
"""

import sys, argparse
from pathlib import Path
SC = Path("/Users/wooyeol/work/stock-v2-candidate-v17")
sys.path.insert(0, str(SC))
import numpy as np
import torch
from scripts.evaluate_node_prediction import build_features_from_ckpt, graph_edge_kwargs, load_model, select_steps
from scripts.evaluate_plan_timing import evaluator_contract_defaults
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.graph_jepa import DOWNSTREAM_AUXILIARY_TASKS
from stock_v2.real_features import make_real_snapshot

p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--horizon", type=int, default=10)
p.add_argument("--top-frac", type=float, default=0.2)
p.add_argument("--lookback", type=int, default=60, help="trailing sessions for sample covariance")
p.add_argument("--shrink", type=float, default=0.3, help="shrink sample cov toward its diagonal")
p.add_argument("--horizons", default="1,2,3,5,10")
p.add_argument("--device", default="mps")
p.add_argument("--max-steps", type=int, default=194)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--cache-dir", default="data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv")
args = evaluator_contract_defaults(p.parse_args())
model, ckpt = load_model(Path(args.model_dir), torch.device(args.device))
model.eval()
ns = argparse.Namespace(**dict(ckpt.get("args", {})))
feats, ckpt_args = build_features_from_ckpt(ckpt, args)
steps = select_steps(feats, ckpt_args, args)
stock = int(feats.tradable_count); H = int(args.horizon); LB = int(args.lookback)
ti = {t: i for i, t in enumerate(DOWNSTREAM_AUXILIARY_TASKS)}
ew = int(ckpt_args.get("edge_window", 60)); tk = int(ckpt_args.get("edge_top_k", 6)); mc = float(ckpt_args.get("min_abs_corr", 0.2))
fwd = max(1, int(rollout_steps_for_offset(ns, H)))
close = np.asarray(feats.close, dtype=np.float64)     # [T, N]
logret = np.diff(np.log(np.clip(close, 1e-9, None)), axis=0)  # [T-1, N]


def minvar_weights(Sigma):
    """Analytic min-var, then clip to long-only and renormalize."""
    n = Sigma.shape[0]
    Sigma = Sigma + np.eye(n) * 1e-6
    try:
        inv1 = np.linalg.solve(Sigma, np.ones(n))
    except np.linalg.LinAlgError:
        return np.ones(n) / n
    w = inv1 / inv1.sum()
    w = np.clip(w, 0, None)
    return w / w.sum() if w.sum() > 0 else np.ones(n) / n


methods = {k: [] for k in ["equal", "inv_var_sample", "minvar_sample", "minvar_modelvol", "minvar_graph"]}
for step in steps:
    s = int(step)
    if s + H >= len(feats.dates) or s + 1 >= len(feats.dates) or s - LB < 1:
        continue
    batch = make_real_snapshot(feats, step=s, full_observation=True, edge_window=ew, top_k=tk, min_abs_corr=mc, **graph_edge_kwargs(ckpt_args, args)).to(next(model.parameters()).device)
    with torch.no_grad():
        ctx = model.encode_temporal_context(batch)
        z = model.rollout_latent(ctx, steps=fwd)
        head = model.predict_downstream_auxiliary(ctx, z, rollout_steps=fwd)[:stock].cpu().numpy()
    ei = batch.edge_index.cpu().numpy() if hasattr(batch, "edge_index") and batch.edge_index is not None else None
    ew_w = batch.edge_weight.cpu().numpy() if hasattr(batch, "edge_weight") and batch.edge_weight is not None else None
    entry = np.asarray(feats.open[s + 1, :stock]); exitp = np.asarray(feats.close[s + H, :stock])
    ok = np.isfinite(entry) & (entry > 0) & np.isfinite(exitp) & (exitp > 0) & np.isfinite(head[:, ti["path_return"]])
    fwd_ret = np.full(stock, np.nan); fwd_ret[ok] = exitp[ok] / entry[ok] - 1.0
    idx = np.where(ok)[0]
    pr = head[idx, ti["path_return"]]
    k = max(5, int(len(idx) * args.top_frac))
    sel = idx[np.argsort(pr)[-k:]]                 # selected stock indices (top-frac by pred return)
    uni_mean = np.nanmean(fwd_ret[idx])
    R = logret[s - LB:s, sel]                        # [LB, k] trailing returns of selected
    good = np.isfinite(R).all(axis=0)                # keep only stocks with complete trailing history
    sel = sel[good]; R = R[:, good]
    k = len(sel)
    if k < 5 or R.shape[0] < 10:
        continue
    Sig = np.cov(R, rowvar=False)                    # sample covariance
    d = np.diag(np.diag(Sig))
    Sig_shrunk = (1 - args.shrink) * Sig + args.shrink * d
    sample_vol = np.sqrt(np.clip(np.diag(Sig), 1e-12, None))
    corr = Sig / np.outer(sample_vol, sample_vol); np.fill_diagonal(corr, 1.0)
    # model vol for selected: standardized head -> positive level via exp, scaled to sample-vol median
    mv_z = head[sel, ti["realized_volatility"]]
    model_vol = np.exp(0.5 * (mv_z - mv_z.mean())) * np.median(sample_vol)
    Sig_modelvol = np.outer(model_vol, model_vol) * corr
    Sig_modelvol = (1 - args.shrink) * Sig_modelvol + args.shrink * np.diag(np.diag(Sig_modelvol))
    # graph correlation among selected (from model's edges), diagonal = model vol
    Rg = np.eye(k)
    if ei is not None and ew_w is not None:
        pos = {int(v): i for i, v in enumerate(sel)}
        for e in range(ei.shape[1]):
            a, b = int(ei[0, e]), int(ei[1, e])
            if a in pos and b in pos:
                Rg[pos[a], pos[b]] = ew_w[e]; Rg[pos[b], pos[a]] = ew_w[e]
    Rg = np.clip(Rg, -0.99, 0.99); np.fill_diagonal(Rg, 1.0)
    Sig_graph = np.outer(model_vol, model_vol) * Rg
    Sig_graph = (1 - args.shrink) * Sig_graph + args.shrink * np.diag(np.diag(Sig_graph))
    r = fwd_ret[sel]
    for name, w in [("equal", np.ones(k) / k), ("inv_var_sample", (1 / np.diag(Sig)) / (1 / np.diag(Sig)).sum()),
                    ("minvar_sample", minvar_weights(Sig_shrunk)), ("minvar_modelvol", minvar_weights(Sig_modelvol)),
                    ("minvar_graph", minvar_weights(Sig_graph))]:
        methods[name].append(float((w * r).sum() - uni_mean))

ppy = 252 / H
print(f"최소분산 사이징 비교 (r5, 상위 {args.top_frac:.0%} 선택 고정, 베타헤지, 비중첩)\n")
print(f"{'사이징 방법':34}{'Sharpe':>8}{'MDD%':>8}{'n':>5}")
for name in methods:
    a = np.array(methods[name])[::H]
    if len(a) < 3 or a.std() == 0:
        print(f"  {name:32}{'n/a':>8}"); continue
    cum = np.cumprod(1 + a); dd = (cum / np.maximum.accumulate(cum) - 1).min()
    print(f"  {name:32}{a.mean()/a.std()*np.sqrt(ppy):>8.2f}{dd*100:>8.1f}{len(a):>5}")
