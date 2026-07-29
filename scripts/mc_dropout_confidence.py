"""User's insight: our predictions have CONFIDENCE (epistemic uncertainty), which
acts like variance and belongs in sizing -- distinct from market vol (aleatoric).

The model has dropout(0.05). MC-dropout: run N stochastic forward passes with
dropout ON -> per-stock spread of pred_return = epistemic uncertainty sigma_epi.

Two questions:
  1. Is the confidence REAL? Does sigma_epi predict the model's actual error
     |realized - point|? (If not, it's noise.)
  2. Does confidence-aware sizing beat equal-weight? (down-weight / exclude names
     the model is unsure about, vs the naive vol-sizing that failed.)

Research only.
"""

import sys, argparse
from pathlib import Path
SC = Path("/Users/wooyeol/work/stock-v2-candidate-v17")
sys.path.insert(0, str(SC))
import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from scripts.evaluate_node_prediction import build_features_from_ckpt, graph_edge_kwargs, load_model, select_steps
from scripts.evaluate_plan_timing import evaluator_contract_defaults
from scripts.run_real_backtest import rollout_steps_for_offset
from stock_v2.graph_jepa import DOWNSTREAM_AUXILIARY_TASKS
from stock_v2.real_features import make_real_snapshot

p = argparse.ArgumentParser()
p.add_argument("--model-dir", required=True)
p.add_argument("--horizon", type=int, default=10)
p.add_argument("--top-frac", type=float, default=0.2)
p.add_argument("--passes", type=int, default=16, help="MC-dropout stochastic passes")
p.add_argument("--horizons", default="1,2,3,5,10")
p.add_argument("--device", default="mps")
p.add_argument("--max-steps", type=int, default=194)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--cache-dir", default="data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv")
args = evaluator_contract_defaults(p.parse_args())
torch.manual_seed(args.seed)
model, ckpt = load_model(Path(args.model_dir), torch.device(args.device))
model.eval()
for m in model.modules():                       # MC-dropout: dropout ON, rest eval
    if isinstance(m, nn.Dropout):
        m.train()
ns = argparse.Namespace(**dict(ckpt.get("args", {})))
feats, ckpt_args = build_features_from_ckpt(ckpt, args)
steps = select_steps(feats, ckpt_args, args)
stock = int(feats.tradable_count); H = int(args.horizon); N = int(args.passes)
ti = {t: i for i, t in enumerate(DOWNSTREAM_AUXILIARY_TASKS)}
ew = int(ckpt_args.get("edge_window", 60)); tk = int(ckpt_args.get("edge_top_k", 6)); mc = float(ckpt_args.get("min_abs_corr", 0.2))
fwd = max(1, int(rollout_steps_for_offset(ns, H)))

rows = []           # (point_ret, epi_std, realized, err)
err_ic = []         # per-date spearman(epi_std, |err|)
for step in list(steps)[::H]:   # non-overlapping dates only (speed: ~20 dates x N passes)
    s = int(step)
    if s + H >= len(feats.dates) or s + 1 >= len(feats.dates):
        continue
    batch = make_real_snapshot(feats, step=s, full_observation=True, edge_window=ew, top_k=tk, min_abs_corr=mc, **graph_edge_kwargs(ckpt_args, args)).to(next(model.parameters()).device)
    samples = []
    with torch.no_grad():
        for _ in range(N):
            ctx = model.encode_temporal_context(batch)
            z = model.rollout_latent(ctx, steps=fwd)
            samples.append(model.predict_downstream_auxiliary(ctx, z, rollout_steps=fwd)[:stock, ti["path_return"]].cpu().numpy())
    samples = np.stack(samples)                  # [N, stock]
    point = samples.mean(0); epi = samples.std(0)
    entry = np.asarray(feats.open[s + 1, :stock]); exitp = np.asarray(feats.close[s + H, :stock])
    ok = np.isfinite(entry) & (entry > 0) & np.isfinite(exitp) & (exitp > 0) & np.isfinite(point)
    realized = exitp / entry - 1.0
    idx = np.where(ok)[0]
    # confidence validity: does epi predict the ranking error? use rank of point vs rank of realized error
    err = np.abs(point[idx] - np.sign(realized[idx]) * np.abs(point[idx]))  # placeholder; use |realized - point-standardized|
    # simpler: cross-sectional error = |z(realized) - z(point)|
    zr = (realized[idx] - realized[idx].mean()) / (realized[idx].std() + 1e-9)
    zp = (point[idx] - point[idx].mean()) / (point[idx].std() + 1e-9)
    cerr = np.abs(zr - zp)
    if len(idx) >= 10:
        ic, _ = spearmanr(epi[idx], cerr)
        if np.isfinite(ic):
            err_ic.append(ic)
    day = feats.dates[s]
    for j in idx:
        rows.append((day, point[j], epi[j], realized[j]))

import pandas as pd
df = pd.DataFrame(rows, columns=["date", "point", "epi", "ret"])
print(f"MC-dropout ({N} passes, {df['date'].nunique()} 세션, dropout={ckpt_args.get('dropout', 0.05)})\n")
print(f"1) confidence 유효성: epi_std가 예측오차를 맞추는 IC = {np.mean(err_ic):+.4f} "
      f"(t={np.mean(err_ic)/(np.std(err_ic)/np.sqrt(len(err_ic))):+.1f}, >0이면 불확실성이 진짜 오차 예측)")
print(f"   epi 횡단면 분산(평균 std/mean): {df['epi'].mean():.5f} / point |mean| {df['point'].abs().mean():.5f}")

# 2) 사이징 비교
df["point_z"] = df.groupby("date")["point"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
df["epi_z"] = df.groupby("date")["epi"].transform(lambda s: (s - s.mean()) / (s.std() + 1e-9))
ppy = 252 / H
def port(fn):
    out = []
    for d, g in df.groupby("date"):
        if len(g) < 10:
            out.append(np.nan); continue
        w = fn(g);
        if w.sum() <= 0: out.append(np.nan); continue
        w = w / w.sum()
        out.append(float((w * g["ret"]).sum() - g["ret"].mean()))
    a = np.array([x for x in out if np.isfinite(x)])   # dates already non-overlapping
    if len(a) < 3 or a.std() == 0: return (0, 0, len(a))
    cum = np.cumprod(1 + a); dd = (cum / np.maximum.accumulate(cum) - 1).min()
    return (a.mean() / a.std() * np.sqrt(ppy), dd * 100, len(a))
def top(g): return (g["point"] >= g["point"].quantile(1 - args.top_frac)).astype(float).values
def top_conf_w(g):  # top-20%, weight by confidence (1/epi)
    sel = g["point"] >= g["point"].quantile(1 - args.top_frac)
    return np.where(sel, np.exp(-g["epi_z"].values), 0.0)
def top_conf_filter(g):  # top-30% by return, then keep the most-confident 20%, equal-weight
    a = g["point"] >= g["point"].quantile(1 - args.top_frac * 1.5)
    sub = g[a]
    keep = sub["epi"] <= sub["epi"].quantile(0.67)
    w = np.zeros(len(g)); w[np.where(a.values)[0][keep.values]] = 1.0
    return w
def ret_per_epi(g):  # weight ~ exp(point_z - epi_z)
    return np.exp(g["point_z"].values - g["epi_z"].values)
print(f"\n2) 사이징 (베타헤지, 비중첩)")
print(f"{'방법':30}{'Sharpe':>8}{'MDD%':>8}")
for name, fn in [("등가중 상위20%", top), ("상위20% confidence가중", top_conf_w),
                 ("상위 confidence필터+등가중", top_conf_filter), ("exp(수익-불확실성)", ret_per_epi)]:
    sh, dd, n = port(fn); print(f"  {name:28}{sh:>8.2f}{dd:>8.1f}")
