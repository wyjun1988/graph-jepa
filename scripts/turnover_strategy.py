"""Turnover reduction = the key lever for NET edge. Test hysteresis (a hold band)
on the h10 rebalance: keep a held name while it stays in the top-`exit_frac`;
only replace names that fall out; fill to the top-`entry_frac` set. Fewer trades
-> lower Korean sell-tax drag -> higher net.

Compares FULL re-selection (top-20% every rebalance) vs several hysteresis bands,
tracking stock identity across the ~20 non-overlapping rebalance points. Net =
gross - turnover * round-trip cost.
"""

import sys, argparse
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
p.add_argument("--model-dir", required=True)
p.add_argument("--horizon", type=int, default=10)
p.add_argument("--entry-frac", type=float, default=0.2)
p.add_argument("--passes", type=int, default=12)
p.add_argument("--roundtrip-cost", type=float, default=0.0033)
p.add_argument("--futures-cost", type=float, default=0.0005)
p.add_argument("--horizons", default="1,2,3,5,10")
p.add_argument("--device", default="mps")
p.add_argument("--max-steps", type=int, default=194)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--cache-dir", default="data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv")
args = evaluator_contract_defaults(p.parse_args())
H = int(args.horizon); N = int(args.passes)
torch.manual_seed(args.seed)
model, ckpt = load_model(Path(args.model_dir), torch.device(args.device))
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

# snapshots at non-overlapping rebalance points: per stock-id -> (pred_ret, epi, realized)
snaps = []
for step in list(steps)[::H]:
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
    snaps.append({"pred": point, "epi": epi, "ret": exitp / entry - 1.0, "idx": idx})


def run(exit_frac):
    """exit_frac=None -> full re-selection each period. Else hysteresis: hold names
    still within top-exit_frac; fill to top-entry_frac with fresh names."""
    prev_w = np.zeros(stock); rets, tos = [], []
    for sn in snaps:
        idx = sn["idx"]; point = sn["pred"]; epi = sn["epi"]
        entry_thr = np.quantile(point[idx], 1 - args.entry_frac)
        held = np.where(prev_w > 0)[0]
        keep = set()
        if exit_frac is not None and len(held):
            exit_thr = np.quantile(point[idx], 1 - exit_frac)
            keep = {i for i in held if i in set(idx) and point[i] >= exit_thr}
        fresh = {i for i in idx if point[i] >= entry_thr}
        sel = np.array(sorted(keep | fresh)) if (keep or fresh) else idx[point[idx] >= entry_thr]
        ez = (epi[sel] - epi[sel].mean()) / (epi[sel].std() + 1e-9)
        w = np.exp(-ez); w = w / w.sum()
        full = np.zeros(stock); full[sel] = w
        tos.append(0.5 * np.abs(full - prev_w).sum())
        r = sn["ret"]
        rets.append(float((full[sel] * r[sel]).sum() - r[idx].mean()))
        prev_w = full
    g = np.array(rets); to = float(np.mean(tos[1:])) if len(tos) > 1 else 1.0
    cost = to * args.roundtrip_cost + args.futures_cost
    net = g - cost
    ppy = 252 / H
    def sh(x): return x.mean() / x.std() * np.sqrt(ppy) if x.std() > 0 else 0
    return sh(g), sh(net), to, cost


print(f"회전율 축소 (히스테리시스 밴드) — {args.model_dir.split('/')[-1]}, 진입 상위{int(args.entry_frac*100)}%\n")
print(f"{'규칙':28}{'회전율':>7}{'비용%':>7}{'gross':>7}{'NET':>7}")
for label, ef in [("풀 재선택(매 기간)", None), ("히스테리시스 상위30% 유지", 0.30),
                  ("히스테리시스 상위40% 유지", 0.40), ("히스테리시스 상위50% 유지", 0.50)]:
    gs, nsr, to, cost = run(ef)
    print(f"  {label:26}{to*100:>6.0f}%{cost*100:>7.2f}{gs:>7.2f}{nsr:>7.2f}")
