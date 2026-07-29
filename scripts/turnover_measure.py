"""Measure the actual turnover of the top-20% long basket -- the key uncertainty
for the cost verdict. The strategy cache dropped stock identity; here we keep the
stock index so we can see how much of the basket persists period-to-period.

Daily turnover = fraction of today's top-20% NOT in yesterday's top-20%.
h-day turnover = same vs the basket h sessions ago (the non-overlapping cadence).

Low daily turnover means a daily-rebalanced book trades only small deltas, so the
real cost is far below the full-rotation worst case used in the cost analysis.

Research only.
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
stock = int(feats.tradable_count); H = int(args.horizon)
ti = {t: i for i, t in enumerate(DOWNSTREAM_AUXILIARY_TASKS)}
ew = int(ckpt_args.get("edge_window", 60)); tk = int(ckpt_args.get("edge_top_k", 6)); mc = float(ckpt_args.get("min_abs_corr", 0.2))
fwd = max(1, int(rollout_steps_for_offset(ns, H)))

# per step: set of stock indices in the top-frac by predicted return (tradable & valid entry)
baskets = []
for step in steps:
    if int(step) + H >= len(feats.dates) or int(step) + 1 >= len(feats.dates):
        continue
    batch = make_real_snapshot(feats, step=int(step), full_observation=True, edge_window=ew, top_k=tk, min_abs_corr=mc, **graph_edge_kwargs(ckpt_args, args)).to(next(model.parameters()).device)
    with torch.no_grad():
        ctx = model.encode_temporal_context(batch)
        z = model.rollout_latent(ctx, steps=fwd)
        head = model.predict_downstream_auxiliary(ctx, z, rollout_steps=fwd)[:stock].cpu().numpy()
    entry = np.asarray(feats.open[int(step) + 1, :stock], dtype=np.float64)
    valid = np.isfinite(entry) & (entry > 0) & np.isfinite(head[:, ti["path_return"]])
    idx = np.where(valid)[0]
    pr = head[idx, ti["path_return"]]
    k = max(1, int(len(idx) * args.top_frac))
    top = set(idx[np.argsort(pr)[-k:]].tolist())
    baskets.append(top)

def turnover(lag):
    ts = []
    for i in range(lag, len(baskets)):
        prev, cur = baskets[i - lag], baskets[i]
        if not cur:
            continue
        ts.append(len(cur - prev) / len(cur))   # fraction of current basket that is NEW
    return float(np.mean(ts)) if ts else float("nan")

print(f"상위 {args.top_frac:.0%} 롱 바스켓 회전율 ({len(baskets)} 세션, 평균 {np.mean([len(b) for b in baskets]):.0f}종목)\n")
for lag, lbl in [(1, "일간(1세션)"), (5, "주간(5세션)"), (H, f"h{H}(비중첩)")]:
    to = turnover(lag)
    print(f"  {lbl:16} 신규편입 비율 {to*100:5.1f}%  → 왕복거래 대상 {to*100:4.1f}% of 바스켓")
d1 = turnover(1)
print(f"\n해석: 일간 회전 {d1*100:.0f}% → 일일 리밸런싱 시 매일 바스켓의 {d1*100:.0f}%만 교체.")
print(f"연 매도세(일간회전): 0.20% x {d1:.2f} x 252 = {0.20*d1*252:.1f}%/년 (풀턴오버 5.0% 대비)")
