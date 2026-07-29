"""Daily operational signal from the world model -- the Monday-ready core.

At the LATEST available session it runs the model (sensing-only inputs, leak-free),
producing per-stock: predicted return, MC-dropout confidence, and a
confidence-weighted long-basket signal + the index-futures hedge notional. Also
emits a tiny order list sized to a small capital for an execution smoke-test.

SAFETY: this GENERATES a signal and an order LIST only. It does NOT place orders,
log in to any broker, or touch live_orders. The user reviews and executes.
"""

import sys, json, argparse
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
p.add_argument("--top-frac", type=float, default=0.2)
p.add_argument("--passes", type=int, default=16)
p.add_argument("--min-price", type=float, default=2000.0, help="exclude names below this price (penny/liquidity filter)")
p.add_argument("--capital", type=float, default=100000.0, help="KRW for the execution smoke-test order list")
p.add_argument("--smoke-names", type=int, default=3, help="how many top names to buy in the small test")
p.add_argument("--horizons", default="1,2,3,5,10")
p.add_argument("--device", default="mps")
p.add_argument("--max-steps", type=int, default=194)
p.add_argument("--seed", type=int, default=17)
p.add_argument("--cache-dir", default="data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv")
p.add_argument("--output", default="")
args = evaluator_contract_defaults(p.parse_args())
torch.manual_seed(args.seed)
model, ckpt = load_model(Path(args.model_dir), torch.device(args.device))
model.eval()
for m in model.modules():
    if isinstance(m, nn.Dropout):
        m.train()                     # MC-dropout for confidence
ns = argparse.Namespace(**dict(ckpt.get("args", {})))
feats, ckpt_args = build_features_from_ckpt(ckpt, args)
stock = int(feats.tradable_count); H = int(args.horizon); N = int(args.passes)
ti = {t: i for i, t in enumerate(DOWNSTREAM_AUXILIARY_TASKS)}
ew = int(ckpt_args.get("edge_window", 60)); tk = int(ckpt_args.get("edge_top_k", 6)); mc = float(ckpt_args.get("min_abs_corr", 0.2))
fwd = max(1, int(rollout_steps_for_offset(ns, H)))

# "today" = last session with a valid snapshot (leak-free: only sensed data up to here)
today = len(feats.dates) - 1
batch = make_real_snapshot(feats, step=today, full_observation=True, edge_window=ew, top_k=tk, min_abs_corr=mc, **graph_edge_kwargs(ckpt_args, args)).to(next(model.parameters()).device)
samples = []
with torch.no_grad():
    for _ in range(N):
        ctx = model.encode_temporal_context(batch)
        z = model.rollout_latent(ctx, steps=fwd)
        samples.append(model.predict_downstream_auxiliary(ctx, z, rollout_steps=fwd)[:stock].cpu().numpy())
S = np.stack(samples)                                  # [N, stock, tasks]
point = S.mean(0); epi = S[:, :, ti["path_return"]].std(0)
pred_ret = point[:, ti["path_return"]]
tickers = list(feats.tickers)[:stock]
names = feats.names if hasattr(feats, "names") else {}
price = np.asarray(feats.close[today, :stock], dtype=np.float64)
prev = np.asarray(feats.close[today - 1, :stock], dtype=np.float64)
tradable = np.isfinite(price) & (price > 0) & np.isfinite(prev) & (prev > 0) & (price != prev)  # exclude halted (flat) bars
tradable &= price >= args.min_price                    # liquidity/penny filter (prudence + slippage)
ok = tradable & np.isfinite(pred_ret)
idx = np.where(ok)[0]

# selection + confidence-weighted sizing (the validated winning method)
thr = np.quantile(pred_ret[idx], 1 - args.top_frac)
sel = idx[pred_ret[idx] >= thr]
epi_sel = epi[sel]
epi_z = (epi_sel - epi_sel.mean()) / (epi_sel.std() + 1e-9)
w = np.exp(-epi_z); w = w / w.sum()                    # weight ~ exp(-confidence_z)
order = np.argsort(-w)
signal = [{"ticker": tickers[sel[i]], "name": names.get(tickers[sel[i]], ""),
           "weight": round(float(w[i]), 5), "rank_score": round(float(pred_ret[sel[i]]), 4),
           "confidence": round(float(-epi[sel[i]]), 5), "price": float(price[sel[i]])} for i in order]

# tiny execution smoke-test: greedily fill affordable top names to use most of the capital
cap = float(args.capital); smoke = []; remaining = cap
pool = [s for s in signal[:15] if s["price"] <= cap]      # affordable top names by weight
wsum = sum(s["weight"] for s in pool) or 1.0
for s in pool:
    if remaining < s["price"]:
        continue
    target = cap * s["weight"] / wsum
    sh = min(int(target // s["price"]), int(remaining // s["price"]))
    if sh >= 1:
        cost = round(sh * s["price"]); remaining -= cost
        smoke.append({"ticker": s["ticker"], "name": s["name"], "shares": sh,
                      "price": s["price"], "cost_krw": cost})
    if len(smoke) >= args.smoke_names:
        break
spent = sum(o["cost_krw"] for o in smoke)

payload = {
    "role": "daily_operational_signal", "live_orders_allowed": False, "auto_execute": False,
    "note": "SIGNAL + ORDER LIST ONLY. No orders placed, no broker login. User reviews and executes.",
    "as_of_session": str(feats.dates[today]), "model_dir": str(args.model_dir),
    "method": "top-%d%% by pred_return, confidence-weighted (exp(-epi_z)), index-futures beta hedge, h%d hold" % (int(args.top_frac*100), H),
    "n_selected": len(signal),
    "full_long_basket": signal,
    "futures_hedge": {"instrument": "KOSPI200/KOSDAQ150 index future (short)",
                      "notional_krw": "= total long notional (beta-neutral)",
                      "note": "infeasible at ~100k capital (margin too large); hedge applies at scale only"},
    "smoke_test_order_list": {"capital_krw": cap, "orders": smoke, "spent_krw": spent,
                              "note": "LONG-ONLY micro test = execution plumbing check, NOT strategy (too small for basket/hedge)"},
}
if args.output:
    Path("/Users/wooyeol/work/stock-v2", args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

print(f"=== 일간 신호 (as-of {str(feats.dates[today])[:10]}, {args.model_dir.split('/')[-1]}) ===")
print(f"방법: 상위{int(args.top_frac*100)}% 수익 + confidence 가중 + 지수선물 헤지 + h{H} 보유")
print(f"선택 종목 {len(signal)}개. 상위 8 (가중치순):\n")
print(f"{'순':>3} {'코드':>7} {'종목명':12}{'가중%':>7}{'수익점수':>8}{'현재가':>9}")
for i, s in enumerate(signal[:8]):
    print(f"{i+1:>3} {s['ticker']:>7} {s['name'][:11]:12}{s['weight']*100:>7.2f}{s['rank_score']:>8.2f}{s['price']:>9.0f}")
print("  (수익점수 = 표준화된 랭킹 점수, 실제 수익률 아님. 선택·가중에만 사용)")
print(f"\n=== 10만원 실행 스모크테스트 (롱-only, 헤지 불가) ===")
for o in smoke:
    print(f"  매수: {o['ticker']} {o['name'][:10]} {o['shares']}주 @ {o['price']:.0f} = {o['cost_krw']:,}원")
print(f"  집행액 {spent:,}원 / 자본 {int(cap):,}원  (잔액 {int(cap)-spent:,}원)")
print(f"\n⚠️ 신호·주문리스트만 생성. 실주문/로그인/live_orders 없음 — 사용자가 검토·실행.")
if args.output: print(f"-> {args.output}")
