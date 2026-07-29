"""Shadow paper-trading ledger for the new candidate -- runs PARALLEL to the live
4-task chain, never touches it. Records each daily signal; reconciles matured
entries (h-day hold) from the OHLCV cache into realized, beta-hedged P&L.

SAFETY: paper only. live_orders=false. Writes to ops/shadow_v17_candidate/ (a NEW
dir), never to ops/prospective_live/post_impact/ledger.jsonl or the 4 model files.

Modes:
  --append <signal.json>   log today's signal (entry = next session open)
  --reconcile              mark matured entries to realized return from OHLCV cache
  --status                 show open/closed entries + cumulative shadow P&L
"""

import sys, json, argparse, glob, os
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/Users/wooyeol/work/stock-v2")
LEDGER_DIR = ROOT / "ops/shadow_v17_candidate"
LEDGER = LEDGER_DIR / "ledger.jsonl"
OHLCV = ROOT / "data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv"
GUARD = ROOT / "ops/prospective_live/post_impact/ledger.jsonl"   # the LIVE ledger -- must never be our target

p = argparse.ArgumentParser()
p.add_argument("--append", default="")
p.add_argument("--reconcile", action="store_true")
p.add_argument("--status", action="store_true")
p.add_argument("--horizon", type=int, default=10)
args = p.parse_args()
assert LEDGER.resolve() != GUARD.resolve(), "refuse: shadow ledger must differ from the live ledger"
LEDGER_DIR.mkdir(parents=True, exist_ok=True)


def read_ledger():
    if not LEDGER.is_file():
        return []
    return [json.loads(l) for l in LEDGER.read_text(encoding="utf-8").splitlines() if l.strip()]


def ohlcv(ticker):
    m = glob.glob(str(OHLCV / f"{ticker}_*.csv"))
    if not m:
        return None
    d = pd.read_csv(m[0], usecols=["Date", "Open", "Close"], parse_dates=["Date"])
    return d.reset_index(drop=True)


if args.append:
    sig = json.loads((ROOT / args.append).read_text(encoding="utf-8") if not os.path.isabs(args.append) else Path(args.append).read_text(encoding="utf-8"))
    entry = {"as_of": sig["as_of_session"], "model_dir": sig.get("model_dir", ""), "horizon": args.horizon,
             "role": "shadow_paper", "live_orders_allowed": False,
             "basket": [{"ticker": s["ticker"], "weight": s["weight"]} for s in sig["full_long_basket"]],
             "status": "open"}
    with LEDGER.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    print(f"shadow 기록됨: as_of {entry['as_of'][:10]}, {len(entry['basket'])}종목 -> {LEDGER}")

if args.reconcile:
    led = read_ledger()
    # universe mean return proxy (index/beta hedge) per (entry_date -> +H) computed from all cached tickers
    changed = 0
    for e in led:
        if e.get("status") != "open":
            continue
        # find a reference calendar from any basket ticker
        rets, valid = [], True
        idx_rets = []
        for b in e["basket"]:
            df = ohlcv(b["ticker"])
            if df is None:
                continue
            hit = df.index[df["Date"] == pd.Timestamp(e["as_of"])]
            if len(hit) == 0 or hit[0] + 1 + e["horizon"] >= len(df):
                valid = False; break                     # not matured yet
            t = hit[0]
            en = df["Open"].iloc[t + 1]; ex = df["Close"].iloc[t + e["horizon"]]
            if en > 0 and ex > 0:
                rets.append((b["weight"], ex / en - 1.0))
        if not valid or not rets:
            continue
        # beta hedge proxy: mean return of all cached tickers over the same window
        allr = []
        for f in glob.glob(str(OHLCV / "*.csv")):
            d = pd.read_csv(f, usecols=["Date", "Open", "Close"], parse_dates=["Date"])
            hit = d.index[d["Date"] == pd.Timestamp(e["as_of"])]
            if len(hit) and hit[0] + 1 + e["horizon"] < len(d):
                en = d["Open"].iloc[hit[0] + 1]; ex = d["Close"].iloc[hit[0] + e["horizon"]]
                if en > 0 and ex > 0:
                    allr.append(ex / en - 1.0)
        wsum = sum(w for w, _ in rets)
        long_ret = sum(w * r for w, r in rets) / wsum
        idx = float(np.mean(allr)) if allr else 0.0
        e["status"] = "closed"; e["long_return"] = round(long_ret, 5)
        e["index_return"] = round(idx, 5); e["hedged_return"] = round(long_ret - idx, 5)
        changed += 1
    LEDGER.write_text("".join(json.dumps(e, ensure_ascii=False) + "\n" for e in led), encoding="utf-8")
    print(f"reconcile: {changed}건 정산 완료")

if args.status or (not args.append and not args.reconcile):
    led = read_ledger()
    op = [e for e in led if e.get("status") == "open"]
    cl = [e for e in led if e.get("status") == "closed"]
    print(f"=== shadow 원장 ({LEDGER}) ===")
    print(f"열림 {len(op)}건, 닫힘 {len(cl)}건")
    if cl:
        hr = np.array([e["hedged_return"] for e in cl])
        cum = np.prod(1 + hr) - 1
        print(f"닫힌 헤지수익: 평균 {hr.mean()*100:.3f}%/건, 적중 {(hr>0).mean()*100:.0f}%, 누적 {cum*100:.2f}%")
    for e in cl[-5:]:
        print(f"  {e['as_of'][:10]} 롱 {e['long_return']*100:+.2f}% - 지수 {e['index_return']*100:+.2f}% = 헤지 {e['hedged_return']*100:+.2f}%")
    print("live_orders=false, 페이퍼 전용, 라이브 체인과 분리.")
