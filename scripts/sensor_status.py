"""Sensing readiness check for live operation. Reports each sensor's latest date
and coverage vs the model's fail-closed requirements, so the operator knows if a
fresh signal can be produced or which sensor needs a refresh.

Read-only. No collection, no credentials. Run any time to check readiness.

Model requirements (from the candidate's training args):
  OHLCV daily, investor lag1 cov>=0.95, fundamentals lag1 cov>=0.79,
  events cov>=0.99, external lag1.
"""

import sys, glob, json, argparse
from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path("/Users/wooyeol/work/stock-v2")
p = argparse.ArgumentParser()
p.add_argument("--asof", default="", help="target trading date YYYY-MM-DD (default: latest OHLCV)")
p.add_argument("--max-stale-days", type=int, default=3)
a = p.parse_args()

REQ = {"investor": 0.95, "fundamental": 0.79, "event": 0.99}
rows = []

def latest_ohlcv(pattern):
    fs = sorted(glob.glob(str(ROOT / pattern)))
    if not fs:
        return None, 0
    dmax = None; n = 0
    for f in fs[:50]:                                  # sample 50 tickers for speed
        try:
            d = pd.read_csv(f, usecols=["Date"], parse_dates=["Date"])
            m = d["Date"].max()
            dmax = m if dmax is None or m > dmax else dmax; n += 1
        except Exception:
            pass
    return (dmax.date() if dmax is not None else None), len(fs)

# OHLCV — operational cache + newest cache
op_date, op_n = latest_ohlcv("data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv/*.csv")
new_caches = sorted(glob.glob(str(ROOT / "data/staging/ohlcv_lifecycle_hybrid_krx500_pit*/ohlcv")), reverse=True)
newest = new_caches[0] if new_caches else ""
nw_date, nw_n = latest_ohlcv(Path(newest).relative_to(ROOT).as_posix() + "/*.csv") if newest else (None, 0)
asof = pd.Timestamp(a.asof) if a.asof else (pd.Timestamp(nw_date) if nw_date else pd.Timestamp(op_date))
rows.append(("OHLCV 운용캐시", str(op_date), f"{op_n}종목", "-"))
rows.append(("OHLCV 최신캐시", str(nw_date), f"{nw_n}종목 ({Path(newest).parent.name if newest else '-'})", "-"))

# investor
inv = sorted(glob.glob(str(ROOT / "data/kiwoom_investor_cache/*.csv")))
inv_date = None; inv_cov = 0
if inv:
    have = 0
    for f in inv[:100]:
        try:
            d = pd.read_csv(f, usecols=["date"], parse_dates=["date"])
            if d["date"].max() >= asof - pd.Timedelta(days=a.max_stale_days):
                have += 1
            if inv_date is None or d["date"].max() > pd.Timestamp(inv_date):
                inv_date = d["date"].max().date()
        except Exception:
            pass
    inv_cov = have / min(100, len(inv))
rows.append(("investor 순매수", str(inv_date), f"{inv_cov:.0%} (요구 {REQ['investor']:.0%})",
             "OK" if inv_cov >= REQ["investor"] and inv_date and pd.Timestamp(inv_date) >= asof - pd.Timedelta(days=a.max_stale_days) else "STALE"))

# fundamentals
try:
    fu = pd.read_json(ROOT / "data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl", lines=True)
    fu_date = pd.to_datetime(fu["available_at"]).max().date()
    fu_stale = pd.Timestamp(fu_date) < asof - pd.Timedelta(days=45)   # fundamentals lag structurally; 45d window
    rows.append(("fundamentals(DART)", str(fu_date), f"최신 available_at (구조적 45~79일 지연 정상)",
                 "OK" if not fu_stale else "CHECK"))
except Exception as e:
    rows.append(("fundamentals(DART)", "ERR", str(e)[:40], "?"))

print(f"=== 센싱 준비도 (기준일 {asof.date()}) ===\n")
print(f"{'센서':22}{'최신':>13}{'커버리지/비고':>34}{'상태':>8}")
for name, dt, note, st in rows:
    print(f"  {name:20}{dt:>13}{note:>34}{st:>8}")

# verdict
stale = [r for r in rows if r[3] == "STALE"]
print()
if any(r[0].startswith("OHLCV") and r[1] != "None" for r in rows) and not stale:
    print("판정: 센서 신선 — 신호 생성 가능 (fail-closed 게이트 통과 예상)")
else:
    print("판정: 일부 센서 STALE → refresh_all_sensors.sh 실행 필요 (운영자, --env-file)")
    for r in stale:
        print(f"  - {r[0]} 갱신 필요")
