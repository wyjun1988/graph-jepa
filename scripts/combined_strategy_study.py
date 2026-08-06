#!/usr/bin/env python3
"""결합 전략 — 통과한 후보들을 합치면 더 좋은가.

── 질문 (사용자, 2026-08-06) ───────────────────────────────────────────────
그동안의 아이디어를 모두 합친 전략을 만들 수 없나.

── 합칠 수 있는 것 / 없는 것 ───────────────────────────────────────────────
5폴드 3게이트를 통과했거나 근접한 것만 대상으로 한다.

  랭크청산     청산 시점  : top20% 이탈시 매도, 캡 D+30      [5/5 통과]
  물타기       진입 규모  : -5% 하락시 추가매수(5일 내)       [5/5 통과]
  신호강도조절   코호트 예산 : meantop z 로 배수 K+0.6         [4/5, 마진 +0.09]
  배분조절     코호트 예산 : 시장변동성 z 로 배수 K+0.3        [마진 +0.05]

**신호강도조절과 배분조절은 같은 축(예산)이라 동시 적용하지 않는다** — 둘 중
하나만 쓴다. 나머지는 축이 직교하므로 결합 가능하다.

갈아타기는 5폴드 최악폴드 -0.26 으로 제외됐다.

── ⚠️ 결합의 두 함정 ───────────────────────────────────────────────────────
1. **중복 계산.** 랭크청산과 물타기는 둘 다 "빠진 뒤 반등"을 먹는다. 물타기는
   싸게 더 사서, 랭크청산은 안 나빠졌으면 계속 들고 있어서. 같은 효과를 두 번
   세면 결합 이득이 부풀려진다 — 개별 이득의 단순 합보다 작을 수 있다.
2. **규칙 충돌.** 물타기는 "빠지면 더 산다", 랭크청산은 "나빠지면 판다".
   빠진 종목의 스코어가 같이 나빠지면 **사자마자 파는** 일이 생긴다.
   → 충돌 처리를 두 가지로 나눠 잰다:
        naive  : 각자 독립 작동 (사고 바로 팔 수도 있음)
        guard  : 추가매수한 포지션은 최소 3세션 랭크청산 유예

── 방법 ────────────────────────────────────────────────────────────────────
전부 인과. 진입 Open[t+1], 비용은 매수 회차마다 41bp + 청산 1회.
예산 배수는 예약자본으로 정규화(최대배수를 늘 잡아둬야 하므로).
벤치 = 유니버스 평균(시가 진입, D+15) — 전 정책 공통.

사용법:
  python scripts/combined_strategy_study.py --folds r5 r4
"""

import argparse
import bisect
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OHLCV = ROOT / "data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv"
NODE_EVAL = ROOT / "reports" / "walk_forward" / "node_eval"
FOLDS = {
    "r4": "fold1_20241106_to_20250908",
    "r5": "fold1_20250905_to_20260710",
    "r3": "fold1_20240104_to_20241107",
    "r2": "fold1_20230307_to_20240105",
    "r1": "fold1_20220510_to_20230306",
}
SIGNAL_H, K_PICKS = 10, 35
COST_BP, TRADING_DAYS = 41.0, 252
BASE_HOLD, CAP = 15, 30
GAP_SKIP = 0.05
Z_W = 40
RANK_BAND = 0.20            # 랭크청산 보유 밴드
ADD_DROP, ADD_WIN = 0.05, 5  # 물타기: -5% / 5일
SIG_K, SIG_LO, SIG_HI = 0.6, 0.2, 2.0
GUARD_DAYS = 3              # 추가매수 후 랭크청산 유예

# (라벨, 랭크청산, 물타기, 예산조절, 충돌가드)
POLICIES = [
    ("현행 (D+15 단일)",              False, False, False, False),
    ("랭크청산만",                    True,  False, False, False),
    ("물타기만",                      False, True,  False, False),
    ("예산조절만",                    False, False, True,  False),
    ("랭크+물타기 (naive)",            True,  True,  False, False),
    ("랭크+물타기 (guard)",            True,  True,  False, True),
    ("랭크+예산",                     True,  False, True,  False),
    ("물타기+예산",                   False, True,  True,  False),
    ("전부 (naive)",                  True,  True,  True,  False),
    ("전부 (guard)",                  True,  True,  True,  True),
]


def load_panel():
    panel = {}
    for path in sorted(OHLCV.glob("*.csv")):
        t = path.name.split("_")[0]
        ds, o, c = [], [], []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    op, cl = float(r["Open"]), float(r["Close"])
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(op) and math.isfinite(cl) and op > 0 and cl > 0):
                    continue
                ds.append(r["Date"][:10]); o.append(op); c.append(cl)
        if ds:
            panel[t] = (ds, o, c, {d: i for i, d in enumerate(ds)})
    return panel


def load_ens(fold, seeds, prefix="ens_s"):
    maps = []
    for s in seeds:
        p = NODE_EVAL / f"{prefix}{s}_{FOLDS[fold]}" / "return_1d_forecasts.csv"
        if not p.exists():
            continue
        d = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if row["horizon"] != str(SIGNAL_H):
                    continue
                try:
                    pr = float(row["prediction_entry_path_return"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(pr):
                    d.setdefault(row["date"], {})[row["ticker"]] = pr
        maps.append(d)
    if not maps:
        return None, 0
    dates = sorted(set.intersection(*(set(m) for m in maps)))
    return ({dt: {t: sum(m[dt][t] for m in maps) / len(maps)
                  for t in set.intersection(*(set(m[dt]) for m in maps))}
             for dt in dates}, len(maps))


def budget_mult(ens, dts):
    """신호강도(meantop) 기반 예산 배수. z 는 진입일 이전 40세션만(인과)."""
    raw = {}
    for d in dts:
        v = sorted(ens[d].values(), reverse=True)
        if len(v) >= 50:
            raw[d] = sum(v[:K_PICKS]) / K_PICKS
    ds = sorted(raw)
    out = {}
    for i, d in enumerate(ds):
        if i < Z_W:
            out[d] = 1.0
            continue
        prev = [raw[x] for x in ds[i - Z_W:i]]
        mu = sum(prev) / len(prev)
        sd = statistics.pstdev(prev)
        out[d] = min(SIG_HI, max(SIG_LO, 1.0 + SIG_K * (raw[d] - mu) / (sd + 1e-12)))
    return out


def simulate(rec, t, dt, tops, sig_dates, use_rank, use_add, use_guard):
    """(자본가중 손익, 투입자본배수) 또는 None.

    손익은 '초기 1.0 기준'. 물타기 추가분은 자본을 1.0 더 쓰므로 배수가 2.0 이 된다.
    """
    ds, o, c, ix = rec
    i = ix.get(dt)
    hold_cap = CAP if use_rank else BASE_HOLD
    if i is None or i + 1 >= len(ds) or i + hold_cap >= len(ds):
        return None
    entry = o[i + 1]
    if entry <= 0:
        return None
    cost = COST_BP / 1e4

    add_px, add_day = None, None
    exit_d = None
    for d in range(1, hold_cap + 1):
        px = c[i + d]
        # 물타기 판정 (진입가 대비)
        if use_add and add_px is None and d <= ADD_WIN and px <= entry * (1 - ADD_DROP):
            add_px, add_day = px, d
        # 랭크청산 판정 — 직전 세션 스코어(인과), d>=2 부터
        if use_rank and d >= 2:
            if use_guard and add_day is not None and d < add_day + GUARD_DAYS:
                pass                                   # 추가매수 직후 유예
            else:
                j = bisect.bisect_right(sig_dates, ds[i + d - 1]) - 1
                if j >= 0 and t not in tops[sig_dates[j]]:
                    exit_d = d
                    break
        if not use_rank and d >= BASE_HOLD:
            exit_d = d
            break
    if exit_d is None:
        exit_d = hold_cap
    exit_px = c[i + exit_d]

    pnl = (exit_px / entry - 1.0 - cost)
    cap = 1.0
    if add_px is not None and add_day < exit_d:
        pnl += (exit_px / add_px - 1.0 - cost)
        cap = 2.0
    return pnl, cap, exit_d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r5", "r4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_panel()

    res, seed_n = {}, {}
    for fold in args.folds:
        ens, n = load_ens(fold, args.seeds, args.prefix)
        if ens is None:
            continue
        seed_n[fold] = n
        sig_dates = sorted(ens)
        tops = {d: set(sorted(ens[d], key=lambda t: ens[d][t], reverse=True)
                       [:max(1, int(len(ens[d]) * RANK_BAND))]) for d in sig_dates}
        dts = [d for d in sig_dates if len(ens[d]) >= 50]
        picks = {d: sorted(ens[d], key=lambda t: ens[d][t], reverse=True)[:K_PICKS] for d in dts}
        bmult = budget_mult(ens, dts)
        print(f"  {fold}: 시드 {n}, {len(dts)}일 …", flush=True)

        # ⚠️ 벤치는 (진입일, **실제 보유일**) 로 잡아야 한다. D+15 하나로 고정하면
        # 보유가 짧은 정책(랭크청산 ~8일)이 15일치 시장수익과 비교돼 부당하게 깎인다.
        bench = {}
        for dt in dts:
            for d in range(1, CAP + 1):
                vals = []
                for t in ens[dt]:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    ds, o, c, ix = rec
                    i = ix.get(dt)
                    if i is None or i + 1 >= len(ds) or i + d >= len(ds) or o[i + 1] <= 0:
                        continue
                    vals.append(c[i + d] / o[i + 1] - 1.0)
                if len(vals) >= 50:
                    bench[(dt, d)] = sum(vals) / len(vals)

        for label, ur, ua, ub, ug in POLICIES:
            daily, holds = [], []
            for dt in dts:
                if (dt, BASE_HOLD) not in bench:
                    continue
                m = bmult.get(dt, 1.0) if ub else 1.0
                pn, cp = 0.0, 0.0
                for t in picks[dt]:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    ds, o, c, ix = rec
                    i = ix.get(dt)
                    if i is None or i + 1 >= len(ds):
                        continue
                    if abs(o[i + 1] / c[i] - 1.0) > GAP_SKIP:
                        continue
                    r = simulate(rec, t, dt, tops, sig_dates, ur, ua, ug)
                    if r is None:
                        continue
                    pnl, cap, hd = r
                    b = bench.get((dt, hd))
                    if b is None:
                        continue
                    pn += pnl - cap * b
                    cp += cap
                    holds.append(hd)
                if cp <= 0:
                    continue
                # 예약자본: 물타기는 2.0, 예산조절은 최대배수 SIG_HI 를 늘 잡아둔다
                reserve = (2.0 if ua else 1.0) * (SIG_HI if ub else 1.0)
                daily.append(pn / cp * m / reserve)
            if len(daily) < 20 or not holds:
                continue
            mu = sum(daily) / len(daily)
            sd = statistics.stdev(daily)
            # 회전도 실측 보유일로 — 짧게 들면 더 자주 돈다
            turns = TRADING_DAYS / (sum(holds) / len(holds))
            res[(fold, label)] = (mu / sd * math.sqrt(turns) if sd > 0 else float("nan"),
                                  mu * turns * 100, sum(holds) / len(holds))

    folds = [f for f in args.folds if any(x[0] == f for x in res)]
    print("\n" + "=" * 78)
    print("결합 전략 (41bp/매수, 예약자본 정규화, 벤치=유니버스 D+15)")
    print("  시드: " + ", ".join(f"{f}={seed_n.get(f,0)}" for f in folds))
    print("=" * 78)
    hdr = f"{'정책':<24}"
    for f in folds:
        hdr += f"{f+' Sh':>10}{f+' 연%':>9}{'보유':>6}"
    print(hdr + f"{'평균Sh':>10}")
    print("-" * 78)
    base = None
    for label, *_ in POLICIES:
        cells, shs = "", []
        for f in folds:
            r = res.get((f, label))
            if r:
                cells += f"{r[0]:>+10.2f}{r[1]:>+9.1f}{r[2]:>6.1f}"; shs.append(r[0])
            else:
                cells += f"{'.':>10}{'.':>9}{'.':>6}"
        if shs:
            avg = sum(shs) / len(shs)
            if base is None:
                base = avg
            print(f"{label:<24}{cells}{avg:>+10.2f}")
    print("\n결합이 개별 최고를 넘어야 합칠 값이 있다. 개별 이득의 단순 합보다 작으면")
    print("중복 계산(같은 반등 효과를 두 번 셈)이 있다는 뜻이다.")
    if args.json:
        import json as _json
        Path(args.json).write_text(_json.dumps({
            "study": "combined", "folds": folds, "seeds": seed_n,
            "rows": {label: {f: res[(f, label)][0] for f in folds if (f, label) in res}
                     for label, *_ in POLICIES},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
