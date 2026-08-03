#!/usr/bin/env python3
"""변동성 기반 청산 — 변동성 '변화'로 매도 타이밍을 잡을 수 있는가.

── 착상 ────────────────────────────────────────────────────────────────────
랭크 청산은 "모델 의견이 나빠지면 판다"(정보원 = 예측). 이건
"변동성 상태가 나빠지면 판다"(정보원 = 실현변동성). 정보원이 달라 겹치지 않고,
시간(D+n)·가격(TP/SL)·랭크에 이은 **제4의 청산 축**이다.

근거는 같은 날 2x2 교차(docs/VOL_HOLDING_INTERACTION_20260803.md):
  저국면·고종목 = D+5 정점,  저국면·저종목 = D+15 정점,  고국면 = 전 구간 0 근처.
보유 중에 이 상태가 바뀌면 예정일을 기다릴 이유가 없다.

── 정책군 ──────────────────────────────────────────────────────────────────
전부 인과: 보유 s 일차의 판정은 s-1 세션까지의 데이터만 쓴다(랭크 청산과 동일).

  vol_up      종목 변동성이 진입 시점 대비 R배를 넘으면 청산 (변동성 급등 이탈)
  regime_up   시장 국면 z 가 임계를 넘으면 청산 (국면 악화 이탈)
  either      위 둘 중 하나라도 발동하면 청산
  adaptive    2x2 결론을 규칙화 — 진입 시점의 (국면, 종목변동성)으로 목표 보유일을
              정하고, 보유 중 국면이 고변동으로 바뀌면 즉시 청산.
                저국면·고종목 -> D+5,  저국면·저종목 -> D+15
                고국면        -> D+5 (그 국면은 어차피 안 먹는다)

비교 기준: D+15(채택안), D+5, 랭크20(현 최고 후보), 사다리(구정책).

사용법:
  python scripts/vol_exit_study.py --folds r5 r4
"""

import argparse
import bisect
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from exit_policy_report import load_prices  # noqa: E402

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
VOL_W, Z_W = 20, 40
MAXD = 30
RANK_FRAC = 0.20

# (라벨, 종류, 파라미터…)
POLICIES = [
    ("D+15 (채택안)",        "time",   15),
    ("D+5",                 "time",    5),
    ("랭크20 (현 후보)",      "rank",  RANK_FRAC),
    ("사다리 (구정책)",        "ladder", None),
    ("vol_up 1.3x",         "volup",  1.3),
    ("vol_up 1.5x",         "volup",  1.5),
    ("vol_up 2.0x",         "volup",  2.0),
    ("regime_up z>+0.5",    "regime", 0.5),
    ("regime_up z>+1.0",    "regime", 1.0),
    ("either 1.5x/z+1.0",   "either", (1.5, 1.0)),
    ("adaptive 2x2",        "adapt",  None),
]
LADDER = (1, 2, 3, 5, 10)


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


def universe_daily_returns(panel):
    acc = {}
    for dates, _o, closes, _i in panel.values():
        for j in range(1, len(dates)):
            if closes[j - 1] <= 0:
                continue
            a = acc.setdefault(dates[j], [0.0, 0])
            a[0] += closes[j] / closes[j - 1] - 1.0
            a[1] += 1
    return {d: s / n for d, (s, n) in acc.items() if n >= 50}


def regime_z_series(rets):
    """모든 날짜의 시장 국면 z. 그 날짜 **미만** 수익률만 사용(인과)."""
    ds = sorted(rets)
    vals = [rets[d] for d in ds]
    out = {}
    for i, d in enumerate(ds):
        if i < VOL_W + Z_W:
            continue
        def vol_at(j):
            seg = vals[j - VOL_W:j]
            return statistics.pstdev(seg) if len(seg) == VOL_W else float("nan")
        cur = vol_at(i)
        prev = [v for v in (vol_at(j) for j in range(i - Z_W, i)) if math.isfinite(v)]
        if not math.isfinite(cur) or len(prev) < 10:
            continue
        sd = statistics.pstdev(prev)
        out[d] = (cur - sum(prev) / len(prev)) / (sd + 1e-12)
    return out


def stock_vol_at(rec, i):
    """인덱스 i **미만** VOL_W 세션 실현변동성."""
    _d, _o, closes, _ix = rec
    if i < VOL_W + 1:
        return None
    seg = closes[i - VOL_W:i + 1]
    if len(seg) != VOL_W + 1 or min(seg) <= 0:
        return None
    return statistics.pstdev([seg[j] / seg[j - 1] - 1.0 for j in range(1, len(seg))])


def simulate(rec, t, dt, kind, param, ctx):
    """(청산 보유일, 수익률) 또는 None."""
    dates, opens, closes, index = rec
    i = index.get(dt)
    if i is None or i + 1 >= len(dates):
        return None
    entry = opens[i + 1]
    if not (entry > 0):
        return None
    zs, tops, sig_dates, zmed, vmed = ctx

    def out(d):
        return (d, closes[i + d] / entry - 1.0) if i + d < len(dates) else None

    if kind == "time":
        return out(param)
    if kind == "ladder":
        if i + max(LADDER) >= len(dates):
            return None
        rs = [closes[i + L] / entry - 1.0 for L in LADDER]
        return (sum(LADDER) / len(LADDER), sum(rs) / len(rs))

    v0 = stock_vol_at(rec, i)
    if v0 is None or v0 <= 0:
        return None

    if kind == "adapt":
        z0 = zs.get(dates[i])
        if z0 is None:
            return None
        hi_stock = v0 >= vmed.get(dt, float("inf"))       # 코호트 내 중앙값 이상
        target = 5 if (z0 >= zmed or hi_stock) else 15
    else:
        target = MAXD

    cap = min(target, MAXD)
    if i + cap >= len(dates):
        return None
    for d in range(2, cap + 1):
        prev_i = i + d - 1                    # 판정은 직전 세션까지
        if kind in ("volup", "either"):
            ratio = param if kind == "volup" else param[0]
            v = stock_vol_at(rec, prev_i)
            if v is not None and v >= v0 * ratio:
                return out(d)
        if kind in ("regime", "either", "adapt"):
            thr = param if kind == "regime" else (param[1] if kind == "either" else zmed)
            z = zs.get(dates[prev_i])
            if z is not None and z >= thr:
                return out(d)
    return out(cap)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r5", "r4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()
    zs = regime_z_series(universe_daily_returns(panel))

    res, seed_n = {}, {}
    for fold in args.folds:
        ens, n = load_ens(fold, args.seeds, args.prefix)
        if ens is None:
            continue
        seed_n[fold] = n
        sig_dates = sorted(ens)
        picks = {dt: sorted(ens[dt], key=lambda t: ens[dt][t], reverse=True)[:K_PICKS]
                 for dt in ens if len(ens[dt]) >= 50}
        dts = sorted(picks)
        tops = {dt: set(sorted(ens[dt], key=lambda t: ens[dt][t], reverse=True)
                        [:max(1, int(len(ens[dt]) * RANK_FRAC))]) for dt in sig_dates}
        zv = sorted(v for d, v in zs.items() if d in picks)
        zmed = zv[len(zv) // 2] if zv else 0.0
        # 코호트별 종목변동성 중앙값 (adaptive 용)
        vmed = {}
        for dt in dts:
            vs = []
            for t in picks[dt]:
                rec = panel.get(t)
                if rec is None:
                    continue
                v = stock_vol_at(rec, rec[3].get(dt, -1)) if dt in rec[3] else None
                if v:
                    vs.append(v)
            if vs:
                vmed[dt] = sorted(vs)[len(vs) // 2]
        ctx = (zs, tops, sig_dates, zmed, vmed)
        print(f"  {fold}: 시드 {n}, {len(dts)}일 …", flush=True)

        bench = {}
        for dt in dts:
            for d in range(1, MAXD + 1):
                vals = []
                for t in ens[dt]:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    dd, oo, cc, ix = rec
                    i = ix.get(dt)
                    if i is None or i + d >= len(dd) or i + 1 >= len(dd) or oo[i + 1] <= 0:
                        continue
                    vals.append(cc[i + d] / oo[i + 1] - 1.0)
                if len(vals) >= 50:
                    bench[(dt, d)] = sum(vals) / len(vals)

        for label, kind, param in POLICIES:
            daily, holds = [], []
            for dt in dts:
                pos = []
                for t in picks[dt]:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    if kind == "rank":
                        ex = None
                        dd, oo, cc, ix = rec
                        i = ix.get(dt)
                        if i is not None and i + 1 < len(dd) and oo[i + 1] > 0 and i + MAXD < len(dd):
                            e = oo[i + 1]
                            for d in range(2, MAXD + 1):
                                j = bisect.bisect_right(sig_dates, dd[i + d - 1]) - 1
                                if j >= 0 and t not in tops[sig_dates[j]]:
                                    ex = (d, cc[i + d] / e - 1.0); break
                            if ex is None:
                                ex = (MAXD, cc[i + MAXD] / e - 1.0)
                    else:
                        ex = simulate(rec, t, dt, kind, param, ctx)
                    if ex is None:
                        continue
                    d, r = ex
                    key = (dt, int(round(d)))
                    if key not in bench:
                        continue
                    pos.append(r - bench[key]); holds.append(d)
                if len(pos) >= K_PICKS // 2:
                    daily.append(sum(pos) / len(pos))
            if len(daily) < 20 or not holds:
                continue
            hold = sum(holds) / len(holds)
            mu = sum(daily) / len(daily) - COST_BP / 1e4
            sd = statistics.stdev(daily)
            turns = TRADING_DAYS / hold
            if sd > 0:
                res[(fold, label)] = (mu / sd * math.sqrt(turns), mu * turns * 100, hold)

    folds = [f for f in args.folds if any(k[0] == f for k in res)]
    print("\n" + "=" * 78)
    print("변동성 기반 청산 연구 (프로덕션 유니버스, 41bp, 회전=실측 보유일)")
    print("  시드: " + ", ".join(f"{f}={seed_n.get(f,0)}" for f in folds)
          + " | 판정은 직전 세션까지의 데이터만 사용(인과)")
    print("=" * 78)
    print(f"{'정책':<20}" + "".join(f"{f+' Sh':>8}{f+' 연%':>8}{f+' 보유':>7}" for f in folds)
          + f"{'평균Sh':>8}")
    print("-" * 78)
    for label, *_ in POLICIES:
        cells, shs = "", []
        for f in folds:
            r = res.get((f, label))
            if r:
                cells += f"{r[0]:>+8.2f}{r[1]:>+8.1f}{r[2]:>7.1f}"; shs.append(r[0])
            else:
                cells += f"{'.':>8}{'.':>8}{'.':>7}"
        if shs:
            print(f"{label:<20}{cells}{sum(shs)/len(shs):>+8.2f}")
    print("\n판정: 변동성 기반 청산이 'D+15(채택안)' 과 '랭크20' 을 넘어야 제4의 축으로")
    print("      값어치가 있다. 넘지 못하면 랭크 청산에 이미 담긴 정보다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
