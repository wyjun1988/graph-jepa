#!/usr/bin/env python3
"""갭하락 종목의 진입 타이밍 — 시가 대신 늦게 사면 나은가.

── 착상 (사용자, 2026-08-06) ───────────────────────────────────────────────
지정가 −1% 는 기각됐다(놓친 종목이 강한 종목이라 기회비용에 짐).
그러면 **갭하락한 종목만** 좀 나중에 사는 건?

앞 결과와 방향이 반대라 따로 잴 값어치가 있다. 두 가설이 경쟁한다:
  (a) 갭하락 = 과잉반응 → 장중 더 빠졌다가 회복. 늦게 사면 싸게 산다.
  (b) 갭하락 = 진짜 악재 → 계속 빠진다. 늦게 사면 더 비싸게(=더 나쁜 상태로) 산다.

── 방법 ────────────────────────────────────────────────────────────────────
갭 = Open[t+1] / Close[t] − 1. 현행 프로덕션은 |갭| > 5% 를 스킵하므로
그 안쪽(−5%~0%)을 구간별로 나눠 본다. 갭하락 종목에만 다른 진입을 적용하고
나머지는 전부 시가 — 즉 **조건부 정책**이다.

지연 진입 방식 세 가지:
  close_t1   당일(t+1) 종가에 매수      — 하루 지켜보고 산다
  open_t2    다음날(t+2) 시가에 매수    — 하루 통째로 넘긴다
  limit_dn   당일 저가-접근 지정가(시가−1%) — 더 빠지면 산다, 아니면 종가

청산은 채택안 D+15 종가 고정(진입일 기준). 비용 41bp.
벤치 = 유니버스 평균(시가 진입) — 전 정책 공통.

사용법:
  python scripts/gap_entry_timing_study.py --folds r5 r4
"""

import argparse
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
HOLD = 15
GAP_SKIP = 0.05                       # 프로덕션과 동일: |갭|>5% 는 진입 자체를 스킵

# (라벨, 갭하락 임계, 지연방식)  갭 <= -임계 인 종목에만 적용
POLICIES = [
    ("시가 전량 (현행)",              None,  None),
    ("갭≤-1% → 당일종가",            0.01,  "close_t1"),
    ("갭≤-2% → 당일종가",            0.02,  "close_t1"),
    ("갭≤-3% → 당일종가",            0.03,  "close_t1"),
    ("갭≤-2% → 익일시가",            0.02,  "open_t2"),
    ("갭≤-2% → 시가-1%/미체결 종가",   0.02,  "limit_dn"),
    ("갭≤-2% → 아예 스킵",           0.02,  "skip"),
]


def load_panel():
    panel = {}
    for path in sorted(OHLCV.glob("*.csv")):
        t = path.name.split("_")[0]
        ds, o, lo, c = [], [], [], []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    vals = [float(r["Open"]), float(r["Low"]), float(r["Close"])]
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(v) and v > 0 for v in vals):
                    continue
                ds.append(r["Date"][:10]); o.append(vals[0]); lo.append(vals[1]); c.append(vals[2])
        if ds:
            panel[t] = (ds, o, lo, c, {d: i for i, d in enumerate(ds)})
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r5", "r4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_panel()

    res, gapstat, seed_n = {}, {}, {}
    for fold in args.folds:
        ens, n = load_ens(fold, args.seeds, args.prefix)
        if ens is None:
            continue
        seed_n[fold] = n
        picks = {dt: sorted(ens[dt], key=lambda t: ens[dt][t], reverse=True)[:K_PICKS]
                 for dt in ens if len(ens[dt]) >= 50}
        print(f"  {fold}: 시드 {n}, {len(picks)}일 …", flush=True)

        bench = {}
        for dt in picks:
            vals = []
            for t in ens[dt]:
                rec = panel.get(t)
                if rec is None:
                    continue
                ds, o, lo, c, ix = rec
                i = ix.get(dt)
                if i is None or i + 1 >= len(ds) or i + HOLD >= len(ds) or o[i + 1] <= 0:
                    continue
                vals.append(c[i + HOLD] / o[i + 1] - 1.0)
            if len(vals) >= 50:
                bench[dt] = sum(vals) / len(vals)

        # 갭 분포·갭하락 종목의 기본 성적 (진단용)
        gd, gu = [], []
        for dt, pk in picks.items():
            if dt not in bench:
                continue
            for t in pk:
                rec = panel.get(t)
                if rec is None:
                    continue
                ds, o, lo, c, ix = rec
                i = ix.get(dt)
                if i is None or i + 1 >= len(ds) or i + HOLD >= len(ds) or o[i + 1] <= 0:
                    continue
                g = o[i + 1] / c[i] - 1.0
                if abs(g) > GAP_SKIP:
                    continue
                ex = c[i + HOLD] / o[i + 1] - 1.0 - bench[dt]
                (gd if g <= -0.02 else gu).append((ex, c[i + 1] / o[i + 1] - 1.0))
        if gd:
            gapstat[fold] = (len(gd), sum(x[0] for x in gd) / len(gd) * 100,
                             sum(x[1] for x in gd) / len(gd) * 100,
                             sum(x[0] for x in gu) / len(gu) * 100 if gu else None)

        for label, thr, mode in POLICIES:
            daily = []
            for dt, pk in picks.items():
                if dt not in bench:
                    continue
                pos = []
                for t in pk:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    ds, o, lo, c, ix = rec
                    i = ix.get(dt)
                    if i is None or i + 1 >= len(ds) or i + HOLD >= len(ds) or o[i + 1] <= 0:
                        continue
                    g = o[i + 1] / c[i] - 1.0
                    if abs(g) > GAP_SKIP:          # 프로덕션 갭 가드
                        continue
                    exit_px = c[i + HOLD]
                    if thr is None or g > -thr:    # 조건 미해당 → 시가
                        pos.append(exit_px / o[i + 1] - 1.0 - bench[dt])
                        continue
                    if mode == "skip":
                        continue
                    if mode == "close_t1":
                        price = c[i + 1]
                    elif mode == "open_t2":
                        if i + 2 >= len(ds) or o[i + 2] <= 0:
                            continue
                        price = o[i + 2]
                    else:                          # limit_dn
                        lim = o[i + 1] * 0.99
                        price = lim if lo[i + 1] <= lim else c[i + 1]
                    pos.append(exit_px / price - 1.0 - bench[dt])
                if len(pos) >= K_PICKS // 3:
                    daily.append(sum(pos) / len(pos))
            if len(daily) < 20:
                continue
            mu = sum(daily) / len(daily) - COST_BP / 1e4
            sd = statistics.stdev(daily)
            turns = TRADING_DAYS / HOLD
            res[(fold, label)] = (mu / sd * math.sqrt(turns) if sd > 0 else float("nan"),
                                  mu * turns * 100)

    folds = [f for f in args.folds if any(k[0] == f for k in res)]
    print("\n" + "=" * 78)
    print(f"갭하락 진입 타이밍 (청산 D+{HOLD} 고정, 41bp, |갭|>{GAP_SKIP:.0%} 는 프로덕션대로 스킵)")
    print("  시드: " + ", ".join(f"{f}={seed_n.get(f,0)}" for f in folds))
    print("=" * 78)
    print(f"{'정책':<28}" + "".join(f"{f+' Sh':>9}{f+' 연%':>9}" for f in folds) + f"{'평균Sh':>9}")
    print("-" * 78)
    for label, *_ in POLICIES:
        cells, shs = "", []
        for f in folds:
            r = res.get((f, label))
            if r:
                cells += f"{r[0]:>+9.2f}{r[1]:>+9.1f}"; shs.append(r[0])
            else:
                cells += f"{'.':>9}{'.':>9}"
        if shs:
            print(f"{label:<28}{cells}{sum(shs)/len(shs):>+9.2f}")
    print("\n── 진단: 갭하락(≤-2%) 종목은 어떤 놈들인가 ──")
    for f in folds:
        if f in gapstat:
            n_, ex, intraday, ex_other = gapstat[f]
            print(f"  {f}: 갭하락 {n_}건 | D+15 초과 {ex:+.2f}% (비갭하락 {ex_other:+.2f}%)"
                  f" | 진입일 시가→종가 {intraday:+.2f}%")
    print("  진입일 시가→종가가 양수면 갭하락 후 당일 반등(늦게 사면 비싸진다).")
    if args.json:
        import json as _json
        Path(args.json).write_text(_json.dumps({
            "study": "gap_entry_timing", "folds": folds, "seeds": seed_n,
            "rows": {label: {f: res[(f, label)][0] for f in folds if (f, label) in res}
                     for label, *_ in POLICIES},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
