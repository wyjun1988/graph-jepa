#!/usr/bin/env python3
"""신호 강도 기반 조절 — 살아남은 컨셉을 남은 고정 파라미터에 적용한다.

── 컨셉 (2026-08-06) ───────────────────────────────────────────────────────
이번 세션에서 살아남은 셋(랭크청산·물타기·갈아타기)의 공통 구조는 하나다:
**진입 시점에 고정하지 않고, 매일 갱신되는 모델 스코어로 사후 조정한다.**
반대로 가격 기반(TP/SL/갭/지정가/평탄)은 예외 없이 기각됐다.

아직 고정돼 있는 파라미터가 셋 있다:
  COHORT_FRAC = 0.15   코호트 예산       → 신호가 강한 날 더 싣는가?
  top_frac    = 0.10   종목 수(35)       → 신호가 갈릴 때 좁히는가?
  weight = exp(-epi_z) 가중치(진입 고정)  → 매일 재조정하는가?

앞 둘은 **진입일에 정하는** 것이라 이 스크립트에서 잰다(셋째는 보유 중 재조정이라
switch_study 의 연속형이므로 별도).

배분조절(vol_deploy_study)과 같은 형식이되 신호원이 다르다:
  거기: 시장 실현변동성 z (외생)
  여기: **우리 모델 스코어에서 나온 강도** (내생)
vol_deploy 는 5폴드에서 마진 +0.05 로 겨우 통과했다. 신호 기반이 그보다 나은지가
이 연구의 질문이다.

── 신호 강도 측정 (전부 그날 스코어에서, 인과) ─────────────────────────────
  mean_top : 상위 35 예측수익의 평균        — "오늘 얼마나 좋은 게 있나"
  disp     : 전 종목 예측의 횡단면 표준편차   — "얼마나 갈리나"
  spread   : 상위10% 평균 − 중앙값          — "상단이 얼마나 튀나"

각 측정치를 ZW세션 z-score 로 바꿔 m_t = clip(1+K·z, lo, hi) 를 만든다.

── 두 적용 ────────────────────────────────────────────────────────────────
  deploy : 코호트 예산 × m_t   (자본 조절)
  breadth: 종목 수 × m_t       (분산도 조절 — 신호 강하면 좁게)

비용 41bp, 청산 D+15 고정. 예약자본 기준으로 정규화한다(scale_in 에서 배운 것 —
트리거를 사전에 모르므로 최대분을 예약해야 공정하다).

사용법:
  python scripts/signal_scaling_study.py --folds r5 r4
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
HOLD, Z_W = 15, 40
GAP_SKIP = 0.05

# (라벨, 적용, 신호원, K, clip 하한, clip 상한)
POLICIES = [
    ("고정 (현행)",                "none",    None,       0.0,  1.0, 1.0),
    ("예산×meantop K+0.3",        "deploy",  "mean_top", 0.3,  0.2, 2.0),
    ("예산×meantop K+0.6",        "deploy",  "mean_top", 0.6,  0.2, 2.0),
    ("예산×meantop K-0.3",        "deploy",  "mean_top", -0.3, 0.2, 2.0),
    ("예산×disp K+0.3",           "deploy",  "disp",     0.3,  0.2, 2.0),
    ("예산×disp K+0.6",           "deploy",  "disp",     0.6,  0.2, 2.0),
    ("예산×spread K+0.3",         "deploy",  "spread",   0.3,  0.2, 2.0),
    ("예산×spread K+0.6",         "deploy",  "spread",   0.6,  0.2, 2.0),
    ("종목수×disp K-0.3(강하면좁게)", "breadth", "disp",    -0.3, 0.4, 1.6),
    ("종목수×disp K+0.3(강하면넓게)", "breadth", "disp",     0.3, 0.4, 1.6),
    ("종목수×spread K-0.3",        "breadth", "spread",  -0.3, 0.4, 1.6),
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


def strength(ens, dt, kind):
    """그날 스코어에서 뽑은 신호 강도 — 그날 정보만 쓰므로 인과."""
    vals = sorted(ens[dt].values(), reverse=True)
    if len(vals) < 50:
        return None
    if kind == "mean_top":
        return sum(vals[:K_PICKS]) / K_PICKS
    if kind == "disp":
        return statistics.pstdev(vals)
    if kind == "spread":
        k = max(1, len(vals) // 10)
        return sum(vals[:k]) / k - vals[len(vals) // 2]
    return None


def mult_series(ens, dts, kind, k, lo, hi):
    """진입일별 배수. z-score 는 **그 날짜 이전** ZW세션만 쓴다(인과)."""
    raw = {d: strength(ens, d, kind) for d in dts}
    raw = {d: v for d, v in raw.items() if v is not None}
    ds = sorted(raw)
    out = {}
    for i, d in enumerate(ds):
        if i < Z_W:
            out[d] = 1.0
            continue
        prev = [raw[x] for x in ds[i - Z_W:i]]
        mu = sum(prev) / len(prev)
        sd = statistics.pstdev(prev)
        z = (raw[d] - mu) / (sd + 1e-12)
        out[d] = min(hi, max(lo, 1.0 + k * z))
    return out


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
        dts = sorted(d for d in ens if len(ens[d]) >= 50)
        ranked = {d: sorted(ens[d], key=lambda t: ens[d][t], reverse=True) for d in dts}
        print(f"  {fold}: 시드 {n}, {len(dts)}일 …", flush=True)

        bench = {}
        for dt in dts:
            vals = []
            for t in ens[dt]:
                rec = panel.get(t)
                if rec is None:
                    continue
                ds, o, c, ix = rec
                i = ix.get(dt)
                if i is None or i + 1 >= len(ds) or i + HOLD >= len(ds) or o[i + 1] <= 0:
                    continue
                vals.append(c[i + HOLD] / o[i + 1] - 1.0)
            if len(vals) >= 50:
                bench[dt] = sum(vals) / len(vals)

        for label, mode, kind, k, lo, hi in POLICIES:
            mult = ({d: 1.0 for d in dts} if mode == "none"
                    else mult_series(ens, dts, kind, k, lo, hi))
            daily, mm = [], []
            for dt in dts:
                if dt not in bench or dt not in mult:
                    continue
                m = mult[dt]
                npick = K_PICKS if mode != "breadth" else max(5, int(round(K_PICKS * m)))
                ex = []
                for t in ranked[dt][:npick]:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    ds, o, c, ix = rec
                    i = ix.get(dt)
                    if i is None or i + 1 >= len(ds) or i + HOLD >= len(ds) or o[i + 1] <= 0:
                        continue
                    if abs(o[i + 1] / c[i] - 1.0) > GAP_SKIP:
                        continue
                    ex.append(c[i + HOLD] / o[i + 1] - 1.0 - bench[dt] - COST_BP / 1e4)
                if len(ex) < 8:
                    continue
                r = sum(ex) / len(ex)
                if mode == "deploy":
                    # 예약자본 기준: 최대배수(hi)를 항상 잡아둬야 한다
                    daily.append(r * m / hi)
                else:
                    daily.append(r)
                mm.append(m)
            if len(daily) < 20:
                continue
            mu = sum(daily) / len(daily)
            sd = statistics.stdev(daily)
            turns = TRADING_DAYS / HOLD
            res[(fold, label)] = (mu / sd * math.sqrt(turns) if sd > 0 else float("nan"),
                                  mu * turns * 100, sum(mm) / len(mm))

    folds = [f for f in args.folds if any(x[0] == f for x in res)]
    print("\n" + "=" * 84)
    print(f"신호 강도 기반 조절 (청산 D+{HOLD}, 41bp, deploy 는 예약자본 기준)")
    print("  시드: " + ", ".join(f"{f}={seed_n.get(f,0)}" for f in folds)
          + " | z-score 는 진입일 이전 40세션만(인과)")
    print("=" * 84)
    hdr = f"{'정책':<26}"
    for f in folds:
        hdr += f"{f+' Sh':>9}{f+' 연%':>8}{'평균m':>7}"
    print(hdr + f"{'평균Sh':>9}")
    print("-" * 84)
    for label, *_ in POLICIES:
        cells, shs = "", []
        for f in folds:
            r = res.get((f, label))
            if r:
                cells += f"{r[0]:>+9.2f}{r[1]:>+8.1f}{r[2]:>7.2f}"
                shs.append(r[0])
            else:
                cells += f"{'.':>9}{'.':>8}{'.':>7}"
        if shs:
            print(f"{label:<26}{cells}{sum(shs)/len(shs):>+9.2f}")
    print("\ndeploy = 코호트 예산 조절(예약자본으로 정규화 — 최대배수를 늘 잡아둬야 하므로).")
    print("breadth = 종목 수 조절. K<0 은 '신호 강하면 좁게', K>0 은 '강하면 넓게'.")
    print("현행(고정)을 넘어야 값이 있다. 배분조절(시장변동성 기반)은 5폴드 마진 +0.05 였다.")
    if args.json:
        import json as _json
        Path(args.json).write_text(_json.dumps({
            "study": "signal_scaling", "folds": folds, "seeds": seed_n,
            "rows": {label: {f: res[(f, label)][0] for f in folds if (f, label) in res}
                     for label, *_ in POLICIES},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
