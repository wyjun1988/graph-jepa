#!/usr/bin/env python3
"""배분 조절(vol deployment scaling) 변형 연구 — 페이퍼 최고 성과 북의 근거 검증.

── 무엇 ────────────────────────────────────────────────────────────────────
페이퍼 섀도우 `vol스케일` 이 실측에서 계속 1위다(2026-08-03 기준 +3.58%,
정정 헤지). 그 규칙은 진입일마다 배분 배수를 거는 것이다:

    m_t = clip(1 + K * z_t, lo, hi)
    z_t = (최근 W세션 실현변동성 - 직전 ZW세션 그 변동성의 평균) / 표준편차

프로덕션 값은 K=+0.3, W=20, ZW=40, clip(0.2, 2.0), 변동성 신호원 = KQ11.
K>0 이므로 **변동성이 높을 때 더 싣는다**(시장중립이라 분산이 클수록 기회가
크다는 논리). 직관과 반대 방향이라 부호부터 검증 대상이다.

── 왜 다시 재는가 ──────────────────────────────────────────────────────────
K=0.3 은 "4시드 OOS 최적"으로 정해졌다는 주석만 있고 5폴드 근거가 없다.
청산에서 2폴드 결론이 5폴드에서 뒤집힌 전례(SL-5%)가 있어 같은 자로 다시 잰다.

── 방법 ────────────────────────────────────────────────────────────────────
청산은 고정하고(기본 D+15 = 채택안) 배분 배수만 바꾼다. 진입일 t 의 코호트
순수익 r_t = (초과수익 - 비용) 에 m_t 를 곱한다. 비용도 함께 스케일된다
(m 배 규모로 거래하므로). m=1 이면 기존 결과와 정확히 일치한다.

변동성 신호원 두 가지를 비교한다:
  uni  — 유니버스(KRX500 동일가중) 일별 수익의 실현변동성. 외부 의존 없음.
  kq11 — 현행 프로덕션. FDR 로 받는다.

사용법:
  python scripts/vol_deploy_study.py --folds r5 r4
  python scripts/vol_deploy_study.py --folds r5 r4 --exit rank20
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
MAXD = 30
RANK_FRAC, RANK_CAP = 0.20, 30      # --exit rank20 일 때

# (라벨, K, 변동성창 W, z창 ZW, clip 하한, clip 상한, 신호원)
POLICIES = [
    ("고정 (m=1)",            0.00, 20, 40, 0.2, 2.0, "uni"),
    ("프로덕션 K+0.3 kq11",   0.30, 20, 40, 0.2, 2.0, "kq11"),
    ("프로덕션 K+0.3 uni",    0.30, 20, 40, 0.2, 2.0, "uni"),
    ("K+0.15",               0.15, 20, 40, 0.2, 2.0, "uni"),
    ("K+0.45",               0.45, 20, 40, 0.2, 2.0, "uni"),
    ("K+0.60",               0.60, 20, 40, 0.2, 2.0, "uni"),
    ("K-0.30 (역방향)",      -0.30, 20, 40, 0.2, 2.0, "uni"),
    ("K-0.60 (역방향)",      -0.60, 20, 40, 0.2, 2.0, "uni"),
    ("K+0.3 W10",            0.30, 10, 40, 0.2, 2.0, "uni"),
    ("K+0.3 W40",            0.30, 40, 40, 0.2, 2.0, "uni"),
    ("K+0.3 ZW20",           0.30, 20, 20, 0.2, 2.0, "uni"),
    ("K+0.3 ZW60",           0.30, 20, 60, 0.2, 2.0, "uni"),
    ("K+0.3 clip.5-1.5",     0.30, 20, 40, 0.5, 1.5, "uni"),
    ("K+0.3 clip.8-1.2",     0.30, 20, 40, 0.8, 1.2, "uni"),
]


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


def universe_daily_returns(panel, all_dates):
    """유니버스 동일가중 일별 수익률 {date: r}. 종가 기준, 전 종목 평균."""
    idx = {d: i for i, d in enumerate(all_dates)}
    acc = {d: [0.0, 0] for d in all_dates}
    for dates, _opens, closes, index in panel.values():
        for j in range(1, len(dates)):
            d = dates[j]
            if d not in idx or closes[j - 1] <= 0:
                continue
            acc[d][0] += closes[j] / closes[j - 1] - 1.0
            acc[d][1] += 1
    return {d: (s / n) for d, (s, n) in acc.items() if n >= 50}


def kq11_daily_returns():
    """KQ11 일별 수익률. FDR 실패 시 None."""
    try:
        import FinanceDataReader as fdr
        df = fdr.DataReader("KQ11")
        c = df["Close"].astype(float)
        r = c.pct_change().dropna()
        return {str(d.date()): float(v) for d, v in r.items()}
    except Exception as e:
        print(f"  (kq11 계열 적재 실패: {type(e).__name__} — 해당 정책 생략)", flush=True)
        return None


def multiplier_series(rets, dates, k, w, zw, lo, hi):
    """진입일별 배분 배수 {date: m}. 판정은 그 날짜 **이전** 수익률만 쓴다(인과)."""
    ds = sorted(rets)
    vals = [rets[d] for d in ds]
    out = {}
    for d in dates:
        i = bisect.bisect_left(ds, d)          # ds[:i] = d 미만 (엄격히 이전)
        if i < w + zw:
            out[d] = 1.0
            continue
        def vol_at(j):
            seg = vals[j - w:j]
            return statistics.pstdev(seg) if len(seg) == w else float("nan")
        cur = vol_at(i)
        prev = [vol_at(j) for j in range(i - zw, i)]
        prev = [v for v in prev if math.isfinite(v)]
        if not math.isfinite(cur) or len(prev) < 10:
            out[d] = 1.0
            continue
        mu = sum(prev) / len(prev)
        sd = statistics.pstdev(prev)
        z = (cur - mu) / (sd + 1e-12)
        out[d] = min(hi, max(lo, 1.0 + k * z))
    return out


def time_exit(rec, dt, legs):
    dates, opens, closes, index = rec
    i = index.get(dt)
    need = max(legs)
    if i is None or i + 1 >= len(dates) or i + need >= len(dates):
        return None
    entry = opens[i + 1]
    if not (entry > 0):
        return None
    return [(L, closes[i + L] / entry - 1.0) for L in legs]


def rank_exit(rec, t, dt, tops, sig_dates, cap):
    dates, opens, closes, index = rec
    i = index.get(dt)
    if i is None or i + 1 >= len(dates) or i + cap >= len(dates):
        return None
    entry = opens[i + 1]
    if not (entry > 0):
        return None
    for d in range(2, cap + 1):
        j = bisect.bisect_right(sig_dates, dates[i + d - 1]) - 1
        if j < 0:
            continue
        if t not in tops[sig_dates[j]]:
            return [(d, closes[i + d] / entry - 1.0)]
    return [(cap, closes[i + cap] / entry - 1.0)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r5", "r4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--exit", default="d15", choices=["d15", "d20", "rank20"],
                    help="배분 조절을 얹을 청산 규칙 (기본 d15 = 채택안)")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()
    all_dates = sorted({d for v in panel.values() for d in v[0]})
    uni_rets = universe_daily_returns(panel, all_dates)
    kq_rets = kq11_daily_returns()
    print(f"  유니버스 일별계열 {len(uni_rets)}일 | kq11 {len(kq_rets) if kq_rets else 0}일", flush=True)

    res, seed_n = {}, {}
    for fold in args.folds:
        ens, n = load_ens(fold, args.seeds, args.prefix)
        if ens is None:
            print(f"  {fold}: 런 없음", flush=True)
            continue
        seed_n[fold] = n
        picks = {dt: sorted(ens[dt], key=lambda t: ens[dt][t], reverse=True)[:K_PICKS]
                 for dt in ens if len(ens[dt]) >= 50}
        print(f"  {fold}: 시드 {n}개, {len(picks)}일 채점 …", flush=True)

        bench = {}
        for dt in picks:
            for d in range(1, MAXD + 1):
                vals = []
                for t in ens[dt]:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    dates, opens, closes, index = rec
                    i = index.get(dt)
                    if i is None or i + d >= len(dates) or i + 1 >= len(dates):
                        continue
                    if opens[i + 1] > 0:
                        vals.append(closes[i + d] / opens[i + 1] - 1.0)
                if len(vals) >= 50:
                    bench[(dt, d)] = sum(vals) / len(vals)

        tops = sig_dates = None
        if args.exit == "rank20":
            sig_dates = sorted(ens)
            tops = {}
            for dt in sig_dates:
                ranked = sorted(ens[dt], key=lambda t: ens[dt][t], reverse=True)
                tops[dt] = set(ranked[:max(1, int(len(ranked) * RANK_FRAC))])

        # 청산 결과는 정책과 무관하다 — 한 번만 계산해 재사용한다.
        base = {}          # dt -> (평균 초과수익, 평균 보유일)
        for dt, pk in picks.items():
            pos, holds = [], []
            for t in pk:
                rec = panel.get(t)
                if rec is None:
                    continue
                if args.exit == "rank20":
                    ex = rank_exit(rec, t, dt, tops, sig_dates, RANK_CAP)
                else:
                    ex = time_exit(rec, dt, (15,) if args.exit == "d15" else (20,))
                if ex is None:
                    continue
                xs = [(r - bench[(dt, d)]) for d, r in ex if (dt, d) in bench]
                if len(xs) == len(ex):
                    pos.append(sum(xs) / len(xs))
                    holds.append(sum(d for d, _ in ex) / len(ex))
            if len(pos) >= K_PICKS // 2:
                base[dt] = (sum(pos) / len(pos), sum(holds) / len(holds))

        if len(base) < 20:
            continue
        dts = sorted(base)
        hold = sum(base[d][1] for d in dts) / len(dts)
        turns = TRADING_DAYS / hold
        cost = COST_BP / 1e4

        for label, k, w, zw, lo, hi, src in POLICIES:
            rets = uni_rets if src == "uni" else kq_rets
            if rets is None:
                continue
            mult = ({d: 1.0 for d in dts} if k == 0
                    else multiplier_series(rets, dts, k, w, zw, lo, hi))
            scaled = [mult[d] * (base[d][0] - cost) for d in dts]
            mu = sum(scaled) / len(scaled)
            sd = statistics.stdev(scaled)
            sh = mu / sd * math.sqrt(turns) if sd > 0 else float("nan")
            mm = sum(mult[d] for d in dts) / len(dts)
            res[(fold, label)] = (sh, mu * turns * 100, mm)

    folds = [f for f in args.folds if any(x[0] == f for x in res)]
    print("\n" + "=" * 78)
    print(f"배분 조절 연구 — 청산={args.exit} (프로덕션 유니버스, 41bp)")
    print("  시드: " + ", ".join(f"{f}={seed_n.get(f, 0)}" for f in folds)
          + " | m_t = clip(1+K·z, lo, hi), z 는 인과(진입일 이전만)")
    print("=" * 78)
    print(f"{'정책':<22}" + "".join(f"{f + ' Sh':>8}{f + ' 연%':>8}{f + ' 평균m':>8}"
                                    for f in folds) + f"{'평균Sh':>8}")
    print("-" * 78)
    for label, *_ in POLICIES:
        cells, shs = "", []
        for f in folds:
            r = res.get((f, label))
            if r:
                cells += f"{r[0]:>+8.2f}{r[1]:>+8.1f}{r[2]:>8.2f}"
                shs.append(r[0])
            else:
                cells += f"{'.':>8}{'.':>8}{'.':>8}"
        if shs:
            print(f"{label:<22}{cells}{sum(shs)/len(shs):>+8.2f}")
    print("\n판정: '고정 (m=1)' 을 넘어야 배분 조절이 값어치가 있다.")
    print("      K 부호가 뒤집히면(K<0 이 이기면) '고변동성에 더 싣는다' 는 전제가 틀린 것이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
