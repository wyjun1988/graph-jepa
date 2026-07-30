#!/usr/bin/env python3
"""TSFM 벤치마크 후속 분석 — 직교성·포트폴리오·결합.

tsfm_benchmark.py --dump 이 저장한 (날짜, 종목, 신호들, 실현) CSV 만 있으면
모델 재실행 없이 돌아간다.

세 가지를 본다:
 1) 직교성  — 신호끼리 얼마나 겹치나(일별 횡단면 상관 평균). 겹치지 않는데
              IC 가 비슷하면 결합에 값어치가 있다.
 2) 포트폴리오 — IC 는 비슷해도 실제 수익은 다를 수 있다(이 프로젝트에서 네 번
              확인된 실패 양식). 매일 상위 K 동일가중, 유니버스 평균 대비 초과분,
              회전율 환산 비용.
 3) 결합    — 횡단면 z-score 평균. 챔프 단독보다 나은지 짝지은 NW t 로 검정.

사용법:
  python scripts/tsfm_analysis.py --csv preds_small.csv
"""

import argparse
import csv
import math
import sys
from itertools import combinations

TRADING_DAYS = 252
HORIZON = 10
SD_SEED = 0.0159      # 지평10 시드 σ (docs/MEASUREMENT_CORRECTIONS_20260730.md)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx, my = sum(xs) / n, sum(ys) / n
    sxy = sxx = syy = 0.0
    for x, y in zip(xs, ys):
        dx, dy = x - mx, y - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    d = math.sqrt(sxx * syy)
    return sxy / d if d > 0 else float("nan")


def newey_west_t(diffs, lag):
    n = len(diffs)
    if n < lag + 2:
        return float("nan")
    m = sum(diffs) / n
    dev = [d - m for d in diffs]
    var = sum(x * x for x in dev) / n
    for k in range(1, lag + 1):
        cov = sum(dev[t] * dev[t - k] for t in range(k, n)) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    return m / math.sqrt(var / n) if var > 0 else float("nan")


def zscore(vals):
    finite = [v for v in vals if math.isfinite(v)]
    if len(finite) < 2:
        return [0.0] * len(vals)
    m = sum(finite) / len(finite)
    sd = (sum((v - m) ** 2 for v in finite) / (len(finite) - 1)) ** 0.5
    if sd <= 0:
        return [0.0] * len(vals)
    return [((v - m) / sd if math.isfinite(v) else 0.0) for v in vals]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--cost-bp", type=float, default=41.0)
    args = ap.parse_args()

    by_date = {}
    with open(args.csv, newline="") as f:
        for row in csv.DictReader(f):
            by_date.setdefault(row["date"], []).append(row)
    dates = sorted(by_date)
    signals = ["champ", "tsfm", "mom20", "rev5"]
    label = {"champ": "챔프(앙상블)", "tsfm": "Chronos(제로샷)",
             "mom20": "20일 모멘텀", "rev5": "5일 반전", "blend": "챔프+Chronos 결합"}
    print(f"평가일 {len(dates)}일, 관측 {sum(len(v) for v in by_date.values()):,}개\n")

    # ── 1) 직교성
    print("── 신호 간 횡단면 상관 (일별 평균) ──")
    pair_corr = {}
    for a, b in combinations(signals, 2):
        cs = []
        for d in dates:
            rows = by_date[d]
            xa = [float(r[a]) for r in rows]
            xb = [float(r[b]) for r in rows]
            pairs = [(p, q) for p, q in zip(xa, xb) if math.isfinite(p) and math.isfinite(q)]
            if len(pairs) > 10:
                c = pearson([p for p, _ in pairs], [q for _, q in pairs])
                if math.isfinite(c):
                    cs.append(c)
        if cs:
            pair_corr[(a, b)] = sum(cs) / len(cs)
    for (a, b), c in sorted(pair_corr.items(), key=lambda kv: -abs(kv[1])):
        tag = "  ← 거의 직교" if abs(c) < 0.15 else ""
        print(f"  {label[a]:<16} vs {label[b]:<16} {c:+.3f}{tag}")

    # ── 2) 결합 신호 만들기 (횡단면 z 평균)
    for d in dates:
        rows = by_date[d]
        zc = zscore([float(r["champ"]) for r in rows])
        zt = zscore([float(r["tsfm"]) for r in rows])
        for r, a, b in zip(rows, zc, zt):
            r["blend"] = (a + b) / 2.0

    # ── 3) IC 와 포트폴리오
    print(f"\n── IC · 포트폴리오 (매일 상위 {args.top_k}종목 동일가중, "
          f"유니버스 평균 대비, {args.cost_bp:.0f}bp) ──")
    print(f"{'신호':<18}{'IC':>9}{'수익%/10d':>11}{'연수익%':>9}{'Sharpe':>8}")
    print("-" * 56)
    daily_ic, daily_ret = {}, {}
    for s in signals + ["blend"]:
        ics, rets = {}, {}
        for d in dates:
            rows = by_date[d]
            pairs = [(float(r[s]), float(r["realized"])) for r in rows
                     if math.isfinite(float(r[s])) and math.isfinite(float(r["realized"]))]
            if len(pairs) < 10:
                continue
            ic = pearson([p for p, _ in pairs], [z for _, z in pairs])
            if math.isfinite(ic):
                ics[d] = ic
            # 포트폴리오: 상위 K 평균 − 유니버스 평균 (시장중립)
            ranked = sorted(pairs, key=lambda pz: pz[0], reverse=True)
            k = min(args.top_k, len(ranked))
            uni = sum(z for _, z in pairs) / len(pairs)
            rets[d] = sum(z for _, z in ranked[:k]) / k - uni
        daily_ic[s], daily_ret[s] = ics, rets
        v = list(rets.values())
        m = sum(v) / len(v)
        sd = (sum((x - m) ** 2 for x in v) / (len(v) - 1)) ** 0.5
        turns = TRADING_DAYS / HORIZON
        net = (m - args.cost_bp / 1e4) * turns
        vol = sd * math.sqrt(turns)
        icm = sum(ics.values()) / len(ics)
        print(f"{label[s]:<18}{icm:>+9.4f}{m*100:>+11.3f}{net*100:>+9.1f}"
              f"{net/vol if vol > 0 else float('nan'):>8.2f}")

    # ── 4) 챔프 대비 짝지은 검정
    print(f"\n── 챔프 대비 짝지은 차이 (겹침보정 NW t, lag={HORIZON}) ──")
    print(f"{'신호':<18}{'ΔIC':>9}{'t':>7}{'Δ수익%/10d':>13}{'t':>7}")
    base_ic, base_ret = daily_ic["champ"], daily_ret["champ"]
    for s in ["tsfm", "mom20", "rev5", "blend"]:
        ci = sorted(set(daily_ic[s]) & set(base_ic))
        cr = sorted(set(daily_ret[s]) & set(base_ret))
        di = [daily_ic[s][d] - base_ic[d] for d in ci]
        dr = [daily_ret[s][d] - base_ret[d] for d in cr]
        print(f"{label[s]:<18}{sum(di)/len(di):>+9.4f}{newey_west_t(di, HORIZON):>7.2f}"
              f"{sum(dr)/len(dr)*100:>+13.3f}{newey_west_t(dr, HORIZON):>7.2f}")

    print(f"\n판정 기준: 지평10 시드 σ={SD_SEED:.4f}. |ΔIC| < {SD_SEED/2:.4f} 는 동등으로 읽는다.")
    print("포트폴리오는 유니버스 평균 대비 초과분 — 시장 방향은 제거돼 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
