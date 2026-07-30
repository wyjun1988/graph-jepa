#!/usr/bin/env python3
"""사전등록 검정 — 챔프를 배제 필터로 쓰는 방식 (docs/PREREG_FILTER_R3_20260731.md)

규칙은 문서에 고정돼 있고 여기서 바꾸지 않는다. 파라미터를 CLI 로 열어두지 않은
것도 의도적이다 — 결과를 보고 조정하면 사전등록이 무의미해진다.

  A (현행)   유동성 top100 중 챔프 상위 20
  C (주가설) 챔프 하위 20% 제외 → 남은 것 중 Chronos 상위 20
  D (통제군) 유동성 top100 중 Chronos 상위 20 (필터 없음)

정정1(문서 참조): 원래 B팔(챔프Q1제외+챔프 상위20)은 A와 수학적으로 동일해
퇴화한다 — 상위 20은 결코 하위 20%에 들지 않는다. D로 교체했다.

주 판정: t(C−A) > 2 (일별 초과수익 짝지음, 겹침보정 NW lag=10)
         그리고 3시드 개별 부호가 2/3 이상 양수
부 판정: t(C−D) — 챔프 필터가 Chronos 단독에 무언가를 더하는가

입력은 tsfm_benchmark.py --dump 이 만든 CSV
(date, ticker, champ, tsfm, mom20, rev5, realized).

사용법:
  python scripts/prereg_filter_test.py --csv preds_r3.csv \
      --seed-csv preds_r3_s3.csv preds_r3_s17.csv preds_r3_s29.csv
"""

import argparse
import csv
import math
import statistics
import sys

# ── 사전등록 고정값 (docs/PREREG_FILTER_R3_20260731.md §4) ──
EXCLUDE_FRAC = 0.20     # 챔프 하위 20% 배제
TOP_K = 20              # 편입 종목수
HORIZON = 10            # 거래일
COST_BP = 41.0          # 회전율 환산 비용
TRADING_DAYS = 252
COST_GRID = (26, 41, 72)


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


def newey_west_t(diffs, lag=HORIZON):
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


def load(path):
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rec = {
                    "ticker": r["ticker"],
                    "champ": float(r["champ"]),
                    "tsfm": float(r["tsfm"]),
                    "realized": float(r["realized"]),
                }
            except (KeyError, TypeError, ValueError):
                continue
            if not all(math.isfinite(rec[k]) for k in ("champ", "tsfm", "realized")):
                continue
            by.setdefault(r["date"], []).append(rec)
    return by


def arm_returns(by, arm):
    """{date: 유니버스 평균 대비 초과수익}"""
    out = {}
    for date, rows in by.items():
        if len(rows) < 30:
            continue
        uni = sum(r["realized"] for r in rows) / len(rows)
        if arm == "A":
            picks = sorted(rows, key=lambda r: -r["champ"])[:TOP_K]
        elif arm == "D":
            picks = sorted(rows, key=lambda r: -r["tsfm"])[:TOP_K]
        else:
            cut = int(len(rows) * EXCLUDE_FRAC)
            kept = sorted(rows, key=lambda r: r["champ"])[cut:]
            picks = sorted(kept, key=lambda r: -r["tsfm"])[:TOP_K]
        if len(picks) < 5:
            continue
        out[date] = sum(r["realized"] for r in picks) / len(picks) - uni
    return out


def stats(daily, cost_bp):
    v = [daily[d] for d in sorted(daily)]
    m = sum(v) / len(v)
    sd = statistics.stdev(v) if len(v) > 1 else float("nan")
    turns = TRADING_DAYS / HORIZON
    net = (m - cost_bp / 1e4) * turns
    vol = sd * math.sqrt(turns)
    return m, net, (net / vol if vol > 0 else float("nan"))


def ic_of(by, key):
    v = []
    for _, rows in by.items():
        if len(rows) < 10:
            continue
        c = pearson([r[key] for r in rows], [r["realized"] for r in rows])
        if math.isfinite(c):
            v.append(c)
    return sum(v) / len(v) if v else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="앙상블 예측 CSV")
    ap.add_argument("--seed-csv", nargs="*", default=[],
                    help="시드별 CSV (부 판정: 부호 일치 2/3 확인용)")
    args = ap.parse_args()

    by = load(args.csv)
    dates = sorted(by)
    print("═" * 62)
    print("사전등록 검정 — docs/PREREG_FILTER_R3_20260731.md")
    print(f"고정값: 배제 하위 {EXCLUDE_FRAC:.0%} | 편입 {TOP_K} | 지평 {HORIZON}일 | {COST_BP:.0f}bp")
    print("═" * 62)
    print(f"\n평가일 {len(dates)}일, 관측 {sum(len(v) for v in by.values()):,}개")
    print(f"참고 IC — 챔프 {ic_of(by,'champ'):+.4f} | Chronos {ic_of(by,'tsfm'):+.4f}")

    arms = {a: arm_returns(by, a) for a in ("A", "C", "D")}
    label = {"A": "A 현행: 챔프 상위20",
             "C": "C 주가설: 챔프Q1제외+Chronos",
             "D": "D 통제군: Chronos 단독"}

    print(f"\n{'팔':<30}{'수익%/10d':>11}{'연수익%':>9}" +
          "".join(f"{'Sh'+str(c):>7}" for c in COST_GRID))
    print("-" * 68)
    for a in ("A", "C", "D"):
        m, net, _ = stats(arms[a], COST_BP)
        row = f"{label[a]:<30}{m*100:>+11.3f}{net*100:>+9.1f}"
        for c in COST_GRID:
            _, _, sh = stats(arms[a], c)
            row += f"{sh:>7.2f}"
        print(row)

    print(f"\n── 짝지은 차이 (겹침보정 NW t, lag={HORIZON}) ──")
    verdict = {}
    for a, b in (("C", "A"), ("D", "A"), ("C", "D")):
        common = sorted(set(arms[a]) & set(arms[b]))
        diffs = [arms[a][d] - arms[b][d] for d in common]
        t = newey_west_t(diffs)
        verdict[(a, b)] = (sum(diffs) / len(diffs), t, len(diffs))
        note = "   ← 부 판정: 필터 기여" if (a, b) == ("C", "D") else ""
        print(f"  {a} − {b} : {verdict[(a,b)][0]*100:+.3f}%/10d   NW t {t:+6.2f}   (n={len(diffs)}일){note}")

    # 부 판정: 시드별 부호
    sign_ok, sign_tot = 0, 0
    if args.seed_csv:
        print("\n── 시드별 C−A 부호 (부 판정) ──")
        for p in args.seed_csv:
            try:
                sb = load(p)
            except OSError:
                print(f"  {p}: 못 읽음"); continue
            a_, c_ = arm_returns(sb, "A"), arm_returns(sb, "C")
            common = sorted(set(a_) & set(c_))
            if len(common) < HORIZON + 2:
                continue
            d = sum(c_[x] - a_[x] for x in common) / len(common)
            sign_tot += 1
            sign_ok += 1 if d > 0 else 0
            print(f"  {p.split('/')[-1]:<28} Δ {d*100:+.3f}%/10d  {'양수' if d > 0 else '음수'}")

    print("\n" + "═" * 62)
    print("판정")
    print("═" * 62)
    dc, tc, _ = verdict[("C", "A")]
    db, tb, _ = verdict[("C", "D")]
    pass_t = math.isfinite(tc) and tc > 2
    pass_sign = (sign_tot == 0) or (sign_ok >= 2)
    print(f"  주 판정  t(C−A) = {tc:+.2f}  (기준 > 2)  → {'통과' if pass_t else '미달'}")
    if sign_tot:
        print(f"           시드 부호 {sign_ok}/{sign_tot} 양수 (기준 2 이상) → "
              f"{'통과' if pass_sign else '미달'}")
    if pass_t and pass_sign:
        print("\n  ✅ 스크린 통과. 단 홀드아웃 1폴드이므로 채택이 아니다 —")
        print("     운영 반영 전 나머지 폴드로 확장 필요 (docs §7-5).")
    else:
        print("\n  ❌ 기각. 사전등록대로 이 축을 닫는다.")
        print("     비율·종목수·모델을 바꿔 재시도하는 것은 금지돼 있다.")
    print(f"\n  부 판정  t(C−D) = {tb:+.2f}  (챔프 필터가 Chronos 에 더하는 몫)")
    if math.isfinite(tb) and math.isfinite(tc):
        if tb > 0:
            print("           → 필터가 Chronos 를 돕는 방향.")
        else:
            print("           → 필터가 Chronos 를 오히려 깎는 방향 (r5·r4 와 동일).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
