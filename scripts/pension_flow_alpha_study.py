#!/usr/bin/env python3
"""연기금 흐름의 알파 상한 — "국민연금이 뭘 살지 알면" 얼마나 벌 수 있나.

── 착상 (사용자, 2026-08-03) ───────────────────────────────────────────────
국민연금이 뭘 살지 예측할 수 있으면 플러스 알파가 있을 것이다. 큰 매수/매도니까.
연기금 보유주식수를 노드로 놓고 미관측 시점을 예측하면 어떤가.

── 먼저 확인할 것: 상한선 ─────────────────────────────────────────────────
예측 기계를 만들기 전에 **완벽히 안다고 가정했을 때** 얼마나 버는지 잰다.
오라클이 안 벌면 예측할 값어치가 없다. 세 신호를 비교한다:

  oracle_fwd   t+1..t+h 의 연기금 순매수 (미래 정보 — 상한선)
  known_past   t-h..t 의 연기금 순매수 (이미 아는 정보 — 모델 입력에 이미 있음)
  oracle_minus_known  오라클에서 과거분을 뺀 순수 '새 정보'

각 신호로 상위 10%(35종) 롱 바스켓을 만들어 진입 Open[t+1] → 종가 청산.
벤치 = 같은 (진입일, 보유일) 유니버스 평균. 41bp. 랭크/IC 둘 다 본다.

데이터: data/kiwoom_investor_cache/*_20200101_*.csv 의 investor_pension_net_m
(종목별 일별 연기금 순매수, 백만원). 거래대금으로 정규화해 크기 편향을 뺀다.

사용법:
  python scripts/pension_flow_alpha_study.py --holds 5 10 15
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from exit_policy_report import load_prices  # noqa: E402

CACHE = ROOT / "data" / "kiwoom_investor_cache"
K_PICKS = 35
COST_BP, TRADING_DAYS = 41.0, 252
MIN_NAMES = 100          # 그 날짜에 신호가 있는 최소 종목수


def load_pension():
    """{ticker: {date: (연기금순매수_백만, 거래대금_백만)}} — 전체이력 파일만."""
    out = {}
    for path in sorted(CACHE.glob("*_20200101_*.csv")):
        ticker = path.name.split("_")[0]
        rows = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    p = float(r["investor_pension_net_m"])
                    v = float(r["investor_traded_value_m"])
                except (TypeError, ValueError, KeyError):
                    continue
                if math.isfinite(p) and math.isfinite(v) and v > 0:
                    rows[r["date"][:10]] = (p, v)
        if rows:
            out[ticker] = rows
    return out


def build_signals(pens, dates_all, h):
    """날짜별 세 신호. 값은 거래대금 대비 비율(크기 편향 제거)."""
    idx = {d: i for i, d in enumerate(dates_all)}
    fwd, past = {}, {}
    for t, rows in pens.items():
        ds = sorted(rows)
        pos = [idx[d] for d in ds if d in idx]
        if len(pos) < 2 * h + 2:
            continue
        arr = [(d, rows[d]) for d in ds if d in idx]
        n = len(arr)
        # 누적합으로 구간합
        cp = [0.0]
        cv = [0.0]
        for _d, (p, v) in arr:
            cp.append(cp[-1] + p)
            cv.append(cv[-1] + v)
        for j in range(h, n - h):
            d = arr[j][0]
            # 미래: j+1 .. j+h  (진입 다음날부터)
            fp = cp[j + 1 + h] - cp[j + 1]
            fv = cv[j + 1 + h] - cv[j + 1]
            # 과거: j-h+1 .. j
            pp = cp[j + 1] - cp[j + 1 - h]
            pv = cv[j + 1] - cv[j + 1 - h]
            if fv > 0:
                fwd.setdefault(d, {})[t] = fp / fv
            if pv > 0:
                past.setdefault(d, {})[t] = pp / pv
    return fwd, past


def evaluate(sig, panel, h, label, bench_cache):
    daily = []
    for d, row in sig.items():
        if len(row) < MIN_NAMES:
            continue
        key = (d, h)
        if key not in bench_cache:
            vals = []
            for t in row:
                rec = panel.get(t)
                if rec is None:
                    continue
                dates, opens, closes, index = rec
                i = index.get(d)
                if i is None or i + 1 >= len(dates) or i + h >= len(dates) or opens[i + 1] <= 0:
                    continue
                vals.append(closes[i + h] / opens[i + 1] - 1.0)
            bench_cache[key] = (sum(vals) / len(vals)) if len(vals) >= 50 else None
        b = bench_cache[key]
        if b is None:
            continue
        picks = sorted(row, key=lambda t: row[t], reverse=True)[:K_PICKS]
        ex = []
        for t in picks:
            rec = panel.get(t)
            if rec is None:
                continue
            dates, opens, closes, index = rec
            i = index.get(d)
            if i is None or i + 1 >= len(dates) or i + h >= len(dates) or opens[i + 1] <= 0:
                continue
            ex.append(closes[i + h] / opens[i + 1] - 1.0 - b)
        if len(ex) >= K_PICKS // 2:
            daily.append(sum(ex) / len(ex))
    if len(daily) < 20:
        return None
    mu = sum(daily) / len(daily) - COST_BP / 1e4
    sd = statistics.stdev(daily)
    turns = TRADING_DAYS / h
    return (mu / sd * math.sqrt(turns) if sd > 0 else float("nan"),
            mu * turns * 100, len(daily))


def rank_ic(sig, panel, h):
    """신호와 h일 선도수익의 일별 스피어만 상관 평균."""
    ics = []
    for d, row in sig.items():
        if len(row) < MIN_NAMES:
            continue
        pairs = []
        for t, s in row.items():
            rec = panel.get(t)
            if rec is None:
                continue
            dates, opens, closes, index = rec
            i = index.get(d)
            if i is None or i + 1 >= len(dates) or i + h >= len(dates) or opens[i + 1] <= 0:
                continue
            pairs.append((s, closes[i + h] / opens[i + 1] - 1.0))
        if len(pairs) < MIN_NAMES:
            continue
        n = len(pairs)
        rs = {v: k for k, v in enumerate(sorted(range(n), key=lambda k: pairs[k][0]))}
        rr = {v: k for k, v in enumerate(sorted(range(n), key=lambda k: pairs[k][1]))}
        ms = (n - 1) / 2
        num = sum((rs[k] - ms) * (rr[k] - ms) for k in range(n))
        den = math.sqrt(sum((rs[k] - ms) ** 2 for k in range(n))
                        * sum((rr[k] - ms) ** 2 for k in range(n)))
        if den > 0:
            ics.append(num / den)
    return (sum(ics) / len(ics), len(ics)) if ics else (float("nan"), 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--holds", nargs="+", type=int, default=[5, 10, 15])
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()
    print("연기금 수급 적재 중 …", flush=True)
    pens = load_pension()
    dates_all = sorted({d for v in panel.values() for d in v[0]})
    print(f"  종목 {len(pens)} | 가격일자 {len(dates_all)}", flush=True)

    print("\n" + "=" * 78)
    print("연기금 흐름 알파 상한 (상위 10% 롱, 41bp, 벤치=유니버스 평균)")
    print("=" * 78)
    print(f"{'신호':<28}{'보유':>5}{'Sharpe':>9}{'연%':>8}{'RankIC':>9}{'일수':>7}")
    print("-" * 78)
    for h in args.holds:
        fwd, past = build_signals(pens, dates_all, h)
        minus = {}
        for d in fwd:
            if d in past:
                minus[d] = {t: fwd[d][t] - past[d].get(t, 0.0)
                            for t in fwd[d] if t in past[d]}
        bench_cache = {}
        for label, sig in (("oracle_fwd (미래 h일)", fwd),
                           ("known_past (과거 h일)", past),
                           ("oracle − known (순수 신정보)", minus)):
            r = evaluate(sig, panel, h, label, bench_cache)
            ic, nic = rank_ic(sig, panel, h)
            if r:
                print(f"{label:<28}{h:>5}{r[0]:>+9.2f}{r[1]:>+8.1f}{ic:>+9.4f}{r[2]:>7}")
            else:
                print(f"{label:<28}{h:>5}{'.':>9}{'.':>8}{ic:>+9.4f}{'.':>7}")
        print()
    print("판정: oracle_fwd 가 크면 '연기금 예측' 에 값어치가 있다.")
    print("      known_past 와 차이가 작으면 이미 아는 정보로 대부분 설명된다는 뜻이고,")
    print("      그러면 예측 기계를 만들 이유가 없다(모델 입력에 이미 들어 있다).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
