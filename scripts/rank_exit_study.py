#!/usr/bin/env python3
"""랭크 청산 연구 — Buy-Hold-Spread 히스테리시스는 시간 청산을 이기는가.

── 가설 (AFA 'The Expected Returns on ML Strategies') ─────────────────────
상위 10%에 들면 사고, 상위 20%에서 떨어질 때만 판다. 시간(D+n)·가격(TP/SL)이
아닌 제3의 청산 축. 모델이 아직 좋다고 하는 종목을 미리 팔지 않고, 나빠졌다고
하는 종목을 캘린더보다 일찍 내보낸다.

── 규칙 (섀도우 북 rankexit 과 동일 — paper_trader.py 2026-08-02) ──────────
진입가 = Open[t+1] (신호일 t 종가 기준 예측, 상위 35). 이후 매 세션 s 의
종가 청산 판정은 **s 직전 세션의 예측**으로 한다(신호 T-1 → 체결 T, 인과).
직전 세션 예측에서 보유 분위(top N%) 밖이면 그날 종가 청산. 예측에 없어진
종목(no_score)도 청산. 캡 D+30. 진입 다음날(d=1)은 진입 신호 자신이 판정
기준이라 구조상 청산 불가 — 최소 보유 2세션.

── 지평 변형 (2026-08-02 확장) ─────────────────────────────────────────────
진입 랭킹과 청산 랭킹의 신호원을 분리해 본다. 청산 질문은 "다음 10일이
좋은가"(h10)가 아니라 "지금 나가야 하나"라서 단기(h5)가 더 맞을 수 있고,
mean(h5,h10)은 8/1 연구에서 진입 신호의 공짜 개선(+0.09)이었다.
D+15 진입mean 참조 행으로 진입 효과와 청산 효과를 분해한다.

비용 41bp, 회전 = 실측 평균 보유일. 유니버스 = 프로덕션(KRX500 → 상위35).
벤치 = 같은 (진입일, 보유일) 유니버스 평균 — sl_exit_study 와 동일한 자.

사용법:
  python scripts/rank_exit_study.py --folds r5 r4
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
K = 35
COST_BP, TRADING_DAYS = 41.0, 252
MAXD = 30
HORIZONS = (5, 10)                # 적재할 지평. mean = (h5+h10)/2 (8/1 연구 관행)

# 랭크 정책: (라벨, 진입 신호, 청산 신호, 보유 분위, 캡)
#   신호원: "h10" | "h5" | "mean"
RANK_POLICIES = [
    ("랭크20 h10/h10",   "h10",  "h10",  0.20, 30),  # ← 섀도우 북 동일 (사전등록 주판정)
    ("랭크20 exit-h5",   "h10",  "h5",   0.20, 30),  # 청산만 단기 지평
    ("랭크20 exit-mean", "h10",  "mean", 0.20, 30),  # 청산만 mean(h5,h10)
    ("랭크20 in&out-mean", "mean", "mean", 0.20, 30),  # 진입·청산 모두 mean (조합)
    ("랭크15 h10/h10",   "h10",  "h10",  0.15, 30),  # 폭 스윕
    ("랭크25 h10/h10",   "h10",  "h10",  0.25, 30),
    ("랭크30 h10/h10",   "h10",  "h10",  0.30, 30),
    ("랭크10 h10/h10",   "h10",  "h10",  0.10, 30),  # 무히스테리시스
    ("랭크20 캡15",      "h10",  "h10",  0.20, 15),  # 랭크 x 시간 하이브리드
]
# 시간 청산 참조: (라벨, 진입 신호, 다리)
TIME_POLICIES = [
    ("D+15 (채택안)",     "h10",  (15,)),
    ("D+15 진입mean",    "mean", (15,)),   # 진입 효과 분리용 (8/1 의 +0.09 재현)
    ("D+20",            "h10",  (20,)),
    ("D+30",            "h10",  (30,)),
    ("사다리 (구정책)",     "h10",  (1, 2, 3, 5, 10)),
]


def load_ens_multi(fold, seeds, prefix="ens_s"):
    """지평별 시드 평균 예측. {h: {date: {ticker: pred}}}, 시드 수."""
    maps = {h: [] for h in HORIZONS}
    want = {str(h) for h in HORIZONS}
    n = 0
    for s in seeds:
        p = NODE_EVAL / f"{prefix}{s}_{FOLDS[fold]}" / "return_1d_forecasts.csv"
        if not p.exists():
            continue
        d = {h: {} for h in HORIZONS}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if row["horizon"] not in want:
                    continue
                try:
                    pr = float(row["prediction_entry_path_return"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(pr):
                    d[int(row["horizon"])].setdefault(row["date"], {})[row["ticker"]] = pr
        for h in HORIZONS:
            maps[h].append(d[h])
        n += 1
    if n == 0:
        return None, 0
    out = {}
    for h in HORIZONS:
        ms = maps[h]
        dates = sorted(set.intersection(*(set(m) for m in ms)))
        out[h] = {dt: {t: sum(m[dt][t] for m in ms) / len(ms)
                       for t in set.intersection(*(set(m[dt]) for m in ms))}
                  for dt in dates}
    return out, n


def build_signal(ens, src):
    """신호원 이름 -> {date: {ticker: score}}. mean 은 h5·h10 공통 종목만."""
    if src == "h10":
        return ens[10]
    if src == "h5":
        return ens[5]
    sig = {}
    for dt in set(ens[5]) & set(ens[10]):
        common = set(ens[5][dt]) & set(ens[10][dt])
        sig[dt] = {t: (ens[5][dt][t] + ens[10][dt][t]) / 2 for t in common}
    return sig


def build_top_sets(sig, frac):
    """날짜별 상위 frac 집합과 정렬된 날짜 리스트."""
    tops, sig_dates = {}, sorted(sig)
    for dt in sig_dates:
        ranked = sorted(sig[dt], key=lambda t: sig[dt][t], reverse=True)
        tops[dt] = set(ranked[:max(1, int(len(ranked) * frac))])
    return tops, sig_dates


def rank_exit(rec, t, dt, tops, sig_dates, cap):
    if rec is None:
        return None
    dates, opens, closes, index = rec
    i = index.get(dt)
    if i is None or i + 1 >= len(dates) or i + cap >= len(dates):
        return None
    entry = opens[i + 1]
    if not (entry > 0):
        return None
    for d in range(2, cap + 1):
        decision_day = dates[i + d - 1]           # 판정 = 직전 세션까지의 최신 예측
        j = bisect.bisect_right(sig_dates, decision_day) - 1
        if j < 0:
            continue                              # 예측 이전 구간 — 보유
        if t not in tops[sig_dates[j]]:           # 분위 밖 (no_score 포함)
            return (d, closes[i + d] / entry - 1.0)
    return (cap, closes[i + cap] / entry - 1.0)


def time_exit(rec, dt, legs):
    if rec is None:
        return None
    dates, opens, closes, index = rec
    i = index.get(dt)
    need = max(legs)
    if i is None or i + 1 >= len(dates) or i + need >= len(dates):
        return None
    entry = opens[i + 1]
    if not (entry > 0):
        return None
    return [(L, closes[i + L] / entry - 1.0) for L in legs]


def score(daily, holds):
    if len(daily) < 20 or not holds:
        return None
    hold = sum(holds) / len(holds)
    m = sum(daily) / len(daily)
    sd = statistics.stdev(daily)
    turns = TRADING_DAYS / hold
    net = (m - COST_BP / 1e4) * turns
    return (net / (sd * math.sqrt(turns)) if sd > 0 else float("nan"),
            net * 100, hold)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r5", "r4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()

    res, seed_n = {}, {}
    for fold in args.folds:
        ens, n = load_ens_multi(fold, args.seeds, args.prefix)
        if ens is None:
            print(f"  {fold}: 런 없음", flush=True)
            continue
        seed_n[fold] = n
        signals = {src: build_signal(ens, src) for src in ("h10", "h5", "mean")}
        print(f"  {fold}: 시드 {n}개, {len(signals['h10'])}일 채점 …", flush=True)

        picks = {src: {dt: sorted(sig[dt], key=lambda t: sig[dt][t], reverse=True)[:K]
                       for dt in sig if len(sig[dt]) >= 50}
                 for src, sig in signals.items() if src in {"h10", "mean"}}
        # 벤치는 유니버스(h10 기준) 평균 — 진입 신호와 무관하게 같은 자
        bench = {}
        for dt in picks["h10"]:
            for d in range(1, MAXD + 1):
                vals = []
                for t in signals["h10"][dt]:
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

        top_cache = {}
        for label, esrc, xsrc, frac, cap in RANK_POLICIES:
            key = (xsrc, frac)
            if key not in top_cache:
                top_cache[key] = build_top_sets(signals[xsrc], frac)
            tops, sig_dates = top_cache[key]
            daily, holds = [], []
            for dt, pk in picks[esrc].items():
                pos = []
                for t in pk:
                    ex = rank_exit(panel.get(t), t, dt, tops, sig_dates, cap)
                    if ex is None or (dt, ex[0]) not in bench:
                        continue
                    pos.append(ex[1] - bench[(dt, ex[0])])
                    holds.append(ex[0])
                if len(pos) >= K // 2:
                    daily.append(sum(pos) / len(pos))
            r = score(daily, holds)
            if r:
                res[(fold, label)] = r

        for label, esrc, legs in TIME_POLICIES:
            daily, holds = [], []
            for dt, pk in picks[esrc].items():
                pos = []
                for t in pk:
                    ex = time_exit(panel.get(t), dt, legs)
                    if ex is None:
                        continue
                    xs = [(r2 - bench[(dt, d)]) for d, r2 in ex if (dt, d) in bench]
                    if len(xs) == len(ex):
                        pos.append(sum(xs) / len(xs))
                        holds.append(sum(d for d, _ in ex) / len(ex))
                if len(pos) >= K // 2:
                    daily.append(sum(pos) / len(pos))
            r = score(daily, holds)
            if r:
                res[(fold, label)] = r

    folds = [f for f in args.folds if any(k[0] == f for k in res)]
    all_pol = [p[0] for p in RANK_POLICIES] + [p[0] for p in TIME_POLICIES]
    print("\n" + "=" * 78)
    print("랭크 청산 연구 (프로덕션 유니버스, 41bp, 회전=실측 보유일)")
    print("  시드: " + ", ".join(f"{f}={seed_n.get(f, 0)}" for f in folds)
          + " | 판정: T-1 예측 → T 종가 체결 (섀도우 북과 동일)")
    print("=" * 78)
    print(f"{'정책':<18}" + "".join(f"{f + ' Sh':>8}{f + ' 연%':>8}{f + ' 보유':>7}"
                                   for f in folds) + f"{'평균Sh':>8}")
    print("-" * 78)
    for label in all_pol:
        cells, shs = "", []
        for f in folds:
            r = res.get((f, label))
            if r:
                cells += f"{r[0]:>+8.2f}{r[1]:>+8.1f}{r[2]:>7.1f}"
                shs.append(r[0])
            else:
                cells += f"{'.':>8}{'.':>8}{'.':>7}"
        if shs:
            print(f"{label:<18}{cells}{sum(shs)/len(shs):>+8.2f}")
    print("\n판정: 주판정 = '랭크20 h10/h10' vs 'D+15 (채택안)' (사전등록 2026-08-02).")
    print("      exit-h5/mean·in&out-mean 은 탐색 — 주판정과 별도로 읽는다.")
    print("      최악폴드 악화 금지 게이트는 5폴드에서만 확정.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
