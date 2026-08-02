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

비용 41bp, 회전 = 실측 평균 보유일. 유니버스 = 프로덕션(KRX500 → 상위35).
벤치 = 같은 (진입일, 보유일) 유니버스 평균 (시장중립 근사) — sl_exit_study 와
동일한 자로 잰다.

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
SIGNAL_H, K = 10, 35
COST_BP, TRADING_DAYS = 41.0, 252
MAXD = 30

# 랭크 정책: (라벨, 보유 분위, 캡)
RANK_POLICIES = [
    ("랭크 top20% 캡30", 0.20, 30),   # ← 섀도우 북과 동일 (본명제)
    ("랭크 top20% 캡15", 0.20, 15),   # 랭크 x 시간 하이브리드
    ("랭크 top30% 캡30", 0.30, 30),   # 느슨한 히스테리시스
    ("랭크 top10% 캡30", 0.10, 30),   # 히스테리시스 없음 (진입 분위 이탈 즉시)
]
# 시간 청산 참조 (같은 런에서 같은 자로): (라벨, 다리)
TIME_POLICIES = [
    ("D+15 (채택안)", (15,)),
    ("D+20", (20,)),
    ("D+30", (30,)),
    ("사다리 (구정책)", (1, 2, 3, 5, 10)),
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


def build_top_sets(ens, fracs):
    """ens 날짜별 상위 frac 집합. {frac: {date: set}}, 정렬된 날짜 리스트도 반환."""
    tops = {fr: {} for fr in fracs}
    sig_dates = sorted(ens)
    for dt in sig_dates:
        ranked = sorted(ens[dt], key=lambda t: ens[dt][t], reverse=True)
        n = len(ranked)
        for fr in fracs:
            tops[fr][dt] = set(ranked[:max(1, int(n * fr))])
    return tops, sig_dates


def rank_exit(rec, t, dt, top_by_date, sig_dates, frac, cap):
    """단일 포지션 시뮬. (청산일오프셋, 수익률) 또는 None(데이터 부족)."""
    if rec is None:
        return None
    dates, opens, closes, index = rec
    i = index.get(dt)
    if i is None or i + 1 >= len(dates) or i + cap >= len(dates):
        return None
    entry = opens[i + 1]
    if not (entry > 0):
        return None
    top = top_by_date[frac]
    for d in range(2, cap + 1):
        # 체결일 dates[i+d] 종가, 판정은 직전 세션 dates[i+d-1] 까지의 최신 예측
        decision_day = dates[i + d - 1]
        j = bisect.bisect_right(sig_dates, decision_day) - 1
        if j < 0:
            continue                      # 예측 이전 구간 — 보유 (라이브의 스킵과 동일)
        sd = sig_dates[j]
        if t not in top[sd]:              # top 분위 밖 (예측에 없는 no_score 포함)
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

    fracs = sorted({fr for _, fr, _ in RANK_POLICIES})
    res, seed_n = {}, {}
    for fold in args.folds:
        ens, n = load_ens(fold, args.seeds, args.prefix)
        if ens is None:
            print(f"  {fold}: 런 없음", flush=True)
            continue
        seed_n[fold] = n
        tops, sig_dates = build_top_sets(ens, fracs)
        print(f"  {fold}: 시드 {n}개, {len(sig_dates)}일 채점 …", flush=True)

        picks_by_date = {dt: sorted(ens[dt], key=lambda t: ens[dt][t], reverse=True)[:K]
                         for dt in ens if len(ens[dt]) >= 50}
        bench = {}
        for dt in picks_by_date:
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

        for label, frac, cap in RANK_POLICIES:
            daily, holds = [], []
            for dt, picks in picks_by_date.items():
                pos = []
                for t in picks:
                    ex = rank_exit(panel.get(t), t, dt, tops, sig_dates, frac, cap)
                    if ex is None or (dt, ex[0]) not in bench:
                        continue
                    pos.append(ex[1] - bench[(dt, ex[0])])
                    holds.append(ex[0])
                if len(pos) >= K // 2:
                    daily.append(sum(pos) / len(pos))
            r = score(daily, holds)
            if r:
                res[(fold, label)] = r

        for label, legs in TIME_POLICIES:
            daily, holds = [], []
            for dt, picks in picks_by_date.items():
                pos = []
                for t in picks:
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
          + f" | 판정: T-1 예측 → T 종가 체결 (섀도우 북과 동일)")
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
    print("\n판정: '랭크 top20% 캡30' 이 D+15 를 넘으면 히스테리시스 채택 후보")
    print("      (최악폴드 악화 금지 게이트는 5폴드에서만 확정). 못 넘으면 D+15 유지.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
