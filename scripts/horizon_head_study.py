#!/usr/bin/env python3
"""지평 헤드 연구 — 어느 헤드(1/2/3/5/10일)가 D+10/15/20 보유에 가장 좋은 신호인가.

── 왜 ─────────────────────────────────────────────────────────────────────
두 가지 질문이 걸려 있다.

1) 사용자 질문(7/29): "혹시 너무 먼 미래를 예측하나? 하루 다음꺼를 예측하게
   해보면 어때?" — 지금까지 신호는 항상 h10 헤드였다. h1 이 더 좋다면 학습
   없이도 신호 교체로 개선된다.

2) 내일 학습 계획의 관문: 청산 연구로 보유가 D+15~30 으로 갈 판인데 모델은
   h10 까지만 예측한다. h15/h20 헤드를 새로 학습할 가치가 있는지는
   "예측지평이 보유기간에 가까울수록 좋은가"에 달렸다. h1→h5→h10 이 단조로
   좋아지면 h15/h20 학습(P0)이 정당화되고, h5≈h10 이면 포화라 학습 불필요.

── 방법 ────────────────────────────────────────────────────────────────────
신호 = 각 헤드의 앙상블(시드 평균) 예측으로 상위 K 선정.
보유 = 단일 D+10 / D+15 / D+20 (전일 종가 대비 유니버스 평균 차감, 41bp).
유니버스 = 프로덕션 프리셋(KRX500 전체 → 상위35, 동일가중; conf 는 별도 연구).
추가로 헤드 평균(h1..h10, h5+h10)도 신호 후보로 넣는다 — 헤드 앙상블이 단일
헤드보다 나은지.

사용법:
  python scripts/horizon_head_study.py --folds r4 r5
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

NODE_EVAL = ROOT / "reports" / "walk_forward" / "node_eval"
FOLDS = {
    "r1": "fold1_20220510_to_20230306",
    "r2": "fold1_20230307_to_20240105",
    "r3": "fold1_20240104_to_20241107",
    "r4": "fold1_20241106_to_20250908",
    "r5": "fold1_20250905_to_20260710",
}
HEADS = (1, 2, 3, 5, 10)
HOLDS = (10, 15, 20)
K = 35
COST_BP, TRADING_DAYS = 41.0, 252


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


def load_fold(fold, seeds, prefix="ens_s"):
    """{date: {ticker: {"liq":…, h: 앙상블예측}}} — 전 지평."""
    per_seed = []
    for s in seeds:
        p = NODE_EVAL / f"{prefix}{s}_{FOLDS[fold]}" / "return_1d_forecasts.csv"
        if not p.exists():
            continue
        d = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                h = int(row["horizon"])
                if h not in HEADS:
                    continue
                try:
                    pr = float(row["prediction_entry_path_return"])
                    lq = float(row["current_value_ma20_log"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(pr):
                    d.setdefault(row["date"], {}).setdefault(row["ticker"],
                                                            {"liq": lq})[h] = pr
        per_seed.append(d)
    if not per_seed:
        return None, 0
    dates = sorted(set.intersection(*(set(d) for d in per_seed)))
    out = {}
    for dt in dates:
        tk = set.intersection(*(set(d[dt]) for d in per_seed))
        row = {}
        for t in tk:
            recs = [d[dt][t] for d in per_seed]
            if not all(all(h in r for h in HEADS) for r in recs):
                continue
            row[t] = {"liq": recs[0]["liq"]}
            for h in HEADS:
                row[t][h] = sum(r[h] for r in recs) / len(recs)
        if len(row) >= 50:
            out[dt] = row
    return out, len(per_seed)


def hold_return(rec, date, hold):
    if rec is None:
        return None
    dates, opens, closes, index = rec
    i = index.get(date)
    if i is None or i + 1 >= len(dates) or i + hold >= len(dates):
        return None
    e = opens[i + 1]
    return closes[i + hold] / e - 1.0 if e > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r4", "r5"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()

    # 신호 후보: 단일 헤드 + 두 가지 헤드 평균
    signals = [(f"h{h}", (h,)) for h in HEADS] + \
              [("mean(h1..h10)", HEADS), ("mean(h5,h10)", (5, 10))]

    all_res = {}          # (fold, sig, hold) -> (sharpe, gross%/hold, ic)
    for fold in args.folds:
        ens, n = load_fold(fold, args.seeds, args.prefix)
        if ens is None:
            print(f"  {fold}: 런 없음", flush=True)
            continue
        print(f"  {fold}: 시드 {n}개, {len(ens)}일 채점 …", flush=True)

        # 보유기간별 실현수익 캐시 + 유니버스 평균
        rz, uni_mean = {}, {}
        for dt, rows in ens.items():
            for hold in HOLDS:
                vals = {}
                for t in rows:
                    r = hold_return(panel.get(t), dt, hold)
                    if r is not None:
                        vals[t] = r
                if len(vals) >= 50:
                    rz[(dt, hold)] = vals
                    uni_mean[(dt, hold)] = sum(vals.values()) / len(vals)

        for signame, heads in signals:
            for hold in HOLDS:
                daily, ics = [], []
                for dt, rows in ens.items():
                    if (dt, hold) not in rz:
                        continue
                    vals = rz[(dt, hold)]
                    scored = [(sum(rows[t][h] for h in heads) / len(heads), t)
                              for t in rows if t in vals]
                    if len(scored) < K + 15:
                        continue
                    scored.sort(reverse=True)
                    picks = [t for _, t in scored[:K]]
                    daily.append(sum(vals[t] for t in picks) / K
                                 - uni_mean[(dt, hold)])
                    c = pearson([s for s, _ in scored], [vals[t] for _, t in scored])
                    if math.isfinite(c):
                        ics.append(c)
                if len(daily) < 20:
                    continue
                m = sum(daily) / len(daily)
                sd = statistics.stdev(daily)
                turns = TRADING_DAYS / hold
                net = (m - COST_BP / 1e4) * turns
                sh = net / (sd * math.sqrt(turns)) if sd > 0 else float("nan")
                all_res[(fold, signame, hold)] = (sh, m * 100, sum(ics) / len(ics))

    folds = [f for f in args.folds if any(k[0] == f for k in all_res)]

    for hold in HOLDS:
        print("\n" + "=" * 72)
        print(f"보유 D+{hold} — 신호(헤드)별 Sharpe / 총초과%/보유 / IC(같은 지평)")
        print("=" * 72)
        print(f"{'신호':<15}" + "".join(
            f"{f + ' Sh':>9}{f + ' %':>8}{f + ' IC':>9}" for f in folds)
            + f"{'평균Sh':>8}")
        print("-" * 72)
        for signame, _ in signals:
            cells, shs = "", []
            for f in folds:
                r = all_res.get((f, signame, hold))
                if r:
                    cells += f"{r[0]:>+9.2f}{r[1]:>+8.3f}{r[2]:>+9.4f}"
                    shs.append(r[0])
                else:
                    cells += f"{'.':>9}{'.':>8}{'.':>9}"
            if shs:
                print(f"{signame:<15}{cells}{sum(shs)/len(shs):>+8.2f}")

    print("\n판정 안내:")
    print("  h1 이 최고 → 신호를 h1 로 교체 (학습 불필요, 사용자 가설 채택)")
    print("  h10 이 최고 & h5→h10 상승 → 지평이 보유에 가까울수록 좋다")
    print("    → 내일 h15/h20 헤드 학습(P0)이 정당화된다")
    print("  h5 ≈ h10 → 포화. 헤드 추가 학습의 기대효과 낮음")
    return 0


if __name__ == "__main__":
    sys.exit(main())
