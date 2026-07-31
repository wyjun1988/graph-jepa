#!/usr/bin/env python3
"""청산 정책 비교 — 프로덕션의 TP+5% 익절을 포함해서.

── 왜 이게 필요했나 ────────────────────────────────────────────────────────
프로덕션(auto_trader) 청산은 사다리 {1,2,3,5,10} + TP 5% 익절이다. 그런데
exit_policy_report 는 사다리만 모델링하고 TP 가 어디에도 없었다. 실측하면
오늘 챔프 매도 145건 중 89건(61%)이 TP 발동이다. 즉 **실제 매도의 과반을
차지하는 규칙이 어떤 백테스트에도 들어간 적이 없다.**

그 상태로 "D+15 가 사다리보다 낫다"고 말하면 프로덕션과 비교한 게 아니다.
페이퍼에서 사다리 계열(+2.51%)이 백테스트 예상과 반대로 잘 도는 것도
TP 때문일 수 있다. 여기서 그걸 가른다.

── 규칙 (paper_trader.manage_exits 와 동일) ────────────────────────────────
진입가 = Open[t+1] (진입경로). 각 다리는 예정일에 청산하되, 종가가 진입가의
1+TP 를 처음 넘는 날 남은 다리를 전부 턴다. 즉

    다리 청산일 = min(예정일, TP 최초발동일)

── 비용을 공짜로 만들지 않기 ───────────────────────────────────────────────
TP 는 보유기간을 줄인다 → 회전이 늘고 비용이 는다. 그래서 고정 보유일을 쓰지
않고 **실측 평균 보유일**로 회전수를 낸다. 이걸 빼먹으면 TP 가 이득만 있는
것처럼 보인다.

수익은 다리마다 같은 보유구간의 유니버스 평균을 뺀 초과분이다(시장중립).
비용은 왕복 41bp 고정.

사용법:
  python scripts/exit_tp_report.py                      # 있는 폴드 전부
  python scripts/exit_tp_report.py --folds r4 r5
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
SIGNAL_H, TOP_N, K = 10, 100, 20
COST_BP, TRADING_DAYS = 41.0, 252
LADDER = (1, 2, 3, 5, 10)
TP = 0.05
MAXD = 30

# (라벨, 다리, TP)
POLICIES = [
    ("사다리+TP5% (프로덕션)", LADDER, TP),
    ("사다리 (TP없음)",        LADDER, None),
    ("D+10",                  (10,),  None),
    ("D+10 +TP5%",            (10,),  TP),
    ("D+15",                  (15,),  None),
    ("D+15 +TP5%",            (15,),  TP),
    ("D+20",                  (20,),  None),
    ("D+30",                  (30,),  None),
]


def load_ens(fold, seeds, prefix="ens_s"):
    """{date: {ticker: (앙상블예측, 유동성)}} — 신호 지평만."""
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
                    lq = float(row["current_value_ma20_log"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(pr) and math.isfinite(lq):
                    d.setdefault(row["date"], {})[row["ticker"]] = (pr, lq)
        maps.append(d)
    if not maps:
        return None, 0
    dates = sorted(set.intersection(*(set(m) for m in maps)))
    out = {}
    for dt in dates:
        tk = set.intersection(*(set(m[dt]) for m in maps))
        out[dt] = {t: (sum(m[dt][t][0] for m in maps) / len(maps), maps[0][dt][t][1])
                   for t in tk}
    return out, len(maps)


def leg_exits(rec, date, legs, tp):
    """[(청산일, 수익)] — 없으면 None. 규칙은 manage_exits 와 동일."""
    if rec is None:
        return None
    dates, opens, closes, index = rec
    i = index.get(date)
    if i is None or i + 1 >= len(dates):
        return None
    need = max(legs)
    if i + need >= len(dates):
        return None
    entry = opens[i + 1]
    if not (entry > 0):
        return None
    tp_day = None
    if tp is not None:
        for d in range(1, need + 1):
            if closes[i + d] >= entry * (1.0 + tp):
                tp_day = d
                break
    out = []
    for L in legs:
        d = min(L, tp_day) if tp_day is not None else L
        out.append((d, closes[i + d] / entry - 1.0))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r1", "r2", "r3", "r4", "r5"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()
    print(f"  종목 {len(panel)}개\n", flush=True)

    res = {}      # fold -> {policy: (sharpe, net연%, 평균보유일)}
    for fold in args.folds:
        ens, nseed = load_ens(fold, args.seeds, args.prefix)
        if ens is None:
            print(f"  {fold}: 런 없음 — 건너뜀", flush=True)
            continue
        print(f"  {fold}: 시드 {nseed}개, {len(ens)}일 채점 중 …", flush=True)

        # 날짜별 유니버스와 픽
        picks_by_date, uni_by_date = {}, {}
        for dt, rows in ens.items():
            uni = sorted(rows, key=lambda t: rows[t][1], reverse=True)[:TOP_N]
            if len(uni) < 30:
                continue
            uni_by_date[dt] = uni
            picks_by_date[dt] = sorted(uni, key=lambda t: rows[t][0], reverse=True)[:K]

        # 유니버스 벤치마크: (날짜, 보유일) -> 평균 수익. 다리마다 같은 구간을 뺀다.
        bench = {}
        for dt, uni in uni_by_date.items():
            for d in range(1, MAXD + 1):
                vals = []
                for t in uni:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    dates, opens, closes, index = rec
                    i = index.get(dt)
                    if i is None or i + d >= len(dates) or i + 1 >= len(dates):
                        continue
                    e = opens[i + 1]
                    if e > 0:
                        vals.append(closes[i + d] / e - 1.0)
                if len(vals) >= 30:
                    bench[(dt, d)] = sum(vals) / len(vals)

        res[fold] = {}
        for label, legs, tp in POLICIES:
            daily, holds = [], []
            for dt, picks in picks_by_date.items():
                pos = []
                for t in picks:
                    ex = leg_exits(panel.get(t), dt, legs, tp)
                    if ex is None:
                        continue
                    xs = [(r - bench[(dt, d)]) for d, r in ex if (dt, d) in bench]
                    if len(xs) == len(ex):
                        pos.append(sum(xs) / len(xs))
                        holds.append(sum(d for d, _ in ex) / len(ex))
                if len(pos) >= K // 2:
                    daily.append(sum(pos) / len(pos))
            if len(daily) < 20 or not holds:
                continue
            hold = sum(holds) / len(holds)
            m = sum(daily) / len(daily)
            sd = statistics.stdev(daily)
            turns = TRADING_DAYS / hold
            net = (m - COST_BP / 1e4) * turns
            sh = net / (sd * math.sqrt(turns)) if sd > 0 else float("nan")
            res[fold][label] = (sh, net * 100, hold)

    folds = [f for f in args.folds if f in res]
    if not folds:
        print("채점된 폴드가 없습니다.")
        return 1

    def avg(v):
        v = [x for x in v if math.isfinite(x)]
        return sum(v) / len(v) if v else float("nan")

    print("\n" + "=" * 78)
    print(f"청산 정책 x {len(folds)}폴드 — Sharpe (41bp 차감, 회전은 실측 보유일)")
    print("=" * 78)
    print(f"{'정책':<22}" + "".join(f"{f:>7}" for f in folds)
          + f"{'평균':>8}{'연%':>7}{'보유일':>7}{'양수':>6}")
    print("-" * 78)
    for label, _, _ in POLICIES:
        row = [res[f].get(label, (float("nan"),) * 3)[0] for f in folds]
        if not any(math.isfinite(x) for x in row):
            continue
        ann = avg([res[f].get(label, (0, float("nan"), 0))[1] for f in folds])
        hd = avg([res[f].get(label, (0, 0, float("nan")))[2] for f in folds])
        pos = sum(1 for x in row if math.isfinite(x) and x > 0)
        print(f"{label:<22}" + "".join(f"{x:>+7.2f}" for x in row)
              + f"{avg(row):>+8.2f}{ann:>+7.1f}{hd:>7.1f}{pos:>4}/{len(folds)}")

    print("\n── 프로덕션 대비 (사다리+TP5%) ──")
    base = [res[f].get("사다리+TP5% (프로덕션)", (float("nan"),))[0] for f in folds]
    for label, _, _ in POLICIES:
        if label.startswith("사다리+TP"):
            continue
        row = [res[f].get(label, (float("nan"),))[0] for f in folds]
        d = [a - b for a, b in zip(row, base)
             if math.isfinite(a) and math.isfinite(b)]
        if not d:
            continue
        win = sum(1 for x in d if x > 0)
        print(f"  {label:<22} Δ{avg(d):>+6.2f}   {win}/{len(d)} 폴드 우세"
              f"{'   ← 전 폴드 우세' if win == len(d) else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
