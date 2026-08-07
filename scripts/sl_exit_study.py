#!/usr/bin/env python3
"""손절(SL) 연구 — TP 가 해로웠다면, 반대 방향(진 종목 자르기)은 이로운가.

── 가설 ────────────────────────────────────────────────────────────────────
챔프의 랭킹력은 Q1(-0.949, 지는 종목 회피)에 몰려 있고 Q5(+0.358)는 약하다.
TP+5% 는 이긴 종목을 일찍 팔아 프로덕션 유니버스에서 부호를 뒤집었다(+0.45→-0.27).
대칭 논리로, 손절은 '지는 종목' — 모델이 잘 가려내는 쪽 — 을 자른다.
가격 기반 SL 이 모델의 Q1 회피와 겹치는 정보라면 무익, 보완이면 유익.

── 규칙 ────────────────────────────────────────────────────────────────────
진입가 = Open[t+1]. 종가가 entry*(1-SL) 를 처음 깨는 날 남은 다리 전부 청산.
그 외는 예정일 청산. 비용은 41bp, 회전은 실측 평균 보유일.
유니버스 = 프로덕션(KRX500 전체 → 상위35, 동일가중).

사용법:
  python scripts/sl_exit_study.py --folds r4 r5
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
    "r4": "fold1_20241106_to_20250908",
    "r5": "fold1_20250905_to_20260710",
    "r3": "fold1_20240104_to_20241107",
    "r2": "fold1_20230307_to_20240105",
    "r1": "fold1_20220510_to_20230306",
}
SIGNAL_H, K = 10, 35
COST_BP, TRADING_DAYS = 41.0, 252
LADDER = (1, 2, 3, 5, 10)
MAXD = 30

# (라벨, 다리, TP, SL)
POLICIES = [
    ("D+15",            (15,),  None, None),
    ("D+15 SL-3%",      (15,),  None, 0.03),
    ("D+15 SL-5%",      (15,),  None, 0.05),
    ("D+15 SL-8%",      (15,),  None, 0.08),
    ("D+15 SL-12%",     (15,),  None, 0.12),
    ("D+15 TP+5% (참조)", (15,),  0.05, None),
    ("D+20",            (20,),  None, None),
    ("D+20 SL-5%",      (20,),  None, 0.05),
    ("D+20 SL-8%",      (20,),  None, 0.08),
    ("사다리 (참조)",      LADDER, None, None),
    ("사다리 SL-8%",     LADDER, None, 0.08),
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
                    lq = float(row["current_value_ma20_log"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(pr) and math.isfinite(lq):
                    d.setdefault(row["date"], {})[row["ticker"]] = (pr, lq)
        maps.append(d)
    if not maps:
        return None, 0
    dates = sorted(set.intersection(*(set(m) for m in maps)))
    return ({dt: {t: (sum(m[dt][t][0] for m in maps) / len(maps),
                      maps[0][dt][t][1])
                  for t in set.intersection(*(set(m[dt]) for m in maps))}
             for dt in dates}, len(maps))


def leg_exits(rec, date, legs, tp, sl):
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
    trig = None
    for d in range(1, need + 1):
        c = closes[i + d]
        if tp is not None and c >= entry * (1 + tp):
            trig = d
            break
        if sl is not None and c <= entry * (1 - sl):
            trig = d
            break
    return [(min(L, trig) if trig else L,
             closes[i + (min(L, trig) if trig else L)] / entry - 1.0)
            for L in legs]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r4", "r5"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--json", default="", help="판정기용 결과 JSON 경로")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()

    res = {}
    for fold in args.folds:
        ens, n = load_ens(fold, args.seeds, args.prefix)
        if ens is None:
            print(f"  {fold}: 런 없음", flush=True)
            continue
        print(f"  {fold}: 시드 {n}개, {len(ens)}일 채점 …", flush=True)

        picks_by_date = {}
        for dt, rows in ens.items():
            if len(rows) >= 50:
                picks_by_date[dt] = sorted(rows, key=lambda t: rows[t][0],
                                           reverse=True)[:K]
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

        for label, legs, tp, sl in POLICIES:
            daily, holds = [], []
            for dt, picks in picks_by_date.items():
                pos = []
                for t in picks:
                    ex = leg_exits(panel.get(t), dt, legs, tp, sl)
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
            res[(fold, label)] = (net / (sd * math.sqrt(turns)) if sd > 0
                                  else float("nan"), net * 100, hold)

    folds = [f for f in args.folds if any(k[0] == f for k in res)]
    print("\n" + "=" * 74)
    print("손절 연구 (프로덕션 유니버스, 41bp, 회전=실측 보유일)")
    print("=" * 74)
    print(f"{'정책':<18}" + "".join(f"{f + ' Sh':>8}{f + ' 연%':>8}{f + ' 보유':>7}"
                                   for f in folds) + f"{'평균Sh':>8}")
    print("-" * 74)
    for label, _, _, _ in POLICIES:
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
    if args.json:
        import json as _json
        Path(args.json).write_text(_json.dumps({
            "study": "sl_exit", "prefix": args.prefix, "folds": folds,
            "rows": {label: {f: res[(f, label)][0] for f in folds
                             if (f, label) in res}
                     for label, _, _, _ in POLICIES},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[json] {args.json}")
    print("\n판정: SL 이 D+15 를 넘으면 '가격기반 손절이 모델 Q1회피를 보완'.")
    print("      못 넘으면 겹치는 정보 — 단순 D+15 유지가 답이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
