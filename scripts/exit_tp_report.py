#!/usr/bin/env python3
"""청산 정책 비교 — 프로덕션의 TP+5% 익절과 프로덕션 유니버스까지 반영해서.

── 왜 이게 필요했나 ────────────────────────────────────────────────────────
두 가지가 백테스트에 빠져 있었다.

1) TP+5% 익절. 프로덕션(auto_trader) 청산은 사다리 {1,2,3,5,10} + TP 5% 인데
   exit_policy_report 는 사다리만 모델링했다. 실측하면 챔프 매도의 61%가 TP
   발동이다 — 실제 매도의 과반을 차지하는 규칙이 어떤 백테스트에도 없었다.

2) 유니버스. 페이퍼 픽 35종목은 35/35 가 KRX500 매니페스트 안에 있지만
   유동성 상위100 에는 1개뿐이다. 즉
     프로덕션 : KRX500 전체(날짜당 ~442) -> 상위 10% (35~44종목), confidence 가중
     백테스트 : 유동성 상위 100 으로 제한 -> 상위 20, 동일가중
   **모집단도 픽 수도 가중도 달랐다.** 다른 조건에서 잰 결론을 옮겨 붙일 수 없다.

그래서 유니버스·픽수·가중을 인자로 빼고 양쪽을 같은 자로 잰다.

── 청산 규칙 (paper_trader.manage_exits 와 동일) ───────────────────────────
진입가 = Open[t+1]. 각 다리는 예정일에 청산하되, 종가가 진입가의 1+TP 를 처음
넘는 날 남은 다리를 전부 턴다:  다리 청산일 = min(예정일, TP 최초발동일)

── 비용을 공짜로 만들지 않기 ───────────────────────────────────────────────
TP 는 보유기간을 줄인다 -> 회전이 늘고 비용이 는다. 고정 보유일이 아니라
**실측 평균 보유일**로 회전수를 낸다. 빼먹으면 TP 가 이득만 있는 것처럼 보인다.

수익은 다리마다 같은 보유구간의 유니버스 평균을 뺀 초과분이다(시장중립).
유니버스=all 이면 그 벤치마크가 곧 KRX500 동일가중이고, 페이퍼가 2026-08-01 부터
쓰는 헤지와 같은 자다.

── confidence 가중에 대하여 ────────────────────────────────────────────────
프로덕션은 exp(-epi_z) 로 가중한다. 예측 파일에 epistemic 열이 없어서 **시드간
예측 표준편차**를 epistemic 대용으로 쓴다(날짜별 횡단면 z 화 후 exp(-z)).
정확한 재현이 아니라 근사다 — 가중 유무의 방향만 보는 용도다.

사용법:
  python scripts/exit_tp_report.py --preset prod       # KRX500 전체, 상위35, conf가중
  python scripts/exit_tp_report.py --preset backtest   # 유동성상위100, 상위20, 동일가중
  python scripts/exit_tp_report.py --compare-presets   # 둘 다 (기본)
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
SIGNAL_H = 10
COST_BP, TRADING_DAYS = 41.0, 252
LADDER = (1, 2, 3, 5, 10)
TP = 0.05
MAXD = 30

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

PRESETS = {
    # 이름:        (유동성상위 N 또는 None=전체, 픽수, 가중)
    "prod":     (None, 35, "conf"),
    "backtest": (100,  20, "equal"),
}


def load_ens(fold, seeds, prefix="ens_s"):
    """{date: {ticker: (평균예측, 유동성, 시드간표준편차)}}"""
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
        row = {}
        for t in tk:
            vs = [m[dt][t][0] for m in maps]
            sd = statistics.stdev(vs) if len(vs) > 1 else 0.0
            row[t] = (sum(vs) / len(vs), maps[0][dt][t][1], sd)
        out[dt] = row
    return out, len(maps)


def leg_exits(rec, date, legs, tp):
    """[(청산일, 수익)] — 없으면 None."""
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
    return [(min(L, tp_day) if tp_day is not None else L,
             closes[i + (min(L, tp_day) if tp_day is not None else L)] / entry - 1.0)
            for L in legs]


def weights_for(picks, rows, mode):
    """프로덕션은 exp(-epi_z) 가중. 시드간 표준편차를 epistemic 대용으로 쓴다."""
    if mode != "conf" or len(picks) < 3:
        return {t: 1.0 / len(picks) for t in picks}
    sds = [rows[t][2] for t in picks]
    m, s = sum(sds) / len(sds), (statistics.stdev(sds) if len(sds) > 1 else 0.0)
    if not s:
        return {t: 1.0 / len(picks) for t in picks}
    w = {t: math.exp(-((rows[t][2] - m) / s)) for t in picks}
    tot = sum(w.values())
    return {t: v / tot for t, v in w.items()}


def score_fold(ens, panel, liq_top, k, wmode):
    """{정책: (sharpe, 연%, 평균보유일)}"""
    picks_by_date, uni_by_date = {}, {}
    for dt, rows in ens.items():
        uni = sorted(rows, key=lambda t: rows[t][1], reverse=True)
        if liq_top:
            uni = uni[:liq_top]
        if len(uni) < max(30, k):
            continue
        uni_by_date[dt] = uni
        picks_by_date[dt] = sorted(uni, key=lambda t: rows[t][0], reverse=True)[:k]

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
                if opens[i + 1] > 0:
                    vals.append(closes[i + d] / opens[i + 1] - 1.0)
            if len(vals) >= 30:
                bench[(dt, d)] = sum(vals) / len(vals)

    out = {}
    for label, legs, tp in POLICIES:
        daily, holds = [], []
        for dt, picks in picks_by_date.items():
            rows = ens[dt]
            w = weights_for(picks, rows, wmode)
            acc, wsum = 0.0, 0.0
            for t in picks:
                ex = leg_exits(panel.get(t), dt, legs, tp)
                if ex is None:
                    continue
                xs = [(r - bench[(dt, d)]) for d, r in ex if (dt, d) in bench]
                if len(xs) != len(ex):
                    continue
                acc += w[t] * (sum(xs) / len(xs))
                wsum += w[t]
                holds.append(sum(d for d, _ in ex) / len(ex))
            if wsum > 0.5:
                daily.append(acc / wsum)
        if len(daily) < 20 or not holds:
            continue
        hold = sum(holds) / len(holds)
        m = sum(daily) / len(daily)
        sd = statistics.stdev(daily)
        turns = TRADING_DAYS / hold
        net = (m - COST_BP / 1e4) * turns
        out[label] = (net / (sd * math.sqrt(turns)) if sd > 0 else float("nan"),
                      net * 100, hold)
    return out


def report(title, res, folds):
    def avg(v):
        v = [x for x in v if math.isfinite(x)]
        return sum(v) / len(v) if v else float("nan")
    print("\n" + "=" * 76)
    print(title)
    print("=" * 76)
    print(f"{'정책':<22}" + "".join(f"{f:>7}" for f in folds)
          + f"{'평균':>8}{'연%':>7}{'보유일':>7}{'양수':>6}")
    print("-" * 76)
    for label, _, _ in POLICIES:
        row = [res[f].get(label, (float("nan"),) * 3)[0] for f in folds]
        if not any(math.isfinite(x) for x in row):
            continue
        ann = avg([res[f].get(label, (0, float("nan"), 0))[1] for f in folds])
        hd = avg([res[f].get(label, (0, 0, float("nan")))[2] for f in folds])
        pos = sum(1 for x in row if math.isfinite(x) and x > 0)
        print(f"{label:<22}" + "".join(f"{x:>+7.2f}" for x in row)
              + f"{avg(row):>+8.2f}{ann:>+7.1f}{hd:>7.1f}{pos:>4}/{len(folds)}")
    base = [res[f].get("사다리+TP5% (프로덕션)", (float("nan"),))[0] for f in folds]
    print("\n  프로덕션 청산(사다리+TP5%) 대비:")
    for label, _, _ in POLICIES:
        if label.startswith("사다리+TP"):
            continue
        row = [res[f].get(label, (float("nan"),))[0] for f in folds]
        d = [a - b for a, b in zip(row, base) if math.isfinite(a) and math.isfinite(b)]
        if not d:
            continue
        win = sum(1 for x in d if x > 0)
        print(f"    {label:<22} Δ{avg(d):>+6.2f}   {win}/{len(d)} 폴드 우세"
              f"{'   ← 전 폴드' if win == len(d) else ''}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r1", "r2", "r3", "r4", "r5"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None)
    args = ap.parse_args()
    presets = [args.preset] if args.preset else ["prod", "backtest"]

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()
    print(f"  종목 {len(panel)}개\n", flush=True)

    loaded = {}
    for fold in args.folds:
        ens, n = load_ens(fold, args.seeds, args.prefix)
        if ens is None:
            print(f"  {fold}: 런 없음 — 건너뜀", flush=True)
            continue
        loaded[fold] = (ens, n)
        print(f"  {fold}: 시드 {n}개, {len(ens)}일", flush=True)
    if not loaded:
        print("채점할 폴드가 없습니다.")
        return 1
    folds = [f for f in args.folds if f in loaded]

    results = {}
    for name in presets:
        liq_top, k, wmode = PRESETS[name]
        print(f"\n[{name}] 유니버스 {'전체' if not liq_top else f'유동성상위{liq_top}'}"
              f" → 상위{k}, {'confidence 가중' if wmode == 'conf' else '동일가중'}",
              flush=True)
        res = {}
        for f in folds:
            print(f"  {f} 채점 …", flush=True)
            res[f] = score_fold(loaded[f][0], panel, liq_top, k, wmode)
        results[name] = res

    for name in presets:
        liq_top, k, wmode = PRESETS[name]
        tag = ("프로덕션 유니버스 (KRX500 전체 → 상위%d, conf가중)" % k
               if name == "prod" else
               "백테스트 유니버스 (유동성상위%d → 상위%d, 동일가중)" % (liq_top, k))
        report(f"{tag} x {len(folds)}폴드 — Sharpe (41bp, 회전은 실측 보유일)",
               results[name], folds)

    if len(presets) == 2:
        print("\n" + "=" * 76)
        print("유니버스만 바꿨을 때 결론이 뒤집히나 (프로덕션 − 백테스트)")
        print("=" * 76)
        for label, _, _ in POLICIES:
            a = [results["prod"][f].get(label, (float("nan"),))[0] for f in folds]
            b = [results["backtest"][f].get(label, (float("nan"),))[0] for f in folds]
            pa = [x for x in a if math.isfinite(x)]
            pb = [x for x in b if math.isfinite(x)]
            if not pa or not pb:
                continue
            ma, mb = sum(pa) / len(pa), sum(pb) / len(pb)
            print(f"  {label:<22} 프로덕션 {ma:>+6.2f}   백테스트 {mb:>+6.2f}"
                  f"   차이 {ma-mb:>+6.2f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
