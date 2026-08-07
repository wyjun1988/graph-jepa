#!/usr/bin/env python3
"""두 변형을 시드로 짝지어 비교 — 헤드 입력 실험용.

같은 시드는 같은 데이터 순서·같은 마스킹을 쓰므로, 시드로 짝지으면 시드 분산의
상당 부분이 상쇄된다. 비짝지은 평균 비교보다 검정력이 높다.
(완전한 짝은 아니다 — 모델 크기가 달라 RNG 스트림이 갈린다.)

두 층위를 본다:
  개별 시드 : 같은 시드끼리 IC 차이 → 짝지은 평균과 표준오차
  앙상블    : 각 변형의 시드 평균 예측으로 만든 앙상블끼리 → 프로덕션 관련 수치

채점은 ensemble_report 와 동일(유동성 top100, 지평 10 진입경로 Pearson).

사용법:
  python scripts/paired_variant_report.py --a ens_s --b ctx_s --seeds 3 5 17 29
"""

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODE_EVAL = ROOT / "reports" / "walk_forward" / "node_eval"
FOLDS = {
    "r5": "fold1_20250905_to_20260710",
    "r4": "fold1_20241106_to_20250908",
    "r3": "fold1_20240104_to_20241107",
    "r2": "fold1_20230307_to_20240105",
    "r1": "fold1_20220510_to_20230306",
}
HORIZON = 10
TOP_N = 100


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


def load(name, suffix):
    p = NODE_EVAL / f"{name}_{suffix}" / "return_1d_forecasts.csv"
    if not p.exists():
        return None
    by_date = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["horizon"]) != HORIZON:
                continue
            pr = float(row["prediction_entry_path_return"])
            rz = float(row["realized_path_return"])
            lq = float(row["current_value_ma20_log"])
            if not (math.isfinite(pr) and math.isfinite(rz)):
                continue
            by_date.setdefault(row["date"], {})[row["ticker"]] = (pr, rz, lq)
    return by_date


def universe(loaded):
    """모든 변형·시드가 공유하는 날짜별 top-100 (실현값·유동성은 변형 무관)."""
    dates = set.intersection(*(set(d) for d in loaded))
    uni = {}
    for date in sorted(dates):
        tk = set.intersection(*(set(d[date]) for d in loaded))
        base = loaded[0][date]
        ranked = sorted(
            (t for t in tk if math.isfinite(base[t][2])),
            key=lambda t: base[t][2], reverse=True,
        )[:TOP_N]
        if len(ranked) >= 2:
            uni[date] = (ranked, [base[t][1] for t in ranked])
    return uni


def daily_ic(by_date, uni, seeds_data=None):
    """변형 하나(또는 시드 평균)의 날짜별 IC."""
    out = {}
    for date, (tickers, rz) in uni.items():
        if seeds_data is None:
            pred = [by_date[date][t][0] for t in tickers]
        else:
            pred = [sum(d[date][t][0] for d in seeds_data) / len(seeds_data)
                    for t in tickers]
        ic = pearson(pred, rz)
        if math.isfinite(ic):
            out[date] = ic
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="기준 변형 접두 (예: ens_s)")
    ap.add_argument("--b", required=True, help="비교 변형 접두 (예: ctx_s)")
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--fold", default="r5", choices=sorted(FOLDS))
    ap.add_argument("--label-a", default=None)
    ap.add_argument("--label-b", default=None)
    ap.add_argument("--json", default="", help="판정기용 결과 JSON 경로")
    args = ap.parse_args()
    suffix = FOLDS[args.fold]
    la = args.label_a or args.a.rstrip("_s")
    lb = args.label_b or args.b.rstrip("_s")

    A, B, used = {}, {}, []
    for s in args.seeds:
        a, b = load(f"{args.a}{s}", suffix), load(f"{args.b}{s}", suffix)
        if a is None or b is None:
            print("  시드 %d: %s 없음 — 제외" % (s, "A" if a is None else "B"))
            continue
        A[s], B[s] = a, b
        used.append(s)
    if not used:
        print("비교할 쌍이 없습니다.")
        return 1

    uni = universe([A[s] for s in used] + [B[s] for s in used])
    print("[폴드 %s] 짝 %s | 공통 거래일 %d일\n" % (args.fold, used, len(uni)))

    print("%6s %12s %12s %10s" % ("시드", la, lb, "차이"))
    print("-" * 44)
    per_seed = {}
    for s in used:
        ia = daily_ic(A[s], uni)
        ib = daily_ic(B[s], uni)
        ma = sum(ia.values()) / len(ia)
        mb = sum(ib.values()) / len(ib)
        per_seed[s] = (ma, mb, ia, ib)
        print("%6d %+12.4f %+12.4f %+10.4f" % (s, ma, mb, mb - ma))

    diffs = [per_seed[s][1] - per_seed[s][0] for s in used]
    n = len(diffs)
    md = sum(diffs) / n
    if n > 1:
        sd = (sum((d - md) ** 2 for d in diffs) / (n - 1)) ** 0.5
        se = sd / math.sqrt(n)
        t = md / se if se > 0 else float("nan")
    else:
        sd = se = t = float("nan")
    print()
    print("짝지은 평균 차이 : %+.4f  (표준편차 %.4f, 표준오차 %.4f, t %+.2f, n=%d)"
          % (md, sd, se, t, n))

    # 앙상블 대 앙상블 — 프로덕션이 쓰는 형태
    ea = daily_ic(None, uni, seeds_data=[A[s] for s in used])
    eb = daily_ic(None, uni, seeds_data=[B[s] for s in used])
    mea = sum(ea.values()) / len(ea)
    meb = sum(eb.values()) / len(eb)
    common = sorted(set(ea) & set(eb))
    dd = [eb[d] - ea[d] for d in common]
    tt = newey_west_t(dd, lag=HORIZON)
    print()
    print("%d시드 앙상블 : %s %+.4f  vs  %s %+.4f  차이 %+.4f"
          % (n, la, mea, lb, meb, meb - mea))
    print("  일별 짝지은 차이 겹침보정 NW t = %+.2f (n=%d일)" % (tt, len(dd)))

    print()
    SD_SEED = 0.0159   # 지평10 시드 σ (n=6, docs/MEASUREMENT_CORRECTIONS_20260730.md)
    print("판정 (지평10 시드 σ=%.4f 기준):" % SD_SEED)
    if abs(md) < SD_SEED / 2:
        print("  → 사실상 동등. %s 가 미래 잠재를 안 쓰고도 같은 성능이면" % lb)
        print("     파라미터·연산을 덜어낼 근거가 된다.")
    elif md > 0:
        print("  → %s 가 나은 방향. 다만 t=%.2f 로 %s" % (lb, t, "유의" if abs(t) > 2 else "유의 미달"))
    else:
        print("  → %s 가 나쁜 방향. 다만 t=%.2f 로 %s" % (lb, t, "유의" if abs(t) > 2 else "유의 미달"))
    if args.json:
        import json as _json
        Path(args.json).write_text(_json.dumps({
            "study": "paired_ic", "a": args.a, "b": args.b, "fold": args.fold,
            "seeds": used, "ic_a": mea, "ic_b": meb, "diff": meb - mea,
            "nw_t": None if tt != tt else tt,
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print("[json] %s" % args.json)
    print("\n단일 폴드·%d쌍 — 채택 판정은 다폴드 필요 (docs §7-5)" % n)
    return 0


if __name__ == "__main__":
    sys.exit(main())
