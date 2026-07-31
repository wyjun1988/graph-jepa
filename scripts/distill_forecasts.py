#!/usr/bin/env python3
"""여러 시드의 종목별 예측을 한 개의 작은 CSV 로 압축 + 챔프 단독 요약 출력.

원본 `return_1d_forecasts.csv` 는 시드당 약 85MB(모든 지평 x 전 종목)라 원격
머신에서 빼오기 어렵다. 실제 분석에 필요한 것은 그중 극히 일부다:

  지평 10 행 x 일별 유동성 상위 100종목 x (앙상블 예측, 실현수익, 유동성)
  = 약 19,400행 ≈ 1.2MB (gzip 시 300KB 남짓)

이 스크립트가 그 압축을 하고, 동시에 챔프 단독 지표(IC·포트폴리오)를 계산해
콘솔에 찍는다. 그래서 파일을 전혀 못 빼오는 상황에서도 화면만 복사하면
"학습이 제대로 됐는지"는 바로 확인된다.

출력 CSV 는 tsfm_benchmark.py --dump 와 같은 스키마(tsfm 열만 비어 있음)라
로컬에서 Chronos 예측을 얹으면 사전등록 검정이 그대로 돌아간다.

사용법:
  python scripts/distill_forecasts.py --fold r3 --seeds 3 17 29 --out r3_compact.csv
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODE_EVAL = ROOT / "reports" / "walk_forward" / "node_eval"
FOLDS = {
    "r5": "fold1_20250905_to_20260710",
    "r4": "fold1_20241106_to_20250908",
    "r3": "fold1_20240104_to_20241107",
}
HORIZON = 10
TOP_N = 100
K = 20
COST_BP = 41.0
TRADING_DAYS = 252


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="r3", choices=sorted(FOLDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 17, 29])
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true",
                    help="어떤 런이 있는지만 보여주고 끝낸다")
    args = ap.parse_args()
    if args.list or not args.out:
        print("=== node_eval 에 있는 런 ===")
        for d in sorted(NODE_EVAL.glob("*_fold1_*")):
            f = d / "return_1d_forecasts.csv"
            mark = f"{f.stat().st_size/1024/1024:6.1f}MB" if f.exists() else "  종목별예측 없음"
            print(f"  {d.name:<44} {mark}")
        if not args.out:
            print("\n--out 을 주면 압축을 실행합니다.")
            return 0
    suffix = FOLDS[args.fold]
    if args.list:
        return 0

    per_seed = []
    for s in args.seeds:
        p = NODE_EVAL / f"{args.prefix}{s}_{suffix}" / "return_1d_forecasts.csv"
        if not p.exists():
            print(f"  시드 {s}: 없음 — 건너뜀 ({p})")
            continue
        d = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if int(row["horizon"]) != HORIZON:
                    continue
                try:
                    pr = float(row["prediction_entry_path_return"])
                    rz = float(row["realized_path_return"])
                    lq = float(row["current_value_ma20_log"])
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(pr) and math.isfinite(rz)):
                    continue
                d.setdefault(row["date"], {})[row["ticker"]] = (pr, rz, lq)
        per_seed.append((s, d))
        print(f"  시드 {s}: {sum(len(v) for v in d.values()):,}행 적재")

    if not per_seed:
        print("적재된 시드가 없습니다.")
        return 1

    dates = sorted(set.intersection(*(set(d) for _, d in per_seed)))
    rows_out = []
    for date in dates:
        tickers = set.intersection(*(set(d[date]) for _, d in per_seed))
        base = per_seed[0][1][date]
        ranked = sorted((t for t in tickers if math.isfinite(base[t][2])),
                        key=lambda t: base[t][2], reverse=True)[:TOP_N]
        for t in ranked:
            champ = sum(d[date][t][0] for _, d in per_seed) / len(per_seed)
            rows_out.append({
                "date": date, "ticker": t,
                "champ": f"{champ:.8g}",
                "tsfm": "",                      # 로컬에서 채운다
                "mom20": "", "rev5": "",
                "realized": f"{base[t][1]:.8g}",
                "liq": f"{base[t][2]:.8g}",
            })

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["date", "ticker", "champ", "tsfm",
                                          "mom20", "rev5", "realized", "liq"])
        w.writeheader()
        w.writerows(rows_out)
    size = Path(args.out).stat().st_size
    print(f"\n압축 완료: {args.out}  ({len(rows_out):,}행, {size/1024/1024:.1f}MB)")

    # ── 챔프 단독 요약 — 파일을 못 빼와도 화면만 보면 학습 성공 여부를 안다
    by = {}
    for r in rows_out:
        by.setdefault(r["date"], []).append((float(r["champ"]), float(r["realized"])))
    ics, ports = [], []
    for d, v in by.items():
        if len(v) < 30:
            continue
        c = pearson([a for a, _ in v], [b for _, b in v])
        if math.isfinite(c):
            ics.append(c)
        uni = sum(b for _, b in v) / len(v)
        top = sorted(v, key=lambda x: -x[0])[:K]
        ports.append(sum(b for _, b in top) / len(top) - uni)
    m = sum(ports) / len(ports)
    sd = statistics.stdev(ports)
    turns = TRADING_DAYS / HORIZON
    net = (m - COST_BP / 1e4) * turns
    sh = net / (sd * math.sqrt(turns)) if sd > 0 else float("nan")

    print("\n" + "═" * 58)
    print(f"챔프 단독 요약 — 폴드 {args.fold}, 시드 {[s for s, _ in per_seed]}")
    print("═" * 58)
    print(f"  평가일          {len(by)}일")
    print(f"  지평10 IC       {sum(ics)/len(ics):+.4f}")
    print(f"  상위{K} 수익     {m*100:+.3f}%/10d")
    print(f"  연수익({COST_BP:.0f}bp) {net*100:+.1f}%")
    print(f"  Sharpe          {sh:+.2f}")
    print("\n  ── 같은 시드 3/17/29 기준 참고값 ──")
    print("    폴드 r5 : IC +0.0588 | 상위20 +0.681%/10d | Sharpe +0.42")
    print("    폴드 r4 : IC +0.0380 | 상위20 +0.449%/10d | Sharpe +0.09")
    print("  IC 가 +0.02~+0.07 밖이면 학습·평가가 잘못됐을 수 있다.")
    print("  (앞서 보고한 r5 +0.0485 는 6시드 앙상블 값이라 다르다)")
    print("═" * 58)
    print(f"\n돌려주실 것: {args.out} (위 크기) — 이것만 있으면 나머지 분석이 됩니다.")
    print("파일도 어려우면 위 요약 화면만 복사해 주셔도 학습 성공 여부는 확인됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
