#!/usr/bin/env python3
"""여러 시드의 종목별 예측을 작은 CSV 하나로 압축 + 화면 요약.

원본 `return_1d_forecasts.csv` 는 시드당 약 97MB(전 지평 x 전 종목)라 원격
머신에서 빼오기 어렵다. 실제 분석에 필요한 건 그중 일부다.

── 스키마를 2026-07-31 에 바꾼 이유 ────────────────────────────────────────
예전 버전은 시드를 평균낸 `champ` 열 하나만, 지평 10 만 남겼다. 그 결과:

  1) 시드 민감도를 로컬에서 못 쟀다. 그런데 r5 에서 시드 3개를 어떻게 고르냐에
     따라 Sharpe 가 +0.42 ~ -0.50 으로 갈렸다 — 지금 이 프로젝트의 최대 불확실성이
     바로 이것이고, 평균낸 열로는 재현이 불가능하다.
  2) 지평 10 만 남겨서 단기 헤드(1/2/3/5)의 IC 를 못 봤다.

그래서 시드별 열 x 전 지평 롱포맷으로 바꿨다. 크기는 gzip 후 폴드당 2MB 안팎이라
여전히 메일로 보낼 수 있고, GPU 를 다시 돌릴 필요가 없어진다.

  date,ticker,horizon,liq,realized,s3,s5,s11,s17,s23,s29

`realized` 는 진입경로 실현수익(Close[t+h]/Open[t+1]-1)이고 시드와 무관하다.
D+15/20/30 청산 분석은 이 파일로 안 된다 — 예측 파일에 지평 15/20/30 이
없기 때문이다. 대신 로컬 OHLCV 캐시(196MB)에서 (date,ticker) 로 직접 계산한다.
즉 이 파일엔 **예측만** 있으면 되고, realized 는 로컬 계산과 맞는지 보는
교차검증용으로만 넣는다.

사용법:
  python scripts/distill_forecasts.py --fold r3 --seeds 3 5 11 17 23 29 --out r3_compact.csv
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
    "r2": "fold1_20230307_to_20240105",
    "r1": "fold1_20220510_to_20230306",
}
HORIZON = 10          # 유니버스 선정·요약에 쓰는 기준 지평
TOP_N = 100
K = 20
COST_BP = 41.0
TRADING_DAYS = 252
REF_SEEDS = [3, 17, 29]   # 5폴드 패널에 쓴 조합 — 비교 기준


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


def read_seed(path, keep=None):
    """(date,ticker,horizon) -> pred. keep 이 주어지면 그 키만 남긴다(메모리 절약).

    keep=None 이면 (base, preds) 를 함께 돌려준다 — base 는 시드 무관 정보.
    """
    preds = {}
    base = {} if keep is None else None
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["date"], row["ticker"], row["horizon"])
            if keep is not None and key not in keep:
                continue
            try:
                pr = float(row["prediction_entry_path_return"])
            except (TypeError, ValueError):
                continue
            if not math.isfinite(pr):
                continue
            preds[key] = pr
            if base is not None:
                try:
                    rz = float(row["realized_path_return"])
                    lq = float(row["current_value_ma20_log"])
                except (TypeError, ValueError):
                    continue
                if math.isfinite(rz):
                    base[key] = (rz, lq)
    return (preds, base) if keep is None else preds


def summarize(rows, seeds, label):
    """지평 10 · 상위K 로 IC·포트폴리오 지표를 낸다."""
    by = {}
    for r in rows:
        if r["horizon"] != str(HORIZON):
            continue
        vals = [r[f"s{s}"] for s in seeds]
        if any(v == "" for v in vals):
            continue
        pred = sum(float(v) for v in vals) / len(vals)
        by.setdefault(r["date"], []).append((pred, float(r["realized"])))
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
    if not ports:
        return None
    m = sum(ports) / len(ports)
    sd = statistics.stdev(ports) if len(ports) > 1 else float("nan")
    turns = TRADING_DAYS / HORIZON
    net = (m - COST_BP / 1e4) * turns
    sh = net / (sd * math.sqrt(turns)) if sd and sd > 0 else float("nan")
    return dict(label=label, days=len(by), ic=sum(ics) / len(ics),
                ret=m * 100, ann=net * 100, sharpe=sh)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", default="r3", choices=sorted(FOLDS))
    ap.add_argument("--seeds", nargs="+", type=int,
                    default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--out", default="")
    ap.add_argument("--list", action="store_true",
                    help="어떤 런이 있는지만 보여주고 끝낸다")
    args = ap.parse_args()

    if args.list or not args.out:
        print("=== node_eval 에 있는 런 ===")
        for d in sorted(NODE_EVAL.glob("*_fold1_*")):
            f = d / "return_1d_forecasts.csv"
            mark = (f"{f.stat().st_size/1024/1024:6.1f}MB" if f.exists()
                    else "  종목별예측 없음")
            print(f"  {d.name:<44} {mark}")
        if not args.out:
            print("\n--out 을 주면 압축을 실행합니다.")
        return 0

    suffix = FOLDS[args.fold]
    avail = [s for s in args.seeds
             if (NODE_EVAL / f"{args.prefix}{s}_{suffix}"
                 / "return_1d_forecasts.csv").exists()]
    missing = [s for s in args.seeds if s not in avail]
    if missing:
        print(f"  없는 시드(건너뜀): {missing}")
    if not avail:
        print(f"적재할 시드가 없습니다 — 폴드 {args.fold}")
        return 1

    # 1패스: 첫 시드로 시드 무관 정보(realized·liq)와 전체 키를 얻는다
    p0 = NODE_EVAL / f"{args.prefix}{avail[0]}_{suffix}" / "return_1d_forecasts.csv"
    print(f"  시드 {avail[0]}: 적재 중 …")
    preds0, base = read_seed(p0)
    print(f"  시드 {avail[0]}: {len(preds0):,}행")

    # 날짜별 유동성 상위 TOP_N — 지평 10 행의 liq 로 정한다
    liq_by_date = {}
    for (d, t, h), (_, lq) in base.items():
        if h == str(HORIZON) and math.isfinite(lq):
            liq_by_date.setdefault(d, []).append((lq, t))
    universe = {d: {t for _, t in sorted(v, reverse=True)[:TOP_N]}
                for d, v in liq_by_date.items()}
    keep = {k for k in base if k[1] in universe.get(k[0], ())}
    print(f"  유니버스: {len(universe)}일 x 상위{TOP_N} → 키 {len(keep):,}개")

    # 2패스: 나머지 시드는 유니버스 키만
    seed_preds = {avail[0]: {k: preds0[k] for k in keep if k in preds0}}
    del preds0
    for s in avail[1:]:
        p = NODE_EVAL / f"{args.prefix}{s}_{suffix}" / "return_1d_forecasts.csv"
        print(f"  시드 {s}: 적재 중 …")
        seed_preds[s] = read_seed(p, keep=keep)
        print(f"  시드 {s}: {len(seed_preds[s]):,}행")

    cols = ["date", "ticker", "horizon", "liq", "realized"] + \
           [f"s{s}" for s in args.seeds]
    rows_out = []
    for k in sorted(keep, key=lambda x: (x[0], x[1], int(x[2]))):
        d, t, h = k
        rz, lq = base[k]
        row = {"date": d, "ticker": t, "horizon": h,
               "liq": f"{lq:.6g}", "realized": f"{rz:.8g}"}
        for s in args.seeds:
            v = seed_preds.get(s, {}).get(k)
            row[f"s{s}"] = f"{v:.8g}" if v is not None else ""
        rows_out.append(row)

    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows_out)
    size = Path(args.out).stat().st_size
    print(f"\n압축 완료: {args.out}  ({len(rows_out):,}행, {size/1024/1024:.1f}MB)")

    # ── 화면 요약 — 파일을 못 빼와도 이것만 보면 학습 성공 여부를 안다 ──
    print("\n" + "=" * 62)
    print(f"폴드 {args.fold} 요약 (지평{HORIZON}, 유동성 top{TOP_N} → 상위{K}, {COST_BP:.0f}bp)")
    print("=" * 62)
    print(f"{'조합':<22}{'평가일':>6}{'IC':>9}{'%/10d':>9}{'연%':>8}{'Sharpe':>8}")
    print("-" * 62)
    combos = [(avail, f"전체 {len(avail)}시드")]
    ref = [s for s in REF_SEEDS if s in avail]
    if ref and set(ref) != set(avail):
        combos.append((ref, f"기준 {ref}"))
    for s in avail:
        combos.append(([s], f"  단일 s{s}"))
    for seeds, label in combos:
        r = summarize(rows_out, seeds, label)
        if r:
            print(f"{label:<22}{r['days']:>6}{r['ic']:>+9.4f}"
                  f"{r['ret']:>+9.3f}{r['ann']:>+8.1f}{r['sharpe']:>+8.2f}")

    print("\n  ── 참고: 시드 3/17/29 로 이미 확인된 값 ──")
    print("    r1 IC +0.0921 | r2 +0.0764 | r3 +0.0637 | r4 +0.0380 | r5 +0.0588")
    print("  같은 폴드의 '기준' 행이 위 값과 크게 다르면 환경이 다른 것이다.")
    print("=" * 62)
    print(f"\n돌려주실 것: {args.out}.gz — 이것만 있으면 나머지 분석이 로컬에서 됩니다.")
    print("못 빼오시면 위 표만 복사해 주셔도 시드 민감도까지는 확인됩니다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
