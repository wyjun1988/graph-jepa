#!/usr/bin/env python3
"""제로샷 TSFM 다중 모델 종합 — 우리 모델이 제로샷을 이기는가.

사용자 기준: "제로샷도 못 이기면 다시 생각해봐야지."
그래서 크기(8.7M~205M)와 아키텍처(Bolt 직접분위 / T5 샘플링)를 훑어
**두 폴드 모두에서** 재고, 부호 일치 여부를 함께 본다. 한 폴드만 좋은 것은
모멘텀에서 이미 겪은 함정이다(r5 +0.48 → r4 -1.44).

포트폴리오는 상위 5일을 뺀 값도 같이 낸다. 소수 이벤트에 실린 성과인지
가려내기 위한 것이다 — Chronos-small 의 Sharpe 0.42 가 상위 5일 제거 시
0.10 으로 무너지는 것을 이미 확인했다.

사용법:
  python scripts/zeroshot_summary.py --glob '/path/z_*.csv'
"""

import argparse
import csv
import glob as globmod
import math
import os
import statistics
import sys

K, H, COST, TRADING_DAYS = 20, 10, 41.0, 252


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


def newey_west_t(diffs, lag=H):
    n = len(diffs)
    if n < lag + 2:
        return float("nan")
    m = sum(diffs) / n
    dev = [d - m for d in diffs]
    var = sum(x * x for x in dev) / n
    for k in range(1, lag + 1):
        var += 2.0 * (1.0 - k / (lag + 1.0)) * \
               sum(dev[i] * dev[i - k] for i in range(k, n)) / n
    return m / math.sqrt(var / n) if var > 0 else float("nan")


def load(path):
    by = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rec = {"champ": float(r["champ"]), "tsfm": float(r["tsfm"]),
                       "realized": float(r["realized"])}
            except (KeyError, TypeError, ValueError):
                continue
            if all(math.isfinite(v) for v in rec.values()):
                by.setdefault(r["date"], []).append(rec)
    return by


def ic_series(by, key):
    out = {}
    for d, rows in by.items():
        if len(rows) < 10:
            continue
        c = pearson([r[key] for r in rows], [r["realized"] for r in rows])
        if math.isfinite(c):
            out[d] = c
    return out


def port_series(by, key):
    out = {}
    for d, rows in by.items():
        if len(rows) < 30:
            continue
        uni = sum(r["realized"] for r in rows) / len(rows)
        picks = sorted(rows, key=lambda r: -r[key])[:K]
        out[d] = sum(r["realized"] for r in picks) / len(picks) - uni
    return out


def sharpe(vals, cost=COST):
    if len(vals) < 10:
        return float("nan")
    m = sum(vals) / len(vals)
    sd = statistics.stdev(vals)
    t = TRADING_DAYS / H
    return ((m - cost / 1e4) * t) / (sd * math.sqrt(t)) if sd > 0 else float("nan")


def drop_top(vals, n=5):
    order = sorted(range(len(vals)), key=lambda i: -vals[i])
    bad = set(order[:n])
    return [v for i, v in enumerate(vals) if i not in bad]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    args = ap.parse_args()

    files = sorted(globmod.glob(args.glob))
    if not files:
        print("파일 없음")
        return 1

    rows = {}
    champ_ref = {}
    for p in files:
        base = os.path.basename(p).replace(".csv", "")
        parts = base.split("_")
        fold = parts[-1]
        model = "_".join(parts[1:-1])
        by = load(p)
        ic_t = ic_series(by, "tsfm")
        pt_t = port_series(by, "tsfm")
        ic_c = ic_series(by, "champ")
        pt_c = port_series(by, "champ")
        champ_ref[fold] = (sum(ic_c.values()) / len(ic_c),
                           sharpe(list(pt_c.values())),
                           sharpe(drop_top(list(pt_c.values()))), pt_c, ic_c)
        rows[(model, fold)] = (sum(ic_t.values()) / len(ic_t),
                               sharpe(list(pt_t.values())),
                               sharpe(drop_top(list(pt_t.values()))), pt_t, ic_t)

    models = sorted({m for m, _ in rows})
    folds = ["r5", "r4"]

    print("═" * 78)
    print("제로샷 TSFM vs 우리 모델 — 두 폴드 동일 조건")
    print("═" * 78)
    print(f"\n{'모델':<24}" + "".join(f"{'IC ' + f:>11}" for f in folds) +
          f"{'평균IC':>9}{'부호':>6}")
    print("-" * 78)
    for f in folds:
        if f in champ_ref:
            pass
    ic5 = champ_ref.get("r5", [float('nan')])[0]
    ic4 = champ_ref.get("r4", [float('nan')])[0]
    print(f"{'★ 우리 챔프(앙상블)':<24}{ic5:>+11.4f}{ic4:>+11.4f}"
          f"{(ic5+ic4)/2:>+9.4f}{'예' if ic5*ic4 > 0 else '아니오':>6}")
    for m in models:
        a = rows.get((m, "r5"), [float("nan")])[0]
        b = rows.get((m, "r4"), [float("nan")])[0]
        ok = "예" if (math.isfinite(a) and math.isfinite(b) and a * b > 0) else "아니오"
        print(f"{m:<24}{a:>+11.4f}{b:>+11.4f}{(a+b)/2:>+9.4f}{ok:>6}")

    print(f"\n{'모델':<24}" + "".join(f"{'Sh ' + f:>11}" for f in folds) +
          f"{'평균':>9}{'상위5일제거(r5/r4)':>20}")
    print("-" * 78)
    s5 = champ_ref.get("r5", [0, float('nan'), float('nan')])[1]
    s4 = champ_ref.get("r4", [0, float('nan'), float('nan')])[1]
    d5 = champ_ref.get("r5", [0, 0, float('nan')])[2]
    d4 = champ_ref.get("r4", [0, 0, float('nan')])[2]
    print(f"{'★ 우리 챔프(앙상블)':<24}{s5:>+11.2f}{s4:>+11.2f}{(s5+s4)/2:>+9.2f}"
          f"{d5:>+10.2f}{d4:>+10.2f}")
    for m in models:
        a = rows.get((m, "r5"), [0, float("nan")])[1]
        b = rows.get((m, "r4"), [0, float("nan")])[1]
        da = rows.get((m, "r5"), [0, 0, float("nan")])[2]
        db = rows.get((m, "r4"), [0, 0, float("nan")])[2]
        print(f"{m:<24}{a:>+11.2f}{b:>+11.2f}{(a+b)/2:>+9.2f}{da:>+10.2f}{db:>+10.2f}")

    print(f"\n── 챔프 대비 짝지은 NW t (lag={H}) ──")
    print(f"{'모델':<24}{'ΔIC r5':>10}{'t':>7}{'ΔIC r4':>10}{'t':>7}"
          f"{'Δ수익 r5':>11}{'t':>7}{'Δ수익 r4':>11}{'t':>7}")
    print("-" * 78)
    for m in models:
        cells = []
        for metric in (4, 3):        # 4=IC 계열, 3=포트폴리오 계열
            for f in folds:
                r = rows.get((m, f))
                c = champ_ref.get(f)
                if not r or not c:
                    cells += ["—", "—"]
                    continue
                common = sorted(set(r[metric]) & set(c[metric]))
                diffs = [r[metric][d] - c[metric][d] for d in common]
                scale = 1 if metric == 4 else 100
                cells += [f"{sum(diffs)/len(diffs)*scale:+.4f}" if metric == 4
                          else f"{sum(diffs)/len(diffs)*scale:+.3f}",
                          f"{newey_west_t(diffs):+.2f}"]
        print(f"{m:<24}" + "".join(f"{c:>10}" if i % 2 == 0 else f"{c:>7}"
                                   for i, c in enumerate(cells)))

    print("\n판정 기준: 지평10 시드 σ=0.0159. 두 폴드 부호 일치 + |t|>2 가 아니면")
    print("어느 쪽도 우세를 주장할 수 없다. 상위5일 제거는 소수 이벤트 의존을 본다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
