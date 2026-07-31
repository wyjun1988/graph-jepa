#!/usr/bin/env python3
"""시드 앙상블 포화곡선 — 시드를 몇 개까지 늘릴 값어치가 있는가.

배경: 시드 앙상블 자체는 이미 프로덕션에 채택돼 있다(daily_signal.py, 2026-07-19
검증: 4시드가 최악폴드 Sharpe 0.5~0.65 → 1.13~1.28). 결합은 예측 평균이다.
답이 없는 쪽은 "몇 개가 적정한가"다. 6시드를 학습해 부분집합을 전수(C(6,k))
평균하면 k=1..6 포화곡선이 나오고, 4에서 이미 평평한지 아닌지 보인다.

채점은 evaluate_node_prediction.py 를 그대로 흉내낸다:
  * 매 거래일 유동성(current_value_ma20_log) 상위 100종목
  * 그 안에서 predicted_entry_path vs realized_path 의 Pearson → 일별 IC 평균
포트폴리오는 같은 유니버스에서 예측 상위 K를 동일가중 10일 보유한 실현 경로수익.
(헤지·비용 없음 — 절대수준이 아니라 변형 간 비교용)

공식 수치와의 관계: 예측 CSV 행은 return_valid(1일 수익률 가용성)로 써지고 공식
지표는 path_valid(지평 수익률 가용성)로 걸러지므로 재채점값이 보고값과 미세하게
다를 수 있다 — 버그가 아니다. 모든 변형을 동일 행 집합으로 재므로 비교는 유효하다.

사용법:
  python scripts/ensemble_report.py --seeds 3 5 11 17 23 29 --fold r5
"""

import argparse
import csv
import math
import sys
from itertools import combinations
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
HORIZON = 10          # 챔프의 보유기간; entry_path IC 도 이 지평에서 채점된다
TOP_N = 100           # 유동성 상위 N — 공식 지표와 동일
TOP_K = 5             # 포트폴리오 편입 종목수 (챔프의 --top-k 5)


def pearson(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    mx = sum(xs) / n
    my = sum(ys) / n
    sxy = sxx = syy = 0.0
    for x, y in zip(xs, ys):
        dx, dy = x - mx, y - my
        sxy += dx * dy
        sxx += dx * dx
        syy += dy * dy
    denom = math.sqrt(sxx * syy)
    return sxy / denom if denom > 0 else float("nan")


def newey_west_t(diffs, lag):
    """겹치는 보유기간이 만드는 자기상관을 보정한 t (평균=0 검정)."""
    n = len(diffs)
    if n < lag + 2:
        return float("nan")
    mean = sum(diffs) / n
    dev = [d - mean for d in diffs]
    var = sum(x * x for x in dev) / n
    for k in range(1, lag + 1):
        cov = sum(dev[t] * dev[t - k] for t in range(k, n)) / n
        var += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    return mean / math.sqrt(var / n) if var > 0 else float("nan")


def load_forecasts(name, suffix):
    """{date: {ticker: (pred, realized, liquidity)}} — horizon 10 행만."""
    path = NODE_EVAL / f"{name}_{suffix}" / "return_1d_forecasts.csv"
    if not path.exists():
        return None
    by_date = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["horizon"]) != HORIZON:
                continue
            pred = float(row["prediction_entry_path_return"])
            realized = float(row["realized_path_return"])
            liq = float(row["current_value_ma20_log"])
            if not (math.isfinite(pred) and math.isfinite(realized)):
                continue
            by_date.setdefault(row["date"], {})[row["ticker"]] = (pred, realized, liq)
    return by_date


def build_universe(loaded):
    """날짜별 (유동성 top-100 종목, 실현수익). 유동성·실현값은 시드와 무관하므로
    한 번만 만들어 모든 부분집합이 공유한다 — 이게 없으면 전수 조합이 너무 느리다."""
    seeds = sorted(loaded)
    dates = set.intersection(*(set(loaded[s]) for s in seeds))
    universe = {}
    for date in sorted(dates):
        tickers = set.intersection(*(set(loaded[s][date]) for s in seeds))
        base = loaded[seeds[0]][date]
        ranked = sorted(
            (t for t in tickers if math.isfinite(base[t][2])),
            key=lambda t: base[t][2],
            reverse=True,
        )[:TOP_N]
        if len(ranked) >= TOP_K:
            universe[date] = (ranked, [base[t][1] for t in ranked])
    return universe


def subset_predictions(loaded, subset, universe):
    """{date: [해당 부분집합 시드 평균 예측]} — 유니버스 순서에 맞춤."""
    out = {}
    for date, (tickers, _) in universe.items():
        rows = [loaded[s][date] for s in subset]
        out[date] = [
            sum(r[t][0] for r in rows) / len(rows) for t in tickers
        ]
    return out


def score_ic(preds, universe):
    daily = []
    for date, vals in preds.items():
        ic = pearson(vals, universe[date][1])
        if math.isfinite(ic):
            daily.append(ic)
    return (sum(daily) / len(daily)) if daily else float("nan")


def portfolio_daily(preds, universe, top_k=TOP_K):
    out = {}
    for date, vals in preds.items():
        realized = universe[date][1]
        order = sorted(range(len(vals)), key=lambda i: vals[i], reverse=True)[:top_k]
        out[date] = sum(realized[i] for i in order) / len(order)
    return out


def stats(daily):
    vals = [daily[d] for d in sorted(daily)]
    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    return m, sd, (m / sd if sd > 0 else float("nan"))


def reported_ic(name, suffix, horizon=HORIZON):
    """future_rollout.csv 가 보고한 top100 IC — 지평을 지정해 뽑는다.

    주의: 이 파일은 (날짜 x 지평) 행을 담고 있어서 통째로 평균하면 지평
    1/2/3/5/10 이 섞인 값이 나온다. 지금까지 실험 간 비교에 쓰인 수치가 그
    혼합 평균이었다. 10일 보유 전략의 성패와 직결되는 건 지평 10 이므로
    여기서는 지평을 명시해 뽑는다.
    """
    path = NODE_EVAL / f"{name}_{suffix}" / "future_rollout.csv"
    if not path.exists():
        return float("nan")
    vals = []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if horizon is not None and int(row["horizon"]) != horizon:
                continue
            v = row.get("realized_entry_path_ic_top100")
            if v not in (None, "", "nan"):
                vals.append(float(v))
    return sum(vals) / len(vals) if vals else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--fold", default="r5", choices=sorted(FOLDS))
    args = ap.parse_args()
    suffix = FOLDS[args.fold]

    loaded = {}
    for seed in args.seeds:
        by_date = load_forecasts(f"{args.prefix}{seed}", suffix)
        if by_date is None:
            print(f"  {args.prefix}{seed}: 예측 파일 없음 — 제외")
            continue
        loaded[seed] = by_date
    if not loaded:
        print("사용할 런이 없습니다.")
        return 1

    universe = build_universe(loaded)
    seeds = sorted(loaded)
    print(f"[폴드 {args.fold}]  시드 {seeds}  공통 거래일 {len(universe)}일\n")

    # ── 단일 시드: 재채점이 공식 보고값을 따라가는지 확인
    print(f"{'시드':>6} {'재채점 IC':>11} {'보고 IC':>11} {'차이':>9}   (둘 다 지평 10)")
    print("-" * 40)
    for s in seeds:
        preds = subset_predictions(loaded, [s], universe)
        ic = score_ic(preds, universe)
        ref = reported_ic(f"{args.prefix}{s}", suffix)
        gap = ic - ref if math.isfinite(ref) else float("nan")
        flag = "  ← 괴리 큼" if math.isfinite(gap) and abs(gap) > 5e-3 else ""
        print(f"{s:>6} {ic:>+11.4f} {ref:>+11.4f} {gap:>+9.5f}{flag}")

    # ── 포화곡선: 각 k 에 대해 C(n,k) 부분집합을 전수 평균
    print(f"\n── 시드 수별 포화곡선 (부분집합 전수 평균) ──")
    print(f"{'시드수':>6} {'조합수':>6} {'IC':>10} {'수익%/10d':>11} {'Sharpe':>8}")
    print("-" * 45)
    curve = {}
    for k in range(1, len(seeds) + 1):
        subsets = list(combinations(seeds, k))
        ics, means, sharpes = [], [], []
        for sub in subsets:
            preds = subset_predictions(loaded, list(sub), universe)
            ics.append(score_ic(preds, universe))
            m, _, sh = stats(portfolio_daily(preds, universe))
            means.append(m)
            sharpes.append(sh)
        ic_k = sum(ics) / len(ics)
        m_k = sum(means) / len(means)
        sh_k = sum(x for x in sharpes if math.isfinite(x)) / max(
            1, sum(1 for x in sharpes if math.isfinite(x))
        )
        curve[k] = (ic_k, m_k, sh_k)
        print(f"{k:>6} {len(subsets):>6} {ic_k:>+10.4f} {m_k*100:>+11.3f} {sh_k:>8.2f}")

    # ── 해석: 4시드(현 프로덕션)가 이미 포화점인가
    n = len(seeds)
    ic1, m1, sh1 = curve[1]
    icn, mn, shn = curve[n]
    print()
    total_ic = icn - ic1
    total_sh = shn - sh1
    print(f"단일 → {n}시드 총 개선 : IC {total_ic:+.4f} | Sharpe {total_sh:+.2f}")
    if 4 in curve and n > 4 and abs(total_sh) > 1e-9:
        got = (curve[4][2] - sh1) / total_sh * 100
        print(f"4시드(현 프로덕션)가 확보한 몫 : Sharpe 기준 {got:.0f}%")
        rest = curve[n][2] - curve[4][2]
        print(f"4 → {n} 추가 이득 : Sharpe {rest:+.2f}")
        if rest < 0.05:
            print("→ 4시드에서 이미 포화. 시드를 더 늘릴 값어치 없음.")
        else:
            print(f"→ 아직 오르는 중 — {n}시드까지 늘릴 근거 있음.")

    # ── 전체 앙상블 vs 단일 평균: 포트폴리오 수준 유의성 (겹침보정)
    full = subset_predictions(loaded, seeds, universe)
    full_daily = portfolio_daily(full, universe)
    singles_daily = [portfolio_daily(subset_predictions(loaded, [s], universe), universe)
                     for s in seeds]
    common = sorted(full_daily)
    mean_single = {d: sum(sd[d] for sd in singles_daily) / len(singles_daily)
                   for d in common}
    diffs = [full_daily[d] - mean_single[d] for d in common]
    t = newey_west_t(diffs, lag=HORIZON)
    avg = sum(diffs) / len(diffs) if diffs else float("nan")
    print()
    print(f"{n}시드 앙상블 − 단일평균 : {avg*100:+.3f}%/10d  (겹침보정 NW t={t:+.2f}, {len(diffs)}일)")
    if math.isfinite(t) and t > 2:
        print("→ IC 개선이 포트폴리오 수익으로도 전환됨.")
    elif math.isfinite(t) and t > 0:
        print("→ 방향은 맞으나 유의하지 않음 (NW t<2).")
    else:
        print("→ 포트폴리오 수준 이득 없음 — IC 개선이 수익으로 전환되지 않음.")

    print("\n단일 폴드 결과 — 채택 판정은 다시드x다폴드 짝지은 t검정 필요 (docs §7-5)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
