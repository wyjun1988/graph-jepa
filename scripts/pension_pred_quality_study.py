#!/usr/bin/env python3
"""예측이 나쁜 건가, 예측 대상이 무가치한 건가 — 둘을 분리한다.

앞선 결론("알파 있는 성분은 예측 불가, 예측되는 성분은 알파 없음")은 추론이었다.
반박(사용자): 오라클이 되는데 우리가 안 되면 그냥 우리 예측이 나쁜 것 아닌가.

분리하려면 세 가지를 따로 재야 한다.

  [1] 예측 품질   corr(예측, 그 예측의 실제 대상)  — 모델이 자기 목표를 맞히나
  [2] 대상 가치   IC(그 실제 대상, 선도수익)      — 완벽히 맞혀도 값이 있나
  [3] 실현 알파   IC(예측, 선도수익)              — 이미 잰 것 (-0.078)

⚠️ 앞선 비교의 결함: 오라클은 `t+1~t+h 누적 순매수`였는데 모델이 예측하는 것은
`t+h 시점의 1일 흐름비율`이다. 서로 다른 대상이라 "오라클은 되는데 예측은 안 된다"
가 성립하지 않는다. 여기서는 **모델의 실제 대상**으로 오라클을 다시 정의해 잰다.

  target_1d@t+h  = investor_pension_flow_ratio_1d 의 t+h 시점 실제값
  target_cum     = t+1~t+h 누적 순매수 / 누적 거래대금 (앞선 오라클)

사용법:
  python scripts/pension_pred_quality_study.py --dirs <d1> <d2> ... --horizon 10
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

CACHE = ROOT / "data" / "kiwoom_investor_cache"
PRED_COL = "prediction_investor_pension_flow_ratio_1d"
RET_COL = "prediction_entry_path_return"
MIN_NAMES = 100


def load_pension():
    out = {}
    for path in sorted(CACHE.glob("*_20200101_*.csv")):
        t = path.name.split("_")[0]
        rows = {}
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    p = float(r["investor_pension_net_m"])
                    v = float(r["investor_traded_value_m"])
                except (TypeError, ValueError, KeyError):
                    continue
                if math.isfinite(p) and math.isfinite(v) and v > 0:
                    rows[r["date"][:10]] = (p, v)
        if rows:
            out[t] = rows
    return out


def load_pred(dirs, horizon):
    acc = {}
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            c = list(p.glob("**/return_1d_forecasts.csv"))
            if not c:
                continue
            p = c[0]
        if not p.exists():
            continue
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["horizon"] != str(horizon):
                    continue
                cell = acc.setdefault(r["date"], {}).setdefault(r["ticker"], {})
                for col in (PRED_COL, RET_COL):
                    try:
                        v = float(r[col])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if math.isfinite(v):
                        cell.setdefault(col, []).append(v)
    return {d: {t: {c: sum(v) / len(v) for c, v in cols.items()}
                for t, cols in rows.items()} for d, rows in acc.items()}


def spearman(pairs):
    n = len(pairs)
    if n < MIN_NAMES:
        return None
    rs = {v: k for k, v in enumerate(sorted(range(n), key=lambda k: pairs[k][0]))}
    rr = {v: k for k, v in enumerate(sorted(range(n), key=lambda k: pairs[k][1]))}
    m = (n - 1) / 2
    num = sum((rs[k] - m) * (rr[k] - m) for k in range(n))
    den = math.sqrt(sum((rs[k] - m) ** 2 for k in range(n))
                    * sum((rr[k] - m) ** 2 for k in range(n)))
    return num / den if den > 0 else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--horizon", type=int, default=10)
    ap.add_argument("--hold", type=int, default=10)
    args = ap.parse_args()
    h = args.horizon

    print("적재 중 …", flush=True)
    panel = load_prices()
    pens = load_pension()
    pred = load_pred(args.dirs, h)
    dates_all = sorted({d for v in panel.values() for d in v[0]})
    pos = {d: i for i, d in enumerate(dates_all)}
    print(f"  예측일 {len(pred)} | 연기금 종목 {len(pens)}", flush=True)

    # 종목별 날짜 인덱스 (연기금 계열)
    pidx = {t: {d: i for i, d in enumerate(sorted(rows))} for t, rows in pens.items()}
    plist = {t: sorted(rows) for t, rows in pens.items()}

    def target_1d(t, d):
        """t+h 시점의 1일 흐름비율 (모델의 실제 예측 대상)."""
        ix = pidx.get(t, {}).get(d)
        if ix is None:
            return None
        ds = plist[t]
        if ix + h >= len(ds):
            return None
        p, v = pens[t][ds[ix + h]]
        return p / v if v > 0 else None

    def target_cum(t, d):
        """t+1~t+h 누적 순매수 / 누적 거래대금 (앞선 오라클)."""
        ix = pidx.get(t, {}).get(d)
        if ix is None:
            return None
        ds = plist[t]
        if ix + h >= len(ds):
            return None
        sp = sv = 0.0
        for j in range(ix + 1, ix + 1 + h):
            p, v = pens[t][ds[j]]
            sp += p
            sv += v
        return sp / sv if sv > 0 else None

    def fwd_ret(t, d):
        rec = panel.get(t)
        if rec is None:
            return None
        dd, oo, cc, ix = rec
        i = ix.get(d)
        if i is None or i + 1 >= len(dd) or i + args.hold >= len(dd) or oo[i + 1] <= 0:
            return None
        return cc[i + args.hold] / oo[i + 1] - 1.0

    def past_1d(t, d):
        """직전 관측 1일 흐름비율 — 순진한 persistence 기준선."""
        ix = pidx.get(t, {}).get(d)
        if ix is None or ix < 1:
            return None
        p, v = pens[t][plist[t][ix]]
        return p / v if v > 0 else None

    rows = {"pred_vs_t1d": [], "pred_vs_cum": [], "past_vs_t1d": [],
            "t1d_vs_ret": [], "cum_vs_ret": [], "past_vs_ret": [], "pred_vs_ret": []}
    for d, per_t in pred.items():
        acc = {k: [] for k in rows}
        for t, cols in per_t.items():
            pv = cols.get(PRED_COL)
            if pv is None:
                continue
            t1 = target_1d(t, d)
            tc = target_cum(t, d)
            pa = past_1d(t, d)
            fr = fwd_ret(t, d)
            if t1 is not None:
                acc["pred_vs_t1d"].append((pv, t1))
                if pa is not None:
                    acc["past_vs_t1d"].append((pa, t1))
                if fr is not None:
                    acc["t1d_vs_ret"].append((t1, fr))
            if tc is not None:
                acc["pred_vs_cum"].append((pv, tc))
                if fr is not None:
                    acc["cum_vs_ret"].append((tc, fr))
            if fr is not None:
                acc["pred_vs_ret"].append((pv, fr))
                if pa is not None:
                    acc["past_vs_ret"].append((pa, fr))
        for k, v in acc.items():
            s = spearman(v)
            if s is not None:
                rows[k].append(s)

    def show(k):
        v = rows[k]
        return f"{sum(v)/len(v):>+9.4f}{len(v):>7}" if v else f"{'.':>9}{'.':>7}"

    print("\n" + "=" * 74)
    print(f"예측 품질 vs 대상 가치 분리 (지평 {h}, 보유 {args.hold}일)")
    print("=" * 74)
    print("[1] 예측 품질 — 모델이 자기 목표를 맞히나")
    print(f"{'  corr(예측, t+h 1일흐름) = 진짜 목표':<48}{show('pred_vs_t1d')}")
    print(f"{'  corr(직전 1일흐름, t+h 1일흐름) = persistence':<48}{show('past_vs_t1d')}")
    print(f"{'  corr(예측, t+1~t+h 누적) = 앞서 쓴 오라클 대상':<48}{show('pred_vs_cum')}")
    print()
    print("[2] 대상 가치 — 완벽히 맞히면 값이 있나 (IC vs 선도수익)")
    print(f"{'  IC(t+h 1일흐름 실제값, 수익)':<48}{show('t1d_vs_ret')}")
    print(f"{'  IC(t+1~t+h 누적 실제값, 수익)':<48}{show('cum_vs_ret')}")
    print(f"{'  IC(직전 1일흐름, 수익)':<48}{show('past_vs_ret')}")
    print()
    print("[3] 실현 알파")
    print(f"{'  IC(예측, 수익)':<48}{show('pred_vs_ret')}")
    print()
    print("해석:")
    print("  [1]이 낮으면 -> 예측이 나쁜 것이다(사용자 반박이 옳다).")
    print("  [1]은 높은데 [2]의 해당 행이 0/음수면 -> 목표 자체가 무가치한 것이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
