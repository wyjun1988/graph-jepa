#!/usr/bin/env python3
"""예측된 연기금 흐름에 알파가 있는가 — 오라클 상한 대비 실현 비율.

`pension_flow_alpha_study.py` 가 상한을 쟀다: 미래 연기금 순매수를 완벽히 알면
RankIC 0.236(h10). 단 동시성 분리 결과 겹침 0에서도 0.102 가 남아, 그건
후행분(오른 뒤에도 계속 산다)이라 알파가 아니다. 가용 상한 ≈ 0.13.

여기서는 **모델이 실제로 예측한 값**으로 같은 자를 잰다. 모델은 이 값을 이미
예측하고 있었고(state_target_features 149개에 포함) 종목 선택에만 안 쓰였다.
`evaluate_node_prediction.py` 에 열을 추가해 뽑은 CSV 를 읽는다.

비교 대상:
  pred_return    prediction_entry_path_return (현행 선택 신호)
  pred_pension   prediction_investor_pension_flow_ratio_1d (미사용 부산물)
  pred_foreign / pred_institution  (대조군)
  combo          z(pred_return) + w·z(pred_pension)

사용법:
  python scripts/pension_pred_alpha_study.py --dirs <d1> <d2> ... --hold 10
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

K_PICKS = 35
COST_BP, TRADING_DAYS = 41.0, 252
MIN_NAMES = 100

SIGNALS = [
    ("pred_return (현행)",      "prediction_entry_path_return"),
    ("pred_pension",            "prediction_investor_pension_flow_ratio_1d"),
    ("pred_foreign",            "prediction_investor_foreign_flow_ratio_1d"),
    ("pred_institution",        "prediction_investor_institution_flow_ratio_1d"),
]
COMBO_WEIGHTS = [0.1, 0.25, 0.5, 1.0]


def load_forecasts(dirs, horizon):
    """시드 평균 {date: {ticker: {col: value}}}."""
    acc = {}
    n_seeds = 0
    for d in dirs:
        p = Path(d)
        if p.is_dir():
            cands = list(p.glob("**/return_1d_forecasts.csv"))
            if not cands:
                continue
            p = cands[0]
        if not p.exists():
            continue
        n_seeds += 1
        with open(p, newline="") as f:
            for r in csv.DictReader(f):
                if r["horizon"] != str(horizon):
                    continue
                cell = acc.setdefault(r["date"], {}).setdefault(r["ticker"], {})
                for _lbl, col in SIGNALS:
                    try:
                        v = float(r[col])
                    except (TypeError, ValueError, KeyError):
                        continue
                    if math.isfinite(v):
                        cell.setdefault(col, []).append(v)
    out = {}
    for dt, rows in acc.items():
        out[dt] = {t: {c: sum(v) / len(v) for c, v in cols.items()}
                   for t, cols in rows.items()}
    return out, n_seeds


def zscore(vals):
    if len(vals) < 2:
        return {k: 0.0 for k in vals}
    m = sum(vals.values()) / len(vals)
    s = statistics.pstdev(list(vals.values()))
    return {k: ((v - m) / s if s > 0 else 0.0) for k, v in vals.items()}


def rank_ic(sig_by_date, panel, hold):
    ics = []
    for d, row in sig_by_date.items():
        pairs = []
        for t, s in row.items():
            rec = panel.get(t)
            if rec is None:
                continue
            dates, opens, closes, index = rec
            i = index.get(d)
            if i is None or i + 1 >= len(dates) or i + hold >= len(dates) or opens[i + 1] <= 0:
                continue
            pairs.append((s, closes[i + hold] / opens[i + 1] - 1.0))
        if len(pairs) < MIN_NAMES:
            continue
        n = len(pairs)
        rs = {v: k for k, v in enumerate(sorted(range(n), key=lambda k: pairs[k][0]))}
        rr = {v: k for k, v in enumerate(sorted(range(n), key=lambda k: pairs[k][1]))}
        m = (n - 1) / 2
        num = sum((rs[k] - m) * (rr[k] - m) for k in range(n))
        den = math.sqrt(sum((rs[k] - m) ** 2 for k in range(n))
                        * sum((rr[k] - m) ** 2 for k in range(n)))
        if den > 0:
            ics.append(num / den)
    return (sum(ics) / len(ics), len(ics)) if ics else (float("nan"), 0)


def basket(sig_by_date, panel, hold, bench_cache):
    daily = []
    for d, row in sig_by_date.items():
        if len(row) < MIN_NAMES:
            continue
        key = (d, hold)
        if key not in bench_cache:
            vals = []
            for t in row:
                rec = panel.get(t)
                if rec is None:
                    continue
                dates, opens, closes, index = rec
                i = index.get(d)
                if i is None or i + 1 >= len(dates) or i + hold >= len(dates) or opens[i + 1] <= 0:
                    continue
                vals.append(closes[i + hold] / opens[i + 1] - 1.0)
            bench_cache[key] = (sum(vals) / len(vals)) if len(vals) >= 50 else None
        b = bench_cache[key]
        if b is None:
            continue
        picks = sorted(row, key=lambda t: row[t], reverse=True)[:K_PICKS]
        ex = []
        for t in picks:
            rec = panel.get(t)
            if rec is None:
                continue
            dates, opens, closes, index = rec
            i = index.get(d)
            if i is None or i + 1 >= len(dates) or i + hold >= len(dates) or opens[i + 1] <= 0:
                continue
            ex.append(closes[i + hold] / opens[i + 1] - 1.0 - b)
        if len(ex) >= K_PICKS // 2:
            daily.append(sum(ex) / len(ex))
    if len(daily) < 20:
        return None
    mu = sum(daily) / len(daily) - COST_BP / 1e4
    sd = statistics.stdev(daily)
    turns = TRADING_DAYS / hold
    return (mu / sd * math.sqrt(turns) if sd > 0 else float("nan"),
            mu * turns * 100, len(daily))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--hold", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=10)
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()
    print("예측 적재 중 …", flush=True)
    fc, n_seeds = load_forecasts(args.dirs, args.horizon)
    print(f"  시드 {n_seeds} | 날짜 {len(fc)}", flush=True)

    bench_cache = {}
    print("\n" + "=" * 78)
    print(f"예측된 수급의 알파 (지평 {args.horizon}, 보유 {args.hold}일, 상위10% 롱, 41bp)")
    print("=" * 78)
    print(f"{'신호':<26}{'RankIC':>10}{'Sharpe':>9}{'연%':>8}{'일수':>7}")
    print("-" * 78)
    for label, col in SIGNALS:
        sig = {d: {t: v[col] for t, v in row.items() if col in v}
               for d, row in fc.items()}
        sig = {d: r for d, r in sig.items() if len(r) >= MIN_NAMES}
        ic, nic = rank_ic(sig, panel, args.hold)
        r = basket(sig, panel, args.hold, bench_cache)
        line = f"{label:<26}{ic:>+10.4f}"
        line += f"{r[0]:>+9.2f}{r[1]:>+8.1f}{r[2]:>7}" if r else f"{'.':>9}{'.':>8}{'.':>7}"
        print(line)

    print()
    ret_col = SIGNALS[0][1]
    pen_col = SIGNALS[1][1]
    for w in COMBO_WEIGHTS:
        sig = {}
        for d, row in fc.items():
            both = {t: v for t, v in row.items() if ret_col in v and pen_col in v}
            if len(both) < MIN_NAMES:
                continue
            zr = zscore({t: v[ret_col] for t, v in both.items()})
            zp = zscore({t: v[pen_col] for t, v in both.items()})
            sig[d] = {t: zr[t] + w * zp[t] for t in both}
        ic, _ = rank_ic(sig, panel, args.hold)
        r = basket(sig, panel, args.hold, bench_cache)
        label = f"combo z(ret)+{w}·z(pension)"
        line = f"{label:<26}{ic:>+10.4f}"
        line += f"{r[0]:>+9.2f}{r[1]:>+8.1f}{r[2]:>7}" if r else f"{'.':>9}{'.':>8}{'.':>7}"
        print(line)

    print("\n판정: pred_pension 의 RankIC 를 가용 오라클 상한(≈0.13, 동행분 제외)과 대조한다.")
    print("      combo 가 pred_return 단독을 넘어야 선택 신호에 넣을 값어치가 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
