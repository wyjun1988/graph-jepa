#!/usr/bin/env python3
"""청산 정책 비교 — 같은 픽을 사다리/단일 D+10/단일 D+20 으로 청산했을 때.

배경(docs §8-18): 동일 모델·동일 픽에서 청산만 바꿔 Sharpe -0.28 → +0.98 이 나왔다.
낮의 IC 실험들 최대 기대효과의 10배이고 모델은 건드리지 않는다. 다만 그 결과는
단일 시드·단일 폴드였다. 여기서는 시드 앙상블 캠페인의 6시드 예측을 써서
(1) 그 발견이 시드를 바꿔도 유지되는지, (2) 앙상블 픽에서도 같은지 본다.

수익 정의는 평가 파이프라인과 동일한 '진입경로':
    D+h 수익 = Close[t+h] / Open[t+1] - 1        (t = 신호일)
수익은 시장중립 기준이다 — 진입과 동시에 같은 금액 KQ11 숏(β=1, 프로덕션과 동일).
비용은 왕복 고정 bp 로 단순화했다(매수 13bp + 매도 28bp = 41bp 가 현행 가정).
사다리의 레그별 스프레드 악화(§8-9~8-12)는 여기 반영돼 있지 않으므로, 사다리에
유리한 쪽으로 치우친 비교다 — 그런데도 사다리가 지면 결론은 더 강해진다.

사용법:
  python scripts/exit_policy_report.py --seeds 3 5 11 17 23 29 --fold r5
"""

import argparse
import csv
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODE_EVAL = ROOT / "reports" / "walk_forward" / "node_eval"
OHLCV = ROOT / "data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv"
INDEX_CSV = ROOT / "data/external_cache/kq11_kosdaq_20200101_20260710.csv"
FOLDS = {
    "r5": "fold1_20250905_to_20260710",
    "r4": "fold1_20241106_to_20250908",
    "r3": "fold1_20240104_to_20241107",
}
SIGNAL_HORIZON = 10   # 픽을 만드는 예측 지평 (모델의 학습 타깃)
TOP_N = 100           # 유동성 상위 N
TOP_K = 5             # 편입 종목수
LADDER = (1, 2, 3, 5, 10)
COST_GRID = (26, 41, 72)   # docs §8-18 과 동일한 왕복 bp 격자
TRADING_DAYS = 252

# 정책별 자본의 평균 보유일 → 연 회전수(252/보유일). docs §8-9 의 핵심:
# 사다리는 주당 평균 4.2일만 들고 있어 연 60회전, 단일 D+20 은 12.6회전이다.
# 거래당 같은 bp 를 물려도 연간 비용은 4.6배 차이가 난다.
POLICY_HOLD = {
    "사다리(1,2,3,5,10)": sum(LADDER) / len(LADDER),   # 4.2
    "단일 D+10": 10.0,
    "단일 D+20": 20.0,
}


def load_prices():
    """{ticker: (dates[], open[], close[])} — 거래일 순."""
    panel = {}
    for path in sorted(OHLCV.glob("*.csv")):
        ticker = path.name.split("_")[0]
        dates, opens, closes = [], [], []
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    o = float(row["Open"])
                    c = float(row["Close"])
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(o) and math.isfinite(c)):
                    continue
                dates.append(row["Date"][:10])
                opens.append(o)
                closes.append(c)
        if dates:
            panel[ticker] = (dates, opens, closes, {d: i for i, d in enumerate(dates)})
    return panel


def load_index():
    """KQ11 (dates, open, close, index) — 헤지 다리 계산용."""
    dates, opens, closes = [], [], []
    with open(INDEX_CSV, newline="") as f:
        for row in csv.DictReader(f):
            try:
                o, c = float(row["Open"]), float(row["Close"])
            except (TypeError, ValueError):
                continue
            if not (math.isfinite(o) and math.isfinite(c) and o > 0):
                continue
            dates.append(row["Date"][:10])
            opens.append(o)
            closes.append(c)
    return dates, opens, closes, {d: i for i, d in enumerate(dates)}


def load_picks(name, suffix):
    """{date: [(ticker, pred, liquidity)]} — 신호 지평 행만."""
    path = NODE_EVAL / f"{name}_{suffix}" / "return_1d_forecasts.csv"
    if not path.exists():
        return None
    by_date = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            if int(row["horizon"]) != SIGNAL_HORIZON:
                continue
            pred = float(row["prediction_entry_path_return"])
            liq = float(row["current_value_ma20_log"])
            if not (math.isfinite(pred) and math.isfinite(liq)):
                continue
            by_date.setdefault(row["date"], {})[row["ticker"]] = (pred, liq)
    return by_date


def path_return(rec, date, horizon):
    """Close[t+h] / Open[t+1] - 1. 데이터가 모자라면 None."""
    if rec is None:
        return None
    dates, opens, closes, index = rec
    i = index.get(date)
    if i is None or i + horizon >= len(dates) or i + 1 >= len(dates):
        return None
    entry = opens[i + 1]
    if not (entry > 0):
        return None
    return closes[i + horizon] / entry - 1.0


def entry_path_return(panel, ticker, date, horizon):
    return path_return(panel.get(ticker), date, horizon)


def select(by_date, date, top_k=TOP_K):
    rows = by_date[date]
    liquid = sorted(rows, key=lambda t: rows[t][1], reverse=True)[:TOP_N]
    if len(liquid) < top_k:
        return []
    return sorted(liquid, key=lambda t: rows[t][0], reverse=True)[:top_k]


def policy_returns(panel, index_rec, picks_by_date):
    """{정책: {날짜: 비용전 수익}}

    D+20 은 미래 20거래일이 있어야 하므로 기간 끝에서 결측이 난다. 정책마다 날짜
    표본이 달라지면 비교가 무의미해지므로, **모든 지평(1,2,3,5,10,20)이 갖춰진
    픽만** 쓰고 편입 종목이 하나라도 모자란 날은 통째로 버린다. 세 정책이 완전히
    같은 날짜·같은 종목을 보게 만드는 것이 목적이다.
    """
    horizons = list(LADDER) + [20]
    out = {"사다리(1,2,3,5,10)": {}, "단일 D+10": {}, "단일 D+20": {}}
    for date, picks in picks_by_date.items():
        if not picks:
            continue
        idx = {h: path_return(index_rec, date, h) for h in horizons}
        if any(v is None for v in idx.values()):
            continue                       # 지수 다리가 없으면 헤지 수익 계산 불가
        rows = []
        for t in picks:
            rs = {h: entry_path_return(panel, t, date, h) for h in horizons}
            if any(v is None for v in rs.values()):
                rows = []
                break                      # 한 종목이라도 불완전하면 그 날은 제외
            rows.append(rs)
        if len(rows) != len(picks):
            continue
        n = len(rows)
        # 시장중립: 진입과 동시에 같은 금액만큼 지수 숏(β=1) — 프로덕션과 동일
        out["사다리(1,2,3,5,10)"][date] = sum(
            sum(r[h] - idx[h] for h in LADDER) / len(LADDER) for r in rows
        ) / n                              # 각 레그 20% 균등
        out["단일 D+10"][date] = sum(r[10] - idx[10] for r in rows) / n
        out["단일 D+20"][date] = sum(r[20] - idx[20] for r in rows) / n
    return out


def annualized(daily, hold_days, cost_bp):
    """회전율로 환산한 연 수익·연 변동성·Sharpe.

    매일 새 코호트가 들어가는 정상상태에서 평균 투입자본은 (코호트 x 평균보유일)
    이므로, 자본 기준 연 회전수는 252/보유일이다. 비용은 회전마다 물고, 수익도
    같은 배수로 늘어난다. 변동성은 사이클이 독립이라 보고 sqrt(회전수)로 키운다.
    사다리를 거래당 비용으로만 재면 회전수 4.6배 차이가 통째로 빠진다 (docs §8-9).
    """
    vals = [v for _, v in sorted(daily.items())]
    if len(vals) < 2:
        return float("nan"), float("nan"), float("nan")
    m = sum(vals) / len(vals)
    sd = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
    turns = TRADING_DAYS / hold_days
    net = (m - cost_bp / 10000.0) * turns
    vol = sd * math.sqrt(turns)
    return net, vol, (net / vol if vol > 0 else float("nan"))



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, required=True)
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--fold", default="r5", choices=sorted(FOLDS))
    args = ap.parse_args()
    suffix = FOLDS[args.fold]

    loaded = {}
    for s in args.seeds:
        d = load_picks(f"{args.prefix}{s}", suffix)
        if d is None:
            print(f"  {args.prefix}{s}: 예측 없음 — 제외")
            continue
        loaded[s] = d
    if not loaded:
        print("사용할 런이 없습니다.")
        return 1

    print("가격 패널 적재 중...", flush=True)
    panel = load_prices()
    index_rec = load_index()
    print(f"  종목 {len(panel)}개, 지수 {len(index_rec[0])}일\n", flush=True)

    seeds = sorted(loaded)
    dates = sorted(set.intersection(*(set(loaded[s]) for s in seeds)))

    # 앙상블 픽 = 시드 평균 예측 기준 상위 K
    ens_by_date = {}
    for date in dates:
        tickers = set.intersection(*(set(loaded[s][date]) for s in seeds))
        base = loaded[seeds[0]][date]
        ens_by_date[date] = {
            t: (sum(loaded[s][date][t][0] for s in seeds) / len(seeds), base[t][1])
            for t in tickers
        }

    variants = {f"seed {s}": {d: select(loaded[s], d) for d in dates} for s in seeds}
    variants["앙상블"] = {d: select(ens_by_date, d) for d in dates}

    print(f"[폴드 {args.fold}] 공통 거래일 {len(dates)}일, 편입 {TOP_K}종목")
    print("수익=시장중립(KQ11 β=1 숏), 비용은 연 회전수 환산\n")
    header = (f"{'변형':>10} {'청산':>20} {'보유일':>6} {'연회전':>6} {'연수익%':>8}"
              + "".join(f"{'Sh' + str(c):>7}" for c in COST_GRID))
    print(header)
    print("-" * len(header))

    summary = {}
    for label, picks_by_date in variants.items():
        pol = policy_returns(panel, index_rec, picks_by_date)
        for policy, daily in pol.items():
            if not daily:
                continue
            hold = POLICY_HOLD[policy]
            turns = TRADING_DAYS / hold
            gross, _, _ = annualized(daily, hold, 0)
            row = (f"{label:>10} {policy:>20} {hold:>6.1f} {turns:>6.1f}"
                   f" {gross*100:>+8.1f}")
            for c in COST_GRID:
                _, _, sh = annualized(daily, hold, c)
                row += f"{sh:>7.2f}"
            print(row)
            summary.setdefault(policy, {})[label] = daily
        print()

    # ── 핵심 비교: 41bp 기준, 사다리 vs D+20 (docs §8-18 재현)
    print("── 41bp 기준 정책별 Sharpe (docs §8-18 재현·확장) ──")
    print(f"{'변형':>10}" + "".join(f"{p:>22}" for p in summary))
    for label in list(variants):
        row = f"{label:>10}"
        for policy in summary:
            daily = summary[policy].get(label)
            if daily:
                _, _, sh = annualized(daily, POLICY_HOLD[policy], 41)
                row += f"{sh:>22.2f}"
            else:
                row += f"{'—':>22}"
        print(row)

    lad = summary.get("사다리(1,2,3,5,10)", {})
    d20 = summary.get("단일 D+20", {})
    if lad and d20:
        wins = 0
        total = 0
        for label in variants:
            if label in lad and label in d20:
                _, _, a = annualized(lad[label], POLICY_HOLD["사다리(1,2,3,5,10)"], 41)
                _, _, b = annualized(d20[label], POLICY_HOLD["단일 D+20"], 41)
                if math.isfinite(a) and math.isfinite(b):
                    total += 1
                    wins += 1 if b > a else 0
        print(f"\nD+20 이 사다리를 이긴 변형: {wins}/{total}")
        if total and wins == total:
            print("→ 시드를 바꿔도 뒤집히지 않음. §8-18 의 청산 결론이 재현됨.")
        elif total and wins >= total / 2:
            print("→ 대체로 재현되나 일부 시드에서 뒤집힘 — 시드 의존성 있음.")
        else:
            print("→ 재현 실패. §8-18 은 단일 시드 특수성이었을 가능성.")

    print("\n비용은 왕복 고정 bp 근사 — 사다리의 레그별 스프레드 악화(§8-9~8-12) 미반영")
    return 0


if __name__ == "__main__":
    sys.exit(main())
