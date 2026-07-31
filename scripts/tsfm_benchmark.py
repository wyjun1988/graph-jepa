#!/usr/bin/env python3
"""오픈소스 시계열 파운데이션 모델을 우리 챔프와 같은 조건에서 비교.

왜 하나: 이 시스템은 지금껏 **자기 자신하고만** 비교돼 왔다(설정 스윕 59건,
저변동성 팩터). 외부에서 독립적으로 학습된 기준선이 없어서 IC +0.05 가
절대적으로 좋은 수치인지 알 수 없었다. 사전학습 시계열 모델은 10^11 시점으로
학습된 시간 표현을 갖고 오므로, 우리가 실패한 지점(시퀀스 어텐션이 학습
컨텍스트 1,247개로는 과적합, -2.3σ)을 정확히 메울 후보다.

공정성을 위해 **완전히 같은 표본**을 쓴다. 챔프의 예측 CSV 에서 (날짜, 종목,
실현수익, 유동성)을 그대로 가져와 유니버스와 정답을 공유하고, TSFM 예측만
새로 얹는다. 채점도 동일하다 — 매일 유동성 top100 에서 예측 vs 실현 Pearson.

TSFM 신호 정의: 종가 시계열 256일을 넣고 10스텝을 예측해
    예측수익 = 예측종가[t+10] / 종가[t] - 1
실현 정답은 진입경로 수익(Close[t+10]/Open[t+1]-1)이라 기준점이 다르지만,
그 차이는 종목 간 거의 공통이라 **순위 상관**에는 영향이 작다. 어차피 재는 것은
순위 능력이다.

라이선스: Apache-2.0 만 쓴다. Moirai 는 CC BY-NC 4.0 이라 실매매 시스템에
부적합해 제외했다.

사용법:
  python scripts/tsfm_benchmark.py --model amazon/chronos-bolt-small --dates 30
  python scripts/tsfm_benchmark.py --model amazon/chronos-bolt-base
"""

import argparse
import csv
import math
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
NODE_EVAL = ROOT / "reports" / "walk_forward" / "node_eval"
OHLCV = ROOT / "data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv"
FOLDS = {"r5": "fold1_20250905_to_20260710",
         "r4": "fold1_20241106_to_20250908",
         "r3": "fold1_20240104_to_20241107",
         "r2": "fold1_20230307_to_20240105",
         "r1": "fold1_20220510_to_20230306"}
HORIZON = 10
TOP_N = 100
CONTEXT = 256          # 컨텍스트로 쓸 거래일 수


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


def load_champ(seeds, suffix):
    """{date: {ticker: (챔프예측, 실현, 유동성)}} — 시드 평균(앙상블)."""
    per_seed = []
    for s in seeds:
        p = NODE_EVAL / f"ens_s{s}_{suffix}" / "return_1d_forecasts.csv"
        if not p.exists():
            continue
        d = {}
        with open(p, newline="") as f:
            for row in csv.DictReader(f):
                if int(row["horizon"]) != HORIZON:
                    continue
                pr = float(row["prediction_entry_path_return"])
                rz = float(row["realized_path_return"])
                lq = float(row["current_value_ma20_log"])
                if not (math.isfinite(pr) and math.isfinite(rz)):
                    continue
                d.setdefault(row["date"], {})[row["ticker"]] = (pr, rz, lq)
        per_seed.append(d)
    if not per_seed:
        return None
    dates = set.intersection(*(set(d) for d in per_seed))
    out = {}
    for date in sorted(dates):
        tk = set.intersection(*(set(d[date]) for d in per_seed))
        base = per_seed[0][date]
        out[date] = {
            t: (sum(d[date][t][0] for d in per_seed) / len(per_seed),
                base[t][1], base[t][2])
            for t in tk
        }
    return out


def load_prices(tickers):
    """{ticker: (closes[], {date: idx})} — 필요한 종목만."""
    want = set(tickers)
    panel = {}
    for path in sorted(OHLCV.glob("*.csv")):
        t = path.name.split("_")[0]
        if t not in want:
            continue
        closes, idx = [], {}
        with open(path, newline="") as f:
            for row in csv.DictReader(f):
                try:
                    c = float(row["Close"])
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(c) and c > 0):
                    continue
                idx[row["Date"][:10]] = len(closes)
                closes.append(c)
        if closes:
            panel[t] = (closes, idx)
    return panel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="amazon/chronos-bolt-small")
    ap.add_argument("--fold", default="r5", choices=sorted(FOLDS))
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--dates", type=int, default=0, help="0=전부 (시험용으로 줄일 때)")
    ap.add_argument("--context", type=int, default=CONTEXT)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--dump", default="",
                    help="예측을 CSV 로 저장 — 모델 재실행 없이 포트폴리오·직교성 분석을 하려면 필수")
    ap.add_argument("--log-price", action="store_true", default=True,
                    help="로그가격으로 넣기 (기본). 가격 수준의 스케일 차이를 없앤다.")
    args = ap.parse_args()
    suffix = FOLDS[args.fold]

    champ = load_champ(args.seeds, suffix)
    if champ is None:
        print("챔프 예측 CSV 가 없습니다.")
        return 1
    dates = sorted(champ)
    if args.dates:
        dates = dates[: args.dates]
    print(f"[폴드 {args.fold}] 평가일 {len(dates)}일 | 챔프 앙상블 {len(args.seeds)}시드")

    # 각 날짜의 유동성 top100 = 채점 유니버스 (챔프와 완전히 동일)
    universe = {}
    need = set()
    for date in dates:
        rows = champ[date]
        ranked = sorted((t for t in rows if math.isfinite(rows[t][2])),
                        key=lambda t: rows[t][2], reverse=True)[:TOP_N]
        if len(ranked) >= 2:
            universe[date] = ranked
            need.update(ranked)
    print(f"필요 종목 {len(need)}개, 가격 적재 중...", flush=True)
    prices = load_prices(need)
    print(f"  적재 {len(prices)}개\n", flush=True)

    import torch
    from chronos import BaseChronosPipeline
    print(f"모델 로드: {args.model} ({args.device})", flush=True)
    pipe = BaseChronosPipeline.from_pretrained(
        args.model, device_map=args.device, dtype=torch.float32)
    nparam = sum(p.numel() for p in pipe.model.parameters())
    print(f"  파라미터 {nparam:,}\n", flush=True)

    daily = {"champ": [], "tsfm": [], "mom20": [], "rev5": []}
    dump_rows = []
    skipped = 0
    t_start = time.time()

    for n, date in enumerate(dates):
        tickers = universe[date]
        rows = champ[date]
        ctxs, keep = [], []
        for t in tickers:
            rec = prices.get(t)
            if rec is None:
                continue
            closes, idx = rec
            i = idx.get(date)
            if i is None or i < 30:            # 최소 이력
                continue
            lo = max(0, i - args.context + 1)
            seq = closes[lo:i + 1]
            if len(seq) < 30:
                continue
            vals = [math.log(c) for c in seq] if args.log_price else list(seq)
            ctxs.append(torch.tensor(vals, dtype=torch.float32))
            keep.append(t)
        if len(keep) < 10:
            skipped += 1
            continue

        _, mean = pipe.predict_quantiles(
            context=ctxs, prediction_length=HORIZON, quantile_levels=[0.1, 0.5, 0.9])
        pred_last = mean[:, -1].tolist()          # t+10 시점 예측

        tsfm_pred, champ_pred, realized, mom, rev = [], [], [], [], []
        for j, t in enumerate(keep):
            closes, idx = prices[t]
            i = idx[date]
            cur = math.log(closes[i]) if args.log_price else closes[i]
            # 예측 10일 수익
            if args.log_price:
                r = math.exp(pred_last[j] - cur) - 1.0
            else:
                r = pred_last[j] / cur - 1.0 if cur > 0 else float("nan")
            if not math.isfinite(r):
                continue
            tsfm_pred.append(r)
            champ_pred.append(rows[t][0])
            realized.append(rows[t][1])
            # 단순 기준선: 20일 모멘텀, 5일 반전
            m20 = (closes[i] / closes[i - 20] - 1.0) if i >= 20 else float("nan")
            r5 = -(closes[i] / closes[i - 5] - 1.0) if i >= 5 else float("nan")
            mom.append(m20)
            rev.append(r5)

        if len(realized) < 10:
            skipped += 1
            continue
        if args.dump:
            for j, t in enumerate(keep[: len(realized)]):
                dump_rows.append({
                    "date": date, "ticker": t,
                    "champ": champ_pred[j], "tsfm": tsfm_pred[j],
                    "mom20": mom[j], "rev5": rev[j], "realized": realized[j],
                })
        for key, series in (("champ", champ_pred), ("tsfm", tsfm_pred),
                            ("mom20", mom), ("rev5", rev)):
            pairs = [(p, z) for p, z in zip(series, realized) if math.isfinite(p)]
            if len(pairs) < 10:
                continue
            ic = pearson([p for p, _ in pairs], [z for _, z in pairs])
            if math.isfinite(ic):
                daily[key].append((date, ic))
        if (n + 1) % 25 == 0:
            el = time.time() - t_start
            print(f"  {n+1}/{len(dates)}  ({el:.0f}s, 남은 예상 {el/(n+1)*(len(dates)-n-1):.0f}s)",
                  flush=True)

    if args.dump and dump_rows:
        with open(args.dump, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(dump_rows[0].keys()))
            w.writeheader(); w.writerows(dump_rows)
        print(f"\n예측 저장: {args.dump} ({len(dump_rows):,}행)")

    print(f"\n제외된 날 {skipped}일")
    print(f"\n{'신호':<28}{'IC':>10}{'일수':>7}")
    print("-" * 46)
    label = {"champ": f"우리 챔프 (앙상블 {len(args.seeds)}시드)",
             "tsfm": f"{args.model.split('/')[-1]} (제로샷)",
             "mom20": "20일 모멘텀 (단순)",
             "rev5": "5일 반전 (단순)"}
    means = {}
    for k in ("champ", "tsfm", "mom20", "rev5"):
        v = [ic for _, ic in daily[k]]
        if not v:
            continue
        means[k] = sum(v) / len(v)
        print(f"{label[k]:<28}{means[k]:>+10.4f}{len(v):>7}")

    # 챔프 대비 짝지은 검정 (같은 날짜)
    base = dict(daily["champ"])
    print(f"\n── 챔프 대비 짝지은 차이 (겹침보정 NW t, lag={HORIZON}) ──")
    for k in ("tsfm", "mom20", "rev5"):
        d = dict(daily[k])
        common = sorted(set(d) & set(base))
        if len(common) < HORIZON + 2:
            continue
        diffs = [d[x] - base[x] for x in common]
        t = newey_west_t(diffs, lag=HORIZON)
        print(f"  {label[k]:<26} Δ {sum(diffs)/len(diffs):+.4f}  NW t {t:+6.2f}  (n={len(diffs)})")

    SD = 0.0159
    print(f"\n판정 기준: 지평10 시드 σ={SD:.4f} (docs/MEASUREMENT_CORRECTIONS_20260730.md)")
    if "tsfm" in means and "champ" in means:
        d = means["tsfm"] - means["champ"]
        print(f"  TSFM − 챔프 = {d:+.4f} ({d/SD:+.1f}σ)")
    print("\n단일 폴드 결과. TSFM 은 제로샷이며 파인튜닝하지 않았다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
