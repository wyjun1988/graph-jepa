#!/usr/bin/env python3
"""변동성 × 보유기간 상호작용 — "변동성 크면 짧게, 작으면 길게" 가 맞는가.

── 가설 (사용자, 2026-08-03) ───────────────────────────────────────────────
변동성이 큰 건 단기에 먹고 빠지기 좋고 오래 들면 나쁘다. 오래 가져갈 거면
변동성 작은 걸 골라야 한다.

이건 두 개의 서로 다른 주장이라 나눠 잰다.

  [A] 국면 층위 — 시장 변동성이 높은 **날 진입한 코호트**는 짧게 털어야 하고,
      낮은 날 진입분은 길게 가져가도 되는가. (배분조절 K 와 같은 신호원,
      다른 용도: 크기가 아니라 보유기간을 바꾼다.)

  [B] 종목 층위 — 같은 코호트 안에서 **변동성 큰 종목**은 짧게, 작은 종목은
      길게가 맞는가. 이건 지금까지 한 번도 안 쟀다.

[B] 가 맞으면 청산이 종목별로 갈려야 한다는 뜻이라 함의가 크다 —
현행은 코호트 전체에 같은 규칙(D+15)을 건다.

── 방법 ────────────────────────────────────────────────────────────────────
전부 인과: 진입일 t 의 판정에는 t **미만** 세션의 수익률만 쓴다.
  국면 z  = 유니버스 동일가중 일별수익의 W세션 실현변동성의 ZW세션 z-score
  종목 vol = 그 종목 종가수익의 W세션 실현변동성 (진입일 미만까지)
비용 41bp, 벤치 = 같은 (진입일, 보유일) 유니버스 평균. 회전 = 실측 보유일.

사용법:
  python scripts/vol_holding_interaction_study.py --folds r5 r4
"""

import argparse
import bisect
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
    "r4": "fold1_20241106_to_20250908",
    "r5": "fold1_20250905_to_20260710",
    "r3": "fold1_20240104_to_20241107",
    "r2": "fold1_20230307_to_20240105",
    "r1": "fold1_20220510_to_20230306",
}
SIGNAL_H, K_PICKS = 10, 35
COST_BP, TRADING_DAYS = 41.0, 252
MAXD = 30
VOL_W, Z_W = 20, 40                      # 실현변동성 창 / z-score 창
HOLDS = [3, 5, 10, 15, 20, 30]           # 비교할 보유기간
MIN_OBS = 8                              # 분할 후 최소 종목수


def load_ens(fold, seeds, prefix="ens_s"):
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
                except (TypeError, ValueError):
                    continue
                if math.isfinite(pr):
                    d.setdefault(row["date"], {})[row["ticker"]] = pr
        maps.append(d)
    if not maps:
        return None, 0
    dates = sorted(set.intersection(*(set(m) for m in maps)))
    return ({dt: {t: sum(m[dt][t] for m in maps) / len(maps)
                  for t in set.intersection(*(set(m[dt]) for m in maps))}
             for dt in dates}, len(maps))


def universe_daily_returns(panel):
    acc = {}
    for dates, _o, closes, _i in panel.values():
        for j in range(1, len(dates)):
            if closes[j - 1] <= 0:
                continue
            a = acc.setdefault(dates[j], [0.0, 0])
            a[0] += closes[j] / closes[j - 1] - 1.0
            a[1] += 1
    return {d: s / n for d, (s, n) in acc.items() if n >= 50}


def regime_z(rets, dates):
    """진입일별 시장변동성 z. 인과: 그 날짜 미만 수익률만."""
    ds = sorted(rets)
    vals = [rets[d] for d in ds]
    out = {}
    for d in dates:
        i = bisect.bisect_left(ds, d)
        if i < VOL_W + Z_W:
            continue
        def vol_at(j):
            seg = vals[j - VOL_W:j]
            return statistics.pstdev(seg) if len(seg) == VOL_W else float("nan")
        cur = vol_at(i)
        prev = [v for v in (vol_at(j) for j in range(i - Z_W, i)) if math.isfinite(v)]
        if not math.isfinite(cur) or len(prev) < 10:
            continue
        sd = statistics.pstdev(prev)
        out[d] = (cur - sum(prev) / len(prev)) / (sd + 1e-12)
    return out


def stock_vol(rec, dt):
    """진입일 미만 VOL_W 세션 실현변동성. 인과."""
    dates, _o, closes, index = rec
    i = index.get(dt)
    if i is None or i < VOL_W + 1:
        return None
    seg = closes[i - VOL_W:i + 1]
    if len(seg) != VOL_W + 1 or min(seg) <= 0:
        return None
    return statistics.pstdev([seg[j] / seg[j - 1] - 1.0 for j in range(1, len(seg))])


def hold_return(rec, dt, hold):
    dates, opens, closes, index = rec
    i = index.get(dt)
    if i is None or i + 1 >= len(dates) or i + hold >= len(dates):
        return None
    entry = opens[i + 1]
    if not (entry > 0):
        return None
    return closes[i + hold] / entry - 1.0


def sharpe(daily, hold):
    if len(daily) < 20:
        return None
    mu = sum(daily) / len(daily) - COST_BP / 1e4
    sd = statistics.stdev(daily)
    if sd <= 0:
        return None
    turns = TRADING_DAYS / hold
    return mu / sd * math.sqrt(turns), mu * turns * 100


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r5", "r4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()
    uni = universe_daily_returns(panel)

    # (fold, 행, 보유) -> Sharpe.  C 는 2x2 교차, LV 는 그 셀의 절대 변동성 수준.
    A, B, C, LV = {}, {}, {}, {}
    seed_n = {}
    for fold in args.folds:
        ens, n = load_ens(fold, args.seeds, args.prefix)
        if ens is None:
            continue
        seed_n[fold] = n
        picks = {dt: sorted(ens[dt], key=lambda t: ens[dt][t], reverse=True)[:K_PICKS]
                 for dt in ens if len(ens[dt]) >= 50}
        dts = sorted(picks)
        zs = regime_z(uni, dts)
        print(f"  {fold}: 시드 {n}, {len(dts)}일 (국면 z 산출 {len(zs)}일) …", flush=True)

        bench = {}
        for dt in dts:
            for h in HOLDS:
                vals = []
                for t in ens[dt]:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    r = hold_return(rec, dt, h)
                    if r is not None:
                        vals.append(r)
                if len(vals) >= 50:
                    bench[(dt, h)] = sum(vals) / len(vals)

        # 국면 분위 (해당 폴드 내 중앙값 기준 — 인과성은 z 자체가 보장)
        zvals = sorted(zs.values())
        zmed = zvals[len(zvals) // 2] if zvals else 0.0

        for h in HOLDS:
            buckets = {"고변동 국면": [], "저변동 국면": [], "전체": []}
            sbuck = {"고변동 종목": [], "저변동 종목": []}
            for dt in dts:
                if (dt, h) not in bench:
                    continue
                rows = []
                for t in picks[dt]:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    r = hold_return(rec, dt, h)
                    v = stock_vol(rec, dt)
                    if r is None or v is None:
                        continue
                    rows.append((t, r - bench[(dt, h)], v))
                if len(rows) < K_PICKS // 2:
                    continue
                ex = [x[1] for x in rows]
                buckets["전체"].append(sum(ex) / len(ex))
                regime = None
                if dt in zs:
                    regime = "고국면" if zs[dt] >= zmed else "저국면"
                    label = "고변동 국면" if regime == "고국면" else "저변동 국면"
                    buckets[label].append(sum(ex) / len(ex))
                # 종목 변동성 분할 — 코호트 내 중앙값
                rows.sort(key=lambda x: x[2])
                half = len(rows) // 2
                lo, hi = rows[:half], rows[half:]
                if len(lo) >= MIN_OBS and len(hi) >= MIN_OBS:
                    lo_ex = sum(x[1] for x in lo) / len(lo)
                    hi_ex = sum(x[1] for x in hi) / len(hi)
                    sbuck["저변동 종목"].append(lo_ex)
                    sbuck["고변동 종목"].append(hi_ex)
                    # [C] 2x2 교차 — 시장 국면 안에서 종목 변동성을 본다.
                    # 같은 "고변동 종목" 이라도 어느 국면에서인지에 따라 다른 물건이다.
                    if regime:
                        C.setdefault((fold, f"{regime}·저종목", h), []).append(lo_ex)
                        C.setdefault((fold, f"{regime}·고종목", h), []).append(hi_ex)
                        # 절대 변동성 수준도 남긴다(교락 확인용)
                        LV.setdefault((fold, f"{regime}·저종목"), []).append(
                            sum(x[2] for x in lo) / len(lo))
                        LV.setdefault((fold, f"{regime}·고종목"), []).append(
                            sum(x[2] for x in hi) / len(hi))
            for k, v in buckets.items():
                r = sharpe(v, h)
                if r:
                    A[(fold, k, h)] = r
            for k, v in sbuck.items():
                r = sharpe(v, h)
                if r:
                    B[(fold, k, h)] = r
            for key in [k for k in C if k[0] == fold and k[2] == h]:
                r = sharpe(C[key], h)
                C[key] = r if r else None

    folds = [f for f in args.folds if any(x[0] == f for x in A)]

    def table(store, rows, title, note):
        print("\n" + "=" * 78)
        print(title)
        print("  시드: " + ", ".join(f"{f}={seed_n.get(f,0)}" for f in folds)
              + f" | 변동성창 {VOL_W}세션, z창 {Z_W}세션, 인과")
        print("=" * 78)
        print(f"{'구분':<14}" + "".join(f"{'D+'+str(h):>9}" for h in HOLDS))
        print("-" * 78)
        for r in rows:
            for f in folds:
                cells = ""
                for h in HOLDS:
                    v = store.get((f, r, h))
                    cells += f"{v[0]:>+9.2f}" if v else f"{'.':>9}"
                print(f"{(r + ' ' + f):<14}{cells}")
            cells = ""
            for h in HOLDS:
                vs = [store[(f, r, h)][0] for f in folds if (f, r, h) in store]
                cells += f"{sum(vs)/len(vs):>+9.2f}" if vs else f"{'.':>9}"
            print(f"{(r + ' 평균'):<14}{cells}")
            print()
        print(note)

    table(A, ["전체", "고변동 국면", "저변동 국면"],
          "[A] 국면 층위 — 진입일 시장변동성 x 보유기간 (Sharpe)",
          "판정: '고변동 국면' 의 최적 보유가 '저변동 국면' 보다 짧으면 가설 지지.")
    table(B, ["저변동 종목", "고변동 종목"],
          "[B] 종목 층위 — 코호트 내 종목변동성 x 보유기간 (Sharpe)",
          "판정: 저변동 종목이 긴 보유에서, 고변동 종목이 짧은 보유에서 상대우위면\n"
          "      가설 지지 → 종목별로 청산일을 달리할 근거가 된다.")
    table(C, ["저국면·저종목", "저국면·고종목", "고국면·저종목", "고국면·고종목"],
          "[C] 2x2 교차 — 시장 국면 x 종목변동성 x 보유기간 (Sharpe)",
          "판정: 같은 '고변동 종목' 도 국면에 따라 다른 물건인가.\n"
          "      국면별로 최적 보유가 갈리면 청산을 국면x종목으로 조건화할 근거다.")

    print("\n" + "=" * 78)
    print("[참고] 각 셀의 평균 절대 변동성 (교락 확인 — 상대순위 분할의 한계)")
    print("=" * 78)
    for f in folds:
        cells = []
        for r in ["저국면·저종목", "저국면·고종목", "고국면·저종목", "고국면·고종목"]:
            v = LV.get((f, r))
            cells.append(f"{r} {sum(v)/len(v)*100:.2f}%" if v else f"{r} .")
        print(f"  {f}: " + " | ".join(cells))
    print("  * '저국면·고종목' 과 '고국면·저종목' 의 절대 수준이 겹치면,")
    print("    코호트 내 상대순위 분할이 절대 변동성을 대변하지 못한다는 뜻이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
