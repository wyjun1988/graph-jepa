#!/usr/bin/env python3
"""갈아타기(switch) 연구 — 새 정보가 오면 더 좋은 종목으로 바꾸는 게 이득인가.

── 착상 (사용자, 2026-08-06) ───────────────────────────────────────────────
바꿀 기회를 주면 바꾸는 게 낫다는 문제처럼, 다음날 가격이 비슷하면 그날 기준으로
더 좋은 종목으로 갈아타면 어떤가.

── 랭크청산과 무엇이 다른가 ────────────────────────────────────────────────
랭크청산: "내 종목이 top20% 밖으로 밀리면 **판다**" (현금화)
갈아타기: "내 종목보다 **더 좋은 게 있으면 교체**" (재투자)

후자는 자본을 놀리지 않는다. 대신 **교체 비용이 왕복 2회**다 — 팔고(1) 사고(1).
41bp 를 왕복으로 보면 교체 1회에 41bp 를 더 낸다. 그래서 스코어 격차가
비용을 넘을 때만 갈아타야 한다.

── 정책군 ──────────────────────────────────────────────────────────────────
매 세션, 직전 세션 스코어(인과)로 판정한다.
  보유 종목의 그날 스코어 순위 vs 미보유 후보 중 최상위.
  후보가 보유분보다 스코어가 gap 이상 높으면 교체(그날 종가 체결).

  switch_top    : 순위 기준 — 보유가 top N% 밖이고, 대체 후보는 top10% 안일 때
  switch_score  : 스코어 격차 기준 — z-score 차이가 임계 이상일 때
  switch_flat   : 사용자 원안 — **가격이 비슷할 때만**(진입가 대비 |수익| < x%)
                  갈아탄다. 이미 오른 놈은 놔두고, 크게 빠진 놈도 놔둔다.

청산은 원래 코호트의 D+15(교체해도 코호트 만기는 유지 — 자본 회전을 공평하게).
비용: 최초 매수 41bp + 교체마다 41bp 추가.
벤치 = 유니버스 평균(시가 진입, 1회) — 전 정책 공통.

사용법:
  python scripts/switch_study.py --folds r5 r4
"""

import argparse
import bisect
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OHLCV = ROOT / "data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv"
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
HOLD = 15
GAP_SKIP = 0.05
MAX_SWITCH = 3          # 포지션당 교체 상한 (비용 폭주 방지)

# (라벨, 종류, 파라미터…)
# (라벨, 종류, 파라미터, 집중허용도)
POLICIES = [
    ("현행 (교체 없음)",              "none",   None,        1),
    ("top20 이탈→top10 (중복금지)",   "top",    (0.20, 0.10), 1),
    ("top30 이탈→top10 (중복금지)",   "top",    (0.30, 0.10), 1),
    # ── 집중 허용 (2026-08-06 사용자 반론: 계속 좋으면 몰려도 되지 않나) ──
    ("top20→top10 (중복 2까지)",     "top",    (0.20, 0.10), 2),
    ("top20→top10 (중복 3까지)",     "top",    (0.20, 0.10), 3),
    ("top20→top10 (중복 무제한)",     "top",    (0.20, 0.10), 99),
    ("top30→top10 (중복 3까지)",     "top",    (0.30, 0.10), 3),
    ("top30→top10 (중복 무제한)",     "top",    (0.30, 0.10), 99),
    ("스코어격차 1.0σ (중복 무제한)",   "score",  1.0,         99),
    ("평탄(|수익|<3%)+격차0.5σ",       "flat",   (0.03, 0.5),  1),
]


def load_panel():
    panel = {}
    for path in sorted(OHLCV.glob("*.csv")):
        t = path.name.split("_")[0]
        ds, o, c = [], [], []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    op, cl = float(r["Open"]), float(r["Close"])
                except (TypeError, ValueError):
                    continue
                if not (math.isfinite(op) and math.isfinite(cl) and op > 0 and cl > 0):
                    continue
                ds.append(r["Date"][:10]); o.append(op); c.append(cl)
        if ds:
            panel[t] = (ds, o, c, {d: i for i, d in enumerate(ds)})
    return panel


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


def zmap(scores):
    vals = list(scores.values())
    if len(vals) < 2:
        return {k: 0.0 for k in scores}
    m = sum(vals) / len(vals)
    s = statistics.pstdev(vals)
    return {k: ((v - m) / s if s > 0 else 0.0) for k, v in scores.items()}


def simulate(t0, dt, panel, sig_dates, zs, tops, kind, param, held_by_day,
             max_dup=1):
    """한 포지션의 (비용차감 총수익, 교체횟수) 또는 None.

    코호트 만기는 D+15 로 고정한다(교체해도 연장 없음) — 그래야 자본 회전이
    현행과 같아 비교가 공평하다. 수익은 구간별 곱으로 이어붙인다.
    판정은 항상 **직전 세션** 스코어만 쓴다(인과).
    """
    rec = panel.get(t0)
    if rec is None:
        return None
    ds, o, c, ix = rec
    i0 = ix.get(dt)
    if i0 is None or i0 + 1 >= len(ds) or i0 + HOLD >= len(ds):
        return None
    entry = o[i0 + 1]
    if entry <= 0:
        return None

    exit_day = ds[i0 + HOLD]
    cur, cur_px = t0, entry
    legs, switches = [], 0

    for d in range(2, HOLD):
        if kind == "none" or switches >= MAX_SWITCH:
            break
        day = ds[i0 + d]
        j = bisect.bisect_right(sig_dates, ds[i0 + d - 1]) - 1
        if j < 0:
            continue
        z, top = zs.get(sig_dates[j]), tops.get(sig_dates[j])
        if not z or cur not in z:
            continue
        crec = panel.get(cur)
        ci = crec[3].get(day) if crec else None
        if ci is None:
            continue
        px_now = crec[2][ci]

        # 1) 교체를 검토할 상태인가
        if kind == "top":
            if cur in top[param[0]]:
                continue
        elif kind == "flat":
            if abs(px_now / cur_px - 1.0) >= param[0]:
                continue
        elif kind == "flattop":
            if abs(px_now / cur_px - 1.0) >= param[0] or cur in top[param[1]]:
                continue

        # 2) 대체 후보 — 그날 최상위 중 미보유·데이터 있는 것
        # 집중 허용도(max_dup)는 **가정이 아니라 정책 파라미터**다.
        #   1  = 중복 금지(35종목 분산 유지)
        #   >1 = 같은 종목을 최대 max_dup 슬롯까지 겹쳐 보유 허용
        #   99 = 무제한 — "계속 좋게 평가되면 몰려도 된다"는 가설의 극단
        best, best_z = None, None
        for cand, cz in sorted(z.items(), key=lambda kv: -kv[1])[:60]:
            if cand == cur or held_by_day.get(cand, 0) >= max_dup:
                continue
            r2 = panel.get(cand)
            if r2 is None:
                continue
            k2 = r2[3].get(day)
            if k2 is None or r2[3].get(exit_day) is None or r2[2][k2] <= 0:
                continue
            best, best_z = cand, cz
            break
        if best is None:
            continue

        # 3) 격차 조건
        gap = best_z - z[cur]
        if kind == "score" and gap < param:
            continue
        if kind == "flat" and gap < param[1]:
            continue
        if kind in ("top", "flattop") and best not in top[0.10]:
            continue

        # 4) 체결 — 현재 종가 매도 → 후보 종가 매수
        legs.append(px_now / cur_px - 1.0)
        r2 = panel.get(best)
        held_by_day[cur] = held_by_day.get(cur, 1) - 1      # 판 종목 슬롯 반환
        if held_by_day[cur] <= 0:
            held_by_day.pop(cur, None)
        held_by_day[best] = held_by_day.get(best, 0) + 1    # 산 종목 슬롯 점유
        cur, cur_px = best, r2[2][r2[3][day]]
        switches += 1

    frec = panel.get(cur)
    fi = frec[3].get(exit_day)
    if fi is None:
        return None
    legs.append(frec[2][fi] / cur_px - 1.0)

    total = 1.0
    for r in legs:
        total *= (1.0 + r)
    cost = (1 + switches) * COST_BP / 1e4
    return total - 1.0 - cost, switches


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r5", "r4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--json", default="")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_panel()

    res, seed_n = {}, {}
    for fold in args.folds:
        ens, n = load_ens(fold, args.seeds, args.prefix)
        if ens is None:
            continue
        seed_n[fold] = n
        sig_dates = sorted(ens)
        zs = {d: zmap(ens[d]) for d in sig_dates}
        tops = {}
        for d in sig_dates:
            ranked = sorted(ens[d], key=lambda t: ens[d][t], reverse=True)
            nn = len(ranked)
            tops[d] = {b: set(ranked[:max(1, int(nn * b))]) for b in (0.10, 0.20, 0.30)}
        picks = {dt: sorted(ens[dt], key=lambda t: ens[dt][t], reverse=True)[:K_PICKS]
                 for dt in ens if len(ens[dt]) >= 50}
        print(f"  {fold}: 시드 {n}, {len(picks)}일 …", flush=True)

        bench = {}
        for dt in picks:
            vals = []
            for t in ens[dt]:
                rec = panel.get(t)
                if rec is None:
                    continue
                ds, o, c, ix = rec
                i = ix.get(dt)
                if i is None or i + 1 >= len(ds) or i + HOLD >= len(ds) or o[i + 1] <= 0:
                    continue
                vals.append(c[i + HOLD] / o[i + 1] - 1.0)
            if len(vals) >= 50:
                bench[dt] = sum(vals) / len(vals)

        for label, kind, param, mdup in POLICIES:
            daily, sw_tot, pos_tot, conc = [], 0, 0, []
            for dt, pk in picks.items():
                if dt not in bench:
                    continue
                # 코호트의 현재 보유 집합 — 초기 35픽으로 시작한다
                held = {t: 1 for t in pk}
                ex = []
                for t in pk:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    ds, o, c, ix = rec
                    i = ix.get(dt)
                    if i is None or i + 1 >= len(ds):
                        continue
                    if abs(o[i + 1] / c[i] - 1.0) > GAP_SKIP:
                        continue
                    r = simulate(t, dt, panel, sig_dates, zs, tops,
                                 kind, param, held, mdup)
                    if r is None:
                        continue
                    ret, sw = r
                    ex.append(ret - bench[dt])
                    sw_tot += sw; pos_tot += 1
                if len(ex) >= K_PICKS // 3:
                    daily.append(sum(ex) / len(ex))
                    conc.append(max(held.values()) if held else 1)
            if len(daily) < 20:
                continue
            mu = sum(daily) / len(daily)
            sd = statistics.stdev(daily)
            turns = TRADING_DAYS / HOLD
            res[(fold, label)] = (mu / sd * math.sqrt(turns) if sd > 0 else float("nan"),
                                  sd * math.sqrt(turns) * 100,
                                  sw_tot / pos_tot if pos_tot else 0.0,
                                  sum(conc) / len(conc) if conc else 1.0)

    folds = [f for f in args.folds if any(k[0] == f for k in res)]
    print("\n" + "=" * 82)
    print(f"갈아타기 연구 (코호트 만기 D+{HOLD} 고정, 교체마다 41bp 추가, 상한 {MAX_SWITCH}회)")
    print("  시드: " + ", ".join(f"{f}={seed_n.get(f,0)}" for f in folds)
          + " | 판정은 직전 세션 스코어(인과)")
    print("=" * 82)
    hdr = f"{'정책':<26}"
    for f in folds:
        hdr += f"{f+' Sh':>9}{f+' 변동%':>8}{'교체':>6}{'최대집중':>8}"
    print(hdr + f"{'평균Sh':>9}")
    print("-" * 82)
    for label, *_ in POLICIES:
        cells, shs = "", []
        for f in folds:
            r = res.get((f, label))
            if r:
                cells += f"{r[0]:>+9.2f}{r[1]:>8.1f}{r[2]:>6.2f}{r[3]:>8.1f}"
                shs.append(r[0])
            else:
                cells += f"{'.':>9}{'.':>8}{'.':>6}{'.':>8}"
        if shs:
            print(f"{label:<26}{cells}{sum(shs)/len(shs):>+9.2f}")
    print("\n변동% = 연환산 변동성. 최대집중 = 코호트당 한 종목 최대 보유 슬롯수(1=분산).")
    print("Sharpe 는 위험조정 후이므로 집중이 이득이면 여기서 이겨야 한다 —")
    print("집중으로 수익만 커지고 변동성이 같이 커지면 Sharpe 는 안 오른다.")
    if args.json:
        import json as _json
        Path(args.json).write_text(_json.dumps({
            "study": "switch", "folds": folds, "seeds": seed_n,
            "rows": {label: {f: res[(f, label)][0] for f in folds if (f, label) in res}
                     for label, *_ in POLICIES},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
