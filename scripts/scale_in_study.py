#!/usr/bin/env python3
"""물타기(스케일인) 연구 — 진입 후 빠지면 더 사는 게 이득인가.

── 착상 (사용자, 2026-08-06) ───────────────────────────────────────────────
어차피 며칠 뒤 오른다고 예측한 종목이니, 다음날 우리 매수가보다 떨어지면
더 사서 같이 파는 게 이득 아닌가.

앞서 기각된 두 연구(지정가 진입·갭하락 지연진입)와 성격이 다르다. 그것들은
**진입을 미뤄** 기대 드리프트를 버렸지만, 물타기는 **추가로 산다** — 드리프트를
버리지 않는다. 따로 잴 값어치가 있다.

── 그러나 공짜가 아니다 ────────────────────────────────────────────────────
1. **자본이 더 든다.** 같은 종목에 2배를 실으면 수익률이 아니라 레버리지를 잰다.
   vol스케일 북에서 이미 겪은 함정이다. → 아래 두 방식을 **분리해서** 잰다:
     scale_up : 초기 1.0 + 추가 1.0 = 총 2.0  (자본 증가 — 레버리지 섞임)
     half_half: 초기 0.5 + 추가 0.5 = 총 1.0  (자본 동일 — 순수 타이밍 효과)
   half_half 가 진짜 비교다. scale_up 이 이겨도 그건 "더 실어서" 일 수 있다.
2. **새 정보의 의미.** 진입 후 하락은 "우리 예측이 아직 틀리다"는 신호다.
   그게 평균회귀(사면 이득)인지 모멘텀(사면 손해)인지는 경험 문제다.
3. **미실행 위험.** 안 빠지면 추가 매수가 안 일어난다 — 그 경우 half_half 는
   0.5 만 실은 채로 끝나 언더인베스트가 된다. 그 손실도 함께 잰다.

── 방법 ────────────────────────────────────────────────────────────────────
진입 Open[t+1]. 이후 W 세션 안에 종가가 진입가×(1−x) 이하로 내려가면 그 종가에
추가 매수. 청산은 채택안 D+15 종가에 전량(가중평균 단가 기준). 비용 41bp 는
**매수 회차마다** 부과한다(2회 매수면 2회분).

벤치 = 유니버스 평균(시가 1회 진입) — 전 정책 공통.

사용법:
  python scripts/scale_in_study.py --folds r5 r4
"""

import argparse
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

# (라벨, 방식, 트리거 하락폭 x, 관찰창 W)
#   base      : 시가 1회 (현행)
#   scale_up  : 1.0 + 1.0  (자본 2배)
#   half_half : 0.5 + 0.5  (자본 동일)
POLICIES = [
    ("시가 1회 (현행)",              "base",      None, None),
    ("scale_up -1% / 3일",         "scale_up",  0.01, 3),
    ("scale_up -3% / 5일",         "scale_up",  0.03, 5),
    ("scale_up -5% / 5일",         "scale_up",  0.05, 5),
    ("half_half -1% / 3일",        "half_half", 0.01, 3),
    ("half_half -3% / 5일",        "half_half", 0.03, 5),
    ("half_half -5% / 5일",        "half_half", 0.05, 5),
    ("half_half -3% / 10일",       "half_half", 0.03, 10),
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


def position_return(rec, dt, mode, x, w):
    """(가중 수익률, 투입자본배수, 추가매수여부) 또는 None.

    수익률은 '투입 1.0 당' 이 아니라 **초기 1.0 기준 손익**으로 돌려준다.
    scale_up 은 자본이 2배 들어가므로 손익도 2배 규모가 되고, 그 사실이
    capital 배수로 함께 나가 아래에서 자본조정 비교가 가능하다.
    """
    ds, o, c, ix = rec
    i = ix.get(dt)
    if i is None or i + 1 >= len(ds) or i + HOLD >= len(ds):
        return None
    entry = o[i + 1]
    if entry <= 0:
        return None
    exit_px = c[i + HOLD]
    cost = COST_BP / 1e4
    if mode == "base":
        return (exit_px / entry - 1.0 - cost), 1.0, False

    w0 = 1.0 if mode == "scale_up" else 0.5
    w1 = 1.0 if mode == "scale_up" else 0.5
    trigger = entry * (1.0 - x)
    add_px = None
    for d in range(1, min(w, HOLD - 1) + 1):
        if c[i + d] <= trigger:
            add_px = c[i + d]
            break
    if add_px is None:                       # 추가 미실행
        pnl = w0 * (exit_px / entry - 1.0 - cost)
        return pnl, w0, False
    pnl = (w0 * (exit_px / entry - 1.0 - cost)
           + w1 * (exit_px / add_px - 1.0 - cost))
    return pnl, w0 + w1, True


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

        # ⚠️ 정책 일치 벤치 — 유니버스 전체에 **같은 물타기 규칙**을 적용한 평균.
        # 물타기 이득이 시장 전체의 단기 반등이라면 이 벤치도 같이 올라가므로
        # 초과수익이 사라진다. 그것이 "우리 알파냐 시장 효과냐"를 가른다.
        pbench = {}
        for dt in picks:
            for label, mode, x, w in POLICIES:
                if mode == "base":
                    continue
                num = den = 0.0
                for t in ens[dt]:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    ds, o, c, ix = rec
                    i = ix.get(dt)
                    if i is None or i + 1 >= len(ds):
                        continue
                    if abs(o[i + 1] / c[i] - 1.0) > GAP_SKIP:
                        continue
                    rr = position_return(rec, dt, mode, x, w)
                    if rr is None:
                        continue
                    pl, cp, _ = rr
                    num += pl; den += cp
                if den > 0:
                    pbench[(dt, label)] = num / den

        for label, mode, x, w in POLICIES:
            daily, daily_r, daily_p, caps, adds, tries = [], [], [], [], 0, 0
            for dt, pk in picks.items():
                if dt not in bench:
                    continue
                pnl_sum, pnl_sum_p, cap_sum, cnt = 0.0, 0.0, 0.0, 0
                for t in pk:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    ds, o, c, ix = rec
                    i = ix.get(dt)
                    if i is None or i + 1 >= len(ds):
                        continue
                    if abs(o[i + 1] / c[i] - 1.0) > GAP_SKIP:     # 프로덕션 갭 가드
                        continue
                    r = position_return(rec, dt, mode, x, w)
                    if r is None:
                        continue
                    pnl, cap, added = r
                    # 벤치는 '투입자본 1.0 기준' 이므로 같은 배수로 맞춘다
                    pnl_sum += pnl - cap * bench[dt]
                    pb = pbench.get((dt, label), bench[dt])
                    pnl_sum_p += pnl - cap * pb
                    cap_sum += cap
                    cnt += 1
                    tries += 1
                    adds += 1 if added else 0
                if cnt >= K_PICKS // 3 and cap_sum > 0:
                    # ⚠️ 분모가 판정을 가른다.
                    #   사용자본: 실제 투입분. 트리거된 것만 2배로 세므로
                    #            "떨어진 놈에만 더 실었다"는 선택 효과가 그대로 남는다.
                    #   예약자본: 전략을 돌리려면 추가분을 **미리 잡아둬야** 한다
                    #            (어느 종목이 트리거될지 사전에 모른다). 놀린 현금도
                    #            자본이다 — 이것이 현행(항상 1.0 투입)과의 공정한 비교다.
                    reserve = (2.0 if mode == "scale_up"
                               else 1.0 if mode == "half_half" else 1.0)
                    daily.append(pnl_sum / cap_sum)
                    daily_r.append(pnl_sum / (reserve * cnt))
                    daily_p.append(pnl_sum_p / (reserve * cnt))
                    caps.append(cap_sum / cnt)
            if len(daily) < 20:
                continue
            turns = TRADING_DAYS / HOLD
            def sh(xs):
                m, d = sum(xs) / len(xs), statistics.stdev(xs)
                return (m / d * math.sqrt(turns) if d > 0 else float("nan"), m * turns * 100)
            s_used, a_used = sh(daily)
            s_res, _ = sh(daily_r)
            s_pol, _ = sh(daily_p)
            res[(fold, label)] = (s_used, s_pol, s_res,
                                  sum(caps) / len(caps),
                                  adds / tries * 100 if tries else 0.0)

    folds = [f for f in args.folds if any(k[0] == f for k in res)]
    print("\n" + "=" * 86)
    print(f"물타기 연구 (청산 D+{HOLD}, 매수 회차마다 41bp, 자본 1.0당 초과수익으로 정규화)")
    print("  시드: " + ", ".join(f"{f}={seed_n.get(f,0)}" for f in folds))
    print("=" * 86)
    hdr = f"{'정책':<22}"
    for f in folds:
        hdr += f"{f+' 예약':>8}{f+' 정책벤치':>10}{'자본':>6}{'추가율':>7}"
    print(hdr + f"{'평균예약':>9}{'평균정책':>9}")
    print("-" * 86)
    for label, *_ in POLICIES:
        cells, su, sr = "", [], []
        for f in folds:
            r = res.get((f, label))
            if r:
                cells += f"{r[2]:>+8.2f}{r[1]:>+10.2f}{r[3]:>6.2f}{r[4]:>6.0f}%"
                su.append(r[2]); sr.append(r[1])
            else:
                cells += f"{'.':>8}{'.':>10}{'.':>6}{'.':>7}"
        if su:
            print(f"{label:<22}{cells}{sum(su)/len(su):>+9.2f}{sum(sr)/len(sr):>+9.2f}")
    print("\n예약 = 미리 잡아둬야 하는 자본 기준(현행과 공정 비교).")
    print("정책벤치 = 유니버스 전체에 **같은 물타기 규칙**을 적용한 벤치 대비 초과.")
    print("  이 값이 현행(+0.60)을 못 넘으면 물타기 이득은 우리 알파가 아니라")
    print("  시장 전체의 단기 반등이다 — 아무 종목에나 해도 되는 것이라 우위가 아니다.")
    print("  트리거될 종목을 사전에 모르므로 추가분은 항상 예약해야 한다(놀린 현금도 자본).")
    print("  **예약 기준이 현행과의 공정한 비교**다. 사용 기준은 '떨어진 놈에만 더 실었다'는")
    print("  선택 효과가 남아 부풀려진다.")
    if args.json:
        import json as _json
        Path(args.json).write_text(_json.dumps({
            "study": "scale_in", "folds": folds, "seeds": seed_n,
            # 주판정 지표는 **정책일치 벤치 기준**(res[..][1])이다.
            # 사용자본 기준은 선택효과가 남아 부풀려지므로 판정에 쓰지 않는다.
            "rows": {label: {f: res[(f, label)][1] for f in folds if (f, label) in res}
                     for label, *_ in POLICIES},
            "rows_reserve": {label: {f: res[(f, label)][2] for f in folds if (f, label) in res}
                             for label, *_ in POLICIES},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
