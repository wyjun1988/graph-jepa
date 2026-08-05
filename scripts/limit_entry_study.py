#!/usr/bin/env python3
"""지정가 진입 연구 — 시가 대신 시가-1% 지정가로 사면 나은가.

── 착상 (사용자, 2026-08-06) ───────────────────────────────────────────────
진입일에는 노이즈가 있으니, 갭상승이 아니면 시가에 사지 말고 시가보다 1% 낮은
지정가를 걸어 두는 게 낫지 않은가.

── 트레이드오프 ────────────────────────────────────────────────────────────
이득: 체결되면 진입가가 ~1% 싸다 (D+15 보유에 1%p 는 크다).
비용: (1) 역선택 — 지정가까지 내려와 체결되는 종목은 그날 약한 종목에 편중.
      (2) 기회비용 — 안 내려오고 날아간 종목(=강한 종목)을 통째로 놓친다.
      우리 픽은 기대수익 양수의 상위 10% 라 (2) 가 특히 아플 수 있다.
어느 쪽이 이기는지는 경험 문제다 — 잰다.

── 체결 가정 (한계 포함) ───────────────────────────────────────────────────
지정가 = Open[t+1] × (1−x). 당일 Low[t+1] ≤ 지정가면 지정가에 전량 체결.
일봉이라 주문 시점(시가 직후) 이전의 저가와 구분 못 하고 호가 큐를 무시하므로
**체결에 낙관적**이다. 즉 지정가 정책의 성적은 여기 나온 것보다 좋을 수 없다 —
그런데도 지면 확실히 지는 것이다.

비용은 기존 연구와 동일 41bp (지정가는 메이커라 실제론 더 쌀 수 있음 — 보수적).
청산은 채택안 D+15 종가 고정. 벤치 = 유니버스 평균(시가 진입) — 전 정책 공통.

사용법:
  python scripts/limit_entry_study.py --folds r5 r4
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
HOLD = 15                                  # 채택안 청산 고정

# (라벨, 종류, 할인 x, 미체결 처리, 갭상승 조건부?)
#   종류: open(전량 시가) | limit
#   미체결: skip(포기) | close(종가 추격)
POLICIES = [
    ("시가 전량 (현행)",          "open",  0.000, None,    False),
    ("지정가 -0.5% / 미체결 포기",  "limit", 0.005, "skip",  False),
    ("지정가 -1.0% / 미체결 포기",  "limit", 0.010, "skip",  False),
    ("지정가 -2.0% / 미체결 포기",  "limit", 0.020, "skip",  False),
    ("지정가 -1.0% / 미체결 종가",  "limit", 0.010, "close", False),
    ("갭상승만 시가, 나머지 -1%/포기", "limit", 0.010, "skip",  True),   # ← 사용자 원안
    ("갭상승만 시가, 나머지 -1%/종가", "limit", 0.010, "close", True),
]


def load_panel():
    """{ticker: (dates, open, high, low, close, index)}"""
    panel = {}
    for path in sorted(OHLCV.glob("*.csv")):
        t = path.name.split("_")[0]
        ds, o, h, lo, c = [], [], [], [], []
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                try:
                    vals = [float(r["Open"]), float(r["High"]),
                            float(r["Low"]), float(r["Close"])]
                except (TypeError, ValueError):
                    continue
                if not all(math.isfinite(v) and v > 0 for v in vals):
                    continue
                ds.append(r["Date"][:10])
                o.append(vals[0]); h.append(vals[1]); lo.append(vals[2]); c.append(vals[3])
        if ds:
            panel[t] = (ds, o, h, lo, c, {d: i for i, d in enumerate(ds)})
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


def entry_price(rec, dt, kind, disc, fallback, cond_gap):
    """(진입가, 체결여부, 시가) 또는 None(데이터 부족). 체결여부 False = 포기."""
    ds, o, h, lo, c, ix = rec
    i = ix.get(dt)
    if i is None or i + 1 >= len(ds) or i + HOLD >= len(ds):
        return None
    op = o[i + 1]
    if op <= 0:
        return None
    if kind == "open":
        return op, True, op
    # 갭상승 조건부: 시가가 전일종가보다 높으면(갭상승) 시가 체결
    if cond_gap and op > c[i]:
        return op, True, op
    limit = op * (1.0 - disc)
    if lo[i + 1] <= limit:                 # 낙관적 체결 가정 (docstring 참조)
        return limit, True, op
    if fallback == "close":
        return c[i + 1], True, op
    return None, False, op                 # 미체결 포기


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["r5", "r4"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--json", default="", help="판정기용 결과 JSON 경로")
    args = ap.parse_args()

    print("가격 패널 적재 중 (O/H/L/C) …", flush=True)
    panel = load_panel()

    res, extra, seed_n = {}, {}, {}
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
                ds, o, h, lo, c, ix = rec
                i = ix.get(dt)
                if i is None or i + 1 >= len(ds) or i + HOLD >= len(ds) or o[i + 1] <= 0:
                    continue
                vals.append(c[i + HOLD] / o[i + 1] - 1.0)
            if len(vals) >= 50:
                bench[dt] = sum(vals) / len(vals)

        for label, kind, disc, fb, cg in POLICIES:
            daily, n_fill, n_try, savings, missed_ex = [], 0, 0, [], []
            for dt, pk in picks.items():
                if dt not in bench:
                    continue
                pos = []
                for t in pk:
                    rec = panel.get(t)
                    if rec is None:
                        continue
                    ep = entry_price(rec, dt, kind, disc, fb, cg)
                    if ep is None:
                        continue
                    price, filled, op = ep
                    n_try += 1
                    ds, o, h, lo, c, ix = rec
                    i = ix.get(dt)
                    exit_px = c[i + HOLD]
                    if filled:
                        n_fill += 1
                        pos.append(exit_px / price - 1.0 - bench[dt])
                        if price < op:
                            savings.append(op / price - 1.0)
                    else:
                        # 반사실: 시가에 샀더라면 — 역선택/기회비용의 직접 증거
                        missed_ex.append(exit_px / op - 1.0 - bench[dt])
                if len(pos) >= K_PICKS // 3:
                    daily.append(sum(pos) / len(pos))
            if len(daily) < 20:
                continue
            mu = sum(daily) / len(daily) - COST_BP / 1e4
            sd = statistics.stdev(daily)
            turns = TRADING_DAYS / HOLD
            sh = mu / sd * math.sqrt(turns) if sd > 0 else float("nan")
            res[(fold, label)] = (sh, mu * turns * 100)
            extra[(fold, label)] = (
                n_fill / n_try if n_try else 0.0,
                (sum(savings) / len(savings) * 1e4) if savings else 0.0,
                (sum(missed_ex) / len(missed_ex) * 100) if missed_ex else None,
                len(missed_ex),
            )

    folds = [f for f in args.folds if any(k[0] == f for k in res)]
    print("\n" + "=" * 96)
    print(f"지정가 진입 연구 (청산 D+{HOLD} 고정, 41bp, 벤치=유니버스 시가진입)")
    print("  시드: " + ", ".join(f"{f}={seed_n.get(f,0)}" for f in folds)
          + " | 체결가정 낙관적(일봉 Low) — 지정가 성적의 상한이다")
    print("=" * 96)
    hdr = f"{'정책':<26}"
    for f in folds:
        hdr += f"{f+' Sh':>8}{f+' 연%':>8}{'체결률':>7}{'절약bp':>7}{'놓친초과%':>9}"
    print(hdr + f"{'평균Sh':>8}")
    print("-" * 96)
    for label, *_ in POLICIES:
        cells, shs = "", []
        for f in folds:
            r = res.get((f, label)); e = extra.get((f, label))
            if r and e:
                fr, sv, mx, nm = e
                mxs = f"{mx:>+9.2f}" if mx is not None else f"{'—':>9}"
                cells += f"{r[0]:>+8.2f}{r[1]:>+8.1f}{fr*100:>6.0f}%{sv:>7.0f}{mxs}"
                shs.append(r[0])
            else:
                cells += f"{'.':>8}{'.':>8}{'.':>7}{'.':>7}{'.':>9}"
        if shs:
            print(f"{label:<26}{cells}{sum(shs)/len(shs):>+8.2f}")
    print("\n놓친초과% = 미체결로 포기한 종목을 시가에 샀더라면의 평균 초과수익(D+15).")
    print("  양수·크면 강한 종목을 놓치고 있다는 뜻 = 기회비용이 절약분을 먹는다.")
    if args.json:
        import json as _json
        Path(args.json).write_text(_json.dumps({
            "study": "limit_entry", "folds": folds, "seeds": seed_n,
            "rows": {label: {f: res[(f, label)][0] for f in folds if (f, label) in res}
                     for label, *_ in POLICIES},
        }, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"[json] {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
