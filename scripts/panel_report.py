#!/usr/bin/env python3
"""6시드 x 5폴드 패널 종합 — 화면으로만 결과를 나르는 환경용.

파일을 빼올 수 없고 화면을 손으로 옮겨 적어야 하는 상황을 전제로 만들었다.
그래서 원칙이 둘이다:

  1) 분석을 데이터가 있는 쪽(GPU 머신)에서 끝낸다. 숫자를 나르지 말고 결론을 나른다.
  2) 가장 중요한 블록을 **맨 마지막에** 찍는다. 위로 스크롤해 사라지지 않게.

블록은 중요도 역순이다 — [C] 선택 → [B] 권장 → [A] 필수.
[A] 10개만 적어도 핵심 판정은 난다.

── 무엇을 판정하나 ──────────────────────────────────────────────────────
5폴드 패널(시드 3/17/29)은 평균 Sharpe +0.35, t=2.13 (p=0.100) 이었다.
그런데 r5 의 6시드를 펼쳐 보니 s3 단독이 +0.87 이고 6시드 전부 쓰면 -0.08 이다.
즉 그 조합은 6개 중 최고 시드를 품고 있었다.

  [A] 6시드 앙상블의 폴드별 IC·Sharpe → 시드운을 뺀 진짜 수준
  [B] 청산 D+10/15/20/30           → 총알파의 69%를 먹는 비용을 줄일 수 있나
  [C] 시드x폴드 격자와 순위 상관     → 좋은 시드가 어디서나 좋은가

[C] 의 상관이 0 근처면 시드운은 폴드마다 독립이므로 r5 만 부풀려진 것이고,
+0.5 이상이면 다섯 폴드가 전부 같은 방향으로 편향돼 있다는 뜻이다.

사용법:
  python scripts/panel_report.py
  python scripts/panel_report.py --seeds 3 5 11 17 23 29 --folds r1 r2 r3 r4 r5
"""

import argparse
import csv
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from exit_policy_report import load_prices, path_return  # noqa: E402

NODE_EVAL = ROOT / "reports" / "walk_forward" / "node_eval"
FOLDS = {
    "r1": "fold1_20220510_to_20230306",
    "r2": "fold1_20230307_to_20240105",
    "r3": "fold1_20240104_to_20241107",
    "r4": "fold1_20241106_to_20250908",
    "r5": "fold1_20250905_to_20260710",
}
HORIZON, TOP_N, K = 10, 100, 20
COST_BP, TRADING_DAYS = 41.0, 252
POLICIES = [10, 15, 20, 30]
# 시드 3/17/29 로 이미 확인된 값 — 대조군
REF = {"r1": (0.0921, 0.04), "r2": (0.0764, 0.24), "r3": (0.0637, 0.94),
       "r4": (0.0380, 0.09), "r5": (0.0588, 0.42)}


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


def spearman(a, b):
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        for pos, i in enumerate(order):
            r[i] = float(pos)
        return r
    return pearson(rank(a), rank(b))


def sharpe_of(excess, horizon):
    """기간초과수익 시계열 → 비용차감 연환산 Sharpe."""
    if len(excess) < 10:
        return float("nan"), float("nan")
    m = sum(excess) / len(excess)
    sd = statistics.stdev(excess)
    turns = TRADING_DAYS / horizon
    net = (m - COST_BP / 1e4) * turns
    return (net / (sd * math.sqrt(turns)) if sd > 0 else float("nan")), net


def load_seed(fold, seed, prefix="ens_s"):
    """{date: {ticker: (pred, liq)}} — 지평 10 만."""
    p = NODE_EVAL / f"{prefix}{seed}_{FOLDS[fold]}" / "return_1d_forecasts.csv"
    if not p.exists():
        return None
    by = {}
    with open(p, newline="") as f:
        for row in csv.DictReader(f):
            if row["horizon"] != str(HORIZON):
                continue
            try:
                pred = float(row["prediction_entry_path_return"])
                liq = float(row["current_value_ma20_log"])
            except (TypeError, ValueError):
                continue
            if math.isfinite(pred) and math.isfinite(liq):
                by.setdefault(row["date"], {})[row["ticker"]] = (pred, liq)
    return by


def score(seed_maps, panel, policies=(HORIZON,)):
    """앙상블(또는 단일) 예측으로 IC 와 정책별 초과수익 시계열을 낸다."""
    dates = sorted(set.intersection(*(set(m) for m in seed_maps)))
    ics = []
    excess = {h: [] for h in policies}
    for d in dates:
        tk = set.intersection(*(set(m[d]) for m in seed_maps))
        base = seed_maps[0][d]
        uni = sorted((t for t in tk), key=lambda t: base[t][1], reverse=True)[:TOP_N]
        if len(uni) < 30:
            continue
        pred = {t: sum(m[d][t][0] for m in seed_maps) / len(seed_maps) for t in uni}
        rz = {t: path_return(panel.get(t), d, HORIZON) for t in uni}
        ok = [t for t in uni if rz[t] is not None]
        if len(ok) >= 30:
            c = pearson([pred[t] for t in ok], [rz[t] for t in ok])
            if math.isfinite(c):
                ics.append(c)
        picks = sorted(uni, key=lambda t: pred[t], reverse=True)[:K]
        for h in policies:
            pv = [path_return(panel.get(t), d, h) for t in picks]
            uv = [path_return(panel.get(t), d, h) for t in uni]
            pv = [v for v in pv if v is not None]
            uv = [v for v in uv if v is not None]
            if len(pv) >= K // 2 and len(uv) >= 30:
                excess[h].append(sum(pv) / len(pv) - sum(uv) / len(uv))
    return (sum(ics) / len(ics) if ics else float("nan")), excess, len(dates)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 5, 11, 17, 23, 29])
    ap.add_argument("--folds", nargs="+", default=["r1", "r2", "r3", "r4", "r5"])
    ap.add_argument("--prefix", default="ens_s")
    ap.add_argument("--brief", action="store_true",
                    help="[A] 필수 블록만 — 화면 복사가 짧아야 할 때")
    args = ap.parse_args()

    print("가격 패널 적재 중 …", flush=True)
    panel = load_prices()
    print(f"  종목 {len(panel)}개\n", flush=True)

    ens, grid, avail_by_fold = {}, {}, {}
    for fold in args.folds:
        maps = {}
        for s in args.seeds:
            m = load_seed(fold, s, args.prefix)
            if m is not None:
                maps[s] = m
        if not maps:
            print(f"  {fold}: 런 없음 — 건너뜀", flush=True)
            continue
        avail_by_fold[fold] = sorted(maps)
        print(f"  {fold}: 시드 {sorted(maps)} 채점 중 …", flush=True)
        ic, exc, ndays = score([maps[s] for s in sorted(maps)], panel, POLICIES)
        ens[fold] = {"ic": ic, "days": ndays,
                     "sh": {h: sharpe_of(exc[h], h)[0] for h in POLICIES},
                     "net": {h: sharpe_of(exc[h], h)[1] for h in POLICIES}}
        grid[fold] = {}
        if not args.brief:      # 시드별 재채점은 [C] 격자에만 쓴다 — brief 면 건너뛴다
            for s in sorted(maps):
                _, e1, _ = score([maps[s]], panel, (HORIZON,))
                grid[fold][s] = sharpe_of(e1[HORIZON], HORIZON)[0]
    if not ens:
        print("채점할 런이 없습니다.")
        return 1
    folds = [f for f in args.folds if f in ens]

    def avg(v):
        v = [x for x in v if math.isfinite(x)]
        return sum(v) / len(v) if v else float("nan")

    if not args.brief:
        # ══ [C] 선택 ═══════════════════════════════════════════════════
        print("\n[C] 시드 x 폴드 Sharpe 격자 (D+10) — 여유되면")
        print("seed  " + "".join(f"{f:>8}" for f in folds))
        for s in args.seeds:
            # 손으로 옮겨 적는 화면이다. 빈 칸은 nan 대신 점 — 0 으로 오독되면 안 된다.
            cells = "".join(
                f"{grid[f][s]:>+8.2f}" if math.isfinite(grid[f].get(s, float("nan")))
                else f"{'.':>8}" for f in folds)
            print(f"{s:>4}  {cells}")
        pairs = []
        for i in range(len(folds)):
            for j in range(i + 1, len(folds)):
                a, b = folds[i], folds[j]
                common = [s for s in args.seeds
                          if math.isfinite(grid[a].get(s, float("nan")))
                          and math.isfinite(grid[b].get(s, float("nan")))]
                if len(common) >= 4:
                    pairs.append(spearman([grid[a][s] for s in common],
                                          [grid[b][s] for s in common]))
        if pairs:
            print(f"  폴드쌍 시드순위 상관 {sum(pairs)/len(pairs):+.2f} ({len(pairs)}쌍)"
                  "  — 0 근처면 시드운은 폴드마다 독립")

        # ══ [B] 권장 ═══════════════════════════════════════════════════
        print("\n[B] 청산정책 — 앙상블 Sharpe (41bp 차감) / 연수익%")
        print("정책  " + "".join(f"{f:>8}" for f in folds) + f"{'평균':>8}{'연%':>8}")
        for h in POLICIES:
            row = [ens[f]["sh"][h] for f in folds]
            print(f"D+{h:<3}" + "".join(f"{v:>+8.2f}" for v in row)
                  + f"{avg(row):>+8.2f}"
                  + f"{avg([ens[f]['net'][h] for f in folds])*100:>+8.1f}")

    # ══ [A] 필수 — 맨 마지막 ═══════════════════════════════════════════
    print("\n" + "█" * 60)
    print("[A] 이것만은 꼭 적어주세요 — 6시드 앙상블")
    print("█" * 60)
    print("            " + "".join(f"{f:>9}" for f in folds))
    print("  IC        " + "".join(f"{ens[f]['ic']:>+9.4f}" for f in folds))
    print("  Sh D+10   " + "".join(f"{ens[f]['sh'][10]:>+9.2f}" for f in folds))
    print("  Sh D+15   " + "".join(f"{ens[f]['sh'][15]:>+9.2f}" for f in folds))
    print("  일수      " + "".join(f"{ens[f]['days']:>9d}" for f in folds))
    mic = avg([ens[f]["ic"] for f in folds])
    m10 = avg([ens[f]["sh"][10] for f in folds])
    m15 = avg([ens[f]["sh"][15] for f in folds])
    print(f"\n  {len(folds)}폴드 평균 :  IC {mic:+.4f}"
          f"   Sharpe D+10 {m10:+.2f}   D+15 {m15:+.2f}")
    rf = [f for f in folds if f in REF]
    print(f"  대조(3/17/29, D+10):  IC {avg([REF[f][0] for f in rf]):+.4f}"
          f"   Sharpe {avg([REF[f][1] for f in rf]):+.2f}")
    print("\n  시드 구성: " + " | ".join(
        f"{f}:{len(avail_by_fold[f])}개" for f in folds))
    print("█" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
