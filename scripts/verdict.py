#!/usr/bin/env python3
"""사전등록 판정기 — 표를 사람이 읽지 않아도 결론이 나오게.

각 스터디가 `--json` 으로 뱉은 결과를 읽어 **사전등록된 게이트**를 코드로 판정한다.
사람은 맨 아래 요약 몇 줄만 전달하면 된다.

게이트는 전부 2026-08-02~03 에 r1~r3 미관측 상태로 등록됐다. 여기서 임계를
바꾸면 사후조정이므로, 값은 상수로 박아두고 출력에 함께 찍는다.

사용법:
  python scripts/verdict.py --dir <json 디렉터리>
"""

import argparse
import json
import sys
from pathlib import Path

# ── 사전등록 상수 (변경 금지 — 바꾸면 사후조정이다) ────────────────────────
PREREG = {
    "rank_exit": {
        "date": "2026-08-02",
        "main": "랭크20 h10/h10",
        "base": "D+15 (채택안)",
        "obs2": "2폴드 사전관측 +1.98 vs +0.57",
    },
    "vol_deploy": {
        "date": "2026-08-03",
        "main": "프로덕션 K+0.3 uni",
        "base": "고정 (m=1)",
        "alt": ["K-0.30 (역방향)", "K-0.60 (역방향)"],
        "obs2": "2폴드 사전관측 K+0.3 이 고정에 짐(d15 0.43<0.57, rank20 1.42<1.98)",
    },
    "vol_holding": {
        "date": "2026-08-03",
        "obs2": "2폴드 사전관측 저국면·고종목 D+5 정점 / 저국면·저종목 D+15 정점",
    },
    "scale_in": {
        "date": "2026-08-06",
        "main": "scale_up -5% / 5일",   # = half_half -5%/5일 (자본정규화 후 동일)
        "base": "시가 1회 (현행)",
        "obs2": "2폴드 사전관측 예약자본 +0.93 / 정책일치벤치 +0.83 vs 현행 +0.60",
    },
    "signal_scaling": {
        "date": "2026-08-06",
        "main": "예산×meantop K+0.6",
        "base": "고정 (현행)",
        "obs2": "2폴드 사전관측 +0.83 vs +0.60, 2/2. 시장변동성 버전은 5폴드 마진 +0.05",
    },
    "combined": {
        "date": "2026-08-06",
        "main": "전부 (guard)",
        "base": "랭크청산만",     # ⚠️ 기준선은 현행이 아니라 개별 최고다
        "obs2": "2폴드 사전관측 +2.47 vs 랭크청산 +1.92 (현행 +0.60). 가산성 98%",
    },
    "switch": {
        "date": "2026-08-06",
        "main": "top30→top10 (중복 3까지)",
        "base": "현행 (교체 없음)",
        "obs2": "2폴드 사전관측 +1.28 vs +0.60. 단 변동성이 3배(12~18% -> 42~52%)",
    },
}
MIN_FOLDS_FOR_ADOPT = 5      # 5폴드 미만이면 '승급'까지만


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(v, w=6):
    return f"{v:+.2f}".rjust(w) if v is not None else "   .  "


def load(d, name):
    p = Path(d) / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception:
        return None


def verdict_rank_exit(data, out):
    pr = PREREG["rank_exit"]
    rows, folds = data["rows"], data["folds"]
    main, base = rows.get(pr["main"], {}), rows.get(pr["base"], {})
    if not main or not base:
        out.append(("랭크청산", "판정불가", "주판정/기준 행 없음"))
        return
    mm, bm = mean(main.values()), mean(base.values())
    wins = [f for f in folds if f in main and f in base and main[f] > base[f]]
    worst_m = min(main.values())
    worst_b = min(base.values())
    g_mean = mm > bm
    g_worst = worst_m >= worst_b
    g_consist = len(wins) == len([f for f in folds if f in main and f in base])
    n = len(folds)
    if g_mean and g_worst and g_consist and n >= MIN_FOLDS_FOR_ADOPT:
        v, why = "채택권고", f"평균·최악폴드·전폴드일관 모두 통과 ({n}폴드)"
    elif g_mean and g_worst and n >= MIN_FOLDS_FOR_ADOPT:
        v, why = "조건부채택", f"평균·최악폴드 통과, 폴드일관 {len(wins)}/{n}"
    elif g_mean and n >= MIN_FOLDS_FOR_ADOPT:
        v, why = "보류", f"평균은 이기나 최악폴드 악화 ({worst_m:+.2f} < {worst_b:+.2f})"
    elif g_mean:
        v, why = "승급", f"{n}폴드에서 평균 우세 — 5폴드 확인 필요"
    else:
        v, why = "기각", f"평균 미달 ({mm:+.2f} <= {bm:+.2f})"
    out.append(("랭크청산", v, why))
    out.append(("", "", f"주판정 {mm:+.2f} vs 기준 {bm:+.2f} | "
                       f"최악폴드 {worst_m:+.2f} vs {worst_b:+.2f} | 우세 {len(wins)}/{n}폴드"))
    # 폭 스윕 참고 (사후선택 금지 — 표시만)
    widths = {k: mean(v.values()) for k, v in rows.items()
              if k.startswith("랭크") and "h10/h10" in k}
    if len(widths) > 2:
        best = max(widths, key=lambda k: widths[k])
        out.append(("", "", f"폭 참고: 최고 {best} {widths[best]:+.2f} "
                            f"(사전등록은 {pr['main']} — 사후 변경 금지)"))


def verdict_vol_deploy(data, out, tag):
    pr = PREREG["vol_deploy"]
    rows, folds = data["rows"], data["folds"]
    main, base = rows.get(pr["main"], {}), rows.get(pr["base"], {})
    if not main or not base:
        out.append((f"배분조절({tag})", "판정불가", "행 없음"))
        return
    mm, bm = mean(main.values()), mean(base.values())
    alts = {k: mean(rows[k].values()) for k in pr["alt"] if k in rows}
    best_alt = max(alts, key=lambda k: alts[k]) if alts else None
    n = len(folds)
    if mm > bm:
        v, why = ("채택권고" if n >= MIN_FOLDS_FOR_ADOPT else "승급",
                  f"K+0.3 이 고정을 이김 ({mm:+.2f} > {bm:+.2f})")
    elif best_alt and alts[best_alt] > bm:
        v, why = ("부호전환 권고" if n >= MIN_FOLDS_FOR_ADOPT else "부호전환 후보",
                  f"K+0.3 기각({mm:+.2f}<={bm:+.2f}), 역방향 {best_alt} {alts[best_alt]:+.2f} > 고정")
    else:
        v, why = "배분조절 제거 권고", f"K+0.3({mm:+.2f})·역방향 모두 고정({bm:+.2f})을 못 넘음"
    out.append((f"배분조절({tag})", v, why))


def verdict_vol_holding(data, out):
    cross, folds = data.get("cross", {}), data["folds"]
    holds = [str(h) for h in data["holds"]]

    def peak(cell, fold=None):
        c = cross.get(cell, {})
        series = ({h: mean([c[f][h] for f in c if h in c[f]]) for h in holds}
                  if fold is None else
                  {h: c.get(fold, {}).get(h) for h in holds})
        series = {h: v for h, v in series.items() if v is not None}
        return max(series, key=lambda h: series[h]) if series else None

    hi, lo = "저국면·고종목", "저국면·저종목"
    p_hi, p_lo = peak(hi), peak(lo)
    if p_hi is None or p_lo is None:
        out.append(("변동성x보유", "판정불가", "2x2 셀 없음"))
        return
    ok_mean = int(p_hi) < int(p_lo)
    per_fold = []
    for f in folds:
        a, b = peak(hi, f), peak(lo, f)
        if a and b:
            per_fold.append(int(a) < int(b))
    consist = per_fold and all(per_fold)
    n = len(folds)
    if ok_mean and consist and n >= MIN_FOLDS_FOR_ADOPT:
        v, why = "채택권고", f"고종목 D+{p_hi} < 저종목 D+{p_lo}, 전폴드 일관"
    elif ok_mean and n >= MIN_FOLDS_FOR_ADOPT:
        v, why = "보류", (f"평균은 성립(D+{p_hi} < D+{p_lo})이나 폴드 일관 "
                        f"{sum(per_fold)}/{len(per_fold)}")
    elif ok_mean:
        v, why = "승급", f"{n}폴드 평균 성립 — 5폴드 확인 필요"
    else:
        v, why = "기각", f"고종목 D+{p_hi} >= 저종목 D+{p_lo}"
    out.append(("변동성x보유", v, why))
    lows = [mean(cross[c][f].values()) for c in (hi, lo) for f in folds
            if c in cross and f in cross[c]]
    out.append(("", "", f"저국면 셀 평균 {mean(lows):+.2f} — 양수면 저변동 국면에서만 전략이 먹는다"))


def verdict_simple(data, out, key, name):
    """평균 + 최악폴드 + 폴드일관 3게이트 (랭크청산과 동일 기준)."""
    pr = PREREG[key]
    rows, folds = data["rows"], data["folds"]
    main, base = rows.get(pr["main"], {}), rows.get(pr["base"], {})
    if not main or not base:
        out.append((name, "판정불가", "주판정/기준 행 없음"))
        return
    mm, bm = mean(main.values()), mean(base.values())
    pairs = [f for f in folds if f in main and f in base]
    wins = [f for f in pairs if main[f] > base[f]]
    wm, wb = min(main.values()), min(base.values())
    n = len(folds)
    if mm > bm and wm >= wb and len(wins) == len(pairs) and n >= MIN_FOLDS_FOR_ADOPT:
        v, why = "채택권고", f"평균·최악폴드·전폴드일관 통과 ({n}폴드)"
    elif mm > bm and wm >= wb and n >= MIN_FOLDS_FOR_ADOPT:
        v, why = "조건부채택", f"평균·최악폴드 통과, 폴드일관 {len(wins)}/{len(pairs)}"
    elif mm > bm and n >= MIN_FOLDS_FOR_ADOPT:
        v, why = "보류", f"평균 우세이나 최악폴드 악화 ({wm:+.2f} < {wb:+.2f})"
    elif mm > bm:
        v, why = "승급", f"{n}폴드 평균 우세 — 5폴드 확인 필요"
    else:
        v, why = "기각", f"평균 미달 ({mm:+.2f} <= {bm:+.2f})"
    out.append((name, v, why))
    out.append(("", "", f"주판정 {mm:+.2f} vs 기준 {bm:+.2f} | "
                       f"최악폴드 {wm:+.2f} vs {wb:+.2f} | 우세 {len(wins)}/{len(pairs)}폴드"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="스터디가 --json 으로 뱉은 디렉터리")
    args = ap.parse_args()

    out = []
    n_folds = None
    d = load(args.dir, "rank_exit")
    if d:
        n_folds = len(d["folds"])
        verdict_rank_exit(d, out)
    for tag in ("d15", "rank20"):
        d = load(args.dir, f"vol_deploy_{tag}")
        if d:
            n_folds = n_folds or len(d["folds"])
            verdict_vol_deploy(d, out, tag)
    d = load(args.dir, "vol_holding")
    if d:
        n_folds = n_folds or len(d["folds"])
        verdict_vol_holding(d, out)
    for key, nm in (("scale_in", "물타기"), ("switch", "갈아타기"),
                    ("signal_scaling", "신호강도조절"), ("combined", "결합전략")):
        d = load(args.dir, key)
        if d:
            n_folds = n_folds or len(d["folds"])
            verdict_simple(d, out, key, nm)

    if not out:
        print("판정할 JSON 이 없다. 스터디를 --json 으로 먼저 돌려라.")
        return 1

    print("\n" + "=" * 74)
    print("사전등록 판정 결과")
    print(f"  폴드 수 {n_folds} | 채택 요건 {MIN_FOLDS_FOR_ADOPT}폴드"
          + ("" if (n_folds or 0) >= MIN_FOLDS_FOR_ADOPT
             else "  ⚠️ 폴드 부족 — '승급'까지만 가능"))
    print("=" * 74)
    for name, v, why in out:
        if name:
            print(f"  {name:<16}{v:<14}{why}")
        else:
            print(f"  {'':<16}{'':<14}{why}")
    print("=" * 74)
    print("이 블록만 전달하면 된다. 상세 표는 같은 디렉터리의 *.json 에 있다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
