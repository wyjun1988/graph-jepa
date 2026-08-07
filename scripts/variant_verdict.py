#!/usr/bin/env python3
"""모델 변형 판정기 — 변형(ts_s/fr_s/epc_s/hz_s)이 ens_s 를 이겼는지 코드로 판정.

`verdict.py` 는 **청산·사이징 정책**을 판정한다. 이건 **모델 변형**용이다.
사람이 S6 의 [C]·[A]·[A2] 표 세 덩어리를 눈으로 대조하던 자리를 대신한다.

입력은 각 스터디가 `--json` 으로 뱉은 파일이다:
  <dir>/paired_<variant>_<fold>.json   (paired_variant_report — h10 IC)
  <dir>/sl_<prefix>.json               (sl_exit_study        — 시간청산 스택)
  <dir>/rank_<prefix>.json             (rank_exit_study      — 랭크청산 스택)

사용법:
  python scripts/variant_verdict.py --dir <json 디렉터리> --variant ts_s
"""

import argparse
import json
import sys
from pathlib import Path

# ── 사전등록 상수 (변경 금지 — 바꾸면 사후조정이다) ────────────────────────
SD_SEED = 0.0159          # 지평10 시드 σ, n=6 (docs/MEASUREMENT_CORRECTIONS_20260730.md)

PREREG = {
    "ts_s": {
        "date": "2026-08-07",
        "what": "잠재 직선화 (Temporal Straightening, λ=0.1)",
        "doc": "docs/DESIGN_STRAIGHTENING_20260807.md",
        "obs": "사전관측 없음 — 학습 전에 등록",
    },
    "fr_s": {
        "date": "2026-08-03",
        "what": "수급 랭킹 손실 (연기금 flow, w=0.25)",
        "doc": "docs/DESIGN_FLOW_RANK_HEAD_20260803.md",
        "obs": "사전관측 없음 — 학습 전에 등록",
    },
    "epc_s": {
        "date": "2026-07-30",
        "what": "진입경로 랭킹 압력 손실 (w=0.25)",
        "doc": "docs/ARCHITECTURE_VERDICTS.md",
        "obs": "5폴드 관측 완료 — 회귀 게이트 미달로 기각됨",
    },
    "hz_s": {
        "date": "2026-07-30",
        "what": "지평 헤드 확장 (h15/h20 추가)",
        "doc": "docs/ARCHITECTURE_VERDICTS.md",
        "obs": "스모크 실패로 미실행",
    },
}

# 게이트 ①  h10 IC 회귀 금지
G1_MEAN_FLOOR = -SD_SEED / 2      # 폴드 평균 ΔIC 하한 (-0.0080)
G1_FOLD_FLOOR = -SD_SEED          # 개별 폴드 ΔIC 하한 (-0.0159), 위반 0개여야 함

# 게이트 ②  청산표 개선 — 두 축(D+15 / 랭크20) 중 최소 하나가 실질 개선
G2_MARGIN = 0.10                  # 실질 개선으로 인정할 Sharpe 마진
G2_NO_HARM = -0.05                # 나머지 축이 이만큼 넘게 나빠지면 실격

# 게이트 ③  폴드 일관성
G3_MIN_WIN = 4                    # 개선 축에서 5폴드 중 우세해야 할 최소 폴드 수
G3_WORST_NO_HARM = 0.0            # 최악폴드 악화 금지

MIN_FOLDS_FOR_ADOPT = 5           # 5폴드 미만이면 '승급'까지만

D15 = "D+15"
RANK20 = "랭크20 h10/h10"


def load(d, name):
    p = Path(d) / f"{name}.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text())
    except Exception as exc:                      # noqa: BLE001
        print(f"  ⚠ {p.name} 파싱 실패: {exc}")
        return None


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def fmt(v, w=6):
    return f"{v:+.2f}".rjust(w) if v is not None else "   .  "


def fmt4(v):
    return f"{v:+.4f}" if v is not None else "   .   "


def gate1(d, variant, folds):
    """h10 IC 회귀 금지."""
    diffs = {}
    for f in folds:
        j = load(d, f"paired_{variant}_{f}")
        if j:
            diffs[f] = j.get("diff")
    if not diffs:
        return None, "IC 짝비교 JSON 없음", diffs
    m = mean(diffs.values())
    bad = [f for f, v in diffs.items() if v is not None and v < G1_FOLD_FLOOR]
    ok = m is not None and m >= G1_MEAN_FLOOR and not bad
    why = f"평균 ΔIC {fmt4(m)} (하한 {G1_MEAN_FLOOR:+.4f})"
    if bad:
        why += f", 폴드 하한 위반 {','.join(bad)}"
    return ok, why, diffs


def _axis(d, variant, study, label):
    """변형·기준의 (폴드→Sharpe) 두 벌을 돌려준다."""
    b = load(d, f"{study}_ens_s")
    v = load(d, f"{study}_{variant}")
    if not b or not v:
        return None, None
    return b.get("rows", {}).get(label), v.get("rows", {}).get(label)


def gate23(d, variant):
    """청산표 개선(②) + 폴드 일관성(③)."""
    axes = {}
    for name, study, label in (("D+15", "sl", D15), ("랭크20", "rank", RANK20)):
        base, var = _axis(d, variant, study, label)
        if not base or not var:
            continue
        common = sorted(set(base) & set(var))
        if not common:
            continue
        mb, mv = mean([base[f] for f in common]), mean([var[f] for f in common])
        wins = sum(1 for f in common if var[f] > base[f])
        worst_b = min(base[f] for f in common)
        worst_v = min(var[f] for f in common)
        axes[name] = {
            "folds": common, "base": mb, "var": mv, "margin": mv - mb,
            "wins": wins, "n": len(common),
            "worst_base": worst_b, "worst_var": worst_v,
            "worst_margin": worst_v - worst_b,
        }
    if not axes:
        return None, None, "청산 JSON 없음", axes

    improved = [k for k, a in axes.items() if a["margin"] >= G2_MARGIN]
    harmed = [k for k, a in axes.items() if a["margin"] < G2_NO_HARM]
    g2 = bool(improved) and not harmed
    why2 = " / ".join(
        f"{k} {fmt(a['var'])} vs {fmt(a['base'])} ({a['margin']:+.2f})"
        for k, a in axes.items())
    if harmed:
        why2 += f" — 악화 축 {','.join(harmed)}"

    if not improved:
        return g2, None, why2, axes
    g3 = all(axes[k]["wins"] >= G3_MIN_WIN
             and axes[k]["worst_margin"] >= G3_WORST_NO_HARM for k in improved)
    why3 = " / ".join(
        f"{k} 우세 {axes[k]['wins']}/{axes[k]['n']}, "
        f"최악폴드 {fmt(axes[k]['worst_var'])} vs {fmt(axes[k]['worst_base'])}"
        for k in improved)
    return g2, g3, why2 + "\n  " + why3, axes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="스터디 JSON 디렉터리")
    ap.add_argument("--variant", required=True, help="변형 접두 (예: ts_s)")
    ap.add_argument("--folds", nargs="+",
                    default=["r5", "r4", "r3", "r2", "r1"])
    args = ap.parse_args()

    pre = PREREG.get(args.variant)
    print("=" * 74)
    print(f"모델 변형 판정: {args.variant}"
          + (f" — {pre['what']}" if pre else "  (사전등록 없음!)"))
    print("=" * 74)
    if pre:
        print(f"사전등록 {pre['date']} | {pre['doc']}")
        print(f"  {pre['obs']}")
    else:
        print("⚠ PREREG 에 등록되지 않은 변형이다. 게이트가 사후에 정해졌을 수 있다.")
    print(f"임계: ①평균ΔIC≥{G1_MEAN_FLOOR:+.4f}·폴드≥{G1_FOLD_FLOOR:+.4f} "
          f"②마진≥{G2_MARGIN:+.2f}·타축≥{G2_NO_HARM:+.2f} "
          f"③우세≥{G3_MIN_WIN}폴드·최악폴드 악화금지")
    print()

    g1, why1, diffs = gate1(args.dir, args.variant, args.folds)
    g2, g3, why23, axes = gate23(args.dir, args.variant)

    if diffs:
        print("① h10 IC 짝비교 (변형 − ens_s)")
        for f in args.folds:
            if f in diffs:
                print(f"   {f}: {fmt4(diffs[f])}")
    print(f"   → {'통과' if g1 else '미달' if g1 is not None else '판정불가'}: {why1}")
    print()
    print("②③ 청산표")
    print(f"   {why23}")

    n_folds = max([a["n"] for a in axes.values()] or [0])
    print()
    print("─" * 74)
    print("결론")
    print("─" * 74)
    if g1 is None or g2 is None:
        verdict = "판정불가 (JSON 누락)"
    elif not g1:
        verdict = "기각 — 회귀 게이트 미달"
    elif not g2:
        verdict = "기각 — 청산 개선 없음"
    elif not g3:
        verdict = "보류 — 평균은 이기나 폴드 일관성 미달"
    elif n_folds < MIN_FOLDS_FOR_ADOPT:
        verdict = f"승급 (전 게이트 통과, 단 {n_folds}폴드 < {MIN_FOLDS_FOR_ADOPT} — 채택 아님)"
    else:
        verdict = "채택권고 — 전 게이트 통과"
    print(f"  {args.variant}: {verdict}")
    for k, a in axes.items():
        print(f"    {k}: {fmt(a['var'])} vs ens_s {fmt(a['base'])} "
              f"| 우세 {a['wins']}/{a['n']} | 최악 {fmt(a['worst_var'])} vs {fmt(a['worst_base'])}")
    if diffs:
        print(f"    h10 IC: 평균 Δ{fmt4(mean(diffs.values()))} ({len(diffs)}폴드)")
    print("─" * 74)
    return 0


if __name__ == "__main__":
    sys.exit(main())
