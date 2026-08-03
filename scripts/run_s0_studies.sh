#!/usr/bin/env bash
# S0-e/f/g 판정만 따로 뽑는다 — 4090_all.sh 를 이미 돌린 뒤 추가된 연구를
# 재실행 없이 붙이고 싶을 때. 전부 CPU, 기존 예측 재사용.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/.."
PY="${PY:-}"
if [ -z "$PY" ]; then
  for c in venv/bin/python .venv/bin/python "$(command -v python3)"; do
    [ -n "$c" ] && [ -x "$c" ] && { PY="$c"; break; }
  done
fi
F="${FOLDS:-r5 r4 r3 r2 r1}"
S="${SEEDS:-3 5 11 17 23 29}"
echo "════ S0 추가 판정 (폴드: $F / 시드: $S) ════"
echo ""
echo "──── S0-e. 랭크청산 사전등록 (2026-08-02) ────"
"$PY" scripts/rank_exit_study.py --folds $F --seeds $S 2>/dev/null | sed -n '/랭크 청산 연구/,$p'
echo ""
echo "──── S0-f. 배분조절 사전등록 (2026-08-03) ────"
for X in d15 rank20; do
  echo ""; echo "  ── 청산=${X} ──"
  "$PY" scripts/vol_deploy_study.py --exit "$X" --folds $F --seeds $S 2>/dev/null | tail -20
done
echo ""
echo "──── S0-g. 변동성x보유 2x2 사전등록 (2026-08-03) ────"
"$PY" scripts/vol_holding_interaction_study.py --folds $F --seeds $S 2>/dev/null | sed -n '/\[A\] 국면/,$p'
echo ""
echo "════ 끝 $(date '+%Y-%m-%d %H:%M') ════"
