#!/usr/bin/env bash
# S0 판정 일괄 실행 + 자동 판정. 전부 CPU, 기존 예측 재사용(재학습 없음).
#
#   bash scripts/run_s0_studies.sh
#
# 끝나면 맨 아래 "사전등록 판정 결과" 블록만 전달하면 된다 — 표를 읽을 필요 없다.
# 상세 표는 ops/verdict/ 의 *.json 과 s0_full.log 에 남는다.
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
OUT="${OUT:-ops/verdict}"
mkdir -p "$OUT"
LOG="$OUT/s0_full.log"
: > "$LOG"

say(){ echo "$@" | tee -a "$LOG"; }
run(){ # run <라벨> <json이름> <명령...>
  local label="$1" name="$2"; shift 2
  say ""
  say "──── $label ────"
  if "$@" --json "$OUT/${name}.json" >> "$LOG" 2>&1; then
    say "  완료 -> $OUT/${name}.json"
  else
    say "  ❌ 실패 — $LOG 끝부분 확인"
    tail -5 "$LOG" | sed 's/^/     | /'
  fi
}

say "════ S0 판정 $(date '+%Y-%m-%d %H:%M') ════"
say "폴드: $F | 시드: $S"

run "S0-e. 랭크청산 사전등록 (2026-08-02)" "rank_exit" \
  "$PY" scripts/rank_exit_study.py --folds $F --seeds $S
for X in d15 rank20; do
  run "S0-f. 배분조절 사전등록 (2026-08-03) — 청산=$X" "vol_deploy_${X}" \
    "$PY" scripts/vol_deploy_study.py --exit "$X" --folds $F --seeds $S
done
run "S0-g. 변동성x보유 2x2 사전등록 (2026-08-03)" "vol_holding" \
  "$PY" scripts/vol_holding_interaction_study.py --folds $F --seeds $S

run "S0-h. 물타기 사전등록 (2026-08-06)" "scale_in" \
  "$PY" scripts/scale_in_study.py --folds $F --seeds $S
run "S0-i. 갈아타기 사전등록 (2026-08-06)" "switch" \
  "$PY" scripts/switch_study.py --folds $F --seeds $S

run "S0-j. 신호강도 조절 사전등록 (2026-08-06)" "signal_scaling" \
  "$PY" scripts/signal_scaling_study.py --folds $F --seeds $S

# fr_s(수급 랭킹 손실)가 학습돼 있으면 게이트 1 도 함께 판정한다.
if ls reports/walk_forward/node_eval/fr_s* >/dev/null 2>&1; then
  say ""
  say "──── fr_s 게이트1. 회귀 금지 (h10 IC) ────"
  "$PY" scripts/paired_variant_report.py --a ens_s --b fr_s \
    --seeds ${SEEDS_EXP:-3 17 29} --fold r5 >> "$LOG" 2>&1 \
    && say "  완료 — s0_full.log 의 '[C]' 절 참조" \
    || say "  ❌ 실패(런 부족일 수 있음)"
fi

"$PY" scripts/verdict.py --dir "$OUT" 2>&1 | tee -a "$LOG"
