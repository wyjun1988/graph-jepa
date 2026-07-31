#!/usr/bin/env bash
# GPU 배치 — 6시드 x 5폴드 패널 완성 (2026-07-31)
#
#   실행:  bash scripts/gpu.sh
#
# 알아서 백그라운드로 넘어간다. nohup 도 & 도 붙일 필요 없고, 터미널을 닫아도
# 계속 돈다. 사전 점검과 할 일 목록은 넘어가기 전에 화면에 먼저 찍힌다.
#
# ── 왜 이걸 돌리나 ────────────────────────────────────────────────────────
# 5폴드 패널(시드 3/17/29)이 완성됐고 결과는 이랬다:
#   IC 5/5 양수, 폴드간 t=+7.29 (p=0.002)
#   Sharpe 평균 +0.35, t=+2.13, p=0.100 — 5% 미달이고 95%CI 가 0 을 포함
#
# 그런데 r5 의 6시드 개별 성적을 처음 펼쳐 보니 이랬다:
#   s3 +0.87 | s17 +0.08 | s29 -0.10 | s23 -0.25 | s5 -0.69 | s11 -0.78
#   6시드 앙상블 -0.08   vs   3/17/29 조합 +0.42
#
# 즉 패널에 쓴 3/17/29 는 6개 중 최고 시드(s3)를 품은 조합이고, 전부 쓰면 r5 는
# 오히려 마이너스다. **시드 선택 효과가 신호보다 크다.** 5폴드 결과 전체가 운 좋은
# 조합 위에 서 있는 셈이다.
#
# 이 배치가 답하는 건 하나다: **시드 품질이 폴드를 건너 상관되는가.**
# 시드는 초기화와 데이터 순서를 정하는데 폴드마다 학습 데이터가 다르니, 원리상
# s3 가 모든 폴드에서 좋을 이유가 없다.
#   상관 없음 → r5 만 부풀려진 것이고 r1~r4 는 편향이 없어 패널은 대체로 살아남는다.
#   상관 있음 → 다섯 폴드가 전부 내려앉는다. 전략을 다시 봐야 한다.
#
# ── 무엇을 하나 ──────────────────────────────────────────────────────────
#   1) 5폴드 x 6시드(3/5/11/17/23/29) 를 채운다. 이미 끝난 런은 건너뛴다.
#   2) 폴드가 끝날 때마다 바로 압축한다 — 중간에 멈춰도 끝난 폴드는 건진다.
#   3) 1차에서 OOM 으로 실패한 런을 동시 1로 한 번 더 줍는다.
#
# 24GB 에 런당 11.8GB 라 동시 2 면 여유가 1GB 뿐이다. 그래서 2차 패스가 있다.
# 불안하면:  CONCURRENCY=1 bash scripts/gpu.sh
#
# ── 끝나면 ───────────────────────────────────────────────────────────────
#   r1_compact.csv.gz ~ r5_compact.csv.gz  (폴드당 약 3.8MB, 합계 20MB 안팎)
#   묶기:  tar czf compact_all.tar.gz r?_compact.csv.gz
# 이 파일들만 있으면 청산정책·시드포화·사전등록 필터 검정이 전부 로컬에서 된다.
# 파일을 못 빼와도 로그의 폴드별 요약표만 있으면 시드 민감도까지는 확인된다.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONCURRENCY="${CONCURRENCY:-2}"
ALL_SEEDS="${ALL_SEEDS:-3 5 11 17 23 29}"
FOLD_LIST="${FOLD_LIST:-r1 r2 r3 r4 r5}"

PY="${PY:-}"
if [ -z "$PY" ]; then
  for c in venv/bin/python .venv/bin/python "$(command -v python3)"; do
    [ -n "$c" ] && [ -x "$c" ] && { PY="$c"; break; }
  done
fi
export PY

suffix_of(){
  case "$1" in
    r5) echo "fold1_20250905_to_20260710" ;;
    r4) echo "fold1_20241106_to_20250908" ;;
    r3) echo "fold1_20240104_to_20241107" ;;
    r2) echo "fold1_20230307_to_20240105" ;;
    r1) echo "fold1_20220510_to_20230306" ;;
  esac
}
done_run(){ [ -f "reports/walk_forward/node_eval/ens_s${2}_$(suffix_of "$1")/future_rollout.csv" ]; }

# ══════════════════════════════════════════════════════════════════════════
# 1부: 부모 — 점검하고 계획을 보여준 뒤 자기 자신을 백그라운드로 띄운다
# ══════════════════════════════════════════════════════════════════════════
if [ "${GPU_SH_CHILD:-}" != "1" ]; then

  echo "════ GPU 배치: 6시드 x 5폴드 ════"
  echo "시각   : $(date '+%Y-%m-%d %H:%M')"
  echo "경로   : $ROOT"
  echo "python : $PY"
  "$PY" -c "import torch;print('torch  :',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" || {
    echo ""
    echo "❌ $PY 에 torch 가 없습니다."
    echo "   이 리포 안에 venv/ 가 없으면 자동탐색이 시스템 python 을 집는다."
    echo "   torch 가 있는 파이썬을 직접 지정하세요:"
    echo "     PY=/경로/venv/bin/python bash scripts/gpu.sh"
    exit 1; }
  echo "동시   : ${CONCURRENCY} | 시드: ${ALL_SEEDS} | 폴드: ${FOLD_LIST}"
  echo ""

  # 이미 돌고 있으면 두 번 띄우지 않는다 — GPU 를 두 배치가 나눠 쓰면 둘 다 OOM 이다.
  # pidfile 로 본다. pgrep -f 로 자기 자신을 잡는 사고가 전에 있었고(무관한 SSH
  # 명령줄에 패턴이 걸려 큐가 한 시간 멈췄다), 환경변수는 -f 로 매칭되지도 않는다.
  mkdir -p ops
  LOCK="$ROOT/ops/gpu.pid"
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "⚠ 이미 돌고 있습니다 (PID $(cat "$LOCK")). 중복 실행하면 OOM 입니다."
    echo "  로그 보기:  tail -f $ROOT/gpu_*.log"
    echo "  중단:       kill $(cat "$LOCK")"
    exit 1
  fi
  # pidfile 이 없어도 학습 프로세스가 살아 있으면 막는다 (수동 실행 등)
  if pgrep -f 'python[^ ]* scripts/run_walk_forward_node_eval\.py' > /dev/null 2>&1; then
    echo "⚠ 학습 프로세스가 이미 떠 있습니다. 중복 실행하면 OOM 입니다."
    echo "  확인:  pgrep -af 'scripts/run_walk_forward_node_eval'"
    exit 1
  fi

  echo "── 사전 점검 ──"
  fail=0
  "$PY" - <<'PYEOF' || fail=1
import re, sys
sys.path.insert(0, ".")
src = open("scripts/evaluate_node_prediction.py").read()
assert "temporal_head_input=ckpt_args" in src, \
    "evaluate_node_prediction 의 load_model 배선이 없다 — git pull 필요"
q = open("scripts/seed_queue_v2.sh").read()
for f in ("r1", "r2", "r3", "r4", "r5"):
    assert f + ") FOLD=" in q, "seed_queue_v2.sh 에 %s 폴드가 없다 — git pull 필요" % f
assert "--save-return-forecasts" in q, "큐에 --save-return-forecasts 가 없다"
bad = [l for l in q.splitlines()
       if not l.lstrip().startswith("#") and re.search(r"^(ROOT|PY)=/workspace", l.strip())]
assert not bad, "seed_queue_v2.sh 에 RunPod 경로가 박혀 있다 — git pull 필요: %s" % bad
d = open("scripts/distill_forecasts.py").read()
assert "REF_SEEDS" in d, "distill_forecasts.py 가 구버전이다(시드별 열 없음) — git pull 필요"
print("  코드 배선 OK")
PYEOF
  for f in data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv \
           data/universes/krx500_pit_20191231.json \
           data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl \
           data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
           data/kiwoom_investor_cache data/external_cache; do
    [ -e "$f" ] || { echo "  ❌ 없음: $f"; fail=1; }
  done
  [ "$fail" = 0 ] && echo "  데이터 OK" || { echo ""; echo "사전 점검 실패 — 중단"; exit 1; }

  echo ""
  echo "── 할 일 ──"
  todo=0
  for F in $FOLD_LIST; do
    have=""; need=""
    for S in $ALL_SEEDS; do
      if done_run "$F" "$S"; then have="$have $S"; else need="$need $S"; todo=$((todo+1)); fi
    done
    printf "  %-3s  보유:%-20s 필요:%s\n" "$F" "${have:- 없음}" "${need:- 없음(완료)}"
  done
  echo ""
  if [ "$todo" -eq 0 ]; then
    echo "  전부 완료돼 있습니다 — 압축만 하고 끝냅니다."
  else
    lo=$(( todo * 25 / 60 )); hi=$(( todo * 45 / 60 ))
    [ "$CONCURRENCY" -gt 1 ] && { lo=$(( lo * 10 / 14 )); hi=$(( hi * 10 / 14 )); }
    echo "  새로 돌릴 런: ${todo}개 → 약 ${lo}~${hi}시간 (런당 25~45분, 동시 ${CONCURRENCY})"
  fi

  LOG="$ROOT/gpu_$(date '+%m%d_%H%M').log"
  GPU_SH_CHILD=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
  CHILD=$!
  echo "$CHILD" > "$LOCK"
  echo ""
  echo "════ 백그라운드로 넘어갑니다 (PID ${CHILD}) ════"
  echo "  로그  : $LOG"
  echo "  보기  : tail -f $LOG"
  echo "  중단  : kill ${CHILD}"
  echo ""
  echo "  터미널을 닫아도 계속 돕니다. 이제 나가셔도 됩니다."
  exit 0
fi

# ══════════════════════════════════════════════════════════════════════════
# 2부: 자식 — 실제 작업
# ══════════════════════════════════════════════════════════════════════════
trap 'rm -f "$ROOT/ops/gpu.pid"' EXIT
echo "════ 시작 $(date '+%Y-%m-%d %H:%M') ════"

distill(){
  echo "── $1 압축 ──"
  "$PY" scripts/distill_forecasts.py --fold "$1" --seeds $ALL_SEEDS \
    --out "$1_compact.csv" && gzip -9kf "$1_compact.csv"
}

# 1차 패스 — 폴드가 끝날 때마다 압축한다
for F in $FOLD_LIST; do
  echo ""
  echo "════════ 폴드 $F (1차, 동시 ${CONCURRENCY}) ════════"
  PREFIX=ens_s bash scripts/seed_queue_v2.sh "$CONCURRENCY" "$F" $ALL_SEEDS
  distill "$F"
done

# 2차 패스 — 1차에서 OOM·실패한 런을 동시 1로 줍는다
left=0
for F in $FOLD_LIST; do
  for S in $ALL_SEEDS; do done_run "$F" "$S" || left=$((left+1)); done
done
if [ "$left" -gt 0 ]; then
  echo ""
  echo "════ 2차 패스: 빠진 ${left}런을 동시 1로 재시도 ════"
  for F in $FOLD_LIST; do
    PREFIX=ens_s bash scripts/seed_queue_v2.sh 1 "$F" $ALL_SEEDS
    distill "$F"
  done
else
  echo ""
  echo "1차에서 전부 성공 — 2차 패스 생략"
fi

echo ""
echo "════ 종료 $(date '+%Y-%m-%d %H:%M') ════"
echo "── 최종 상태 ──"
for F in $FOLD_LIST; do
  ok=0; miss=""
  for S in $ALL_SEEDS; do
    if done_run "$F" "$S"; then ok=$((ok+1)); else miss="$miss $S"; fi
  done
  printf "  %-3s  %d/%s 시드%s\n" "$F" "$ok" "$(echo $ALL_SEEDS | wc -w | tr -d ' ')" \
    "$([ -n "$miss" ] && echo "  (실패:$miss)")"
done
echo ""
echo "── 돌려주실 것 ──"
for F in $FOLD_LIST; do
  [ -f "${F}_compact.csv.gz" ] && printf "  %-22s %s\n" "${F}_compact.csv.gz" "$(du -h "${F}_compact.csv.gz" | cut -f1)"
done
echo ""
echo "  한 번에 묶기:  tar czf compact_all.tar.gz r?_compact.csv.gz"
