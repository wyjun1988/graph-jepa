#!/usr/bin/env bash
# 4090 통합 큐 (2026-08-01 작성) — 복구되면 이것 하나만 돌리면 된다.
#
#   git pull && bash scripts/4090_all.sh
#
# 알아서 백그라운드로 넘어간다. nohup/& 불필요, 터미널 닫아도 계속.
# 중간에 죽어도 다시 돌리면 끝난 것은 건너뛴다.
#
# ── 왜 이 순서인가 ───────────────────────────────────────────────────────
# 이 머신은 파일 반출이 안 된다(화면 복사만 가능). 그래서 판정을 전부 여기서
# 끝내고 로그에 남긴다. 순서는 "GPU 가 죽어도 건지는 것부터":
#
#  S0  CPU 판정 (즉시, GPU 무관) — r1~r3 예측이 이 머신에만 있어서 지금까지
#      2폴드로만 봤던 것들을 5폴드로 확정한다:
#        청산 스택(현행 vs D+15 vs SL-5%) · 손절 사전등록 · 지평 헤드 · 6시드 패널
#        · 랭크청산 사전등록 (2026-08-02 추가 — Buy-Hold-Spread top20% 히스테리시스,
#          섀도우 북 rankexit 과 동일 규칙. 2폴드 +1.98 vs D+15 +0.57 의 5폴드 재판)
#  S1  보충 학습 — r5 ens_s23 (지난 배치에서 실패한 1런). 6시드 x 5폴드 완성.
#  S2  스모크 — path-horizons 15,20 배선 확인(3에폭). 실패 시 hz 단계만 접는다.
#  S3  실험 학습 — hz_s(지평 헤드) · epc_s(랭킹 압력) x 5폴드 x 시드 3/17/29.
#      비교 대상 ens_s 가 전부 이 머신에서 학습됐으므로 머신 짝지음이 저절로 된다.
#  S4  2차 패스 — 실패분 동시 1 재시도.
#  S5  (옵션) Chronos r3 사전등록 검정 — chronos 미설치면 자동 생략.
#  S6  최종 판정 재출력 — 중요한 표일수록 아래에. 로그 끝부분만 복사하면 된다.
#
# ── 조절 ─────────────────────────────────────────────────────────────────
#   CONCURRENCY=1        기본 2 (24GB 에 11.8GB x2 — 이전 배치에서 검증됨)
#   FOLDS_EXP="r5 r4"    실험 폴드 축소 (기본 5폴드 전부)
#   CHRONOS=0            사전등록 단계 생략 (기본 auto: 설치돼 있으면 실행)
#
# 예상 소요: S1 1런 + S2 1스모크 + S3 30런.
#   4090 실측이 런당 ~25분이면 동시2 기준 약 7시간, 45분이면 밤새.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONCURRENCY="${CONCURRENCY:-2}"
SEEDS_EXP="${SEEDS_EXP:-3 17 29}"
SEEDS_ALL="3 5 11 17 23 29"
FOLDS_ALL="r5 r4 r3 r2 r1"
FOLDS_EXP="${FOLDS_EXP:-r5 r4 r3 r2 r1}"
CHRONOS="${CHRONOS:-auto}"
HZ_EXTRA="--path-horizons 1,2,3,5,10,15,20"
EPC_EXTRA="--entry-path-correlation-loss-weight 0.25"

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
done_run(){ [ -f "reports/walk_forward/node_eval/${2}${3}_$(suffix_of "$1")/future_rollout.csv" ]; }

# ══════════════════════════════════════════════════════════════════════════
# 1부: 부모 — 점검·계획 표시 후 자기 자신을 백그라운드로
# ══════════════════════════════════════════════════════════════════════════
if [ "${GPU_SH_CHILD:-}" != "1" ]; then
  echo "════ 4090 통합 큐: 5폴드 확정 + 보충 + hz/epc 실험 + 사전등록 ════"
  echo "시각   : $(date '+%Y-%m-%d %H:%M')"
  echo "python : $PY"
  "$PY" -c "import torch;print('torch  :',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" || {
    echo "❌ torch 없음. PY=/경로/venv/bin/python bash scripts/4090_all.sh"; exit 1; }
  echo "동시   : ${CONCURRENCY} | 실험 시드: ${SEEDS_EXP} | 실험 폴드: ${FOLDS_EXP}"

  mkdir -p ops
  LOCK="$ROOT/ops/gpu3.pid"
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "⚠ 이미 돌고 있습니다 (PID $(cat "$LOCK"))."; exit 1
  fi
  if pgrep -f 'python[^ ]* scripts/run_walk_forward_node_eval\.py' > /dev/null 2>&1; then
    echo "⚠ 학습 프로세스가 이미 떠 있습니다. 중복이면 OOM."; exit 1
  fi

  echo ""
  echo "── 사전 점검 ──"
  fail=0
  "$PY" - <<'PYEOF' || fail=1
import sys, pathlib
sys.path.insert(0, ".")
from stock_v2.graph_jepa import StockGraphJEPA
m = StockGraphJEPA(num_features=20, hidden_dim=32, num_layers=2,
                   temporal_state_mode="horizon_residual_heads",
                   temporal_head_steps=[1, 2], temporal_state_context_skip=True)
assert m.temporal_head_steps == (1, 2)
assert "temporal_head_input=ckpt_args" in open("scripts/evaluate_node_prediction.py").read(), \
    "load_model 배선 없음 — git pull"
q = open("scripts/seed_queue_v2.sh").read()
assert "PREFIX" in q and "EXTRA" in q and "r1) FOLD=" in q
for s in ("exit_tp_report", "sl_exit_study", "horizon_head_study",
          "paired_variant_report", "panel_report", "prereg_filter_test",
          "tsfm_benchmark", "rank_exit_study"):
    assert pathlib.Path(f"scripts/{s}.py").exists(), f"{s}.py 없음 — git pull"
print("  코드 배선 OK")
PYEOF
  for f in data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv \
           data/universes/krx500_pit_20191231.json \
           data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl \
           data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
           data/kiwoom_investor_cache data/external_cache; do
    [ -e "$f" ] || { echo "  ❌ 없음: $f"; fail=1; }
  done
  # r1~r3 예측이 실제로 이 머신에 있는지 — 없으면 S0 5폴드 판정이 2폴드가 된다
  for F in $FOLDS_ALL; do
    done_run "$F" ens_s 3 || echo "  ⚠ ens_s3 ${F} 예측 없음 (S0 판정에서 해당 폴드 빠짐)"
  done
  [ "$fail" = 0 ] && echo "  데이터 OK" || { echo "사전 점검 실패 — 중단"; exit 1; }

  echo ""
  echo "── 할 일 (GPU) ──"
  todo=0
  need=""
  for F in $FOLDS_ALL; do
    for S in $SEEDS_ALL; do done_run "$F" ens_s "$S" || { need="$need ens_s$S($F)"; todo=$((todo+1)); }; done
  done
  printf "  보충(S1)  :%s\n" "${need:- 없음}"
  for P in hz_s epc_s; do
    n=0
    for F in $FOLDS_EXP; do
      for S in $SEEDS_EXP; do done_run "$F" "$P" "$S" || n=$((n+1)); done
    done
    printf "  %-9s : %d런\n" "$P(S3)" "$n"
    todo=$((todo+n))
  done
  echo "  + 스모크 1런"
  echo "  합계 ${todo}런 — 런당 25분이면 동시${CONCURRENCY} 기준 약 $(( todo * 25 / 60 / CONCURRENCY ))시간, 45분이면 $(( todo * 45 / 60 / CONCURRENCY ))시간"
  echo "  Chronos 사전등록(S5): ${CHRONOS} ($("$PY" -c 'import chronos' 2>/dev/null && echo 설치됨 || echo '미설치 — 자동 생략'))"

  LOG="$ROOT/4090_all_$(date '+%m%d_%H%M').log"
  GPU_SH_CHILD=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
  CHILD=$!
  echo "$CHILD" > "$LOCK"
  echo ""
  echo "════ 백그라운드 시작 (PID ${CHILD}) ════"
  echo "  로그: tail -f $LOG"
  echo "  중단: kill ${CHILD}"
  echo "  S0(CPU 5폴드 판정)는 몇 분 안에 로그에 찍힙니다 — GPU 와 무관하게 먼저 보셔도 됩니다."
  exit 0
fi

# ══════════════════════════════════════════════════════════════════════════
# 2부: 자식 — 실제 작업
# ══════════════════════════════════════════════════════════════════════════
trap 'rm -f "$ROOT/ops/gpu3.pid"' EXIT
echo "════ 시작 $(date '+%Y-%m-%d %H:%M') ════"

# ── S0: CPU 판정 — 지금 있는 파일로 5폴드 확정 (GPU 죽어도 이건 남는다) ──
echo ""
echo "╔════ S0. 5폴드 확정 판정 (기존 파일, CPU) ════╗"
echo ""
echo "──── S0-a. 6시드 패널 (panel_report) ────"
"$PY" scripts/panel_report.py --seeds $SEEDS_ALL --folds $FOLDS_ALL 2>/dev/null
echo ""
echo "──── S0-b. 청산 스택: 유니버스 2종 (exit_tp_report) ────"
"$PY" scripts/exit_tp_report.py --folds $FOLDS_ALL --seeds $SEEDS_ALL 2>/dev/null
echo ""
echo "──── S0-c. 손절 사전등록 판정: D+15 SL-5% (sl_exit_study) ────"
echo "  사전등록(8/1): 주 판정 = D+15 SL-5% 가 D+15 를 5폴드에서 이기는가"
"$PY" scripts/sl_exit_study.py --folds $FOLDS_ALL --seeds $SEEDS_ALL 2>/dev/null | tail -22
echo ""
echo "──── S0-d. 지평 헤드 5폴드 (horizon_head_study) ────"
"$PY" scripts/horizon_head_study.py --folds $FOLDS_ALL --seeds $SEEDS_ALL 2>/dev/null | tail -45
echo ""
echo "──── S0-e. 랭크 청산 사전등록 판정 (rank_exit_study, 2026-08-02 추가) ────"
echo "  사전등록(8/2, r1~r3 미관측 상태에서 등록): 주 판정 = '랭크 top20% 캡30' 이"
echo "  D+15 를 5폴드 평균에서 이기고 + 최악폴드를 악화시키지 않는가."
echo "  2폴드 사전관측치: 랭크 +1.98 vs D+15 +0.57 (r5 +1.32/+0.55, r4 +2.64/+0.60)."
echo "  경고: SL-5% 가 같은 2폴드에서 +0.92 로 보이다 5폴드에서 죽었다. 그 재판이다."
"$PY" scripts/rank_exit_study.py --folds $FOLDS_ALL --seeds $SEEDS_ALL 2>/dev/null | tail -30
echo ""
echo "╚════ S0 끝 — 여기까지가 5폴드 확정. 아래는 신규 학습 ════╝"

# ── S1: 보충 — 6시드 x 5폴드 완성 (스킵 인지라 빠진 것만 돈다) ──────────
echo ""
echo "════ S1. 보충 학습 (빠진 ens_s 채우기) ════"
for F in $FOLDS_ALL; do
  PREFIX=ens_s bash scripts/seed_queue_v2.sh "$CONCURRENCY" "$F" $SEEDS_ALL
done

# ── S2: 스모크 — h15/h20 배선 ────────────────────────────────────────────
HZ_OK=1
if ! done_run r5 smokehz_s 17; then
  echo ""
  echo "════ S2. 스모크: path-horizons 15,20 (3에폭) ════"
  T0=$(date +%s)
  PREFIX=smokehz_s EXTRA="$HZ_EXTRA --epochs 3 --checkpoint-epochs 2" \
    bash scripts/seed_queue_v2.sh 1 r5 17
  echo "  스모크 소요: $(( ($(date +%s)-T0)/60 ))분 (본 런 시간 추정에 참고)"
fi
"$PY" - <<'PYEOF' || HZ_OK=0
import csv, sys, collections
p = "reports/walk_forward/node_eval/smokehz_s17_fold1_20250905_to_20260710/return_1d_forecasts.csv"
try:
    c = collections.Counter()
    with open(p, newline="") as f:
        for r in csv.DictReader(f):
            c[r["horizon"]] += 1
except FileNotFoundError:
    print("  ❌ 스모크 예측 파일 없음"); sys.exit(1)
print("  지평 분포:", dict(sorted(c.items(), key=lambda x: int(x[0]))))
if {"15", "20"} - set(c):
    print("  ❌ h15/h20 행 없음 — hz 본 런 생략"); sys.exit(1)
print("  ✅ h15/h20 예측 생성 확인")
PYEOF
[ "$HZ_OK" = 1 ] && echo "→ hz 본 런 진행" || echo "→ ⚠ hz 접음 (epc 는 계속)"

# ── S3: 실험 학습 — 폴드별 hz → epc ──────────────────────────────────────
for F in $FOLDS_EXP; do
  echo ""
  echo "════════ S3. 폴드 $F ════════"
  [ "$HZ_OK" = 1 ] && PREFIX=hz_s EXTRA="$HZ_EXTRA" \
    bash scripts/seed_queue_v2.sh "$CONCURRENCY" "$F" $SEEDS_EXP
  PREFIX=epc_s EXTRA="$EPC_EXTRA" \
    bash scripts/seed_queue_v2.sh "$CONCURRENCY" "$F" $SEEDS_EXP
done

# ── S4: 2차 패스 — 실패분 동시 1 ─────────────────────────────────────────
left=0
for F in $FOLDS_ALL; do
  for S in $SEEDS_ALL; do done_run "$F" ens_s "$S" || left=$((left+1)); done
done
for F in $FOLDS_EXP; do
  for P in epc_s $([ "$HZ_OK" = 1 ] && echo hz_s); do
    for S in $SEEDS_EXP; do done_run "$F" "$P" "$S" || left=$((left+1)); done
  done
done
if [ "$left" -gt 0 ]; then
  echo ""
  echo "════ S4. 2차 패스: 빠진 ${left}런 (동시 1) ════"
  for F in $FOLDS_ALL; do PREFIX=ens_s bash scripts/seed_queue_v2.sh 1 "$F" $SEEDS_ALL; done
  for F in $FOLDS_EXP; do
    [ "$HZ_OK" = 1 ] && PREFIX=hz_s EXTRA="$HZ_EXTRA" bash scripts/seed_queue_v2.sh 1 "$F" $SEEDS_EXP
    PREFIX=epc_s EXTRA="$EPC_EXTRA" bash scripts/seed_queue_v2.sh 1 "$F" $SEEDS_EXP
  done
fi

# ── S5: (옵션) Chronos r3 사전등록 검정 ──────────────────────────────────
if [ "$CHRONOS" != "0" ] && "$PY" -c 'import chronos' 2>/dev/null; then
  echo ""
  echo "════ S5. 사전등록 필터 검정 (r3, Chronos-Bolt-small, CUDA) ════"
  echo "  등록 문서: docs/PREREG_FILTER_R3_20260731.md (고정값은 스크립트에 하드코딩)"
  if [ ! -f prereg_r3_ens.csv ]; then
    "$PY" scripts/tsfm_benchmark.py --model amazon/chronos-bolt-small --fold r3 \
      --seeds 3 17 29 --device cuda --dump prereg_r3_ens.csv 2>/dev/null | tail -6
  fi
  for S in 3 17 29; do
    [ -f "prereg_r3_s${S}.csv" ] || \
      "$PY" scripts/tsfm_benchmark.py --model amazon/chronos-bolt-small --fold r3 \
        --seeds "$S" --device cuda --dump "prereg_r3_s${S}.csv" 2>/dev/null | tail -3
  done
  echo ""
  "$PY" scripts/prereg_filter_test.py --csv prereg_r3_ens.csv \
    --seed-csv prereg_r3_s3.csv prereg_r3_s17.csv prereg_r3_s29.csv
else
  echo ""
  echo "════ S5. 생략 — chronos 미설치 (원하면: pip install chronos-forecasting 후 재실행) ════"
fi

# ── S6: 최종 판정 — 중요한 표일수록 아래에 ───────────────────────────────
echo ""
echo "╔════ S6. 최종 판정 (전부 이 머신에서 학습 — 머신 짝지음) ════╗"
for F in $FOLDS_EXP; do
  echo ""
  echo "[C] h10 IC 짝비교 — 폴드 $F (회귀 금지 게이트)"
  [ "$HZ_OK" = 1 ] && "$PY" scripts/paired_variant_report.py --a ens_s --b hz_s \
    --seeds $SEEDS_EXP --fold "$F" 2>/dev/null | tail -6
  "$PY" scripts/paired_variant_report.py --a ens_s --b epc_s \
    --seeds $SEEDS_EXP --fold "$F" 2>/dev/null | tail -6
done
if [ "$HZ_OK" = 1 ]; then
  echo ""
  echo "[B] hz: h15/h20 헤드가 D+15/20 보유에서 기존 조합을 이기나"
  "$PY" scripts/horizon_head_study.py --prefix hz_s --seeds $SEEDS_EXP \
    --folds $FOLDS_EXP --heads 1,2,3,5,10,15,20 2>/dev/null | tail -50
fi
echo ""
echo "[A] 최종 자: 청산표 (D+15 SL-5% 행이 판정 기준) — 베이스라인 → 변형 순"
for P in ens_s $([ "$HZ_OK" = 1 ] && echo hz_s) epc_s; do
  echo ""
  echo "  ── ${P} (시드 ${SEEDS_EXP}) ──"
  "$PY" scripts/sl_exit_study.py --prefix "$P" --seeds $SEEDS_EXP \
    --folds $FOLDS_EXP 2>/dev/null | sed -n '/손절 연구/,$p' | head -20
done
echo ""
echo "[A2] 랭크 청산이 모델 변형에서도 유지되는가 (2026-08-02 추가)"
for P in $([ "$HZ_OK" = 1 ] && echo hz_s) epc_s; do
  echo ""
  echo "  ── ${P} (시드 ${SEEDS_EXP}) ──"
  "$PY" scripts/rank_exit_study.py --prefix "$P" --seeds $SEEDS_EXP \
    --folds $FOLDS_EXP 2>/dev/null | sed -n '/랭크 청산 연구/,$p' | head -28
done
echo ""
echo "════ 종료 $(date '+%Y-%m-%d %H:%M') ════"
echo "복사해 주실 것: S0-c(손절), S0-e(랭크청산 사전등록), S5(필터), S6 [A]·[A2]."
echo "그 덩어리들이면 채택 판정이 전부 됩니다."