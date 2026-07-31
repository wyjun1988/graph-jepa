#!/usr/bin/env bash
# GPU 하루 배치 (2026-08-02) — 지평 헤드 · 랭킹 압력 · 머신 짝지은 베이스라인
#
#   실행:  bash scripts/gpu_day2.sh
#
# 알아서 백그라운드로 넘어간다. nohup 도 & 도 불필요, 터미널 닫아도 계속 돈다.
# 계획·근거는 docs/TRAINING_PLAN_20260802.md.
#
# ── 설계 요점 ────────────────────────────────────────────────────────────
# 1) 스모크 먼저: --path-horizons 에 15,20 을 넣으면 롤아웃 스텝이 1→2 로
#    바뀐다(아키텍처 변경). 3에폭 스모크로 h15/h20 예측행 생성과 런타임
#    증가율을 확인한 뒤에만 본 런을 태운다. 실패하면 hz 단계 전체를 접는다.
# 2) 베이스라인 재학습: 같은 시드도 머신이 바뀌면 재현 안 된다(4090 s3
#    +0.57 vs A5000 +0.87). 변형 비교는 이 팟에서 학습한 ens_s 와만 한다.
# 3) 폴드가 끝날 때마다 압축(distill) — 중간에 죽어도 끝난 것은 건진다.
# 4) 마지막에 판정표를 화면에 찍는다. 중요한 것일수록 아래에.

set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

CONCURRENCY="${CONCURRENCY:-2}"
SEEDS="${SEEDS:-3 17 29}"
FOLD_LIST="${FOLD_LIST:-r5 r4}"
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
# 1부: 부모 — 점검·계획을 보여주고 자기 자신을 백그라운드로
# ══════════════════════════════════════════════════════════════════════════
if [ "${GPU_SH_CHILD:-}" != "1" ]; then
  echo "════ GPU 하루 배치: hz(지평헤드) · epc(랭킹압력) · ens(베이스라인) ════"
  echo "시각   : $(date '+%Y-%m-%d %H:%M')"
  echo "python : $PY"
  "$PY" -c "import torch;print('torch  :',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')" || {
    echo "❌ torch 없음. PY=/경로/venv/bin/python bash scripts/gpu_day2.sh"; exit 1; }
  echo "동시   : ${CONCURRENCY} | 시드: ${SEEDS} | 폴드: ${FOLD_LIST}"

  mkdir -p ops
  LOCK="$ROOT/ops/gpu2.pid"
  if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
    echo "⚠ 이미 돌고 있습니다 (PID $(cat "$LOCK")). kill $(cat "$LOCK") 후 재실행."
    exit 1
  fi
  if pgrep -f 'python[^ ]* scripts/run_walk_forward_node_eval\.py' > /dev/null 2>&1; then
    echo "⚠ 학습 프로세스가 이미 떠 있습니다. 중복이면 OOM."
    exit 1
  fi

  echo ""
  echo "── 사전 점검 ──"
  fail=0
  "$PY" - <<'PYEOF' || fail=1
import sys
sys.path.insert(0, ".")
from stock_v2.graph_jepa import StockGraphJEPA
m = StockGraphJEPA(num_features=20, hidden_dim=32, num_layers=2,
                   temporal_state_mode="horizon_residual_heads",
                   temporal_head_steps=[1, 2], temporal_state_context_skip=True)
assert m.temporal_head_steps == (1, 2)
src = open("scripts/evaluate_node_prediction.py").read()
assert "temporal_head_input=ckpt_args" in src, "load_model 배선 없음 — git pull"
q = open("scripts/seed_queue_v2.sh").read()
assert "PREFIX" in q and "EXTRA" in q, "seed_queue_v2 가 PREFIX/EXTRA 를 모른다"
for s in ("horizon_head_study", "sl_exit_study", "paired_variant_report"):
    assert __import__("pathlib").Path(f"scripts/{s}.py").exists(), f"{s}.py 없음 — git pull"
print("  코드 배선 OK")
PYEOF
  for f in data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv \
           data/universes/krx500_pit_20191231.json \
           data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl \
           data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
           data/kiwoom_investor_cache data/external_cache; do
    [ -e "$f" ] || { echo "  ❌ 없음: $f"; fail=1; }
  done
  [ "$fail" = 0 ] && echo "  데이터 OK" || { echo "사전 점검 실패 — 중단"; exit 1; }

  echo ""
  echo "── 할 일 ──"
  todo=0
  for F in $FOLD_LIST; do
    for P in ens_s hz_s epc_s; do
      need=""
      for S in $SEEDS; do done_run "$F" "$P" "$S" || { need="$need $S"; todo=$((todo+1)); }; done
      printf "  %-3s %-6s 필요:%s\n" "$F" "$P" "${need:- 없음}"
    done
  done
  echo "  + 스모크 1런 (smokehz_s17, 3에폭)"
  echo "  본 런 ${todo}개 × 41~45분 ≈ 동시${CONCURRENCY} 기준 $(( todo * 43 / 60 / CONCURRENCY ))시간"

  LOG="$ROOT/gpu2_$(date '+%m%d_%H%M').log"
  GPU_SH_CHILD=1 nohup bash "$0" "$@" > "$LOG" 2>&1 &
  CHILD=$!
  echo "$CHILD" > "$LOCK"
  echo ""
  echo "════ 백그라운드 시작 (PID ${CHILD}) ════"
  echo "  로그: tail -f $LOG"
  echo "  중단: kill ${CHILD}"
  echo "  끝나면 로그 맨 아래 판정표만 복사해 주시면 됩니다."
  exit 0
fi

# ══════════════════════════════════════════════════════════════════════════
# 2부: 자식 — 실제 작업
# ══════════════════════════════════════════════════════════════════════════
trap 'rm -f "$ROOT/ops/gpu2.pid"' EXIT
echo "════ 시작 $(date '+%Y-%m-%d %H:%M') ════"

# ── S0: 스모크 — h15/h20 배선이 실제로 도는지 3에폭으로 확인 ─────────────
HZ_OK=1
if ! done_run r5 smokehz_s 17; then
  echo ""
  echo "════ S0 스모크: path-horizons 15,20 (3에폭) ════"
  T0=$(date +%s)
  PREFIX=smokehz_s EXTRA="$HZ_EXTRA --epochs 3 --checkpoint-epochs 2" \
    bash scripts/seed_queue_v2.sh 1 r5 17
  T1=$(date +%s)
  echo "  스모크 소요: $(( (T1-T0)/60 ))분"
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
print("  스모크 지평 분포:", dict(sorted(c.items(), key=lambda x: int(x[0]))))
missing = {"15", "20"} - set(c)
if missing:
    print(f"  ❌ 지평 {missing} 행이 없다 — hz 본 런 생략"); sys.exit(1)
print("  ✅ h15/h20 예측 생성 확인")
PYEOF
[ "$HZ_OK" = 1 ] && echo "→ hz 본 런 진행" || echo "→ ⚠ hz 본 런 접음 (ens/epc 는 계속)"

# ── S1~S3: 본 런 — 폴드별로 ens → hz → epc ────────────────────────────────
for F in $FOLD_LIST; do
  echo ""
  echo "════════ 폴드 $F ════════"
  PREFIX=ens_s bash scripts/seed_queue_v2.sh "$CONCURRENCY" "$F" $SEEDS
  if [ "$HZ_OK" = 1 ]; then
    PREFIX=hz_s EXTRA="$HZ_EXTRA" bash scripts/seed_queue_v2.sh "$CONCURRENCY" "$F" $SEEDS
  fi
  PREFIX=epc_s EXTRA="$EPC_EXTRA" bash scripts/seed_queue_v2.sh "$CONCURRENCY" "$F" $SEEDS
  echo "── $F 압축 ──"
  for P in ens_s hz_s epc_s; do
    "$PY" scripts/distill_forecasts.py --fold "$F" --seeds $SEEDS --prefix "$P" \
      --out "${F}_${P%_s}_compact.csv" 2>/dev/null | tail -3 \
      && gzip -9kf "${F}_${P%_s}_compact.csv" 2>/dev/null
  done
done

# ── 2차 패스: 실패분 동시 1 로 재시도 ─────────────────────────────────────
left=0
for F in $FOLD_LIST; do
  for P in ens_s epc_s $([ "$HZ_OK" = 1 ] && echo hz_s); do
    for S in $SEEDS; do done_run "$F" "$P" "$S" || left=$((left+1)); done
  done
done
if [ "$left" -gt 0 ]; then
  echo ""
  echo "════ 2차 패스: 빠진 ${left}런 재시도 (동시 1) ════"
  for F in $FOLD_LIST; do
    PREFIX=ens_s bash scripts/seed_queue_v2.sh 1 "$F" $SEEDS
    [ "$HZ_OK" = 1 ] && PREFIX=hz_s EXTRA="$HZ_EXTRA" bash scripts/seed_queue_v2.sh 1 "$F" $SEEDS
    PREFIX=epc_s EXTRA="$EPC_EXTRA" bash scripts/seed_queue_v2.sh 1 "$F" $SEEDS
  done
fi

# ── 판정 — 중요한 것일수록 아래에 ─────────────────────────────────────────
echo ""
echo "════ 판정 (같은 팟에서 학습된 것끼리만 비교) ════"
for F in $FOLD_LIST; do
  echo ""
  echo "[C] h10 IC 짝비교 — 폴드 $F (회귀 금지 게이트)"
  [ "$HZ_OK" = 1 ] && "$PY" scripts/paired_variant_report.py --a ens_s --b hz_s \
    --seeds $SEEDS --fold "$F" 2>/dev/null | tail -8
  "$PY" scripts/paired_variant_report.py --a ens_s --b epc_s \
    --seeds $SEEDS --fold "$F" 2>/dev/null | tail -8
done
if [ "$HZ_OK" = 1 ]; then
  echo ""
  echo "[B] hz: h15/h20 헤드가 D+15/20 보유에서 h10·mean(h5,h10) 을 이기나"
  "$PY" scripts/horizon_head_study.py --prefix hz_s --seeds $SEEDS \
    --folds $FOLD_LIST --heads 1,2,3,5,10,15,20 2>/dev/null | tail -40
fi
echo ""
echo "[A] 최종 자 (D+15 SL-5% 포함 청산표) — 베이스라인 → 변형 순"
for P in ens_s $([ "$HZ_OK" = 1 ] && echo hz_s) epc_s; do
  echo ""
  echo "  ── ${P} ──"
  "$PY" scripts/sl_exit_study.py --prefix "$P" --seeds $SEEDS \
    --folds $FOLD_LIST 2>/dev/null | sed -n '/손절 연구/,$p' | head -18
done
echo ""
echo "════ 종료 $(date '+%Y-%m-%d %H:%M') ════"
echo "위 [A] 세 표에서 변형이 ens_s 를 D+15 SL-5% 행에서 이기는지가 결론입니다."
echo "파일 반출 가능하면: tar czf day2_compact.tar.gz r?_*_compact.csv.gz"
