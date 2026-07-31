#!/usr/bin/env bash
# 4090 배치 — 5폴드 패널 완성 (2026-07-31 개정 2)
#
# ── 왜 이걸 돌리나 ────────────────────────────────────────────────────────
# 지금 이 프로젝트의 거의 모든 결론이 "단일 폴드" 또는 "두 폴드"에 걸려 있다.
# 그런데 채택 기준(docs §7-5)은 다시드 x 5폴드 짝지은 t검정 + 최악폴드 악화
# 금지다. 즉 지금 상태로는 무엇도 채택 판정을 내릴 수 없다.
#
# 확보:    r5(6시드) · r4(3시드) · r3(3시드)
# 이 배치: r2 · r1 을 같은 시드로 채워 5폴드 패널을 완성한다.
#
# 완성되면 GPU 없이 로컬에서 다음이 전부 5폴드로 재계산된다:
#   청산 정책(사다리 vs D+10/15/20/30) ← 현재 최대 개선축
#   제로샷 TSFM 비교 / 사전등록 필터 검정 / 시드 앙상블 포화곡선
# 즉 이 배치 하나가 앞으로의 재분석을 전부 해금한다.
#
# ── 시드를 3/17/29 로 고정하는 이유와 주의 ───────────────────────────────
# 폴드 간 비교는 같은 시드로 짝지어야 시드 분산이 상쇄된다.
# 주의: r5 에서 seed 3(+0.0661)·17(+0.0539)이 6개 중 1·2위였다. 이 조합은
# 다소 운이 좋은 편이고(약한 3개 조합은 Sharpe -0.50), 절대 수준은 낙관 쪽이다.
# 폴드 간 상대 비교에만 쓸 것.
#
# ── 이전 큐들을 왜 버렸나 ─────────────────────────────────────────────────
# ema 0.99 / VICReg: 지평 혼합 IC 로 매긴 순위였고, 지평 10 으로 다시 재면
#   둘 다 챔프보다 나쁘다(-0.8σ / -0.9σ). docs/MEASUREMENT_CORRECTIONS_20260730.md
# context 모드 x r4: r5 에서 4/4 열세(짝지은 t -3.83)로 이미 결론이 났다.
#
# ── 사전 준비 ────────────────────────────────────────────────────────────
#   1) git pull   ← r1·r2 폴드 정의가 오늘 추가됐다. 없으면 "폴드는 r1~r5" 오류
#   2) data/ 를 graph-jepa-4090.tar.gz 에서 풀어 넣기
#   3) venv 준비 (torch + numpy 2.3.5 + pandas 2.3.3 권장 — 버전 고정이 재현에 중요)
#
# ── 실행 ─────────────────────────────────────────────────────────────────
#   nohup bash scripts/4090_queue.sh > 4090.log 2>&1 &
#   tail -f 4090.log
#
# ── 끝나고 돌려줄 것 ──────────────────────────────────────────────────────
#   r2_compact.csv.gz · r1_compact.csv.gz  (각 수백 KB, 스크립트가 자동 생성)
#
# 예상 소요: 6런(2폴드 x 3시드) x 35~45분 = 4~5시간
# 중간에 멈춰도 안전하다 — 큐는 이미 끝난 시드를 건너뛴다.

set -u
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# venv 자동 탐색 — 없으면 시스템 python3
PY="${PY:-}"
if [ -z "$PY" ]; then
  for c in venv/bin/python .venv/bin/python "$(command -v python3)"; do
    [ -x "$c" ] && { PY="$c"; break; }
  done
fi
export PY

echo "════ 4090 배치: 5폴드 패널 완성 (r2 · r1) ════"
echo "python : $PY"
"$PY" -c "import torch;print('torch  :',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
echo ""

# 사전 점검 — 없는 채로 2시간 돌리고 평가에서 죽는 일을 막는다
echo "── 사전 점검 ──"
fail=0
"$PY" - <<'PYEOF' || fail=1
import re, sys
sys.path.insert(0, ".")
from stock_v2.graph_jepa import StockGraphJEPA, TEMPORAL_HEAD_INPUTS
assert "context" in TEMPORAL_HEAD_INPUTS, "graph_jepa 에 context 모드가 없다 — git pull 필요"
m = StockGraphJEPA(num_features=20, hidden_dim=32, num_layers=2,
                   temporal_state_mode="horizon_residual_heads",
                   temporal_head_steps=[1, 10], temporal_head_input="context")
assert m.temporal_head_width == 1
print("  graph_jepa context 모드 OK")
src = open("scripts/evaluate_node_prediction.py").read()
assert "temporal_head_input=ckpt_args" in src, \
    "evaluate_node_prediction 의 load_model 배선이 없다 — 평가가 크기 불일치로 죽는다. git pull 필요"
q = open("scripts/seed_queue_v2.sh").read()
assert "r1) FOLD=" in q and "r2) FOLD=" in q, \
    "seed_queue_v2.sh 에 r1·r2 폴드가 없다 — git pull 필요"
assert "--save-return-forecasts" in q, "큐에 --save-return-forecasts 가 없다"
print("  평가 load_model 배선 · r1/r2 폴드 · 예측저장 OK")
# 주석이 아니라 실제 대입문만 본다 (설명 주석에 /workspace 가 들어 있다)
bad = [l for l in q.splitlines()
       if not l.lstrip().startswith("#") and re.search(r"^(ROOT|PY)=/workspace", l.strip())]
assert not bad, "seed_queue_v2.sh 에 RunPod 경로가 대입돼 있다 — git pull 필요: %s" % bad
assert "nproc" in q, "워커 수가 코어 기반이 아니다 — git pull 필요"
print("  RunPod 하드코딩 없음 · 워커 코어기반 OK")
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

for F in r2 r1; do
  echo "──────── 폴드 $F ────────"
  PREFIX=ens_s bash scripts/seed_queue_v2.sh 1 "$F" 3 17 29
done

echo ""
echo "════ 결과 압축·요약 ════"
# 원본 예측은 시드당 약 85MB 라 빼오기 어렵다. 분석에 실제로 필요한 부분만
# 1MB 남짓으로 줄이고, 동시에 챔프 단독 지표를 화면에 찍는다.
for F in r2 r1; do
  echo "──────── $F ────────"
  "$PY" scripts/distill_forecasts.py --fold "$F" --seeds 3 17 29 --out "${F}_compact.csv" \
    && gzip -kf "${F}_compact.csv"
done
echo ""
echo "════ 돌려주실 것 ════"
ls -lh r2_compact.csv.gz r1_compact.csv.gz 2>/dev/null | awk '{print "  "$NF"  "$5}'
echo ""
echo "  이 두 파일(각 수백 KB)만 있으면 5폴드 재분석이 전부 됩니다."
echo "  못 빼오시면 위 '챔프 단독 요약' 화면 두 개를 복사해 주세요."
