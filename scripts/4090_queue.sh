#!/usr/bin/env bash
# 4090 큐 — 사전등록 검정용 r3 챔프 예측 생성 (2026-07-31 개정)
#
# ── 왜 이걸 돌리나 ────────────────────────────────────────────────────────
# docs/PREREG_FILTER_R3_20260731.md 사전등록 검정에 r3 폴드의 챔프 종목별
# 예측이 필요하다. r3 기존 런 3개는 future_rollout.csv(일별 집계)만 있고
# 종목별 예측이 없다 — --save-return-forecasts 는 2026-07-30 에 추가됐다.
#
# 이 큐가 만드는 것: ens_s3/17/29 @ r3 의 return_1d_forecasts.csv
# 그것이 있어야 A(챔프 상위20) / C(챔프필터+Chronos) / D(Chronos 단독) 검정이 된다.
#
# ── 이전 큐들을 왜 버렸나 ─────────────────────────────────────────────────
# ema 0.99 / VICReg: 지평 혼합 IC 로 매긴 순위였고, 지평 10 으로 다시 재면
#   둘 다 챔프보다 나쁘다(-0.8σ / -0.9σ). docs/MEASUREMENT_CORRECTIONS_20260730.md
# context 모드 x r4: r5 에서 4/4 열세(짝지은 t -3.83)로 이미 결론이 났다.
#
# ── 사전 준비 ────────────────────────────────────────────────────────────
#   1) git clone git@github.com:wyjun1988/graph-jepa.git  (또는 git pull)
#      → 오늘 수정분이 반드시 포함돼야 한다:
#         - graph_jepa.py 의 --temporal-head-input
#         - evaluate_node_prediction.py 의 load_model 배선 (없으면 평가가 죽는다)
#         - seed_queue_v2.sh 의 guard 오탐 수정
#   2) data/ 를 graph-jepa-4090.tar.gz 에서 풀어 넣기
#   3) venv 준비 (torch + numpy 2.3.5 + pandas 2.3.3 권장 — 버전 고정이 재현에 중요)
#
# ── 실행 ─────────────────────────────────────────────────────────────────
#   nohup bash scripts/4090_queue.sh > 4090.log 2>&1 &
#   tail -f 4090.log
#
# ── 끝나고 돌려줄 것 ──────────────────────────────────────────────────────
#   reports/walk_forward/node_eval/ens_s*_fold1_20240104_to_20241107/
#     ├── future_rollout.csv          (필수)
#     └── return_1d_forecasts.csv     (필수 — 재분석 전부 여기서 나온다)
#   tar -czf r3_results.tar.gz \
#     reports/walk_forward/node_eval/ens_s*_fold1_20240104_to_20241107/{future_rollout,return_1d_forecasts}.csv
#
# 예상 소요: 3시드 x 약 35~45분 = 2시간 내외 (4090 이 A5000 보다 다소 빠름)
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

echo "════ 4090 큐: 사전등록 검정용 r3 챔프 3시드 ════"
echo "python : $PY"
"$PY" -c "import torch;print('torch  :',torch.__version__,'| cuda',torch.cuda.is_available(),'|',torch.cuda.get_device_name(0) if torch.cuda.is_available() else '-')"
echo ""

# 사전 점검 — 없는 채로 2시간 돌리고 평가에서 죽는 일을 막는다
echo "── 사전 점검 ──"
fail=0
"$PY" - <<'PYEOF' || fail=1
import sys
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
print("  평가 load_model 배선 OK")
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

PREFIX=ens_s bash scripts/seed_queue_v2.sh 1 r3 3 17 29

echo ""
echo "════ 결과 압축·요약 ════"
# 원본 예측은 시드당 약 85MB 라 빼오기 어렵다. 분석에 실제로 필요한 부분만
# 1MB 남짓으로 줄이고, 동시에 챔프 단독 지표를 화면에 찍는다.
"$PY" scripts/distill_forecasts.py --fold r3 --seeds 3 17 29 --out r3_compact.csv
gzip -kf r3_compact.csv 2>/dev/null && echo "  gzip: r3_compact.csv.gz ($(du -h r3_compact.csv.gz 2>/dev/null | cut -f1))"
echo ""
echo "════ 돌려주실 것 ════"
echo "  1순위: r3_compact.csv.gz  (수백 KB — 이것만 있으면 사전등록 검정이 끝납니다)"
echo "  2순위: 위 '챔프 단독 요약' 화면 복사 (파일을 못 빼올 때)"
echo ""
echo "원본이 필요하면(선택):"
echo "  tar -czf r3_full.tar.gz reports/walk_forward/node_eval/ens_s*_fold1_20240104_to_20241107/{future_rollout,return_1d_forecasts}.csv"
