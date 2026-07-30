#!/usr/bin/env bash
# 4090 큐 — 헤드 입력 실험을 폴드 r4 로 확장 (2026-07-30 개정)
#
# ── 왜 이걸 돌리나 ────────────────────────────────────────────────────────
# A5000 에서 폴드 r5 로 context 모드(미래 잠재 미사용) 4시드를 돌리고 있다.
# 4090 은 같은 실험을 폴드 r4 로 돌려 다폴드 근거를 만든다(docs §7-5 기준).
# 기준선 ens_s3/17/29 의 r4 결과는 이미 확보돼 있어 짝지은 비교가 바로 된다.
#
# ── 왜 이전 큐(ema 0.99 / VICReg)를 버렸나 ────────────────────────────────
# 그 후보들은 지평을 섞은 IC 로 순위를 매긴 결과였다. 10일 보유와 직결되는
# 지평 10 으로 다시 재면 둘 다 챔프보다 나쁘다:
#     챔프 +0.0549 | ema 0.99 +0.0424 (-0.8σ) | VICReg lat1 +0.0400 (-0.9σ)
# 자세한 근거: docs/MEASUREMENT_CORRECTIONS_20260730.md
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
#   reports/walk_forward/node_eval/ctx_s*_fold1_20241106_to_20250908/
#     ├── future_rollout.csv          (필수)
#     └── return_1d_forecasts.csv     (필수 — 재분석 전부 여기서 나온다)
#   tar -czf ctx_r4_results.tar.gz \
#     reports/walk_forward/node_eval/ctx_s*_fold1_20241106_to_20250908/{future_rollout,return_1d_forecasts}.csv
#
# 예상 소요: 3시드 x 약 35~45분 = 2시간 내외 (4090 이 A5000 보다 다소 빠름)

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

echo "════ 4090 큐: context 모드 x 폴드 r4 ════"
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

PREFIX=ctx_s EXTRA="--temporal-head-input context" \
  bash scripts/seed_queue_v2.sh 1 r4 3 17 29

echo ""
echo "════ 완료 — 아래를 압축해 돌려주세요 ════"
ls -d reports/walk_forward/node_eval/ctx_s*_fold1_20241106_to_20250908 2>/dev/null
echo ""
echo "tar -czf ctx_r4_results.tar.gz reports/walk_forward/node_eval/ctx_s*_fold1_20241106_to_20250908/{future_rollout,return_1d_forecasts}.csv"
