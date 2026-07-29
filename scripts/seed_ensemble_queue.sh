#!/usr/bin/env bash
# 챔프(vf_s0.0_seed17) 설정을 시드만 바꿔 재학습 — 시드 앙상블 재료 생성.
#
# 왜: 59개 실험 중 2σ를 넘긴 설정이 하나도 없었고, 시드 산포(σ=0.0094)가
# 설정 간 차이보다 컸다. 지금 프로덕션 신호는 seed 17 단일 복권이다.
# 예측을 여러 시드에 걸쳐 평균하면 독립 노이즈가 상쇄돼 IC가 오를 수 있다 —
# 설정 탐색보다 기대값이 높고 실패 위험이 없는 유일한 남은 카드.
#
# 검증 겸용: seed 3/5/11/23/29 는 이전 pod의 sig_s* 와 동일 설정이므로
# IC가 각각 +0.0521/+0.0261/+0.0299/+0.0419/+0.0400 을 재현해야 한다.
# 재현되면 새 pod의 데이터·코드·torch 환경이 검증된 것.
#
# 사용법:  bash scripts/seed_ensemble_queue.sh <동시실행수> <폴드:r4|r5> <시드...>
#   예)    bash scripts/seed_ensemble_queue.sh 1 r5 3
#          bash scripts/seed_ensemble_queue.sh 2 r5 5 11 17 23 29
#          bash scripts/seed_ensemble_queue.sh 2 r4 3 5 11 17 23 29

set -u
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

ROOT=/workspace/stock-v2-candidate-v17
PY=/workspace/venv/bin/python
cd "$ROOT"

CONCURRENCY="${1:-1}"; shift
FOLD_TAG="${1:-r5}"; shift
case "$FOLD_TAG" in
  r5) FOLD="2025-09-05:2026-07-10"; SUF="20250905_to_20260710" ;;
  r4) FOLD="2024-11-06:2025-09-08"; SUF="20241106_to_20250908" ;;
  *)  echo "폴드는 r4 또는 r5"; exit 1 ;;
esac

SEEDS=("$@")
[ ${#SEEDS[@]} -eq 0 ] && { echo "시드를 지정하세요"; exit 1; }

# 설정 변형용 (환경변수). BASE 뒤에 붙으므로 같은 플래그면 이쪽이 이긴다.
#   예) PREFIX=ema_s EXTRA="--ema-decay 0.99 --hidden-completion-weight 0.25" \
#         bash scripts/seed_ensemble_queue.sh 2 r5 3 5 11 23
PREFIX="${PREFIX:-ens_s}"
EXTRA="${EXTRA:-}"

# 동시 실행 시 96 vCPU를 나눠 쓴다 (경합하면 동시실행 이득이 사라짐)
if [ "$CONCURRENCY" -gt 1 ]; then WORKERS=40; else WORKERS=48; fi

# 챔프 설정 — sig_s* 큐와 동일한 BASE (sequence-window 0 = 어텐션 없음)
BASE="--hidden-dim 1024 --layers 10 --train-batch-size 16 --snapshot-workers $WORKERS \
--device cuda --eval-device cuda --max-steps 0 --lr 3e-4 --horizon 10 \
--universe krx --universe-manifest data/universes/krx500_pit_20191231.json --max-tickers 500 \
--cache-dir data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv \
--edge-correlation-mode signed --industry-edge-scale 0.2 --edge-top-k 6 \
--external-node-mode nodes --external-preset kr_global_rates --external-cache-dir data/external_cache \
--external-lag-days 1 --require-all-external-factors \
--event-path data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl \
--fundamental-path data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
--investor-cache-dir data/kiwoom_investor_cache \
--require-event-sensors --require-fundamental-sensors --require-investor-sensors \
--min-event-coverage 0.99 --min-fundamental-coverage 0.79 --min-investor-coverage 0.95 \
--event-coverage-mode mask_uncovered --fundamental-lag-days 1 --investor-flow-lag-days 1 \
--graph-neighbor-scale 1.0 --temporal-graph-neighbor-scale 0.0 \
--temporal-state-mode horizon_residual_heads --temporal-state-context-skip \
--mask-strategy mixed --training-manifest-schema-version 4 \
--temporal-exclude-feature-prefix fund_ --policy-rate-edge-scale 0.0 --amp-dtype bfloat16 \
--ema-decay 0.9995 --state-loss-weight 1.0 --downstream-auxiliary-loss-weight 0.25 \
--current-imputation-loss-weight 1.0 --entry-path-correlation-loss-weight 0.05 \
--latent-loss-weight 0.25 --sequence-window 0 \
--save-return-forecasts"

mkdir -p ops/training

# A5000 24GB 에서 이 설정은 11.8GB 를 쓴다 → 동시 2개는 OOM. 앞 런이 비면 시작.
guard(){
  while pgrep -f 'run_real_backtest|evaluate_node_prediction' > /dev/null 2>&1; do
    sleep 30
  done
}

run_seed(){
  local S="$1"
  local N="${PREFIX}${S}"
  local LOG="ops/training/${N}_${FOLD_TAG}.log"
  guard
  if [ -f "reports/walk_forward/node_eval/${N}_fold1_${SUF}/future_rollout.csv" ]; then
    echo "  skip ${N}(${FOLD_TAG}) — 이미 완료"; return 0
  fi
  echo "  ▶ [$(date '+%H:%M:%S')] ${N}(${FOLD_TAG}) 시작"
  "$PY" scripts/run_walk_forward_node_eval.py --name "$N" --fold "$FOLD" --start 2020-01-01 \
    --epochs 24 --checkpoint-epochs 12 --seed "$S" $BASE $EXTRA \
    > "$LOG" 2>&1
  local rc=$?
  if [ $rc -eq 0 ]; then
    echo "  ✅ [$(date '+%H:%M:%S')] ${N}(${FOLD_TAG}) 완료"
  else
    echo "  ❌ [$(date '+%H:%M:%S')] ${N}(${FOLD_TAG}) 실패 (rc=$rc)"
    grep -iE "error|out of memory|Traceback" "$LOG" | tail -3
  fi
}

echo "════ 시드 앙상블 큐 $(date '+%Y-%m-%d %H:%M') ════"
echo "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader)"
echo "폴드: ${FOLD_TAG} (${FOLD}) | 시드: ${SEEDS[*]} | 동시실행: ${CONCURRENCY} | 워커: ${WORKERS}"
echo "이름: ${PREFIX}* | 추가플래그: ${EXTRA:-(없음)}"
echo ""

running=0
for S in "${SEEDS[@]}"; do
  run_seed "$S" &
  running=$((running+1))
  if [ "$running" -ge "$CONCURRENCY" ]; then wait -n; running=$((running-1)); fi
done
wait

echo ""
echo "════ 큐 종료 ${FOLD_TAG} $(date '+%Y-%m-%d %H:%M') ════"
for S in "${SEEDS[@]}"; do
  f="reports/walk_forward/node_eval/${PREFIX}${S}_fold1_${SUF}/future_rollout.csv"
  [ -f "$f" ] && echo "  ${PREFIX}${S}: 완료" || echo "  ${PREFIX}${S}: 없음"
done
