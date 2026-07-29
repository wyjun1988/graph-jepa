#!/usr/bin/env bash
# 시드 앙상블 캠페인 분석 일괄 실행.
#   bash scripts/run_reports.sh r5 3 5 11 17 23 29
set -u
ROOT=/workspace/stock-v2-candidate-v17
PY=/workspace/venv/bin/python
cd "$ROOT"

FOLD="${1:-r5}"; shift
SEEDS="$*"
OUT="reports/ensemble_campaign_${FOLD}.txt"
mkdir -p reports

{
  echo "════ 시드 앙상블 캠페인 리포트 — 폴드 ${FOLD} ════"
  echo "생성 $(date '+%Y-%m-%d %H:%M') | 시드 ${SEEDS}"
  echo ""
  echo "########## 1. 시드 수 포화곡선 ##########"
  "$PY" scripts/ensemble_report.py --seeds $SEEDS --fold "$FOLD" 2>&1
  echo ""
  echo "########## 2. 청산 정책 (편입 5종목) ##########"
  "$PY" scripts/exit_policy_report.py --seeds $SEEDS --fold "$FOLD" --top-k 5 2>&1
  echo ""
  echo "########## 3. 청산 정책 (편입 20종목 — 선별폭 의존성 확인) ##########"
  "$PY" scripts/exit_policy_report.py --seeds $SEEDS --fold "$FOLD" --top-k 20 2>&1
} | tee "$OUT"

echo ""
echo "저장: $OUT"
