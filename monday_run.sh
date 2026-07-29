#!/bin/bash
# 월요일 원터치 러너: 데이터 신선도 확인 → 일간 신호 → shadow 기록 → 주문리스트 출력.
# 신호·주문리스트만 생성. 실주문/증권사로그인/live_orders 없음 — 사용자가 검토·실행.
set -e
SC=/Users/wooyeol/work/stock-v2-candidate-v17
PY=/Users/wooyeol/work/stock-v2/.venv-mps-max/bin/python
cd /Users/wooyeol/work/stock-v2
MODEL="${1:-models/v16_buysell_5fold_seed17_20260718/r5}"   # v17 최적에폭 나오면 교체
CAPITAL="${2:-100000}"
OUT="reports/daily_signal_$(python3 -c 'import datetime;print(datetime.date.today().isoformat())' 2>/dev/null || echo latest).json"

echo "================ 월요일 운용 러너 ================"
echo "모델: $MODEL   자본(스모크): ${CAPITAL}원"

echo "--- 1) 데이터 신선도 ---"
$PY - <<PY
import sys; sys.path.insert(0,"$SC")
import pandas as pd, glob
f=sorted(glob.glob("data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4/ohlcv/*.csv"))[0]
d=pd.read_csv(f,usecols=["Date"],parse_dates=["Date"])
last=d["Date"].max(); import datetime; today=pd.Timestamp(datetime.date.today())
gap=(today-last).days
print(f"  OHLCV 최신: {last.date()} (오늘 대비 {gap}일 전)")
print("  ⚠️ 데이터가 3일 이상 오래됨 → 라이브 신호 전 데이터 갱신 필요 (ops 수집 파이프라인)" if gap>3 else "  데이터 신선 — 진행 가능")
PY

echo "--- 2) 일간 신호 + 10만원 주문리스트 ---"
$PY "$SC/scripts/daily_signal.py" --model-dir "$MODEL" --capital "$CAPITAL" --smoke-names 5 --passes 16 --device mps --output "$OUT" 2>/dev/null | grep -vE "scikit|Torch|warn" | sed -n '/일간 신호/,$p'

echo "--- 3) shadow 페이퍼 원장 기록 ---"
$PY "$SC/scripts/shadow_log.py" --append "$OUT" --horizon 10 2>/dev/null | tail -1
$PY "$SC/scripts/shadow_log.py" --status 2>/dev/null | grep -E "열림|닫힌"

echo "--- 4) 정산 (만기된 과거 shadow 항목) ---"
$PY "$SC/scripts/shadow_log.py" --reconcile 2>/dev/null | tail -1

echo "================================================="
echo "⚠️ 안전: 위는 신호·주문리스트·페이퍼기록뿐. 실주문·로그인·live_orders 없음."
echo "   10만원 실테스트는 주문리스트를 사용자가 직접 증권사에서 검토·실행."
