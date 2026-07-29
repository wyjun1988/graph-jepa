#!/bin/bash
# 전체 센서 일일 갱신 — 라이브 센싱을 최신 거래일까지 끌어올린다.
# 운영자가 크레덴셜(--env-file)로 실행. OHLCV는 기존 daily-causal-shadow가 평일 15:45
# 자동 갱신하지만 investor/fundamentals/news는 빠져 있어 여기서 함께 갱신한다.
#
# 사용: bash refresh_all_sensors.sh [END_DATE] [ENV_FILE]
#   END_DATE  기본 오늘 (YYYY-MM-DD)
#   ENV_FILE  기본 /Users/wooyeol/work/stock/.env (Kiwoom+DART 키)
#
# 크레덴셜(.env)에 필요: KIWOOM app key/secret, DART API key. 수집기는 stock 프로젝트
# venv(kiwoom/dart 클라이언트 포함)를 쓴다.
set -uo pipefail
LEDGER=/Users/wooyeol/work/stock-v2
END="${1:-$(date +%F)}"
ENVF="${2:-/Users/wooyeol/work/stock/.env}"
UNIV="data/universes/krx500_pit_20191231.json"
PY_COLLECT=/Users/wooyeol/work/stock/venv/bin/python     # 수집기용 (kiwoom/dart deps)
PY_V2=/Users/wooyeol/work/stock-v2/.venv-mps-max/bin/python
INCR_START="2026-07-11"                                  # 마지막 안정일 다음날(운영자 확인)
cd "$LEDGER"
# ⚠️ Kiwoom은 IP 허용목록 검증 → 등록 IP 머신(M1 Pro 182.224.205.217)에서만 인증됨.
#    M1 Max는 동적 IP(원래 .219 → 현재 .199, 미등록)라 인증 거부. 수집기는 M1 Pro에서 실행.
MYIP=$(curl -s --max-time 8 https://api.ipify.org 2>/dev/null)
case "$MYIP" in
  182.222.134.219|182.224.100.173|182.224.205.217) echo "IP $MYIP 허용됨 — 진행" ;;
  *) echo "ABORT: 현재 IP [$MYIP] 미등록. M1 Pro(허용 IP)에서 실행하세요."; exit 1 ;;
esac
echo "================ 센서 전체 갱신 (END=$END) ================"
[ -f "$ENVF" ] || { echo "ABORT: env-file 없음: $ENVF (크레덴셜 필요)"; exit 1; }

run() { echo "--- $1 $(date +%T) ---"; shift; "$@"; echo "  exit=$?"; }

# 1) OHLCV (일봉) — 기존 daily job과 동일 경로. 실패 시 daily-causal-shadow 로그 참조.
run "OHLCV 수집" $PY_COLLECT scripts/backfill_kiwoom_ohlcv.py \
  --universe-manifest "$UNIV" --start "$INCR_START" --end "$END" \
  --basis causal --cache-dir data/kiwoom_ohlcv_cache_live --raw-cache-dir data/kiwoom_ohlcv_raw_live \
  --env-file "$ENVF" --sleep-sec 0.22 --max-pages 20 --resume

# 2) 투자자 순매수 (외국인/기관) — 빠져있던 센서
run "investor 순매수 수집" $PY_COLLECT scripts/cache_kiwoom_investor_flows.py \
  --universe-manifest "$UNIV" --start "$INCR_START" --end "$END" \
  --cache-dir data/kiwoom_investor_cache --env-file "$ENVF" --sleep-sec 0.22 --resume

# 3) 재무 (DART) — 구조적 지연이나 신규 공시 반영
run "fundamentals(DART) 수집" $PY_COLLECT scripts/backfill_dart_fundamentals.py \
  --universe-manifest "$UNIV" --start-year 2026 --end-year 2026 \
  --output data/fundamentals/opendart_krx500_pit_2020_2026_clean.jsonl \
  --api-key-env OPENDART_API_KEY --env-file "$ENVF" --sleep-sec 0.22 --resume

# 4) 뉴스/이벤트
run "news/events 수집" $PY_COLLECT scripts/backfill_news_events.py \
  --universe-manifest "$UNIV" --start "$INCR_START" --end "$END" \
  --output data/staging/news_krx500_dart_pit_v2_20260712/neutral_events.jsonl --resume

# 5) 운용 패널 전방 확장 (과거 불변 → 데이터계약 manifest 일치 유지)
run "lifecycle 패널 전방확장" $PY_V2 scripts/extend_lifecycle_release.py \
  --base-manifest data/staging/ohlcv_lifecycle_hybrid_krx500_pit_20260710_v4 \
  --universe-manifest "$UNIV" --incremental-start "$INCR_START" --end "$END" \
  --output-root data/staging --current-link data/ops/current_ohlcv_release

# 6) 센싱 준비도 확인 (제가 만든 체커)
echo "--- 센서 준비도 확인 ---"
$PY_V2 scripts/../../stock-v2-candidate-v17/scripts/sensor_status.py --asof "$END"
echo "==========================================================="
echo "완료. 준비 OK면:  bash /Users/wooyeol/work/stock-v2-candidate-v17/monday_run.sh"
echo "⚠️ 과거 조정값이 바뀌면 manifest 불일치 → 재학습 필요(A4000). 그 경우 로그에 'manifest' 오류."
