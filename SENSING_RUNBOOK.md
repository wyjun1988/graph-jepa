# 센싱 운용 런북 (v17 후보)

라이브 신호를 내려면 매일: **센서 갱신 → 준비도 확인 → 신호 생성**.

## 1. 센싱 설계 (완료·검증)
모든 입력이 인과(누수없음) 센서 + fail-closed:
- OHLCV(당일), 투자자순매수(lag1), 재무 DART(lag1+available_at), 이벤트(cov≥0.99), 외부팩터(lag1).
- 커버리지 미달 시 모델이 실행 거부 → 오염 신호 대신 안전.

## 2. 매일 절차 (운영자, 장마감 후)

### (a) 센서 갱신
- **OHLCV**: 이미 자동 — launchd `daily-causal-shadow`가 **평일 15:45 KST** 수집+패널빌드.
- **전체(OHLCV+투자자+재무+뉴스)**: 아래 한 번에
  ```
  bash /Users/wooyeol/work/stock-v2-candidate-v17/refresh_all_sensors.sh [END_DATE] [ENV_FILE]
  ```
  - `ENV_FILE` 기본 `/Users/wooyeol/work/stock/.env` (Kiwoom·DART 키 — **존재 확인됨**).
  - ⚠️ **최초 실행 전**: 스크립트 안의 경로/인자(cache-dir, INCR_START, base-manifest)를 실제 `daily-causal-shadow`(run_daily_causal_shadow.py) 호출과 대조해 확인. 조립본이라 첫 실행은 감독 필요.

### (b) 준비도 확인 (검증됨, 언제든 실행 가능)
```
.venv-mps-max/bin/python /Users/wooyeol/work/stock-v2-candidate-v17/scripts/sensor_status.py --asof YYYY-MM-DD
```
→ 센서별 최신일·커버리지·상태(OK/STALE). "센서 신선"이면 신호 생성 가능.

### (c) 신호 생성
```
bash /Users/wooyeol/work/stock-v2-candidate-v17/monday_run.sh
```
→ 신호·shadow기록·10만원 주문리스트. (실주문·로그인은 운영자가 직접.)

## 3. 데이터계약 주의
- 패널 **전방확장**(과거 불변)이면 manifest 일치 → v16 그대로 추론. `extend_lifecycle_release.py`가 이를 수행.
- 과거 조정값이 바뀌면(vendor 조정계수·정정공시) manifest 불일치 → 로그에 `manifest` 오류 → **A4000 재학습** 필요.

## 4. 현재 상태 (2026-07-19)
- 센싱 설계·준비도 체커·수집기·크레덴셜·venv: **모두 준비**.
- 데이터 신선도: OHLCV 07-14, 투자자 07-10, 재무 07-09 → 신선 신호는 `refresh_all_sensors.sh` 1회 실행 후.
- refresh_all_sensors.sh: **조립 완료, 크레덴셜 필요라 미테스트** — 첫 실행 감독 권장.
