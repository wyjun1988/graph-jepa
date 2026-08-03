#!/bin/bash
# hz 스모크 실패 원인 특정 — 4090 박스에서 실행
cd ~/work/stock-v2-candidate-v17
L=ops/training/smokehz_s17_r5.log
[ -f "$L" ] || { echo "학습 로그 없음: $L"; exit 1; }
echo "════ 단계 도달 여부 ════"
grep -c "^RUN "  "$L" | xargs echo "  RUN(학습 호출) 줄 수 :"
grep -c "^EVAL " "$L" | xargs echo "  EVAL(평가 호출) 줄 수:"
echo
echo "════ 판정 ════"
if grep -q "^EVAL " "$L"; then
  echo "  → 학습은 통과, EVAL(evaluate_node_prediction) 에서 실패"
else
  echo "  → EVAL 줄이 없다 = 학습(run_real_backtest) 단계에서 실패"
fi
echo
echo "════ 진짜 원인 (CalledProcessError 위쪽 = 내부 프로세스 stderr) ════"
grep -nE "Traceback|Error|error|Killed|out of memory|CUDA|assert|RuntimeError|ValueError" "$L" | tail -25
echo
echo "════ 로그 끝 40줄 ════"
tail -40 "$L"
