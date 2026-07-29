#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/workspace/stock-v2}"
PYTHON="${PYTHON:-/root/venvs/news-vllm-cu128/bin/python}"
HF_HOME="${HF_HOME:-/workspace/hf-cache}"
QUEUE="${QUEUE:-${ROOT}/data/staging/news_krx500_pit_20260710_v1/structure_queue.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-${ROOT}/data/staging/news_krx500_pit_20260710_v1}"
SAMPLE_SIZE="${SAMPLE_SIZE:-500}"
SAMPLE_SEED="${SAMPLE_SEED:-20260712}"

MODEL_4B_REVISION="851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a"
MODEL_9B_REVISION="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
MODEL_4B_PATH="${HF_HOME}/hub/models--Qwen--Qwen3.5-4B/snapshots/${MODEL_4B_REVISION}"
MODEL_9B_PATH="${HF_HOME}/hub/models--Qwen--Qwen3.5-9B/snapshots/${MODEL_9B_REVISION}"
OUTPUT_4B="${OUTPUT_DIR}/structured_qwen35_4b_sample${SAMPLE_SIZE}_seed${SAMPLE_SEED}_v5.jsonl"
OUTPUT_9B="${OUTPUT_DIR}/structured_qwen35_9b_sample${SAMPLE_SIZE}_seed${SAMPLE_SEED}_v5.jsonl"

for required in "$PYTHON" "$QUEUE" "$MODEL_4B_PATH" "$MODEL_9B_PATH"; do
  test -e "$required" || { echo "missing required path: $required" >&2; exit 1; }
done
if test -e "$OUTPUT_4B" || test -e "$OUTPUT_9B"; then
  echo "comparison output already exists; choose a new OUTPUT_DIR or remove the incomplete run" >&2
  exit 1
fi

cd "$ROOT"
export HF_HOME CUDA_VISIBLE_DEVICES=0

status=0
"$PYTHON" scripts/structure_news_queue_hf.py \
  --input "$QUEUE" \
  --output "$OUTPUT_4B" \
  --model-path "$MODEL_4B_PATH" \
  --model-id Qwen/Qwen3.5-4B \
  --model-revision "$MODEL_4B_REVISION" \
  --batch-size 32 \
  --max-new-tokens 256 \
  --repair-retries 2 \
  --sample-size "$SAMPLE_SIZE" \
  --sample-seed "$SAMPLE_SEED" \
  --print-every 100 || status=$?

"$PYTHON" scripts/structure_news_queue_hf.py \
  --input "$QUEUE" \
  --output "$OUTPUT_9B" \
  --model-path "$MODEL_9B_PATH" \
  --model-id Qwen/Qwen3.5-9B \
  --model-revision "$MODEL_9B_REVISION" \
  --batch-size 12 \
  --max-new-tokens 256 \
  --repair-retries 2 \
  --sample-size "$SAMPLE_SIZE" \
  --sample-seed "$SAMPLE_SEED" \
  --print-every 100 || status=$?

exit "$status"
