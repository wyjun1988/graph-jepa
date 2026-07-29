#!/usr/bin/env bash
set -Eeuo pipefail

MODEL="${QWEN_MODEL_PATH:-/Users/wooyeol/models/qwen/Qwen3.6-27B-Q5_K_M/Qwen_Qwen3.6-27B-Q5_K_M.gguf}"
LLAMA_SERVER="${LLAMA_SERVER:-/Users/wooyeol/tools/llama.cpp/build/bin/llama-server}"
HOST="${QWEN_HOST:-127.0.0.1}"
PORT="${QWEN_PORT:-18080}"
CTX_SIZE="${QWEN_CTX_SIZE:-32768}"

if [[ ! -f "$MODEL" ]]; then
  echo "Qwen model not found: $MODEL" >&2
  exit 1
fi
if [[ ! -x "$LLAMA_SERVER" ]]; then
  echo "llama-server not found: $LLAMA_SERVER" >&2
  exit 1
fi

exec "$LLAMA_SERVER" \
  -m "$MODEL" \
  -ngl 99 \
  -c "$CTX_SIZE" \
  -fa on \
  --threads 8 \
  --parallel 1 \
  --host "$HOST" \
  --port "$PORT" \
  --reasoning off \
  "$@"
