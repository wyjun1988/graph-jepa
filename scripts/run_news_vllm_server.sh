#!/usr/bin/env bash
set -euo pipefail

MODEL_ID=${1:?model id required}
MODEL_REVISION=${2:?model revision required}
SERVED_NAME=${3:?served model name required}
PORT=${4:-8000}
VLLM_ENV=${VLLM_ENV:-/root/venvs/news-vllm-cu128}
HF_HOME=${HF_HOME:-/workspace/hf-cache}

export HF_HOME
export TOKENIZERS_PARALLELISM=true

exec "${VLLM_ENV}/bin/vllm" serve "${MODEL_ID}" \
  --revision "${MODEL_REVISION}" \
  --served-model-name "${SERVED_NAME}" \
  --host 127.0.0.1 \
  --port "${PORT}" \
  --tensor-parallel-size 1 \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.92 \
  --language-model-only
