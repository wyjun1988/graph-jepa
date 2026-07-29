#!/usr/bin/env bash
set -euo pipefail

wait_pid="${1:?usage: run_alias16_balanced_v14_after_pid.sh WAIT_PID}"
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python_bin="${PYTHON_BIN:-/root/venvs/news-vllm-cu128/bin/python}"
model_revision="c202236235762e1c871ad0ccb60c8ee5ba337b9a"
model_path="${MODEL_PATH:-/workspace/hf-cache/hub/models--Qwen--Qwen3.5-9B/snapshots/${model_revision}}"
input_path="data/staging/news_alias16_partial_v14_contract_20260712/structure_queue_ticker_balanced300_seed20260713.jsonl"
output_path="data/staging/news_alias16_partial_v14_contract_20260712/structured_qwen35_9b_ticker_balanced300_seed20260713_v14.jsonl"

while kill -0 "${wait_pid}" 2>/dev/null; do
  sleep 30
done

cd "${repo_root}"
if [[ -e "${output_path}" ]]; then
  echo "refusing to overwrite existing output: ${output_path}" >&2
  exit 1
fi

exec "${python_bin}" scripts/structure_news_queue_hf.py \
  --input "${input_path}" \
  --output "${output_path}" \
  --model-path "${model_path}" \
  --model-id Qwen/Qwen3.5-9B \
  --model-revision "${model_revision}" \
  --batch-size 6 \
  --max-new-tokens 256 \
  --repair-retries 1 \
  --dtype bfloat16 \
  --device cuda:0 \
  --print-every 300
