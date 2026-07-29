#!/usr/bin/env bash
set -Eeuo pipefail

cd /workspace/stock-v2

python=/root/venvs/news-vllm-cu128/bin/python
base=data/staging/news_alias16_partial_v14_contract_v2_20260712
queue=$base/structure_queue.jsonl
embedding=$base/qwen3_embedding_06b_dim512.npy
embedding_manifest=$base/qwen3_embedding_06b_dim512.manifest.json
semantic_queue=$base/structure_queue_semantic095_v2.jsonl
semantic_clusters=$base/semantic_clusters_095_v2.jsonl
sample=$base/structure_queue_semantic095_ticker_balanced300_seed20260713.jsonl
sample_report=reports/news_alias16_v2_semantic095_ticker_balanced300_seed20260713.json
structured=$base/structured_qwen35_9b_semantic095_ticker_balanced300_seed20260713_v14.jsonl

expected_queue_sha=21db0a0dc7e017d144197f7b34301074aed71316786e878b0070bce9fe7eab90
expected_embedding_sha=a8852d6bdeb529c85bc4b2a4c7a27a7ca41ac12cd5fefd050d6673b34b75df5c

for input in "$queue" "$embedding" "$embedding_manifest" "$semantic_queue" "$semantic_clusters"; do
  if [[ ! -s "$input" ]]; then
    echo "required completed stage is missing: $input" >&2
    exit 3
  fi
done
for output in "$sample" "$sample_report" "$structured"; do
  if [[ -e "$output" ]]; then
    echo "refusing to overwrite existing output: $output" >&2
    exit 4
  fi
done

actual_queue_sha=$(sha256sum "$queue" | cut -d' ' -f1)
actual_embedding_sha=$(sha256sum "$embedding" | cut -d' ' -f1)
if [[ "$actual_queue_sha" != "$expected_queue_sha" ]]; then
  echo "queue hash mismatch: $actual_queue_sha" >&2
  exit 5
fi
if [[ "$actual_embedding_sha" != "$expected_embedding_sha" ]]; then
  echo "embedding hash mismatch: $actual_embedding_sha" >&2
  exit 6
fi

"$python" - "$embedding_manifest" "$semantic_queue" <<'PY'
from __future__ import annotations

import json
from pathlib import Path
import sys

manifest = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
if manifest.get("input_sha256") != "21db0a0dc7e017d144197f7b34301074aed71316786e878b0070bce9fe7eab90":
    raise SystemExit("embedding manifest input hash mismatch")
if manifest.get("output_sha256") != "a8852d6bdeb529c85bc4b2a4c7a27a7ca41ac12cd5fefd050d6673b34b75df5c":
    raise SystemExit("embedding manifest output hash mismatch")
if manifest.get("rows") != 96638 or manifest.get("embedding_dimension") != 512:
    raise SystemExit("embedding manifest shape mismatch")

semantic_path = Path(sys.argv[2])
rows = sum(1 for line in semantic_path.open(encoding="utf-8") if line.strip())
if rows != 95208:
    raise SystemExit(f"unexpected semantic queue rows: {rows}")
print(json.dumps({"validated_embedding_rows": 96638, "semantic_queue_rows": rows}))
PY

"$python" scripts/sample_news_queue.py \
  --queue "$semantic_queue" \
  --per-ticker 300 \
  --seed 20260713 \
  --output "$sample" \
  --report "$sample_report"

"$python" scripts/structure_news_queue_hf.py \
  --input "$sample" \
  --output "$structured" \
  --model-path /workspace/hf-cache/hub/models--Qwen--Qwen3.5-9B/snapshots/c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --model-id Qwen/Qwen3.5-9B \
  --model-revision c202236235762e1c871ad0ccb60c8ee5ba337b9a \
  --batch-size 6 \
  --max-new-tokens 256 \
  --repair-retries 1 \
  --dtype bfloat16 \
  --device cuda:0 \
  --print-every 300

"$python" - "$sample" "$structured" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

sample_path = Path(sys.argv[1])
structured_path = Path(sys.argv[2])

def rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

sample = rows(sample_path)
structured = rows(structured_path)
summary = {
    "sample_rows": len(sample),
    "structured_rows": len(structured),
    "llm_errors": sum(not bool(row.get("llm_used")) for row in structured),
    "input_hash_policies": sorted(
        {str(row.get("input_hash_policy") or "missing") for row in sample}
    ),
    "sample_sha256": hashlib.sha256(sample_path.read_bytes()).hexdigest(),
    "structured_sha256": hashlib.sha256(structured_path.read_bytes()).hexdigest(),
}
print(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True))
if (
    summary["sample_rows"] != summary["structured_rows"]
    or summary["llm_errors"]
    or summary["input_hash_policies"] != ["news-structure-input-v2"]
):
    raise SystemExit(7)
PY
