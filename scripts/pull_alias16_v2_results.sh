#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/Users/wooyeol/work/stock-v2"
POD="root@194.68.245.170"
SSH_PORT="22045"
SSH_KEY="/Users/wooyeol/.ssh/id_ed25519"
REMOTE_ROOT="/workspace/stock-v2"
BASE="data/staging/news_alias16_partial_v14_contract_v2_20260712"
REMOTE_PID_FILE="reports/news_alias16_v2_pipeline_resume_20260712.pid"
LOG="$ROOT/logs/pull_alias16_v2_results_20260713.log"

mkdir -p "$ROOT/logs"
exec > >(tee -a "$LOG") 2>&1

timestamp() {
  date '+%Y-%m-%dT%H:%M:%S%z'
}

remote() {
  ssh -p "$SSH_PORT" -i "$SSH_KEY" -o BatchMode=yes "$POD" "$@"
}

echo "[$(timestamp)] waiting for alias16 v2 resume pipeline"
while remote "pid=\$(cat '$REMOTE_ROOT/$REMOTE_PID_FILE' 2>/dev/null || true); test -n \"\$pid\" && kill -0 \"\$pid\" 2>/dev/null"; do
  sleep 300
done

STRUCTURED="$BASE/structured_qwen35_9b_semantic095_ticker_balanced300_seed20260713_v14.jsonl"
remote "test -s '$REMOTE_ROOT/$STRUCTURED'" || {
  echo "[$(timestamp)] pipeline exited without a structured output" >&2
  exit 1
}

echo "[$(timestamp)] pulling immutable alias16 v2 artifacts"
mkdir -p "$ROOT/$BASE" "$ROOT/reports"
RSYNC_SSH="ssh -p $SSH_PORT -i $SSH_KEY -o BatchMode=yes"
for file in \
  qwen3_embedding_06b_dim512.npy \
  qwen3_embedding_06b_dim512.manifest.json \
  structure_queue_semantic095_v2.jsonl \
  semantic_clusters_095_v2.jsonl \
  structure_queue_semantic095_ticker_balanced300_seed20260713.jsonl \
  structured_qwen35_9b_semantic095_ticker_balanced300_seed20260713_v14.jsonl; do
  rsync -rt --partial -e "$RSYNC_SSH" \
    "$POD:$REMOTE_ROOT/$BASE/$file" "$ROOT/$BASE/"
done
for file in \
  news_alias16_v2_semantic095_ticker_balanced300_seed20260713.json \
  news_alias16_v2_pipeline_20260712.log \
  news_alias16_v2_pipeline_resume_20260712.log; do
  rsync -rt --partial -e "$RSYNC_SSH" \
    "$POD:$REMOTE_ROOT/reports/$file" "$ROOT/reports/"
done

cd "$ROOT"
"/Users/wooyeol/work/stock/venv/bin/python" - "$BASE" <<'PY'
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

base = Path(sys.argv[1])
paths = {
    "semantic": base / "structure_queue_semantic095_v2.jsonl",
    "sample": base / "structure_queue_semantic095_ticker_balanced300_seed20260713.jsonl",
    "structured": base / "structured_qwen35_9b_semantic095_ticker_balanced300_seed20260713_v14.jsonl",
}
def records(path: Path):
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)

expected = {"semantic": 95208, "sample": 3900, "structured": 3900}
actual = {name: sum(1 for _row in records(path)) for name, path in paths.items()}
if actual != expected:
    raise SystemExit(f"row-count mismatch: expected={expected} actual={actual}")
if any(not bool(row.get("llm_used")) for row in records(paths["structured"])):
    raise SystemExit("structured output contains an LLM error row")
if {row.get("input_hash_policy") for row in records(paths["sample"])} != {
    "news-structure-input-v2"
}:
    raise SystemExit("sample contains a non-v2 input hash policy")
print(
    json.dumps(
        {
            "rows": actual,
            "sha256": {
                name: hashlib.sha256(path.read_bytes()).hexdigest()
                for name, path in paths.items()
            },
        },
        indent=2,
        sort_keys=True,
    )
)
PY

touch "$ROOT/$BASE/PULLED_FROM_RUNPOD"
echo "[$(timestamp)] alias16 v2 result pull complete"
