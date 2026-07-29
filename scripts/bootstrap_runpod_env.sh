#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${1:-/root/venvs/news-vllm-cu128}"
PYTHON="$VENV/bin/python"
PIP="$VENV/bin/pip"

if [[ ! -x "$PYTHON" || ! -x "$PIP" ]]; then
  echo "missing Python environment: $VENV" >&2
  exit 2
fi

cd "$ROOT"
torch_before="$($PYTHON -c 'import torch; print(torch.__version__)')"
transformers_before="$($PYTHON -c 'import transformers; print(transformers.__version__)')"

"$PIP" install --no-cache-dir -r requirements-runpod.txt

torch_after="$($PYTHON -c 'import torch; print(torch.__version__)')"
transformers_after="$($PYTHON -c 'import transformers; print(transformers.__version__)')"
if [[ "$torch_before" != "$torch_after" ]]; then
  echo "refusing changed Torch runtime: $torch_before -> $torch_after" >&2
  exit 3
fi
if [[ "$transformers_before" != "$transformers_after" ]]; then
  echo "refusing changed Transformers runtime: $transformers_before -> $transformers_after" >&2
  exit 4
fi

"$PYTHON" - <<'PY'
from __future__ import annotations

import json

import FinanceDataReader
import joblib
import numpy
import pandas
import scipy
import sklearn
import torch
import transformers

print(
    json.dumps(
        {
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_runtime": torch.version.cuda,
            "transformers": transformers.__version__,
            "numpy": numpy.__version__,
            "pandas": pandas.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
            "finance_datareader": FinanceDataReader.__version__,
            "joblib": joblib.__version__,
        },
        indent=2,
        sort_keys=True,
    )
)
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available in the RunPod runtime")
PY

CUDA_VISIBLE_DEVICES= "$PYTHON" -m pytest -q
