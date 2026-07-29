#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/work/stock-v2}"
VENV="${VENV:-$ROOT/.venv-mps-max}"
PYTHON="$VENV/bin/python"

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This repair is only applicable to macOS." >&2
  exit 2
fi
if [[ ! -x "$PYTHON" ]]; then
  echo "Python environment is missing: $PYTHON" >&2
  exit 2
fi

SITE_PACKAGES="$($PYTHON -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
LIGHTGBM_DYLIB="$SITE_PACKAGES/lightgbm/lib/lib_lightgbm.dylib"
TORCH_LIBOMP="$SITE_PACKAGES/torch/lib/libomp.dylib"
TORCH_RELATIVE='@loader_path/../../torch/lib/libomp.dylib'

if [[ ! -f "$LIGHTGBM_DYLIB" || ! -f "$TORCH_LIBOMP" ]]; then
  echo "Both LightGBM and PyTorch must be installed in the same environment." >&2
  exit 2
fi

CURRENT_DEPENDENCY="$(otool -L "$LIGHTGBM_DYLIB" | awk '/libomp\.dylib/{print $1; exit}')"
if [[ -z "$CURRENT_DEPENDENCY" ]]; then
  echo "LightGBM has no libomp dependency to align." >&2
  exit 2
fi
if [[ "$CURRENT_DEPENDENCY" != "$TORCH_RELATIVE" ]]; then
  install_name_tool -change \
    "$CURRENT_DEPENDENCY" \
    "$TORCH_RELATIVE" \
    "$LIGHTGBM_DYLIB"
fi

ACTUAL_DEPENDENCY="$(otool -L "$LIGHTGBM_DYLIB" | awk '/libomp\.dylib/{print $1; exit}')"
if [[ "$ACTUAL_DEPENDENCY" != "$TORCH_RELATIVE" ]]; then
  echo "Failed to align LightGBM with PyTorch libomp: $ACTUAL_DEPENDENCY" >&2
  exit 1
fi

"$PYTHON" - <<'PY'
import numpy as np
import lightgbm as lgb
import torch

rng = np.random.default_rng(17)
x = rng.normal(size=(256, 16)).astype(np.float32)
y = rng.normal(size=256).astype(np.float32)
booster = lgb.train(
    {"objective": "regression", "verbosity": -1, "num_threads": 4},
    lgb.Dataset(x, label=y),
    num_boost_round=2,
)
prediction = booster.predict(x, num_threads=4)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
tensor = torch.as_tensor(x, device=device)
result = (tensor @ tensor.T).sum()
if device.type == "mps":
    torch.mps.synchronize()
if not np.isfinite(prediction).all() or not bool(torch.isfinite(result).item()):
    raise RuntimeError("OpenMP alignment smoke test produced non-finite output")
print(
    f"aligned: torch={torch.__version__} lightgbm={lgb.__version__} "
    f"device={device.type} rows={len(prediction)}"
)
PY
