#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="${ROOT:-$HOME/work/stock-v2}"
PYTHON_BIN="${PYTHON_BIN:-/Library/Developer/CommandLineTools/Library/Frameworks/Python3.framework/Versions/3.9/bin/python3}"
VENV="${QLIB_VENV:-$ROOT/.venv-qlib}"
MICROMAMBA="$HOME/.local/bin/micromamba"
OPENMP_PREFIX="$HOME/.local/opt/llvm-openmp"

cd "$ROOT"
if [[ ! -x "$VENV/bin/python" ]]; then
  "$PYTHON_BIN" -m venv "$VENV"
fi
"$VENV/bin/python" -m pip install --upgrade pip setuptools wheel
"$VENV/bin/python" -m pip install -r requirements-qlib.txt

if ! "$VENV/bin/python" -c 'import lightgbm' >/dev/null 2>&1; then
  mkdir -p "$(dirname "$MICROMAMBA")" "$HOME/.local/opt"
  if [[ ! -x "$MICROMAMBA" ]]; then
    archive="$(mktemp -t micromamba).tar.bz2"
    extract_dir="$(mktemp -d -t micromamba)"
    curl -L --fail --silent --show-error \
      https://micro.mamba.pm/api/micromamba/osx-arm64/latest \
      -o "$archive"
    tar -xjf "$archive" -C "$extract_dir" bin/micromamba
    install -m 755 "$extract_dir/bin/micromamba" "$MICROMAMBA"
    rm -rf "$archive" "$extract_dir"
  fi
  "$MICROMAMBA" create -y -p "$OPENMP_PREFIX" -c conda-forge llvm-openmp=22.1.8
  lgb_dir="$VENV/lib/python3.9/site-packages/lightgbm/lib"
  cp "$OPENMP_PREFIX/lib/libomp.dylib" "$lgb_dir/libomp.dylib"
  if otool -L "$lgb_dir/lib_lightgbm.dylib" | grep -q '@rpath/libomp.dylib'; then
    install_name_tool -change \
      @rpath/libomp.dylib \
      @loader_path/libomp.dylib \
      "$lgb_dir/lib_lightgbm.dylib"
  fi
fi

"$VENV/bin/python" - <<'PY'
import lightgbm
import qlib

print(f"Qlib {qlib.__version__}; LightGBM {lightgbm.__version__}")
PY
