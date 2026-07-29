#!/usr/bin/env bash
set -Eeuo pipefail

HOST="${QWEN_HOST:-127.0.0.1}"
PORT="${QWEN_PORT:-18081}"

if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "Qwen server: stopped"
  sysctl vm.swapusage
  exit 1
fi

echo "Qwen server: listening at http://$HOST:$PORT/v1"
curl --fail --silent --show-error "http://$HOST:$PORT/api/status"
echo
sysctl vm.swapusage
