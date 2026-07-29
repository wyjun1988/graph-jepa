#!/usr/bin/env bash
set -Eeuo pipefail

PORT="${QWEN_PORT:-18081}"
pids="$(lsof -tiTCP:"$PORT" -sTCP:LISTEN 2>/dev/null || true)"

if [[ -z "$pids" ]]; then
  echo "Qwen server is not listening on port $PORT."
  exit 0
fi

for pid in $pids; do
  command_line="$(ps -p "$pid" -o command= 2>/dev/null || true)"
  if [[ "$command_line" != *omlx-server* && "$command_line" != *"omlx serve"* ]]; then
    echo "Refusing to stop unrelated process on port $PORT: $pid $command_line" >&2
    exit 1
  fi
  kill -TERM "$pid"
done

for _ in {1..30}; do
  if ! lsof -nP -iTCP:"$PORT" -sTCP:LISTEN >/dev/null 2>&1; then
    echo "Qwen server stopped."
    exit 0
  fi
  sleep 1
done

echo "Qwen server did not stop within 30 seconds." >&2
exit 1
