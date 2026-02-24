#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FRONTEND_PORT="${FRONTEND_PORT:-3000}"
FS_PORT="${FS_PORT:-8001}"
FS_HOST="${FS_HOST:-0.0.0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if [[ "${1:-}" != "" ]]; then
  FRONTEND_PORT="$1"
fi
if [[ "${2:-}" != "" ]]; then
  FS_PORT="$2"
fi

cleanup() {
  local exit_code=$?
  if [[ -n "${FRONTEND_PID:-}" ]]; then
    kill "${FRONTEND_PID}" 2>/dev/null || true
  fi
  if [[ -n "${FS_PID:-}" ]]; then
    kill "${FS_PID}" 2>/dev/null || true
  fi
  wait 2>/dev/null || true
  exit "${exit_code}"
}

trap cleanup INT TERM EXIT

echo "[start] frontend: http://localhost:${FRONTEND_PORT}"
echo "[start] fs api:   http://localhost:${FS_PORT}"

"${PYTHON_BIN}" -m http.server "${FRONTEND_PORT}" --directory "${ROOT_DIR}/playground" &
FRONTEND_PID=$!

"${PYTHON_BIN}" -m playground.fs_api --host "${FS_HOST}" --port "${FS_PORT}" &
FS_PID=$!

wait -n "${FRONTEND_PID}" "${FS_PID}"
