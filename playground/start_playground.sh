#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Start the playground: backend API server + frontend + filesystem API.
#
# Usage:
#   ./playground/start_playground.sh --pipeline qwen3-omni --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct
#   CUDA_VISIBLE_DEVICES=5 ./playground/start_playground.sh --pipeline qwen3-omni --model-id <id>
#   ./playground/start_playground.sh --pipeline qwen3-omni --model-id <id> --backend-port 8080
# ---------------------------------------------------------------------------

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FRONTEND_PORT="${FRONTEND_PORT:-3000}"
FS_PORT="${FS_PORT:-8001}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FS_HOST="${FS_HOST:-0.0.0.0}"
PYTHON_BIN="${PYTHON_BIN:-python}"

# Parse arguments: extract our flags, forward the rest to sglang-omni-server
BACKEND_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --frontend-port)  FRONTEND_PORT="$2"; shift 2 ;;
    --backend-port)   BACKEND_PORT="$2"; shift 2 ;;
    --fs-port)        FS_PORT="$2"; shift 2 ;;
    --port)           BACKEND_PORT="$2"; shift 2 ;;
    *)                BACKEND_ARGS+=("$1"); shift ;;
  esac
done

if [[ ${#BACKEND_ARGS[@]} -eq 0 ]]; then
  echo "Usage: $0 --pipeline <name> --model-id <model> [--backend-port PORT] [--frontend-port PORT]"
  echo ""
  echo "Example:"
  echo "  CUDA_VISIBLE_DEVICES=5 $0 --pipeline qwen3-omni --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct"
  exit 1
fi

cleanup() {
  local exit_code=$?
  echo ""
  echo "[stop] shutting down..."
  if [[ -n "${BACKEND_PID:-}" ]]; then
    kill "${BACKEND_PID}" 2>/dev/null || true
  fi
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

echo "============================================================"
echo "  SGLang-Omni Playground"
echo "============================================================"
echo ""
echo "  Frontend :  http://localhost:${FRONTEND_PORT}"
echo "  FS API   :  http://localhost:${FS_PORT}"
echo "  Backend  :  http://localhost:${BACKEND_PORT}"
echo ""
echo "============================================================"
echo ""

# Start frontend static file server
"${PYTHON_BIN}" -m http.server "${FRONTEND_PORT}" \
  --directory "${ROOT_DIR}/playground" &
FRONTEND_PID=$!
echo "[playground] frontend started  (pid ${FRONTEND_PID})"

# Start filesystem API
"${PYTHON_BIN}" -m playground.fs_api \
  --host "${FS_HOST}" --port "${FS_PORT}" &
FS_PID=$!
echo "[playground] fs api started    (pid ${FS_PID})"

# Start backend API server (model loading — this takes a while)
echo "[playground] starting backend (model loading may take a few minutes)..."
"${PYTHON_BIN}" -m sglang_omni.serve.launcher \
  "${BACKEND_ARGS[@]}" \
  --port "${BACKEND_PORT}" &
BACKEND_PID=$!

echo ""
echo "[playground] all processes launched. Ctrl+C to stop."
echo ""

# Wait for any process to exit
wait -n "${BACKEND_PID}" "${FRONTEND_PID}" "${FS_PID}"
