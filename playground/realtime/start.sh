#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Start the realtime playground: launches the backend server, waits for health,
# then starts the standalone frontend app.
#
# Prerequisites:
#   uv pip install -e ".[realtime]"
#
# Usage:
#   ./playground/realtime/start.sh --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
#   ./playground/realtime/start.sh --model-path <path> --port 8080 --playground-port 7861
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

BACKEND_PORT="${PORT:-8000}"
PLAYGROUND_PORT="7861"

BACKEND_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)               BACKEND_PORT="$2"; shift 2 ;;
    --playground-port)    PLAYGROUND_PORT="$2"; shift 2 ;;
    --pipeline)           shift 2 ;;
    *)                    BACKEND_ARGS+=("$1"); shift ;;
  esac
done

if [[ ${#BACKEND_ARGS[@]} -eq 0 ]]; then
  echo "Usage: $0 --model-path <model> [--port PORT] [--playground-port PORT]"
  echo ""
  echo "Example:"
  echo "  CUDA_VISIBLE_DEVICES=0 $0 --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct"
  exit 1
fi

if ! python -c "import socket; s=socket.socket(); s.bind(('0.0.0.0',${BACKEND_PORT})); s.close()" 2>/dev/null; then
  echo "WARNING: Port ${BACKEND_PORT} is already in use."
  BACKEND_PORT=$(python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()")
  echo "Using port ${BACKEND_PORT} instead."
fi

API_BASE="http://localhost:${BACKEND_PORT}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  SGLang-Omni Realtime Playground"
echo "============================================================"
echo ""
echo "  Backend API:   ${API_BASE}"
echo "  Frontend UI:   http://localhost:${PLAYGROUND_PORT}"
echo ""
echo "============================================================"
echo ""

echo "[1/2] Starting backend server with arguments: ${BACKEND_ARGS[@]}"
"${PYTHON_BIN}" -m sglang_omni.cli.cli serve \
  "${BACKEND_ARGS[@]}" \
  --port "${BACKEND_PORT}" &
SERVER_PID=$!

echo "[2/2] Waiting for server to be ready..."
for i in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: Backend server exited unexpectedly."
    exit 1
  fi
  if curl -s "${API_BASE}/health" 2>/dev/null | grep -q "healthy"; then
    echo "Server is ready."
    break
  fi
  if [[ $i -eq 120 ]]; then
    echo "ERROR: Server did not become healthy within 600s."
    exit 1
  fi
  sleep 5
done

echo ""
echo "============================================================"
echo "  Server is ready!"
echo "============================================================"
echo ""
echo "  Frontend UI:   http://localhost:${PLAYGROUND_PORT}"
echo "  Backend API:   ${API_BASE}"
echo ""
echo "============================================================"
echo ""

export SGLANG_OMNI_API_BASE="${API_BASE}"
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/app.py" \
  --api-base "${API_BASE}" \
  --port "${PLAYGROUND_PORT}"
