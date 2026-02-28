#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Start the Gradio playground: launches the backend server, waits for it to
# become healthy, then starts the Gradio UI.  One command, one terminal.
#
# Prerequisites:
#   pip install "sglang-omni[gradio]"   # or: pip install gradio httpx
#
# Usage:
#   ./playground/gradio/start.sh --pipeline qwen3-omni --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct
#   CUDA_VISIBLE_DEVICES=0 ./playground/gradio/start.sh --pipeline qwen3-omni --model-id <id>
#   ./playground/gradio/start.sh --pipeline qwen3-omni --model-id <id> --port 8080 --gradio-port 7861
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

BACKEND_PORT="${PORT:-8000}"
GRADIO_PORT="7860"
GRADIO_SHARE=""

# Parse arguments: split into backend args vs gradio-specific flags
BACKEND_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)          BACKEND_PORT="$2"; shift 2 ;;
    --gradio-port)   GRADIO_PORT="$2"; shift 2 ;;
    --share)         GRADIO_SHARE="--share"; shift ;;
    *)               BACKEND_ARGS+=("$1"); shift ;;
  esac
done

if [[ ${#BACKEND_ARGS[@]} -eq 0 ]]; then
  echo "Usage: $0 --pipeline <name> --model-id <model> [--port PORT] [--gradio-port PORT] [--share]"
  echo ""
  echo "Example:"
  echo "  CUDA_VISIBLE_DEVICES=0 $0 --pipeline qwen3-omni --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct"
  exit 1
fi

API_BASE="http://localhost:${BACKEND_PORT}"

# Clean up background server on exit
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  SGLang-Omni Gradio Playground"
echo "============================================================"
echo ""
echo "  Backend API:  ${API_BASE}"
echo "  Gradio UI:    http://localhost:${GRADIO_PORT}"
echo ""
echo "============================================================"
echo ""

# 1. Start the backend server in the background
echo "[1/2] Starting backend server with arguments: ${BACKEND_ARGS[@]}"
"${PYTHON_BIN}" -m sglang_omni.cli.cli serve \
  "${BACKEND_ARGS[@]}" \
  --port "${BACKEND_PORT}" &
SERVER_PID=$!

# 2. Wait for the server to become healthy
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

# 3. Launch the Gradio UI (foreground — Ctrl-C stops everything via trap)
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/app.py" \
  --api-base "${API_BASE}" \
  --port "${GRADIO_PORT}" \
  ${GRADIO_SHARE}
