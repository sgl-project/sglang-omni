#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Start the playground: single-process backend serving API + UI + file browser.
#
# Usage:
#   ./playground/web/start.sh --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
#   CUDA_VISIBLE_DEVICES=5 ./playground/web/start.sh --model-path <path>
#   ./playground/web/start.sh --model-path <path> --port 8080
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

PORT=8000
PLAYGROUND_PORT=7860

# Parse arguments: extract our flags, forward the rest to sglang-omni-server
BACKEND_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port)  PORT="$2"; shift 2 ;;
    --playground-port)   PLAYGROUND_PORT="$2"; shift 2 ;;
    --pipeline)  shift 2 ;;
    *)       BACKEND_ARGS+=("$1"); shift ;;
  esac
done

if [[ ${#BACKEND_ARGS[@]} -eq 0 ]]; then
  echo "Usage: $0 --model-id <model> [--port SERVER_PORT] [--playground-port PLAYGROUND_PORT]"
  echo ""
  echo "Example:"
  echo "  CUDA_VISIBLE_DEVICES=5 $0 --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct"
  exit 1
fi

API_BASE="http://localhost:${PORT}"

# Clean up background server on exit
cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  SGLang-Omni Playground"
echo "============================================================"
echo ""
echo "  Backend API: http://localhost:${PORT}"
echo "  Playground UI: http://localhost:${PLAYGROUND_PORT}"
echo ""
echo "============================================================"
echo ""

# 1. Start the backend server in the background
echo "[1/2] Starting backend server with arguments: ${BACKEND_ARGS[@]}"
"${PYTHON_BIN}" -m sglang_omni.cli.cli serve \
  "${BACKEND_ARGS[@]}" \
  --port "${PORT}" &
SERVER_PID=$!

API_BASE="http://localhost:${PORT}"

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
export SGLANG_OMNI_API_BASE=${API_BASE}
exec "${PYTHON_BIN}" "${SCRIPT_DIR}/app.py" --port "${PLAYGROUND_PORT}"
