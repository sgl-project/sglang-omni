#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Start the S2-Pro TTS playground: launches the backend, waits for health,
# then starts the Gradio UI.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./playground/tts/start.sh
#   ./playground/tts/start.sh --port 8080 --gradio-port 7861 --share
#   ./playground/tts/start.sh --model-path /path/to/s2-pro
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

BACKEND_PORT="${PORT:-8000}"
GRADIO_PORT="7899"
GRADIO_SHARE=""
MODEL_PATH="${S2PRO_CKPT:-${MODEL_PATH:-fishaudio/s2-pro}}"
CONFIG_PATH="${REPO_DIR}/examples/configs/s2pro_tts.yaml"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)    MODEL_PATH="$2"; shift 2 ;;
    --port)          BACKEND_PORT="$2"; shift 2 ;;
    --gradio-port)   GRADIO_PORT="$2"; shift 2 ;;
    --share)         GRADIO_SHARE="--share"; shift ;;
    --config)        CONFIG_PATH="$2"; shift 2 ;;
    *)               echo "Unknown arg: $1"; exit 1 ;;
  esac
done

# Check port availability before loading models (try actual bind)
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
echo "  S2-Pro TTS Playground"
echo "============================================================"
echo ""
echo "  Model:       ${MODEL_PATH}"
echo "  Backend API: ${API_BASE}"
echo "  Gradio UI:   http://localhost:${GRADIO_PORT}"
echo ""
echo "============================================================"

# 1. Start backend
echo "[1/2] Starting S2-Pro server..."
"${PYTHON_BIN}" -m sglang_omni.cli.cli serve \
  --model-path "${MODEL_PATH}" \
  --config "${CONFIG_PATH}" \
  --port "${BACKEND_PORT}" &
SERVER_PID=$!

# 2. Wait for health
echo "[2/2] Waiting for server..."
for i in $(seq 1 120); do
  if ! kill -0 "${SERVER_PID}" 2>/dev/null; then
    echo "ERROR: Server exited unexpectedly."
    exit 1
  fi
  if curl -s "${API_BASE}/health" 2>/dev/null | grep -q "ok\|healthy\|true"; then
    echo "Server is ready."
    break
  fi
  if [[ $i -eq 120 ]]; then
    echo "ERROR: Server did not start within 600s."
    exit 1
  fi
  sleep 5
done

# 3. Launch Gradio
echo ""
echo "============================================================"
echo "  Server is ready!"
echo "============================================================"
echo ""
echo "  Gradio UI:   http://localhost:${GRADIO_PORT}"
echo "  Backend API: ${API_BASE}"
echo ""
echo "  To access from your local machine, run:"
echo "    ssh -L ${GRADIO_PORT}:localhost:${GRADIO_PORT} <user>@$(hostname)"
echo ""
echo "  Then open http://localhost:${GRADIO_PORT} in your browser."
echo "============================================================"
echo ""

exec "${PYTHON_BIN}" "${SCRIPT_DIR}/app.py" \
  --api-base "${API_BASE}" \
  --port "${GRADIO_PORT}" \
  ${GRADIO_SHARE}
