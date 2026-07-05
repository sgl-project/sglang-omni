#!/usr/bin/env bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Start the Higgs Audio v3 TTS playground: launches the sgl-omni backend,
# waits for health, then starts the static HTML/CSS/JS UI.
#
# Usage:
#   CUDA_VISIBLE_DEVICES=0 ./playground/higgs/start.sh
#   ./playground/higgs/start.sh --port 8080 --playground-port 7861
#   ./playground/higgs/start.sh --model-path /path/to/higgs-tts
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"

BACKEND_PORT="${PORT:-8000}"
PLAYGROUND_PORT="7860"
MODEL_PATH="${HIGGS_CKPT:-${MODEL_PATH:-bosonai/higgs-tts-3-4b}}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model-path)        MODEL_PATH="$2"; shift 2 ;;
    --port)              BACKEND_PORT="$2"; shift 2 ;;
    --playground-port)   PLAYGROUND_PORT="$2"; shift 2 ;;
    *)                   echo "Unknown arg: $1"; exit 1 ;;
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
echo "  Higgs Audio v3 TTS Playground"
echo "============================================================"
echo ""
echo "  Model:         ${MODEL_PATH}"
echo "  Backend API:   ${API_BASE}"
echo "  Playground UI: http://localhost:${PLAYGROUND_PORT}"
echo ""
echo "============================================================"

echo "[1/2] Starting Higgs backend..."
"${PYTHON_BIN}" -m sglang_omni.cli serve \
  --model-path "${MODEL_PATH}" \
  --port "${BACKEND_PORT}" &
SERVER_PID=$!

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

echo ""
echo "============================================================"
echo "  Server is ready!"
echo "============================================================"
echo ""
echo "  Playground UI: http://localhost:${PLAYGROUND_PORT}"
echo "  Backend API:   ${API_BASE}"
echo ""
echo "  To access from your local machine, run:"
echo "    ssh -L ${PLAYGROUND_PORT}:localhost:${PLAYGROUND_PORT} <user>@$(hostname)"
echo ""
echo "  Then open http://localhost:${PLAYGROUND_PORT} in your browser."
echo "============================================================"
echo ""

cd "${REPO_DIR}"

exec "${PYTHON_BIN}" -m playground.higgs.app \
  --api-base "${API_BASE}" \
  --port "${PLAYGROUND_PORT}"
