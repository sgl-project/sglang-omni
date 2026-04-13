#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND_ENTRY="${REPO_ROOT}/examples/run_qwen3_omni_speech_server.py"

BACKEND_PORT="${PORT:-8000}"
PLAYGROUND_PORT="7862"
MOCK_BACKEND="0"
MOCK_ARGS=()
BACKEND_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./playground/realtime-ws/start.sh [--mock] [realtime-options] [backend-options...]

Description:
  Launch the standalone websocket realtime frontend plus either:
  - the mock websocket backend with --mock, or
  - the Qwen3 Omni speech server backend with forwarded backend-options.

Realtime options:
  --mock                     Use the mock backend instead of the Qwen3 Omni speech server.
  --port PORT                Backend API port. Default: 8000.
  --playground-port PORT     Frontend UI port. Default: 7862.

Mock-only options:
  --response-text TEXT
  --audio-mode MODE
  --dump-audio-dir DIR
  --model-name NAME
  --sample-rate HZ
  --chunk-duration SECONDS
  --chunk-delay SECONDS
  --total-duration SECONDS
  --tone-frequency HZ

Backend options:
  Any unrecognized options are forwarded to:
    python examples/run_qwen3_omni_speech_server.py

Examples:
  ./playground/realtime-ws/start.sh --mock
  ./playground/realtime-ws/start.sh --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
EOF
}

pick_free_port() {
  "${PYTHON_BIN}" - <<'PY'
import socket

s = socket.socket()
s.bind(("", 0))
print(s.getsockname()[1])
s.close()
PY
}

require_websocket_runtime() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys

if importlib.util.find_spec("websockets") or importlib.util.find_spec("wsproto"):
    raise SystemExit(0)
print(
    "ERROR: WebSocket runtime support is missing. Install the realtime extra with\n"
    "  uv pip install -e \".[realtime]\"\n"
    "or install one of:\n"
    "  pip install websockets\n"
    "  pip install wsproto"
)
raise SystemExit(1)
PY
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)           usage; exit 0 ;;
    --mock)              MOCK_BACKEND="1"; shift ;;
    --port)              BACKEND_PORT="$2"; shift 2 ;;
    --playground-port)   PLAYGROUND_PORT="$2"; shift 2 ;;
    --response-text|--audio-mode|--dump-audio-dir|--model-name|--sample-rate|--chunk-duration|--chunk-delay|--total-duration|--tone-frequency)
                         MOCK_ARGS+=("$1" "$2"); shift 2 ;;
    --pipeline)          shift 2 ;;
    *)                   BACKEND_ARGS+=("$1"); shift ;;
  esac
done

if [[ "${MOCK_BACKEND}" != "1" && ${#BACKEND_ARGS[@]} -eq 0 ]]; then
  usage
  exit 1
fi

if ! "${PYTHON_BIN}" -c "import socket; s=socket.socket(); s.bind(('0.0.0.0',${BACKEND_PORT})); s.close()" 2>/dev/null; then
  echo "WARNING: Port ${BACKEND_PORT} is already in use."
  BACKEND_PORT=$(pick_free_port)
  echo "Using port ${BACKEND_PORT} instead."
fi

API_BASE="http://localhost:${BACKEND_PORT}"
if ! require_websocket_runtime; then
  exit 1
fi

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "============================================================"
echo "  SGLang-Omni Realtime WebSocket Playground"
echo "============================================================"
echo ""
echo "  Backend API:   ${API_BASE}"
echo "  Frontend UI:   http://localhost:${PLAYGROUND_PORT}"
echo ""
echo "============================================================"
echo ""

if [[ "${MOCK_BACKEND}" == "1" ]]; then
  echo "[1/2] Starting mock websocket realtime API server..."
  "${PYTHON_BIN}" "${SCRIPT_DIR}/mock_server.py" \
    --port "${BACKEND_PORT}" \
    "${MOCK_ARGS[@]}" &
else
  echo "[1/2] Starting backend server with arguments: ${BACKEND_ARGS[*]}"
  "${PYTHON_BIN}" "${BACKEND_ENTRY}" \
    "${BACKEND_ARGS[@]}" \
    --port "${BACKEND_PORT}" &
fi
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
"${PYTHON_BIN}" "${SCRIPT_DIR}/app.py" \
  --api-base "${API_BASE}" \
  --port "${PLAYGROUND_PORT}"
