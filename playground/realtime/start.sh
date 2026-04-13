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
#   ./playground/realtime/start.sh [--mock] [realtime-options] [serve-options...]
#   ./playground/realtime/start.sh --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct --gpu-talker 1 --gpu-code-predictor 1 --mem-fraction-static 0.9 --with-turn --turn-host IP_ADDRESS
# ---------------------------------------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python}"
BACKEND_ENTRY="${REPO_ROOT}/examples/run_qwen3_omni_speech_server.py"

BACKEND_PORT="${PORT:-8000}"
PLAYGROUND_PORT="7861"
MOCK_BACKEND="0"
MOCK_ARGS=()
ICE_URLS=()
ICE_USERNAME="${SGLANG_OMNI_ICE_USERNAME:-}"
ICE_CREDENTIAL="${SGLANG_OMNI_ICE_CREDENTIAL:-}"
TURN_ENABLED="0"
TURN_HOST="${TURN_HOST:-}"
TURN_PORT="${TURN_PORT:-3479}"
TURN_PORT_EXPLICIT="0"
TURN_ALT_PORT=""
TURN_USER="${TURN_USER:-realtime}"
TURN_PASSWORD="${TURN_PASSWORD:-realtime-demo}"
TURN_REALM="${TURN_REALM:-sglang-omni}"
TURN_PUBLIC_IP="${TURN_PUBLIC_IP:-}"
TURN_MIN_PORT="${TURN_MIN_PORT:-49160}"
TURN_MAX_PORT="${TURN_MAX_PORT:-49200}"
TURN_PID=""
TURN_USERDB_PATH=""
TURN_PIDFILE_PATH=""

BACKEND_ARGS=()

usage() {
  cat <<'EOF'
Usage:
  ./playground/realtime/start.sh [--mock] [realtime-options] [backend-options...]

Description:
  Launch the standalone realtime frontend plus either:
  - the mock realtime backend with --mock, or
  - the Qwen3 Omni speech server backend with forwarded backend-options.

Minimal usable commands:
  Local smoke test:
    ./playground/realtime/start.sh --mock

  Remote smoke test with TURN:
    ./playground/realtime/start.sh --mock --with-turn

  Real model:
    ./playground/realtime/start.sh --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct

Realtime options:
  --mock                     Use the mock backend instead of the Qwen3 Omni speech server.
  --port PORT                Backend API port. Default: 8000.
  --playground-port PORT     Frontend UI port. Default: 7861.
  --with-turn                Start a local coturn process for ICE relay.
  --turn-host HOST           Public or tailnet host/IP advertised to browsers.
                             Default: first Tailscale IPv4 if available.
  --turn-port PORT           TURN listener port. Default: auto from 3479 if free.
  --turn-user USER           TURN username. Default: realtime.
  --turn-password PASS       TURN password. Default: realtime-demo.
  --turn-realm REALM         TURN realm. Default: sglang-omni.
  --turn-public-ip IP        External IP for coturn when behind NAT.
  --turn-min-port PORT       TURN relay min port. Default: 49160.
  --turn-max-port PORT       TURN relay max port. Default: 49200.
  --ice-server URL           Extra ICE server URL. May be repeated.
  --ice-username USER        ICE username for the configured ICE servers.
  --ice-credential PASS      ICE credential for the configured ICE servers.

Mock-only options:
  --response-text TEXT
  --audio-mode MODE         Mock audio mode: tone or echo. Default: tone.
  --dump-audio-dir DIR      Save mock turn-context WAVs into DIR.
  --model-name NAME
  --sample-rate HZ
  --chunk-duration SECONDS
  --chunk-delay SECONDS
  --total-duration SECONDS
  --tone-frequency HZ

Backend options:
  Any unrecognized options are forwarded to:
    python examples/run_qwen3_omni_speech_server.py

  In normal backend mode you typically want:
    --model-path MODEL_OR_PATH

Examples:
  ./playground/realtime/start.sh --mock
  ./playground/realtime/start.sh --mock --with-turn
  ./playground/realtime/start.sh --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct
  ./playground/realtime/start.sh --port 8080 --playground-port 7861 --model-path /models/qwen-omni
EOF
}

detect_turn_host() {
  if command -v tailscale >/dev/null 2>&1; then
    local ts_ip
    ts_ip="$(tailscale ip -4 2>/dev/null | head -n1 | tr -d '[:space:]')"
    if [[ -n "${ts_ip}" ]]; then
      echo "${ts_ip}"
      return 0
    fi
  fi
  return 1
}

check_tcp_port() {
  "${PYTHON_BIN}" - "$1" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
sock.settimeout(0.5)
try:
    rc = sock.connect_ex(("127.0.0.1", port))
finally:
    sock.close()
raise SystemExit(0 if rc == 0 else 1)
PY
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

wait_for_tcp_port() {
  local port="$1"
  local label="$2"
  local attempts="${3:-120}"
  for i in $(seq 1 "${attempts}"); do
    if check_tcp_port "${port}"; then
      echo "${label} is ready on tcp:${port}."
      return 0
    fi
    sleep 1
  done
  echo "ERROR: ${label} did not become ready on tcp:${port}."
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)           usage; exit 0 ;;
    --mock)               MOCK_BACKEND="1"; shift ;;
    --with-turn)          TURN_ENABLED="1"; shift ;;
    --port)               BACKEND_PORT="$2"; shift 2 ;;
    --playground-port)    PLAYGROUND_PORT="$2"; shift 2 ;;
    --ice-server)         ICE_URLS+=("$2"); shift 2 ;;
    --ice-username)       ICE_USERNAME="$2"; shift 2 ;;
    --ice-credential)     ICE_CREDENTIAL="$2"; shift 2 ;;
    --turn-host)          TURN_HOST="$2"; shift 2 ;;
    --turn-port)          TURN_PORT="$2"; TURN_PORT_EXPLICIT="1"; shift 2 ;;
    --turn-user)          TURN_USER="$2"; shift 2 ;;
    --turn-password)      TURN_PASSWORD="$2"; shift 2 ;;
    --turn-realm)         TURN_REALM="$2"; shift 2 ;;
    --turn-public-ip)     TURN_PUBLIC_IP="$2"; shift 2 ;;
    --turn-min-port)      TURN_MIN_PORT="$2"; shift 2 ;;
    --turn-max-port)      TURN_MAX_PORT="$2"; shift 2 ;;
    --response-text|--audio-mode|--dump-audio-dir|--model-name|--sample-rate|--chunk-duration|--chunk-delay|--total-duration|--tone-frequency)
                         MOCK_ARGS+=("$1" "$2"); shift 2 ;;
    --pipeline)           shift 2 ;;
    *)                    BACKEND_ARGS+=("$1"); shift ;;
  esac
done

if [[ "${MOCK_BACKEND}" != "1" && ${#BACKEND_ARGS[@]} -eq 0 ]]; then
  usage
  exit 1
fi

if [[ "${TURN_ENABLED}" == "1" && -z "${TURN_HOST}" ]]; then
  if TURN_HOST="$(detect_turn_host)"; then
    echo "Using auto-detected TURN host ${TURN_HOST}."
  else
    echo "ERROR: --with-turn requires --turn-host HOST, or a working Tailscale IPv4 for auto-detection."
    exit 1
  fi
fi

if ! "${PYTHON_BIN}" -c "import socket; s=socket.socket(); s.bind(('0.0.0.0',${BACKEND_PORT})); s.close()" 2>/dev/null; then
  echo "WARNING: Port ${BACKEND_PORT} is already in use."
  BACKEND_PORT=$(pick_free_port)
  echo "Using port ${BACKEND_PORT} instead."
fi

if [[ "${TURN_ENABLED}" == "1" ]]; then
  if ! "${PYTHON_BIN}" -c "import socket; s=socket.socket(); s.bind(('0.0.0.0',${TURN_PORT})); s.close()" 2>/dev/null; then
    if [[ "${TURN_PORT_EXPLICIT}" == "1" ]]; then
      echo "ERROR: TURN port ${TURN_PORT} is already in use."
      exit 1
    fi
    echo "WARNING: TURN port ${TURN_PORT} is already in use."
    TURN_PORT=$(pick_free_port)
    echo "Using TURN port ${TURN_PORT} instead."
  fi
  TURN_ALT_PORT=$(pick_free_port)
  while [[ "${TURN_ALT_PORT}" == "${TURN_PORT}" ]]; do
    TURN_ALT_PORT=$(pick_free_port)
  done
fi

API_BASE="http://localhost:${BACKEND_PORT}"

cleanup() {
  if [[ -n "${SERVER_PID:-}" ]]; then
    kill "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  if [[ -n "${TURN_PID:-}" ]]; then
    kill "${TURN_PID}" 2>/dev/null || true
    wait "${TURN_PID}" 2>/dev/null || true
  fi
  if [[ -n "${TURN_USERDB_PATH:-}" ]]; then
    rm -f "${TURN_USERDB_PATH}" 2>/dev/null || true
  fi
  if [[ -n "${TURN_PIDFILE_PATH:-}" ]]; then
    rm -f "${TURN_PIDFILE_PATH}" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

if [[ "${TURN_ENABLED}" == "1" ]]; then
  ICE_URLS+=(
    "turn:${TURN_HOST}:${TURN_PORT}?transport=tcp"
    "turn:${TURN_HOST}:${TURN_PORT}?transport=udp"
  )
  TURN_USERDB_PATH="$(mktemp /tmp/sglang-omni-turn-XXXXXX.db)"
  TURN_PIDFILE_PATH="$(mktemp /tmp/sglang-omni-turn-XXXXXX.pid)"
  if [[ -z "${ICE_USERNAME}" ]]; then
    ICE_USERNAME="${TURN_USER}"
  fi
  if [[ -z "${ICE_CREDENTIAL}" ]]; then
    ICE_CREDENTIAL="${TURN_PASSWORD}"
  fi
fi

echo "============================================================"
echo "  SGLang-Omni Realtime Playground"
echo "============================================================"
echo ""
echo "  Backend API:   ${API_BASE}"
echo "  Frontend UI:   http://localhost:${PLAYGROUND_PORT}"
if [[ "${TURN_ENABLED}" == "1" ]]; then
  echo "  TURN relay:    turn:${TURN_HOST}:${TURN_PORT}"
fi
echo ""
echo "============================================================"
echo ""

if [[ ${#ICE_URLS[@]} -gt 0 ]]; then
  export SGLANG_OMNI_ICE_URLS="$(IFS=,; echo "${ICE_URLS[*]}")"
fi
if [[ -n "${ICE_USERNAME}" ]]; then
  export SGLANG_OMNI_ICE_USERNAME="${ICE_USERNAME}"
fi
if [[ -n "${ICE_CREDENTIAL}" ]]; then
  export SGLANG_OMNI_ICE_CREDENTIAL="${ICE_CREDENTIAL}"
fi

STEP_LABEL="1/2"
WAIT_LABEL="2/2"
if [[ "${TURN_ENABLED}" == "1" ]]; then
  STEP_LABEL="1/3"
  WAIT_LABEL="3/3"
  echo "[${STEP_LABEL}] Starting TURN relay..."
  if command -v turnserver >/dev/null 2>&1; then
    TURN_CMD=(
      turnserver
      -n
      --fingerprint
      --lt-cred-mech
      --user "${TURN_USER}:${TURN_PASSWORD}"
      --realm "${TURN_REALM}"
      --server-name "${TURN_REALM}"
      --userdb "${TURN_USERDB_PATH}"
      --pidfile "${TURN_PIDFILE_PATH}"
      --listening-port "${TURN_PORT}"
      --alt-listening-port "${TURN_ALT_PORT}"
      --listening-ip "0.0.0.0"
      --min-port "${TURN_MIN_PORT}"
      --max-port "${TURN_MAX_PORT}"
      --no-cli
      --no-tls
      --no-dtls
    )
    if [[ -n "${TURN_PUBLIC_IP}" ]]; then
      TURN_CMD+=(--external-ip "${TURN_PUBLIC_IP}")
    fi
    "${TURN_CMD[@]}" &
    TURN_PID=$!
  else
    echo "ERROR: --with-turn requires coturn's 'turnserver' binary to be installed."
    exit 1
  fi
  wait_for_tcp_port "${TURN_PORT}" "TURN relay" 180
  echo "[2/3] TURN relay is ready."
fi

if [[ "${MOCK_BACKEND}" == "1" ]]; then
  echo "[${STEP_LABEL/1/2}] Starting mock realtime API server..."
  "${PYTHON_BIN}" "${SCRIPT_DIR}/mock_server.py" \
    --port "${BACKEND_PORT}" \
    "${MOCK_ARGS[@]}" &
else
  echo "[${STEP_LABEL/1/2}] Starting backend server with arguments: ${BACKEND_ARGS[@]}"
  "${PYTHON_BIN}" "${BACKEND_ENTRY}" \
    "${BACKEND_ARGS[@]}" \
    --port "${BACKEND_PORT}" &
fi
SERVER_PID=$!

echo "[${WAIT_LABEL}] Waiting for server to be ready..."
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
if [[ ${#ICE_URLS[@]} -gt 0 ]]; then
  echo "  ICE servers:   ${ICE_URLS[*]}"
fi
if [[ "${TURN_ENABLED}" == "1" ]]; then
  echo "  TURN user:     ${TURN_USER}"
  echo "  Relay ports:   ${TURN_MIN_PORT}-${TURN_MAX_PORT}"
  if [[ -n "${TURN_PUBLIC_IP}" ]]; then
    echo "  External IP:   ${TURN_PUBLIC_IP}"
  fi
fi
echo ""
echo "============================================================"
echo ""
if [[ "${TURN_ENABLED}" == "1" ]]; then
  echo "  NOTE: Your browser must be able to reach ${TURN_HOST}:${TURN_PORT}"
  echo "  and the coturn relay port range ${TURN_MIN_PORT}-${TURN_MAX_PORT}."
  echo "  If you are forwarding from Windows into WSL, forward both the TURN"
  echo "  listener port and that relay range."
  echo ""
fi

export SGLANG_OMNI_API_BASE="${API_BASE}"
"${PYTHON_BIN}" "${SCRIPT_DIR}/app.py" \
  --api-base "${API_BASE}" \
  --port "${PLAYGROUND_PORT}"
