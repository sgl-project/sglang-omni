#!/bin/bash
# One acceptance arm: start one server arm, wait for readiness, run the fixed-
# order eval, then stop that server.
#
# usage: ab_arm.sh <base|opt> <group> <concurrency> <limit> <run_id>
#
# Required environment:
#   BENCH_DIR       directory with en_manifest.json and run_eval.py
#   LAUNCHER_BASE   server launcher script for the baseline arm
#   LAUNCHER_OPT    server launcher script for the optimized arm
#
# Optional environment:
#   PYTHON          interpreter that runs run_eval.py   (default: python3)
#   BENCH_WRAPPER   wrapper prefix for that interpreter (default: none)
#   BASE_URL        server base URL          (default: http://127.0.0.1:8000)
#   MANIFEST        eval manifest            (default: $BENCH_DIR/en_manifest.json)
#   LOG_DIR         server/eval logs         (default: $BENCH_DIR/logs)
#   RESULTS_DIR     summaries and WAVs       (default: $BENCH_DIR/results)
#   START_TIMEOUT   seconds to wait for ready (default: 2400)
#   DEVICE_INDEX    NPU index for npu-smi polling (default: 0)
#   NPU_SMI         npu-smi binary           (default: npu-smi)
#   NPU_SMI_WRAPPER wrapper prefix for npu-smi (default: none)
#   STOP_TIMEOUT    seconds to wait for a clean stop (default: 60)
#
# The script only ever signals the process group it started. It refuses to
# start when the base URL is already serving, and never matches or kills other
# user processes.
set -uo pipefail

ARM=${1:?arm (base|opt)}
GROUP=${2:?group}
CONC=${3:?concurrency}
LIMIT=${4:?limit}
RUN_ID=${5:?run_id}

: "${BENCH_DIR:?set BENCH_DIR to the directory containing run_eval.py}"
: "${LAUNCHER_BASE:?set LAUNCHER_BASE to the baseline server launcher}"
: "${LAUNCHER_OPT:?set LAUNCHER_OPT to the optimized server launcher}"
PYTHON=${PYTHON:-python3}
BENCH_WRAPPER=${BENCH_WRAPPER:-}
BASE_URL=${BASE_URL:-http://127.0.0.1:8000}
MANIFEST=${MANIFEST:-$BENCH_DIR/en_manifest.json}
LOG_DIR=${LOG_DIR:-$BENCH_DIR/logs}
RESULTS_DIR=${RESULTS_DIR:-$BENCH_DIR/results}
START_TIMEOUT=${START_TIMEOUT:-2400}
DEVICE_INDEX=${DEVICE_INDEX:-0}
NPU_SMI=${NPU_SMI:-npu-smi}
NPU_SMI_WRAPPER=${NPU_SMI_WRAPPER:-}
STOP_TIMEOUT=${STOP_TIMEOUT:-60}

case "$ARM" in
  base) LAUNCHER=$LAUNCHER_BASE ;;
  opt) LAUNCHER=$LAUNCHER_OPT ;;
  *) echo "bad arm: $ARM (expected base|opt)" >&2; exit 2 ;;
esac
[ -x "$LAUNCHER" ] || { echo "launcher is not executable: $LAUNCHER" >&2; exit 2; }
[ -f "$MANIFEST" ] || { echo "manifest not found: $MANIFEST" >&2; exit 2; }

mkdir -p "$LOG_DIR" "$RESULTS_DIR"
SRVLOG=$LOG_DIR/server_${ARM}_g${GROUP}_c${CONC}.log
SRV_PID=""
RC=0

step() { echo "[$(date +%H:%M:%S)] $*"; }

stop_server() {
  [ -n "$SRV_PID" ] || return 0
  kill -0 "$SRV_PID" 2>/dev/null || { SRV_PID=""; return 0; }
  step "stopping server process group $SRV_PID"
  kill -TERM -- "-$SRV_PID" 2>/dev/null || kill -TERM "$SRV_PID" 2>/dev/null
  for _ in $(seq 1 "$STOP_TIMEOUT"); do
    kill -0 "$SRV_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$SRV_PID" 2>/dev/null; then
    step "server did not stop; sending SIGKILL to process group $SRV_PID"
    kill -KILL -- "-$SRV_PID" 2>/dev/null || kill -KILL "$SRV_PID" 2>/dev/null
  fi
  wait "$SRV_PID" 2>/dev/null
  SRV_PID=""
}

on_exit() {
  local status=$?
  stop_server
  exit "$status"
}
trap on_exit EXIT INT TERM HUP

if curl -sf -m 5 "$BASE_URL/v1/models" >/dev/null 2>&1; then
  echo "a server is already serving $BASE_URL; refusing to start another" >&2
  echo "stop it yourself, or point BASE_URL at a free port" >&2
  exit 3
fi

step "starting $ARM server (log $SRVLOG)"
if command -v setsid >/dev/null 2>&1; then
  setsid bash "$LAUNCHER" >"$SRVLOG" 2>&1 &
else
  bash "$LAUNCHER" >"$SRVLOG" 2>&1 &
fi
SRV_PID=$!

READY=0
for _ in $(seq 1 "$START_TIMEOUT"); do
  if ! kill -0 "$SRV_PID" 2>/dev/null; then
    step "server exited during startup"
    tail -n 40 "$SRVLOG"
    exit 4
  fi
  if grep -qE "Stage process pipeline failed|crashed|Traceback \(most recent call last\)" "$SRVLOG"; then
    step "startup failure detected"
    tail -n 40 "$SRVLOG"
    exit 4
  fi
  if curl -sf -m 5 "$BASE_URL/v1/models" >/dev/null 2>&1; then
    READY=1
    break
  fi
  sleep 1
done
if [ "$READY" != 1 ]; then
  step "server not ready after ${START_TIMEOUT}s"
  tail -n 40 "$SRVLOG"
  exit 5
fi
step "server ready"

PY_CMD=()
if [ -n "$BENCH_WRAPPER" ]; then PY_CMD+=("$BENCH_WRAPPER"); fi
PY_CMD+=("$PYTHON")

MODEL=$("${PY_CMD[@]}" -c \
  'import json,sys,urllib.request; print(json.load(urllib.request.urlopen(sys.argv[1]))["data"][0]["id"])' \
  "$BASE_URL/v1/models")
step "served model: $MODEL"
grep -E "Captured MOSS-TTS|graph" "$SRVLOG" | tail -n 40 >>"$RESULTS_DIR/${RUN_ID}_graphcov.txt" || true

OUT=$RESULTS_DIR/$RUN_ID
EVAL_ARGS=(
  --base-url "$BASE_URL"
  --manifest "$MANIFEST"
  --out-dir "$OUT"
  --model "$MODEL"
  --seed 1234
  --concurrency "$CONC"
  --warmup 8
  --limit "$LIMIT"
  --run-id "$RUN_ID"
  --device-index "$DEVICE_INDEX"
  --npu-smi "$NPU_SMI"
)
if [ -n "$NPU_SMI_WRAPPER" ]; then
  EVAL_ARGS+=(--npu-smi-wrapper "$NPU_SMI_WRAPPER")
fi

step "running eval -> $OUT"
"${PY_CMD[@]}" "$BENCH_DIR/run_eval.py" "${EVAL_ARGS[@]}"
RC=$?
step "eval exit=$RC"
exit "$RC"
