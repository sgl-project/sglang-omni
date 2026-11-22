#!/bin/bash
# One acceptance arm: (re)start one server arm, wait for readiness, run the
# fixed-order SeedTTS eval, then stop the server.
# usage: ab_arm.sh <base|opt> <group> <concurrency> <limit> <run_id>
set -u
ARM=${1:?arm}
GROUP=${2:?group}
CONC=${3:?concurrency}
LIMIT=${4:?limit}
RUN_ID=${5:?run_id}
BENCH=/root/work/bench
LOGD=$BENCH/logs
RESD=$BENCH/results
mkdir -p "$LOGD" "$RESD"
case "$ARM" in
  base) LAUNCHER=$BENCH/launch_server_base.sh ;;
  opt) LAUNCHER=$BENCH/launch_server.sh ;;
  *) echo "bad arm $ARM"; exit 2 ;;
esac
SRVLOG=$LOGD/server_${ARM}_g${GROUP}_c${CONC}.log

step() { echo "[$(date +%H:%M:%S)] $*"; }

step "stopping any running server"
pkill -f "[s]glang_omni.cli serve" || true
for _ in $(seq 1 60); do
  pgrep -f "[s]glang_omni.cli serve" >/dev/null || break
  sleep 2
done
pgrep -f "[s]glang_omni.cli serve" >/dev/null && { step "server still running"; exit 3; }

step "starting $ARM server (log $SRVLOG)"
nohup bash "$LAUNCHER" >"$SRVLOG" 2>&1 &
SRV_PID=$!

READY=0
for _ in $(seq 1 2400); do
  if grep -qE "Stage process pipeline failed|crashed|Traceback \(most recent call last\)" "$SRVLOG"; then
    step "startup failure detected"; tail -n 40 "$SRVLOG"; kill $SRV_PID 2>/dev/null; pkill -f "[s]glang_omni.cli serve" || true; exit 4
  fi
  if curl -sf -m 5 http://127.0.0.1:8000/v1/models >/dev/null 2>&1; then READY=1; break; fi
  sleep 1
done
if [ "$READY" != 1 ]; then step "not ready in 2400s"; tail -n 40 "$SRVLOG"; kill $SRV_PID 2>/dev/null; pkill -f "[s]glang_omni.cli serve" || true; exit 5; fi
step "server ready"
PY=/tmp/e31-stack/container-swr/rootfs/usr/local/python3.12.13/bin/python3.12
MODEL=$(curl -sf http://127.0.0.1:8000/v1/models | /tmp/e31-stack/native "$PY" -c 'import json,sys;print(json.load(sys.stdin)["data"][0]["id"])')
step "served model: $MODEL"
grep -E "Captured MOSS-TTS|graph" "$SRVLOG" | tail -n 40 >> "$RESD/${RUN_ID}_graphcov.txt" || true

OUT=$RESD/$RUN_ID
step "running eval -> $OUT"
/tmp/e31-stack/native "$PY" "$BENCH/run_eval.py" \
  --base-url http://127.0.0.1:8000 \
  --manifest "$BENCH/en_manifest.json" \
  --out-dir "$OUT" \
  --model "$MODEL" \
  --seed 1234 \
  --concurrency "$CONC" \
  --warmup 8 \
  --limit "$LIMIT" \
  --run-id "$RUN_ID"
RC=$?
step "eval exit=$RC"

step "stopping server"
pkill -f "[s]glang_omni.cli serve" || true
sleep 5
exit $RC
