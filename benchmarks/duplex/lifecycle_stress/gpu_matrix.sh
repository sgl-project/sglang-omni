#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Every lifecycle stress cell against one turn-based model server on this host, in one go.
#
#   MODEL=... MAX_SESSIONS=... SERVER_PID=... SERVER_LOG=... PAIR=... \
#     bash benchmarks/duplex/lifecycle_stress/gpu_matrix.sh OUTPUT_DIRECTORY
#
# PAIR is a directory with input.wav, in which the user talks over the reply, and
# clean_input.wav, in which nobody does, such as one Full-Duplex-Bench v1.5
# user_interruption pair. Each cell first runs one cycle of two clients. A cell
# that fails there is skipped; the others run CYCLES cycles (10 by default) at
# MAX_SESSIONS clients. No failure stops the script, and a rerun with the same
# OUTPUT_DIRECTORY skips what is already there.
set -u
OUT=$1
: "${MODEL:?}" "${MAX_SESSIONS:?}" "${SERVER_PID:?}" "${SERVER_LOG:?}" "${PAIR:?}"
URL=${URL:-ws://127.0.0.1:8000/v1/realtime}
CYCLES=${CYCLES:-10}
PYTHON=${PYTHON:-python}
# DEVICE_MEMORY=0 leaves out the nvidia-smi reading, for a host without one.
DEVICE_MEMORY=${DEVICE_MEMORY:-1}
REVISION=$(git rev-parse HEAD)
OVERFLOW=$((MAX_SESSIONS + (MAX_SESSIONS + 3) / 4))
mkdir -p "$OUT"

stress() {
  directory=$1
  shift
  device_flag=()
  [ "$DEVICE_MEMORY" = 1 ] && device_flag=(--device-memory)
  "$PYTHON" -m benchmarks.duplex.lifecycle_stress run --url "$URL" \
    --server-revision "$REVISION" --model "$MODEL" --no-debug-route \
    --max-connections "$MAX_SESSIONS" --server-pid "$SERVER_PID" \
    --server-log "$SERVER_LOG" ${device_flag[@]+"${device_flag[@]}"} --output "$directory" "$@" \
    >"$directory.log" 2>&1
}

passed() {
  "$PYTHON" -c 'import json, sys; sys.exit(bool(json.load(open(sys.argv[1]))["failed_checks"]))' "$1/summary.json" 2>/dev/null
}

# cell NAME CLIENTS AUDIO RUN_ARGUMENTS...
cell() {
  name=$1
  clients=$2
  audio=$PAIR/$3
  shift 3
  if [ ! -f "$OUT/smoke/$name/summary.json" ]; then
    rm -rf "$OUT/smoke/$name"
    mkdir -p "$OUT/smoke"
    stress "$OUT/smoke/$name" --concurrency 2 --cycles 1 --audio-file "$audio" "$@"
  fi
  if ! passed "$OUT/smoke/$name"; then
    echo "$name smoke FAILED, full run skipped: see $OUT/smoke/$name.log" | tee -a "$OUT/progress.log"
    return
  fi
  if [ ! -f "$OUT/full/$name/summary.json" ]; then
    rm -rf "$OUT/full/$name"
    mkdir -p "$OUT/full"
    stress "$OUT/full/$name" --concurrency "$clients" --cycles "$CYCLES" --audio-file "$audio" "$@"
  fi
  if passed "$OUT/full/$name"; then
    echo "$name passed" | tee -a "$OUT/progress.log"
  else
    echo "$name full run FAILED: see $OUT/full/$name.log" | tee -a "$OUT/progress.log"
  fi
}

: >"$OUT/progress.log"
# The two interrupt cells come first: they show whether the server cancels a reply the user talks over at all.
cell interrupt "$MAX_SESSIONS" input.wav --scenario interrupt
cell interrupt_control "$MAX_SESSIONS" clean_input.wav --scenario interrupt
cell churn "$MAX_SESSIONS" clean_input.wav --scenario churn
cell churn_overflow "$OVERFLOW" clean_input.wav --scenario churn
cell long "$MAX_SESSIONS" clean_input.wav --scenario long
cell drop_update_reset "$MAX_SESSIONS" clean_input.wav --scenario drop --leave-point update --leave-style reset
cell drop_append_reset "$MAX_SESSIONS" clean_input.wav --scenario drop --leave-point append --leave-style reset
cell drop_reply_reset "$MAX_SESSIONS" clean_input.wav --scenario drop --leave-point reply --leave-style reset
cell drop_reply_frame "$MAX_SESSIONS" clean_input.wav --scenario drop --leave-point reply --leave-style close_frame
cell cancel "$MAX_SESSIONS" clean_input.wav --scenario cancel
cell cancel_reply_reset "$MAX_SESSIONS" clean_input.wav --scenario cancel --leave-point reply --leave-style reset
cell reconnect_reply_reset "$MAX_SESSIONS" clean_input.wav --scenario reconnect --reconnects 2 --leave-point reply --leave-style reset

"$PYTHON" - "$OUT" <<'PYTHON_END' | tee "$OUT/report.txt"
import json
import sys
from pathlib import Path

FIELDS = (
    "session_outcomes",
    "failed_checks",
    "unmeasured_checks",
    "errors",
    "interrupted_replies",
    "cancel_statuses",
    "drops_mid_reply",
    "drops_after_reply",
    "capacity_unrecovered_cycles",
    "rss_growth_megabytes_per_cycle",
    "device_growth_megabytes_per_cycle",
    "thread_count_growth",
    "open_files_growth",
    "process_count_growth",
    "server_log",
)
root = Path(sys.argv[1])
for stage in ("smoke", "full"):
    for path in sorted(root.glob(f"{stage}/*/summary.json")):
        summary = json.loads(path.read_text())
        print(f"== {stage}/{path.parent.name}")
        for field in FIELDS:
            print(f"   {field}: {json.dumps(summary[field])}")
PYTHON_END
echo "done: send back $OUT" | tee -a "$OUT/progress.log"
