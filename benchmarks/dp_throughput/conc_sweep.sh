#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# conc_sweep.sh <port> <concs_csv> [secs_per]
# Drive an already-running server at increasing concurrency; print qps per point so you can
# find the throughput plateau. A plateau (more offered concurrency stops raising qps) is the
# first half of the client-control check: it means the server is saturated, not the client.
# Read-only on the serving path. Prints CSV: conc,qps_window,cv.
set -u
source "$(dirname "$0")/common.sh"
DPT="$DPT_DIR"
PORT=$1; CONCS=$2; SECS=${3:-25}
: "${OUTDIR:=./dp_runs}"; mkdir -p "$OUTDIR/sweep"
VCFLAG=""; [ "${VOICE_CLONE:-1}" = "1" ] && VCFLAG="--voice-clone"
IFS=',' read -ra CS <<< "$CONCS"
echo "conc,qps_window,cv"
for c in "${CS[@]}"; do
  numactl -C "$CLIENT_CORES" --membind="$NUMA_NODE" \
    python "$DPT/load_client.py" --port "$PORT" --model "$MODEL" --conc "$c" \
    --secs "$SECS" --warmup-secs 6 $VCFLAG --out "$OUTDIR/sweep/c${c}.json" \
    > "$OUTDIR/sweep/c${c}.log" 2>&1
  q=$(python -c "import json;d=json.load(open('$OUTDIR/sweep/c${c}.json'));print(f\"{d['qps_window']},{d['bucket_qps_cv']}\")" 2>/dev/null || echo "ERR,ERR")
  echo "$c,$q"
done
