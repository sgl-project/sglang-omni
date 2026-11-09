#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# scan_dp.sh <max_dp> <use_mps 0|1> [reps]
# Same-card data-parallel sweep DP=1..max_dp on $CARD. Per DP: (fresh MPS if use_mps=1) ->
# launch DP replicas on $SERVER_CORES (shared) with a fit mem-fraction -> wait health ->
# run <reps> measured conditions (one continuous client per replica at CONC, clients split
# over $CLIENT_CORES) -> graceful teardown. The ONLY variable across DP is replica count.
#
# Per-DP mem-fraction: MEM_FRACS="0.40,0.40,0.27,0.20" (csv, index=DP-1) else ~0.85/DP.
# NOTE: >1 replica per card needs the mem-fraction patch documented in the README (stock
# code hardcodes it, so 2 replicas OOM). CONC (per-replica offered) default 96. Requires
# the model's AR engine to expose enough max_running_requests to saturate at CONC.
set -u
source "$(dirname "$0")/common.sh"
DPT="$DPT_DIR"
MAXDP=${1:-3}; USE_MPS=${2:-0}; REPS=${3:-2}
export USE_MPS
: "${OUTDIR:=./dp_runs}"; mkdir -p "$OUTDIR"
CONC=${CONC:-96}; DUR=${DUR:-60}
IFS=',' read -ra MFA <<< "${MEM_FRACS:-}"

split_cores() {  # <a-b> <n> -> n comma-separated sub-ranges of the core range
  local lo=${1%-*} hi=${1#*-} n=$2 total per i start end out=""
  total=$((hi-lo+1)); per=$((total/n))
  for ((i=0; i<n; i++)); do
    start=$((lo+i*per)); end=$((start+per-1)); [ $i -eq $((n-1)) ] && end=$hi
    out="$out,${start}-${end}"
  done
  echo "${out#,}"
}

graceful_teardown() {
  pkill -TERM -f "sglang_omni.cli serve" 2>/dev/null
  pkill -TERM -f "sgl-omni serve" 2>/dev/null
  sleep 8
  pkill -9 -f multiprocessing.spawn 2>/dev/null
  pkill -9 -f "load_client.py" 2>/dev/null
  sleep 2
  local uu; uu=$(dpt_gpu_uuid)
  if [ -n "$uu" ]; then
    for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | grep "$uu" | awk -F',' '{print $1}'); do kill -9 "$pid" 2>/dev/null; done
  fi
  sleep 2
}

for ((DP=1; DP<=MAXDP; DP++)); do
  graceful_teardown
  [ "$USE_MPS" = "1" ] && bash "$DPT/mps.sh" start
  MF="${MFA[$((DP-1))]:-}"
  [ -z "$MF" ] && MF=$(python -c "print(round(0.85/$DP,2))")
  ports=""; concs=""
  ccores=$(split_cores "$CLIENT_CORES" "$DP")
  ok=1
  for ((k=0; k<DP; k++)); do
    p=$((8901+k)); ports="$ports,$p"; concs="$concs,$CONC"
    bash "$DPT/launch_server.sh" "$p" "$MF"
    bash "$DPT/wait_health.sh" "$p" 80 || { echo "[scan] DP=$DP port $p FAILED (VRAM/OOM?) -> recording limit"; ok=0; break; }
  done
  ports="${ports#,}"; concs="${concs#,}"
  if [ "$ok" != "1" ]; then
    echo "[scan] DP=$DP DID_NOT_FIT (mem_frac=$MF mps=$USE_MPS)"
    graceful_teardown; [ "$USE_MPS" = "1" ] && bash "$DPT/mps.sh" stop
    continue
  fi
  echo "[scan] DP=$DP healthy (mem_frac=$MF mps=$USE_MPS); card mem:"; nvidia-smi --query-gpu=index,memory.used --format=csv,noheader | awk -F',' -v c="$CARD" '$1+0==c{print}'
  for ((R=1; R<=REPS; R++)); do
    lbl="dp${DP}_mps${USE_MPS}_r${R}"
    OUTDIR="$OUTDIR" bash "$DPT/run_condition.sh" "$lbl" "$DUR" "$ports" "$concs" "$ccores" "$OUTDIR/$lbl"
  done
  graceful_teardown
  [ "$USE_MPS" = "1" ] && bash "$DPT/mps.sh" stop
done
echo "[scan] SCAN DONE (max_dp=$MAXDP mps=$USE_MPS reps=$REPS)"
