#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# run_router_condition.sh <nworkers> [rep]
# Measure the STOCK sglang-omni router (no router code change) driving <nworkers> same-card
# replicas, so you can compare "manual half/half split" vs "via router" under the SAME
# workers/card. The router spawns the workers (launcher local backend, worker_gpu_ids all =
# $CARD); router + workers are pinned to $SERVER_CORES via taskset (inherited by the workers)
# and inherit HIGGS_MEM_FRAC + (if USE_MPS=1) the MPS pipe. Client sends ALL requests to the
# router port; the router least_request-splits. Also samples the router process CPU% (its
# single async event loop is the usual bottleneck).
#
# One rep per call with a FRESH MPS daemon: router+MPS is unstable across back-to-back reps
# (2nd sustained rep tends to crash a worker with cudaErrorMpsRpcFailure). Call once per rep.
set -u
source "$(dirname "$0")/../common.sh"
DPT="$DPT_DIR"                 # dp_throughput dir (parent of router_bench)
RB="$(cd "$(dirname "$0")" && pwd)"
N=${1:-2}; REP=${2:-1}
: "${OUTDIR:=./dp_runs}"
ROUTER_PORT=${ROUTER_PORT:-8900}
CONC=${CONC:-96}; DUR=${DUR:-60}
MF="${MEM_FRAC:-$(python -c "print(round(0.85/$N,2))")}"
OUT="$OUTDIR/router_dp${N}_mps${USE_MPS:-0}_r${REP}"; mkdir -p "$OUT"
RLOG="$OUT/router.log"; CFG="$OUT/launcher.yaml"

graceful_teardown() {
  pkill -TERM -f sglang_omni_router.serve 2>/dev/null; sleep 8
  pkill -TERM -f "sgl-omni serve" 2>/dev/null; pkill -TERM -f "sglang_omni.cli serve" 2>/dev/null; sleep 8
  pkill -9 -f load_client.py 2>/dev/null
  [ "${USE_MPS:-0}" = "1" ] && bash "$DPT/mps.sh" stop
  local uu; uu=$(dpt_gpu_uuid)
  [ -n "$uu" ] && for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader 2>/dev/null | grep "$uu" | awk -F',' '{print $1}'); do kill -9 "$pid" 2>/dev/null; done
  sleep 3
}

graceful_teardown
[ "${USE_MPS:-0}" = "1" ] && bash "$DPT/mps.sh" start

gpu_ids="[$(for ((k=0;k<N;k++)); do printf '"%s",' "$CARD"; done | sed 's/,$//')]"
sed -e "s|__MODEL__|$MODEL|g" -e "s|__NWORKERS__|$N|" -e "s|__WORKER_GPU_IDS__|$gpu_ids|" "$RB/launcher.yaml.template" > "$CFG"

pre="PYTHONPATH=$REPO HIGGS_MEM_FRAC=$MF"
[ "${USE_MPS:-0}" = "1" ] && pre="$pre CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_LOG_DIRECTORY=$MPS_LOG"
cd "$REPO"
env $pre numactl --membind="$NUMA_NODE" taskset -c "$SERVER_CORES" \
  python -m sglang_omni_router.serve --host 0.0.0.0 --port "$ROUTER_PORT" \
  --launcher-config "$CFG" --policy least_request \
  --health-success-threshold 1 --health-failure-threshold 2 \
  --health-check-interval-secs 2 --log-level info > "$RLOG" 2>&1 &
RPID=$!
echo "[router] pid=$RPID nworkers=$N mem_frac=$MF mps=${USE_MPS:-0}"

for ((k=0; k<N; k++)); do
  bash "$DPT/wait_health.sh" $((8901+k)) 100 || { echo "[router] worker $((8901+k)) FAILED"; graceful_teardown; exit 1; }
done
RR=""; for i in $(seq 1 30); do RR=$(curl -s -o /dev/null -w "%{http_code}" -m 3 "127.0.0.1:$ROUTER_PORT/ready" 2>/dev/null); [ "$RR" = "200" ] && break; sleep 4; done
echo "[router] ready=$RR"

# smoke one request through the router
VCFLAG=""; [ "${VOICE_CLONE:-1}" = "1" ] && VCFLAG="--voice-clone"
numactl -C "$CLIENT_CORES" --membind="$NUMA_NODE" python "$DPT/load_client.py" \
  --port "$ROUTER_PORT" --model "$MODEL" --conc 4 --secs 12 --warmup-secs 3 $VCFLAG \
  --out "$OUT/smoke.json" > "$OUT/smoke.log" 2>&1
SE=$(python -c "import json;print(json.load(open('$OUT/smoke.json'))['errors'])" 2>/dev/null || echo PF)
echo "[router] smoke errors=$SE"
[ "$SE" != "0" ] && { echo "[router] SMOKE_FAIL"; graceful_teardown; exit 1; }

# all clients -> router port; sample router CPU% in the background
ports=$(for ((k=0;k<N;k++)); do printf '%s,' "$ROUTER_PORT"; done | sed 's/,$//')
concs=$(for ((k=0;k<N;k++)); do printf '%s,' "$CONC"; done | sed 's/,$//')
lo=${CLIENT_CORES%-*}; hi=${CLIENT_CORES#*-}; per=$(((hi-lo+1)/N)); ccores=""
for ((k=0;k<N;k++)); do s=$((lo+k*per)); e=$((s+per-1)); [ $k -eq $((N-1)) ] && e=$hi; ccores="$ccores,${s}-${e}"; done
ccores="${ccores#,}"
( top -b -n 42 -d 2 -p "$RPID" 2>/dev/null | awk -v r="$RPID" '$1==r{print $9}' > "$OUT/router_cpu.txt" ) &

SAT_LOGS="$RLOG" OUTDIR="$OUTDIR" bash "$DPT/run_condition.sh" \
  "router_dp${N}_mps${USE_MPS:-0}_r${REP}" "$DUR" "$ports" "$concs" "$ccores" "$OUT"

python3 -c "import statistics as s;v=[float(x) for x in open('$OUT/router_cpu.txt').read().split() if x.strip()];print('router cpu%% med=%.1f max=%.1f (one core=100%%)'%(s.median(v),max(v))) if v else print('router cpu%% no-samples')" | tee -a "$OUT/run.log"
graceful_teardown
echo "[router] DONE nworkers=$N rep=$REP"
