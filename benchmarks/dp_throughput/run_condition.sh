#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# run_condition.sh <label> <dur> <ports_csv> <concs_csv> <client_cores_csv> <outdir>
# One measurement condition against already-running server(s) or a router: start one
# pinned continuous load client per (port,conc,client_cores), ramp, capture device-level
# nsys gpu-metrics on $CARD over the steady window, parse SM%/DRAM%, wait clients, sum
# throughput, and prove saturation from the server logs. Reuses parse_gpu_metrics.py +
# analyze_saturation.py + load_client.py.
#
# Server logs for the saturation check default to <outdir>/../srv_<port>.log (where
# launch_server.sh writes them); override with SAT_LOGS="log1 log2 ..." (used for router,
# whose workers log into the router log). Env: VOICE_CLONE=1, MAX_NEW_TOKENS=512,
# GPU_METRICS_SET=gh100 (Hopper; use your arch's nsys metric set).
set -u
source "$(dirname "$0")/common.sh"
LABEL=$1; DUR=$2; PORTS=$3; CONCS=$4; CCORES=$5; OUT=$6
DPT="$DPT_DIR"
mkdir -p "$OUT"
IFS=',' read -ra PA <<< "$PORTS"
IFS=',' read -ra CA <<< "$CONCS"
IFS=',' read -ra CC <<< "$CCORES"
VCFLAG=""; [ "${VOICE_CLONE:-1}" = "1" ] && VCFLAG="--voice-clone"
MNT="${MAX_NEW_TOKENS:-512}"; GMS="${GPU_METRICS_SET:-gh100}"

echo "[run] $LABEL card=$CARD dur=${DUR}s ports=$PORTS concs=$CONCS ccores=$CCORES" | tee "$OUT/run.log"
CPIDS=()
for i in "${!PA[@]}"; do
  numactl -C "${CC[$i]}" --membind="$NUMA_NODE" \
    python "$DPT/load_client.py" --port "${PA[$i]}" --model "$MODEL" \
    --conc "${CA[$i]}" --secs $((DUR+40)) --warmup-secs 20 $VCFLAG \
    --max-new-tokens "$MNT" --out "$OUT/client_${i}_${PA[$i]}.json" \
    > "$OUT/client_${i}_${PA[$i]}.log" 2>&1 &
  CPIDS+=($!)
  echo "[run] client$i port=${PA[$i]} conc=${CA[$i]} cores=${CC[$i]} pid=$!" | tee -a "$OUT/run.log"
done

echo "[run] ramp 22s..." | tee -a "$OUT/run.log"
sleep 22
date +%s > "$OUT/window_start.txt"
REP="$OUT/gm_${LABEL}"
nsys profile --gpu-metrics-devices "$CARD" --gpu-metrics-set "$GMS" \
  --gpu-metrics-frequency 10000 -d "$DUR" -o "$REP" -f true sleep $((DUR+3)) \
  > "$OUT/nsys_${LABEL}.log" 2>&1
date +%s > "$OUT/window_end.txt"
python "$DPT/parse_gpu_metrics.py" "$REP.nsys-rep" "$LABEL" | tee "$OUT/sol_${LABEL}.txt"

for p in "${CPIDS[@]}"; do wait "$p"; done

echo "==== THROUGHPUT ($LABEL) ====" | tee -a "$OUT/run.log"
TOT=0
for i in "${!PA[@]}"; do
  q=$(python -c "import json;print(json.load(open('$OUT/client_${i}_${PA[$i]}.json'))['qps_window'])" 2>/dev/null || echo 0)
  cv=$(python -c "import json;print(json.load(open('$OUT/client_${i}_${PA[$i]}.json'))['bucket_qps_cv'])" 2>/dev/null || echo NA)
  echo "  client$i port ${PA[$i]} conc ${CA[$i]} : qps_window=$q cv=$cv" | tee -a "$OUT/run.log"
  TOT=$(python -c "print(round($TOT+$q,4))")
done
echo "  AGGREGATE qps_window = $TOT" | tee -a "$OUT/run.log"

echo "==== SATURATION ($LABEL) ====" | tee -a "$OUT/run.log"
if [ -n "${SAT_LOGS:-}" ]; then SLOGS="$SAT_LOGS"; else
  SLOGS=""; for i in "${!PA[@]}"; do SLOGS="$SLOGS $OUT/../srv_${PA[$i]}.log"; done
fi
python "$DPT/analyze_saturation.py" "$OUT" $SLOGS 2>/dev/null | tee -a "$OUT/run.log"
echo "[run] DONE $LABEL" | tee -a "$OUT/run.log"
