#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# launch_server.sh <port> [mem_frac]
# Launch ONE omni server on physical $CARD, pinned to $SERVER_CORES / NUMA $NUMA_NODE.
# USE_MPS=1 routes CUDA through the per-user MPS daemon (see mps.sh). mem_frac (optional)
# is passed as HIGGS_MEM_FRAC so >1 replica can share a card -- this only takes effect
# with the one-line patch documented in the README (NOT part of this tooling PR).
# Detached; writes $OUTDIR/srv_<port>.log.
set -u
source "$(dirname "$0")/common.sh"
PORT=$1; MF="${2:-$MEM_FRAC}"
: "${OUTDIR:=./dp_runs}"; mkdir -p "$OUTDIR"

pre="CUDA_VISIBLE_DEVICES=$CARD PYTHONPATH=$REPO"
[ -n "$MF" ] && pre="$pre HIGGS_MEM_FRAC=$MF"
if [ "${USE_MPS:-0}" = "1" ]; then
  pre="$pre CUDA_MPS_PIPE_DIRECTORY=$MPS_PIPE CUDA_MPS_LOG_DIRECTORY=$MPS_LOG"
fi

cd "$REPO"
env $pre numactl -C "$SERVER_CORES" --membind="$NUMA_NODE" \
  python -m sglang_omni.cli serve \
  --model-path "$MODEL" --model-name "$MODEL_NAME" \
  --host 127.0.0.1 --port "$PORT" --allowed-local-media-path / \
  > "$OUTDIR/srv_${PORT}.log" 2>&1 &
echo "launched port=$PORT cores=$SERVER_CORES card=$CARD mem_frac=${MF:-default} mps=${USE_MPS:-0} pid=$!"
