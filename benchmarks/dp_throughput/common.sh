# SPDX-License-Identifier: Apache-2.0
# Shared config for the dp_throughput measurement tools. Override any of these via env.
# Source this from the other scripts. NOT executable on its own.
DPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
: "${REPO:=$(cd "$DPT_DIR/../.." && pwd)}"     # repo root (tool lives in benchmarks/dp_throughput)
: "${MODEL:?export MODEL=<served model path or HF id> (an omni TTS pipeline model)}"
: "${MODEL_NAME:=$MODEL}"                        # served model name; MUST equal the model the client sends
: "${CARD:=0}"                                   # physical GPU index (nvidia-smi) to run everything on
: "${NUMA_NODE:=0}"                              # NUMA node of $CARD (see: nvidia-smi topo -m)
: "${SERVER_CORES:=0-15}"                        # NUMA-local cores for server replicas
: "${CLIENT_CORES:=16-23}"                       # cores for load clients, disjoint from SERVER_CORES
: "${MEM_FRAC:=}"                                # optional per-replica HIGGS_MEM_FRAC (needs the manual patch; see README)
: "${MPS_PIPE:=/tmp/dpt_mps_pipe}"
: "${MPS_LOG:=/tmp/dpt_mps_log}"
: "${GPU_METRICS_SET:=gh100}"                    # nsys --gpu-metrics-set for your arch (gh100 = Hopper H100/H200)
export REPO MODEL MODEL_NAME CARD NUMA_NODE SERVER_CORES CLIENT_CORES MEM_FRAC MPS_PIPE MPS_LOG GPU_METRICS_SET

# UUID of $CARD, used to sweep orphan compute-apps on it.
dpt_gpu_uuid() { nvidia-smi --query-gpu=index,uuid --format=csv,noheader | awk -F', *' -v c="$CARD" '$1==c{print $2}'; }
