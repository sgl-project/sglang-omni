#!/bin/bash
# autodp.sh — size and launch same-GPU DP automatically.
#
# Computes the maximum SAFE DP for this GPU + model + KV cap from the
# two-constraint sizing model (see note/2026-07-18-h200-same-gpu-dp-ipc-profiling.md):
#
#   1. capacity:  D*(s + h) <= M          (no-WS)
#                 W + D*(s - W + h) <= M  (WS)
#   2. profiler:  the LAST replica's mem-fraction budget must clear the KV cap;
#                 autodp derives --mem-fraction-static instead of trusting the
#                 model default (which under-resolves the last replica).
#
#   M = GPU memory, s = measured per-replica static footprint at the cap,
#   W = backbone weight bytes (shared once under WEIGHT_SHARE=1),
#   h = per-replica dynamic headroom (cuBLAS/activations/vocoder; NOT covered
#       by the static budget — undersizing h is how a run boots and then OOMs).
#
# Usage:
#   [env] bash examples/mps_dp/autodp.sh plan     # probe + print sizing only
#   [env] bash examples/mps_dp/autodp.sh up       # probe + size + launch
#
# Environment (defaults in parentheses):
#   MODEL (bosonai/higgs-tts-3-4b), MODEL_NAME (higgs), GPU_ID (0),
#   CONFIG ()                  config-file models (e.g. Qwen3-TTS): forwarded to
#                              launch.sh, which serves via --config instead of
#                              --model-path (MODEL/MODEL_NAME defaults are
#                              suppressed; launch.sh forbids MODEL with CONFIG)
#   MAX_TOTAL_TOKENS (100000)  common per-replica KV cap, required
#   WEIGHT_SHARE (1)           1 = share the AR backbone over CUDA IPC; use 0
#                              for models without weight-share support
#   HEADROOM_GIB (1.5)         h; measured: 1.3 works, 0.5 OOMs (Higgs, bs<=64).
#                              Re-measure if you raise max_running_requests.
#   N ()                       override replica count (must be <= computed max)
#   MAX_DP (16)                hard clamp on computed D (small models can size
#                              absurdly high before CPU becomes the limit)
#   MIN_CORES_PER_REPLICA (2)  clamp D so each replica keeps this many cores
#   STATIC_GIB ()              skip the probe: known per-replica footprint s
#   WEIGHTS_GIB ()             skip log-derived W (needed for WS sizing)
#   CORE_BLOCKS ()             forwarded to launch.sh; derived from the current
#                              cpuset when unset (server share = 3/4 of cores)
#
# The probe boots ONE replica at the cap on the target GPU, reads its
# memory.used as s, extracts W from the weight-share export line, and tears the
# probe down before sizing. Skipped when STATIC_GIB (+ WEIGHTS_GIB) are given.
set -euo pipefail

HERE=$(cd "$(dirname "$0")" && pwd)
CMD=${1:-plan}
CONFIG=${CONFIG:-}
if [ -n "$CONFIG" ]; then
  # launch.sh forbids MODEL alongside CONFIG; empty values read as unset there.
  MODEL=
  MODEL_NAME=${MODEL_NAME:-}
else
  MODEL=${MODEL:-bosonai/higgs-tts-3-4b}
  MODEL_NAME=${MODEL_NAME:-higgs}
fi
GPU_ID=${GPU_ID:-0}
CAP=${MAX_TOTAL_TOKENS:-100000}
WS=${WEIGHT_SHARE:-1}
H_GIB=${HEADROOM_GIB:-1.5}

die() { echo "error: $*" >&2; exit 1; }
[[ "$CMD" =~ ^(plan|up)$ ]] || die "usage: autodp.sh plan|up"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (run on a GPU node)"

M_MIB=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits -i "$GPU_ID")
USED_MIB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID")
[ "$USED_MIB" -lt 1024 ] || die "GPU $GPU_ID already has ${USED_MIB} MiB in use; sizing assumes an idle card"

# ---- derive CORE_BLOCKS lazily (server share = 3/4 of the cpuset) ------------
core_blocks_for() {
  python3 - "$1" <<'EOF'
import os, sys
n = int(sys.argv[1])
cores = sorted(os.sched_getaffinity(0))
srv = cores[: max(n, len(cores) * 3 // 4)]
k, r = divmod(len(srv), n)
assert k >= 1, f"need >= {n} server cores, have {len(srv)}"
out, i = [], 0
for b in range(n):
    sz = k + (1 if b < r else 0)
    out.append(",".join(map(str, srv[i:i+sz]))); i += sz
print(" ".join(out))
EOF
}

# ---- probe: boot one replica at the cap, measure s and W ---------------------
STATIC_GIB=${STATIC_GIB:-}
WEIGHTS_GIB=${WEIGHTS_GIB:-}
if [ -z "$STATIC_GIB" ]; then
  echo "[autodp] probing: booting 1 replica at cap=$CAP to measure the static footprint..."
  probe_blocks=$(core_blocks_for 1)
  MODEL=$MODEL MODEL_NAME=$MODEL_NAME CONFIG=$CONFIG \
  GPU_ID=$GPU_ID N=1 BASE_PORT=${BASE_PORT:-8801} \
  CORE_BLOCKS="$probe_blocks" MAX_TOTAL_TOKENS=$CAP WEIGHT_SHARE=$WS \
    bash "$HERE/launch.sh" up > /tmp/autodp_probe.$$.log 2>&1 \
    || { echo "--- probe log (/tmp/autodp_probe.$$.log) ---" >&2; \
         cat /tmp/autodp_probe.$$.log >&2; \
         echo "--- diagnostics ---" >&2; \
         df -h /tmp >&2; nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2; \
         ls /tmp/sglang-omni-same-gpu-dp/$UID/gpu-*/ 2>/dev/null >&2; \
         die "probe boot failed"; }
  probe_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID")
  STATIC_GIB=$(python3 -c "print(round($probe_used/1024, 2))")
  run_id=$(basename "$(bash "$HERE/launch.sh" list | head -1)")
  state_dir=$(bash "$HERE/launch.sh" list | head -1)
  if [ -z "$WEIGHTS_GIB" ]; then
    WEIGHTS_GIB=$(grep -hoE "leader exported .*\(([0-9.]+) GiB" "$state_dir"/logs/replica_0.log 2>/dev/null \
      | grep -oE "[0-9.]+" | tail -1 || true)
  fi
  bash "$HERE/launch.sh" down "$run_id" > /dev/null
  echo "[autodp] probe: s=${STATIC_GIB} GiB per replica, W=${WEIGHTS_GIB:-?} GiB weights"
fi
[ -n "$WEIGHTS_GIB" ] || { [ "$WS" = 0 ] || die "WS sizing needs WEIGHTS_GIB (probe could not extract it)"; WEIGHTS_GIB=0; }

# ---- sizing ------------------------------------------------------------------
SRV_CORE_COUNT=$(python3 -c "import os; c=len(os.sched_getaffinity(0)); print(max(1, c*3//4))")
readarray -t SIZED < <(python3 - <<EOF
m = $M_MIB / 1024
s = float("$STATIC_GIB")
w = float("$WEIGHTS_GIB")
h = float("$H_GIB")
cap = $CAP
ws = $WS
if ws:
    d = int((m - w) // (s - w + h))
else:
    d = int(m // (s + h))
# clamps: hard MAX_DP + keep MIN_CORES_PER_REPLICA server cores per replica
d = min(d, ${MAX_DP:-16}, $SRV_CORE_COUNT // ${MIN_CORES_PER_REPLICA:-2})
d = max(d, 1)
# required mem fraction for the LAST replica (earlier ones need less);
# +0.02 margin, clamped. The launcher's exact-KV check validates it at boot.
prev = (d - 1) * ((s - w) if ws else s) + (w if ws else 0)
free_last = m - prev
mf = min(0.97, round(s / free_last + 0.02, 3)) if free_last > 0 else 0.97
static_total = w + d * (s - w) if ws else d * s
print(d); print(mf); print(round(static_total, 1)); print(round(m - static_total, 1))
EOF
)
D_MAX=${SIZED[0]}; MF_REQ=${SIZED[1]}; STATIC_TOT=${SIZED[2]}; FREE_LEFT=${SIZED[3]}
N=${N:-$D_MAX}
[ "$N" -le "$D_MAX" ] || die "N=$N exceeds computed safe maximum $D_MAX"

M_GIB=$(python3 -c "print(round($M_MIB/1024,1))")
echo "[autodp] plan: GPU ${GPU_ID} M=${M_GIB} GiB | s=${STATIC_GIB} GiB W=${WEIGHTS_GIB} GiB h=${H_GIB} GiB cap=${CAP} weight_share=${WS}"
echo "[autodp] plan: max safe DP = ${D_MAX} (launching N=${N}); static total ~${STATIC_TOT} GiB, dynamic pool ~${FREE_LEFT} GiB; derived MF=${MF_REQ}"
[ "$CMD" = plan ] && exit 0

# ---- launch ------------------------------------------------------------------
BLOCKS=${CORE_BLOCKS:-$(core_blocks_for "$N")}
echo "[autodp] launching N=$N (blocks: $BLOCKS)"
launch_log=$(mktemp /tmp/autodp_up.XXXXXX.log)
if ! MODEL=$MODEL MODEL_NAME=$MODEL_NAME CONFIG=$CONFIG \
     GPU_ID=$GPU_ID N=$N BASE_PORT=${BASE_PORT:-8801} \
     CORE_BLOCKS="$BLOCKS" MAX_TOTAL_TOKENS=$CAP WEIGHT_SHARE=$WS MF=$MF_REQ \
     bash "$HERE/launch.sh" up 2>&1 | tee "$launch_log"; then
  # Config-file pipelines (e.g. Qwen3-TTS) reject --mem-fraction-static.
  # The derived MF is only *required* when it exceeds the model default;
  # retry without it and let the launcher's exact-KV check arbitrate.
  if grep -qE "mem-fraction-static requires a pipeline|sets mem_fraction_static through both" "$launch_log"; then
    echo "[autodp] pipeline rejects --mem-fraction-static; retrying with the model default (derived MF=$MF_REQ was advisory)"
    # The failed attempt keeps its state dir for diagnostics; remove it (all
    # recorded PIDs are dead) so the retry can start.
    for st in /tmp/sglang-omni-same-gpu-dp/$UID/gpu-$GPU_ID/run-*; do
      [ -f "$st/replicas.tsv" ] || { rm -rf "$st" 2>/dev/null; continue; }
      live=0
      while IFS=$'\t' read -r _ pid _ _ _ _; do
        kill -0 "$pid" 2>/dev/null && live=1
      done < "$st/replicas.tsv"
      [ "$live" = 0 ] && rm -rf "$st"
    done
    MODEL=$MODEL MODEL_NAME=$MODEL_NAME CONFIG=$CONFIG \
    GPU_ID=$GPU_ID N=$N BASE_PORT=${BASE_PORT:-8801} \
    CORE_BLOCKS="$BLOCKS" MAX_TOTAL_TOKENS=$CAP WEIGHT_SHARE=$WS \
      bash "$HERE/launch.sh" up
  else
    die "launch failed (see above)"
  fi
fi
rm -f "$launch_log"

# ---- post-launch headroom verification --------------------------------------
used_after=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID")
free_after=$(python3 -c "print(round(($M_MIB-$used_after)/1024,1))")
need=$(python3 -c "print(round($N*$H_GIB,1))")
if python3 -c "exit(0 if $M_MIB-$used_after >= $N*$H_GIB*1024 else 1)"; then
  echo "[autodp] headroom check PASS: ${free_after} GiB free >= ${need} GiB (${N} x ${H_GIB})"
else
  echo "[autodp] headroom check FAIL: ${free_after} GiB free < ${need} GiB — expect OOM under load; bring the run down and lower N or the cap" >&2
  exit 1
fi
