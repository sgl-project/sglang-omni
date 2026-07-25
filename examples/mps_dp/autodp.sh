#!/bin/bash
# autodp.sh — size and launch same-GPU DP automatically.
#
# Computes the maximum ESTIMATED safe DP (boot-validated, not workload-proven)
# for this GPU + model + KV cap from the
# two-constraint sizing model:
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
#   MODEL (), MODEL_NAME (), GPU_ID (0),
#   CONFIG (configs/higgs_h100_dp3.yaml when MODEL is unset)
#                              pipeline config forwarded to launch.sh, which
#                              serves via --config instead of --model-path
#                              (launch.sh forbids MODEL with CONFIG and
#                              requires CONFIG when WEIGHT_SHARE=1, so the
#                              supported-model preflight can run)
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
if [ -z "$CONFIG" ] && [ -z "${MODEL:-}" ]; then
  # Default flow: Higgs through its shipped config, so the weight-share
  # supported-model preflight can run before any resource is created.
  CONFIG=$HERE/configs/higgs_h100_dp3.yaml
fi
if [ -n "$CONFIG" ]; then
  # launch.sh forbids MODEL alongside CONFIG; empty values read as unset there.
  MODEL=
  MODEL_NAME=${MODEL_NAME:-}
else
  MODEL=${MODEL:-}
  MODEL_NAME=${MODEL_NAME:-higgs}
fi
GPU_ID=${GPU_ID:-0}
CAP=${MAX_TOTAL_TOKENS:-100000}
# Note (Jiaxin Deng): weight sharing supports only validated configs (the
# WEIGHT_SHARE_VALIDATED_CONFIGS registry in config.py); others are rejected in
# preflight. Size those with WEIGHT_SHARE=0.
WS=${WEIGHT_SHARE:-1}
H_GIB=${HEADROOM_GIB:-1.5}

MAX_DP=${MAX_DP:-16}
MIN_CORES_PER_REPLICA=${MIN_CORES_PER_REPLICA:-2}

die() { echo "error: $*" >&2; exit 1; }
[[ "$CMD" =~ ^(plan|up)$ ]] || die "usage: autodp.sh plan|up"
command -v nvidia-smi >/dev/null || die "nvidia-smi not found (run on a GPU node)"

# Note (Yueying Li): every public knob is validated before any resource is
# created — a sizing tool that promises a safe plan must reject nonsensical
# inputs (WEIGHT_SHARE=2, MIN_CORES_PER_REPLICA=0, nonnumeric MAX_DP, N=0)
# instead of folding them into arithmetic.
[[ "$GPU_ID" =~ ^[0-9]+$ ]] \
  || die "GPU_ID must be a non-negative integer, got '$GPU_ID'"
[[ "$CAP" =~ ^[1-9][0-9]*$ ]] \
  || die "MAX_TOTAL_TOKENS must be a positive integer, got '$CAP'"
[[ "$WS" =~ ^[01]$ ]] \
  || die "WEIGHT_SHARE must be 0 or 1, got '$WS'"
[[ "$H_GIB" =~ ^[0-9]+([.][0-9]+)?$ ]] \
  || die "HEADROOM_GIB must be a non-negative number, got '$H_GIB'"
[[ "$MAX_DP" =~ ^[1-9][0-9]*$ ]] \
  || die "MAX_DP must be a positive integer, got '$MAX_DP'"
[[ "$MIN_CORES_PER_REPLICA" =~ ^[1-9][0-9]*$ ]] \
  || die "MIN_CORES_PER_REPLICA must be a positive integer, got '$MIN_CORES_PER_REPLICA'"
[ -z "${N:-}" ] || [[ "$N" =~ ^[1-9][0-9]*$ ]] \
  || die "N must be a positive integer, got '$N'"

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
  # Note (Jiaxin Deng): pin a unique run id so we read and tear down exactly the
  # run we started, never a rediscovered newest dir (which a concurrent launcher
  # on another GPU could win).
  state_root=${STATE_ROOT:-/tmp/sglang-omni-same-gpu-dp/$UID}
  # Note (Jiaxin Deng): the run id must start with run- so launch.sh's find_runs
  # / down / stale-run guard (all glob run-*) can see and tear it down.
  probe_run="run-autodp-probe-$GPU_ID-$$"
  probe_dir="$state_root/gpu-$GPU_ID/$probe_run"
  RUN_ID=$probe_run MODEL=$MODEL MODEL_NAME=$MODEL_NAME CONFIG=$CONFIG \
  GPU_ID=$GPU_ID N=1 BASE_PORT=${BASE_PORT:-8801} \
  CORE_BLOCKS="$probe_blocks" MAX_TOTAL_TOKENS=$CAP WEIGHT_SHARE=$WS \
    bash "$HERE/launch.sh" up > /tmp/autodp_probe.$$.log 2>&1 \
    || { echo "--- probe log (/tmp/autodp_probe.$$.log) ---" >&2; \
         cat /tmp/autodp_probe.$$.log >&2; \
         echo "--- diagnostics ---" >&2; \
         df -h /tmp >&2; nvidia-smi --query-compute-apps=pid,used_memory --format=csv >&2; \
         bash "$HERE/launch.sh" down "$probe_run" >/dev/null 2>&1 || true; \
         die "probe boot failed"; }
  probe_used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$GPU_ID")
  STATIC_GIB=$(python3 -c "print(round($probe_used/1024, 2))")
  if [ -z "$WEIGHTS_GIB" ]; then
    WEIGHTS_GIB=$(grep -hoE "leader exported .*\(([0-9.]+) GiB" "$probe_dir"/logs/replica_0.log 2>/dev/null \
      | grep -oE "[0-9.]+" | tail -1 || true)
  fi
  bash "$HERE/launch.sh" down "$probe_run" > /dev/null
  echo "[autodp] probe: s=${STATIC_GIB} GiB per replica, W=${WEIGHTS_GIB:-?} GiB weights"
fi
[ -n "$WEIGHTS_GIB" ] || { [ "$WS" = 0 ] || die "WS sizing needs WEIGHTS_GIB (probe could not extract it)"; WEIGHTS_GIB=0; }

# ---- sizing ------------------------------------------------------------------
SRV_CORE_COUNT=$(python3 -c "import os; c=len(os.sched_getaffinity(0)); print(max(1, c*3//4))")
SIZED_RAW=$(python3 - <<EOF
import sys

m = $M_MIB / 1024
s = float("$STATIC_GIB")
w = float("$WEIGHTS_GIB")
h = float("$H_GIB")
ws = $WS

if not (m > 0 and s > 0 and h >= 0 and w >= 0 and (w < s or not ws)):
    sys.exit("autodp: invalid sizing inputs m=%s s=%s w=%s h=%s" % (m, s, w, h))

denom = (s - w + h) if ws else (s + h)
if denom <= 0:
    sys.exit("autodp: non-positive per-replica cost %.3f GiB; cannot size" % denom)

d = int((m - w) // denom) if ws else int(m // denom)
d = min(d, $MAX_DP, $SRV_CORE_COUNT // $MIN_CORES_PER_REPLICA)

# Note (Jiaxin Deng): reject an unsizable card instead of forcing d=1, which
# would report "safe" for a plan that cannot even boot one replica.
if d < 1:
    need = (w + denom) if ws else denom
    sys.exit("autodp: no replica fits on %.1f GiB (needs >= %.1f GiB); lower the KV cap or free the GPU" % (m, need))

prev = (d - 1) * ((s - w) if ws else s) + (w if ws else 0)
free_last = m - prev
mf = min(0.97, round(s / free_last + 0.02, 3)) if free_last > 0 else 0.97
static_total = (w + d * (s - w)) if ws else (d * s)

print(d)
print(mf)
print(round(static_total, 1))
print(round(m - static_total, 1))
EOF
) || die "autodp sizing failed (see message above)"
readarray -t SIZED <<< "$SIZED_RAW"
D_MAX=${SIZED[0]}; MF_REQ=${SIZED[1]}; STATIC_TOT=${SIZED[2]}; FREE_LEFT=${SIZED[3]}
N=${N:-$D_MAX}
[ "$N" -le "$D_MAX" ] || die "N=$N exceeds computed max estimated DP $D_MAX"

M_GIB=$(python3 -c "print(round($M_MIB/1024,1))")
echo "[autodp] plan: GPU ${GPU_ID} M=${M_GIB} GiB | s=${STATIC_GIB} GiB W=${WEIGHTS_GIB} GiB h=${H_GIB} GiB cap=${CAP} weight_share=${WS}"
echo "[autodp] plan: max estimated DP = ${D_MAX} (launching N=${N}); static total ~${STATIC_TOT} GiB, dynamic pool ~${FREE_LEFT} GiB; derived MF=${MF_REQ} (boot-validated estimate, validate under sustained load)"
[ "$CMD" = plan ] && exit 0

# ---- launch ------------------------------------------------------------------
BLOCKS=${CORE_BLOCKS:-$(core_blocks_for "$N")}
echo "[autodp] launching N=$N (blocks: $BLOCKS)"
launch_run="run-autodp-$GPU_ID-$$"
launch_log=$(mktemp /tmp/autodp_up.XXXXXX.log)
if ! RUN_ID=$launch_run MODEL=$MODEL MODEL_NAME=$MODEL_NAME CONFIG=$CONFIG \
     GPU_ID=$GPU_ID N=$N BASE_PORT=${BASE_PORT:-8801} \
     CORE_BLOCKS="$BLOCKS" MAX_TOTAL_TOKENS=$CAP WEIGHT_SHARE=$WS MF=$MF_REQ \
     bash "$HERE/launch.sh" up 2>&1 | tee "$launch_log"; then
  # A config-file pipeline may reject --mem-fraction-static; the derived MF is
  # only required when it exceeds the model default, so retry without it and
  # let the launcher's exact-KV check arbitrate.
  if grep -qE "mem-fraction-static requires a pipeline|sets mem_fraction_static through both" "$launch_log"; then
    echo "[autodp] pipeline rejects --mem-fraction-static; retrying with the model default (derived MF=$MF_REQ was advisory)"
    # Note (Jiaxin Deng): tear down exactly this run (pinned RUN_ID) through the
    # launcher, which does the start-time / PGID / MPS-client checks that a bare
    # rm -rf skips; never a rediscovered newest dir.
    bash "$HERE/launch.sh" down "$launch_run" \
      || die "autodp: could not tear down the failed --mem-fraction-static attempt $launch_run; inspect and retry"
    RUN_ID=$launch_run MODEL=$MODEL MODEL_NAME=$MODEL_NAME CONFIG=$CONFIG \
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
