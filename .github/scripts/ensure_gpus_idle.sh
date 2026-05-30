#!/usr/bin/env bash
# Hard GPU cleanup between CI benchmark stages.
#
# delete_gpu_process.sh only kills PIDs nvidia-smi lists as compute apps.
# Colocated Omni workers often leave orphan multiprocessing.spawn children
# that still hold VRAM while nvidia-smi reports "No running processes".
# This script kills those first, then delegates to delete_gpu_process.sh.
set -uo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
memory_threshold_mb="${OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB:-1024}"
wait_timeout_seconds="${OMNI_CI_GPU_CLEAN_WAIT_SECONDS:-600}"
poll_seconds="${OMNI_CI_GPU_CLEAN_POLL_SECONDS:-5}"

kill_orphan_gpu_processes() {
    local patterns=(
        "multiprocessing.spawn"
        "sglang_omni_router.serve"
        "sgl-omni serve"
        "stage_process"
    )
    for pattern in "${patterns[@]}"; do
        pkill -9 -f "${pattern}" 2>/dev/null || true
    done
    rm -f /tmp/sglang_omni_gpu_*_startup.lock

    # nvidia-smi often misses zombie CUDA contexts; scan open /dev/nvidia* fds.
    local pid cmdline
    for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$' || true); do
        if ls -l "/proc/${pid}/fd" 2>/dev/null | grep -q nvidia; then
            cmdline="$(tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
            if [ -n "${cmdline}" ]; then
                echo "  killing orphan GPU PID ${pid}: ${cmdline}"
                kill -9 "${pid}" 2>/dev/null || true
            fi
        fi
    done
}

echo "=== ensure_gpus_idle (threshold=${memory_threshold_mb} MiB, timeout=${wait_timeout_seconds}s) ==="
kill_orphan_gpu_processes
sleep 2

attempt=0
while true; do
    attempt=$((attempt + 1))
    if OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB="${memory_threshold_mb}" \
       OMNI_CI_GPU_CLEAN_WAIT_SECONDS="${wait_timeout_seconds}" \
       OMNI_CI_GPU_CLEAN_POLL_SECONDS="${poll_seconds}" \
       bash "${repo_root}/.github/scripts/delete_gpu_process.sh"; then
        echo "ensure_gpus_idle: all GPUs below ${memory_threshold_mb} MiB"
        exit 0
    fi
    if [ "${attempt}" -ge 3 ]; then
        echo "::error::ensure_gpus_idle failed after ${attempt} cleanup rounds"
        nvidia-smi
        exit 1
    fi
    echo "  cleanup round ${attempt} incomplete — retrying orphan kill..."
    kill_orphan_gpu_processes
    sleep 3
done
