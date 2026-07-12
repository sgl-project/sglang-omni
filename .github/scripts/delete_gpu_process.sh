#!/usr/bin/env bash
# Scoped GPU cleanup for concurrent calibration groups.
#
# CUDA_VISIBLE_DEVICES selects the *physical* GPU indices to clean. The script
# then unsets CUDA_VISIBLE_DEVICES before calling nvidia-smi so --id=N always
# means physical GPU N (CVD remapping would otherwise make --id=2 fail or hit
# the wrong device when CVD=2,3).
#
# Unscoped cleanup (every visible GPU) is refused unless explicitly allowed:
#   OMNI_CI_ALLOW_UNSCOPED_GPU_CLEAN=1  or  GITHUB_ACTIONS=true  or  CVD=all
set -uo pipefail

memory_threshold_mb="${OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB:-1024}"
wait_timeout_seconds="${OMNI_CI_GPU_CLEAN_WAIT_SECONDS:-120}"
poll_seconds="${OMNI_CI_GPU_CLEAN_POLL_SECONDS:-5}"

# Capture physical targets BEFORE unsetting CVD for nvidia-smi.
_raw_cvd="${CUDA_VISIBLE_DEVICES:-}"
target_gpu_ids="${_raw_cvd}"
unset CUDA_VISIBLE_DEVICES

_TR=/usr/bin/tr

_normalize_gpu_list() {
    printf '%s\n' "$1" | ${_TR} ',' '\n' \
        | sed 's/^[[:space:]]*//;s/[[:space:]]*$//' \
        | sed -n '/^[0-9][0-9]*$/p'
}

selected_gpu_ids() {
    if [ -n "${target_gpu_ids}" ] && [ "${target_gpu_ids}" != "all" ]; then
        _normalize_gpu_list "${target_gpu_ids}"
        return 0
    fi
    if [ "${target_gpu_ids}" = "all" ] \
        || [ "${OMNI_CI_ALLOW_UNSCOPED_GPU_CLEAN:-}" = "1" ] \
        || [ "${GITHUB_ACTIONS:-}" = "true" ]; then
        nvidia-smi --query-gpu=index --format=csv,noheader,nounits
        return 0
    fi
    echo "::error::delete_gpu_process.sh refuses unscoped cleanup. Set CUDA_VISIBLE_DEVICES to the physical GPU ids owned by this job (e.g. 0,1), or OMNI_CI_ALLOW_UNSCOPED_GPU_CLEAN=1 for single-tenant CI." >&2
    return 1
}

# note (Yue Yin): kill orphan processes that hold /dev/nvidia* fds but are
# invisible to nvidia-smi (e.g. multiprocessing.spawn workers after a crash).
_kill_orphan_gpu_processes() {
    # Do not global-pkill by pattern — concurrent calibration may share the host.
    # Orphans are removed only when they hold /dev/nvidiaN fds on selected GPUs.
    local pid cmdline gpu_regex fd_target gpu_id
    local -a gpu_list=()

    if [ -n "${target_gpu_ids}" ] && [ "${target_gpu_ids}" != "all" ]; then
        mapfile -t gpu_list < <(_normalize_gpu_list "${target_gpu_ids}")
        if [ "${#gpu_list[@]}" -eq 0 ]; then
            echo "::error::CUDA_VISIBLE_DEVICES='${target_gpu_ids}' produced empty GPU list; refusing unscoped orphan kill"
            return 1
        fi
        gpu_regex="$(printf '%s\n' "${gpu_list[@]}" | paste -sd'|' -)"
        for gpu_id in "${gpu_list[@]}"; do
            rm -f "/tmp/sglang_omni_gpu_${gpu_id}_startup.lock"
        done
        echo "  orphan kill scoped to physical GPU(s): ${gpu_list[*]} (regex=/dev/nvidia(${gpu_regex})\$)"
    else
        if [ "${target_gpu_ids}" != "all" ] \
            && [ "${OMNI_CI_ALLOW_UNSCOPED_GPU_CLEAN:-}" != "1" ] \
            && [ "${GITHUB_ACTIONS:-}" != "true" ]; then
            echo "::error::refusing unscoped orphan kill without CUDA_VISIBLE_DEVICES"
            return 1
        fi
        rm -f /tmp/sglang_omni_gpu_*_startup.lock
        gpu_regex=""
        echo "  orphan kill UNSCOPED (single-tenant / explicit allow)"
    fi

    for pid in $(ls /proc 2>/dev/null | grep -E '^[0-9]+$' || true); do
        # Never kill self / parent shell.
        if [ "${pid}" = "$$" ] || [ "${pid}" = "${PPID}" ]; then
            continue
        fi
        if [ -n "${gpu_regex}" ]; then
            fd_target="$(find "/proc/${pid}/fd" -maxdepth 1 -type l -printf '%l\n' 2>/dev/null || true)"
            # Match only /dev/nvidiaN for selected physical ids (not nvidiactl/uvm).
            if ! printf '%s\n' "${fd_target}" | grep -Eq "/dev/nvidia(${gpu_regex})$"; then
                continue
            fi
            # Extra guard for concurrent calibration: if the process explicitly
            # pinned a disjoint CUDA_VISIBLE_DEVICES set, do not kill it even
            # when it also holds a matching nvidia fd (driver/ctl edge cases).
            proc_cvd="$(${_TR} '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null \
                | sed -n 's/^CUDA_VISIBLE_DEVICES=//p' | head -n 1 || true)"
            if [ -n "${proc_cvd}" ] && [ "${proc_cvd}" != "all" ]; then
                overlap=0
                while IFS= read -r proc_gpu; do
                    [ -n "${proc_gpu}" ] || continue
                    for gpu_id in "${gpu_list[@]}"; do
                        if [ "${proc_gpu}" = "${gpu_id}" ]; then
                            overlap=1
                            break
                        fi
                    done
                    [ "${overlap}" = "1" ] && break
                done < <(_normalize_gpu_list "${proc_cvd}")
                if [ "${overlap}" = "0" ]; then
                    echo "  skip PID ${pid}: CVD=${proc_cvd} disjoint from cleanup scope ${gpu_list[*]}"
                    continue
                fi
            fi
        elif ! ls -l "/proc/${pid}/fd" 2>/dev/null | grep -q '/dev/nvidia[0-9]'; then
            continue
        fi
        cmdline="$(${_TR} '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null || true)"
        if [ -z "${cmdline}" ]; then
            continue
        fi
        # Never kill ephemeral precheck probes from a sibling calibration group.
        case "${cmdline}" in
            *"importlib.metadata"*|*"import sglang"*|*"print(sglang.__version__)"*|*"m.version("*)
                echo "  skip PID ${pid}: version-probe cmdline"
                continue
                ;;
        esac
        echo "  killing orphan GPU PID ${pid}: ${cmdline}"
        kill -9 "${pid}" 2>/dev/null || true
    done
}

kill_orphans=0
for arg in "$@"; do
    [ "${arg}" = "--kill-orphans" ] && kill_orphans=1
done

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; skipping GPU cleanup."
    exit 0
fi

# Resolve and validate scope up front (also used by the wait loop).
if ! mapfile -t _SELECTED_GPUS < <(selected_gpu_ids); then
    exit 2
fi
if [ "${#_SELECTED_GPUS[@]}" -eq 0 ]; then
    echo "::error::no GPUs selected for cleanup"
    exit 2
fi
echo "=== GPU cleanup scope (physical): ${_SELECTED_GPUS[*]} (input CVD='${_raw_cvd}') ==="

if [ "${kill_orphans}" = "1" ]; then
    _kill_orphan_gpu_processes || exit 2
    sleep 2
fi

echo "=== Checking GPU Utilization ==="

for gpu_index in "${_SELECTED_GPUS[@]}"; do
    gpu_index=$(printf '%s' "$gpu_index" | ${_TR} -d ' ')
    [ -n "${gpu_index}" ] || continue
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="${gpu_index}" 2>/dev/null || true)
    if [ -z "$pids" ]; then
        echo "  No processes found on GPU $gpu_index."
    else
        echo "  Killing processes on GPU $gpu_index: $pids"
        for pid in $pids; do
            pid=$(printf '%s' "$pid" | ${_TR} -d ' ')
            [ -n "${pid}" ] || continue
            echo "  Killing PID $pid..."
            kill -9 "$pid" || true
        done
    fi
done

if ! [[ "${memory_threshold_mb}" =~ ^[0-9]+$ ]] || [ "${memory_threshold_mb}" -lt 1 ]; then
    echo "::error::OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB must be a positive integer; got '${memory_threshold_mb}'"
    exit 2
fi

if ! [[ "${wait_timeout_seconds}" =~ ^[0-9]+$ ]] || ! [[ "${poll_seconds}" =~ ^[0-9]+$ ]] || [ "${poll_seconds}" -lt 1 ]; then
    echo "::error::OMNI_CI_GPU_CLEAN_WAIT_SECONDS and OMNI_CI_GPU_CLEAN_POLL_SECONDS must be non-negative integers, with poll >= 1"
    exit 2
fi

echo "Waiting for selected GPU memory.used to drop below ${memory_threshold_mb} MiB..."
deadline=$((SECONDS + wait_timeout_seconds))
while true; do
    max_used_mb=0
    for gpu_index in "${_SELECTED_GPUS[@]}"; do
        gpu_index=$(printf '%s' "$gpu_index" | ${_TR} -d ' ')
        [ -n "${gpu_index}" ] || continue
        used_mb=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits --id="${gpu_index}" | head -n 1 | ${_TR} -d ' ')
        if [ -z "${used_mb}" ]; then
            continue
        fi
        if [ "${used_mb}" -gt "${max_used_mb}" ]; then
            max_used_mb="${used_mb}"
        fi
        echo "  GPU ${gpu_index}: ${used_mb} MiB used"
    done

    if [ "${max_used_mb}" -lt "${memory_threshold_mb}" ]; then
        echo "GPU memory cleanup complete: max memory.used=${max_used_mb} MiB."
        break
    fi

    if [ "${SECONDS}" -ge "${deadline}" ]; then
        echo "::error::Timed out waiting for GPU memory.used < ${memory_threshold_mb} MiB; max memory.used=${max_used_mb} MiB."
        exit 1
    fi

    sleep "${poll_seconds}"
done

echo ""
