set -uo pipefail

memory_threshold_mb="${OMNI_CI_GPU_MEMORY_CLEAN_THRESHOLD_MB:-1024}"
wait_timeout_seconds="${OMNI_CI_GPU_CLEAN_WAIT_SECONDS:-120}"
poll_seconds="${OMNI_CI_GPU_CLEAN_POLL_SECONDS:-5}"

if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; skipping GPU cleanup."
    exit 0
fi

echo "=== Checking GPU Utilization ==="

# Get GPU indices and their utilization
nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits | while IFS=',' read -r gpu_index utilization; do
    gpu_index=$(echo "$gpu_index" | tr -d ' ')
    utilization=$(echo "$utilization" | tr -d ' ')

    # Get PIDs running on this GPU
    pids=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader --id="$gpu_index")
    if [ -z "$pids" ]; then
        echo "  No processes found on GPU $gpu_index."
    else
        echo "  Killing processes on GPU $gpu_index: $pids"
        for pid in $pids; do
            pid=$(echo "$pid" | tr -d ' ')
            echo "  Killing PID $pid..."
            # kill -9 "$pid" && echo "  PID $pid killed." || echo "  Failed to kill PID $pid (may need sudo)."
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

echo "Waiting for every GPU memory.used to drop below ${memory_threshold_mb} MiB..."
deadline=$((SECONDS + wait_timeout_seconds))
while true; do
    max_used_mb=0
    while IFS=',' read -r gpu_index used_mb; do
        gpu_index=$(echo "$gpu_index" | tr -d ' ')
        used_mb=$(echo "$used_mb" | tr -d ' ')
        if [ -z "${used_mb}" ]; then
            continue
        fi
        if [ "${used_mb}" -gt "${max_used_mb}" ]; then
            max_used_mb="${used_mb}"
        fi
        echo "  GPU ${gpu_index}: ${used_mb} MiB used"
    done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)

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
