#!/usr/bin/env bash
set -euo pipefail

label="omni.cpu-job=${GITHUB_RUN_ID:?}-${GITHUB_RUN_ATTEMPT:?}"
logs="${RUNNER_TEMP:?}/omni-cpu-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir -p "$logs"
# note (0xtoward): the Actions cleanup step can overlap the interrupted launcher's exit trap.
exec 9>"$logs/.collect.lock"
flock 9
containers=$(docker ps -aq --filter "label=$label")

collect() {
    local container=$1 shard state code
    shard=$(docker inspect --format '{{index .Config.Labels "omni.cpu-shard"}}' "$container")
    if [[ $shard == probe ]]; then
        docker rm -f "$container" >/dev/null
        return
    fi
    [[ $shard =~ ^[0-5]$ ]] || return 1
    state=$(docker inspect --format '{{.State.Status}}' "$container")
    case "$state" in
        running|restarting|paused)
            docker kill "$container" >/dev/null || {
                [[ $(docker inspect --format '{{.State.Running}}' "$container") == false ]] || return 1
            }
            docker wait "$container" >/dev/null
            ;;
    esac
    state=$(docker inspect --format '{{.State.Status}}' "$container")
    [[ -n $state ]] || return 1
    if [[ $state == exited ]]; then
        code=$(docker inspect --format '{{.State.ExitCode}}' "$container")
        [[ $code =~ ^[0-9]+$ ]] || return 1
        printf '%s\n' "$code" > "$logs/.$shard.exit"
        mv "$logs/.$shard.exit" "$logs/$shard.exit"
    fi
    docker logs "$container" > "$logs/.$shard.log" 2>&1
    mv "$logs/.$shard.log" "$logs/$shard.log"
    printf '%s\n' "$state" > "$logs/.$shard.state"
    mv "$logs/.$shard.state" "$logs/$shard.state"
    docker rm -f "$container" >/dev/null
}

pids=()
for container in $containers; do
    collect "$container" &
    pids+=("$!")
done
rc=0
for pid in "${pids[@]}"; do
    wait "$pid" || rc=1
done
exit "$rc"
