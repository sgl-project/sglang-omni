#!/usr/bin/env bash
# Run one PR's CPU partitions inside one self-hosted runner job.
set -euo pipefail
count=${1:?expected shard count}
shift
if [[ $# == 0 ]]; then set -- tests/; fi
workspace=${GITHUB_WORKSPACE:?}
logs="${RUNNER_TEMP:?}/omni-cpu-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
mkdir -p "$logs"
prefix=$(basename "$(mktemp -d "$logs/run-XXXXXXXX")")
prefix="omni-cpu-$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT-$prefix"
label="omni.cpu-job=$GITHUB_RUN_ID-$GITHUB_RUN_ATTEMPT"
image=hongccc/sglang-omni@sha256:d2fbd80ec42c8a6d9dff4b8a4fdbee35d50c5a1b41aa8dc872925da6e0c82f6a

masks=$(python3 -I - "$count" <<'PY'
import os
import sys
from pathlib import Path

def cpus(spec):
    result = set()
    for part in spec.split(","):
        start, _, stop = part.partition("-")
        result.update(range(int(start), int(stop or start) + 1))
    return result

count = int(sys.argv[1])
if count not in (3, 6):
    sys.exit("Use 3 or 6 CPU partitions")
allowed = os.sched_getaffinity(0)
if os.getenv("OMNI_CI_CPUSET"):
    allowed &= cpus(os.environ["OMNI_CI_CPUSET"])
cores = {}
for cpu in sorted(allowed):
    topology = Path(f"/sys/devices/system/cpu/cpu{cpu}/topology")
    siblings = (topology / "thread_siblings_list").read_text().strip()
    cores.setdefault(siblings, []).append(cpu)
width = min(2, len(cores) // count)
if not width:
    sys.exit(f"Only {len(cores)} physical cores available for {count} shards")
groups = list(cores.values())
for shard in range(count):
    mask = groups[shard * width:(shard + 1) * width]
    print(",".join(str(cpu) for core in mask for cpu in core[:2]))
PY
)
mapfile -t masks <<< "$masks"
names=()
collector="$(cd "$(dirname "$0")" && pwd)/collect_cpu_shards.sh"
trap 'rc=$?; bash "$collector" || rc=1; exit "$rc"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

# note (0xtoward): some rootless runners do not delegate the cpuset controller.
container_cpuset=false
if docker run --rm --name "$prefix-probe" --label "$label" --label omni.cpu-shard=probe --cpuset-cpus "${masks[0]}" \
    --env NVIDIA_VISIBLE_DEVICES=void --entrypoint python3 "$image" \
    -I -c 'import os, sys; sys.exit(os.sched_getaffinity(0) != set(map(int, sys.argv[1].split(","))))' "${masks[0]}"; then
    container_cpuset=true
else
    docker rm -f "$prefix-probe" >/dev/null 2>&1 || true
    echo "Container cpuset unavailable; pinning the entry process with taskset."
fi

for ((i=0; i<count; i++)); do
    name="$prefix-$i"
    names+=("$name")
    limits=()
    if [[ $container_cpuset == true ]]; then
        limits=(--cpuset-cpus "${masks[i]}")
    fi
    docker create "${limits[@]}" --name "$name" --label "$label" --label "omni.cpu-shard=$i" \
        --volume "$workspace:/ci-source:ro" --volume /data/omni-ci:/data/omni-ci \
        --volume /data/cache/huggingface:/github/home/.cache/huggingface \
        --volume /data/cache/huggingface:/root/.cache/huggingface --volume /dev/shm:/dev/shm \
        --env NVIDIA_VISIBLE_DEVICES=void --env CUDA_VISIBLE_DEVICES= --env OMNI_CI_HOME \
        --env "OMNI_CI_CPUSET=${masks[i]}" --env GITHUB_ACTIONS --env GITHUB_SHA --env GITHUB_REPOSITORY \
        --entrypoint taskset "$image" -c "${masks[i]}" bash /ci-source/.github/scripts/run_cpu_shard.sh "$i" "$count" "$@" >/dev/null
    echo "shard $i CPUs ${masks[i]} container $name"
done
docker start "${names[@]}"
result=0
for i in "${!names[@]}"; do
    docker wait "${names[i]}" > "$logs/.wait-$i" &
    wait "$!"
    rc=$(<"$logs/.wait-$i")
    echo "shard $i exited $rc"
    if [[ $rc != 0 ]]; then
        docker logs --tail 60 "${names[i]}"
        result=1
    fi
done
exit "$result"
