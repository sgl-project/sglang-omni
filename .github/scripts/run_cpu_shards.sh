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

masks=$(python3 - "$count" <<'PY'
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
cleanup() {
    if (( ${#names[@]} )); then
        docker rm -f "${names[@]}" >/dev/null || echo "::warning::check CPU container cleanup"
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

for ((i=0; i<count; i++)); do
    name="$prefix-$i"
    names+=("$name")
    docker create --name "$name" --label "$label" \
        --volume "$workspace:/ci-source:ro" --volume /data/omni-ci:/data/omni-ci \
        --volume /data/cache/huggingface:/github/home/.cache/huggingface \
        --volume /data/cache/huggingface:/root/.cache/huggingface --volume /dev/shm:/dev/shm \
        --env NVIDIA_VISIBLE_DEVICES --env CUDA_VISIBLE_DEVICES= --env OMNI_CI_HOME \
        --env "OMNI_CI_CPUSET=${masks[i]}" --env GITHUB_ACTIONS --env GITHUB_SHA \
        --workdir "/__w/${GITHUB_REPOSITORY##*/}/${GITHUB_REPOSITORY##*/}" \
        --entrypoint bash "$image" /ci-source/.github/scripts/run_cpu_shard.sh "$i" "$count" "$@" >/dev/null
    echo "shard $i CPUs ${masks[i]} container $name"
done
docker start "${names[@]}"
result=0
for i in "${!names[@]}"; do
    docker wait "${names[i]}" > "$logs/$i.exit" &
    wait "$!"
    rc=$(<"$logs/$i.exit")
    docker logs "${names[i]}" > "$logs/$i.log" 2>&1
    echo "shard $i exited $rc"
    if [[ $rc != 0 ]]; then
        tail -n 60 "$logs/$i.log"
        result=1
    fi
done
exit "$result"
