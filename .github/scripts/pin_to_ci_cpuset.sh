# Pin the sourcing shell (and children) to the runner lane's OMNI_CI_CPUSET;
# no-op when unset (forks, local runs).
# note (Jiaxin Deng): docker container options cannot read the runner-local
# .env value, so the reachable layer is the script shell, not --cpuset-cpus.
if [ -n "${OMNI_CI_CPUSET:-}" ]; then
  if ! taskset -pc "${OMNI_CI_CPUSET}" $$ > /dev/null; then
    echo "::error::failed to pin to OMNI_CI_CPUSET='${OMNI_CI_CPUSET}'"
    exit 2
  fi
  # note (Jiaxin Deng): readback catches the kernel silently narrowing the
  # mask (e.g. offline CPUs in the lane spec), mirroring the perf fixture.
  python3 - "$$" "${OMNI_CI_CPUSET}" <<'PYTHON' || exit 2
import os
import sys

requested = set()
for part in sys.argv[2].split(","):
    first, _, last = part.partition("-")
    requested.update(range(int(first), int(last or first) + 1))
effective = os.sched_getaffinity(int(sys.argv[1]))
if effective != requested:
    sys.exit(f"::error::requested CPUs {sorted(requested)}, got {sorted(effective)}")
PYTHON
fi
