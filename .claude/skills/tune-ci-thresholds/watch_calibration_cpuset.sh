#!/bin/bash
# Tab C: foreign-load supervisor for a calibration cpuset.
# usage: watch_calibration_cpuset.sh <cpuset-spec> [interval-s]
# One line per interval with the cpuset's total busy fraction. During the
# idle gap between rounds any busy is foreign; during a round the pytest
# session's own sampler separates foreign load in its end-of-run summary.
set -u
SPEC="${1:?usage: watch_calibration_cpuset.sh <cpuset-spec> [interval-s]}"
INTERVAL="${2:-10}"
exec python3 - "$SPEC" "$INTERVAL" <<'PY'
import sys, time

spec, interval = sys.argv[1], float(sys.argv[2])
cpus = set()
for part in spec.split(","):
    lo, _, hi = part.partition("-")
    cpus.update(range(int(lo), int(hi or lo) + 1))


def snap():
    vals = {}
    with open("/proc/stat") as f:
        for line in f:
            if line.startswith("cpu") and line[3:4].isdigit():
                p = line.split()
                idx = int(p[0][3:])
                if idx in cpus:
                    n = list(map(int, p[1:]))
                    vals[idx] = (sum(n), n[3] + n[4])
    return vals


prev = snap()
while True:
    time.sleep(interval)
    cur = snap()
    tot = sum(cur[i][0] - prev[i][0] for i in cur if i in prev)
    idle = sum(cur[i][1] - prev[i][1] for i in cur if i in prev)
    busy = 1 - idle / tot if tot else 0.0
    flag = "  <-- BUSY" if busy > 0.2 else ""
    print(f"[{time.strftime('%H:%M:%S')}] cpuset {spec} busy {busy:5.1%}{flag}",
          flush=True)
    prev = cur
PY
