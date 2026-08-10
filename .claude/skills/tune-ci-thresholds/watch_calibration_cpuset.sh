#!/bin/bash
# Tab C: foreign-load supervisor for one calibration lane's cpuset.
# usage: watch_calibration_cpuset.sh <gpu-group|cpuset-spec> [interval-s]
#
# Pass the same GPU pair as TUNE_GPU_INCLUDE (e.g. 0,1 / 2,3 / 4,5 / 6,7);
# the script resolves that lane's 32-core cpuset from hosts/*.yaml. A raw
# cpulist (e.g. 16-31,80-95) is also accepted.
#
# Operator-visible only. tune.py enforces the policy for whichever GPUs
# that process owns:
#   - precheck refuses a busy cpuset
#   - live monitor aborts a contaminated stage attempt, discards its
#     artifacts, waits for CPU+GPU recovery, then retries (run continues)
# One line per interval with the cpuset's total busy fraction.
set -u
ARG="${1:?usage: watch_calibration_cpuset.sh <gpu-group|cpuset-spec> [interval-s]}"
INTERVAL="${2:-10}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
HOST_YAML="${TUNE_HOST_YAML:-$SCRIPT_DIR/hosts/${TUNE_HOST:-sglang-h100-ci}.yaml}"
exec python3 - "$ARG" "$INTERVAL" "$HOST_YAML" <<'PY'
import sys, time
from pathlib import Path

arg, interval, host_yaml = sys.argv[1], float(sys.argv[2]), Path(sys.argv[3])

# Built-in lane table (kept in sync with hosts/sglang-h100-ci.yaml).
DEFAULT_GPU_GROUP_CPUSETS = {
    "0,1": "0-15,64-79",
    "2,3": "16-31,80-95",
    "4,5": "48-63,112-127",
    "6,7": "32-47,96-111",
}


def load_table(path: Path) -> dict[str, str]:
    table = dict(DEFAULT_GPU_GROUP_CPUSETS)
    if not path.is_file():
        return table
    # Minimal YAML subset: lines like `  "0,1": "0-15,64-79"`.
    in_block = False
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith("gpu_group_cpusets:"):
            in_block = True
            continue
        if in_block:
            if not stripped or stripped.startswith("#"):
                continue
            if not line.startswith(" ") and not line.startswith("\t"):
                break
            if ":" not in stripped:
                continue
            key, _, val = stripped.partition(":")
            key = key.strip().strip("\"'")
            val = val.strip().strip("\"'")
            if key and val:
                table[key] = val
    return table


def resolve_spec(raw: str, table: dict[str, str]) -> tuple[str, str | None]:
    """Return (cpuset_spec, gpu_group_or_None)."""
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    # GPU group: every token is an integer index (e.g. 2,3).
    if parts and all(p.isdigit() for p in parts):
        key = ",".join(str(int(p)) for p in sorted(parts, key=int))
        if key in table:
            return table[key], key
        # Single GPU: inherit the lane that owns it.
        picked = {int(p) for p in parts}
        for group_key, cpuset in table.items():
            group = {int(x) for x in group_key.split(",") if x.strip()}
            if picked and picked.issubset(group):
                return cpuset, group_key
        raise SystemExit(
            f"no cpuset for GPU group {key!r}; known lanes: "
            + ", ".join(sorted(table))
        )
    # Otherwise treat the argument as a raw cpulist.
    return raw, None


def parse_cpus(spec: str) -> set[int]:
    cpus: set[int] = set()
    for part in spec.split(","):
        lo, _, hi = part.partition("-")
        cpus.update(range(int(lo), int(hi or lo) + 1))
    if not cpus:
        raise SystemExit(f"empty cpuset {spec!r}")
    return cpus


table = load_table(host_yaml)
spec, gpu_group = resolve_spec(arg, table)
cpus = parse_cpus(spec)
label = f"gpus={gpu_group} cpuset={spec}" if gpu_group else f"cpuset={spec}"
print(f"[Tab C] watching {label} (interval={interval}s)", flush=True)


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
    print(f"[{time.strftime('%H:%M:%S')}] {label} busy {busy:5.1%}{flag}",
          flush=True)
    prev = cur
PY
