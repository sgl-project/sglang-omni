#!/usr/bin/env bash
# Confine interactive sessions to the CI host's system cores.
#
# IDE servers and coding agents spawned from SSH sessions (cursor-server,
# codex, their node trees) run with full-machine affinity, so the kernel
# schedules them onto runner-lane cores. Measured on the CI host: three such
# trees held a sustained 2.5 to 3.0 foreign cores on one lane's cpuset,
# which trips the calibration contention gate (2.0 cores) and pollutes any
# perf gate running on that lane. pin_to_ci_cpuset.sh covers CI's own
# processes; this covers everything interactive.
#
# Run once as root: sudo bash confine_interactive_cpuset.sh [ALLOWED_CPUS]
# Default matches the system-core set already used for categraf/dockerd.
#
# note (Jiaxin Deng): user.slice is the systemd cgroup holding every login
# session (any user, including root SSH). CI job containers and calibration
# containers live under system.slice, so they keep their lane masks. The
# property applies to running members immediately and persists for future
# sessions, which per-PID taskset sweeps cannot do.
set -euo pipefail

ALLOWED_CPUS="${1:-0,1,64,65}"

[ "$(id -u)" -eq 0 ] || { echo "error: must run as root" >&2; exit 1; }
command -v systemctl >/dev/null || { echo "error: systemd required" >&2; exit 1; }
[ -f /sys/fs/cgroup/cgroup.controllers ] || {
    echo "error: cgroup v2 required (AllowedCPUs is a cgroup v2 property)" >&2
    exit 1
}

systemctl set-property user.slice AllowedCPUs="$ALLOWED_CPUS"

effective="$(cat /sys/fs/cgroup/user.slice/cpuset.cpus.effective)"
echo "user.slice cpuset.cpus.effective: $effective"

# note (Jiaxin Deng): readback catches the kernel dropping offline CPUs from
# the request, mirroring pin_to_ci_cpuset.sh; an empty effective set would
# starve every login shell, so fail loudly instead.
[ -n "$effective" ] || { echo "error: effective cpuset is empty" >&2; exit 1; }

echo "done: interactive sessions confined to $ALLOWED_CPUS (persistent)"
