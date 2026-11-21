# SPDX-License-Identifier: Apache-2.0
"""Resident memory, threads and open files of a process, and the models and memory of this host's GPUs, read without extra dependencies."""

from __future__ import annotations

import os
import platform
import resource
import subprocess
from pathlib import Path

from benchmarks.duplex.lifecycle_stress.records import ServerSnapshot

BYTES_PER_MEGABYTE = 1024 * 1024
# note (Yi Zhao): ps prints the resident set in kilobytes on both Linux and macOS.
KILOBYTES_PER_MEGABYTE = 1024
PROC_ROOT = Path("/proc")
PS_TIMEOUT_S = 5.0
THREADS_KEY = "Threads:"
PROCESS_LIST_COMMAND = ("ps", "-A", "-o", "pid=", "-o", "ppid=")
# note (Yi Zhao): without units nvidia-smi prints one number of mebibytes per GPU, one GPU per line.
NVIDIA_SMI_COMMAND = (
    "nvidia-smi",
    "--query-gpu=memory.used",
    "--format=csv,noheader,nounits",
)
NVIDIA_SMI_TIMEOUT_S = 10.0
NVIDIA_SMI_MODELS_COMMAND = (
    "nvidia-smi",
    "--query-gpu=name,memory.total",
    "--format=csv,noheader",
)


def read_rss_megabytes(pid: int) -> float | None:
    """Current resident set of a process; None when the process is gone."""
    statm_path = PROC_ROOT / str(pid) / "statm"
    if statm_path.exists():
        resident_pages = int(statm_path.read_text().split()[1])
        return resident_pages * os.sysconf("SC_PAGE_SIZE") / BYTES_PER_MEGABYTE
    else:
        completed = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True,
            text=True,
            timeout=PS_TIMEOUT_S,
            check=False,
        )
        resident_text = completed.stdout.strip()
        if completed.returncode != 0 or not resident_text:
            return None
        else:
            return int(resident_text) / KILOBYTES_PER_MEGABYTE


def read_thread_count(pid: int) -> int | None:
    """Threads of a process as Linux reports them; None where there is no /proc or the process is gone."""
    status_path = PROC_ROOT / str(pid) / "status"
    if status_path.exists():
        thread_lines = [
            line
            for line in status_path.read_text().splitlines()
            if line.startswith(THREADS_KEY)
        ]
        return int(thread_lines[0].split()[1]) if thread_lines else None
    else:
        return None


def read_open_files(pid: int) -> int | None:
    """Open file descriptors of a process, its sockets among them, so a connection that is never closed shows here; None where they cannot be listed."""
    descriptor_directory = PROC_ROOT / str(pid) / "fd"
    try:
        return len(list(descriptor_directory.iterdir()))
    except (FileNotFoundError, PermissionError):
        # note (Yi Zhao): no /proc on this system, or the process belongs to another user.
        return None


def process_tree(pid: int) -> list[int]:
    """The process and every descendant alive now; a model server runs its stages in child processes, and most of its memory is theirs."""
    completed = subprocess.run(
        PROCESS_LIST_COMMAND,
        capture_output=True,
        text=True,
        timeout=PS_TIMEOUT_S,
        check=True,
    )
    children: dict[int, list[int]] = {}
    for line in completed.stdout.splitlines():
        child, parent = (int(field) for field in line.split())
        children.setdefault(parent, []).append(child)
    tree = [pid]
    for parent in tree:
        tree.extend(children.get(parent, []))
    return tree


def read_process_tree(pid: int) -> ServerSnapshot:
    """Resident memory, threads and open files of a server process and its descendants together; a figure no process gives stays empty."""
    pids = process_tree(pid)
    resident = [
        megabytes
        for megabytes in map(read_rss_megabytes, pids)
        if megabytes is not None
    ]
    threads = [count for count in map(read_thread_count, pids) if count is not None]
    files = [count for count in map(read_open_files, pids) if count is not None]
    return ServerSnapshot(
        process_count=len(pids),
        rss_megabytes=sum(resident) if resident else None,
        thread_count=sum(threads) if threads else None,
        open_files=sum(files) if files else None,
    )


def peak_rss_megabytes() -> float:
    """Peak resident set of this process since it started."""
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if platform.system() == "Darwin":
        return peak / BYTES_PER_MEGABYTE
    else:
        return peak / KILOBYTES_PER_MEGABYTE


def read_device_megabytes() -> float:
    """Memory in use on all GPUs of this host together; a failing nvidia-smi raises, because the caller asked for this reading."""
    # note (Yi Zhao): this is what every process holds on the device, cached and reserved memory included, so a session that leaks may not move it.
    completed = subprocess.run(
        NVIDIA_SMI_COMMAND,
        capture_output=True,
        text=True,
        timeout=NVIDIA_SMI_TIMEOUT_S,
        check=True,
    )
    return sum(float(line) for line in completed.stdout.split())


def read_device_models() -> list[str]:
    """Model name and total memory of every GPU of this host, one entry per GPU as nvidia-smi prints it."""
    completed = subprocess.run(
        NVIDIA_SMI_MODELS_COMMAND,
        capture_output=True,
        text=True,
        timeout=NVIDIA_SMI_TIMEOUT_S,
        check=True,
    )
    return completed.stdout.strip().splitlines()
