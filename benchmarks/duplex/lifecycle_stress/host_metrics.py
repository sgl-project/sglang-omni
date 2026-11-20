# SPDX-License-Identifier: Apache-2.0
"""Resident memory of a process, and the models and memory of this host's GPUs, read without extra dependencies."""

from __future__ import annotations

import os
import platform
import resource
import subprocess
from pathlib import Path

BYTES_PER_MEGABYTE = 1024 * 1024
# note (Yi Zhao): ps prints the resident set in kilobytes on both Linux and macOS.
KILOBYTES_PER_MEGABYTE = 1024
PROC_ROOT = Path("/proc")
PS_TIMEOUT_S = 5.0
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
