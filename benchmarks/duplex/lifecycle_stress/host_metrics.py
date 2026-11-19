# SPDX-License-Identifier: Apache-2.0
"""Resident memory of a process, read without extra dependencies."""

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
