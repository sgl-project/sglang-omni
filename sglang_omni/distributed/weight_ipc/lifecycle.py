# SPDX-License-Identifier: Apache-2.0
"""Leader liveness helpers for weight IPC followers."""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


def _is_zombie(pid: int) -> bool:
    try:
        stat_line = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
    except OSError:
        return False

    # Note (guozhihao): /proc/<pid>/stat stores the process state after (comm);
    # Z means zombie, which can still pass kill(pid, 0) until it is reaped.
    closing_paren = stat_line.rfind(")")
    return (
        closing_paren >= 0
        and len(stat_line) > closing_paren + 2
        and stat_line[closing_paren + 2] == "Z"
    )


def pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return not _is_zombie(pid)


class LeaderLivenessMonitor:
    """Background poller that fail-fast exits when the leader PID disappears."""

    def __init__(
        self,
        leader_pid: int,
        *,
        poll_interval_s: float = 1.0,
        exit_code: int = 70,
    ) -> None:
        self.leader_pid = leader_pid
        self.poll_interval_s = poll_interval_s
        self.exit_code = exit_code
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(
            target=self._run,
            name="weight-ipc-leader-liveness",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None

    def _run(self) -> None:
        while not self._stop.wait(self.poll_interval_s):
            if pid_is_alive(self.leader_pid):
                continue
            logger.critical(
                "weight IPC leader pid=%s exited; terminating follower",
                self.leader_pid,
            )
            # Note (guozhihao): mapped CUDA IPC storage is undefined after owner exit.
            os._exit(self.exit_code)


def wait_leader_death(leader_pid: int, timeout_s: float) -> bool:
    """Test helper: return True if leader dies within timeout."""
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if not pid_is_alive(leader_pid):
            return True
        time.sleep(0.05)
    return False
