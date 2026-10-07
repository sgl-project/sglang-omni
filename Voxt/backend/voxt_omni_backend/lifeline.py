# SPDX-License-Identifier: Apache-2.0
"""Runs the server so that it cannot outlive the supervisor that started it.

    python -m voxt_omni_backend.lifeline -- <server argv...>

The supervisor starts this module as the leader of a new process group. It runs
the server as a child in that group and waits for it. A kqueue watch on the
supervisor fires when the supervisor exits for any reason, including SIGKILL,
and the whole group is then killed: the server, its stage processes and any
helper they started, even ones already reparented away from the server.
"""

from __future__ import annotations

import os
import select
import signal
import subprocess
import sys
import threading
from types import FrameType

SUPERVISOR_PID_ENV = "VOXT_OMNI_SUPERVISOR_PID"
# Mirrors the supervisor's stop signals.
STOP_SIGNALS = (signal.SIGTERM, signal.SIGINT, signal.SIGHUP, signal.SIGQUIT)


def kill_group_when_parent_exits(parent_pid: int) -> None:
    queue = select.kqueue()
    watch = select.kevent(
        parent_pid,
        filter=select.KQ_FILTER_PROC,
        flags=select.KQ_EV_ADD,
        fflags=select.KQ_NOTE_EXIT,
    )
    try:
        queue.control([watch], 0, None)
    except OSError:
        os.killpg(os.getpgrp(), signal.SIGKILL)
        return
    if os.getppid() != parent_pid:
        os.killpg(os.getpgrp(), signal.SIGKILL)
    else:
        queue.control(None, 1, None)
        os.killpg(os.getpgrp(), signal.SIGKILL)


def main() -> int:
    if len(sys.argv) < 3 or sys.argv[1] != "--":
        sys.stderr.write(__doc__ or "")
        return 2
    else:
        pass
    supervisor_pid = int(os.environ.get(SUPERVISOR_PID_ENV, os.getppid()))
    threading.Thread(
        target=kill_group_when_parent_exits, args=(supervisor_pid,), daemon=True
    ).start()
    server: subprocess.Popen[bytes] | None = None
    early: list[int] = []

    def forward(signal_number: int, frame: FrameType | None) -> None:
        del frame
        if server is None:
            early.append(signal_number)
        else:
            server.send_signal(signal_number)

    for stop_signal in STOP_SIGNALS:
        signal.signal(stop_signal, forward)
    # The supervisor blocks these while it spawns us, and a blocked mask survives
    # exec: unblocked here, so the server starts able to shut down gracefully.
    signal.pthread_sigmask(signal.SIG_UNBLOCK, STOP_SIGNALS)
    server = subprocess.Popen(sys.argv[2:], stdin=subprocess.DEVNULL)
    for signal_number in early:
        server.send_signal(signal_number)
    return server.wait()


if __name__ == "__main__":
    sys.exit(main())
