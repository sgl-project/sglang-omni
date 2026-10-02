# SPDX-License-Identifier: Apache-2.0
"""Own one backend process group, including workers left by a crashed parent."""

import os
import signal
import subprocess
import sys
import time


def signal_group(group, signum):
    try:
        os.killpg(group, signum)
        return True
    except ProcessLookupError:
        return False


def run(command, *, shutdown_timeout=30, descendant_timeout=5):
    requested_signal = None

    def request_stop(signum, _frame):
        nonlocal requested_signal
        requested_signal = signum

    previous = {
        signum: signal.signal(signum, request_stop)
        for signum in (signal.SIGINT, signal.SIGTERM)
    }
    process = None
    try:
        if requested_signal is not None:
            return 128 + requested_signal
        # The backend and its workers cannot share the launcher's or another
        # service's process group. No shell, process-name matching or PID scan.
        process = subprocess.Popen(command, start_new_session=True)
        while process.poll() is None and requested_signal is None:
            time.sleep(0.1)
    finally:
        try:
            if process is not None:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=shutdown_timeout)
                    except subprocess.TimeoutExpired:
                        pass
                # Also run after an unexpected parent exit. Descendants still
                # belong to the captured group even after being reparented.
                if signal_group(process.pid, signal.SIGTERM):
                    deadline = time.monotonic() + descendant_timeout
                    while signal_group(process.pid, 0) and time.monotonic() < deadline:
                        time.sleep(0.05)
                    if signal_group(process.pid, 0):
                        signal_group(process.pid, signal.SIGKILL)
                process.wait()
        finally:
            for signum, handler in previous.items():
                signal.signal(signum, handler)
    if requested_signal is not None:
        return 128 + requested_signal
    return process.returncode if process.returncode >= 0 else 128 - process.returncode


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("Usage: service_process.py COMMAND [ARG ...]")
    raise SystemExit(run(sys.argv[1:]))
