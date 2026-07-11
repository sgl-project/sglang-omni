# SPDX-License-Identifier: Apache-2.0
"""Subprocess stdout line-waiting helpers shared by integration and unit tests.

Kept stdlib-only so a lightweight unit test can import them without pulling in
heavy model/CI dependencies.
"""

from __future__ import annotations

import queue
import subprocess
import threading
import time


def _wait_for(predicate, timeout: float, interval: float = 2.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return False


def _wait_for_process_line(
    proc: subprocess.Popen,
    marker: str,
    timeout: float,
) -> str:
    assert proc.stdout is not None
    line_queue = getattr(proc, "_omni_stdout_line_queue", None)
    if line_queue is None:
        line_queue = queue.Queue()

        def _read_stdout() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                line_queue.put(line)
            line_queue.put(None)

        threading.Thread(target=_read_stdout, daemon=True).start()
        setattr(proc, "_omni_stdout_line_queue", line_queue)

    deadline = time.time() + timeout
    output: list[str] = []
    while time.time() < deadline:
        remaining = max(0.0, deadline - time.time())
        try:
            line = line_queue.get(timeout=min(0.2, remaining))
        except queue.Empty:
            if proc.poll() is not None:
                raise AssertionError(
                    f"trainer exited with {proc.returncode} while waiting for {marker}: "
                    + "".join(output)
                )
            continue
        if line is None:
            raise AssertionError(
                f"trainer exited with {proc.returncode} while waiting for {marker}: "
                + "".join(output)
            )
        output.append(line)
        if marker in line:
            return line
    raise AssertionError(f"timed out waiting for trainer marker {marker}")
