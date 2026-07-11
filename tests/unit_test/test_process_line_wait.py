# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import subprocess
import sys

from tests.unit_test._util.process import _wait_for_process_line


def test_wait_for_process_line_keeps_prefetched_stdout_lines():
    proc = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import time; print('first', flush=True); print('second', flush=True); time.sleep(10)",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        assert "first" in _wait_for_process_line(proc, "first", timeout=2)
        assert "second" in _wait_for_process_line(proc, "second", timeout=2)
    finally:
        proc.kill()
        proc.wait(timeout=5)
