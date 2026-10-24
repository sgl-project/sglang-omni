# SPDX-License-Identifier: Apache-2.0
"""Process exit helpers."""

from __future__ import annotations

import atexit
import os
import sys
import threading
import traceback
from typing import NoReturn


def exit_without_teardown(code: int) -> NoReturn:
    """End a process whose work and cleanup are done, minus interpreter teardown.

    Runs what a normal exit runs first: it waits for non-daemon threads as the
    interpreter does at shutdown, runs the atexit callbacks (including those of
    logging and multiprocessing), releases the shared memory file torch keeps
    for CUDA tensors sent to other processes, and flushes the standard streams.
    Only the teardown of every module and object is skipped, which in a process
    holding a model takes seconds, while the operating system and the device
    driver release the process's memory at once.
    """
    # The standard library has no public spelling of these three steps.
    try:
        threading._shutdown()  # noqa: leading-underscore
    except BaseException:
        # A normal exit reports a failing thread exit callback and keeps its
        # exit code, as atexit._run_exitfuncs does for an atexit callback.
        print("Exception ignored in thread shutdown:", file=sys.stderr)
        traceback.print_exc()
    atexit._run_exitfuncs()  # noqa: leading-underscore
    release_cuda_ipc_file()
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.flush()
        except Exception:
            pass
    os._exit(code)  # noqa: leading-underscore


def release_cuda_ipc_file() -> None:
    """Remove torch's CUDA IPC reference counter file from shared memory.

    A process that sent CUDA tensors to another process keeps that file, and
    the C++ teardown that os._exit skips removes it once no consumer holds a
    sent tensor. A process that never initialized CUDA has no such file.
    """
    torch = sys.modules.get("torch")
    if torch is None or not torch.cuda.is_initialized():
        return
    else:
        pass
    try:
        torch.cuda.ipc_collect()
    except Exception:
        traceback.print_exc()
