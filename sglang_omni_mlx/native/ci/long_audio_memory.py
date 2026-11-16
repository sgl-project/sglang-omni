# SPDX-License-Identifier: Apache-2.0
"""Sample a server's macOS memory and stop it once its peak passes a limit."""

from __future__ import annotations

import ctypes
import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import TracebackType

MEMORY_SAMPLE_INTERVAL_SECONDS = 0.25
RUSAGE_INFO_V4 = 4
SAMPLER_JOIN_TIMEOUT_SECONDS = 5


@dataclass(frozen=True, kw_only=True)
class MemorySample:
    phase: str
    elapsed_seconds: float
    footprint_bytes: int
    peak_footprint_bytes: int
    resident_bytes: int


class ProcessUsage(ctypes.Structure):
    """The layout of rusage_info_v4 in sys/resource.h."""

    _fields_ = [("uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "user_time",
            "system_time",
            "idle_wakeups",
            "interrupt_wakeups",
            "pageins",
            "wired_bytes",
            "resident_bytes",
            "footprint_bytes",
            "start_time",
            "exit_time",
            "child_user_time",
            "child_system_time",
            "child_idle_wakeups",
            "child_interrupt_wakeups",
            "child_pageins",
            "child_elapsed_time",
            "disk_bytes_read",
            "disk_bytes_written",
            "cpu_time_qos_default",
            "cpu_time_qos_maintenance",
            "cpu_time_qos_background",
            "cpu_time_qos_utility",
            "cpu_time_qos_legacy",
            "cpu_time_qos_user_initiated",
            "cpu_time_qos_user_interactive",
            "billed_system_time",
            "serviced_system_time",
            "logical_writes",
            "peak_footprint_bytes",
            "instructions",
            "cycles",
            "billed_energy",
            "serviced_energy",
            "interval_peak_footprint_bytes",
            "runnable_time",
        )
    ]


class MemorySampler:
    """Reads a server's memory on a thread while its context is open."""

    def __init__(
        self,
        process: subprocess.Popen[bytes],
        destination: Path,
        limit_bytes: int,
    ) -> None:
        self.process = process
        self.destination = destination
        self.limit_bytes = limit_bytes
        self.samples: list[MemorySample] = []
        self.phase = "startup"
        self.exceeded = False
        self.error = ""
        self.done = threading.Event()
        self.library = ctypes.CDLL(None, use_errno=True)
        self.library.proc_pid_rusage.argtypes = [
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        self.library.proc_pid_rusage.restype = ctypes.c_int
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def __enter__(self) -> MemorySampler:
        self.thread.start()
        return self

    def __exit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.done.set()
        self.thread.join(timeout=SAMPLER_JOIN_TIMEOUT_SECONDS)

    def stop_on_error(self, message: str) -> None:
        self.error = message
        self.process.kill()

    def sample(self) -> None:
        started = time.monotonic()
        try:
            with self.destination.open("w") as output:
                while not self.done.is_set() and self.process.poll() is None:
                    usage = ProcessUsage()
                    if (
                        self.library.proc_pid_rusage(
                            self.process.pid, RUSAGE_INFO_V4, ctypes.byref(usage)
                        )
                        != 0
                    ):
                        self.stop_on_error(
                            f"proc_pid_rusage failed: errno {ctypes.get_errno()}"
                        )
                        return
                    else:
                        sample = MemorySample(
                            phase=self.phase,
                            elapsed_seconds=time.monotonic() - started,
                            footprint_bytes=usage.footprint_bytes,
                            peak_footprint_bytes=usage.peak_footprint_bytes,
                            resident_bytes=usage.resident_bytes,
                        )
                        self.samples.append(sample)
                        output.write(json.dumps(asdict(sample)) + "\n")
                        output.flush()
                    if usage.peak_footprint_bytes > self.limit_bytes:
                        self.exceeded = True
                        self.process.kill()
                        return
                    else:
                        self.done.wait(MEMORY_SAMPLE_INTERVAL_SECONDS)
        except OSError as error:
            self.stop_on_error(f"Memory trace failed: {error}")
