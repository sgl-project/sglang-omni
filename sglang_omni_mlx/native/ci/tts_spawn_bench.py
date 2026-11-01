# SPDX-License-Identifier: Apache-2.0
"""Measures a native TTS server's spawn-to-first-audio time and idle footprint.

    python tts_spawn_bench.py --server BIN --model-path DIR --request BODY.json

Each run starts the server, polls /health, sends the request, sends it again
for a steady-state time, waits, reads the server's physical footprint and
stops it. Cold runs evict the checkpoint from the page cache first and check
with mincore that no page stayed; warm runs read it into the cache first.
"""

from __future__ import annotations

import argparse
import ctypes
import http.client
import io
import json
import mmap
import os
import signal
import socket
import statistics
import subprocess
import time
import wave
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

HEALTH_POLL_SECONDS = 0.01
READ_CHUNK_BYTES = 1 << 20
RUSAGE_INFO_V4 = 4
MS_INVALIDATE = 0x0002
LIBC = ctypes.CDLL(None, use_errno=True)
LIBC.mmap.restype = ctypes.c_void_p
LIBC.mmap.argtypes = [
    ctypes.c_void_p,
    ctypes.c_size_t,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int,
    ctypes.c_int64,
]
LIBC.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
LIBC.msync.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
LIBC.mincore.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p]
LIBC.proc_pid_rusage.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_void_p]


class RusageInfoV4(ctypes.Structure):
    """The rusage_info_v4 layout from sys/resource.h."""

    _fields_ = [("ri_uuid", ctypes.c_uint8 * 16)] + [
        (name, ctypes.c_uint64)
        for name in (
            "ri_user_time ri_system_time ri_pkg_idle_wkups ri_interrupt_wkups "
            "ri_pageins ri_wired_size ri_resident_size ri_phys_footprint "
            "ri_proc_start_abstime ri_proc_exit_abstime ri_child_user_time "
            "ri_child_system_time ri_child_pkg_idle_wkups ri_child_interrupt_wkups "
            "ri_child_pageins ri_child_elapsed_abstime ri_diskio_bytesread "
            "ri_diskio_byteswritten ri_cpu_time_qos_default "
            "ri_cpu_time_qos_maintenance ri_cpu_time_qos_background "
            "ri_cpu_time_qos_utility ri_cpu_time_qos_legacy "
            "ri_cpu_time_qos_user_initiated ri_cpu_time_qos_user_interactive "
            "ri_billed_system_time ri_serviced_system_time ri_logical_writes "
            "ri_lifetime_max_phys_footprint ri_instructions ri_cycles "
            "ri_billed_energy ri_serviced_energy ri_interval_max_phys_footprint "
            "ri_runnable_time"
        ).split()
    ]


@dataclass(kw_only=True)
class SpawnRun:
    cache_state: Literal["cold", "warm"]
    resident_pages_before_spawn: int
    checkpoint_pages: int
    ready_seconds: float
    first_audio_byte_seconds: float
    first_audio_seconds: float
    steady_request_seconds: float
    audio_duration_seconds: float | None
    idle_footprint_bytes: int
    peak_footprint_bytes: int


def page_cache_pages(path: Path, should_evict: bool) -> tuple[int, int]:
    """Returns (resident pages, total pages) of a file, evicting it first if asked."""
    size_bytes = path.stat().st_size
    if size_bytes == 0:
        return 0, 0
    else:
        pass
    page_bytes = mmap.PAGESIZE
    page_count = (size_bytes + page_bytes - 1) // page_bytes
    file_descriptor = os.open(path, os.O_RDONLY)
    try:
        address = LIBC.mmap(
            None, size_bytes, mmap.PROT_READ, mmap.MAP_SHARED, file_descriptor, 0
        )
        if address == ctypes.c_void_p(-1).value:
            raise OSError(ctypes.get_errno(), f"mmap failed for {path}")
        else:
            pass
        try:
            if should_evict and LIBC.msync(address, size_bytes, MS_INVALIDATE) != 0:
                raise OSError(ctypes.get_errno(), f"msync failed for {path}")
            else:
                pass
            residency = ctypes.create_string_buffer(page_count)
            if LIBC.mincore(address, size_bytes, residency) != 0:
                raise OSError(ctypes.get_errno(), f"mincore failed for {path}")
            else:
                pass
            resident_count = sum(byte & 1 for byte in residency.raw)
        finally:
            LIBC.munmap(address, size_bytes)
    finally:
        os.close(file_descriptor)
    return resident_count, page_count


def footprint_bytes(process_id: int) -> tuple[int, int]:
    """Returns (current, lifetime peak) physical footprint of a process."""
    usage = RusageInfoV4()
    if LIBC.proc_pid_rusage(process_id, RUSAGE_INFO_V4, ctypes.byref(usage)) != 0:
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage failed for {process_id}")
    else:
        pass
    return usage.ri_phys_footprint, usage.ri_lifetime_max_phys_footprint


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def post_speech(port: int, body: bytes, started: float) -> tuple[float, float, bytes]:
    """Sends one speech request; returns first-byte and done times since started."""
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=600)
    connection.request(
        "POST", "/v1/audio/speech", body, {"Content-Type": "application/json"}
    )
    response = connection.getresponse()
    first_chunk = response.read1(READ_CHUNK_BYTES)
    first_byte_seconds = time.monotonic() - started
    audio = first_chunk + response.read()
    done_seconds = time.monotonic() - started
    connection.close()
    if response.status != 200 or not audio:
        raise RuntimeError(f"speech request failed: {response.status} {audio[:300]!r}")
    else:
        pass
    return first_byte_seconds, done_seconds, audio


def run_once(
    arguments: argparse.Namespace,
    cache_state: Literal["cold", "warm"],
    checkpoint_files: list[Path],
    body: bytes,
) -> SpawnRun:
    for path in checkpoint_files:
        if cache_state == "warm":
            with path.open("rb") as checkpoint_file:
                while checkpoint_file.read(READ_CHUNK_BYTES):
                    pass
        else:
            page_cache_pages(path, should_evict=True)
    # note (Minyi Tang): count residency only after every file is prepared, since warming a later shard can evict an earlier one.
    resident_pages = 0
    checkpoint_pages = 0
    for path in checkpoint_files:
        resident, total = page_cache_pages(path, should_evict=False)
        resident_pages += resident
        checkpoint_pages += total
    port = free_port()
    command = [
        str(arguments.server.resolve()),
        "--model-path", str(arguments.model_path),
        "--host", "127.0.0.1",
        "--port", str(port),
        *arguments.server_arg,
    ]  # fmt: skip
    started = time.monotonic()
    process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=arguments.server_log
    )
    try:
        while True:
            if process.poll() is not None:
                raise RuntimeError(f"server exited with {process.returncode}")
            elif time.monotonic() - started > arguments.startup_timeout_seconds:
                raise TimeoutError(
                    f"server not healthy after {arguments.startup_timeout_seconds} s"
                )
            else:
                pass
            try:
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                connection.request("GET", "/health")
                is_healthy = connection.getresponse().status == 200
                connection.close()
            except OSError:
                is_healthy = False
            if is_healthy:
                break
            else:
                time.sleep(HEALTH_POLL_SECONDS)
        ready_seconds = time.monotonic() - started
        first_byte_seconds, first_audio_seconds, audio = post_speech(
            port, body, started
        )
        steady_started = time.monotonic()
        post_speech(port, body, steady_started)
        steady_request_seconds = time.monotonic() - steady_started
        time.sleep(arguments.idle_seconds)
        idle_footprint, peak_footprint = footprint_bytes(process.pid)
    finally:
        process.send_signal(signal.SIGTERM)
        process.wait(timeout=30)
    if audio[:4] == b"RIFF":
        with wave.open(io.BytesIO(audio)) as wave_file:
            audio_duration_seconds = wave_file.getnframes() / wave_file.getframerate()
    else:
        audio_duration_seconds = None
    return SpawnRun(
        cache_state=cache_state,
        resident_pages_before_spawn=resident_pages,
        checkpoint_pages=checkpoint_pages,
        ready_seconds=ready_seconds,
        first_audio_byte_seconds=first_byte_seconds,
        first_audio_seconds=first_audio_seconds,
        steady_request_seconds=steady_request_seconds,
        audio_duration_seconds=audio_duration_seconds,
        idle_footprint_bytes=idle_footprint,
        peak_footprint_bytes=peak_footprint,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument(
        "--request", type=Path, required=True, help="JSON body for /v1/audio/speech"
    )
    parser.add_argument(
        "--server-arg",
        action="append",
        default=[],
        help="extra server argument; repeat for each",
    )
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--idle-seconds",
        type=float,
        default=5.0,
        help="wait after the last request before reading the footprint",
    )
    parser.add_argument("--startup-timeout-seconds", type=float, default=600.0)
    parser.add_argument("--output", type=Path, help="write every run as JSON")
    parser.add_argument(
        "--server-log", type=argparse.FileType("w"), default="/dev/null"
    )
    arguments = parser.parse_args()
    if not os.access(arguments.server, os.X_OK):
        parser.error(f"not an executable: {arguments.server}")
    elif not arguments.model_path.is_dir():
        parser.error(f"not a directory: {arguments.model_path}")
    else:
        pass
    body = arguments.request.read_bytes()
    request = json.loads(body)
    if not isinstance(request, dict):
        parser.error("--request must hold a JSON object")
    elif request.get("stream") is True:
        # note (Minyi Tang): a streamed response is HTTP 200 before synthesis, so a failed generation would count as audio.
        parser.error("--request must not stream")
    else:
        pass
    checkpoint_files = sorted(
        path for path in arguments.model_path.rglob("*") if path.is_file()
    )
    runs = []
    for cache_state in ("cold", "warm"):
        for _ in range(arguments.runs):
            run = run_once(arguments, cache_state, checkpoint_files, body)
            print(json.dumps(asdict(run)), flush=True)
            runs.append(run)
    for cache_state in ("cold", "warm"):
        selected = [run for run in runs if run.cache_state == cache_state]
        print(
            f"{cache_state}: median over {len(selected)} runs: "
            f"ready {statistics.median(run.ready_seconds for run in selected):.3f} s, "
            f"first audio byte "
            f"{statistics.median(run.first_audio_byte_seconds for run in selected):.3f} s, "
            f"first audio {statistics.median(run.first_audio_seconds for run in selected):.3f} s, "
            f"steady request "
            f"{statistics.median(run.steady_request_seconds for run in selected):.3f} s, "
            f"idle footprint "
            f"{statistics.median(run.idle_footprint_bytes for run in selected) / 1e9:.2f} GB, "
            f"peak footprint "
            f"{statistics.median(run.peak_footprint_bytes for run in selected) / 1e9:.2f} GB, "
            f"max resident pages before spawn "
            f"{max(run.resident_pages_before_spawn for run in selected)}"
            f"/{selected[0].checkpoint_pages}"
        )
    if arguments.output is not None:
        arguments.output.write_text(
            json.dumps([asdict(run) for run in runs], indent=2) + "\n"
        )
    else:
        pass


if __name__ == "__main__":
    main()
