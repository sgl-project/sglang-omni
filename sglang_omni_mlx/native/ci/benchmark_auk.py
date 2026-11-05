# SPDX-License-Identifier: Apache-2.0
"""Benchmark AuK HTTP speech latency, throughput, and macOS process memory."""

from __future__ import annotations

import argparse
import base64
import ctypes
import ctypes.util
import hashlib
import importlib.metadata
import io
import json
import math
import platform
import statistics
import sys
import threading
import time
import wave
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, TypedDict

import httpx
import numpy as np
import psutil

SAMPLE_RATE = 24000
DURATION_TOLERANCE_SECONDS = 0.021
MINIMUM_AUDIO_RMS = 1e-5
MEMORY_SAMPLE_INTERVAL_SECONDS = 0.25
RUSAGE_INFO_VERSION = 4


@dataclass(frozen=True, kw_only=True)
class BenchmarkCase:
    sample_id: str
    language: Literal["en", "zh"]
    text: str
    reference_file: Path
    generation_seconds: float
    seed: int


class SpeechRequest(TypedDict):
    input: str
    ref_audio: str
    seed: int
    response_format: Literal["wav"]
    stage_params: dict[Literal["auk_engine"], dict[Literal["gen_seconds"], float]]


@dataclass(kw_only=True)
class ProcessMemory:
    process_id: int
    physical_footprint_bytes: int
    lifetime_peak_physical_footprint_bytes: int
    resident_bytes: int


@dataclass(kw_only=True)
class MemorySample:
    elapsed_seconds: float
    processes: list[ProcessMemory]
    aggregate_physical_footprint_bytes: int


@dataclass(kw_only=True)
class RequestMeasurement:
    sample_id: str
    language: Literal["en", "zh"]
    repeat: int
    index: int
    latency_seconds: float
    audio_seconds: float
    realtime_factor: float
    wav: str
    wav_sha256: str
    text: str
    reference_file: str


class RusageInfoV4(ctypes.Structure):
    ri_phys_footprint: int
    ri_lifetime_max_phys_footprint: int
    ri_resident_size: int
    _fields_ = [  # noqa: leading-underscore  # ctypes structure layout contract
        ("ri_uuid", ctypes.c_uint8 * 16),
        *[
            (f"ri_{field_name}", ctypes.c_uint64)
            for field_name in (
                "user_time system_time pkg_idle_wkups interrupt_wkups pageins "
                "wired_size resident_size phys_footprint proc_start_abstime "
                "proc_exit_abstime child_user_time child_system_time "
                "child_pkg_idle_wkups child_interrupt_wkups child_pageins "
                "child_elapsed_abstime diskio_bytesread diskio_byteswritten "
                "cpu_time_qos_default cpu_time_qos_maintenance "
                "cpu_time_qos_background cpu_time_qos_utility cpu_time_qos_legacy "
                "cpu_time_qos_user_initiated cpu_time_qos_user_interactive "
                "billed_system_time serviced_system_time logical_writes "
                "lifetime_max_phys_footprint instructions cycles billed_energy "
                "serviced_energy interval_max_phys_footprint runnable_time"
            ).split()
        ],
    ]


def string_field(record: Mapping[str, object], field_name: str) -> str:
    value = record.get(field_name)
    if not isinstance(value, str) or not value:
        raise ValueError(f"Each case requires a nonempty {field_name} string")
    else:
        return value


def load_cases(path: Path) -> list[BenchmarkCase]:
    raw_cases: object = json.loads(path.read_text())
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError("Cases must be a nonempty JSON list")
    else:
        pass
    cases = []
    for record in raw_cases:
        if not isinstance(record, dict):
            raise ValueError("Each case must be a JSON object")
        else:
            pass
        language = string_field(record, "language")
        seconds, seed = record.get("generation_seconds"), record.get("seed")
        if language not in ("en", "zh"):
            raise ValueError("Case language must be en or zh")
        elif isinstance(seconds, bool) or not isinstance(seconds, (int, float)):
            raise ValueError("generation_seconds must be numeric")
        elif not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("generation_seconds must be finite and positive")
        elif isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("Case seed must be an integer")
        else:
            cases.append(
                BenchmarkCase(
                    sample_id=string_field(record, "id"),
                    language="en" if language == "en" else "zh",
                    text=string_field(record, "text"),
                    reference_file=(
                        path.parent / string_field(record, "reference_file")
                    ).resolve(),
                    generation_seconds=float(seconds),
                    seed=seed,
                )
            )
    return cases


def process_memory(library: ctypes.CDLL, process_id: int) -> ProcessMemory:
    usage = RusageInfoV4()
    if library.proc_pid_rusage(process_id, RUSAGE_INFO_VERSION, ctypes.byref(usage)):
        raise OSError(ctypes.get_errno(), "proc_pid_rusage failed")
    else:
        return ProcessMemory(
            process_id=process_id,
            physical_footprint_bytes=usage.ri_phys_footprint,
            lifetime_peak_physical_footprint_bytes=usage.ri_lifetime_max_phys_footprint,
            resident_bytes=usage.ri_resident_size,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--url", required=True)
    parser.add_argument("--backend", required=True)
    parser.add_argument("--variant", choices=["AuK", "AuK-Flash"], required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=1, dest="repeat_count")
    parser.add_argument("--warmup", type=int, default=0, dest="warmup_request_count")
    parser.add_argument("--server-pid", type=int, dest="server_process_id")
    parser.add_argument("--timeout", type=float, default=300, dest="timeout_seconds")
    options = parser.parse_args()
    if options.repeat_count < 1 or options.warmup_request_count < 0:
        parser.error("repeats must be positive and warmup must be nonnegative")
    elif options.output.exists() and any(options.output.iterdir()):
        parser.error("Output directory must be empty")
    elif options.server_process_id is not None and platform.system() != "Darwin":
        parser.error("Process physical-footprint sampling requires macOS")
    else:
        pass
    cases = load_cases(options.cases)
    requests: list[SpeechRequest] = []
    reference_hashes: dict[str, str] = {}
    for case in cases:
        reference_bytes = case.reference_file.read_bytes()
        reference_hashes[str(case.reference_file)] = hashlib.sha256(
            reference_bytes
        ).hexdigest()
        requests.append(
            SpeechRequest(
                input=case.text,
                ref_audio="data:audio/wav;base64,"
                + base64.b64encode(reference_bytes).decode(),
                seed=case.seed,
                response_format="wav",
                stage_params={"auk_engine": {"gen_seconds": case.generation_seconds}},
            )
        )
    options.output.mkdir(parents=True, exist_ok=True)
    protocol = {
        "backend": options.backend,
        "variant": options.variant,
        "url": options.url,
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": sys.version,
            "packages": {
                name: importlib.metadata.version(name)
                for name in ("httpx", "numpy", "psutil")
            },
        },
        "cases_sha256": hashlib.sha256(options.cases.read_bytes()).hexdigest(),
        "reference_sha256": reference_hashes,
        "samples": len(cases),
        "concurrency": 1,
        "repetitions": options.repeat_count,
        "warmup_requests": options.warmup_request_count,
        "seeds": [case.seed for case in cases],
        "latency": "HTTP request wall time; audio loading/base64 encoding precedes timing",
        "throughput": "Actual measured window, including client validation/result writes between requests; excludes warmup/startup/scoring",
        "memory": "macOS process-tree physical footprint sampled every 250 ms; individual OS lifetime peaks retained separately; RSS is not GPU memory",
        "load_time": "External launcher records launch-to-health separately",
    }
    (options.output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    samples: list[MemorySample] = []
    stop = threading.Event()
    started_seconds = time.perf_counter()

    def monitor_memory() -> None:
        library_name = ctypes.util.find_library("proc")
        if library_name is None:
            raise RuntimeError("macOS libproc is unavailable")
        else:
            library = ctypes.CDLL(library_name, use_errno=True)
        while not stop.is_set():
            try:
                parent = psutil.Process(options.server_process_id)
                processes = []
                for process in [parent, *parent.children(recursive=True)]:
                    try:
                        processes.append(process_memory(library, process.pid))
                    except OSError:
                        continue
                samples.append(
                    MemorySample(
                        elapsed_seconds=time.perf_counter() - started_seconds,
                        processes=processes,
                        aggregate_physical_footprint_bytes=sum(
                            process.physical_footprint_bytes for process in processes
                        ),
                    )
                )
            except psutil.NoSuchProcess:
                break
            stop.wait(MEMORY_SAMPLE_INTERVAL_SECONDS)

    watcher = (
        threading.Thread(target=monitor_memory, daemon=True)
        if options.server_process_id is not None
        else None
    )
    if watcher is not None:
        watcher.start()
    else:
        pass
    measurements: list[RequestMeasurement] = []
    try:
        with httpx.Client(
            base_url=options.url, timeout=options.timeout_seconds
        ) as client:
            client.get("/health").raise_for_status()
            for index in range(options.warmup_request_count):
                client.post(
                    "/v1/audio/speech", json=requests[index % len(requests)]
                ).raise_for_status()
            measured_started_seconds = time.perf_counter()
            for repeat in range(options.repeat_count):
                for index, (case, request) in enumerate(zip(cases, requests)):
                    request_started_seconds = time.perf_counter()
                    response = client.post("/v1/audio/speech", json=request)
                    latency_seconds = time.perf_counter() - request_started_seconds
                    response.raise_for_status()
                    with wave.open(io.BytesIO(response.content)) as audio:
                        if (
                            audio.getnchannels(),
                            audio.getframerate(),
                            audio.getsampwidth(),
                        ) != (1, SAMPLE_RATE, 2):
                            raise ValueError("Expected mono 24 kHz PCM16 WAV")
                        else:
                            audio_seconds = audio.getnframes() / SAMPLE_RATE
                            waveform = (
                                np.frombuffer(
                                    audio.readframes(audio.getnframes()), dtype="<i2"
                                ).astype(np.float32)
                                / 32768
                            )
                    if (
                        not waveform.size
                        or not np.isfinite(waveform).all()
                        or np.sqrt(np.mean(waveform**2)) <= MINIMUM_AUDIO_RMS
                    ):
                        raise ValueError(
                            "Generated waveform is empty, nonfinite, or silent"
                        )
                    elif (
                        abs(audio_seconds - case.generation_seconds)
                        > DURATION_TOLERANCE_SECONDS
                    ):
                        raise ValueError(
                            f"Unexpected generated duration: {audio_seconds}"
                        )
                    else:
                        pass
                    filename = f"r{repeat + 1}-{case.language}-{index:03d}.wav"
                    (options.output / filename).write_bytes(response.content)
                    measurements.append(
                        RequestMeasurement(
                            sample_id=case.sample_id,
                            language=case.language,
                            repeat=repeat + 1,
                            index=index,
                            latency_seconds=latency_seconds,
                            audio_seconds=audio_seconds,
                            realtime_factor=latency_seconds / audio_seconds,
                            wav=filename,
                            wav_sha256=hashlib.sha256(response.content).hexdigest(),
                            text=case.text,
                            reference_file=str(case.reference_file),
                        )
                    )
                    (options.output / "requests.json").write_text(
                        json.dumps(
                            [asdict(measurement) for measurement in measurements],
                            ensure_ascii=False,
                            indent=2,
                        )
                        + "\n"
                    )
                    print(
                        f"{options.backend} {options.variant} repeat={repeat + 1} sample={index + 1}/{len(cases)} latency={latency_seconds:.3f}s",
                        flush=True,
                    )
            measured_wall_seconds = time.perf_counter() - measured_started_seconds
    finally:
        stop.set()
        if watcher is not None:
            watcher.join()
        else:
            pass
        (options.output / "memory.json").write_text(
            json.dumps([asdict(sample) for sample in samples], indent=2) + "\n"
        )
    latencies_seconds = [measurement.latency_seconds for measurement in measurements]
    process_peaks: dict[str, int] = {}
    for sample in samples:
        for process in sample.processes:
            process_id = str(process.process_id)
            process_peaks[process_id] = max(
                process_peaks.get(process_id, 0),
                process.lifetime_peak_physical_footprint_bytes,
            )
    summary = {
        "requests": len(measurements),
        "mean_latency_seconds": statistics.mean(latencies_seconds),
        "p95_latency_seconds": float(np.percentile(latencies_seconds, 95)),
        "mean_realtime_factor": statistics.mean(
            measurement.realtime_factor for measurement in measurements
        ),
        "total_latency_seconds": sum(latencies_seconds),
        "measured_wall_seconds": measured_wall_seconds,
        "audio_seconds_per_second": sum(
            measurement.audio_seconds for measurement in measurements
        )
        / measured_wall_seconds,
        "requests_per_second": len(measurements) / measured_wall_seconds,
        "sampled_peak_process_tree_physical_bytes": max(
            (sample.aggregate_physical_footprint_bytes for sample in samples),
            default=None,
        ),
        "os_process_lifetime_peak_bytes": process_peaks,
    }
    (options.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
else:
    pass
