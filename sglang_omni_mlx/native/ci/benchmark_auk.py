# SPDX-License-Identifier: Apache-2.0
"""SeedTTS generation benchmark for a running AuK server, with macOS process-tree memory sampling."""

from __future__ import annotations

import asyncio
import ctypes
import ctypes.util
import json
import logging
import platform
import threading
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path

from benchmarks.eval.benchmark_tts_seedtts import (
    _build_arg_parser,
    _config_from_args,
    _load_benchmark_samples,
    _parse_args,
    _validate_args,
    run_tts_seedtts_benchmark,
)
from benchmarks.metrics.performance import print_speed_summary
from benchmarks.tasks.tts import preload_reference_audio
from sglang_omni.utils.logging import configure_dependency_loggers

MEMORY_SAMPLE_INTERVAL_SECONDS = 0.25
RUSAGE_INFO_VERSION = 4
MAXIMUM_CHILD_PROCESSES = 256
logger = logging.getLogger(__name__)


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


def process_memory(library: ctypes.CDLL, process_id: int) -> ProcessMemory:
    usage = RusageInfoV4()
    if library.proc_pid_rusage(process_id, RUSAGE_INFO_VERSION, ctypes.byref(usage)):
        raise OSError(ctypes.get_errno(), f"proc_pid_rusage failed for {process_id}")
    else:
        return ProcessMemory(
            process_id=process_id,
            physical_footprint_bytes=usage.ri_phys_footprint,
            lifetime_peak_physical_footprint_bytes=usage.ri_lifetime_max_phys_footprint,
            resident_bytes=usage.ri_resident_size,
        )


def process_tree(library: ctypes.CDLL, root_process_id: int) -> list[int]:
    """The root and all its descendants, so a multi-process Omni server is measured whole."""
    process_ids = [root_process_id]
    children = (ctypes.c_int * MAXIMUM_CHILD_PROCESSES)()
    for process_id in process_ids:
        child_count = library.proc_listchildpids(
            process_id, children, ctypes.sizeof(children)
        )
        process_ids.extend(children[:child_count])
    return process_ids


def sample_memory(
    library: ctypes.CDLL, root_process_id: int, started_seconds: float
) -> MemorySample:
    processes = []
    for process_id in process_tree(library, root_process_id):
        try:
            processes.append(process_memory(library, process_id))
        except OSError:
            # A worker that exits between listing and sampling has no usage left.
            if process_id == root_process_id:
                raise
            else:
                continue
    return MemorySample(
        elapsed_seconds=time.perf_counter() - started_seconds,
        processes=processes,
        aggregate_physical_footprint_bytes=sum(
            process.physical_footprint_bytes for process in processes
        ),
    )


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    configure_dependency_loggers()
    parser = _build_arg_parser()
    parser.description = __doc__
    parser.add_argument(
        "--server-pid",
        type=int,
        dest="server_process_id",
        help="macOS server PID whose process tree is sampled with proc_pid_rusage.",
    )
    parser.set_defaults(
        model="tencent/AuK-Flash", generate_only=True, use_existing_server=True
    )
    args, _ = _parse_args(parser)
    if (
        args.transcribe_only
        or args.similarity_only
        or args.utmos_only
        or args.sustained_overshoot
        or args.concurrencies is not None
        or args.repeats > 1
        or args.fingerprint
    ):
        parser.error(
            "benchmark_auk runs one generation pass against a running server; "
            "use benchmarks.eval.benchmark_tts_seedtts for scoring and sweeps"
        )
    elif args.server_process_id is not None and platform.system() != "Darwin":
        parser.error("Process physical-footprint sampling requires macOS")
    else:
        _validate_args(parser, args)
    config = _config_from_args(args)
    samples = _load_benchmark_samples(config)
    if config.voice_clone:
        # The native server reads reference audio only from data URIs.
        reference_audio_by_path = preload_reference_audio(samples)
        samples = [
            replace(sample, ref_audio=reference_audio_by_path[sample.ref_audio])
            for sample in samples
        ]
    else:
        pass
    Path(config.output_dir).mkdir(parents=True, exist_ok=True)

    memory_samples: list[MemorySample] = []
    stop = threading.Event()
    watcher = None
    if args.server_process_id is not None:
        library = ctypes.CDLL(ctypes.util.find_library("proc"), use_errno=True)
        started_seconds = time.perf_counter()
        memory_samples.append(
            sample_memory(library, args.server_process_id, started_seconds)
        )

        def monitor_memory() -> None:
            while not stop.wait(MEMORY_SAMPLE_INTERVAL_SECONDS):
                try:
                    memory_samples.append(
                        sample_memory(library, args.server_process_id, started_seconds)
                    )
                except OSError:
                    logger.warning(
                        f"server {args.server_process_id} exited; memory sampling stopped"
                    )
                    return

        watcher = threading.Thread(target=monitor_memory, daemon=True)
        watcher.start()
    else:
        pass
    try:
        results = asyncio.run(run_tts_seedtts_benchmark(config, samples=samples))
        print_speed_summary(
            results["summary"], config.model, concurrency=config.concurrency
        )
    finally:
        stop.set()
        if watcher is not None:
            watcher.join()
            process_peaks: dict[str, int] = {}
            for memory_sample in memory_samples:
                for process in memory_sample.processes:
                    process_peaks[str(process.process_id)] = max(
                        process_peaks.get(str(process.process_id), 0),
                        process.lifetime_peak_physical_footprint_bytes,
                    )
            Path(config.output_dir, "memory.json").write_text(
                json.dumps(
                    {
                        "server_pid": args.server_process_id,
                        "sampled_peak_process_tree_physical_bytes": max(
                            memory_sample.aggregate_physical_footprint_bytes
                            for memory_sample in memory_samples
                        ),
                        "os_process_lifetime_peak_bytes": process_peaks,
                        "samples": [asdict(sample) for sample in memory_samples],
                    },
                    indent=2,
                )
                + "\n"
            )
        else:
            pass


if __name__ == "__main__":
    main()
else:
    pass
