# SPDX-License-Identifier: Apache-2.0
"""Measure PersonaPlex component costs with paired unprofiled and profiled runs."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict
from pathlib import Path

from benchmarks.benchmarker.fingerprint import collect_environment_fingerprint
from benchmarks.eval.personaplex_profiling_report import (
    ComponentReport,
    build_component_report,
)
from benchmarks.eval.personaplex_profiling_workload import (
    PassMeasurement,
    measure_pass,
    measure_request,
)
from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, SamplingParams
from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.personaplex.config import PersonaPlexPipelineConfig
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.profiler.profiler_control import ProfilerControlClient
from sglang_omni.profiler.views import ProfilerReport, build_report, format_table
from sglang_omni.proto.request import EXPLICIT_GENERATION_PARAMS_KEY

PROFILER_POLL_SECONDS = 0.1


def format_summary(
    baseline: PassMeasurement,
    profiled: PassMeasurement,
    outputs_match: bool,
    component_report: ComponentReport,
    request_report: ProfilerReport,
) -> str:
    request_rows: list[dict[str, str | int | float]] = [
        {
            "Request": request.request_id,
            "Latency (ms)": f"{request.latency_milliseconds:.3f}",
            "First audio (ms)": f"{request.time_to_first_audio_milliseconds:.3f}",
            "RTF": f"{request.real_time_factor:.3f}",
        }
        for measurement in (baseline, profiled)
        for request in measurement.requests
    ]
    cost_columns = ["CPU scope", "GPU kernels", "CUDA launch", "CUDA sync", "GPU copy"]
    component_rows: list[dict[str, str | int | float]] = []
    for component in component_report["components"]:
        durations_milliseconds = [
            component["cpu_scope_milliseconds"],
            component["gpu_kernel_milliseconds"],
            component["cuda_launch_milliseconds"],
            component["cuda_synchronization_milliseconds"],
            component["gpu_transfer_milliseconds"],
        ]
        component_rows.append(
            {
                "Component": component["name"].removeprefix("personaplex."),
                **{
                    column: (
                        "N/A"
                        if duration_milliseconds is None
                        else f"{duration_milliseconds:.3f}"
                    )
                    for column, duration_milliseconds in zip(
                        cost_columns, durations_milliseconds, strict=True
                    )
                },
            }
        )
    queue_rows = [
        row
        for row in request_report["stage_breakdown"]
        if row["interval"]
        in (
            "scheduler_queue_enter->scheduler_prefill_start",
            "stage_dispatch->preprocess_start",
            "stage_dispatch->encoder_start",
        )
    ]
    lines = [
        "\nPersonaPlex profiling summary",
        f"Wall time: baseline {baseline.wall_seconds:.3f}s; profiled {profiled.wall_seconds:.3f}s; ratio {profiled.wall_seconds / baseline.wall_seconds:.3f}x",
        f"Outputs match: {outputs_match}; recorded requests: {request_report['request_count']}/{len(profiled.requests)}",
        "\nRequests (RTF = latency / output audio duration):",
        format_table(
            request_rows, ["Request", "Latency (ms)", "First audio (ms)", "RTF"]
        ),
        "Components (ms across the profiled pass; N/A = missing scope):",
        format_table(component_rows, ["Component", *cost_columns]),
        f"Unattributed GPU work: kernels {component_report['unattributed_gpu_kernel_milliseconds']:.3f}ms; copies {component_report['unattributed_gpu_transfer_milliseconds']:.3f}ms; memset {component_report['unattributed_gpu_memset_milliseconds']:.3f}ms",
        *[
            f"GPU {device['host_name']}:{device['device_id']}: window {device['window_milliseconds']:.3f}ms; busy {device['gpu_busy_milliseconds']:.3f}ms; idle {device['gpu_idle_milliseconds']:.3f}ms ({device['window_source']})"
            for device in component_report["devices"]
        ],
        "\nQueue waits (ms; LM admission and dispatch-to-compute):",
        format_table(queue_rows, ["stage", "count", "total_ms", "avg_ms", "max_ms"]),
        "Stage hops (ms; IPC, payload materialization and receiver scheduling):",
        format_table(
            request_report["hop_breakdown"],
            ["src", "dst", "kind", "count", "avg_ms", "max_ms"],
        ),
        "Use baseline latency for performance; profiled timings include recording overhead.",
        "CPU scopes, launches, synchronization, GPU work and waits overlap; do not add them.",
        "GPU idle includes host scheduling; it is not pure launch overhead.",
        "Streaming Mimi per-chunk queue time is unavailable.",
    ]
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path", required=True, help="Checkpoint directory or repository id"
    )
    parser.add_argument("--audio", type=Path, required=True, help="Caller recording")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Parent directory for a unique run directory",
    )
    parser.add_argument(
        "--voice", default="NATF2", help="Packaged voice, voice file, or empty string"
    )
    parser.add_argument(
        "--text-prompt", default=None, help="Override the assistant role prompt"
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for paired greedy requests"
    )
    parser.add_argument(
        "--requests", type=int, default=1, help="Requests in each measured pass"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent requests; the LM admits one at a time",
    )
    parser.add_argument(
        "--warmup-requests",
        type=int,
        default=1,
        help="Unmeasured warmup requests before both passes",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=900.0,
        help="Pipeline startup timeout in seconds",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=600.0,
        help="Timeout per request in seconds, including queueing",
    )
    parser.add_argument(
        "--profiler-timeout",
        type=float,
        default=180.0,
        help="Timeout for profiler startup/export in seconds",
    )
    arguments, stage_overrides = parser.parse_known_args()
    arguments.stage_overrides = stage_overrides
    if (
        arguments.requests < 1
        or arguments.concurrency < 1
        or arguments.warmup_requests < 0
    ):
        parser.error(
            "Requests and concurrency must be positive; warmup requests cannot be negative"
        )
    elif (
        min(
            arguments.startup_timeout,
            arguments.request_timeout,
            arguments.profiler_timeout,
        )
        <= 0
    ):
        parser.error("Timeouts must be positive")
    elif not arguments.audio.is_file():
        parser.error(f"Caller recording does not exist: {arguments.audio}")
    else:
        return arguments


async def wait_for_profiler_files(
    directory: Path,
    pattern: str,
    expected_count: int,
    timeout_seconds: float,
) -> list[Path]:
    deadline_seconds = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline_seconds:
        paths = sorted(directory.glob(pattern))
        if pattern.endswith(".gz"):
            paths = [path for path in paths if not path.with_suffix("").exists()]
        else:
            pass
        if len(paths) == expected_count:
            return paths
        else:
            await asyncio.sleep(PROFILER_POLL_SECONDS)
    raise TimeoutError(
        f"Expected {expected_count} completed {pattern} files in {directory}"
    )


async def run(arguments: argparse.Namespace) -> int:
    config = PersonaPlexPipelineConfig(model_path=arguments.model_path)
    if arguments.stage_overrides:
        config_manager = ConfigManager(config)
        config = config_manager.merge_config(
            config_manager.parse_extra_args(arguments.stage_overrides)
        )
    else:
        pass
    run_id = f"personaplex-{uuid.uuid4().hex}"
    run_directory = arguments.output_dir.expanduser().resolve() / run_id
    event_directory = run_directory / "events"
    trace_directory = run_directory / "traces"
    trace_directory.mkdir(parents=True)
    event_directory.mkdir()
    extra_parameters: dict[str, str | int | float] = {
        "voice": arguments.voice,
        "audio_temperature": 0.0,
        "seed": arguments.seed,
    }
    if arguments.text_prompt is not None:
        extra_parameters["text_prompt"] = arguments.text_prompt
    else:
        pass
    request = GenerateRequest(
        model=config.name,
        prompt={"audio_path": str(arguments.audio.resolve())},
        sampling=SamplingParams(temperature=0.0, seed=arguments.seed),
        extra_params=extra_parameters,
        metadata={EXPLICIT_GENERATION_PARAMS_KEY: ["temperature", "seed"]},
        output_modalities=["text", "audio"],
        stream=True,
    )
    runner = MultiProcessPipelineRunner(config)
    control_client: ProfilerControlClient | None = None
    recorder = get_recorder()
    try:
        started_seconds = time.perf_counter()
        await runner.start(timeout=arguments.startup_timeout)
        startup_seconds = time.perf_counter() - started_seconds
        print(f"Pipeline ready in {startup_seconds:.1f}s; results: {run_directory}")
        client = Client(runner.coordinator)
        for request_index in range(arguments.warmup_requests):
            await measure_request(
                client, request, f"warmup-{request_index}", arguments.request_timeout
            )
        baseline = await measure_pass(
            client,
            request,
            "baseline",
            arguments.requests,
            arguments.concurrency,
            arguments.request_timeout,
        )
        print(f"Unprofiled pass: {baseline.wall_seconds:.3f}s")
        worker_count = sum(len(group.process_specs) for group in runner.groups)
        control_client = ProfilerControlClient(
            stage_endpoints=runner.stage_control_endpoints
        )
        recorder.start(run_id, str(event_directory), "coordinator")
        await control_client.broadcast_start(
            run_id,
            str(trace_directory / "{stage}"),
            event_dir=str(event_directory),
            enable_torch=True,
        )
        await wait_for_profiler_files(
            event_directory,
            "events_*.jsonl",
            worker_count + 1,
            arguments.profiler_timeout,
        )
        profiled = await measure_pass(
            client,
            request,
            "profiled",
            arguments.requests,
            arguments.concurrency,
            arguments.request_timeout,
        )
        await control_client.broadcast_stop(run_id)
        recorder.stop(run_id=run_id)
        trace_paths = await wait_for_profiler_files(
            trace_directory,
            "*.trace.json.gz",
            worker_count,
            arguments.profiler_timeout,
        )
    finally:
        recorder.stop(run_id=run_id)
        try:
            if control_client is not None:
                try:
                    await control_client.broadcast_stop(run_id)
                finally:
                    await control_client.close()
            else:
                pass
        finally:
            await runner.stop()

    parity = [
        {
            "baseline_request_id": original.request_id,
            "profiled_request_id": measured.request_id,
            "text_equal": original.text == measured.text,
            "audio_equal": original.audio_sha256 == measured.audio_sha256,
            "sample_count_equal": original.output_samples == measured.output_samples,
            "sample_rate_equal": original.sample_rate == measured.sample_rate,
        }
        for original, measured in zip(baseline.requests, profiled.requests, strict=True)
    ]
    outputs_match = all(
        comparison["text_equal"]
        and comparison["audio_equal"]
        and comparison["sample_count_equal"]
        and comparison["sample_rate_equal"]
        for comparison in parity
    )
    component_report = build_component_report(trace_paths)
    request_report = build_report(event_directory)
    report_path = run_directory / "report.json"
    report = {
        "environment": collect_environment_fingerprint(arguments.model_path),
        "configuration": config.model_dump(mode="json"),
        "workload": {
            "audio_path": str(arguments.audio.resolve()),
            "audio_file_sha256": hashlib.sha256(
                arguments.audio.read_bytes()
            ).hexdigest(),
            "voice": arguments.voice,
            "text_prompt": arguments.text_prompt,
            "seed": arguments.seed,
            "requests": arguments.requests,
            "concurrency": arguments.concurrency,
            "warmup_requests": arguments.warmup_requests,
        },
        "startup_seconds": startup_seconds,
        "baseline": asdict(baseline),
        "profiled": asdict(profiled),
        "profiler_wall_time_ratio": profiled.wall_seconds / baseline.wall_seconds,
        "outputs_match": outputs_match,
        "parity": parity,
        "components": component_report,
        "request_events": request_report,
        "notes": [
            "CPU scopes, CUDA launches, synchronization, GPU kernels, transfers, and queue intervals overlap; do not add them.",
            "GPU idle is measured within component activity windows; it includes host scheduling and cannot be called pure launch overhead.",
            "Hop latency includes IPC, payload materialization, and receiver scheduling; it is not pure device transfer time.",
            "Queue events measure LM admission and dispatch-to-compute waits for preprocessing and Mimi encoding; streaming Mimi has no per-chunk scheduler queue metric.",
            "Warmup and startup are excluded. Profiled latency includes instrumentation overhead; use the unprofiled pass for performance.",
            "Audio parity compares the complete little-endian float32 waveform, including streamed/terminal agreement.",
        ],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    summary = format_summary(
        baseline, profiled, outputs_match, component_report, request_report
    )
    summary_path = run_directory / "summary.txt"
    summary_path.write_text(summary, encoding="utf-8")
    print(summary, end="")
    print(f"Report: {report_path}")
    print(f"Summary: {summary_path}")
    if not outputs_match:
        return 1
    elif (
        component_report["missing_components"]
        or request_report["request_count"] != arguments.requests
    ):
        print(
            "Profiling coverage is incomplete; inspect missing components and request events"
        )
        return 1
    else:
        return 0


def main() -> None:
    arguments = parse_args()
    os.environ["SGLANG_TORCH_PROFILER_PROFILE_ALL_THREADS"] = "1"
    raise SystemExit(asyncio.run(run(arguments)))


if __name__ == "__main__":
    main()
