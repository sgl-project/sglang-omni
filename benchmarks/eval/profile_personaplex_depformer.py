# SPDX-License-Identifier: Apache-2.0
"""Measure frame rollouts and summarize CUDA activity without CPU operator tracing."""

from __future__ import annotations

import gzip
import json
import shutil
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import torch
from pydantic import BaseModel, Field


class FrameGenerator(Protocol):
    def __call__(self) -> torch.Tensor: ...


class TraceEvent(BaseModel):
    category: str | None = Field(default=None, alias="cat")
    phase: str = Field(default="", alias="ph")
    name: str = ""
    timestamp_microseconds: float = Field(default=0.0, alias="ts")
    duration_microseconds: float = Field(default=0.0, alias="dur")


class CudaTrace(BaseModel):
    events: list[TraceEvent] = Field(alias="traceEvents")


@dataclass(kw_only=True)
class FrameProfile:
    frames: int
    unprofiled_durations_seconds: list[float]
    unprofiled_mean_seconds: float
    unprofiled_p95_seconds: float
    profiled_wall_seconds: float
    gpu_span_seconds: float
    gpu_active_seconds: float
    gpu_idle_percent: float
    kernel_count: int
    kernels_per_frame: float
    host_kernel_launch_calls: int
    host_graph_launch_calls: int
    cuda_api_counts: dict[str, int]
    device_activity_counts: dict[str, int]
    kernel_names: dict[str, int]
    compressed_trace: str


def measure_frames(
    generate: FrameGenerator,
    frame_counts: list[int],
    repeats: int,
) -> dict[int, list[float]]:
    frame_durations_seconds: dict[int, list[float]] = {}
    for frame_count in frame_counts:
        durations_seconds: list[float] = []
        for _ in range(repeats):
            torch.cuda.synchronize()
            started_seconds = time.perf_counter()
            for _ in range(frame_count):
                generate()
                # note (Codex): The next temporal step consumes this frame's codes.
                torch.cuda.synchronize()
            durations_seconds.append(time.perf_counter() - started_seconds)
        frame_durations_seconds[frame_count] = durations_seconds
    return frame_durations_seconds


def profile_frames(
    generate: FrameGenerator,
    frame_counts: list[int],
    destination: Path,
    frame_durations_seconds: dict[int, list[float]],
) -> list[FrameProfile]:
    destination.mkdir(parents=True, exist_ok=True)
    profiles: list[FrameProfile] = []
    for frame_count in frame_counts:
        durations_seconds = frame_durations_seconds[frame_count]
        trace_path = destination / f"frames-{frame_count}.json"
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CUDA],
            experimental_config=torch._C._profiler._ExperimentalConfig(trace_only=True),
        ) as profiler:
            started_seconds = time.perf_counter()
            for _ in range(frame_count):
                generate()
                torch.cuda.synchronize()
            profiled_wall_seconds = time.perf_counter() - started_seconds
        profiler.export_chrome_trace(str(trace_path))
        trace = CudaTrace.model_validate_json(trace_path.read_text())
        device_events = [
            event
            for event in trace.events
            if event.phase == "X"
            and event.category in ("kernel", "gpu_memcpy", "gpu_memset")
        ]
        assert device_events, "The profiler did not record CUDA device activity"
        intervals_microseconds = sorted(
            (
                event.timestamp_microseconds,
                event.timestamp_microseconds + event.duration_microseconds,
            )
            for event in device_events
        )
        active_microseconds = 0.0
        merged_start_microseconds, merged_end_microseconds = intervals_microseconds[0]
        for start_microseconds, end_microseconds in intervals_microseconds[1:]:
            if start_microseconds <= merged_end_microseconds:
                merged_end_microseconds = max(merged_end_microseconds, end_microseconds)
            else:
                active_microseconds += (
                    merged_end_microseconds - merged_start_microseconds
                )
                merged_start_microseconds, merged_end_microseconds = (
                    start_microseconds,
                    end_microseconds,
                )
        active_microseconds += merged_end_microseconds - merged_start_microseconds
        span_microseconds = merged_end_microseconds - intervals_microseconds[0][0]
        kernel_names = Counter(
            event.name for event in device_events if event.category == "kernel"
        )
        api_counts = Counter(
            event.name
            for event in trace.events
            if event.phase == "X" and event.category in ("cuda_runtime", "cuda_driver")
        )
        compressed_path = trace_path.with_suffix(".json.gz")
        with (
            trace_path.open("rb") as source,
            gzip.open(compressed_path, "wb") as output,
        ):
            shutil.copyfileobj(source, output)
        trace_path.unlink()
        result = FrameProfile(
            frames=frame_count,
            unprofiled_durations_seconds=durations_seconds,
            unprofiled_mean_seconds=float(np.mean(durations_seconds)),
            unprofiled_p95_seconds=float(np.percentile(durations_seconds, 95)),
            profiled_wall_seconds=profiled_wall_seconds,
            gpu_span_seconds=span_microseconds / 1e6,
            gpu_active_seconds=active_microseconds / 1e6,
            gpu_idle_percent=100 * (1 - active_microseconds / span_microseconds),
            kernel_count=kernel_names.total(),
            kernels_per_frame=kernel_names.total() / frame_count,
            host_kernel_launch_calls=sum(
                count
                for name, count in api_counts.items()
                if "LaunchKernel" in name or "LaunchCooperativeKernel" in name
            ),
            host_graph_launch_calls=sum(
                count for name, count in api_counts.items() if "GraphLaunch" in name
            ),
            cuda_api_counts=dict(api_counts),
            device_activity_counts=dict(
                Counter(event.category for event in device_events)
            ),
            kernel_names=dict(kernel_names),
            compressed_trace=str(compressed_path),
        )
        profiles.append(result)
        (destination / "summary.json").write_text(
            json.dumps([asdict(profile) for profile in profiles], indent=2) + "\n"
        )
        print(
            json.dumps(
                {
                    "profile": str(destination),
                    "frames": result.frames,
                    "wall_seconds": result.unprofiled_mean_seconds,
                    "kernels_per_frame": result.kernels_per_frame,
                    "gpu_idle_percent": result.gpu_idle_percent,
                }
            ),
            flush=True,
        )
    return profiles
