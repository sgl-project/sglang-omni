# SPDX-License-Identifier: Apache-2.0
"""Attribute PersonaPlex Chrome traces without summing overlapping activity."""

from __future__ import annotations

import gzip
from bisect import bisect_right
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
from pathlib import Path
from typing import Literal, TypedDict, get_args

from pydantic import BaseModel, Field

ComponentName = Literal[
    "personaplex.temporal_transformer",
    "personaplex.text_logits",
    "personaplex.depformer",
    "personaplex.mimi_encode",
    "personaplex.mimi_decode",
    "personaplex.h2d",
    "personaplex.d2h",
    "personaplex.embeddings",
]
COMPONENT_NAMES: tuple[ComponentName, ...] = get_args(ComponentName)
MICROSECONDS_PER_MILLISECOND = 1000.0
CPU_SCOPE_CATEGORIES = {"user_annotation"}
GPU_ACTIVITY_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


class TraceArguments(BaseModel):
    external_id: int | None = Field(default=None, alias="External id")
    correlation: int | None = None
    device: int | None = None


class TraceEvent(BaseModel):
    phase: str = Field(alias="ph")
    category: str = Field(default="", alias="cat")
    name: str = ""
    process_id: int | str = Field(default=0, alias="pid")
    thread_id: int | str = Field(default=0, alias="tid")
    start_microseconds: float = Field(default=0, alias="ts")
    duration_microseconds: float = Field(default=0, alias="dur", ge=0)
    arguments: TraceArguments = Field(default_factory=TraceArguments, alias="args")

    @property
    def end_microseconds(self) -> float:
        return self.start_microseconds + self.duration_microseconds


class ChromeTrace(BaseModel):
    events: list[TraceEvent] = Field(alias="traceEvents")
    base_time_nanoseconds: int = Field(default=0, alias="baseTimeNanoseconds")
    host_name: str = "local"


class ComponentCost(TypedDict):
    name: ComponentName
    scope_status: Literal["measured", "missing"]
    scope_count: int
    cpu_scope_milliseconds: float | None
    gpu_kernel_milliseconds: float | None
    cuda_launch_milliseconds: float | None
    cuda_synchronization_milliseconds: float | None
    gpu_transfer_milliseconds: float | None
    gpu_memset_milliseconds: float | None
    kernel_count: int
    transfer_count: int


class DeviceActivity(TypedDict):
    host_name: str
    device_id: int
    window_milliseconds: float
    gpu_busy_milliseconds: float
    gpu_idle_milliseconds: float
    gpu_kernel_milliseconds: float
    gpu_transfer_milliseconds: float
    gpu_memset_milliseconds: float
    window_source: Literal["component_scopes", "gpu_events"]


class ComponentReport(TypedDict):
    trace_paths: list[str]
    components: list[ComponentCost]
    devices: list[DeviceActivity]
    cpu_scope_busy_milliseconds: float
    gpu_kernel_milliseconds: float
    gpu_transfer_milliseconds: float
    gpu_memset_milliseconds: float
    unattributed_gpu_kernel_milliseconds: float
    unattributed_gpu_transfer_milliseconds: float
    unattributed_gpu_memset_milliseconds: float
    kernel_attribution_fraction: float | None
    transfer_attribution_fraction: float | None
    dominant_gpu_components: list[ComponentName]
    missing_components: list[ComponentName]
    timing_semantics: str


@dataclass(kw_only=True)
class ScopeIndex:
    scopes: list[TraceEvent]
    starts_microseconds: list[float]
    maximum_ends_microseconds: list[float]

    def containing_component(self, event: TraceEvent) -> ComponentName | None:
        scope_position = (
            bisect_right(self.starts_microseconds, event.start_microseconds) - 1
        )
        while scope_position >= 0:
            if self.maximum_ends_microseconds[scope_position] < event.end_microseconds:
                break
            else:
                scope = self.scopes[scope_position]
                if scope.end_microseconds >= event.end_microseconds:
                    return COMPONENT_NAME_LOOKUP[scope.name]
                else:
                    scope_position -= 1
        return None


COMPONENT_NAME_LOOKUP: dict[str, ComponentName] = {
    component_name: component_name for component_name in COMPONENT_NAMES
}
Interval = tuple[float, float]
ActivityLane = tuple[str, int | str, int | str]
ActivityIntervals = dict[ActivityLane, list[Interval]]
CpuThread = tuple[int | str, int | str]


def interval_milliseconds(intervals: list[Interval]) -> float:
    """Return the union duration of intervals on one clock and activity lane."""
    total_microseconds = 0.0
    previous_end_microseconds = float("-inf")
    for start_microseconds, end_microseconds in sorted(intervals):
        total_microseconds += max(
            0.0, end_microseconds - max(start_microseconds, previous_end_microseconds)
        )
        previous_end_microseconds = max(previous_end_microseconds, end_microseconds)
    return total_microseconds / MICROSECONDS_PER_MILLISECOND


def lane_milliseconds(intervals_by_lane: ActivityIntervals) -> float:
    return sum(
        interval_milliseconds(intervals) for intervals in intervals_by_lane.values()
    )


def build_component_report(trace_paths: list[Path]) -> ComponentReport:
    """Report inclusive CPU scopes and independently measured CUDA activity."""
    if not trace_paths:
        raise ValueError("At least one Chrome trace is required")
    else:
        traces: list[ChromeTrace] = []
        for trace_path in trace_paths:
            if trace_path.suffix == ".gz":
                serialized_trace = gzip.decompress(trace_path.read_bytes())
            else:
                serialized_trace = trace_path.read_bytes()
            traces.append(ChromeTrace.model_validate_json(serialized_trace))
    common_base_nanoseconds = min(trace.base_time_nanoseconds for trace in traces)
    components: dict[ComponentName, ComponentCost] = {}
    for component_name in COMPONENT_NAMES:
        components[component_name] = ComponentCost(
            name=component_name,
            scope_status="missing",
            scope_count=0,
            cpu_scope_milliseconds=None,
            gpu_kernel_milliseconds=None,
            cuda_launch_milliseconds=None,
            cuda_synchronization_milliseconds=None,
            gpu_transfer_milliseconds=None,
            gpu_memset_milliseconds=None,
            kernel_count=0,
            transfer_count=0,
        )
    component_intervals: dict[tuple[ComponentName, str], ActivityIntervals] = (
        defaultdict(lambda: defaultdict(list))
    )
    cpu_intervals: ActivityIntervals = defaultdict(list)
    gpu_intervals: dict[str, ActivityIntervals] = defaultdict(lambda: defaultdict(list))
    device_scope_intervals: dict[tuple[str, int], list[Interval]] = defaultdict(list)
    host_scope_intervals: dict[str, list[Interval]] = defaultdict(list)
    kernel_count = transfer_count = 0
    attributed_kernel_count = attributed_transfer_count = 0
    for trace_path, trace in zip(trace_paths, traces, strict=True):
        clock_offset_microseconds = (
            trace.base_time_nanoseconds - common_base_nanoseconds
        ) / 1000.0
        events = [event for event in trace.events if event.phase == "X"]
        scopes_by_thread: dict[CpuThread, list[TraceEvent]] = defaultdict(list)
        external_events: dict[tuple[int | str, int], TraceEvent] = {}
        runtimes_by_correlation: dict[int, list[TraceEvent]] = defaultdict(list)
        for event in events:
            event.start_microseconds += clock_offset_microseconds
            if (
                event.category in CPU_SCOPE_CATEGORIES
                and event.arguments.device is None
                and event.name in COMPONENT_NAME_LOOKUP
            ):
                component_name = COMPONENT_NAME_LOOKUP[event.name]
                scopes_by_thread[event.process_id, event.thread_id].append(event)
                components[component_name]["scope_status"] = "measured"
                components[component_name]["scope_count"] += 1
                cpu_lane = (str(trace_path), event.process_id, event.thread_id)
                interval = (event.start_microseconds, event.end_microseconds)
                component_intervals[component_name, "cpu_scope"][cpu_lane].append(
                    interval
                )
                cpu_intervals[cpu_lane].append(interval)
                host_scope_intervals[trace.host_name].append(interval)
            else:
                pass
            if (
                event.category in {"cpu_op", "user_annotation"}
                and event.arguments.device is None
                and event.arguments.external_id is not None
            ):
                external_events[event.process_id, event.arguments.external_id] = event
            else:
                pass
            if (
                event.category in {"cuda_runtime", "cuda_driver"}
                and event.arguments.correlation is not None
            ):
                runtimes_by_correlation[event.arguments.correlation].append(event)
            else:
                pass
        scope_indexes: dict[CpuThread, ScopeIndex] = {}
        for thread_lane, scopes in scopes_by_thread.items():
            scopes.sort(
                key=lambda event: (
                    event.start_microseconds,
                    -event.duration_microseconds,
                )
            )
            scope_indexes[thread_lane] = ScopeIndex(
                scopes=scopes,
                starts_microseconds=[scope.start_microseconds for scope in scopes],
                maximum_ends_microseconds=list(
                    accumulate((scope.end_microseconds for scope in scopes), max)
                ),
            )

        def cpu_component(event: TraceEvent) -> ComponentName | None:
            external_event = external_events.get(
                (event.process_id, event.arguments.external_id)
            )
            source_event = external_event if external_event is not None else event
            cpu_thread = (source_event.process_id, source_event.thread_id)
            scope_index = scope_indexes.get(cpu_thread)
            return (
                scope_index.containing_component(source_event)
                if scope_index is not None
                else None
            )

        for event in events:
            interval = (event.start_microseconds, event.end_microseconds)
            if event.category in {"cuda_runtime", "cuda_driver"}:
                component_name = cpu_component(event)
                if component_name is not None:
                    if "Launch" in event.name:
                        activity_name = "cuda_launch"
                    elif "Synchronize" in event.name:
                        activity_name = "cuda_synchronization"
                    else:
                        continue
                    cpu_lane = (str(trace_path), event.process_id, event.thread_id)
                    component_intervals[component_name, activity_name][cpu_lane].append(
                        interval
                    )
                else:
                    pass
            elif event.category in GPU_ACTIVITY_CATEGORIES:
                runtime_candidates = runtimes_by_correlation.get(
                    event.arguments.correlation, []
                )
                if event.arguments.external_id is not None:
                    matching_runtimes = [
                        runtime
                        for runtime in runtime_candidates
                        if runtime.arguments.external_id == event.arguments.external_id
                    ]
                    runtime_candidates = matching_runtimes or runtime_candidates
                else:
                    pass
                process_runtimes = [
                    runtime
                    for runtime in runtime_candidates
                    if runtime.process_id == event.process_id
                ]
                runtime_candidates = process_runtimes or runtime_candidates
                owners = {cpu_component(runtime) for runtime in runtime_candidates}
                if not runtime_candidates and event.arguments.external_id is not None:
                    owners = {
                        cpu_component(source_event)
                        for external_identifier, source_event in external_events.items()
                        if external_identifier[1] == event.arguments.external_id
                    }
                else:
                    pass
                component_name = next(iter(owners)) if len(owners) == 1 else None
                device_id = (
                    event.arguments.device if event.arguments.device is not None else 0
                )
                gpu_lane = (trace.host_name, device_id, "gpu")
                if event.category == "kernel":
                    activity_name = "gpu_kernel"
                    kernel_count += 1
                    attributed_kernel_count += component_name is not None
                    counter_name = "kernel_count"
                elif event.category == "gpu_memcpy":
                    activity_name = "gpu_transfer"
                    transfer_count += 1
                    attributed_transfer_count += component_name is not None
                    counter_name = "transfer_count"
                else:
                    activity_name = "gpu_memset"
                    counter_name = None
                gpu_intervals[activity_name][gpu_lane].append(interval)
                if component_name is not None:
                    component_intervals[component_name, activity_name][gpu_lane].append(
                        interval
                    )
                    if counter_name is not None:
                        components[component_name][counter_name] += 1
                    else:
                        pass
                    device_scope_intervals[trace.host_name, device_id].append(interval)
                else:
                    gpu_intervals[f"unattributed_{activity_name}"][gpu_lane].append(
                        interval
                    )
            else:
                pass
    for component_name, component in components.items():
        if component["scope_status"] == "measured":
            for activity_name in (
                "cpu_scope",
                "gpu_kernel",
                "cuda_launch",
                "cuda_synchronization",
                "gpu_transfer",
                "gpu_memset",
            ):
                component[f"{activity_name}_milliseconds"] = lane_milliseconds(
                    component_intervals[component_name, activity_name]
                )
        else:
            pass
    devices: list[DeviceActivity] = []
    for host_name, device_id in sorted(
        {
            (host_name, device_id)
            for intervals_by_lane in gpu_intervals.values()
            for host_name, device_id, lane_name in intervals_by_lane
        }
    ):
        kernel_intervals = gpu_intervals["gpu_kernel"][host_name, device_id, "gpu"]
        transfer_intervals = gpu_intervals["gpu_transfer"][host_name, device_id, "gpu"]
        memset_intervals = gpu_intervals["gpu_memset"][host_name, device_id, "gpu"]
        activity_intervals = kernel_intervals + transfer_intervals + memset_intervals
        measured_intervals = (
            host_scope_intervals[host_name]
            + device_scope_intervals[host_name, device_id]
        )
        window_intervals = measured_intervals or activity_intervals
        start_microseconds = min(interval[0] for interval in window_intervals)
        end_microseconds = max(interval[1] for interval in window_intervals)
        busy_milliseconds = interval_milliseconds(
            [
                (max(start_microseconds, start), min(end_microseconds, end))
                for start, end in activity_intervals
                if start < end_microseconds and end > start_microseconds
            ]
        )
        window_milliseconds = (
            end_microseconds - start_microseconds
        ) / MICROSECONDS_PER_MILLISECOND
        devices.append(
            DeviceActivity(
                host_name=host_name,
                device_id=device_id,
                window_milliseconds=window_milliseconds,
                gpu_busy_milliseconds=busy_milliseconds,
                gpu_idle_milliseconds=window_milliseconds - busy_milliseconds,
                gpu_kernel_milliseconds=interval_milliseconds(kernel_intervals),
                gpu_transfer_milliseconds=interval_milliseconds(transfer_intervals),
                gpu_memset_milliseconds=interval_milliseconds(memset_intervals),
                window_source=(
                    "component_scopes" if measured_intervals else "gpu_events"
                ),
            )
        )
    return ComponentReport(
        trace_paths=[str(trace_path) for trace_path in trace_paths],
        components=list(components.values()),
        devices=devices,
        cpu_scope_busy_milliseconds=lane_milliseconds(cpu_intervals),
        gpu_kernel_milliseconds=lane_milliseconds(gpu_intervals["gpu_kernel"]),
        gpu_transfer_milliseconds=lane_milliseconds(gpu_intervals["gpu_transfer"]),
        gpu_memset_milliseconds=lane_milliseconds(gpu_intervals["gpu_memset"]),
        unattributed_gpu_kernel_milliseconds=lane_milliseconds(
            gpu_intervals["unattributed_gpu_kernel"]
        ),
        unattributed_gpu_transfer_milliseconds=lane_milliseconds(
            gpu_intervals["unattributed_gpu_transfer"]
        ),
        unattributed_gpu_memset_milliseconds=lane_milliseconds(
            gpu_intervals["unattributed_gpu_memset"]
        ),
        kernel_attribution_fraction=(
            attributed_kernel_count / kernel_count if kernel_count else None
        ),
        transfer_attribution_fraction=(
            attributed_transfer_count / transfer_count if transfer_count else None
        ),
        dominant_gpu_components=[
            component["name"]
            for component in sorted(
                components.values(),
                key=lambda component: component["gpu_kernel_milliseconds"] or 0.0,
                reverse=True,
            )
            if component["gpu_kernel_milliseconds"]
        ],
        missing_components=[
            component["name"]
            for component in components.values()
            if component["scope_status"] == "missing"
        ],
        timing_semantics="Intervals are unioned per CPU thread or physical GPU. CPU scopes are inclusive; component costs may overlap and must not be summed. GPU transfers measure memcpy; memset is separate. Attribution fractions count events. GPU idle is a measured window gap, not CUDA launch or synchronization time.",
    )
