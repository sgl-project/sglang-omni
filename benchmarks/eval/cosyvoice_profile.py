# SPDX-License-Identifier: Apache-2.0
"""Summarize CosyVoice first-audio boundaries and the tail after AR completion."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

from pydantic import BaseModel, Field

from sglang_omni.profiler.views import (
    assess_completeness,
    event_paths,
    percentile,
    reconstruct_timelines,
    stage_breakdown,
)

FIRST_AUDIO_BOUNDARIES = (
    ("coordinator", "request_admission"),
    ("preprocessing", "stage_complete"),
    ("tts_engine", "scheduler_prefill_start"),
    ("tts_engine", "stage_first_stream_chunk_sent"),
    ("vocoder", "vocoder_launch_start"),
    ("vocoder", "vocoder_flow_submitted"),
    ("vocoder", "vocoder_hift_submitted"),
    ("vocoder", "vocoder_audio_ready"),
    ("coordinator", "coordinator_stream_received"),
)
FIRST_AUDIO_PHASES = (
    "admission_to_preprocessing_complete",
    "preprocessing_complete_to_prefill_start",
    "prefill_start_to_first_token_chunk",
    "first_token_chunk_to_vocoder_launch",
    "vocoder_launch_to_flow_submitted",
    "flow_submitted_to_hift_submitted",
    "hift_submitted_to_audio_ready",
    "audio_ready_to_coordinator",
)


class ProfileMetadata(BaseModel):
    batch_size: int | None = None
    ready_since_seconds: float | None = None
    monotonic_seconds: float | None = None


class ProfileEvent(BaseModel):
    request_id: str
    stage: str
    event_name: str
    timestamp_ns: int
    run_id: str | None = None
    metadata: ProfileMetadata = Field(default_factory=ProfileMetadata)


def request_durations_milliseconds(events: list[ProfileEvent]) -> dict[str, float]:
    """Use a telescoping first-output path; AR residence overlaps vocoding."""
    first_timestamps_nanoseconds: dict[tuple[str, str], int] = {}
    ordered_events = sorted(events, key=lambda event: event.timestamp_ns)
    for event in ordered_events:
        first_timestamps_nanoseconds.setdefault(
            (event.stage, event.event_name), event.timestamp_ns
        )
    durations_milliseconds: dict[str, float] = {}
    if all(
        boundary in first_timestamps_nanoseconds for boundary in FIRST_AUDIO_BOUNDARIES
    ):
        timestamps_nanoseconds = [
            first_timestamps_nanoseconds[boundary]
            for boundary in FIRST_AUDIO_BOUNDARIES
        ]
        if any(
            end_nanoseconds < start_nanoseconds
            for start_nanoseconds, end_nanoseconds in zip(
                timestamps_nanoseconds, timestamps_nanoseconds[1:]
            )
        ):
            raise ValueError("First-audio boundaries are not causally ordered")
        else:
            for phase, start_nanoseconds, end_nanoseconds in zip(
                FIRST_AUDIO_PHASES, timestamps_nanoseconds, timestamps_nanoseconds[1:]
            ):
                durations_milliseconds[phase] = (
                    end_nanoseconds - start_nanoseconds
                ) / 1e6
            durations_milliseconds["first_audio_internal"] = (
                timestamps_nanoseconds[-1] - timestamps_nanoseconds[0]
            ) / 1e6
    else:
        pass
    first_launch_event = next(
        (
            event
            for event in ordered_events
            if event.event_name == "vocoder_launch_start" and event.stage == "vocoder"
        ),
        None,
    )
    if first_launch_event is not None:
        ready_seconds = first_launch_event.metadata.ready_since_seconds
        launch_seconds = first_launch_event.metadata.monotonic_seconds
        if ready_seconds is not None and launch_seconds is not None:
            durations_milliseconds["first_vocoder_ready_wait"] = (
                launch_seconds - ready_seconds
            ) * 1000
        else:
            pass
    else:
        pass
    admission_nanoseconds = first_timestamps_nanoseconds.get(
        ("coordinator", "request_admission")
    )
    terminal_nanoseconds = first_timestamps_nanoseconds.get(
        ("coordinator", "terminal_response")
    )
    ar_start_nanoseconds = first_timestamps_nanoseconds.get(
        ("tts_engine", "scheduler_prefill_start")
    )
    ar_end_nanoseconds = first_timestamps_nanoseconds.get(
        ("tts_engine", "stage_complete")
    )
    if None not in (
        admission_nanoseconds,
        terminal_nanoseconds,
        ar_start_nanoseconds,
        ar_end_nanoseconds,
    ):
        assert admission_nanoseconds is not None and terminal_nanoseconds is not None
        assert ar_start_nanoseconds is not None and ar_end_nanoseconds is not None
        if (
            not admission_nanoseconds
            <= ar_start_nanoseconds
            <= ar_end_nanoseconds
            <= terminal_nanoseconds
        ):
            raise ValueError("AR completion boundaries are not causally ordered")
        else:
            durations_milliseconds["before_ar"] = (
                ar_start_nanoseconds - admission_nanoseconds
            ) / 1e6
            durations_milliseconds["ar_residence"] = (
                ar_end_nanoseconds - ar_start_nanoseconds
            ) / 1e6
            durations_milliseconds["after_ar_tail"] = (
                terminal_nanoseconds - ar_end_nanoseconds
            ) / 1e6
            durations_milliseconds["request_internal"] = (
                terminal_nanoseconds - admission_nanoseconds
            ) / 1e6
    else:
        pass
    return durations_milliseconds


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--skip-first-requests",
        type=int,
        default=0,
        help="Exclude warmup requests in admission order.",
    )
    arguments = parser.parse_args()
    if arguments.skip_first_requests < 0:
        parser.error("--skip-first-requests must be nonnegative")
    else:
        pass
    completeness = assess_completeness(arguments.events)
    if not completeness["complete"]:
        raise ValueError(f"Incomplete event storage: {completeness}")
    else:
        pass
    paths = event_paths(arguments.events)
    requests: dict[str, list[ProfileEvent]] = defaultdict(list)
    run_ids: set[str | None] = set()
    for path in paths:
        for line in path.read_text().splitlines():
            event = ProfileEvent.model_validate_json(line)
            requests[event.request_id].append(event)
            run_ids.add(event.run_id)
    if len(run_ids) != 1:
        raise ValueError("Analyze one profiling run at a time")
    else:
        pass
    admissions = sorted(
        (event.timestamp_ns, request_id)
        for request_id, events in requests.items()
        for event in events
        if event.stage == "coordinator" and event.event_name == "request_admission"
    )
    if arguments.skip_first_requests >= len(admissions):
        raise ValueError("No admitted requests remain after excluding warmup")
    else:
        pass
    for _, request_id in admissions[: arguments.skip_first_requests]:
        requests.pop(request_id)
    timelines = {
        request_id: timeline
        for request_id, timeline in reconstruct_timelines(arguments.events).items()
        if request_id in requests
    }
    measurements: dict[str, list[float]] = defaultdict(list)
    request_measurements: dict[str, dict[str, float]] = {}
    for request_id, events in requests.items():
        durations_milliseconds = request_durations_milliseconds(events)
        request_measurements[request_id] = durations_milliseconds
        for phase, duration_milliseconds in durations_milliseconds.items():
            measurements[phase].append(duration_milliseconds)
    summaries: dict[str, dict[str, int | float]] = {}
    for phase, durations_milliseconds in measurements.items():
        if durations_milliseconds:
            ordered_milliseconds = sorted(durations_milliseconds)
            summaries[phase] = {
                "count": len(ordered_milliseconds),
                "mean_ms": mean(ordered_milliseconds),
                "p50_ms": percentile(ordered_milliseconds, 0.5),
                "p95_ms": percentile(ordered_milliseconds, 0.95),
                "maximum_ms": ordered_milliseconds[-1],
            }
        else:
            pass
    vocoder_batches = {
        event.timestamp_ns: event.metadata.batch_size
        for events in requests.values()
        for event in events
        if event.stage == "vocoder"
        and event.event_name == "vocoder_launch_start"
        and event.metadata.batch_size is not None
    }
    report = {
        "completeness": completeness,
        "skipped_initial_requests": arguments.skip_first_requests,
        "requests": len(requests),
        "first_audio_complete_requests": len(measurements["first_audio_internal"]),
        "terminal_complete_requests": len(measurements["request_internal"]),
        "duration_summaries": summaries,
        "vocoder_batch_size_counts": dict(Counter(vocoder_batches.values())),
        "stage_breakdown": [row.to_dict() for row in stage_breakdown(timelines)],
        "request_durations_ms": request_measurements,
        "interpretation": (
            "Host timestamps on one machine. Submission spans include host work and CUDA waits; "
            "submitted-to-ready includes deferred completion and is not pure GPU time. "
            "AR residence overlaps vocoding. Stage intervals count every batch participant. "
            "Sum phase means only over a common request set; "
            "do not sum phase percentiles. HTTP transport lies outside these boundaries."
        ),
    }
    arguments.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
else:
    pass
