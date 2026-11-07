# SPDX-License-Identifier: Apache-2.0
"""Critical-path accounting for overlapping speech generation and vocoding."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from benchmarks.eval.cosyvoice_profile import (
    FIRST_AUDIO_BOUNDARIES,
    FIRST_AUDIO_PHASES,
    ProfileEvent,
    main,
    request_durations_milliseconds,
)
from sglang_omni.profiler.event_recorder_writer import RequestEventRecorder


def test_first_audio_phases_telescope_and_ar_tail_excludes_overlap() -> None:
    timestamps_milliseconds = [0, 10, 15, 50, 55, 70, 80, 90, 95]
    events = [
        ProfileEvent(
            request_id="request",
            stage=stage,
            event_name=name,
            timestamp_ns=timestamp_milliseconds * 1_000_000,
        )
        for (stage, name), timestamp_milliseconds in zip(
            FIRST_AUDIO_BOUNDARIES, timestamps_milliseconds
        )
    ]
    events.extend(
        [
            ProfileEvent(
                request_id="request",
                stage="tts_engine",
                event_name="stage_complete",
                timestamp_ns=200_000_000,
            ),
            ProfileEvent(
                request_id="request",
                stage="coordinator",
                event_name="terminal_response",
                timestamp_ns=230_000_000,
            ),
            ProfileEvent(
                request_id="request",
                stage="vocoder",
                event_name="vocoder_audio_ready",
                timestamp_ns=220_000_000,
            ),
        ]
    )
    durations = request_durations_milliseconds(list(reversed(events)))
    assert sum(durations[phase] for phase in FIRST_AUDIO_PHASES) == 95
    assert durations["first_audio_internal"] == 95
    assert durations["ar_residence"] == 185
    assert durations["after_ar_tail"] == 30
    assert (
        durations["before_ar"] + durations["ar_residence"] + durations["after_ar_tail"]
        == durations["request_internal"]
    )


def test_incomplete_profile_does_not_fabricate_durations() -> None:
    events = [
        ProfileEvent(
            request_id="request",
            stage="coordinator",
            event_name="request_admission",
            timestamp_ns=0,
        )
    ]
    assert request_durations_milliseconds(events) == {}


def test_out_of_order_first_audio_path_is_rejected() -> None:
    timestamps_nanoseconds = [0, 10, 20, 30, 40, 50, 60, 80, 70]
    events = [
        ProfileEvent(
            request_id="request",
            stage=stage,
            event_name=name,
            timestamp_ns=timestamp_nanoseconds,
        )
        for (stage, name), timestamp_nanoseconds in zip(
            FIRST_AUDIO_BOUNDARIES, timestamps_nanoseconds
        )
    ]
    with pytest.raises(ValueError, match="causally ordered"):
        request_durations_milliseconds(events)


def test_recorder_overflow_cannot_produce_an_attribution_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An overflowing diagnostic previously summarized a biased request subset."""
    recorder = RequestEventRecorder()
    recorder.start("overflow", str(tmp_path), "coordinator", event_buffer_capacity=1)
    for event_name in ["request_admission", "terminal_response"]:
        recorder.emit(request_id="request", stage="coordinator", event_name=event_name)
    recorder.stop(wait=True)
    output = tmp_path / "attribution.json"
    monkeypatch.setattr(
        sys, "argv", ["cosyvoice_profile", str(tmp_path), "--output", str(output)]
    )
    with pytest.raises(ValueError, match="Incomplete event storage"):
        main()
    assert not output.exists()
