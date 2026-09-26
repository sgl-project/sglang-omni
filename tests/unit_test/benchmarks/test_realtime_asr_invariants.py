# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import pytest

from benchmarks.realtime_asr.client import ReceivedEvent, SessionTrace
from benchmarks.realtime_asr.metrics import check_invariants


@pytest.fixture
def valid_events() -> list[dict[str, Any]]:
    return [
        {"type": "session.created", "session": {}, "event_index": 1},
        {
            "type": "input_audio_buffer.committed",
            "segment_id": 0,
            "event_index": 2,
        },
        {
            "type": "transcription.segment",
            "segment_id": 0,
            "text": "hello",
            "is_final": True,
            "event_index": 3,
        },
        {"type": "transcription.completed", "text": "hello", "event_index": 4},
    ]


def trace_from_events(events: list[dict[str, Any]]) -> SessionTrace:
    return SessionTrace(
        url="ws://example.test/v1/realtime?intent=transcription",
        received=[
            ReceivedEvent(recv_s=position / 10, event=event)
            for position, event in enumerate(events)
        ],
    )


def test_valid_committed_final_completed_order(
    valid_events: list[dict[str, Any]],
) -> None:
    assert check_invariants(trace_from_events(valid_events)) == []


@pytest.mark.parametrize("position", range(4))
def test_missing_event_index_fails_even_when_other_indexes_increase(
    valid_events: list[dict[str, Any]], position: int
) -> None:
    del valid_events[position]["event_index"]

    violations = check_invariants(trace_from_events(valid_events))

    assert any("event_index missing or non-integer" in item for item in violations)


def test_all_missing_event_indexes_fail(valid_events: list[dict[str, Any]]) -> None:
    for event in valid_events:
        del event["event_index"]

    violations = check_invariants(trace_from_events(valid_events))

    assert any("event_index missing or non-integer" in item for item in violations)


@pytest.mark.parametrize("event_index", [None, "1", 1.0, True, False])
def test_non_integer_event_index_fails(
    valid_events: list[dict[str, Any]], event_index: Any
) -> None:
    valid_events[0]["event_index"] = event_index

    violations = check_invariants(trace_from_events(valid_events))

    assert any("event_index missing or non-integer" in item for item in violations)


def test_duplicate_completed_fails_with_increasing_indexes(
    valid_events: list[dict[str, Any]],
) -> None:
    valid_events.append(
        {"type": "transcription.completed", "text": "hello", "event_index": 5}
    )

    violations = check_invariants(trace_from_events(valid_events))

    assert any("duplicate transcription.completed" in item for item in violations)


def test_partial_after_completed_fails_for_a_different_segment(
    valid_events: list[dict[str, Any]],
) -> None:
    valid_events.append(
        {
            "type": "transcription.segment",
            "segment_id": 1,
            "text": "late",
            "is_final": False,
            "event_index": 5,
        }
    )

    violations = check_invariants(trace_from_events(valid_events))

    assert any("after transcription.completed" in item for item in violations)


def test_completed_before_final_fails_despite_matching_segment_ids(
    valid_events: list[dict[str, Any]],
) -> None:
    final = valid_events[2]
    completed = valid_events[3]
    final["event_index"] = 4
    completed["event_index"] = 3
    valid_events[2:] = [completed, final]

    violations = check_invariants(trace_from_events(valid_events))

    assert any("after transcription.completed" in item for item in violations)


def test_final_before_committed_fails_despite_matching_segment_ids(
    valid_events: list[dict[str, Any]],
) -> None:
    committed = valid_events[1]
    final = valid_events[2]
    committed["event_index"] = 3
    final["event_index"] = 2
    valid_events[1:3] = [final, committed]

    violations = check_invariants(trace_from_events(valid_events))

    assert any("before its committed event" in item for item in violations)


def test_session_control_event_after_completed_is_not_transcription_output(
    valid_events: list[dict[str, Any]],
) -> None:
    valid_events.append({"type": "session.updated", "session": {}, "event_index": 5})

    assert check_invariants(trace_from_events(valid_events)) == []


def test_empty_transcription_can_complete_without_segments() -> None:
    events = [
        {"type": "session.created", "session": {}, "event_index": 1},
        {"type": "transcription.completed", "text": "", "event_index": 2},
    ]

    assert check_invariants(trace_from_events(events)) == []
