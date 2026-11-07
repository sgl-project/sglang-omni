# SPDX-License-Identifier: Apache-2.0
"""Regression for review finding P2: stop without run_id was a no-op.

The original bug:

1. ``POST /start_request_profile`` with no ``run_id`` generates one of the
   form ``run_<timestamp>``.
2. ``POST /stop_request_profile`` with no ``run_id`` used ``"default"`` as
   the value to broadcast.
3. Each stage's stop handler only stopped when its active run id matched
   the message — so a default-value stop was silently dropped, and the
   recorder kept writing.

The fix makes ``ProfilerStopMessage.run_id`` optional. ``None`` is a
wildcard: stop whatever's active.
"""

from __future__ import annotations

from sglang_omni.proto.messages import (
    ProfilerStartMessage,
    ProfilerStopMessage,
    parse_message,
)


def test_profiler_stop_message_supports_optional_run_id() -> None:
    """The proto must accept run_id=None and round-trip cleanly."""
    msg = ProfilerStopMessage()
    assert msg.run_id is None
    d = msg.to_dict()
    assert d == {"type": "profiler_stop", "run_id": None}
    round_trip = ProfilerStopMessage.from_dict(d)
    assert round_trip.run_id is None


def test_profiler_stop_message_legacy_run_id_round_trip() -> None:
    """Existing callers that DO pass a run id must continue to work."""
    msg = ProfilerStopMessage(run_id="abc")
    d = msg.to_dict()
    round_trip = ProfilerStopMessage.from_dict(d)
    assert round_trip.run_id == "abc"


def test_profiler_start_message_capacity_round_trip_and_legacy_payload() -> None:
    message = ProfilerStartMessage(
        run_id="performance_test",
        trace_path_template="",
        event_dir="/tmp/performance_test/events",
        enable_torch=False,
        event_buffer_capacity=1500,
    )

    payload = message.to_dict()
    round_trip = parse_message(payload)

    assert isinstance(round_trip, ProfilerStartMessage)
    assert round_trip.event_buffer_capacity == 1500

    legacy_message = parse_message(
        {
            "type": "profiler_start",
            "run_id": "legacy",
            "trace_path_template": "",
            "event_dir": "/tmp/legacy/events",
            "enable_torch": False,
        }
    )

    assert isinstance(legacy_message, ProfilerStartMessage)
    assert legacy_message.event_buffer_capacity is None
    assert "event_buffer_capacity" not in legacy_message.to_dict()


def test_runtime_profiler_start_message_round_trip() -> None:
    message = ProfilerStartMessage(
        run_id="runtime_test",
        trace_path_template="",
        event_dir="/tmp/runtime_test/events",
        enable_torch=False,
        event_buffer_capacity=1024,
        mode="runtime",
        sampling_unit="session",
        sample_rate=0.01,
        max_event_file_bytes=268435456,
    )

    round_trip = parse_message(message.to_dict())

    assert isinstance(round_trip, ProfilerStartMessage)
    assert round_trip.mode == "runtime"
    assert round_trip.sampling_unit == "session"
    assert round_trip.sample_rate == 0.01
    assert round_trip.max_event_file_bytes == 268435456
    assert round_trip.enable_torch is False
