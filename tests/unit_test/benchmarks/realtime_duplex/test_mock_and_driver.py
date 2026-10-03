# SPDX-License-Identifier: Apache-2.0
"""Driver-versus-mock end-to-end runs on a localhost websocket.

These cover the lifecycle scenarios the benchmark pins: continuous
interaction, explicit cancellation, continued interaction after cancel,
negative protocol probes, malformed server frames, and mid-session
aborts. Pure CPU; no model or GPU.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from benchmarks.realtime_duplex import schema
from benchmarks.realtime_duplex.harness import MALFORMED_EVENT_TYPE, DuplexSessionDriver
from benchmarks.realtime_duplex.mock_server import DuplexMockServer, MockBehavior
from benchmarks.realtime_duplex.replay import group_responses, replay
from benchmarks.realtime_duplex.session_log import read_session_log

GOLDEN = Path(__file__).parent / "golden_continuous.jsonl"


def make_audio(frames: int) -> bytes:
    return b"\x55\xaa" * (schema.frame_bytes() * frames // 2)


def run_session(behavior: MockBehavior, **driver_kwargs: object) -> object:
    async def once() -> object:
        async with DuplexMockServer(behavior) as server:
            driver = DuplexSessionDriver(server.url, make_audio(6), **driver_kwargs)
            return await driver.run()

    return asyncio.run(once())


def test_continuous_session_passes_all_verdicts():
    record = run_session(MockBehavior())
    assert record.client_error is None
    report = replay(record.manifest, record.entries)
    assert report.passed, report.summary
    assert report.counts["s2c.response.audio.delta"] == 3
    assert report.counts["c2s.input_audio_buffer.append"] == 6


def test_cancel_lifecycle_closes_response_as_cancelled():
    record = run_session(MockBehavior(), mode="cancel")
    assert record.client_error is None
    report = replay(record.manifest, record.entries)
    assert report.passed, report.summary
    dones = [
        entry.event.get("response", {}).get("status")
        for entry in record.entries
        if entry.type == "response.done"
    ]
    assert dones == ["cancelled"]


def test_continue_after_cancel_gets_second_response():
    record = run_session(MockBehavior(), mode="continue")
    assert record.client_error is None
    report = replay(record.manifest, record.entries)
    assert report.passed, report.summary
    windows = group_responses(record.entries)
    assert len(windows) == 2
    statuses = [
        window.done.event.get("response", {}).get("status") if window.done else None
        for window in windows
    ]
    assert statuses == ["cancelled", "completed"]


def test_negative_unknown_event_is_expected_error():
    record = run_session(MockBehavior(), mode="negative:unknown_event")
    assert record.client_error is None
    report = replay(record.manifest, record.entries)
    assert report.passed, report.summary
    assert len(report.errors) == 1
    assert report.errors[0]["error"]["type"] == "invalid_request_error"


def test_malformed_server_line_is_recorded_not_fatal():
    record = run_session(MockBehavior(send_malformed_on_session=True))
    assert record.client_error is None
    types = {entry.type for entry in record.entries}
    assert MALFORMED_EVENT_TYPE in types
    report = replay(record.manifest, record.entries)
    assert report.passed, report.summary


def test_server_abort_is_captured_with_complete_accounting(tmp_path):
    record = run_session(MockBehavior(abort_close_code=1011))
    assert record.client_error is not None
    path = tmp_path / "abort.jsonl"
    record.save(path)
    manifest, entries = read_session_log(path)
    assert manifest.scenario == "continuous"
    assert entries, "aborted session must still account for what happened"


def test_golden_fixture_replays_clean():
    manifest, entries = read_session_log(GOLDEN)
    report = replay(manifest, entries)
    assert report.passed, report.summary
    assert report.timing_profile["responses"][0]["delta_count"] == 3
    assert report.counts["c2s.input_audio_buffer.append"] == 3
