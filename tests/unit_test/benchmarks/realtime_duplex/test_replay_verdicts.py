# SPDX-License-Identifier: Apache-2.0
"""Replay verdicts over synthetic logs: golden flow plus every negative
case the verdicts must catch (stdlib only)."""

from __future__ import annotations

import copy

from benchmarks.realtime_duplex.replay import replay
from benchmarks.realtime_duplex.session_log import LogEntry
from tests.unit_test.benchmarks.realtime_duplex.test_session_log import make_manifest

C2S = "c2s"
S2C = "s2c"


def line(
    direction: str, t_s: float, event_type: str, audio_bytes: int = 0, **fields: object
) -> LogEntry:
    event: dict[str, object] = {"type": event_type}
    event.update(fields)
    return LogEntry(t_s=t_s, direction=direction, event=event, audio_bytes=audio_bytes)


def golden_entries() -> list[LogEntry]:
    return [
        line(S2C, 0.001, "session.created"),
        line(C2S, 0.002, "session.update"),
        line(S2C, 0.003, "session.updated"),
        line(C2S, 0.080, "input_audio_buffer.append", audio_bytes=2560),
        line(C2S, 0.160, "input_audio_buffer.append", audio_bytes=2560),
        line(C2S, 0.240, "input_audio_buffer.append", audio_bytes=2560),
        line(S2C, 0.250, "input_audio_buffer.speech_started"),
        line(S2C, 0.251, "input_audio_buffer.speech_stopped"),
        line(S2C, 0.252, "input_audio_buffer.committed"),
        line(S2C, 0.253, "conversation.item.created"),
        line(S2C, 0.260, "response.created"),
        line(S2C, 0.280, "response.audio.delta", audio_bytes=2560),
        line(S2C, 0.360, "response.audio.delta", audio_bytes=2560),
        line(S2C, 0.440, "response.audio.delta", audio_bytes=2560),
        line(S2C, 0.441, "response.audio.done", output_audio_bytes=7680),
        line(S2C, 0.442, "response.done", response={"status": "completed"}),
        line(C2S, 0.500, "client.close", code=1000),
    ]


def verdict(report, name: str):
    return next(item for item in report.verdicts if item.name == name)


def with_cancel_sent(entries: list[LogEntry]) -> list[LogEntry]:
    """Cancel mid-response, after the first delta."""
    result = list(entries)
    result.insert(12, line(C2S, 0.300, "response.cancel"))
    return result


def test_golden_flow_passes_every_verdict():
    report = replay(make_manifest(), golden_entries())
    assert report.passed, report.summary
    profile = report.timing_profile["responses"][0]
    assert profile["delta_count"] == 3
    assert profile["first_delta_after_created_s"] == 0.280 - 0.260
    assert profile["underrun_gaps"] == 0


def test_delta_after_done_fails_response_lifecycle():
    entries = golden_entries()
    entries.insert(15, line(S2C, 0.4415, "response.audio.delta", audio_bytes=2560))
    report = replay(make_manifest(), entries)
    assert not verdict(report, "response_lifecycle").passed
    assert "delta after response" in verdict(report, "response_lifecycle").detail


def test_missing_response_done_fails():
    entries = [item for item in golden_entries() if item.type != "response.done"]
    report = replay(make_manifest(), entries)
    assert not verdict(report, "response_lifecycle").passed


def test_cancel_without_cancelled_status_fails():
    report = replay(make_manifest(), with_cancel_sent(golden_entries()))
    assert not verdict(report, "response_lifecycle").passed
    assert "cancelled by client" in verdict(report, "response_lifecycle").detail


def test_cancelled_response_with_status_passes():
    entries = with_cancel_sent(golden_entries())
    for item in entries:
        if item.type == "response.done":
            item.event["response"] = {"status": "cancelled"}
    report = replay(make_manifest(), entries)
    assert verdict(report, "response_lifecycle").passed
    assert verdict(report, "ack_closure").passed


def test_audio_byte_mismatch_fails_accounting():
    entries = golden_entries()
    for item in entries:
        if item.type == "response.audio.done":
            item.event["output_audio_bytes"] = 9999
    report = replay(make_manifest(), entries)
    assert not verdict(report, "byte_accounting").passed


def test_error_fails_clean_scenario_but_passes_negative_scenario():
    entries = golden_entries()
    entries.insert(
        4,
        line(S2C, 0.100, "error", error={"type": "server_error", "message": "boom"}),
    )
    clean = replay(make_manifest(), entries)
    assert not verdict(clean, "error_accounting").passed
    manifest = copy.deepcopy(make_manifest())
    manifest.scenario = "negative:abort"
    negative = replay(manifest, entries)
    assert verdict(negative, "error_accounting").passed


def test_server_event_after_close_fails_clean_shutdown():
    entries = golden_entries()
    entries.append(line(S2C, 0.600, "response.created"))
    report = replay(make_manifest(), entries)
    assert not verdict(report, "clean_shutdown").passed


def test_missing_client_close_fails_clean_shutdown():
    entries = [item for item in golden_entries() if item.type != "client.close"]
    report = replay(make_manifest(), entries)
    assert not verdict(report, "clean_shutdown").passed


def test_replay_is_deterministic():
    first = replay(make_manifest(), golden_entries()).to_dict()
    second = replay(make_manifest(), golden_entries()).to_dict()
    assert first == second
