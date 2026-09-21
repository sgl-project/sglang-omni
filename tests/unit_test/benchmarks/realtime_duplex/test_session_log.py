# SPDX-License-Identifier: Apache-2.0
"""Session-log round trip and strict-reader checks (stdlib only)."""

from __future__ import annotations

import json

import pytest

from benchmarks.realtime_duplex.session_log import (
    LogEntry,
    SessionManifest,
    read_session_log,
    write_session_log,
)


def make_manifest() -> SessionManifest:
    return SessionManifest(
        url="ws://127.0.0.1:8/v1/realtime?intent=conversation",
        model="mock-duplex",
        started_at_wall="2026-09-21T00:00:00+00:00",
        input_sha256="0" * 64,
        input_duration_s=0.64,
        frame_ms=80,
        pace=1.0,
        scenario="continuous",
    )


def make_entries() -> list[LogEntry]:
    return [
        LogEntry(t_s=0.001, direction="s2c", event={"type": "session.created"}),
        LogEntry(t_s=0.002, direction="c2s", event={"type": "session.update"}),
        LogEntry(t_s=0.003, direction="s2c", event={"type": "session.updated"}),
        LogEntry(
            t_s=0.010,
            direction="c2s",
            event={"type": "input_audio_buffer.append", "audio": "VQ=="},
            audio_bytes=1,
        ),
        LogEntry(t_s=0.900, direction="c2s", event={"type": "client.close", "code": 1000}),
    ]


def test_round_trip(tmp_path):
    path = tmp_path / "session.jsonl"
    write_session_log(path, make_manifest(), make_entries())
    manifest, entries = read_session_log(path)
    assert manifest == make_manifest()
    assert [entry.type for entry in entries] == [
        "session.created",
        "session.update",
        "session.updated",
        "input_audio_buffer.append",
        "client.close",
    ]
    assert entries[3].audio_bytes == 1


def test_manifest_is_first_line(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text(json.dumps({"type": "session.created"}) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="not a manifest"):
        read_session_log(path)


def test_blank_line_rejected(tmp_path):
    path = tmp_path / "session.jsonl"
    write_session_log(path, make_manifest(), make_entries()[:2])
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    with pytest.raises(ValueError, match="blank line"):
        read_session_log(path)


def test_empty_log_rejected(tmp_path):
    path = tmp_path / "session.jsonl"
    path.write_text("", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        read_session_log(path)
