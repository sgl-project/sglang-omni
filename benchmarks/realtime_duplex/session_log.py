# SPDX-License-Identifier: Apache-2.0
"""Session log: the on-disk record a replay verdicts run against.

One JSON object per line. The first line is the manifest (kind: manifest)
pinning identities and run settings; every following line is one event
with its direction and the session-relative monotonic timestamp it was
observed at. Replay reads only this file, so verdicts are reproducible
without the server, the audio, or the wall clock.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

MANIFEST_KIND = "manifest"
ENTRY_KIND = "entry"
CLIENT_TO_SERVER = "c2s"
SERVER_TO_CLIENT = "s2c"


@dataclass(slots=True)
class SessionManifest:
    """Pinned identities and settings for one recorded session."""

    url: str
    model: str
    started_at_wall: str
    input_sha256: str
    input_duration_s: float
    frame_ms: int
    pace: float
    scenario: str
    client: str = "duplex-harness"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class LogEntry:
    """One observed event, client-to-server or server-to-client."""

    t_s: float
    """Session-relative perf_counter seconds (t0 = WebSocket connect)."""
    direction: str
    event: dict[str, Any]
    audio_bytes: int = 0
    """Decoded audio payload size when the event carries one, else 0."""

    @property
    def type(self) -> str:
        return str(self.event.get("type", ""))


def sha256_of(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_session_log(path: str | Path, manifest: SessionManifest, entries: list[LogEntry]) -> None:
    """Write manifest + entries as JSONL. Untimely mid-write crashes leave
    a truncated file; the reader rejects those instead of guessing."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "w", encoding="utf-8") as handle:
        header = asdict(manifest)
        header["kind"] = MANIFEST_KIND
        handle.write(json.dumps(header, ensure_ascii=False) + "\n")
        for entry in entries:
            line = {
                "kind": ENTRY_KIND,
                "t_s": entry.t_s,
                "direction": entry.direction,
                "event": entry.event,
                "audio_bytes": entry.audio_bytes,
            }
            handle.write(json.dumps(line, ensure_ascii=False) + "\n")


def read_session_log(path: str | Path) -> tuple[SessionManifest, list[LogEntry]]:
    source = Path(path)
    with open(source, encoding="utf-8") as handle:
        lines = handle.readlines()
    if not lines:
        raise ValueError(f"session log is empty: {source}")
    header = json.loads(lines[0])
    if header.get("kind") != MANIFEST_KIND:
        raise ValueError(f"first line is not a manifest: {source}")
    header.pop("kind")
    manifest = SessionManifest(**header)
    entries: list[LogEntry] = []
    for lineno, line in enumerate(lines[1:], start=2):
        if not line.strip():
            raise ValueError(f"blank line {lineno} in session log: {source}")
        record = json.loads(line)
        if record.get("kind") != ENTRY_KIND:
            raise ValueError(f"line {lineno} is not an entry: {source}")
        entries.append(
            LogEntry(
                t_s=float(record["t_s"]),
                direction=record["direction"],
                event=record["event"],
                audio_bytes=int(record.get("audio_bytes", 0)),
            )
        )
    return manifest, entries
