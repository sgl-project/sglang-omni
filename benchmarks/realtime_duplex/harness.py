# SPDX-License-Identifier: Apache-2.0
"""Duplex session driver: paced input, concurrent output recording.

The driver runs one conversation session against any endpoint speaking
the realtime event protocol. It records everything it sends and receives
into a session log; scenario modes cover the lifecycle checks the
benchmark pins down: continuous interaction, explicit cancellation,
continued interaction after a cancel, and negative protocol probes.
"""

from __future__ import annotations

import asyncio
import base64
import datetime as dt
import json
import time
from dataclasses import dataclass, field
from typing import Any

import websockets

from benchmarks.realtime_duplex import schema
from benchmarks.realtime_duplex.session_log import (
    CLIENT_TO_SERVER,
    SERVER_TO_CLIENT,
    LogEntry,
    SessionManifest,
    sha256_of,
    write_session_log,
)

MODES = ("continuous", "cancel", "continue", "negative:unknown_event")
MALFORMED_EVENT_TYPE = "malformed.line"
"""Recorded type for a server frame that is not a JSON object."""


@dataclass(slots=True)
class SessionRecord:
    """Everything one session did; replay verdicts derive from this."""

    manifest: SessionManifest
    entries: list[LogEntry] = field(default_factory=list)
    client_error: str | None = None

    def save(self, path: str) -> None:
        write_session_log(path, self.manifest, self.entries)


class DuplexSessionDriver:
    """Drive one session; construct per run, call run() once."""

    def __init__(
        self,
        url: str,
        audio_pcm: bytes,
        *,
        mode: str = "continuous",
        frame_ms: int = schema.DEFAULT_FRAME_MS,
        pace: float = 1.0,
        model: str = "unknown",
        session_config: dict[str, Any] | None = None,
        idle_timeout_s: float = 10.0,
        quiet_drain_s: float = 0.2,
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if pace <= 0:
            raise ValueError("pace must be positive")
        self.url = url
        self.audio_pcm = audio_pcm
        self.mode = mode
        self.frame_ms = frame_ms
        self.pace = pace
        self.model = model
        self.session_config = session_config or dict(schema.DEFAULT_SESSION_CONFIG)
        self.idle_timeout_s = idle_timeout_s
        self.quiet_drain_s = quiet_drain_s

    async def run(self) -> SessionRecord:
        entries: list[LogEntry] = []
        client_error: str | None = None
        websocket = await websockets.connect(self.url, max_size=None)
        t0 = time.perf_counter()

        async def note(direction: str, event: dict[str, Any], audio_bytes: int = 0) -> None:
            entries.append(
                LogEntry(
                    t_s=time.perf_counter() - t0,
                    direction=direction,
                    event=event,
                    audio_bytes=audio_bytes,
                )
            )

        async def send(event: dict[str, Any]) -> None:
            audio = event.get("audio")
            audio_bytes = len(base64.b64decode(audio)) if isinstance(audio, str) else 0
            await note(CLIENT_TO_SERVER, event, audio_bytes)
            await websocket.send(json.dumps(event, ensure_ascii=False))

        async def pump(timeout_s: float) -> LogEntry | None:
            """Record one server event; None on quiet timeout."""
            try:
                raw = await asyncio.wait_for(websocket.recv(), timeout_s)
            except asyncio.TimeoutError:
                return None
            try:
                parsed = json.loads(raw)
            except ValueError:
                parsed = None
            if isinstance(parsed, dict):
                event = parsed
                delta = event.get("delta")
                audio_bytes = len(base64.b64decode(delta)) if isinstance(delta, str) else 0
            else:
                event = {"type": MALFORMED_EVENT_TYPE}
                audio_bytes = 0
            entry = LogEntry(
                t_s=time.perf_counter() - t0,
                direction=SERVER_TO_CLIENT,
                event=event,
                audio_bytes=audio_bytes,
            )
            entries.append(entry)
            return entry

        async def wait_for(event_type: str, timeout_s: float) -> LogEntry:
            deadline = time.monotonic() + timeout_s
            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"timed out waiting for {event_type}")
                entry = await pump(remaining)
                if entry is not None and entry.type == event_type:
                    return entry

        async def paced_sender() -> None:
            chunk = schema.frame_bytes(self.frame_ms)
            frame_s = self.frame_ms / 1000.0 * self.pace
            start = time.perf_counter()
            for index, offset in enumerate(range(0, len(self.audio_pcm), chunk)):
                delay = start + index * frame_s - time.perf_counter()
                if delay > 0:
                    await asyncio.sleep(delay)
                payload = base64.b64encode(self.audio_pcm[offset : offset + chunk]).decode("ascii")
                await send({"type": "input_audio_buffer.append", "audio": payload})

        manifest = SessionManifest(
            url=self.url,
            model=self.model,
            started_at_wall=dt.datetime.now(dt.timezone.utc).isoformat(),
            input_sha256=sha256_of(self.audio_pcm),
            input_duration_s=len(self.audio_pcm) / (schema.SAMPLE_RATE * schema.BYTES_PER_SAMPLE),
            frame_ms=self.frame_ms,
            pace=self.pace,
            scenario=self.mode,
            config=self.session_config,
        )
        sender: asyncio.Task[None] | None = None
        try:
            await wait_for("session.created", self.idle_timeout_s)
            await send({"type": "session.update", "session": self.session_config})
            await wait_for("session.updated", self.idle_timeout_s)

            if self.mode == "negative:unknown_event":
                await send({"type": "bogus.probe", "payload": "not in the protocol"})
                await wait_for("error", self.idle_timeout_s)
            else:
                cancelled = False
                open_responses = 0
                chunk = schema.frame_bytes(self.frame_ms)
                frames_total = max(1, -(-len(self.audio_pcm) // chunk))
                frame_s = self.frame_ms / 1000.0 * self.pace
                sender = asyncio.create_task(paced_sender())
                sender_end = time.perf_counter() + (frames_total - 1) * frame_s
                last_event_perf = time.perf_counter()
                while True:
                    # Exit once the input is fully sent and no response is in
                    # flight: a short quiet window covers the commit latency of
                    # a response triggered by the final frames.
                    quiet_exit = sender.done() and open_responses == 0
                    timeout = self.quiet_drain_s if quiet_exit else self.idle_timeout_s
                    if not quiet_exit and open_responses == 0:
                        # Cap the wait so the quiet-exit reassessment happens
                        # promptly once the sender's schedule runs out.
                        remaining = sender_end - time.perf_counter() + 2 * self.quiet_drain_s
                        timeout = max(2 * self.quiet_drain_s, min(timeout, remaining))
                    entry = await pump(timeout)
                    if entry is None:
                        # A capped wait expiring while the sender still runs is
                        # expected; only a genuinely eventless stretch past the
                        # idle budget is an error.
                        if quiet_exit:
                            break
                        if time.perf_counter() - last_event_perf > self.idle_timeout_s:
                            client_error = "idle timeout waiting for server events"
                            break
                        continue
                    last_event_perf = time.perf_counter()
                    if entry.type == "response.created":
                        open_responses += 1
                    elif entry.type == "response.done":
                        open_responses = max(0, open_responses - 1)
                    if entry.type == "response.audio.delta":
                        if self.mode in ("cancel", "continue") and not cancelled:
                            await send({"type": "response.cancel"})
                            cancelled = True
                    elif entry.type == "response.done":
                        status = str(entry.event.get("response", {}).get("status", ""))
                        if status == "cancelled" and self.mode == "cancel":
                            break
            if sender is not None and not sender.done():
                sender.cancel()
                try:
                    await sender
                except asyncio.CancelledError:
                    pass
            # Quiet drain: absorb anything still in flight before closing.
            drain_deadline = time.monotonic() + 2.0
            while time.monotonic() < drain_deadline:
                if await pump(self.quiet_drain_s) is None:
                    break
            await note(CLIENT_TO_SERVER, {"type": "client.close", "code": 1000})
        except (TimeoutError, websockets.exceptions.WebSocketException) as exc:
            client_error = f"{type(exc).__name__}: {exc}"
        finally:
            await websocket.close()
        return SessionRecord(manifest=manifest, entries=entries, client_error=client_error)
