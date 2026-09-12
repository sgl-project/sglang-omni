# SPDX-License-Identifier: Apache-2.0
"""One realtime transcription session: push audio, collect events, record time.

This module only records. For computing any metric, see
benchmarks.realtime_asr.metrics for datails.

The event flow:

    session.created  <-
    session.update   ->  (language, turn_detection)
    session.updated  <-
    input_audio_buffer.append x N  ->   paced to wall-clock when “paced”
    input_audio_buffer.commit      ->   manual mode only
    transcription.done             ->
    ... transcription.completed    <-   terminal event
"""

from __future__ import annotations

import asyncio
import base64
import json
import time
from dataclasses import dataclass, field
from typing import Any

import websockets

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2
TERMINAL_EVENT = "transcription.completed"
DEFAULT_TURN_DETECTION: dict[str, Any] = {"type": "server_vad"}
"""Server default; pass ``turn_detection=None`` to disable VAD (manual mode)."""


@dataclass(slots=True)
class SentPacket:
    """One input_audio_buffer.append as observed by the sender."""

    send_s: float
    """perf_counter right before the packet was handed to the socket."""
    audio_end_samples: int
    """Cumulative samples sent once this packet is included. Integer so the
    refresh-point lookup in metrics never suffers float accumulation."""
    num_bytes: int

    @property
    def audio_end_s(self) -> float:
        return self.audio_end_samples / SAMPLE_RATE


@dataclass(slots=True)
class ReceivedEvent:
    recv_s: float
    event: dict[str, Any]

    @property
    def type(self) -> str:
        return str(self.event.get("type", ""))


@dataclass(slots=True)
class SessionTrace:
    """Everything one session did, with timestamps, and nothing derived."""

    url: str
    session: dict[str, Any] = field(default_factory=dict)
    """The session payload from session.created/session.updated."""
    sent: list[SentPacket] = field(default_factory=list)
    received: list[ReceivedEvent] = field(default_factory=list)
    audio_duration_s: float = 0.0
    """Seconds of audio the client intended to send (excluding padding)."""
    connect_s: float | None = None
    """perf_counter when the WebSocket handshake completed."""
    first_send_s: float | None = None
    commit_sent_s: float | None = None
    done_sent_s: float | None = None
    end_s: float | None = None
    """perf_counter when the terminal event arrived or the session gave up."""
    error: str | None = None
    """Client-side failure (timeout, transport). Server error events stay in received."""

    # -- convenience views (no derivation, only filtering) -----------------

    def events(self, event_type: str) -> list[ReceivedEvent]:
        return [item for item in self.received if item.type == event_type]

    def segments(self, *, is_final: bool) -> list[ReceivedEvent]:
        return [
            item
            for item in self.events("transcription.segment")
            if bool(item.event.get("is_final")) is is_final
        ]

    @property
    def completed_text(self) -> str | None:
        completed = self.events(TERMINAL_EVENT)
        if not completed:
            return None
        return str(completed[-1].event.get("text", ""))

    @property
    def wall_s(self) -> float | None:
        if self.first_send_s is None or self.end_s is None:
            return None
        return self.end_s - self.first_send_s


def realtime_url(host: str, port: int) -> str:
    return f"ws://{host}:{port}/v1/realtime?intent=transcription"


def split_packets(pcm: bytes, *, packet_ms: int) -> list[bytes]:
    packet_bytes = SAMPLE_RATE * packet_ms // 1000 * BYTES_PER_SAMPLE
    if packet_bytes <= 0:
        raise ValueError("packet_ms must yield at least one sample")
    return [
        pcm[offset : offset + packet_bytes]
        for offset in range(0, len(pcm), packet_bytes)
    ]


async def _send_event(websocket: Any, event: dict[str, Any]) -> None:
    await websocket.send(json.dumps(event))


async def _recv_event(websocket: Any) -> dict[str, Any]:
    return json.loads(await websocket.recv())


async def _wait_for(trace: SessionTrace, websocket: Any, event_type: str) -> dict:
    """Receive until ``event_type``; records every event on the way."""
    while True:
        event = await _recv_event(websocket)
        trace.received.append(ReceivedEvent(recv_s=time.perf_counter(), event=event))
        if event.get("type") == event_type:
            return event
        if event.get("type") == "error":
            raise RuntimeError(f"server error while waiting for {event_type}: {event}")


async def _sender(
    trace: SessionTrace,
    websocket: Any,
    packets: list[bytes],
    *,
    packet_ms: int,
    paced: bool,
    manual_commit: bool,
) -> None:
    packet_s = packet_ms / 1000.0
    audio_end_samples = 0
    t0 = time.perf_counter()
    trace.first_send_s = t0
    for index, packet in enumerate(packets):
        if paced:
            # Absolute schedule: packet i leaves at t0 + i * packet_s, so
            # scheduler jitter does not accumulate into drift.
            delay = t0 + index * packet_s - time.perf_counter()
            if delay > 0:
                await asyncio.sleep(delay)
        audio_end_samples += len(packet) // BYTES_PER_SAMPLE
        send_s = time.perf_counter()
        await _send_event(
            websocket,
            {
                "type": "input_audio_buffer.append",
                "audio": base64.b64encode(packet).decode(),
            },
        )
        trace.sent.append(
            SentPacket(
                send_s=send_s,
                audio_end_samples=audio_end_samples,
                num_bytes=len(packet),
            )
        )
    if manual_commit:
        trace.commit_sent_s = time.perf_counter()
        await _send_event(websocket, {"type": "input_audio_buffer.commit"})
    trace.done_sent_s = time.perf_counter()
    await _send_event(websocket, {"type": "transcription.done"})


async def _receiver(trace: SessionTrace, websocket: Any) -> None:
    while True:
        event = await _recv_event(websocket)
        trace.received.append(ReceivedEvent(recv_s=time.perf_counter(), event=event))
        if event.get("type") == TERMINAL_EVENT:
            return


async def run_session(
    url: str,
    pcm: bytes,
    *,
    packet_ms: int = 200,
    paced: bool = True,
    turn_detection: dict[str, Any] | None = DEFAULT_TURN_DETECTION,
    manual_commit: bool = False,
    language: str | None = None,
    trailing_silence_ms: int = 0,
    timeout_s: float = 120.0,
) -> SessionTrace:
    """Stream pcm (mono 16 kHz PCM16) through one realtime session.

    Args:
        packet_ms: audio per input_audio_buffer.append.
        paced: send packet *i* at t0 + i * packet_ms; False sends as
            fast as the socket accepts (throughput only, latencies are then
            meaningless).
        turn_detection: the session.update value; defaults to server VAD.
            None disables VAD, and the caller then usually wants manual_commit=True.
        trailing_silence_ms: zero samples appended after pcm so server VAD
            can close the last turn on its own instead of at transcription.done.
        timeout_s: bound on the whole session, from connect to the terminal event.

    Never raises for server-side errors: they are recorded as events. Client
    failures (timeout, transport) set trace.error and return the partial trace.
    """
    trace = SessionTrace(url=url)
    trace.audio_duration_s = len(pcm) / (SAMPLE_RATE * BYTES_PER_SAMPLE)
    if trailing_silence_ms > 0:
        pcm = pcm + b"\x00" * (
            SAMPLE_RATE * trailing_silence_ms // 1000 * BYTES_PER_SAMPLE
        )
    packets = split_packets(pcm, packet_ms=packet_ms)

    session_update: dict[str, Any] = {"turn_detection": turn_detection}
    if language is not None:
        session_update["language"] = language

    async def _drive() -> None:
        async with websockets.connect(url, max_size=None) as websocket:
            trace.connect_s = time.perf_counter()
            created = await _wait_for(trace, websocket, "session.created")
            trace.session = dict(created.get("session", {}))
            await _send_event(
                websocket, {"type": "session.update", "session": session_update}
            )
            updated = await _wait_for(trace, websocket, "session.updated")
            trace.session = dict(updated.get("session", trace.session))

            receiver = asyncio.create_task(_receiver(trace, websocket))
            try:
                await _sender(
                    trace,
                    websocket,
                    packets,
                    packet_ms=packet_ms,
                    paced=paced,
                    manual_commit=manual_commit,
                )
                await receiver
            finally:
                if not receiver.done():
                    receiver.cancel()
                    await asyncio.gather(receiver, return_exceptions=True)

    try:
        # asyncio.wait_for keeps the module importable on Python 3.10.
        await asyncio.wait_for(_drive(), timeout_s)
    except asyncio.TimeoutError:
        trace.error = f"timeout after {timeout_s}s"
    except (OSError, RuntimeError, websockets.WebSocketException) as exc:
        trace.error = f"{type(exc).__name__}: {exc}"
    finally:
        trace.end_s = time.perf_counter()
    return trace


__all__ = [
    "DEFAULT_TURN_DETECTION",
    "SAMPLE_RATE",
    "ReceivedEvent",
    "SentPacket",
    "SessionTrace",
    "realtime_url",
    "run_session",
    "split_packets",
]
