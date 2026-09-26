# SPDX-License-Identifier: Apache-2.0
"""Local WebSocket mock of the conversation realtime protocol.

Single-session, CPU-only, dependency-light (the websockets package the
benchmark suite already uses). Behavior is scriptable so tests can pin
golden flows and inject negative cases: unknown-event errors, malformed
frames, and mid-session aborts. It speaks the same event subset as the
serve layer's conversation protocol.
"""

from __future__ import annotations

import asyncio
import base64
import logging
from dataclasses import dataclass, field
from typing import Any

import websockets

try:  # websockets >= 13 exposes the new asyncio server API
    from websockets.asyncio.server import serve as ws_serve
except ImportError:  # websockets 12 (repo minimum)
    from websockets.serve import serve as ws_serve  # type: ignore[attr-defined]

from benchmarks.realtime_duplex import schema

logger = logging.getLogger(__name__)

TONE_BYTE = 0x55
"""Every mock audio payload is this byte repeated; sizes are what matter."""


@dataclass(slots=True)
class MockBehavior:
    """Scriptable knobs for one mock server instance."""

    model: str = "mock-duplex"
    commit_after_frames: int = 3
    """Server-VAD emulation: commit after this many 80 ms frames arrive."""
    delta_count: int = 3
    delta_payload_bytes: int = 0
    """Per response.audio.delta payload; 0 selects one frame of silence."""
    delta_delay_s: float = 0.01
    error_on_event: dict[str, dict[str, Any]] = field(default_factory=dict)
    """Event types to answer with a scripted error instead of handling."""
    send_malformed_on_session: bool = False
    """After session.created, send one non-JSON-object line."""
    abort_close_code: int | None = None
    """When set, close the socket with this code instead of finishing."""


class DuplexMockServer:
    """Run with `async with DuplexMockServer(behavior) as server:` and point
    the driver at server.url."""

    def __init__(self, behavior: MockBehavior | None = None) -> None:
        self.behavior = behavior or MockBehavior()
        self.received: list[dict[str, Any]] = []
        self.sent: list[dict[str, Any]] = []
        self._server: Any = None
        self._port: int | None = None
        self._session_ready = asyncio.Event()

    @property
    def url(self) -> str:
        if self._port is None:
            raise RuntimeError("mock server is not running")
        return schema.conversation_url("127.0.0.1", self._port)

    async def __aenter__(self) -> DuplexMockServer:
        self._server = await ws_serve(self._handler, "127.0.0.1", 0, max_size=None)
        self._port = self._server.sockets[0].getsockname()[1]
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        self._server.close()
        await self._server.wait_closed()

    async def wait_session(self, timeout_s: float = 5.0) -> None:
        await asyncio.wait_for(self._session_ready.wait(), timeout_s)

    # -- connection handler -------------------------------------------------

    async def _send(self, websocket: Any, event: dict[str, Any]) -> None:
        self.sent.append(event)
        await websocket.send(_dumps(event))

    async def _handler(self, websocket: Any) -> None:
        behavior = self.behavior
        session_id = "sess_mock"
        session_object = {
            "id": session_id,
            "object": "realtime.session",
            "model": behavior.model,
            "modalities": ["text", "audio"],
            "input_audio_format": "pcm16",
            "output_audio_format": "pcm16",
        }
        await self._send(
            websocket, {"type": "session.created", "session": session_object}
        )
        self._session_ready.set()
        if behavior.send_malformed_on_session:
            await websocket.send("this-line-is-not-a-json-object")

        audio_frames = 0
        committed = False
        response_job: asyncio.Task[None] | None = None

        async def run_response() -> None:
            nonlocal committed
            committed = True
            await self._send(
                websocket, {"type": "response.created", "response": {"id": "resp_mock"}}
            )
            payload = base64.b64encode(
                TONE_BYTE.to_bytes()
                * (behavior.delta_payload_bytes or schema.frame_bytes())
            ).decode("ascii")
            try:
                for _ in range(behavior.delta_count):
                    await asyncio.sleep(behavior.delta_delay_s)
                    await self._send(
                        websocket,
                        {
                            "type": "response.audio.delta",
                            "delta": payload,
                            "response_id": "resp_mock",
                        },
                    )
            except (websockets.exceptions.ConnectionClosed, asyncio.CancelledError):
                return
            total = behavior.delta_count * (
                behavior.delta_payload_bytes or schema.frame_bytes()
            )
            await self._send(
                websocket,
                {
                    "type": "response.audio.done",
                    "output_audio_bytes": total,
                    "response_id": "resp_mock",
                },
            )
            await self._send(
                websocket,
                {
                    "type": "response.done",
                    "response": {"id": "resp_mock", "status": "completed"},
                },
            )

        try:
            async for raw in websocket:
                if not isinstance(raw, str):
                    continue
                try:
                    event: Any = _loads(raw)
                except ValueError:
                    await self._send(
                        websocket,
                        {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": "malformed JSON",
                            },
                        },
                    )
                    continue
                if not isinstance(event, dict):
                    await self._send(
                        websocket,
                        {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": "event is not an object",
                            },
                        },
                    )
                    continue
                self.received.append(event)
                event_type = str(event.get("type", ""))
                if schema.parse_client_event(event) is None:
                    # Same rejection the real serve loop applies: outside
                    # the conversation protocol is an invalid request.
                    await self._send(
                        websocket,
                        {
                            "type": "error",
                            "error": {
                                "type": "invalid_request_error",
                                "message": f"unknown event type {event_type!r}",
                            },
                        },
                    )
                    continue

                if event_type in behavior.error_on_event:
                    await self._send(websocket, behavior.error_on_event[event_type])
                    continue

                if event_type == "session.update":
                    await self._send(
                        websocket,
                        {"type": "session.updated", "session": session_object},
                    )
                elif event_type == "input_audio_buffer.append":
                    audio_frames += 1
                    if not committed and audio_frames >= behavior.commit_after_frames:
                        await self._send(
                            websocket, {"type": "input_audio_buffer.speech_started"}
                        )
                        await self._send(
                            websocket, {"type": "input_audio_buffer.speech_stopped"}
                        )
                        await self._send(
                            websocket, {"type": "input_audio_buffer.committed"}
                        )
                        await self._send(
                            websocket,
                            {
                                "type": "conversation.item.created",
                                "item": {"role": "user"},
                            },
                        )
                        response_job = asyncio.create_task(run_response())
                elif event_type == "input_audio_buffer.clear":
                    await self._send(websocket, {"type": "input_audio_buffer.cleared"})
                elif event_type == "response.cancel":
                    if response_job is not None and not response_job.done():
                        response_job.cancel()
                        try:
                            await response_job
                        except asyncio.CancelledError:
                            pass
                    committed = False
                    await self._send(
                        websocket,
                        {
                            "type": "response.done",
                            "response": {"id": "resp_mock", "status": "cancelled"},
                        },
                    )
                elif event_type == "conversation.item.truncate":
                    await self._send(
                        websocket,
                        {
                            "type": "conversation.item.truncated",
                            "item_id": event.get("item_id"),
                        },
                    )
                if (
                    behavior.abort_close_code is not None
                    and audio_frames >= behavior.commit_after_frames
                ):
                    await websocket.close(code=behavior.abort_close_code)
                    return
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if response_job is not None and not response_job.done():
                response_job.cancel()


def _dumps(event: dict[str, Any]) -> str:
    import json

    return json.dumps(event, ensure_ascii=False)


def _loads(raw: str) -> Any:
    import json

    return json.loads(raw)
