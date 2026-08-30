# SPDX-License-Identifier: Apache-2.0
"""Fixed-frame realtime session for lockstep duplex audio pipelines."""

from __future__ import annotations

import asyncio
import base64
import json
import uuid
from contextlib import aclosing
from typing import Any

from fastapi import WebSocket
from starlette.websockets import WebSocketState

from sglang_omni.client import Client, GenerateRequest, SamplingParams
from sglang_omni.config import RealtimeAudioConfig


class FrameRealtimeSession:
    """Continuously submit exact PCM frames without a VAD turn boundary.

    The model pipeline owns conversation state. Each coordinator request is one
    frame and carries a stable session id, so stage schedulers can retain their
    model-specific caches without putting those caches in the serving layer.
    """

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        config: RealtimeAudioConfig,
        session_id: str | None = None,
    ) -> None:
        if config.mode != "frame" or config.frame_samples is None:
            raise ValueError("FrameRealtimeSession requires frame-mode audio config")
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.config = config
        self.session_id = session_id or f"sess_{uuid.uuid4().hex}"
        self.instructions = "You are a helpful realtime voice assistant."
        self.closed = False
        self._partial_pcm = bytearray()
        self._frames: asyncio.Queue[bytes | None] = asyncio.Queue(
            maxsize=config.max_pending_frames
        )
        self._worker: asyncio.Task[None] | None = None
        self._inflight: set[asyncio.Task[None]] = set()
        self._inflight_slots = asyncio.Semaphore(config.max_inflight_frames)
        self._frame_error: BaseException | None = None
        self._active_request_ids: set[str] = set()
        self._frame_index = 0

    async def run(self) -> None:
        await self.send(
            {
                "type": "session.created",
                "session": {
                    "id": self.session_id,
                    "model": self.model_name,
                    "modalities": ["text", "audio"],
                    "input_audio_format": "pcm16",
                    "input_sample_rate": self.config.input_sample_rate,
                    "output_audio_format": "pcm16",
                    "output_sample_rate": self.config.output_sample_rate,
                    "frame_samples": self.config.frame_samples,
                },
            }
        )
        self._worker = asyncio.create_task(self._drain_frames())
        while not self.closed:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue
            raw = message.get("text")
            if raw is None:
                raise ValueError("Realtime events must be JSON text messages")
            event = json.loads(raw)
            if not isinstance(event, dict):
                raise TypeError("Top-level payload must be a JSON object")
            await self.dispatch(event)

    async def dispatch(self, event: dict[str, Any]) -> None:
        event_type = event.get("type")
        if event_type == "session.update":
            session = event.get("session") or {}
            if not isinstance(session, dict):
                raise ValueError("session.update.session must be an object")
            instructions = session.get("instructions")
            if instructions is not None:
                self.instructions = str(instructions)
            await self.send(
                {
                    "type": "session.updated",
                    "session": {
                        "id": self.session_id,
                        "instructions": self.instructions,
                        "input_sample_rate": self.config.input_sample_rate,
                        "output_sample_rate": self.config.output_sample_rate,
                        "frame_samples": self.config.frame_samples,
                    },
                }
            )
            return
        if event_type == "input_audio_buffer.append":
            await self._append_audio(event.get("audio"))
            return
        if event_type == "input_audio_buffer.clear":
            self._partial_pcm.clear()
            await self.send({"type": "input_audio_buffer.cleared"})
            return
        if event_type == "input_audio_buffer.commit":
            await self._flush_partial_frame()
            await self._frames.join()
            if self._frame_error is not None:
                error = self._frame_error
                self._frame_error = None
                raise RuntimeError("Realtime frame generation failed") from error
            await self.send({"type": "input_audio_buffer.committed"})
            return
        if event_type == "session.close":
            self.closed = True
            return
        if event_type == "response.cancel":
            # VoiceChat's output is one lockstep frame, not a cancellable turn.
            return
        raise ValueError(f"Unsupported frame-realtime event type: {event_type!r}")

    async def _append_audio(self, encoded: Any) -> None:
        if not isinstance(encoded, str):
            raise TypeError("input_audio_buffer.append.audio must be base64 text")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) % 2:
            raise ValueError("PCM16 input must contain complete 16-bit samples")
        self._partial_pcm.extend(raw)
        frame_bytes = int(self.config.frame_samples) * 2
        while len(self._partial_pcm) >= frame_bytes:
            frame = bytes(self._partial_pcm[:frame_bytes])
            del self._partial_pcm[:frame_bytes]
            await self._frames.put(frame)

    async def _flush_partial_frame(self) -> None:
        if not self._partial_pcm:
            return
        frame_bytes = int(self.config.frame_samples) * 2
        frame = bytes(self._partial_pcm) + bytes(frame_bytes - len(self._partial_pcm))
        self._partial_pcm.clear()
        await self._frames.put(frame)

    async def _drain_frames(self) -> None:
        try:
            while True:
                frame = await self._frames.get()
                if frame is None:
                    self._frames.task_done()
                    break
                await self._inflight_slots.acquire()
                frame_index = self._frame_index
                self._frame_index += 1
                task = asyncio.create_task(
                    self._run_frame_and_report(frame, frame_index)
                )
                self._inflight.add(task)
                task.add_done_callback(self._frame_done)
            if self._inflight:
                await asyncio.gather(*self._inflight)
        finally:
            for task in self._inflight:
                if not task.done():
                    task.cancel()
            await asyncio.gather(*self._inflight, return_exceptions=True)
            self._inflight.clear()

    def _frame_done(self, task: asyncio.Task[None]) -> None:
        self._inflight.discard(task)
        self._inflight_slots.release()
        self._frames.task_done()
        if not task.cancelled() and task.exception() is not None:
            self._frame_error = task.exception()

    async def _run_frame_and_report(self, frame: bytes, frame_index: int) -> None:
        try:
            await self._run_frame(frame, frame_index)
        except asyncio.CancelledError:
            raise
        except Exception as error:  # noqa: BLE001 - report frame failures to the client
            self._frame_error = error
            if not self.closed:
                await self.send(
                    {
                        "type": "error",
                        "error": {
                            "type": "server_error",
                            "code": "frame_generation_failed",
                            "message": str(error),
                        },
                    }
                )

    async def _run_frame(self, frame: bytes, frame_index: int) -> None:
        request_id = f"rt-{self.session_id}-{frame_index}"
        self._active_request_ids.add(request_id)
        request = GenerateRequest(
            model=self.model_name,
            prompt={
                "event": "audio_frame",
                "session_id": self.session_id,
                "frame_index": frame_index,
                "pcm16": base64.b64encode(frame).decode("ascii"),
                "instructions": self.instructions,
            },
            sampling=SamplingParams(temperature=0.0, max_new_tokens=1),
            output_modalities=["text", "audio"],
            stream=True,
        )
        try:
            stream = self.client.completion_stream(
                request, request_id=request_id, audio_format="pcm"
            )
            async with aclosing(stream):
                async for chunk in stream:
                    if chunk.text:
                        await self.send(
                            {
                                "type": "response.text.delta",
                                "delta": chunk.text,
                                "frame_index": frame_index,
                            }
                        )
                    if chunk.audio_b64:
                        await self.send(
                            {
                                "type": "response.audio.delta",
                                "delta": chunk.audio_b64,
                                "sample_rate": self.config.output_sample_rate,
                                "frame_index": frame_index,
                            }
                        )
        finally:
            self._active_request_ids.discard(request_id)

    async def _close_model_session(self) -> None:
        request_id = f"rt-{self.session_id}-close"
        request = GenerateRequest(
            model=self.model_name,
            prompt={"event": "session_close", "session_id": self.session_id},
            sampling=SamplingParams(temperature=0.0, max_new_tokens=1),
            output_modalities=["audio"],
            stream=False,
        )
        async for _ in self.client.generate(request, request_id=request_id):
            pass

    async def send(self, event: dict[str, Any]) -> None:
        if self.closed or self.websocket.application_state != WebSocketState.CONNECTED:
            return
        event.setdefault("event_id", f"evt_{uuid.uuid4().hex}")
        await self.websocket.send_text(json.dumps(event))

    async def teardown(self) -> None:
        if self.closed and self._worker is None:
            return
        self.closed = True
        if self._active_request_ids:
            await asyncio.gather(
                *(
                    self.client.abort(request_id)
                    for request_id in self._active_request_ids
                ),
                return_exceptions=True,
            )
        if self._worker is not None:
            self._worker.cancel()
            await asyncio.gather(self._worker, return_exceptions=True)
            self._worker = None
        try:
            await self._close_model_session()
        finally:
            if self.websocket.client_state == WebSocketState.CONNECTED:
                await self.websocket.close()


__all__ = ["FrameRealtimeSession"]
