# SPDX-License-Identifier: Apache-2.0
"""Model-native full-duplex Realtime WebSocket adapter."""

from __future__ import annotations

import asyncio
import base64
import json
import math
import uuid
from contextlib import suppress
from typing import Any

import numpy as np
from fastapi import WebSocket
from starlette.websockets import WebSocketState

from sglang_omni.client import Client, DuplexSession, GenerateRequest
from sglang_omni.serve.realtime.audio_buffer import RealtimeAudioBuffer
from sglang_omni.serve.realtime.events import (
    InputAudioBufferAppend,
    InputAudioBufferClear,
    PlaybackAck,
    ResponseCancel,
    SessionClose,
    SessionObject,
    SessionUpdate,
    TurnDetectionType,
    make_event,
    parse_client_event,
)

_INPUT_SAMPLE_RATE = 16000
_OUTPUT_SAMPLE_RATE = 24000
_UNIT_SAMPLES = 16000
_UNIT_BYTES = _UNIT_SAMPLES * 2
_MAX_VIDEO_FRAMES_PER_UNIT = 8
_MAX_VIDEO_BYTES_PER_UNIT = 3 * 1024 * 1024
_MAX_REFERENCE_AUDIO_BYTES = 30 * _INPUT_SAMPLE_RATE * 4
_VOICE_BASE64_FIELDS = frozenset({"ref_audio_base64", "tts_ref_audio_base64"})
_QUEUE_END = object()
DEFAULT_INSTRUCTIONS = (
    "You are a helpful realtime voice assistant. Respond conversationally."
)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


class DuplexRealtimeSession:
    """Bridge one WebSocket to one long-lived model-native duplex session."""

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        session_id: str | None = None,
        max_pending_units: int = 2,
        session_ttl_s: float = 600.0,
        idle_timeout_s: float = 60.0,
    ) -> None:
        if max_pending_units < 1:
            raise ValueError("max_pending_units must be positive")
        if session_ttl_s <= 0 or idle_timeout_s <= 0:
            raise ValueError("duplex session timeouts must be positive")
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.session_id = session_id or new_id("sess")
        self.session_object = SessionObject(
            id=self.session_id,
            model=model_name,
            modalities=["text", "audio"],
            instructions=DEFAULT_INSTRUCTIONS,
            input_audio_format="pcm16",
            output_audio_format="pcm16",
            turn_detection={"type": "model_native"},
        )
        self.audio_buffer = RealtimeAudioBuffer(
            source_sr=_INPUT_SAMPLE_RATE,
            target_sr=_INPUT_SAMPLE_RATE,
            max_bytes=_UNIT_BYTES * (max_pending_units + 1),
        )
        self._unit_queue: asyncio.Queue[dict[str, Any] | object] = asyncio.Queue(
            maxsize=max_pending_units
        )
        self._session_ttl_s = float(session_ttl_s)
        self._idle_timeout_s = float(idle_timeout_s)
        self._opened_at = 0.0
        self._last_input_at = 0.0
        self._duplex: DuplexSession | None = None
        self._start_lock = asyncio.Lock()
        self._response_lock = asyncio.Lock()
        self._closed_signal = asyncio.Event()
        self._input_task: asyncio.Task[None] | None = None
        self._output_task: asyncio.Task[None] | None = None
        self._watchdog_task: asyncio.Task[None] | None = None
        self._pending_frames: list[str] = []
        self._pending_frame_bytes = 0
        self._pending_force_listen = False
        self._pending_max_slice_nums = 1
        self._pending_timestamp_ms: int | None = None
        self._playback_cursor_ms = 0
        self._emitted_audio_end_ms = 0.0
        self._active_response_id: str | None = None
        self._active_item_id: str | None = None
        self._active_text: list[str] = []
        self._active_response_epoch = 0
        self._teardown_complete = False
        self._teardown_task: asyncio.Task[None] | None = None
        self.closed = False

    async def run(self) -> None:
        loop = asyncio.get_running_loop()
        self._opened_at = loop.time()
        self._last_input_at = self._opened_at
        await self.send(
            make_event(
                "session.created",
                session=self._session_payload(),
            )
        )
        self._input_task = asyncio.create_task(
            self._input_loop(), name=f"duplex-input-{self.session_id}"
        )
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(), name=f"duplex-watchdog-{self.session_id}"
        )
        while not self.closed:
            receive_task = asyncio.create_task(self.websocket.receive())
            closed_task = asyncio.create_task(self._closed_signal.wait())
            done, pending = await asyncio.wait(
                {receive_task, closed_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            if closed_task in done:
                if receive_task in done:
                    await asyncio.gather(receive_task, return_exceptions=True)
                break
            message = receive_task.result()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue
            raw = message.get("text")
            if raw is None:
                await self.send_error(
                    "invalid_request_error",
                    "binary_event_not_supported",
                    "Realtime events must be JSON text frames",
                )
                continue
            try:
                payload = json.loads(raw)
                if not isinstance(payload, dict):
                    raise TypeError("Top-level payload must be a JSON object")
                await self.dispatch(payload)
            except Exception as exc:
                await self.send_error(
                    "invalid_request_error",
                    "invalid_realtime_event",
                    str(exc) or type(exc).__name__,
                )

    async def dispatch(self, payload: dict[str, Any]) -> None:
        event = parse_client_event(payload)
        if event is None:
            raise ValueError(f"Unsupported event type: {payload.get('type')!r}")
        if isinstance(event, SessionUpdate):
            await self.handle_session_update(event)
        elif isinstance(event, InputAudioBufferAppend):
            await self.handle_audio_append(event)
        elif isinstance(event, InputAudioBufferClear):
            await self.handle_audio_clear(event)
        elif isinstance(event, ResponseCancel):
            await self.handle_response_cancel(event)
        elif isinstance(event, PlaybackAck):
            await self.handle_playback_ack(event)
        elif isinstance(event, SessionClose):
            await self.handle_session_close(event)
        else:
            raise ValueError(f"Unsupported event type: {event.type!r}")

    async def handle_session_update(self, event: SessionUpdate) -> None:
        update = event.session.model_dump(exclude_none=True, exclude_unset=True)
        candidate = SessionObject.model_validate(
            self.session_object.model_dump() | update
        )
        if candidate.input_audio_format != "pcm16":
            raise ValueError("Native duplex currently supports only pcm16 input")
        if candidate.output_audio_format != "pcm16":
            raise ValueError("Native duplex currently supports only pcm16 output")
        if set(candidate.modalities) != {"text", "audio"}:
            raise ValueError("Native duplex requires text and audio modalities")
        if (
            candidate.turn_detection is None
            or candidate.turn_detection.type != TurnDetectionType.MODEL_NATIVE
        ):
            raise ValueError("Native duplex requires model_native turn detection")
        supported_updates = {
            "modalities",
            "instructions",
            "input_audio_format",
            "output_audio_format",
            "voice",
            "turn_detection",
        }
        unsupported_updates = set(update) - supported_updates
        if unsupported_updates:
            raise ValueError(
                "Native duplex does not support per-session updates for: "
                f"{sorted(unsupported_updates)}"
            )
        turn_detection_update = update.get("turn_detection")
        if isinstance(turn_detection_update, dict) and set(turn_detection_update) - {
            "type"
        }:
            raise ValueError(
                "Native duplex turn_detection supports only type=model_native"
            )
        _validate_realtime_voice(candidate.voice)
        if self._duplex is not None and (
            candidate.instructions != self.session_object.instructions
            or candidate.voice != self.session_object.voice
        ):
            raise RuntimeError(
                "instructions and voice cannot change after the duplex model "
                "session has started"
            )
        self.session_object = candidate
        await self.send(
            make_event(
                "session.updated",
                session=self._session_payload(),
            )
        )

    def _session_payload(self) -> dict[str, Any]:
        return self.session_object.model_dump(
            exclude_none=True,
            exclude={"temperature", "max_response_output_tokens"},
        )

    async def handle_audio_append(self, event: InputAudioBufferAppend) -> None:
        self._last_input_at = asyncio.get_running_loop().time()
        max_audio_b64_chars = ((self.audio_buffer.max_bytes + 2) // 3) * 4
        if len(event.audio) > max_audio_b64_chars:
            raise ValueError("audio append exceeds the bounded duplex input buffer")
        audio = base64.b64decode(event.audio, validate=True)
        if len(audio) % 2:
            raise ValueError("pcm16 input must contain a whole number of samples")
        frames, frame_bytes = _validate_video_frames(event.video_frames or [])
        if len(self._pending_frames) + len(frames) > _MAX_VIDEO_FRAMES_PER_UNIT:
            raise ValueError(
                f"a duplex unit accepts at most {_MAX_VIDEO_FRAMES_PER_UNIT} frames"
            )
        if self._pending_frame_bytes + frame_bytes > _MAX_VIDEO_BYTES_PER_UNIT:
            raise ValueError("video frames exceed the duplex unit byte limit")
        complete_units = (self.audio_buffer.num_bytes + len(audio)) // _UNIT_BYTES
        available_units = self._unit_queue.maxsize - self._unit_queue.qsize()
        if complete_units > available_units:
            raise ValueError("duplex input queue is full; retry after backpressure")

        self.audio_buffer.append_bytes(audio)
        self._pending_frames.extend(frames)
        self._pending_frame_bytes += frame_bytes
        self._pending_force_listen = self._pending_force_listen or event.force_listen
        self._pending_max_slice_nums = event.max_slice_nums
        if event.timestamp_ms is not None:
            self._pending_timestamp_ms = event.timestamp_ms

        while self.audio_buffer.num_bytes >= _UNIT_BYTES:
            unit = {
                "audio_pcm16": self.audio_buffer.pop_left(_UNIT_BYTES),
                "sample_rate": _INPUT_SAMPLE_RATE,
                "video_frames": self._pending_frames,
                "force_listen": self._pending_force_listen,
                "max_slice_nums": self._pending_max_slice_nums,
                "timestamp_ms": self._pending_timestamp_ms,
            }
            self._pending_frames = []
            self._pending_frame_bytes = 0
            self._pending_force_listen = False
            self._pending_timestamp_ms = None
            self._unit_queue.put_nowait(unit)

    async def handle_audio_clear(self, event: InputAudioBufferClear) -> None:
        del event
        self.audio_buffer.clear()
        while True:
            try:
                queued = self._unit_queue.get_nowait()
            except asyncio.QueueEmpty:
                break
            if queued is _QUEUE_END:
                self._unit_queue.put_nowait(_QUEUE_END)
                break
        self._pending_frames.clear()
        self._pending_frame_bytes = 0
        self._pending_force_listen = False
        self._pending_max_slice_nums = 1
        self._pending_timestamp_ms = None
        await self.send(make_event("input_audio_buffer.cleared"))

    async def handle_response_cancel(self, event: ResponseCancel) -> None:
        del event
        self._pending_force_listen = True
        if self._duplex is None:
            return
        duplex = self._duplex
        async with self._response_lock:
            epoch = await duplex.interrupt()
            await self._finish_response_locked("client_cancelled", cancelled=True)
            self._active_response_epoch = epoch

    async def handle_playback_ack(self, event: PlaybackAck) -> None:
        if not math.isfinite(event.audio_end_ms):
            raise ValueError("playback cursor must be finite")
        if event.audio_end_ms < self._playback_cursor_ms:
            raise ValueError("playback cursor must be monotonic")
        if event.audio_end_ms > self._emitted_audio_end_ms + 1e-6:
            raise ValueError("playback cursor exceeds emitted audio")
        if self._duplex is not None:
            await self._duplex.playback_ack(event.audio_end_ms)
        self._playback_cursor_ms = event.audio_end_ms
        await self.send(
            make_event(
                "response.audio.playback_acknowledged",
                audio_end_ms=event.audio_end_ms,
            )
        )

    async def handle_session_close(self, event: SessionClose) -> None:
        if await self._close_before_model_start(reason=event.reason):
            return
        await self._close_and_wait_for_terminal(reason=event.reason)

    async def _ensure_started(self) -> DuplexSession:
        if self.closed:
            raise RuntimeError(f"Realtime session {self.session_id} is closed")
        if self._duplex is not None:
            return self._duplex
        async with self._start_lock:
            if self.closed:
                raise RuntimeError(f"Realtime session {self.session_id} is closed")
            if self._duplex is not None:
                return self._duplex
            voice = self.session_object.voice
            open_config = {
                "session_id": self.session_id,
                "system_prompt": self.session_object.instructions,
                "voice": voice,
                "input_sample_rate": _INPUT_SAMPLE_RATE,
                "output_sample_rate": _OUTPUT_SAMPLE_RATE,
                "turn_source": "model_native",
            }
            request = GenerateRequest(
                model=self.model_name,
                prompt=self.session_object.instructions or DEFAULT_INSTRUCTIONS,
                stream=True,
                output_modalities=["text", "audio"],
                metadata={"_duplex_session": open_config},
            )
            self._duplex = await self.client.open_duplex_session(
                request,
                session_id=self.session_id,
                output_queue_size=32,
            )
            self._output_task = asyncio.create_task(
                self._output_loop(), name=f"duplex-output-{self.session_id}"
            )
        return self._duplex

    async def _input_loop(self) -> None:
        try:
            while not self.closed:
                unit = await self._unit_queue.get()
                if unit is _QUEUE_END:
                    return
                if self.closed:
                    return
                assert isinstance(unit, dict)
                duplex = await self._ensure_started()
                await duplex.append(unit, wait_processed=True)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            # Another task may have won terminal ownership while this input was
            # waiting to start or append.  Never publish an error after close.
            if self.closed:
                return
            with suppress(Exception):
                await self.send_error(
                    "server_error",
                    "duplex_input_failed",
                    str(exc) or type(exc).__name__,
                )
            self._mark_closed()
            await self._close_websocket()

    async def _output_loop(self) -> None:
        assert self._duplex is not None
        try:
            async for event in self._duplex.events():
                event_type = event["type"]
                if event_type == "session.created":
                    await self.send(
                        make_event(
                            "session.ready",
                            session_id=self.session_id,
                            generation=event.get("generation"),
                            output_seq=event.get("output_seq"),
                        )
                    )
                elif event_type == "session.input_processed":
                    await self.send(
                        make_event(
                            "input_audio_buffer.processed",
                            input_seq=event.get("input_seq"),
                            response_epoch=event.get("response_epoch"),
                            output_seq=event.get("output_seq"),
                            metrics=event.get("metrics"),
                        )
                    )
                elif event_type == "response.output.delta":
                    await self._handle_output_delta(event)
                elif event_type == "response.output.done":
                    await self._handle_output_done(event)
                elif event_type == "session.error":
                    with suppress(Exception):
                        await self.send_error(
                            "server_error",
                            "duplex_session_failed",
                            str(event.get("error") or "duplex session failed"),
                        )
                    self._mark_closed()
                    await self._close_websocket()
                    return
                elif event_type == "session.closed":
                    await self._finish_response("session_closed")
                    await self.send(
                        make_event(
                            "session.closed",
                            session_id=self.session_id,
                            generation=event.get("generation"),
                            response_epoch=event.get("response_epoch"),
                            output_seq=event.get("output_seq"),
                            reason=event.get("reason"),
                            cleanup_error=event.get("cleanup_error"),
                        )
                    )
                    self._mark_closed()
                    await self._close_websocket()
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with suppress(Exception):
                await self.send_error(
                    "server_error",
                    "duplex_output_failed",
                    str(exc) or type(exc).__name__,
                )
            self._mark_closed()
            await self._close_websocket()

    async def _watchdog_loop(self) -> None:
        try:
            while not self.closed:
                await asyncio.sleep(1.0)
                now = asyncio.get_running_loop().time()
                if now - self._opened_at >= self._session_ttl_s:
                    reason = "max_session_duration"
                elif (
                    self._duplex is not None
                    and now - self._last_input_at >= self._idle_timeout_s
                ):
                    reason = "input_idle_timeout"
                else:
                    continue
                if not await self._close_before_model_start(reason=reason):
                    await self._close_and_wait_for_terminal(reason=reason)
                return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            with suppress(Exception):
                await self.send_error(
                    "server_error",
                    "duplex_watchdog_failed",
                    str(exc) or type(exc).__name__,
                )
            self._mark_closed()
            await self._close_websocket()

    async def _handle_output_delta(self, event: dict[str, Any]) -> None:
        async with self._response_lock:
            await self._handle_output_delta_locked(event)

    async def _handle_output_done(self, event: dict[str, Any]) -> None:
        async with self._response_lock:
            epoch = int(event.get("response_epoch", 0))
            if epoch < self._active_response_epoch:
                return
            if epoch > self._active_response_epoch:
                await self._finish_response_locked("interrupted", cancelled=True)
                self._active_response_epoch = epoch
            reason = str(event.get("reason") or "model_turn_end")
            await self._ensure_response_locked(epoch)
            await self._finish_response_locked(reason)

    async def _handle_output_delta_locked(self, event: dict[str, Any]) -> None:
        epoch = int(event.get("response_epoch", 0))
        if epoch < self._active_response_epoch:
            return
        if epoch > self._active_response_epoch:
            await self._finish_response_locked("interrupted", cancelled=True)
            self._active_response_epoch = epoch

        kind = event.get("kind")
        if kind == "listen":
            await self.send(
                make_event(
                    "response.output.delta",
                    kind="listen",
                    response_id=self._active_response_id,
                    input_seq=event.get("input_seq"),
                    response_epoch=epoch,
                    output_seq=event.get("output_seq"),
                    metrics=event.get("metrics"),
                )
            )
            await self._finish_response_locked("model_listen")
            return

        await self._ensure_response_locked(epoch)
        assert self._active_response_id is not None
        assert self._active_item_id is not None
        if kind == "text":
            text = str(event.get("text") or event.get("delta") or "")
            if not text:
                return
            self._active_text.append(text)
            await self.send(
                make_event(
                    "response.text.delta",
                    response_id=self._active_response_id,
                    item_id=self._active_item_id,
                    output_index=0,
                    content_index=0,
                    delta=text,
                    response_epoch=epoch,
                    output_seq=event.get("output_seq"),
                )
            )
            return
        if kind == "audio":
            audio = event.get("audio") or event.get("delta")
            if not isinstance(audio, str) or not audio:
                return
            output_format = str(
                event.get("audio_format") or event.get("format") or "float32"
            )
            if output_format in {"float32", "f32le"}:
                audio = _float32_audio_to_pcm16_b64(audio)
            elif output_format != "pcm16":
                raise ValueError(f"Unsupported duplex audio format: {output_format}")
            sample_rate = int(event.get("sample_rate") or _OUTPUT_SAMPLE_RATE)
            if sample_rate <= 0:
                raise ValueError("duplex audio sample_rate must be positive")
            audio_bytes = base64.b64decode(audio, validate=True)
            if len(audio_bytes) % 2:
                raise ValueError("pcm16 output has an invalid byte length")
            audio_start_ms = self._emitted_audio_end_ms
            self._emitted_audio_end_ms += len(audio_bytes) / 2 / sample_rate * 1000.0
            await self.send(
                make_event(
                    "response.audio.delta",
                    response_id=self._active_response_id,
                    item_id=self._active_item_id,
                    output_index=0,
                    content_index=1,
                    delta=audio,
                    sample_rate=sample_rate,
                    audio_start_ms=audio_start_ms,
                    audio_end_ms=self._emitted_audio_end_ms,
                    response_epoch=epoch,
                    output_seq=event.get("output_seq"),
                )
            )
            return
        raise ValueError(f"Unsupported duplex output kind: {kind!r}")

    async def _ensure_response(self, epoch: int) -> None:
        async with self._response_lock:
            await self._ensure_response_locked(epoch)

    async def _ensure_response_locked(self, epoch: int) -> None:
        if self._active_response_id is not None:
            return
        self._active_response_id = new_id("resp")
        self._active_item_id = new_id("item")
        self._active_text = []
        self._active_response_epoch = epoch
        await self.send(
            make_event(
                "response.created",
                response={
                    "id": self._active_response_id,
                    "object": "realtime.response",
                    "status": "in_progress",
                    "output": [],
                    "response_epoch": epoch,
                },
            )
        )

    async def _finish_response(self, reason: str, *, cancelled: bool = False) -> None:
        async with self._response_lock:
            await self._finish_response_locked(reason, cancelled=cancelled)

    async def _finish_response_locked(
        self,
        reason: str,
        *,
        cancelled: bool = False,
    ) -> None:
        response_id = self._active_response_id
        item_id = self._active_item_id
        if response_id is None or item_id is None:
            return
        text = "".join(self._active_text)
        status = "cancelled" if cancelled else "completed"
        await self.send(
            make_event(
                "response.text.done",
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=0,
                text=text,
            )
        )
        await self.send(
            make_event(
                "response.audio.done",
                response_id=response_id,
                item_id=item_id,
                output_index=0,
                content_index=1,
            )
        )
        await self.send(
            make_event(
                "response.done",
                response={
                    "id": response_id,
                    "object": "realtime.response",
                    "status": status,
                    "status_details": {"reason": reason},
                    "output": [
                        {
                            "id": item_id,
                            "object": "realtime.item",
                            "type": "message",
                            "role": "assistant",
                            "content": [
                                {"type": "text", "text": text},
                                {"type": "audio"},
                            ],
                        }
                    ],
                    "response_epoch": self._active_response_epoch,
                },
            )
        )
        self._active_response_id = None
        self._active_item_id = None
        self._active_text = []

    def _mark_closed(self) -> None:
        self.closed = True
        self._closed_signal.set()

    async def _close_before_model_start(self, *, reason: str) -> bool:
        """Atomically own close or observe a model session created in flight.

        Opening the backend awaits while holding ``_start_lock``.  Close paths
        must take the same lock before deciding that no backend exists;
        otherwise they can publish ``session.closed`` while an input task is
        still completing ``open_duplex_session``.
        """

        async with self._start_lock:
            if self.closed:
                return True
            if self._duplex is not None:
                return False
            await self._finish_response(reason)
            await self.send(
                make_event(
                    "session.closed",
                    session_id=self.session_id,
                    reason=reason,
                )
            )
            # Wake run()/teardown only after the terminal frame is on the
            # wire.  In particular, a watchdog task must not cancel itself via
            # teardown between claiming close and publishing session.closed.
            self._mark_closed()
            await self._close_websocket()
            return True

    async def _close_and_wait_for_terminal(self, *, reason: str) -> None:
        """Close the model first, then let the output owner forward terminal.

        ``DuplexSession.close`` is complete once its stream consumer has queued the
        terminal event.  The Realtime output task still owns forwarding that event
        to the WebSocket, so closing the socket immediately would race with the
        acknowledgement.  Waiting for that task preserves the wire ordering.
        """

        cleanup_error = await self._close_model_session(reason=reason)
        output_task = self._output_task
        if output_task is not None and output_task is not asyncio.current_task():
            try:
                await asyncio.wait_for(asyncio.shield(output_task), timeout=10.0)
            except TimeoutError:
                output_task.cancel()
                await asyncio.gather(output_task, return_exceptions=True)
                cleanup_error = cleanup_error or "terminal event forwarding timed out"
            except Exception as exc:
                cleanup_error = cleanup_error or (str(exc) or type(exc).__name__)

        if not self.closed:
            with suppress(Exception):
                await self.send(
                    make_event(
                        "session.closed",
                        session_id=self.session_id,
                        reason=reason,
                        cleanup_error=cleanup_error,
                    )
                )
            self._mark_closed()
            await self._close_websocket()

    async def _close_model_session(self, *, reason: str) -> str | None:
        cleanup_errors: list[str] = []
        try:
            await self._finish_response(reason)
        except Exception as exc:
            cleanup_errors.append(str(exc) or type(exc).__name__)
        if self._duplex is not None and not self._duplex.is_closed:
            try:
                await self._duplex.close(reason=reason, timeout_s=10.0)
            except Exception as exc:
                cleanup_errors.append(str(exc) or type(exc).__name__)
        return "; ".join(cleanup_errors) or None

    async def send(self, event: dict[str, Any]) -> None:
        if (
            self.websocket.application_state != WebSocketState.CONNECTED
            or self.websocket.client_state != WebSocketState.CONNECTED
        ):
            return
        event.setdefault("event_id", new_id("evt"))
        try:
            await self.websocket.send_text(json.dumps(event))
        except Exception:
            self._mark_closed()
            raise

    async def send_error(self, type_: str, code: str, message: str) -> None:
        await self.send(
            make_event(
                "error",
                error={"type": type_, "code": code, "message": message},
            )
        )

    async def _close_websocket(self) -> None:
        if (
            self.websocket.application_state == WebSocketState.CONNECTED
            and self.websocket.client_state == WebSocketState.CONNECTED
        ):
            with suppress(Exception):
                await self.websocket.close()

    async def teardown(self) -> None:
        task = self._teardown_task
        if task is None:
            task = asyncio.create_task(
                self._teardown_impl(), name=f"duplex-teardown-{self.session_id}"
            )
            self._teardown_task = task
        try:
            await asyncio.shield(task)
        except asyncio.CancelledError:
            # Teardown owns the only model-session lease.  Do not leak it merely
            # because the request task was cancelled during disconnect handling.
            with suppress(asyncio.CancelledError):
                await asyncio.shield(task)
            raise

    async def _teardown_impl(self) -> None:
        if self._teardown_complete:
            return
        self._mark_closed()
        with suppress(asyncio.QueueFull):
            self._unit_queue.put_nowait(_QUEUE_END)
        current_task = asyncio.current_task()
        try:
            for task in (self._input_task, self._watchdog_task):
                if task is None or task is current_task:
                    continue
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

            await self._close_model_session(reason="client_disconnected")
        finally:
            output_task = self._output_task
            if output_task is not None and output_task is not current_task:
                if not output_task.done():
                    output_task.cancel()
                await asyncio.gather(output_task, return_exceptions=True)
            await self._close_websocket()
            self._teardown_complete = True


def _float32_audio_to_pcm16_b64(audio_b64: str) -> str:
    raw = base64.b64decode(audio_b64, validate=True)
    if len(raw) % np.dtype("<f4").itemsize:
        raise ValueError("float32 audio payload has an invalid byte length")
    waveform = np.frombuffer(raw, dtype="<f4")
    pcm = np.round(np.clip(waveform, -1.0, 1.0) * 32767.0).astype("<i2")
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def _validate_video_frames(frames: list[str]) -> tuple[list[str], int]:
    if len(frames) > _MAX_VIDEO_FRAMES_PER_UNIT:
        raise ValueError(
            f"a duplex unit accepts at most {_MAX_VIDEO_FRAMES_PER_UNIT} frames"
        )
    total_bytes = 0
    for frame in frames:
        encoded = frame.split(",", 1)[1] if frame.startswith("data:") else frame
        if len(encoded) > ((_MAX_VIDEO_BYTES_PER_UNIT + 2) // 3) * 4:
            raise ValueError("video frame exceeds the duplex unit byte limit")
        raw = base64.b64decode(encoded, validate=True)
        total_bytes += len(raw)
        if total_bytes > _MAX_VIDEO_BYTES_PER_UNIT:
            raise ValueError("video frames exceed the duplex unit byte limit")
    return list(frames), total_bytes


def _validate_realtime_voice(voice: Any) -> None:
    if voice is None:
        return
    if not isinstance(voice, dict):
        raise ValueError(
            "MiniCPM-o voice must be bounded base64 audio; named voices and "
            "request-provided file paths are not supported"
        )
    unknown = set(voice) - _VOICE_BASE64_FIELDS
    if unknown:
        raise ValueError(
            "MiniCPM-o voice accepts only base64 reference audio fields; "
            f"unsupported fields: {sorted(unknown)}"
        )
    total_bytes = 0
    for name, value in voice.items():
        if not isinstance(value, str) or not value:
            raise ValueError(f"voice.{name} must be non-empty base64 text")
        encoded = value.split(",", 1)[1] if value.startswith("data:") else value
        if len(encoded) > ((_MAX_REFERENCE_AUDIO_BYTES + 2) // 3) * 4:
            raise ValueError("reference audio exceeds the session byte limit")
        raw = base64.b64decode(encoded, validate=True)
        if len(raw) % 4:
            raise ValueError("reference audio must contain raw float32 samples")
        total_bytes += len(raw)
        if total_bytes > _MAX_REFERENCE_AUDIO_BYTES:
            raise ValueError("reference audio exceeds the session byte limit")
