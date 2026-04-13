# SPDX-License-Identifier: Apache-2.0
"""WebSocket transport for realtime sessions."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from sglang_omni.client import Client
from sglang_omni.realtime.backend import OmniResponseBackend, ResponseBackend
from sglang_omni.realtime.media import mono_float32, resample_linear
from sglang_omni.realtime.session import RealtimeSession, RealtimeSessionConfig
from sglang_omni.realtime.vad import VadConfig

logger = logging.getLogger(__name__)

BackendFactory = Callable[[str, int], ResponseBackend]


class RealtimeVadRequest(BaseModel):
    aggressiveness: int = 3
    frame_duration_ms: int = 20
    min_speech_s: float = 0.25
    min_silence_s: float = 0.60
    preroll_s: float = 0.18
    # Legacy energy-VAD fields retained for compatibility with older clients.
    start_threshold: float = 0.020
    stop_threshold: float = 0.012


class WebSocketEventChannel:
    """Queue-backed event channel with the subset RealtimeSession expects."""

    def __init__(self, send_text: Callable[[str], None]) -> None:
        self._send_text = send_text
        self.readyState = "open"

    def send(self, message: str) -> None:
        if self.readyState != "open":
            raise RuntimeError("WebSocket event channel is closed")
        self._send_text(message)

    def close(self) -> None:
        self.readyState = "closed"


class WebSocketAudioOutputSink:
    """PCM sink that streams assistant audio back over the websocket."""

    def __init__(
        self,
        *,
        sample_rate: int,
        send_bytes: Callable[[bytes], None],
        send_text: Callable[[str], None],
        session_id: str,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self._send_bytes = send_bytes
        self._send_text = send_text
        self._session_id = session_id
        self._pending_samples = 0
        self._pending_updated_at: float | None = None

    @property
    def pending_samples(self) -> int:
        return self._drain_pending()

    def _drain_pending(self) -> int:
        now = time.monotonic()
        if self._pending_updated_at is None:
            return 0
        elapsed = max(now - self._pending_updated_at, 0.0)
        drained = int(round(elapsed * self.sample_rate))
        if drained <= 0:
            return int(self._pending_samples)
        self._pending_samples = max(int(self._pending_samples) - drained, 0)
        self._pending_updated_at = now if self._pending_samples > 0 else None
        return int(self._pending_samples)

    async def clear(self) -> None:
        self._pending_samples = 0
        self._pending_updated_at = None
        self._send_text(
            json.dumps(
                {
                    "type": "output_audio_buffer.cleared",
                    "session_id": self._session_id,
                }
            )
        )

    async def enqueue(self, audio: np.ndarray, sample_rate: int) -> None:
        pcm = mono_float32(audio)
        pcm = resample_linear(pcm, sample_rate, self.sample_rate)
        if pcm.size == 0:
            return

        pcm_i16 = np.clip(pcm * 32767.0, -32768.0, 32767.0).astype("<i2")
        pending = self._drain_pending()
        self._pending_samples = pending + int(pcm_i16.size)
        self._pending_updated_at = time.monotonic()
        self._send_bytes(pcm_i16.tobytes())


@dataclass
class WebSocketSessionHandle:
    session: RealtimeSession
    event_channel: WebSocketEventChannel
    output_sink: WebSocketAudioOutputSink
    input_sample_rate: int = 16000


def create_realtime_ws_router(
    client: Client | None = None,
    *,
    model_name: str,
    backend_factory: BackendFactory | None = None,
) -> APIRouter:
    if backend_factory is None:
        if client is None:
            raise ValueError(
                "create_realtime_ws_router requires either client or backend_factory"
            )

        def backend_factory(
            resolved_model: str,
            max_new_tokens: int,
        ) -> ResponseBackend:
            return OmniResponseBackend(
                client=client,
                model=resolved_model,
                max_new_tokens=max_new_tokens,
                output_modalities=("text", "audio"),
            )

    router = APIRouter()

    @router.websocket("/v1/realtime/ws")
    async def realtime_ws(websocket: WebSocket) -> None:
        await websocket.accept()

        send_queue: asyncio.Queue[tuple[str, str | bytes]] = asyncio.Queue()
        session_id = uuid.uuid4().hex

        def send_text_nowait(payload: str) -> None:
            send_queue.put_nowait(("text", payload))

        def send_bytes_nowait(payload: bytes) -> None:
            send_queue.put_nowait(("bytes", payload))

        async def sender() -> None:
            while True:
                kind, payload = await send_queue.get()
                if kind == "text":
                    await websocket.send_text(str(payload))
                else:
                    await websocket.send_bytes(bytes(payload))

        sender_task = asyncio.create_task(sender())

        model = websocket.query_params.get("model") or model_name
        instructions = websocket.query_params.get("instructions")
        input_audio_mode = websocket.query_params.get("input_audio_mode") or "vad"
        max_new_tokens_raw = websocket.query_params.get("max_new_tokens") or "256"
        try:
            max_new_tokens = max(int(max_new_tokens_raw), 1)
        except ValueError:
            max_new_tokens = 256

        vad_config = VadConfig()
        vad_raw = websocket.query_params.get("vad")
        if vad_raw:
            try:
                vad = RealtimeVadRequest.model_validate_json(vad_raw)
            except Exception:
                logger.warning("Ignoring invalid websocket VAD config: %s", vad_raw)
            else:
                vad_config = VadConfig(
                    sample_rate=vad_config.sample_rate,
                    aggressiveness=vad.aggressiveness,
                    frame_duration_ms=vad.frame_duration_ms,
                    min_speech_s=vad.min_speech_s,
                    min_silence_s=vad.min_silence_s,
                    preroll_s=vad.preroll_s,
                    start_threshold=vad.start_threshold,
                    stop_threshold=vad.stop_threshold,
                )

        output_sink = WebSocketAudioOutputSink(
            sample_rate=24000,
            send_bytes=send_bytes_nowait,
            send_text=send_text_nowait,
            session_id=session_id,
        )
        event_channel = WebSocketEventChannel(send_text_nowait)
        backend = backend_factory(model, max_new_tokens)
        session = RealtimeSession(
            session_id=session_id,
            backend=backend,
            output_track=output_sink,
            config=RealtimeSessionConfig(
                instructions=(
                    instructions
                    or "You are a concise, natural voice assistant. Answer conversationally."
                ),
                input_audio_mode=input_audio_mode,
                vad=vad_config,
            ),
        )
        handle = WebSocketSessionHandle(
            session=session,
            event_channel=event_channel,
            output_sink=output_sink,
        )
        session.attach_event_channel(event_channel)

        await session.emit_event(
            "session.created",
            model=session.backend.model_name,
            instructions=session.instructions,
            audio={
                "input_mode": session.turn_mode,
                "input_encoding": "pcm16le",
                "output_encoding": "pcm16le",
                "output_sample_rate": output_sink.sample_rate,
            },
            transport={"type": "websocket"},
        )

        try:
            while True:
                message = await websocket.receive()
                message_type = message.get("type")
                if message_type == "websocket.disconnect":
                    break

                payload_text = message.get("text")
                if payload_text is not None:
                    try:
                        payload = json.loads(payload_text)
                    except json.JSONDecodeError:
                        await session.emit_event(
                            "error",
                            error={
                                "message": "Invalid JSON control message",
                                "session_id": session.session_id,
                            },
                        )
                        continue

                    event_type = str(payload.get("type") or "").strip()
                    if event_type == "input_audio_format":
                        sample_rate = payload.get("sample_rate")
                        if isinstance(sample_rate, int) and sample_rate > 0:
                            handle.input_sample_rate = int(sample_rate)
                            await session.emit_event(
                                "input_audio_format.updated",
                                sample_rate=handle.input_sample_rate,
                                encoding="pcm16le",
                            )
                        continue

                    await session.handle_client_event(payload)
                    continue

                payload_bytes = message.get("bytes")
                if payload_bytes is None:
                    continue
                if not payload_bytes:
                    continue

                if len(payload_bytes) % 2 != 0:
                    payload_bytes = payload_bytes[:-1]
                    if not payload_bytes:
                        continue

                audio = np.frombuffer(payload_bytes, dtype="<i2")
                await session.handle_audio_chunk(
                    audio,
                    sample_rate=handle.input_sample_rate,
                )
        except WebSocketDisconnect:
            pass
        finally:
            event_channel.close()
            await session.close()
            sender_task.cancel()
            try:
                await sender_task
            except asyncio.CancelledError:
                pass

    return router
