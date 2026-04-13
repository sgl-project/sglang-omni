# SPDX-License-Identifier: Apache-2.0
"""WebRTC + VAD prototype routes."""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal

from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

try:
    from aiortc import (
        RTCConfiguration,
        RTCIceServer,
        RTCPeerConnection,
        RTCSessionDescription,
    )
    from aiortc.exceptions import InvalidStateError
    from aiortc.mediastreams import MediaStreamError

    from sglang_omni.realtime.audio_track import BufferedAudioStreamTrack

    AIORTC_AVAILABLE = True
except ImportError:  # pragma: no cover - handled at runtime
    BufferedAudioStreamTrack = None
    RTCConfiguration = None
    RTCIceServer = None
    RTCPeerConnection = None
    RTCSessionDescription = None
    InvalidStateError = RuntimeError
    MediaStreamError = Exception
    AIORTC_AVAILABLE = False

from sglang_omni.client import Client
from sglang_omni.realtime.backend import OmniResponseBackend, ResponseBackend
from sglang_omni.realtime.media import audio_frame_to_ndarray
from sglang_omni.realtime.session import RealtimeSession, RealtimeSessionConfig
from sglang_omni.realtime.vad import VadConfig

logger = logging.getLogger(__name__)

BackendFactory = Callable[[str, int, bool], ResponseBackend]


def _load_rtc_configuration_from_env() -> Any | None:
    ice_urls = [
        value.strip()
        for value in os.environ.get("SGLANG_OMNI_ICE_URLS", "").split(",")
        if value.strip()
    ]
    if not ice_urls or RTCConfiguration is None or RTCIceServer is None:
        return None

    username = os.environ.get("SGLANG_OMNI_ICE_USERNAME") or None
    credential = os.environ.get("SGLANG_OMNI_ICE_CREDENTIAL") or None
    return RTCConfiguration(
        iceServers=[
            RTCIceServer(
                urls=ice_urls,
                username=username,
                credential=credential,
            )
        ]
    )


class RealtimeVadRequest(BaseModel):
    aggressiveness: int = 3
    frame_duration_ms: int = 20
    min_speech_s: float = 0.25
    min_silence_s: float = 0.60
    preroll_s: float = 0.18
    # Legacy energy-VAD fields retained for compatibility with older clients.
    start_threshold: float = 0.020
    stop_threshold: float = 0.012


class RealtimeOfferRequest(BaseModel):
    sdp: str
    type: str = Field(default="offer", pattern="^offer$")
    model: str | None = None
    instructions: str | None = None
    max_new_tokens: int = 256
    output_text: bool = True
    input_audio_mode: Literal["vad", "manual"] = "vad"
    vad: RealtimeVadRequest | None = None


@dataclass
class SessionHandle:
    session: RealtimeSession
    peer_connection: Any
    consumer_tasks: list[asyncio.Task[None]]


class RealtimeSessionManager:
    """Owns active peer connections and their session state."""

    def __init__(
        self,
        *,
        backend_factory: BackendFactory,
        default_model: str,
        rtc_configuration: Any | None = None,
    ) -> None:
        self._backend_factory = backend_factory
        self._default_model = default_model
        self._rtc_configuration = rtc_configuration
        self._sessions: dict[str, SessionHandle] = {}
        self._lock = asyncio.Lock()

    async def create(
        self,
        *,
        model: str | None,
        instructions: str | None,
        max_new_tokens: int,
        output_text: bool,
        input_audio_mode: str,
        vad: RealtimeVadRequest | None,
    ) -> SessionHandle:
        if not AIORTC_AVAILABLE:
            raise HTTPException(
                status_code=503,
                detail=(
                    "Realtime prototype requires aiortc. "
                    "Install the project with the realtime extra."
                ),
            )

        session_id = uuid.uuid4().hex
        output_track = BufferedAudioStreamTrack()
        backend = self._backend_factory(
            model or self._default_model,
            max_new_tokens,
            output_text,
        )
        vad_config = VadConfig()
        if vad is not None:
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

        config = RealtimeSessionConfig(
            instructions=(
                instructions
                or "You are a concise, natural voice assistant. Answer conversationally."
            ),
            input_audio_mode=input_audio_mode,
            vad=vad_config,
        )
        session = RealtimeSession(
            session_id=session_id,
            backend=backend,
            output_track=output_track,
            config=config,
        )
        pc = RTCPeerConnection(configuration=self._rtc_configuration)
        pc.addTrack(output_track)
        handle = SessionHandle(
            session=session,
            peer_connection=pc,
            consumer_tasks=[],
        )
        async with self._lock:
            self._sessions[session_id] = handle
        return handle

    async def close(self, session_id: str) -> None:
        async with self._lock:
            handle = self._sessions.pop(session_id, None)
        if handle is None:
            return

        with contextlib.suppress(Exception):
            await handle.peer_connection.close()

        for task in handle.consumer_tasks:
            task.cancel()
        for task in handle.consumer_tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task

        await handle.session.close()

    async def get(self, session_id: str) -> SessionHandle | None:
        async with self._lock:
            return self._sessions.get(session_id)


def create_realtime_router(
    client: Client | None = None,
    *,
    model_name: str,
    backend_factory: BackendFactory | None = None,
) -> APIRouter:
    if backend_factory is None:
        if client is None:
            raise ValueError(
                "create_realtime_router requires either client or backend_factory"
            )

        def backend_factory(
            resolved_model: str,
            max_new_tokens: int,
            output_text: bool,
        ) -> ResponseBackend:
            output_modalities = ("text", "audio") if output_text else ("audio",)
            return OmniResponseBackend(
                client=client,
                model=resolved_model,
                max_new_tokens=max_new_tokens,
                output_modalities=output_modalities,
            )

    router = APIRouter()
    manager = RealtimeSessionManager(
        backend_factory=backend_factory,
        default_model=model_name,
        rtc_configuration=_load_rtc_configuration_from_env(),
    )

    @router.post("/v1/realtime/webrtc/offer")
    async def realtime_offer(req: RealtimeOfferRequest) -> JSONResponse:
        handle = await manager.create(
            model=req.model,
            instructions=req.instructions,
            max_new_tokens=req.max_new_tokens,
            output_text=req.output_text,
            input_audio_mode=req.input_audio_mode,
            vad=req.vad,
        )
        session = handle.session
        pc = handle.peer_connection

        @pc.on("datachannel")
        def on_datachannel(channel: Any) -> None:
            session.attach_event_channel(channel)

            @channel.on("open")
            def on_open() -> None:
                session.attach_event_channel(channel)
                asyncio.create_task(
                    session.emit_event(
                        "session.created",
                        model=session.backend.model_name,
                        instructions=session.instructions,
                        audio={"input_mode": session.turn_mode},
                    )
                )

            @channel.on("message")
            def on_message(message: Any) -> None:
                if isinstance(message, bytes):
                    return
                try:
                    payload = json.loads(message)
                except json.JSONDecodeError:
                    return
                asyncio.create_task(session.handle_client_event(payload))

        @pc.on("track")
        def on_track(track: Any) -> None:
            if track.kind == "audio":
                task = asyncio.create_task(_consume_audio_track(track, session))
                handle.consumer_tasks.append(task)
            elif track.kind == "video":
                task = asyncio.create_task(_consume_video_track(track, session))
                handle.consumer_tasks.append(task)

        @pc.on("connectionstatechange")
        async def on_connectionstatechange() -> None:
            logger.info(
                "Realtime session %s connection state=%s",
                session.session_id,
                pc.connectionState,
            )
            if pc.connectionState in {"failed", "closed", "disconnected"}:
                await manager.close(session.session_id)

        await pc.setRemoteDescription(RTCSessionDescription(sdp=req.sdp, type=req.type))
        answer = await pc.createAnswer()
        await pc.setLocalDescription(answer)
        await _wait_for_ice_complete(pc)

        local_description = pc.localDescription
        if local_description is None:
            raise HTTPException(status_code=500, detail="Failed to create SDP answer")

        return JSONResponse(
            {
                "session_id": session.session_id,
                "sdp": local_description.sdp,
                "type": local_description.type,
                "model": session.backend.model_name,
            }
        )

    @router.delete("/v1/realtime/sessions/{session_id}")
    async def close_session(session_id: str) -> JSONResponse:
        await manager.close(session_id)
        return JSONResponse({"ok": True, "session_id": session_id})

    return router


async def _wait_for_ice_complete(pc: Any, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while pc.iceGatheringState != "complete" and time.monotonic() < deadline:
        await asyncio.sleep(0.05)


async def _consume_audio_track(track: Any, session: RealtimeSession) -> None:
    while True:
        try:
            frame = await track.recv()
        except (MediaStreamError, InvalidStateError):
            return
        except Exception:
            logger.exception(
                "Audio track consumer failed for session %s", session.session_id
            )
            return

        audio = audio_frame_to_ndarray(frame)
        sample_rate = frame.sample_rate or 48000
        await session.handle_audio_chunk(
            audio,
            sample_rate,
            timestamp=time.monotonic(),
        )


async def _consume_video_track(track: Any, session: RealtimeSession) -> None:
    while True:
        try:
            frame = await track.recv()
        except (MediaStreamError, InvalidStateError):
            return
        except Exception:
            logger.exception(
                "Video track consumer failed for session %s", session.session_id
            )
            return

        frame_rgb = frame.to_ndarray(format="rgb24")
        await session.handle_video_frame(frame_rgb, timestamp=time.monotonic())
