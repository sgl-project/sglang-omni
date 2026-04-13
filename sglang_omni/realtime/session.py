# SPDX-License-Identifier: Apache-2.0
"""Realtime session orchestration shared across interactive transports."""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from sglang_omni.realtime.backend import ResponseBackend, TurnContext
from sglang_omni.realtime.media import mono_float32, resample_linear, resize_rgb_frame
from sglang_omni.realtime.utils import throttle
from sglang_omni.realtime.vad import EnergyVad, VadConfig


@dataclass
class VideoFrameSample:
    ts_monotonic: float
    frame_rgb: np.ndarray


@dataclass
class VideoBufferConfig:
    ingest_fps: float = 2.0
    clip_window_s: float = 4.0
    max_buffer_s: float = 8.0
    max_frames: int = 16
    resize_width: int = 224
    resize_height: int = 224


@dataclass
class VideoBufferState:
    config: VideoBufferConfig = field(default_factory=VideoBufferConfig)
    frames: deque[VideoFrameSample] = field(default_factory=deque)
    last_ingest_ts: float | None = None
    total_frames_received: int = 0


@dataclass
class AudioTurnState:
    sample_rate: int
    chunks: list[np.ndarray] = field(default_factory=list)
    speech_start_ts: float | None = None
    speech_end_ts: float | None = None


@dataclass
class PendingTurn:
    audio: np.ndarray | None
    sample_rate: int | None
    user_text: str | None
    speech_end_ts: float | None
    turn_index: int


@dataclass
class RealtimeSessionConfig:
    instructions: str = (
        "You are a concise, natural voice assistant. Answer conversationally."
    )
    input_audio_sample_rate: int = 16000
    input_audio_mode: str = "vad"
    vad: VadConfig = field(default_factory=VadConfig)
    video: VideoBufferConfig = field(default_factory=VideoBufferConfig)


class RealtimeSession:
    """Conversation/session state above the request-oriented pipeline."""

    def __init__(
        self,
        *,
        session_id: str,
        backend: ResponseBackend,
        output_track: Any,
        config: RealtimeSessionConfig,
    ) -> None:
        self.session_id = session_id
        self.backend = backend
        self.output_track = output_track
        self.config = config

        self.instructions = config.instructions.strip()
        self.history: list[dict[str, str]] = []
        self.current_user_text: str | None = None
        self.current_audio = AudioTurnState(sample_rate=config.input_audio_sample_rate)
        self.video = VideoBufferState(config=config.video)
        self._audio_chunk_count = 0
        self.turn_mode = "vad"
        self.manual_recording = False

        self.vad = EnergyVad(config.vad)
        self._preroll_chunks: deque[np.ndarray] = deque()
        self._preroll_samples = 0
        self._event_channel: Any | None = None
        self._event_backlog: list[dict[str, Any]] = []
        self._response_lock = asyncio.Lock()
        self._closed = False
        self._throttle_state: dict[str, float] = {}
        self._turn_index = 0
        self._queued_pending_turn: PendingTurn | None = None

        self.active_response_id: str | None = None
        self.active_task: asyncio.Task[None] | None = None
        self.assistant_playing = False
        self._set_turn_mode(config.input_audio_mode)

    async def emit_event(self, event_type: str, **payload: Any) -> None:
        event = {"type": event_type, "session_id": self.session_id, **payload}
        channel = self._event_channel
        if channel is None or getattr(channel, "readyState", None) != "open":
            self._event_backlog.append(event)
            return
        try:
            channel.send(json.dumps(event))
        except Exception:
            self._event_backlog.append(event)

    def attach_event_channel(self, channel: Any) -> None:
        self._event_channel = channel
        self._flush_event_backlog()

    def _flush_event_backlog(self) -> None:
        channel = self._event_channel
        if channel is None or getattr(channel, "readyState", None) != "open":
            return
        pending = list(self._event_backlog)
        self._event_backlog.clear()
        for event in pending:
            try:
                channel.send(json.dumps(event))
            except Exception:
                self._event_backlog.append(event)
                break

    async def handle_client_event(self, payload: dict[str, Any]) -> None:
        event_type = str(payload.get("type") or "").strip()
        if event_type == "session.update":
            session = payload.get("session") or {}
            updated_session: dict[str, Any] = {}
            instructions = session.get("instructions") or payload.get("instructions")
            if isinstance(instructions, str):
                self.instructions = instructions.strip()
                updated_session["instructions"] = self.instructions
            audio_config = session.get("audio") or {}
            input_mode = (
                audio_config.get("input_mode")
                or audio_config.get("turn_mode")
                or payload.get("input_audio_mode")
            )
            if isinstance(input_mode, str) and self._set_turn_mode(input_mode):
                updated_session["audio"] = {"input_mode": self.turn_mode}
            if updated_session:
                await self.emit_event("session.updated", session=updated_session)
            return

        if event_type == "input_audio_buffer.start":
            if self._closed:
                return
            if self.active_task is not None or self.assistant_playing:
                await self._interrupt_active_response(reason="barge_in")
            self.turn_mode = "manual"
            self.manual_recording = True
            self.current_audio = AudioTurnState(
                sample_rate=self.config.input_audio_sample_rate,
                speech_start_ts=time.monotonic(),
            )
            self._preroll_chunks.clear()
            self._preroll_samples = 0
            await self.emit_event("input_audio_buffer.manual_started")
            return

        if event_type == "input_audio_buffer.commit":
            if self.turn_mode != "manual":
                return
            self.manual_recording = False
            if not self.current_audio.chunks:
                await self.emit_event("input_audio_buffer.manual_committed", empty=True)
                self.current_audio = AudioTurnState(
                    sample_rate=self.config.input_audio_sample_rate
                )
                return

            self.current_audio.speech_end_ts = time.monotonic()
            await self.emit_event(
                "input_audio_buffer.manual_committed",
                empty=False,
                sample_count=int(
                    sum(chunk.size for chunk in self.current_audio.chunks)
                ),
            )
            pending = self._consume_pending_turn()
            if pending is not None:
                await self._start_or_queue_response(pending)
            return

        if event_type == "conversation.item.create":
            item = payload.get("item") or {}
            if item.get("role") == "user" and isinstance(item.get("content"), str):
                self.current_user_text = item["content"]
                await self.emit_event("conversation.item.created", item=item)
            return

        if event_type == "response.create":
            if self._closed:
                return
            if self.active_task is not None or self.assistant_playing:
                await self._interrupt_active_response(reason="barge_in")
            pending = self._consume_pending_turn()
            if pending is not None:
                await self._start_or_queue_response(pending)
            return

        if event_type == "response.cancel" and (
            self.active_response_id is not None or self.active_task is not None
        ):
            await self._interrupt_active_response(reason="client")

    async def close(self) -> None:
        self._closed = True
        self._queued_pending_turn = None
        if (
            self.active_response_id is not None
            and self.backend.capabilities.supports_cancel
        ):
            await self.backend.cancel(self.active_response_id)
        if self.active_task is not None:
            self.active_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self.active_task
        await self.output_track.clear()

    async def handle_audio_chunk(
        self,
        audio: np.ndarray,
        sample_rate: int,
        *,
        timestamp: float | None = None,
    ) -> None:
        if self._closed:
            return

        ts = timestamp if timestamp is not None else time.monotonic()
        chunk = mono_float32(audio)
        if sample_rate != self.config.input_audio_sample_rate:
            chunk = resample_linear(
                chunk,
                sample_rate,
                self.config.input_audio_sample_rate,
            )
        if chunk.size == 0:
            return

        self._audio_chunk_count += 1
        if self.turn_mode == "manual":
            if not self.manual_recording:
                return
            self.current_audio.chunks.append(chunk)
            rms = self.vad.measure_level(chunk)
            dc_offset = float(np.mean(chunk)) if chunk.size else 0.0
            await self._emit_audio_chunk_received(
                timestamp=ts,
                chunk_count=self._audio_chunk_count,
                sample_count=int(chunk.size),
                sample_rate=int(self.config.input_audio_sample_rate),
                rms=rms,
                dc_offset=dc_offset,
                frame_count=0,
                voiced_frame_count=0,
                speech_ratio=0.0,
                speaking_before=True,
                speaking_after=True,
            )
            return

        was_speaking = self.vad.speaking
        if not was_speaking:
            self._append_preroll(chunk)

        rms = self.vad.measure_level(chunk)
        dc_offset = float(np.mean(chunk)) if chunk.size else 0.0
        event = self.vad.process(chunk)
        await self._emit_audio_chunk_received(
            timestamp=ts,
            chunk_count=self._audio_chunk_count,
            sample_count=int(chunk.size),
            sample_rate=int(self.config.input_audio_sample_rate),
            rms=rms,
            dc_offset=dc_offset,
            frame_count=int(getattr(self.vad, "last_frame_count", 0)),
            voiced_frame_count=int(getattr(self.vad, "last_voiced_frame_count", 0)),
            speech_ratio=float(getattr(self.vad, "last_speech_ratio", 0.0)),
            speaking_before=bool(was_speaking),
            speaking_after=bool(self.vad.speaking),
        )
        if event.speech_started:
            if self.active_task is not None or self.assistant_playing:
                await self._interrupt_active_response(reason="barge_in")
            self.current_audio = AudioTurnState(
                sample_rate=self.config.input_audio_sample_rate,
                chunks=list(self._preroll_chunks),
                speech_start_ts=ts,
            )
            self._preroll_chunks.clear()
            self._preroll_samples = 0
            await self.emit_event("input_audio_buffer.speech_started")
        elif was_speaking:
            self.current_audio.chunks.append(chunk)

        if event.speech_stopped and self.current_audio.chunks:
            self.current_audio.speech_end_ts = ts
            await self.emit_event("input_audio_buffer.speech_stopped")
            pending = self._consume_pending_turn()
            if pending is not None:
                await self._start_or_queue_response(pending)

    async def handle_video_frame(
        self,
        frame_rgb: np.ndarray,
        *,
        timestamp: float | None = None,
    ) -> None:
        if self._closed:
            return
        ts = timestamp if timestamp is not None else time.monotonic()
        cfg = self.video.config
        if (
            self.video.last_ingest_ts is not None
            and cfg.ingest_fps > 0
            and (ts - self.video.last_ingest_ts) < (1.0 / cfg.ingest_fps)
        ):
            return

        resized = resize_rgb_frame(
            frame_rgb,
            width=cfg.resize_width,
            height=cfg.resize_height,
        )
        self.video.frames.append(VideoFrameSample(ts_monotonic=ts, frame_rgb=resized))
        self.video.last_ingest_ts = ts
        self.video.total_frames_received += 1

        min_allowed_ts = ts - cfg.max_buffer_s
        while self.video.frames and self.video.frames[0].ts_monotonic < min_allowed_ts:
            self.video.frames.popleft()
        while len(self.video.frames) > cfg.max_frames:
            self.video.frames.popleft()

        await self._emit_video_frame_received(
            timestamp=ts,
            frame_count=self.video.total_frames_received,
            buffered_frames=len(self.video.frames),
            width=int(resized.shape[1]),
            height=int(resized.shape[0]),
        )

    def _append_preroll(self, chunk: np.ndarray) -> None:
        max_samples = int(
            round(self.config.vad.preroll_s * self.config.input_audio_sample_rate)
        )
        self._preroll_chunks.append(chunk)
        self._preroll_samples += chunk.shape[0]
        while self._preroll_chunks and self._preroll_samples > max_samples:
            removed = self._preroll_chunks.popleft()
            self._preroll_samples -= removed.shape[0]

    def _consume_pending_turn(self) -> PendingTurn | None:
        user_text = self.current_user_text.strip() if self.current_user_text else None
        if not self.current_audio.chunks and not user_text:
            return None
        audio = None
        sample_rate: int | None = None
        if self.current_audio.chunks:
            audio = np.concatenate(self.current_audio.chunks, axis=0)
            sample_rate = self.current_audio.sample_rate
        self._turn_index += 1
        pending = PendingTurn(
            audio=audio,
            sample_rate=sample_rate,
            user_text=user_text,
            speech_end_ts=self.current_audio.speech_end_ts,
            turn_index=self._turn_index,
        )
        self.current_audio = AudioTurnState(
            sample_rate=self.config.input_audio_sample_rate
        )
        self.current_user_text = None
        return pending

    def _set_turn_mode(self, mode: str) -> bool:
        normalized = str(mode).strip().lower()
        if normalized not in {"vad", "manual"}:
            return False
        self.turn_mode = normalized
        self.manual_recording = False
        self.current_audio = AudioTurnState(
            sample_rate=self.config.input_audio_sample_rate
        )
        self._preroll_chunks.clear()
        self._preroll_samples = 0
        reset_vad = getattr(self.vad, "reset", None)
        if callable(reset_vad):
            reset_vad()
        else:
            self.vad.speaking = False
        return True

    async def _start_or_queue_response(self, pending: PendingTurn) -> None:
        if self.active_task is None:
            self.assistant_playing = True
            self.active_task = asyncio.create_task(self._run_response(pending))
            return
        self._queued_pending_turn = pending

    async def _interrupt_active_response(self, *, reason: str) -> None:
        had_active_response = (
            self.active_task is not None
            or self.active_response_id is not None
            or self.assistant_playing
        )
        if not had_active_response:
            return

        response_id = self.active_response_id
        task = self.active_task
        if response_id is not None and self.backend.capabilities.supports_cancel:
            await self.backend.cancel(response_id)
        elif task is not None:
            task.cancel()

        await self.output_track.clear()
        self.assistant_playing = False
        await self.emit_event(
            "response.cancelled",
            response_id=response_id,
            reason=reason,
        )

    @throttle(1.0, timestamp_kw="timestamp")
    async def _emit_audio_chunk_received(
        self,
        *,
        timestamp: float,
        chunk_count: int,
        sample_count: int,
        sample_rate: int,
        rms: float,
        dc_offset: float,
        frame_count: int,
        voiced_frame_count: int,
        speech_ratio: float,
        speaking_before: bool,
        speaking_after: bool,
    ) -> None:
        await self.emit_event(
            "input_audio_buffer.chunk_received",
            chunk_count=chunk_count,
            sample_count=sample_count,
            sample_rate=sample_rate,
            rms=rms,
            dc_offset=dc_offset,
            frame_count=frame_count,
            voiced_frame_count=voiced_frame_count,
            speech_ratio=speech_ratio,
            speaking_before=speaking_before,
            speaking_after=speaking_after,
        )

    @throttle(1.0, timestamp_kw="timestamp")
    async def _emit_video_frame_received(
        self,
        *,
        timestamp: float,
        frame_count: int,
        buffered_frames: int,
        width: int,
        height: int,
    ) -> None:
        await self.emit_event(
            "input_video_buffer.frame_received",
            frame_count=frame_count,
            buffered_frames=buffered_frames,
            width=width,
            height=height,
        )

    def sample_recent_video_clip(
        self,
        *,
        anchor_ts: float | None,
    ) -> tuple[torch.Tensor | None, float | None]:
        if anchor_ts is None:
            return None, None

        window_start = anchor_ts - self.video.config.clip_window_s
        frames = [
            item.frame_rgb
            for item in self.video.frames
            if window_start <= item.ts_monotonic <= anchor_ts
        ]
        if not frames:
            return None, None

        clip = torch.stack(
            [
                torch.from_numpy(np.array(frame, copy=True)).permute(2, 0, 1)
                for frame in frames
            ],
            dim=0,
        )
        fps = len(frames) / max(self.video.config.clip_window_s, 1e-6)
        return clip, fps

    async def _run_response(self, pending: PendingTurn) -> None:
        async with self._response_lock:
            self.assistant_playing = True
            finish_reason = "stop"
            assistant_text_parts: list[str] = []

            await self.output_track.clear()
            clip, fps = self.sample_recent_video_clip(anchor_ts=pending.speech_end_ts)
            video_frame_count = int(clip.shape[0]) if clip is not None else 0
            await self.emit_event(
                "turn.prepared",
                audio_sample_count=(
                    int(pending.audio.size) if pending.audio is not None else 0
                ),
                audio_sample_rate=(
                    int(pending.sample_rate)
                    if pending.sample_rate is not None
                    else None
                ),
                video_frame_count=video_frame_count,
                video_fps=float(fps) if fps is not None else None,
            )
            turn = TurnContext(
                session_id=self.session_id,
                history=list(self.history),
                instructions=self.instructions,
                user_text=pending.user_text,
                user_audio=pending.audio,
                user_audio_sample_rate=pending.sample_rate,
                recent_video=clip,
                recent_video_fps=fps,
                turn_index=pending.turn_index,
            )

            try:
                async for event in self.backend.stream_response(turn):
                    if event.type == "response_started":
                        self.active_response_id = event.response_id
                        await self.emit_event(
                            "response.created",
                            response_id=event.response_id,
                            model=self.backend.model_name,
                        )
                        continue

                    if event.type == "text_delta" and event.text:
                        response_id = event.response_id
                        assistant_text_parts.append(event.text)
                        await self.emit_event(
                            "response.output_text.delta",
                            response_id=response_id,
                            delta=event.text,
                        )
                        continue

                    if event.type == "audio_chunk" and event.audio is not None:
                        response_id = event.response_id
                        sample_rate = (
                            event.sample_rate or self.config.input_audio_sample_rate
                        )
                        audio_np = np.asarray(event.audio)
                        await self.output_track.enqueue(audio_np, sample_rate)
                        await self.emit_event(
                            "response.output_audio.delta",
                            response_id=response_id,
                            sample_rate=sample_rate,
                            sample_count=int(audio_np.size),
                        )
                        continue

                    if event.type == "done":
                        finish_reason = event.finish_reason or "stop"
                        continue

                    if event.type == "error":
                        await self.emit_event(
                            "error",
                            error={
                                "message": event.error or "Unknown backend error",
                                "response_id": event.response_id,
                            },
                        )
                        return
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                await self.emit_event(
                    "error",
                    error={
                        "message": str(exc),
                        "response_id": self.active_response_id,
                    },
                )
            else:
                assistant_text = "".join(assistant_text_parts).strip()
                if pending.user_text:
                    self.history.append({"role": "user", "content": pending.user_text})
                if assistant_text:
                    self.history.append(
                        {"role": "assistant", "content": assistant_text}
                    )
                await self.emit_event(
                    "response.done",
                    response_id=self.active_response_id,
                    finish_reason=finish_reason,
                    text=assistant_text,
                )
            finally:
                drain_deadline = time.monotonic() + 10.0
                while (
                    getattr(self.output_track, "pending_samples", 0) > 0
                    and time.monotonic() < drain_deadline
                ):
                    await asyncio.sleep(0.05)
                self.assistant_playing = False
                self.active_response_id = None
                self.active_task = None
                queued_pending = self._queued_pending_turn
                self._queued_pending_turn = None
                if queued_pending is not None and not self._closed:
                    self.assistant_playing = True
                    self.active_task = asyncio.create_task(
                        self._run_response(queued_pending)
                    )
