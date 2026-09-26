from __future__ import annotations

import asyncio
import base64
import binascii
import enum
import json
import logging
import uuid
from collections import deque
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np
from fastapi import WebSocket
from pydantic import ValidationError
from starlette.websockets import WebSocketState

from sglang_omni.client import Client, GenerateRequest
from sglang_omni.config import RealtimeTranscriptionConfig
from sglang_omni.serve.realtime.audio_buffer import (
    PCM16_BYTES_PER_SAMPLE,
    PCM_SAMPLE_RATE,
    BufferOverflow,
    RealtimeAudioBuffer,
)
from sglang_omni.serve.realtime.events import (
    InputAudioBufferAppend,
    InputAudioBufferClear,
    InputAudioBufferCommit,
    TranscriptionCleared,
    TranscriptionCommitted,
    TranscriptionCompleted,
    TranscriptionDone,
    TranscriptionError,
    TranscriptionErrorBody,
    TranscriptionSegment,
    TranscriptionServerEvent,
    TranscriptionSessionCreated,
    TranscriptionSessionObject,
    TranscriptionSessionUpdate,
    TranscriptionSessionUpdated,
    TranscriptionSpeechStarted,
    TranscriptionSpeechStopped,
    TurnDetection,
    TurnDetectionType,
    parse_transcription_client_event,
)
from sglang_omni.serve.realtime.vad import (
    VAD_FRAME_SAMPLES,
    StatelessVAD,
    VADConfig,
    offsets_to_ms,
)
from sglang_omni.serve.transcription_chunking import (
    SILENT_CHUNK_PEAK_THRESHOLD,
    join_transcript_parts,
)

logger = logging.getLogger(__name__)

_SILENT_PCM16_PEAK = round(SILENT_CHUNK_PEAK_THRESHOLD * 32768)
_VAD_FRAME_MS = VAD_FRAME_SAMPLES * 1000 // PCM_SAMPLE_RATE
# Memory guard for models that never split a segment on length. This bounds
# the PCM held per session
_UNBOUNDED_BUFFER_S = 600.0


class StreamingASRStrategy(Protocol):
    def create_state(self, *, model_name: str, language: str | None) -> object: ...

    def build_decode_request(
        self,
        *,
        audio: bytes,
        state: object,
        is_final: bool,
        request_id: str,
    ) -> GenerateRequest: ...

    def update_hypothesis(
        self,
        *,
        generated_text: str,
        language: str | None,
        state: object,
    ) -> str: ...


class SessionState(enum.Enum):
    RECEIVING = "receiving"
    # transcription.done was received; only the pending finals are still running.
    INPUT_DONE = "input_done"
    CLOSED = "closed"


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex}"


@dataclass(slots=True)
class TranscriptionSessionSettings:
    language: str | None = None
    decode_interval_ms: int = 2000
    # the session seeds this from the model's server_vad declaration,
    # and the client may change it via session.update.
    turn_detection: TurnDetection | None = None


@dataclass(slots=True)
class ActiveTranscriptionSegment:
    segment_id: int
    start_sample: int
    strategy_state: object
    next_refresh_sample: int
    decode_attempt: int = 0
    last_text: str = ""


@dataclass(slots=True)
class FinalDecode:
    segment: ActiveTranscriptionSegment
    pcm: bytes
    audio: bytes
    done: asyncio.Future[None]


@dataclass(frozen=True, slots=True)
class CommittedTranscriptionSegment:
    segment_id: int
    text: str


class RealtimeTranscriptionSession:
    handlers = {
        TranscriptionSessionUpdate: "handle_session_update",
        InputAudioBufferAppend: "handle_audio_append",
        InputAudioBufferClear: "handle_audio_clear",
        InputAudioBufferCommit: "handle_audio_commit",
        TranscriptionDone: "handle_transcription_done",
    }

    def __init__(
        self,
        websocket: WebSocket,
        *,
        client: Client,
        model_name: str,
        transcription_config: RealtimeTranscriptionConfig,
        strategy: StreamingASRStrategy,
        session_id: str | None = None,
    ) -> None:
        self.websocket = websocket
        self.client = client
        self.model_name = model_name
        self.session_id = session_id or new_id("sess")
        self.state = SessionState.RECEIVING
        self.event_index = 0
        self._send_lock = asyncio.Lock()
        self.transcription_config = transcription_config
        self.strategy = strategy
        self.settings = TranscriptionSessionSettings(
            decode_interval_ms=transcription_config.decode_interval_ms,
            turn_detection=(
                TurnDetection(type=TurnDetectionType.SERVER_VAD)
                if transcription_config.server_vad
                else None
            ),
        )
        max_segment_s = transcription_config.max_segment_s
        self.max_segment_samples = (
            int(max_segment_s * PCM_SAMPLE_RATE) if max_segment_s is not None else None
        )
        self.refresh_interval_samples = (
            transcription_config.decode_interval_ms * PCM_SAMPLE_RATE // 1000
        )
        max_buffer_seconds = (
            max_segment_s + 4 if max_segment_s is not None else _UNBOUNDED_BUFFER_S
        )
        max_buffer_bytes = int(
            max_buffer_seconds * PCM_SAMPLE_RATE * PCM16_BYTES_PER_SAMPLE
        )
        self.audio_buffer = RealtimeAudioBuffer(
            source_sr=PCM_SAMPLE_RATE,
            target_sr=PCM_SAMPLE_RATE,
            max_bytes=max_buffer_bytes,
        )
        self.vad: StatelessVAD | None = self.new_vad(self.settings.turn_detection)
        self.active_segment: ActiveTranscriptionSegment | None = None
        self.committed_segments: list[CommittedTranscriptionSegment] = []
        self._next_segment_id = 0
        self._last_closed_segment_id: int | None = (
            None  # The segment most recently closed by queue_final.
        )
        self._decode_event = asyncio.Event()
        self._pending_finals: deque[FinalDecode] = deque()
        self._final_waiters: set[asyncio.Future[None]] = set()
        self._inflight_request_id: str | None = None
        self._decode_worker_task = self.spawn_decode_worker()

    async def run(self) -> None:
        await self.send(TranscriptionSessionCreated(session=self.session_object()))
        while True:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            if message["type"] != "websocket.receive":
                continue
            try:
                payload = json.loads(message["text"])
            except (KeyError, TypeError, json.JSONDecodeError):
                await self.send_error(
                    "invalid_request_error", "invalid_json", "Invalid JSON event."
                )
                continue
            if not isinstance(payload, dict):
                await self.send_error(
                    "invalid_request_error",
                    "invalid_event",
                    "Top-level payload must be a JSON object.",
                )
                continue
            await self.dispatch(payload)

    async def dispatch(self, payload: dict[str, Any]) -> None:
        if self.state is SessionState.CLOSED:
            return
        try:
            event = parse_transcription_client_event(payload)
        except ValidationError as exc:
            await self.send_error("invalid_request_error", "invalid_event", str(exc))
            return
        if event is None or type(event) not in self.handlers:
            await self.send_error(
                "invalid_request_error",
                "unsupported_event",
                f"Unsupported event type: {payload.get('type')!r}",
            )
            return
        if self.state is SessionState.INPUT_DONE:
            await self.send_error(
                "invalid_request_error",
                "input_already_done",
                f"{event.type} is not accepted after transcription.done.",
            )
            return
        try:
            await getattr(self, self.handlers[type(event)])(event)
        except Exception:
            logger.exception(
                "Realtime transcription handler failed: session=%s event=%s",
                self.session_id,
                event.type,
            )
            await self.send_error(
                "server_error",
                "internal_error",
                f"Internal error while handling {event.type}.",
            )

    async def send(self, event: dict[str, Any] | TranscriptionServerEvent) -> None:
        if self.state is SessionState.CLOSED:
            return
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        if isinstance(event, TranscriptionServerEvent):
            event = event.model_dump(exclude={"event_id", "event_index"})
        async with self._send_lock:
            self.event_index += 1
            event.setdefault("event_id", new_id("evt"))
            event.setdefault("event_index", self.event_index)
            await self.websocket.send_text(json.dumps(event))

    async def send_error(self, type_: str, code: str, message: str) -> None:
        await self.send(
            TranscriptionError(
                error=TranscriptionErrorBody(type=type_, code=code, message=message)
            )
        )

    async def cancel_and_abort(
        self, task: asyncio.Task[Any] | None, request_id: str | None
    ) -> None:
        """Cancel the decode worker, abort its engine request, and absorb the result.

        Gathering with return_exceptions=True prevents a normal cancellation
        from surfacing as a WebSocket handler failure.
        """
        if task is None or task.done():
            return
        task.cancel()
        try:
            if request_id is not None:
                await self.client.abort(request_id)
        except Exception as exc:
            asyncio.get_running_loop().call_exception_handler(
                {
                    "message": "Realtime transcription abort failed",
                    "exception": exc,
                    "task": task,
                }
            )
        finally:
            await asyncio.gather(task, return_exceptions=True)

    def session_object(self) -> TranscriptionSessionObject:
        return TranscriptionSessionObject(
            id=self.session_id,
            model=self.model_name,
            language=self.settings.language,
            decode_interval_ms=self.settings.decode_interval_ms,
            turn_detection=self.settings.turn_detection,
        )

    @staticmethod
    def vad_config(turn_detection: TurnDetection) -> VADConfig:
        settings = turn_detection.model_dump(
            include={"threshold", "prefix_padding_ms", "silence_duration_ms"},
            exclude_none=True,
        )
        return VADConfig(**settings)

    @staticmethod
    def vad_config_error(config: VADConfig) -> str | None:
        # Note (Jeffro): A new segment starts prefix_padding_ms before the frame that woke the VAD,
        # and the previous segment ended silence_duration_ms before that frame.
        # Padding must fit inside the silence window (minus the one frame the VAD reports late) or segments would overlap.
        if config.prefix_padding_ms < 0:
            return "prefix_padding_ms must not be negative."
        if config.silence_duration_ms <= 0:
            return "silence_duration_ms must be positive."
        if config.prefix_padding_ms + _VAD_FRAME_MS > config.silence_duration_ms:
            return (
                "prefix_padding_ms must be at most silence_duration_ms minus "
                f"{_VAD_FRAME_MS} ms."
            )
        return None

    @classmethod
    def new_vad(cls, turn_detection: TurnDetection | None) -> StatelessVAD | None:
        if turn_detection is None:
            return None
        return StatelessVAD(cls.vad_config(turn_detection))

    async def handle_session_update(self, event: TranscriptionSessionUpdate) -> None:
        update = event.session.model_dump(exclude_unset=True)
        if not self.audio_buffer.is_empty() and any(
            key in update for key in ("language", "turn_detection")
        ):
            await self.send_error(
                "invalid_request_error",
                "session_active",
                "Language and VAD settings cannot change "
                "while uncommitted audio is buffered.",
            )
            return

        if "language" in update:
            language = update["language"]
            self.settings.language = language.strip() if language else None
        if "turn_detection" in update:
            turn_detection = event.session.turn_detection
            if (
                turn_detection is not None
                and turn_detection.type != TurnDetectionType.SERVER_VAD
            ):
                await self.send_error(
                    "invalid_request_error",
                    "unsupported_turn_detection",
                    "Realtime transcription supports only server_vad or null.",
                )
                return
            if turn_detection is not None and not self.transcription_config.server_vad:
                await self.send_error(
                    "invalid_request_error",
                    "unsupported_turn_detection",
                    "This model does not support server-side turn detection.",
                )
                return
            if turn_detection is not None:
                problem = self.vad_config_error(self.vad_config(turn_detection))
                if problem is not None:
                    await self.send_error(
                        "invalid_request_error", "invalid_turn_detection", problem
                    )
                    return
            if self.vad is not None:
                self.vad.reset()
            self.settings.turn_detection = turn_detection
            self.vad = self.new_vad(turn_detection)
        await self.send(TranscriptionSessionUpdated(session=self.session_object()))

    async def handle_audio_append(self, event: InputAudioBufferAppend) -> None:
        try:
            pcm = base64.b64decode(event.audio, validate=False)
        except (ValueError, binascii.Error):
            await self.send_error(
                "invalid_request_error", "invalid_audio", "Audio must be base64 PCM16."
            )
            return
        if len(pcm) % 2:
            await self.send_error(
                "invalid_request_error",
                "invalid_audio",
                "PCM16 audio must contain complete 16-bit samples.",
            )
            return
        append_start_sample = self.audio_buffer.end_sample
        try:
            self.audio_buffer.append_bytes(pcm)
        except BufferOverflow as exc:
            await self.send_error(
                "invalid_request_error", "audio_buffer_overflow", str(exc)
            )
            return
        if self.vad is None:
            # Note (Jeffro): The VAD = None means as long as pcm comes in and there is no active segment currently,
            # we can just start a new segment.
            if self.active_segment is None and pcm:
                self.start_segment(append_start_sample)
        else:

            async def vad_on_started(start_sample: int) -> None:
                # Prefix padding cannot reach audio the buffer no longer holds.
                start_sample = max(self.audio_buffer.start_sample, start_sample)
                segment = self.start_segment(start_sample)
                await self.send(
                    TranscriptionSpeechStarted(
                        audio_start_ms=offsets_to_ms(start_sample),
                        segment_id=segment.segment_id,
                    )
                )

            async def vad_on_stopped(end_sample: int) -> None:
                segment = self.active_segment

                has_speech = end_sample > segment.start_sample
                await self.send(
                    TranscriptionSpeechStopped(
                        audio_end_ms=offsets_to_ms(end_sample),
                        segment_id=(
                            segment.segment_id
                            if has_speech
                            else self._last_closed_segment_id
                        ),
                    )
                )
                if has_speech:
                    await self.finalize_through(end_sample)
                elif segment.last_text:
                    # A partial was already reported, so it still owes a final.
                    await self.finalize_through(self.audio_buffer.end_sample)
                else:
                    self.active_segment = None

            await self.vad.process(
                pcm,
                append_start_sample,
                # active_segment is the ONLY record of whether someone is speaking.
                in_speech=lambda: self.active_segment is not None,
                on_started=vad_on_started,
                on_stopped=vad_on_stopped,
            )

        await self.enforce_hard_limit()
        self.trim_idle_prefix()
        self.maybe_schedule_partial()

    def start_segment(self, start_sample: int) -> ActiveTranscriptionSegment:
        segment = ActiveTranscriptionSegment(
            segment_id=self._next_segment_id,
            start_sample=start_sample,
            strategy_state=self.strategy.create_state(
                model_name=self.model_name,
                language=self.settings.language,
            ),
            next_refresh_sample=start_sample + self.refresh_interval_samples,
        )
        self._next_segment_id += 1
        self.active_segment = segment
        return segment

    async def enforce_hard_limit(self) -> None:
        segment = self.active_segment
        max_samples = self.max_segment_samples
        if segment is None or max_samples is None:
            return
        full_blocks = (
            self.audio_buffer.end_sample - segment.start_sample
        ) // max_samples
        if full_blocks == 0:
            return
        cut = segment.start_sample + full_blocks * max_samples
        await self.finalize_through(cut)
        self.start_segment(cut)

    def trim_idle_prefix(self) -> None:
        """Bound the buffer while server VAD holds no active speech turn.

        Audio that arrives between turns is silence the model never decodes
        and that :meth:`_queue_final` never drains. Retaining only the VAD
        prefix padding (plus a frame of slack for audio the VAD has not
        consumed yet) keeps a long silent stretch from growing the buffer
        into BufferOverflow.
        """
        if self.vad is None or self.active_segment is not None:
            return
        keep_samples = (
            self.vad.config.prefix_padding_ms * PCM_SAMPLE_RATE // 1000
            + 2 * VAD_FRAME_SAMPLES
        )
        self.audio_buffer.drop_before(self.audio_buffer.end_sample - keep_samples)

    async def finalize_through(self, end_sample: int) -> None:
        if self.active_segment is None:
            return
        end_sample = min(end_sample, self.audio_buffer.end_sample)
        max_samples = self.max_segment_samples
        while (
            max_samples is not None
            and end_sample - self.active_segment.start_sample > max_samples
        ):
            cut = self.active_segment.start_sample + max_samples
            await self.queue_final(cut)
            self.start_segment(cut)
        if (
            self.active_segment is not None
            and end_sample > self.active_segment.start_sample
        ):
            await self.queue_final(end_sample)

    async def queue_final(self, end_sample: int) -> None:
        segment = self.active_segment
        if segment is None or end_sample <= segment.start_sample:
            return
        pcm = self.audio_buffer.slice(segment.start_sample, end_sample)
        done = asyncio.get_running_loop().create_future()
        done.add_done_callback(self._final_waiters.discard)
        self._final_waiters.add(done)
        self._pending_finals.append(
            FinalDecode(
                segment=segment,
                pcm=pcm,
                audio=self.audio_buffer.pcm_to_wav_bytes(pcm),
                done=done,
            )
        )

        self.audio_buffer.drop_before(end_sample)
        self.active_segment = None
        self._last_closed_segment_id = segment.segment_id
        self._decode_event.set()
        await self.send(
            TranscriptionCommitted(
                segment_id=segment.segment_id,
            )
        )

    def maybe_schedule_partial(self) -> None:
        segment = self.active_segment
        if segment is None:
            return
        if self.audio_buffer.end_sample >= segment.next_refresh_sample:
            self._decode_event.set()

    def spawn_decode_worker(self) -> asyncio.Task[None]:
        task = asyncio.create_task(self.decode_worker())
        task.add_done_callback(self.log_decode_worker_exit)
        return task

    def log_decode_worker_exit(self, task: asyncio.Task[None]) -> None:
        # A worker that dies leaves later appends signalling an event nobody
        # waits on, make sure the cause reaches the log.
        if task.cancelled() or task.exception() is None:
            return
        logger.error(
            "Realtime transcription decode worker crashed: session=%s",
            self.session_id,
            exc_info=task.exception(),
        )

    async def decode_worker(self) -> None:
        while True:
            await self._decode_event.wait()
            self._decode_event.clear()
            if self.state is SessionState.CLOSED:
                return

            while self._pending_finals:
                final = self._pending_finals.popleft()
                try:
                    if self.is_silent(final.pcm):
                        await self.emit_hypothesis(final.segment, "", is_final=True)
                    else:
                        await self.decode_and_emit(
                            final.segment, final.audio, is_final=True
                        )
                finally:
                    if not final.done.done():
                        final.done.set_result(None)

            segment = self.active_segment
            if segment is None:
                continue
            end_sample = self.audio_buffer.end_sample
            if end_sample < segment.next_refresh_sample:
                continue
            segment.next_refresh_sample = end_sample + self.refresh_interval_samples
            pcm = self.audio_buffer.slice(segment.start_sample)
            if self.is_silent(pcm):
                continue
            await self.decode_and_emit(
                segment,
                self.audio_buffer.pcm_to_wav_bytes(pcm),
                is_final=False,
            )

    @staticmethod
    def is_silent(pcm: bytes) -> bool:
        if not pcm:
            return True
        samples = np.frombuffer(pcm, dtype="<i2")
        return bool(
            samples.size == 0
            or np.max(np.abs(samples.astype(np.int32))) < _SILENT_PCM16_PEAK
        )

    async def decode_and_emit(
        self,
        segment: ActiveTranscriptionSegment,
        audio: bytes,
        *,
        is_final: bool,
    ) -> None:
        segment.decode_attempt += 1
        request_id = f"{self.session_id}:{segment.segment_id}:{segment.decode_attempt}"
        request = self.strategy.build_decode_request(
            audio=audio,
            state=segment.strategy_state,
            is_final=is_final,
            request_id=request_id,
        )
        self._inflight_request_id = request_id
        try:
            result = await self.client.completion(request, request_id=request_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.send_error("server_error", "transcription_failed", str(exc))
            return
        finally:
            if self._inflight_request_id == request_id:
                self._inflight_request_id = None
        if not is_final and self.active_segment is not segment:
            return
        text = self.strategy.update_hypothesis(
            generated_text=result.text,
            language=result.language,
            state=segment.strategy_state,
        )
        if not is_final and text == segment.last_text:
            return
        await self.emit_hypothesis(segment, text, is_final=is_final)

    async def emit_hypothesis(
        self,
        segment: ActiveTranscriptionSegment,
        text: str,
        *,
        is_final: bool,
    ) -> None:
        segment.last_text = text
        await self.send(
            TranscriptionSegment(
                segment_id=segment.segment_id,
                text=text,
                is_final=is_final,
            )
        )
        if is_final:
            self.committed_segments.append(
                CommittedTranscriptionSegment(
                    segment_id=segment.segment_id,
                    text=text,
                )
            )

    async def handle_audio_commit(self, event: InputAudioBufferCommit) -> None:
        del event
        await self.commit_buffer("client_commit")

    async def handle_audio_clear(self, event: InputAudioBufferClear) -> None:
        del event
        request_id = self._inflight_request_id
        await self.cancel_and_abort(self._decode_worker_task, request_id)
        for final in self._pending_finals:
            if not final.done.done():
                final.done.cancel()
        for waiter in list(self._final_waiters):
            if not waiter.done():
                waiter.cancel()
        self._pending_finals.clear()
        self._final_waiters.clear()
        self._decode_event.clear()

        self.audio_buffer.clear()
        self.active_segment = None
        if self.vad is not None:
            self.vad.reset()

        self._decode_worker_task = self.spawn_decode_worker()
        await self.send(TranscriptionCleared())

    async def commit_buffer(self, reason: str) -> None:
        end_sample = self.audio_buffer.end_sample
        if self.active_segment is None and not self.audio_buffer.is_empty():
            if reason == "session_end" and self.vad is not None:
                # Note (Akazaakane): With server VAD, buffered audio outside an
                # active speech turn is trailing silence and must not free-run ASR.
                self.audio_buffer.clear()
                self.vad.reset()
                return
            self.start_segment(self.audio_buffer.start_sample)
        await self.finalize_through(end_sample)
        if self.vad is not None:
            self.vad.reset()

    async def handle_transcription_done(self, event: TranscriptionDone) -> None:
        del event
        self.state = SessionState.INPUT_DONE
        await self.commit_buffer("session_end")
        if self._final_waiters:
            await asyncio.gather(*list(self._final_waiters))
        await self.cancel_and_abort(self._decode_worker_task, None)
        ordered = sorted(self.committed_segments, key=lambda item: item.segment_id)
        await self.send(
            TranscriptionCompleted(
                text=join_transcript_parts(item.text for item in ordered),
            )
        )

    async def teardown(self) -> None:
        self.state = SessionState.CLOSED
        self.active_segment = None
        request_id = self._inflight_request_id
        await self.cancel_and_abort(self._decode_worker_task, request_id)
        for final in self._pending_finals:
            if not final.done.done():
                final.done.cancel()
        for waiter in list(self._final_waiters):
            if not waiter.done():
                waiter.cancel()
        self._pending_finals.clear()
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.close()


__all__ = ["RealtimeTranscriptionSession", "StreamingASRStrategy"]
