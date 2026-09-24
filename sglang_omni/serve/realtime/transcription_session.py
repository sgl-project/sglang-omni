from __future__ import annotations

import asyncio
import base64
import binascii
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
    StreamingVAD,
    VADConfig,
    VADEvent,
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
        self.closed = False
        self.event_index = 0
        self.send_lock = asyncio.Lock()
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
        max_buffer_seconds = (
            max_segment_s + 4 if max_segment_s is not None else _UNBOUNDED_BUFFER_S
        )
        max_buffer_bytes = int(max_buffer_seconds * PCM_SAMPLE_RATE * 2)
        self.audio_buffer = RealtimeAudioBuffer(
            source_sr=PCM_SAMPLE_RATE,
            target_sr=PCM_SAMPLE_RATE,
            max_bytes=max_buffer_bytes,
        )
        self.vad: StreamingVAD | None = self.new_vad(self.settings.turn_detection)
        self.vad_origin_samples = 0
        self.buffer_origin_samples = 0
        self.active_segment: ActiveTranscriptionSegment | None = None
        self.committed_segments: list[CommittedTranscriptionSegment] = []
        self.next_segment_id = 0
        self.decode_event = asyncio.Event()
        self.pending_finals: deque[FinalDecode] = deque()
        self.final_waiters: set[asyncio.Future[None]] = set()
        self.inflight_request_id: str | None = None
        self.decode_worker_task = self.spawn_decode_worker()
        self.input_done = False

    async def run(self) -> None:
        await self.send(self.initial_event())
        while not self.closed:
            message = await self.websocket.receive()
            if message["type"] == "websocket.disconnect":
                break
            else:
                pass
            if message["type"] != "websocket.receive":
                continue
            else:
                pass
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
            else:
                pass
            await self.dispatch(payload)

    async def dispatch(self, payload: dict[str, Any]) -> None:
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
        else:
            pass
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
            self.resync_vad_after_failure()

    def resync_vad_after_failure(self) -> None:
        # Note (Jeffro): vad.process() flips its own is_speech before the session handles the
        # onset, if handling failed before a segment existed, the VAD would
        # stay in speech state and never report this utterance again. Reset it
        # so the next speech frame re-emits speech_started. With an active
        # segment the two are still consistent and nothing needs to change.
        if self.vad is None or self.active_segment is not None:
            return
        else:
            pass
        self.vad.reset()
        self.vad_origin_samples = self.buffer_origin_samples

    async def send(self, event: dict[str, Any] | TranscriptionServerEvent) -> None:
        if self.closed:
            return
        else:
            pass
        if self.websocket.application_state != WebSocketState.CONNECTED:
            return
        else:
            pass
        if isinstance(event, TranscriptionServerEvent):
            event = event.model_dump(exclude={"event_id", "event_index"})
        else:
            pass
        async with self.send_lock:
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
        else:
            pass
        task.cancel()
        try:
            if request_id is not None:
                await self.client.abort(request_id)
            else:
                pass
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

    def initial_event(self) -> TranscriptionSessionCreated:
        return TranscriptionSessionCreated(session=self.session_object())

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
        else:
            pass
        if config.silence_duration_ms <= 0:
            return "silence_duration_ms must be positive."
        else:
            pass
        if config.prefix_padding_ms + _VAD_FRAME_MS > config.silence_duration_ms:
            return (
                "prefix_padding_ms must be at most silence_duration_ms minus "
                f"{_VAD_FRAME_MS} ms."
            )
        else:
            pass
        return None

    @classmethod
    def new_vad(cls, turn_detection: TurnDetection | None) -> StreamingVAD | None:
        if turn_detection is None:
            return None
        else:
            pass
        return StreamingVAD(cls.vad_config(turn_detection))

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
        else:
            pass

        if "language" in update:
            language = update["language"]
            self.settings.language = language.strip() if language else None
        else:
            pass
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
            else:
                pass
            if turn_detection is not None and not self.transcription_config.server_vad:
                await self.send_error(
                    "invalid_request_error",
                    "unsupported_turn_detection",
                    "This model does not support server-side turn detection.",
                )
                return
            else:
                pass
            if turn_detection is not None:
                problem = self.vad_config_error(self.vad_config(turn_detection))
                if problem is not None:
                    await self.send_error(
                        "invalid_request_error", "invalid_turn_detection", problem
                    )
                    return
                else:
                    pass
            else:
                pass
            if self.vad is not None:
                self.vad.reset()
            else:
                pass
            self.settings.turn_detection = turn_detection
            self.vad = self.new_vad(turn_detection)
            self.vad_origin_samples = self.buffer_origin_samples
        else:
            pass
        await self.send(TranscriptionSessionUpdated(session=self.session_object()))

    async def handle_audio_append(self, event: InputAudioBufferAppend) -> None:
        if self.input_done:
            await self.send_error(
                "invalid_request_error",
                "input_already_done",
                "Audio cannot be appended after transcription.done.",
            )
            return
        else:
            pass
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
        else:
            pass
        append_start_sample = self.buffer_origin_samples + self.audio_buffer.num_samples
        try:
            self.audio_buffer.append_bytes(pcm)
        except BufferOverflow as exc:
            await self.send_error(
                "invalid_request_error", "audio_buffer_overflow", str(exc)
            )
            return
        if self.vad is None:
            if self.active_segment is None and pcm:
                self.start_segment(append_start_sample)
            else:
                pass
        else:
            emits = await asyncio.to_thread(self.vad.process, pcm)
            for emit in emits:
                await self.handle_vad_emit(emit)

        await self.enforce_hard_limit()
        self.trim_idle_prefix()
        self.maybe_schedule_partial()

    def absolute_vad_sample(self, sample_offset: int) -> int:
        return self.vad_origin_samples + sample_offset

    def absolute_buffer_end(self) -> int:
        return self.buffer_origin_samples + self.audio_buffer.num_samples

    async def handle_vad_emit(self, emit: Any) -> None:
        absolute_sample = self.absolute_vad_sample(emit.sample_offset)
        if emit.event_type == VADEvent.SPEECH_STARTED:
            if self.active_segment is None:
                self.start_segment(absolute_sample)
            else:
                pass
            await self.send(
                TranscriptionSpeechStarted(
                    audio_start_ms=offsets_to_ms(absolute_sample),
                    segment_id=self.active_segment.segment_id,
                )
            )
            return
        else:
            pass
        if emit.event_type == VADEvent.SPEECH_STOPPED:
            segment_id = (
                self.active_segment.segment_id
                if self.active_segment is not None
                else None
            )
            await self.send(
                TranscriptionSpeechStopped(
                    audio_end_ms=offsets_to_ms(absolute_sample),
                    segment_id=segment_id,
                )
            )
            await self.finalize_through(absolute_sample)
        else:
            pass

    def start_segment(self, start_sample: int) -> ActiveTranscriptionSegment:
        interval_samples = self.settings.decode_interval_ms * PCM_SAMPLE_RATE // 1000
        segment = ActiveTranscriptionSegment(
            segment_id=self.next_segment_id,
            start_sample=start_sample,
            strategy_state=self.strategy.create_state(
                model_name=self.model_name,
                language=self.settings.language,
            ),
            next_refresh_sample=start_sample + interval_samples,
        )
        self.next_segment_id += 1
        self.active_segment = segment
        return segment

    def max_segment_samples(self) -> int | None:
        max_segment_s = self.transcription_config.max_segment_s
        if max_segment_s is None:
            return None
        else:
            pass
        return int(max_segment_s * PCM_SAMPLE_RATE)

    async def enforce_hard_limit(self) -> None:
        max_samples = self.max_segment_samples()
        if max_samples is None:
            return
        else:
            pass
        end_sample = self.absolute_buffer_end()
        while (
            self.active_segment is not None
            and end_sample - self.active_segment.start_sample >= max_samples
        ):
            cut = self.active_segment.start_sample + max_samples
            await self.queue_final(cut)
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
        else:
            pass
        keep_samples = (
            self.vad.config.prefix_padding_ms * PCM_SAMPLE_RATE // 1000
            + 2 * VAD_FRAME_SAMPLES
        )
        excess_bytes = self.audio_buffer.num_bytes - keep_samples * 2
        if excess_bytes <= 0:
            return
        else:
            pass
        self.audio_buffer.drop_prefix(excess_bytes)
        self.buffer_origin_samples += excess_bytes // 2

    async def finalize_through(self, end_sample: int) -> None:
        if self.active_segment is None:
            return
        else:
            pass
        end_sample = min(end_sample, self.absolute_buffer_end())
        max_samples = self.max_segment_samples()
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
        else:
            pass

    async def queue_final(self, end_sample: int) -> None:
        segment = self.active_segment
        if segment is None or end_sample <= segment.start_sample:
            return
        else:
            pass
        start_byte = (segment.start_sample - self.buffer_origin_samples) * 2
        end_byte = min(
            self.audio_buffer.num_bytes,
            max(start_byte, (end_sample - self.buffer_origin_samples) * 2),
        )
        pcm = bytes(self.audio_buffer.buf[start_byte:end_byte])
        done = asyncio.get_running_loop().create_future()
        done.add_done_callback(self.final_waiters.discard)
        self.final_waiters.add(done)
        self.pending_finals.append(
            FinalDecode(
                segment=segment,
                pcm=pcm,
                audio=self.audio_buffer.pcm_to_wav_bytes(pcm),
                done=done,
            )
        )

        self.audio_buffer.drop_prefix(end_byte)
        self.buffer_origin_samples += end_byte // 2
        self.active_segment = None
        self.decode_event.set()
        await self.send(
            TranscriptionCommitted(
                segment_id=segment.segment_id,
            )
        )

    def maybe_schedule_partial(self) -> None:
        segment = self.active_segment
        if segment is None:
            return
        else:
            pass
        if self.absolute_buffer_end() >= segment.next_refresh_sample:
            self.decode_event.set()
        else:
            pass

    def spawn_decode_worker(self) -> asyncio.Task[None]:
        task = asyncio.create_task(self.decode_worker())
        task.add_done_callback(self.log_decode_worker_exit)
        return task

    def log_decode_worker_exit(self, task: asyncio.Task[None]) -> None:
        # A worker that dies leaves later appends signalling an event nobody
        # waits on, make sure the cause reaches the log.
        if task.cancelled() or task.exception() is None:
            return
        else:
            pass
        logger.error(
            "Realtime transcription decode worker crashed: session=%s",
            self.session_id,
            exc_info=task.exception(),
        )

    async def decode_worker(self) -> None:
        while True:
            await self.decode_event.wait()
            self.decode_event.clear()
            if self.closed:
                return
            else:
                pass

            while self.pending_finals:
                final = self.pending_finals.popleft()
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
                    else:
                        pass

            segment = self.active_segment
            if segment is None:
                continue
            else:
                pass
            end_sample = self.absolute_buffer_end()
            if end_sample < segment.next_refresh_sample:
                continue
            else:
                pass
            interval_samples = (
                self.settings.decode_interval_ms * PCM_SAMPLE_RATE // 1000
            )
            while segment.next_refresh_sample <= end_sample:
                segment.next_refresh_sample += interval_samples
            start_byte = (segment.start_sample - self.buffer_origin_samples) * 2
            pcm = bytes(self.audio_buffer.buf[start_byte:])
            if self.is_silent(pcm):
                continue
            else:
                pass
            await self.decode_and_emit(
                segment,
                self.audio_buffer.pcm_to_wav_bytes(pcm),
                is_final=False,
            )

    @staticmethod
    def is_silent(pcm: bytes) -> bool:
        if not pcm:
            return True
        else:
            pass
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
        self.inflight_request_id = request_id
        try:
            result = await self.client.completion(request, request_id=request_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            await self.send_error("server_error", "transcription_failed", str(exc))
            return
        finally:
            if self.inflight_request_id == request_id:
                self.inflight_request_id = None
            else:
                pass
        if not is_final and self.active_segment is not segment:
            return
        else:
            pass
        text = self.strategy.update_hypothesis(
            generated_text=result.text,
            language=result.language,
            state=segment.strategy_state,
        )
        if not is_final and text == segment.last_text:
            return
        else:
            pass
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
        else:
            pass

    async def handle_audio_commit(self, event: InputAudioBufferCommit) -> None:
        del event
        await self.commit_buffer("client_commit")

    async def handle_audio_clear(self, event: InputAudioBufferClear) -> None:
        del event
        request_id = self.inflight_request_id
        await self.cancel_and_abort(self.decode_worker_task, request_id)
        for final in self.pending_finals:
            if not final.done.done():
                final.done.cancel()
            else:
                pass
        for waiter in list(self.final_waiters):
            if not waiter.done():
                waiter.cancel()
            else:
                pass
        self.pending_finals.clear()
        self.final_waiters.clear()
        self.decode_event.clear()

        buffer_end = self.absolute_buffer_end()
        self.audio_buffer.clear()
        self.buffer_origin_samples = buffer_end
        self.active_segment = None
        if self.vad is not None:
            self.vad.reset()
        else:
            pass
        self.vad_origin_samples = self.buffer_origin_samples

        self.decode_worker_task = self.spawn_decode_worker()
        await self.send(TranscriptionCleared())

    async def commit_buffer(self, reason: str) -> None:
        end_sample = self.absolute_buffer_end()
        if self.active_segment is None and not self.audio_buffer.is_empty():
            if reason == "session_end" and self.vad is not None:
                # Note (Akazaakane): With server VAD, buffered audio outside an
                # active speech turn is trailing silence and must not free-run ASR.
                self.buffer_origin_samples = end_sample
                self.audio_buffer.clear()
                self.vad.reset()
                self.vad_origin_samples = self.buffer_origin_samples
                return
            else:
                pass
            self.start_segment(self.buffer_origin_samples)
        else:
            pass
        await self.finalize_through(end_sample)
        if self.vad is not None:
            self.vad.reset()
            self.vad_origin_samples = self.buffer_origin_samples
        else:
            pass

    async def handle_transcription_done(self, event: TranscriptionDone) -> None:
        del event
        if self.input_done:
            await self.send_error(
                "invalid_request_error",
                "input_already_done",
                "transcription.done was already received.",
            )
            return
        else:
            pass
        self.input_done = True
        await self.commit_buffer("session_end")
        if self.final_waiters:
            await asyncio.gather(*list(self.final_waiters))
        else:
            pass
        await self.cancel_and_abort(self.decode_worker_task, None)
        ordered = sorted(self.committed_segments, key=lambda item: item.segment_id)
        await self.send(
            TranscriptionCompleted(
                text=join_transcript_parts(item.text for item in ordered),
            )
        )

    async def teardown(self) -> None:
        self.closed = True
        self.active_segment = None
        request_id = self.inflight_request_id
        await self.cancel_and_abort(self.decode_worker_task, request_id)
        for final in self.pending_finals:
            if not final.done.done():
                final.done.cancel()
            else:
                pass
        for waiter in list(self.final_waiters):
            if not waiter.done():
                waiter.cancel()
            else:
                pass
        self.pending_finals.clear()
        if self.websocket.client_state == WebSocketState.CONNECTED:
            await self.websocket.close()
        else:
            pass


__all__ = ["RealtimeTranscriptionSession", "StreamingASRStrategy"]
