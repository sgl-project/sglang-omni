# SPDX-License-Identifier: Apache-2.0
"""The VAD stage: one silero detector per Session, speech boundaries on the input timeline."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from typing import Protocol

from sglang_omni.audio.vad import StreamingVAD, VADConfig, VADEvent
from sglang_omni.models.vad.node import TURN_DETECTION_FIELDS
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    CONTROL,
    PCM16,
    SESSION_CONTROL,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    control_type,
)
from sglang_omni.scheduling.session import (
    DEFAULT_MAX_CONCURRENCY,
    NODE_STAGE_MAX_OPEN_SESSIONS,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)


class SpeechDetector(Protocol):
    """Frame-level speech detection over 16 kHz pcm16; offsets are samples since the last reset."""

    config: VADConfig
    is_speech: bool
    last_speech_offset: int

    def boundaries(self, pcm: bytes) -> list[tuple[bool, int]]: ...

    def reset(self) -> None: ...


class SileroSpeechDetector(StreamingVAD):
    def boundaries(self, pcm: bytes) -> list[tuple[bool, int]]:
        return [
            (emit.event_type == VADEvent.SPEECH_STARTED, emit.sample_offset)
            for emit in self.process(pcm)
        ]


DetectorFactory = Callable[[VADConfig], SpeechDetector]


def turn_detection_config(
    config: VADConfig, settings: Mapping[str, object]
) -> VADConfig:
    """`config` with the `turn_detection` fields `settings` sets."""
    turn_detection = settings.get("turn_detection")
    if turn_detection is None:
        return config
    elif not isinstance(turn_detection, Mapping):
        raise ValueError("turn_detection must be an object")
    else:
        return replace(
            config,
            **{
                field: turn_detection[field]
                for field in TURN_DETECTION_FIELDS
                if turn_detection.get(field) is not None
            },
        )


def boundary_chunk(started: bool, t_ms: float) -> TimedChunk:
    return TimedChunk(
        CONTROL,
        t_ms,
        0,
        0,
        {"type": SPEECH_STARTED if started else SPEECH_STOPPED, "t_ms": t_ms},
        format=SESSION_CONTROL,
    )


@dataclass(kw_only=True)
class DetectorSession:
    detector: SpeechDetector
    is_enabled: bool = True
    # Note (Junnan Li): Input time of the first audio sample since the last reset; None until audio arrives.
    origin_ms: float | None = None

    def boundary_ms(self, offset: int) -> float:
        assert self.origin_ms is not None
        return self.origin_ms + offset * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ


class VADHooks(SessionHooks):
    def __init__(self, config: VADConfig, new_detector: DetectorFactory) -> None:
        self.config = config
        self.new_detector = new_detector
        self.sessions: dict[SessionIdentity, DetectorSession] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        config = turn_detection_config(self.config, request.params)
        self.sessions[session_identity] = DetectorSession(
            detector=self.new_detector(config),
            is_enabled=not (
                "turn_detection" in request.params
                and request.params["turn_detection"] is None
            ),
        )

    def close(self, session_identity: SessionIdentity) -> None:
        del self.sessions[session_identity]

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        session = self.sessions[context.session_identity]
        chunk_control_type = control_type(chunk)
        if chunk.modality == AUDIO:
            self.append_audio(session, chunk, context)
        elif chunk_control_type == COMMIT:
            session.detector.reset()
            session.origin_ms = None
        elif chunk_control_type == CONFIGURE:
            session.detector.config = turn_detection_config(
                session.detector.config, chunk.payload
            )
            if "turn_detection" not in chunk.payload:
                pass
            elif chunk.payload["turn_detection"] is None:
                session.is_enabled = False
                session.detector.reset()
                session.origin_ms = None
            else:
                session.is_enabled = True
        elif chunk.format == SESSION_CONTROL:
            pass
        else:
            raise ValueError("expected pcm16 audio or a session control")
        if chunk.eos and session.detector.is_speech:
            context.emit(
                boundary_chunk(
                    False, session.boundary_ms(session.detector.last_speech_offset)
                )
            )
        else:
            pass
        payload.data = {"accepted": True}
        return payload

    def append_audio(
        self, session: DetectorSession, chunk: TimedChunk, context: SessionContext
    ) -> None:
        if chunk.format != PCM16 or not isinstance(chunk.payload, bytes):
            raise ValueError("audio must be 16 kHz mono pcm16 bytes")
        elif not chunk.payload or not session.is_enabled:
            return
        else:
            pass
        if session.origin_ms is None:
            session.origin_ms = chunk.t_start_ms
        else:
            pass
        for started, offset in session.detector.boundaries(chunk.payload):
            context.emit(boundary_chunk(started, session.boundary_ms(offset)))


def create_scheduler(
    *,
    threshold: float = VADConfig.threshold,
    prefix_padding_ms: int = VADConfig.prefix_padding_ms,
    silence_duration_ms: int = VADConfig.silence_duration_ms,
    max_open_sessions: int = NODE_STAGE_MAX_OPEN_SESSIONS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> SessionScheduler:
    config = VADConfig(
        threshold=threshold,
        prefix_padding_ms=prefix_padding_ms,
        silence_duration_ms=silence_duration_ms,
    )
    return SessionScheduler(
        VADHooks(config, SileroSpeechDetector),
        max_open_sessions=max_open_sessions,
        max_concurrency=max_concurrency,
    )
