# SPDX-License-Identifier: Apache-2.0
"""The VAD stage: one silero detector per Session, speech boundaries on the input timeline."""

from __future__ import annotations

from dataclasses import replace

from sglang_omni.audio.vad import (
    SileroSpeechDetector,
    SpeechDetectorFactory,
    VADConfig,
    VADSessionState,
    turn_detection_vad_config,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    CONTROL,
    SESSION_CONTROL,
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


def boundary_chunk(started: bool, t_ms: float) -> TimedChunk:
    return TimedChunk(
        CONTROL,
        t_ms,
        0,
        0,
        {"type": SPEECH_STARTED if started else SPEECH_STOPPED, "t_ms": t_ms},
        format=SESSION_CONTROL,
    )


class VADHooks(SessionHooks):
    def __init__(self, config: VADConfig, new_detector: SpeechDetectorFactory) -> None:
        self.config = config
        self.new_detector = new_detector
        self.sessions: dict[SessionIdentity, VADSessionState] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        config = turn_detection_vad_config(self.config, request.params)
        self.sessions[session_identity] = VADSessionState(
            detector=self.new_detector(config),
            is_enabled=not (
                "turn_detection" in request.params
                and request.params["turn_detection"] is None
            ),
        )

    def close(self, session_identity: SessionIdentity) -> None:
        session = self.sessions.pop(session_identity)
        session.reset()

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        session = self.sessions[context.session_identity]
        chunk_control_type = control_type(chunk)
        if chunk.modality == AUDIO:
            self.append_audio(session, chunk, context)
        elif chunk_control_type == COMMIT:
            session.reset()
        elif chunk_control_type == CONFIGURE:
            if "turn_detection" not in chunk.payload:
                pass
            elif chunk.payload["turn_detection"] is None:
                session.reset()
                session.is_enabled = False
            else:
                session.detector.config = turn_detection_vad_config(
                    replace(
                        session.detector.config,
                        silence_duration_ms=self.config.silence_duration_ms,
                    ),
                    chunk.payload,
                )
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
        self, session: VADSessionState, chunk: TimedChunk, context: SessionContext
    ) -> None:
        result = session.append_audio(chunk)
        for boundary in result.boundaries:
            context.emit(boundary_chunk(boundary.started, boundary.t_ms))


def create_scheduler(
    *,
    threshold: float = VADConfig.threshold,
    prefix_padding_ms: int = VADConfig.prefix_padding_ms,
    silence_duration_ms: int = VADConfig.silence_duration_ms,
    background_margin_db: float | None = VADConfig.background_margin_db,
    max_open_sessions: int = NODE_STAGE_MAX_OPEN_SESSIONS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> SessionScheduler:
    config = VADConfig(
        threshold=threshold,
        prefix_padding_ms=prefix_padding_ms,
        silence_duration_ms=silence_duration_ms,
        background_margin_db=background_margin_db,
    )
    return SessionScheduler(
        VADHooks(config, SileroSpeechDetector),
        max_open_sessions=max_open_sessions,
        max_concurrency=max_concurrency,
    )
