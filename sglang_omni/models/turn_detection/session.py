# SPDX-License-Identifier: Apache-2.0
"""The turn detection stage: the VAD's speech boundaries become turn boundaries, a pause inside a turn scored by an end-of-turn model."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field

from sglang_omni.audio.semantic_vad import (
    SemanticEOUModel,
    SemanticVADConfig,
    end_of_turn_probability,
)
from sglang_omni.audio.smart_turn import SmartTurnEOU
from sglang_omni.models.turn_detection.node import SMART_TURN_MODEL_REPO
from sglang_omni.models.vad.session import boundary_chunk
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    PCM16,
    SEMANTIC_VAD,
    SERVER_VAD,
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


def turn_detection_policy(
    policy: SemanticVADConfig | None, eagerness: str, settings: Mapping[str, object]
) -> SemanticVADConfig | None:
    """The policy `settings["turn_detection"]` selects: a preset for `semantic_vad`, None for `server_vad`, which passes the VAD's boundaries through; `policy` when it names neither."""
    turn_detection = settings.get("turn_detection")
    if turn_detection is None:
        return policy
    elif not isinstance(turn_detection, Mapping):
        raise ValueError("turn_detection must be an object")
    elif turn_detection.get("type") == SERVER_VAD:
        return None
    elif turn_detection.get("type") == SEMANTIC_VAD:
        requested = turn_detection.get("eagerness")
        return SemanticVADConfig.from_eagerness(
            eagerness if requested in (None, "auto") else str(requested)
        )
    else:
        return policy


@dataclass(kw_only=True)
class TurnSession:
    policy: SemanticVADConfig | None
    is_open: bool = False
    # Note (Junnan Li): The t_ms of the VAD's speech.stopped the open turn waits in, and the silence after it that ends the turn.
    pause_start_ms: float | None = None
    required_silence_ms: int = 0
    turn_start_ms: float = 0.0
    audio_end_ms: float = 0.0
    audio: deque[TimedChunk] = field(default_factory=deque)


class TurnDetectionHooks(SessionHooks):
    def __init__(self, eou_model: SemanticEOUModel, eagerness: str) -> None:
        self.eou_model = eou_model
        self.eagerness = eagerness
        self.sessions: dict[SessionIdentity, TurnSession] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.sessions[session_identity] = TurnSession(
            policy=turn_detection_policy(
                SemanticVADConfig.from_eagerness(self.eagerness),
                self.eagerness,
                request.params,
            )
        )

    def close(self, session_identity: SessionIdentity) -> None:
        del self.sessions[session_identity]

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        session = self.sessions[context.session_identity]
        chunk_control_type = control_type(chunk)
        if chunk.modality == AUDIO:
            self.append_audio(session, chunk)
        elif chunk_control_type in (SPEECH_STARTED, SPEECH_STOPPED):
            self.append_boundary(session, chunk, context)
        elif chunk_control_type == COMMIT:
            session.is_open = False
            session.pause_start_ms = None
        elif chunk_control_type == CONFIGURE:
            session.policy = turn_detection_policy(
                session.policy, self.eagerness, chunk.payload
            )
            if session.policy is None:
                session.is_open = False
                session.pause_start_ms = None
            else:
                pass
        elif chunk.format == SESSION_CONTROL:
            pass
        else:
            raise ValueError("expected pcm16 audio or a session control")
        if chunk.eos and session.is_open:
            self.end_turn(session, context)
        elif (
            session.is_open
            and session.pause_start_ms is not None
            and session.audio_end_ms - session.pause_start_ms
            >= session.required_silence_ms
        ):
            self.end_turn(session, context)
        else:
            pass
        payload.data = {"accepted": True}
        return payload

    def append_audio(self, session: TurnSession, chunk: TimedChunk) -> None:
        if chunk.format != PCM16 or not isinstance(chunk.payload, bytes):
            raise ValueError("audio must be 16 kHz mono pcm16 bytes")
        elif not chunk.payload:
            return
        else:
            pass
        session.audio_end_ms = max(
            session.audio_end_ms, chunk.t_start_ms + chunk.duration_ms
        )
        session.audio.append(chunk)
        horizon_ms = (
            session.audio_end_ms - SemanticVADConfig.max_utterance_seconds * 1000
        )
        while session.audio[0].t_start_ms + session.audio[0].duration_ms <= horizon_ms:
            session.audio.popleft()

    def append_boundary(
        self, session: TurnSession, chunk: TimedChunk, context: SessionContext
    ) -> None:
        t_ms = float(chunk.payload["t_ms"])
        started = control_type(chunk) == SPEECH_STARTED
        if session.policy is None:
            context.emit(boundary_chunk(started, t_ms))
        elif started and not session.is_open:
            session.is_open = True
            session.turn_start_ms = t_ms
            context.emit(boundary_chunk(True, t_ms))
        elif started:
            session.pause_start_ms = None
        elif session.is_open:
            session.pause_start_ms = t_ms
            utterance = b"".join(
                audio.payload
                for audio in session.audio
                if audio.t_start_ms + audio.duration_ms > session.turn_start_ms
            )
            session.required_silence_ms = session.policy.required_silence_ms(
                end_of_turn_probability(
                    self.eou_model, utterance, session.policy.max_utterance_seconds
                )
            )
        else:
            pass

    def end_turn(self, session: TurnSession, context: SessionContext) -> None:
        t_ms = (
            session.pause_start_ms
            if session.pause_start_ms is not None
            else session.audio_end_ms
        )
        session.is_open = False
        session.pause_start_ms = None
        context.emit(boundary_chunk(False, t_ms))


def create_scheduler(
    *,
    model_path: str = SMART_TURN_MODEL_REPO,
    eagerness: str = "medium",
    max_open_sessions: int = NODE_STAGE_MAX_OPEN_SESSIONS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> SessionScheduler:
    return SessionScheduler(
        TurnDetectionHooks(SmartTurnEOU.load(model_path), eagerness),
        max_open_sessions=max_open_sessions,
        max_concurrency=max_concurrency,
    )
