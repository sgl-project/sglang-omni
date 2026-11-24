# SPDX-License-Identifier: Apache-2.0
"""One session owner for VAD boundaries and semantic endpoint decisions."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from typing import Literal

from sglang_omni.audio.semantic_vad import (
    SemanticEOUModel,
    SemanticVADConfig,
    end_of_turn_probability,
)
from sglang_omni.audio.smart_turn import SmartTurnEOU
from sglang_omni.audio.vad import VADConfig
from sglang_omni.models.turn_detection.node import SMART_TURN_MODEL_REPO
from sglang_omni.models.vad.session import (
    DetectorFactory,
    DetectorSession,
    SileroSpeechDetector,
    boundary_chunk,
    turn_detection_config,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    SEMANTIC_VAD,
    SERVER_VAD,
    SESSION_CONTROL,
    SESSION_INPUT_SAMPLE_RATE_HZ,
    control_type,
)
from sglang_omni.scheduling.session import (
    DEFAULT_MAX_CONCURRENCY,
    NODE_STAGE_MAX_OPEN_SESSIONS,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)

TurnDetectionMode = Literal["server_vad", "semantic_vad"] | None


def turn_detection_policy(
    mode: TurnDetectionMode,
    policy: SemanticVADConfig | None,
    eagerness: str,
    settings: Mapping[str, object],
) -> tuple[TurnDetectionMode, SemanticVADConfig | None]:
    """Resolve a session's semantic, server, or manual turn mode."""
    if "turn_detection" not in settings:
        return mode, policy
    else:
        turn_detection = settings["turn_detection"]
    if turn_detection is None:
        return None, None
    elif not isinstance(turn_detection, Mapping):
        raise ValueError("turn_detection must be an object")
    elif turn_detection.get("type") == SERVER_VAD:
        return SERVER_VAD, None
    elif turn_detection.get("type") == SEMANTIC_VAD:
        requested = turn_detection.get("eagerness")
        selected_eagerness = (
            eagerness if requested in (None, "auto") else str(requested)
        )
        return SEMANTIC_VAD, SemanticVADConfig.from_eagerness(selected_eagerness)
    else:
        return mode, policy


@dataclass(kw_only=True)
class TurnSession:
    detector_session: DetectorSession
    mode: TurnDetectionMode
    policy: SemanticVADConfig | None
    is_open: bool = False
    pause_start_ms: float | None = None
    pause_probability: float | None = None
    required_silence_ms: int = 0
    turn_start_ms: float = 0.0
    audio_received_end_ms: float = 0.0
    audio_received_sample_count: int = 0
    audio_history: deque[TimedChunk] = field(default_factory=deque)

    def clear_pause(self) -> None:
        self.pause_start_ms = None
        self.pause_probability = None
        self.required_silence_ms = 0


class TurnDetectionHooks(SessionHooks):
    def __init__(
        self,
        eou_model: SemanticEOUModel,
        eagerness: str,
        vad_config: VADConfig | None = None,
        new_detector: DetectorFactory | None = None,
    ) -> None:
        self.eou_model = eou_model
        self.eagerness = eagerness
        self.vad_config = vad_config or VADConfig()
        self.new_detector = new_detector or SileroSpeechDetector
        self.sessions: dict[SessionIdentity, TurnSession] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        mode, policy = turn_detection_policy(
            SEMANTIC_VAD,
            SemanticVADConfig.from_eagerness(self.eagerness),
            self.eagerness,
            request.params,
        )
        detector_config = turn_detection_config(self.vad_config, request.params)
        self.sessions[session_identity] = TurnSession(
            detector_session=DetectorSession(
                detector=self.new_detector(detector_config),
                is_enabled=mode is not None,
            ),
            mode=mode,
            policy=policy,
        )

    def close(self, session_identity: SessionIdentity) -> None:
        del self.sessions[session_identity]

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        session = self.sessions[context.session_identity]
        chunk_control_type = control_type(chunk)
        boundaries: list[tuple[bool, float]] | None = None
        if chunk.modality == AUDIO:
            boundaries = self.append_audio(session, chunk)
            for started, boundary_time_ms in boundaries:
                self.append_boundary(session, started, boundary_time_ms, context)
            if (
                chunk.eos
                and session.mode is not None
                and session.detector_session.detector.is_speech
            ):
                self.append_boundary(
                    session,
                    False,
                    session.detector_session.boundary_ms(
                        session.detector_session.detector.last_speech_offset
                    ),
                    context,
                )
            else:
                pass
        elif chunk_control_type == COMMIT:
            self.reset_after_commit(session)
        elif chunk_control_type == CONFIGURE:
            self.configure(session, chunk.payload)
        elif chunk.format == SESSION_CONTROL:
            pass
        else:
            raise ValueError("expected pcm16 audio or a session control")

        if chunk.eos and session.mode is not None and session.is_open:
            self.end_turn(session, context)
        elif (
            boundaries is not None
            and session.mode in (SERVER_VAD, SEMANTIC_VAD)
            and session.is_open
            and session.pause_start_ms is not None
            and session.detector_session.completed_frame_end_ms is not None
        ):
            decision_horizon_ms = min(
                session.audio_received_end_ms,
                session.detector_session.completed_frame_end_ms,
            )
            if (
                decision_horizon_ms - session.pause_start_ms
                >= session.required_silence_ms
            ):
                self.end_turn(session, context)
            else:
                pass
        else:
            pass
        payload.data = {"accepted": True}
        return payload

    def append_audio(
        self, session: TurnSession, chunk: TimedChunk
    ) -> list[tuple[bool, float]]:
        expected_start_sample = (
            session.audio_received_sample_count
            if session.detector_session.is_enabled
            and session.detector_session.origin_ms is not None
            else None
        )
        boundaries = session.detector_session.process_audio(
            chunk,
            expected_start_sample=expected_start_sample,
            require_duration_match=True,
        )
        if chunk.payload:
            assert isinstance(chunk.payload, bytes)
            received_sample_count = len(chunk.payload) // 2
            if session.detector_session.is_enabled:
                assert session.detector_session.origin_ms is not None
                session.audio_received_sample_count += received_sample_count
                session.audio_received_end_ms = (
                    session.detector_session.origin_ms
                    + session.audio_received_sample_count
                    * 1000
                    / SESSION_INPUT_SAMPLE_RATE_HZ
                )
            else:
                session.audio_received_end_ms = chunk.t_start_ms + chunk.duration_ms
            self.append_audio_history(session, chunk)
        else:
            pass
        return boundaries

    def append_audio_history(self, session: TurnSession, chunk: TimedChunk) -> None:
        session.audio_history.append(chunk)
        history_start_ms = (
            session.audio_received_end_ms
            - SemanticVADConfig.max_utterance_seconds * 1000
        )
        while (
            session.audio_history[0].t_start_ms + session.audio_history[0].duration_ms
            <= history_start_ms
        ):
            session.audio_history.popleft()

    def append_boundary(
        self, session: TurnSession, started: bool, t_ms: float, context: SessionContext
    ) -> None:
        if session.mode == SERVER_VAD:
            if started and not session.is_open:
                session.is_open = True
                session.turn_start_ms = t_ms
                session.clear_pause()
                context.emit(boundary_chunk(True, t_ms))
            elif started:
                session.clear_pause()
            else:
                session.is_open = False
                session.clear_pause()
                context.emit(boundary_chunk(False, t_ms))
        elif session.mode == SEMANTIC_VAD and started and not session.is_open:
            session.is_open = True
            session.turn_start_ms = t_ms
            session.clear_pause()
            context.emit(boundary_chunk(True, t_ms))
        elif session.mode == SEMANTIC_VAD and started:
            session.clear_pause()
        elif session.mode == SEMANTIC_VAD and session.is_open:
            session.pause_start_ms = t_ms
            utterance = b"".join(
                audio.payload
                for audio in session.audio_history
                if audio.t_start_ms + audio.duration_ms > session.turn_start_ms
            )
            assert session.policy is not None
            session.pause_probability = end_of_turn_probability(
                self.eou_model, utterance, session.policy.max_utterance_seconds
            )
            session.required_silence_ms = session.policy.required_silence_ms(
                session.pause_probability
            )
        else:
            pass

    def configure(self, session: TurnSession, settings: Mapping[str, object]) -> None:
        if "turn_detection" not in settings:
            return
        else:
            pass
        mode, policy = turn_detection_policy(
            session.mode,
            session.policy,
            self.eagerness,
            settings,
        )
        detector_config = turn_detection_config(
            replace(
                session.detector_session.detector.config,
                silence_duration_ms=self.vad_config.silence_duration_ms,
            ),
            settings,
        )
        if mode is None:
            session.detector_session.reset()
            session.detector_session.is_enabled = False
            session.audio_received_sample_count = 0
            session.mode = None
            session.policy = None
            session.clear_pause()
        else:
            session.detector_session.detector.config = detector_config
            session.detector_session.is_enabled = True
            session.mode = mode
            session.policy = policy
            if session.pause_start_ms is None:
                session.clear_pause()
            elif mode == SEMANTIC_VAD:
                assert policy is not None
                session.required_silence_ms = policy.required_silence_ms(
                    session.pause_probability
                )
            elif mode == SERVER_VAD:
                session.required_silence_ms = detector_config.silence_duration_ms
            else:
                pass

    def reset_after_commit(self, session: TurnSession) -> None:
        session.detector_session.reset()
        session.audio_received_sample_count = 0
        session.audio_received_end_ms = 0.0
        session.is_open = False
        session.clear_pause()
        session.audio_history.clear()

    def end_turn(self, session: TurnSession, context: SessionContext) -> None:
        boundary_time_ms = (
            session.pause_start_ms
            if session.pause_start_ms is not None
            else session.audio_received_end_ms
        )
        session.is_open = False
        session.clear_pause()
        context.emit(boundary_chunk(False, boundary_time_ms))


def create_scheduler(
    *,
    model_path: str = SMART_TURN_MODEL_REPO,
    eagerness: str = "medium",
    threshold: float = VADConfig.threshold,
    prefix_padding_ms: int = VADConfig.prefix_padding_ms,
    silence_duration_ms: int = VADConfig.silence_duration_ms,
    background_margin_db: float | None = VADConfig.background_margin_db,
    max_open_sessions: int = NODE_STAGE_MAX_OPEN_SESSIONS,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> SessionScheduler:
    vad_config = VADConfig(
        threshold=threshold,
        prefix_padding_ms=prefix_padding_ms,
        silence_duration_ms=silence_duration_ms,
        background_margin_db=background_margin_db,
    )
    return SessionScheduler(
        TurnDetectionHooks(SmartTurnEOU.load(model_path), eagerness, vad_config),
        max_open_sessions=max_open_sessions,
        max_concurrency=max_concurrency,
    )
