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
from sglang_omni.audio.vad import VAD_FRAME_SAMPLES, VADConfig
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
    CONTROL,
    INTERRUPT,
    SEGMENTATION_MODE,
    SEMANTIC_VAD,
    SERVER_VAD,
    SESSION_CONTROL,
    SESSION_CONTROL_INTERNAL,
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


def timeline_sample_at_time_ms(time_ms: float, origin_ms: float | None) -> int:
    if origin_ms is None:
        return round(time_ms * SESSION_INPUT_SAMPLE_RATE_HZ / 1000)
    else:
        return round(origin_ms * SESSION_INPUT_SAMPLE_RATE_HZ / 1000) + round(
            (time_ms - origin_ms) * SESSION_INPUT_SAMPLE_RATE_HZ / 1000
        )


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
    segmentation_mode_announced: bool = False

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
        if not session.segmentation_mode_announced:
            if session.mode is not None:
                self.emit_segmentation_mode(session, chunk, context)
            else:
                pass
            session.segmentation_mode_announced = True
        else:
            pass
        chunk_control_type = control_type(chunk)
        if chunk.modality == AUDIO:
            self.append_audio(session, chunk, context)
        elif chunk_control_type == COMMIT:
            self.reset_after_commit(session)
            context.emit(replace(chunk, format=SESSION_CONTROL_INTERNAL))
        elif chunk_control_type == CONFIGURE:
            self.configure(session, chunk.payload)
            context.emit(replace(chunk, format=SESSION_CONTROL_INTERNAL))
            self.emit_segmentation_mode(session, chunk, context)
        elif chunk_control_type == INTERRUPT:
            pass
        elif chunk.format == SESSION_CONTROL:
            context.emit(replace(chunk, format=SESSION_CONTROL_INTERNAL))
        else:
            raise ValueError("expected pcm16 audio or a session control")

        if chunk.eos and session.mode is not None and session.is_open:
            if session.detector_session.detector.is_speech:
                session.pause_start_ms = session.detector_session.boundary_ms(
                    session.detector_session.detector.last_speech_offset
                )
            else:
                pass
            context.emit(self.end_turn(session))
        else:
            pass
        if chunk.modality == AUDIO and chunk.eos:
            context.emit(
                TimedChunk(
                    AUDIO,
                    chunk.t_start_ms + chunk.duration_ms,
                    0,
                    chunk.seq,
                    b"",
                    format=chunk.format,
                    eos=True,
                    turn=chunk.turn,
                )
            )
        else:
            pass
        payload.data = {"accepted": True}
        return payload

    def emit_segmentation_mode(
        self, session: TurnSession, chunk: TimedChunk, context: SessionContext
    ) -> None:
        context.emit(
            TimedChunk(
                CONTROL,
                chunk.t_start_ms,
                0,
                chunk.seq,
                {"type": SEGMENTATION_MODE, "enabled": session.mode is not None},
                format=SESSION_CONTROL_INTERNAL,
                turn=chunk.turn,
            )
        )

    def append_audio(
        self, session: TurnSession, chunk: TimedChunk, context: SessionContext
    ) -> None:
        expected_start_sample = (
            session.audio_received_sample_count
            if session.mode is not None
            and session.detector_session.origin_ms is not None
            else None
        )
        sample_count = session.detector_session.validate_audio_chunk(
            chunk,
            expected_start_sample=expected_start_sample,
            require_duration_match=True,
        )
        if sample_count == 0:
            return
        else:
            session.audio_received_end_ms = chunk.t_start_ms + chunk.duration_ms
        if session.mode is None:
            self.append_audio_history(session, chunk)
            self.emit_audio_span([chunk], context)
        else:
            assert isinstance(chunk.payload, bytes)
            byte_offset = 0
            pending_audio: list[TimedChunk] = []
            while byte_offset < len(chunk.payload):
                partial_sample_count = (
                    session.audio_received_sample_count
                    - session.detector_session.detector.samples_consumed
                )
                slice_sample_count = min(
                    VAD_FRAME_SAMPLES - partial_sample_count,
                    sample_count - byte_offset // 2,
                )
                assert slice_sample_count > 0
                start_byte = byte_offset
                end_byte = start_byte + slice_sample_count * 2
                pcm_slice = chunk.payload[start_byte:end_byte]
                consumed_before = session.detector_session.detector.samples_consumed
                boundaries = session.detector_session.process_pcm(pcm_slice)
                session.audio_received_sample_count += slice_sample_count
                assert session.detector_session.origin_ms is not None
                audio_slice = TimedChunk(
                    AUDIO,
                    chunk.t_start_ms
                    + byte_offset // 2 * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ,
                    slice_sample_count * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ,
                    chunk.seq,
                    pcm_slice,
                    format=chunk.format,
                    turn=chunk.turn,
                )
                self.append_audio_history(session, audio_slice)
                pending_audio.append(audio_slice)
                byte_offset = end_byte
                if session.detector_session.detector.samples_consumed > consumed_before:
                    for started, boundary_time_ms in boundaries:
                        boundary = self.append_boundary(
                            session, started, boundary_time_ms
                        )
                        if boundary is not None:
                            self.emit_audio_span(pending_audio, context)
                            pending_audio.clear()
                            context.emit(boundary)
                        else:
                            pass
                    if (
                        session.mode is not None
                        and session.is_open
                        and session.pause_start_ms is not None
                    ):
                        origin_ms = session.detector_session.origin_ms
                        assert origin_ms is not None
                        completed_frame_end_sample = (
                            timeline_sample_at_time_ms(origin_ms, origin_ms)
                            + session.detector_session.detector.samples_consumed
                        )
                        pause_start_sample = timeline_sample_at_time_ms(
                            session.pause_start_ms,
                            origin_ms,
                        )
                        deadline_sample = pause_start_sample + (
                            session.required_silence_ms
                            * SESSION_INPUT_SAMPLE_RATE_HZ
                            // 1000
                        )
                        if completed_frame_end_sample >= deadline_sample:
                            self.emit_audio_span(pending_audio, context)
                            pending_audio.clear()
                            context.emit(self.end_turn(session))
                        else:
                            pass
                    else:
                        pass
                else:
                    pass
            self.emit_audio_span(pending_audio, context)

    def emit_audio_span(
        self, chunks: list[TimedChunk], context: SessionContext
    ) -> None:
        """Coalesce PCM up to the next causal control without crossing appends."""
        if not chunks:
            return
        else:
            pass
        first = chunks[0]
        if len(chunks) == 1:
            assert isinstance(first.payload, bytes)
            audio_payload = first.payload
        else:
            audio_payload = b"".join(chunk.payload for chunk in chunks)
        context.emit(
            replace(
                first,
                duration_ms=len(audio_payload)
                // 2
                * 1000
                / SESSION_INPUT_SAMPLE_RATE_HZ,
                payload=audio_payload,
                eos=False,
            )
        )

    def append_audio_history(self, session: TurnSession, chunk: TimedChunk) -> None:
        session.audio_history.append(chunk)
        assert isinstance(chunk.payload, bytes)
        history_end_sample = (
            timeline_sample_at_time_ms(
                chunk.t_start_ms, session.detector_session.origin_ms
            )
            + len(chunk.payload) // 2
        )
        history_start_sample = history_end_sample - (
            SemanticVADConfig.max_utterance_seconds * SESSION_INPUT_SAMPLE_RATE_HZ
        )
        while session.audio_history:
            first_chunk = session.audio_history[0]
            assert isinstance(first_chunk.payload, bytes)
            first_start_sample = timeline_sample_at_time_ms(
                first_chunk.t_start_ms, session.detector_session.origin_ms
            )
            first_end_sample = first_start_sample + len(first_chunk.payload) // 2
            if first_end_sample <= history_start_sample:
                session.audio_history.popleft()
            elif first_start_sample < history_start_sample:
                first_payload_offset = (history_start_sample - first_start_sample) * 2
                first_payload = first_chunk.payload[first_payload_offset:]
                origin_ms = session.detector_session.origin_ms
                if origin_ms is None:
                    first_start_ms = (
                        history_start_sample * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ
                    )
                else:
                    origin_sample = timeline_sample_at_time_ms(origin_ms, origin_ms)
                    first_start_ms = (
                        origin_ms
                        + (history_start_sample - origin_sample)
                        * 1000
                        / SESSION_INPUT_SAMPLE_RATE_HZ
                    )
                session.audio_history[0] = TimedChunk(
                    AUDIO,
                    first_start_ms,
                    len(first_payload) // 2 * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ,
                    first_chunk.seq,
                    first_payload,
                    format=first_chunk.format,
                    eos=first_chunk.eos,
                    turn=first_chunk.turn,
                )
                break
            else:
                break

    def utterance_pcm_at_frame_end(self, session: TurnSession) -> bytes:
        origin_ms = session.detector_session.origin_ms
        assert origin_ms is not None
        frame_end_sample = (
            timeline_sample_at_time_ms(origin_ms, origin_ms)
            + session.detector_session.detector.samples_consumed
        )
        window_start_sample = max(
            timeline_sample_at_time_ms(session.turn_start_ms, origin_ms),
            frame_end_sample
            - SemanticVADConfig.max_utterance_seconds * SESSION_INPUT_SAMPLE_RATE_HZ,
        )
        pcm_fragments: list[bytes] = []
        for audio_chunk in session.audio_history:
            assert isinstance(audio_chunk.payload, bytes)
            chunk_start_sample = timeline_sample_at_time_ms(
                audio_chunk.t_start_ms, origin_ms
            )
            chunk_end_sample = chunk_start_sample + len(audio_chunk.payload) // 2
            fragment_start_sample = max(window_start_sample, chunk_start_sample)
            fragment_end_sample = min(frame_end_sample, chunk_end_sample)
            if fragment_start_sample < fragment_end_sample:
                start_byte = (fragment_start_sample - chunk_start_sample) * 2
                end_byte = (fragment_end_sample - chunk_start_sample) * 2
                pcm_fragments.append(audio_chunk.payload[start_byte:end_byte])
            else:
                pass
        return b"".join(pcm_fragments)

    def append_boundary(
        self, session: TurnSession, started: bool, t_ms: float
    ) -> TimedChunk | None:
        if started:
            if not session.is_open:
                session.is_open = True
                session.turn_start_ms = t_ms
                session.clear_pause()
                return boundary_chunk(True, t_ms)
            else:
                session.clear_pause()
                return None
        elif session.mode == SERVER_VAD:
            session.is_open = False
            session.clear_pause()
            return boundary_chunk(False, t_ms)
        elif session.mode == SEMANTIC_VAD and session.is_open:
            session.pause_start_ms = t_ms
            assert session.policy is not None
            session.pause_probability = end_of_turn_probability(
                self.eou_model,
                self.utterance_pcm_at_frame_end(session),
                session.policy.max_utterance_seconds,
            )
            session.required_silence_ms = session.policy.required_silence_ms(
                session.pause_probability
            )
            return None
        else:
            return None

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
        session.detector_session.is_enabled = mode is not None
        if mode is None:
            session.detector_session.reset()
            session.audio_received_sample_count = 0
            session.mode = None
            session.policy = None
            session.clear_pause()
        else:
            session.detector_session.detector.config = detector_config
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

    def end_turn(self, session: TurnSession) -> TimedChunk:
        boundary_time_ms = (
            session.pause_start_ms
            if session.pause_start_ms is not None
            else session.audio_received_end_ms
        )
        session.is_open = False
        session.clear_pause()
        return boundary_chunk(False, boundary_time_ms)


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
