# SPDX-License-Identifier: Apache-2.0
"""VAD session state and its standalone Session stage."""

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
    SEMANTIC_VAD,
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
    """Frame-level detection over 16 kHz PCM; sample offsets reset with the detector."""

    config: VADConfig
    is_speech: bool
    last_speech_offset: int
    samples_consumed: int

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
    """Apply VAD options from a realtime turn_detection setting."""
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
                and not (
                    field == "silence_duration_ms"
                    and turn_detection.get("type") == SEMANTIC_VAD
                )
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
    origin_ms: float | None = None

    @property
    def completed_frame_end_ms(self) -> float | None:
        if self.origin_ms is None:
            return None
        else:
            return self.boundary_ms(self.detector.samples_consumed)

    def boundary_ms(self, sample_offset: int) -> float:
        assert self.origin_ms is not None
        return self.origin_ms + sample_offset * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ

    def process_audio(
        self,
        chunk: TimedChunk,
        *,
        expected_start_sample: int | None = None,
        require_duration_match: bool = False,
    ) -> list[tuple[bool, float]]:
        self.validate_audio_chunk(
            chunk,
            expected_start_sample=expected_start_sample,
            require_duration_match=require_duration_match,
        )
        if not chunk.payload or not self.is_enabled:
            return []
        else:
            assert isinstance(chunk.payload, bytes)
            return self.process_pcm(chunk.payload)

    def validate_audio_chunk(
        self,
        chunk: TimedChunk,
        *,
        expected_start_sample: int | None = None,
        require_duration_match: bool = False,
    ) -> int:
        if chunk.format != PCM16 or not isinstance(chunk.payload, bytes):
            raise ValueError("audio must be 16 kHz mono pcm16 bytes")
        elif len(chunk.payload) % 2:
            raise ValueError("pcm16 audio must contain whole samples")
        else:
            sample_count = len(chunk.payload) // 2
        if (
            require_duration_match
            and chunk.duration_ms != sample_count * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ
        ):
            raise ValueError("audio duration must match its pcm16 sample count")
        else:
            pass
        if (
            self.is_enabled
            and sample_count
            and self.origin_ms is not None
            and expected_start_sample is not None
        ):
            actual_start_sample = round(
                (chunk.t_start_ms - self.origin_ms)
                * SESSION_INPUT_SAMPLE_RATE_HZ
                / 1000
            )
            if actual_start_sample != expected_start_sample:
                expected_start_ms = self.origin_ms + (
                    expected_start_sample * 1000 / SESSION_INPUT_SAMPLE_RATE_HZ
                )
                raise ValueError(
                    "fused turn_detection requires contiguous PCM within an enabled "
                    "detector epoch; expected the next sample at "
                    f"{expected_start_ms:g} ms, got {chunk.t_start_ms:g} ms"
                )
            else:
                pass
        else:
            pass
        if not chunk.payload or not self.is_enabled:
            return sample_count
        elif self.origin_ms is None:
            self.origin_ms = chunk.t_start_ms
        else:
            pass
        return sample_count

    def process_pcm(self, pcm: bytes) -> list[tuple[bool, float]]:
        if not pcm or not self.is_enabled:
            return []
        else:
            return [
                (started, self.boundary_ms(sample_offset))
                for started, sample_offset in self.detector.boundaries(pcm)
            ]

    def reset(self) -> None:
        self.detector.reset()
        self.origin_ms = None


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
            for started, boundary_time_ms in session.process_audio(chunk):
                context.emit(boundary_chunk(started, boundary_time_ms))
        elif chunk_control_type == COMMIT:
            session.reset()
        elif chunk_control_type == CONFIGURE:
            if "turn_detection" not in chunk.payload:
                pass
            elif chunk.payload["turn_detection"] is None:
                session.reset()
                session.is_enabled = False
            else:
                session.detector.config = turn_detection_config(
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
