# SPDX-License-Identifier: Apache-2.0
"""The fused turn detection stage and its realtime Session contract."""

import asyncio
import threading
from collections.abc import Sequence
from dataclasses import dataclass
from multiprocessing.queues import Queue

import numpy as np
import pytest

from sglang_omni.audio.segmentation import UtteranceSegmenter
from sglang_omni.audio.semantic_vad import SemanticVADConfig
from sglang_omni.audio.vad import (
    VAD_FRAME_SAMPLES,
    VAD_SAMPLE_RATE,
    Emit,
    VADConfig,
    VADProcessResult,
)
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.session_audio import AudioSessionPipelineConfig
from sglang_omni.models.qwen3_omni.payload_types import UNIT_FORMAT
from sglang_omni.models.qwen3_omni.session_input import QwenInputHooks
from sglang_omni.models.streaming_cascade.base import text_chunk, transcript_chunk
from sglang_omni.models.streaming_cascade.config import StreamingCascadePipelineConfig
from sglang_omni.models.turn_detection.node import (
    SESSION_PORTS,
    SMART_TURN_MODEL_REPO,
    TURN_DETECTION_STAGE,
    TurnDetectionStageConfig,
    turn_detection_stage,
)
from sglang_omni.models.turn_detection.session import TurnDetectionHooks, TurnSession
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SessionGraph,
    SessionIdentity,
    SessionNode,
    TimedChunk,
)
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    PCM16,
    SEMANTIC_VAD,
    SERVER_VAD,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)
from sglang_omni.scheduling.session import SessionContext, SessionScheduler
from sglang_omni.serve.realtime.session_bridge import capabilities_of
from tests.unit_test.fixtures.cascade_fakes import build, drain
from tests.unit_test.fixtures.session_outputs import control_chunk
from tests.unit_test.fixtures.session_pipeline import pipeline

UNIT_MS = 200
UNIT_SAMPLES = UNIT_MS * VAD_SAMPLE_RATE // 1000
MEDIUM = SemanticVADConfig.from_eagerness("medium")
PROFILE_PATH = (
    "examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-base-clone/launch.yaml"
)
SERVER_PROFILE_PATH = "examples/streaming_cascade/profiles/qwen3-asr-qwen3.8-27b-vivian-tts-tool/launch.yaml"


@dataclass(frozen=True)
class ScriptedBoundary:
    ready_sample: int
    started: bool
    boundary_sample: int


def sample_at_ms(time_ms: float) -> int:
    return round(time_ms * VAD_SAMPLE_RATE / 1000)


def scripted_boundary(
    ready_ms: float, started: bool, boundary_ms: float
) -> ScriptedBoundary:
    return ScriptedBoundary(sample_at_ms(ready_ms), started, sample_at_ms(boundary_ms))


class ScriptedDetector:
    """A frame-progress detector whose ordered boundaries come from a script."""

    def __init__(
        self, script_by_reset: Sequence[Sequence[ScriptedBoundary]], config: VADConfig
    ) -> None:
        self.script_by_reset = [list(script) for script in script_by_reset]
        self.config = config
        self.samples_consumed = 0
        self.partial_samples = 0
        self.is_speech = False
        self.last_speech_offset = 0
        self.reset_count = 0
        self.boundary_index = 0

    def process_with_progress(self, pcm_bytes: bytes) -> VADProcessResult:
        self.partial_samples += len(pcm_bytes) // 2
        completed_samples = self.partial_samples // VAD_FRAME_SAMPLES
        self.partial_samples %= VAD_FRAME_SAMPLES
        self.samples_consumed += completed_samples * VAD_FRAME_SAMPLES
        if self.is_speech:
            self.last_speech_offset = self.samples_consumed
        else:
            pass

        script = self.script_by_reset[
            min(self.reset_count, len(self.script_by_reset) - 1)
        ]
        events: list[Emit] = []
        while (
            self.boundary_index < len(script)
            and script[self.boundary_index].ready_sample <= self.samples_consumed
        ):
            boundary = script[self.boundary_index]
            events.append(
                Emit(
                    event_type=(
                        "speech_started" if boundary.started else "speech_stopped"
                    ),
                    sample_offset=boundary.boundary_sample,
                )
            )
            self.is_speech = boundary.started
            if boundary.started:
                self.last_speech_offset = self.samples_consumed
            else:
                pass
            self.boundary_index += 1
        return VADProcessResult(tuple(events), self.samples_consumed)

    def reset(self) -> None:
        self.reset_count += 1
        self.samples_consumed = 0
        self.partial_samples = 0
        self.is_speech = False
        self.last_speech_offset = 0
        self.boundary_index = 0


class FakeEndOfTurn:
    """Probabilities read from a script, one per scored pause."""

    def __init__(self, probabilities: Sequence[float]) -> None:
        self.probabilities = list(probabilities)
        self.scored_s: list[float] = []

    def predict(self, audio: np.ndarray, sample_rate: int) -> float:
        assert sample_rate == VAD_SAMPLE_RATE and audio.dtype == np.float32
        self.scored_s.append(audio.size / sample_rate)
        return self.probabilities.pop(0)


class Node:
    """One hook session receiving PCM through its scheduler append method."""

    def __init__(
        self,
        probabilities: Sequence[float] = (),
        *,
        eagerness: str = "medium",
        params: dict[str, object] | None = None,
        script_by_reset: Sequence[Sequence[ScriptedBoundary]] = ((),),
        vad_config: VADConfig | None = None,
        origin_ms: float = 0,
        hooks: TurnDetectionHooks | None = None,
        identity: SessionIdentity | None = None,
    ) -> None:
        self.model = FakeEndOfTurn(probabilities)
        self.detectors: list[ScriptedDetector] = []

        def new_detector(config: VADConfig) -> ScriptedDetector:
            detector = ScriptedDetector(script_by_reset, config)
            self.detectors.append(detector)
            return detector

        self.hooks = hooks or TurnDetectionHooks(
            self.model,
            eagerness,
            vad_config or VADConfig(silence_duration_ms=160),
            new_detector,
        )
        self.outputs: list[TimedChunk] = []
        self.identity = identity or SessionIdentity("a")
        self.context = SessionContext(
            session_identity=self.identity,
            cancelled=threading.Event(),
            emit=self.outputs.append,
        )
        self.request = OmniRequest(None, params=params or {})
        self.hooks.open(self.identity, self.request)
        self.origin_ms = origin_ms
        self.input_samples = 0
        self.sequence = 0

    @property
    def session(self) -> TurnSession:
        return self.hooks.sessions[self.identity]

    @property
    def detector(self) -> ScriptedDetector:
        return self.session.vad_session.detector

    @property
    def input_end_ms(self) -> float:
        return self.origin_ms + self.input_samples * 1000 / VAD_SAMPLE_RATE

    def append(self, chunk: TimedChunk) -> None:
        self.hooks.append(chunk, StagePayload("x", self.request, None), self.context)

    def audio_until_ms(
        self, until_ms: float, *, chunk_samples: int = UNIT_SAMPLES, eos: bool = False
    ) -> None:
        target_samples = sample_at_ms(until_ms - self.origin_ms)
        if target_samples < self.input_samples:
            raise ValueError("audio timeline must move forward")
        else:
            pass
        eos_sent = False
        while self.input_samples < target_samples:
            sample_count = min(chunk_samples, target_samples - self.input_samples)
            chunk_eos = eos and self.input_samples + sample_count == target_samples
            chunk = TimedChunk(
                AUDIO,
                self.input_end_ms,
                sample_count * 1000 / VAD_SAMPLE_RATE,
                self.sequence,
                bytes(sample_count * 2),
                format=PCM16,
                eos=chunk_eos,
            )
            self.input_samples += sample_count
            self.sequence += 1
            self.append(chunk)
            eos_sent = chunk_eos
        if eos and not eos_sent:
            self.append(
                TimedChunk(
                    AUDIO,
                    self.input_end_ms,
                    0,
                    self.sequence,
                    b"",
                    format=PCM16,
                    eos=True,
                )
            )
            self.sequence += 1
        else:
            pass

    def empty_eos(self) -> None:
        self.append(
            TimedChunk(
                AUDIO,
                self.input_end_ms,
                0,
                self.sequence,
                b"",
                format=PCM16,
                eos=True,
            )
        )
        self.sequence += 1

    def empty_audio(self) -> None:
        self.append(
            TimedChunk(
                AUDIO,
                self.input_end_ms,
                0,
                self.sequence,
                b"",
                format=PCM16,
            )
        )
        self.sequence += 1

    def control(self, kind: str, **fields: object) -> None:
        self.append(control_chunk(kind, **fields))

    def boundaries(self) -> list[tuple[str, float]]:
        return [(o.payload["type"], o.payload["t_ms"]) for o in self.outputs]


def make_scripted_hooks(
    scripts: Sequence[Sequence[ScriptedBoundary]],
) -> tuple[TurnDetectionHooks, list[ScriptedDetector]]:
    detectors: list[ScriptedDetector] = []

    def new_detector(config: VADConfig) -> ScriptedDetector:
        detector = ScriptedDetector(scripts, config)
        detectors.append(detector)
        return detector

    hooks = TurnDetectionHooks(
        FakeEndOfTurn([0.99] * 8),
        "medium",
        VADConfig(silence_duration_ms=160),
        new_detector,
    )
    return hooks, detectors


def test_a_confident_end_waits_for_processed_silence_after_the_candidate_pause() -> (
    None
):
    node = Node(
        [0.9],
        script_by_reset=(
            (
                scripted_boundary(800, True, 800),
                scripted_boundary(2000, False, 2000),
            ),
        ),
    )
    node.audio_until_ms(1000)
    node.audio_until_ms(2600)
    assert node.boundaries() == [(SPEECH_STARTED, 800)]
    assert node.model.scored_s == [1.4]
    node.audio_until_ms(2800)
    assert node.boundaries() == [
        (SPEECH_STARTED, 800),
        (SPEECH_STOPPED, 2000),
    ]


def test_an_immediate_end_closes_at_the_candidate_pause_after_required_silence() -> (
    None
):
    node = Node(
        [0.99],
        script_by_reset=(
            (
                scripted_boundary(800, True, 800),
                scripted_boundary(2000, False, 2000),
            ),
        ),
    )
    node.audio_until_ms(2600)
    assert node.boundaries() == [
        (SPEECH_STARTED, 800),
        (SPEECH_STOPPED, 2000),
    ]


def test_an_unsure_end_waits_for_the_longest_pause() -> None:
    node = Node(
        [0.1],
        script_by_reset=(
            (
                scripted_boundary(800, True, 800),
                scripted_boundary(2000, False, 2000),
            ),
        ),
    )
    node.audio_until_ms(2600)
    node.audio_until_ms(3800)
    assert node.boundaries() == [(SPEECH_STARTED, 800)]
    node.audio_until_ms(4000)
    assert node.boundaries() == [
        (SPEECH_STARTED, 800),
        (SPEECH_STOPPED, 2000),
    ]


def test_speech_resumption_cancels_the_pause_and_a_later_pause_is_rescored() -> None:
    node = Node(
        [0.1, 0.99],
        script_by_reset=(
            (
                scripted_boundary(800, True, 800),
                scripted_boundary(2000, False, 2000),
                scripted_boundary(2700, True, 2700),
                scripted_boundary(3800, False, 3800),
            ),
        ),
    )
    node.audio_until_ms(1000)
    node.audio_until_ms(2600)
    node.audio_until_ms(3000)
    node.audio_until_ms(4400)
    assert node.boundaries() == [
        (SPEECH_STARTED, 800),
        (SPEECH_STOPPED, 3800),
    ]
    assert node.model.scored_s == [1.4, 3.2]


def test_the_semantic_history_is_bounded_to_eight_seconds() -> None:
    node = Node(
        [0.99],
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(11400, False, 11400),
            ),
        ),
    )
    node.audio_until_ms(12000)
    assert node.model.scored_s == [MEDIUM.max_utterance_seconds]
    assert node.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 11400),
    ]


@pytest.mark.parametrize(
    ("eagerness", "probability", "required_silence_ms"),
    [("high", 0.85, 320), ("medium", 0.85, 2000), ("low", 0.95, 1200)],
)
def test_eagerness_presets_keep_the_existing_probability_thresholds(
    eagerness: str, probability: float, required_silence_ms: int
) -> None:
    node = Node(
        [probability],
        eagerness=eagerness,
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(928, False, 928),
            ),
        ),
    )
    node.audio_until_ms(928)
    assert node.session.required_silence_ms == required_silence_ms


def test_resumed_speech_inside_the_deadline_fragment_prevents_a_premature_stop() -> (
    None
):
    node = Node(
        [0.85],
        eagerness="high",
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(1184, False, 928),
                scripted_boundary(1248, True, 1248),
            ),
        ),
    )
    node.audio_until_ms(1184)
    assert node.session.pause_start_ms == 928
    assert node.session.vad_session.completed_frame_end_ms == 1184
    node.audio_until_ms(1280)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert node.session.is_open and node.session.pause_start_ms is None


def test_no_resumption_commits_when_the_completed_frame_horizon_reaches_the_deadline() -> (
    None
):
    node = Node(
        [0.85],
        eagerness="high",
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(1184, False, 928),
            ),
        ),
    )
    node.audio_until_ms(1184)
    node.audio_until_ms(1248)
    assert node.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 928),
    ]


def test_received_audio_past_the_deadline_waits_for_a_complete_vad_frame() -> None:
    node = Node(
        [0.99],
        eagerness="high",
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(928, False, 928),
            ),
        ),
    )
    node.audio_until_ms(928)
    node.audio_until_ms(1130)
    assert node.input_end_ms == 1130
    assert node.session.vad_session.completed_frame_end_ms == 1120
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    node.audio_until_ms(1160)
    assert node.session.vad_session.completed_frame_end_ms == 1152
    assert node.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 928),
    ]


def test_a_timeline_gap_is_rejected_instead_of_counted_as_vad_progress() -> None:
    node = Node()
    node.audio_until_ms(100)
    with pytest.raises(ValueError, match="requires contiguous input audio timing"):
        node.append(
            TimedChunk(
                AUDIO,
                102,
                32,
                node.sequence,
                bytes(32 * 32),
                format=PCM16,
            )
        )


def test_multiple_boundaries_in_one_append_are_processed_before_the_timer() -> None:
    node = Node(
        [0.85],
        eagerness="high",
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(928, False, 928),
                scripted_boundary(1248, True, 1248),
            ),
        ),
    )
    node.audio_until_ms(1280, chunk_samples=1280 * VAD_SAMPLE_RATE // 1000)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert node.session.is_open and node.session.pause_start_ms is None
    assert node.model.scored_s == [1.28]


def test_empty_and_short_fragments_do_not_advance_completed_frame_progress() -> None:
    node = Node(
        [0.99],
        eagerness="high",
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(928, False, 928),
            ),
        ),
    )
    node.audio_until_ms(928)
    node.empty_audio()
    node.empty_audio()
    node.audio_until_ms(948)
    assert node.session.vad_session.completed_frame_end_ms == 928
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    node.audio_until_ms(1152)
    assert node.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 928),
    ]


def test_server_vad_emits_each_detector_boundary_without_smart_turn_scoring() -> None:
    node = Node(
        params={"turn_detection": {"type": SERVER_VAD}},
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(256, False, 160),
                scripted_boundary(480, True, 400),
                scripted_boundary(640, False, 560),
            ),
        ),
    )
    node.audio_until_ms(800)
    assert node.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 160),
        (SPEECH_STARTED, 400),
        (SPEECH_STOPPED, 560),
    ]
    assert node.model.scored_s == []


def test_vad_options_and_turn_detection_types_are_typed_and_round_trip() -> None:
    config = AudioSessionPipelineConfig(
        model_path="unused",
        session_nodes={
            "turn_detection": {
                "stages": ["turn_detection"],
                "inputs": ["input:pcm16", "input:session"],
            }
        },
        session_publish={"turn_detection": ["session"]},
    )
    assert [stage.name for stage in config.stages] == [TURN_DETECTION_STAGE]
    assert isinstance(config.stages[0], TurnDetectionStageConfig)
    assert config.session_graph().nodes["turn_detection"].inputs == (
        "input:pcm16",
        "input:session",
    )
    assert config.session_ports() == SESSION_PORTS
    assert capabilities_of(config.session_ports()).turn_control == (
        SERVER_VAD,
        SEMANTIC_VAD,
        None,
    )

    declared = turn_detection_stage().model_dump()
    tuned = TurnDetectionStageConfig.model_validate(
        {
            **declared,
            "factory": {
                "model_path": "/models/smart-turn.onnx",
                "eagerness": "high",
                "threshold": 0.6,
                "prefix_padding_ms": 200,
                "silence_duration_ms": 160,
                "background_margin_db": 3.0,
            },
        }
    )
    config.stages[0] = tuned
    resolved = resolve_stage_factory_args(tuned, config)
    assert resolved == {
        "model_path": "/models/smart-turn.onnx",
        "eagerness": "high",
        "threshold": 0.6,
        "prefix_padding_ms": 200,
        "silence_duration_ms": 160,
        "background_margin_db": 3.0,
    }
    rebuilt = AudioSessionPipelineConfig.model_validate(config.model_dump())
    assert rebuilt.session_graph() == config.session_graph()
    with pytest.raises(ValueError, match="eagerness"):
        TurnDetectionStageConfig.model_validate(
            {**declared, "factory": {"eagerness": "eager"}}
        )
    assert turn_detection_stage().factory.model_path == SMART_TURN_MODEL_REPO


def test_the_old_split_graph_is_rejected_with_a_migration_message() -> None:
    with pytest.raises(ValueError, match="split vad -> turn_detection graph"):
        AudioSessionPipelineConfig(
            model_path="unused",
            session_nodes={
                "vad": {"stages": ["vad"], "inputs": ["input:pcm16"]},
                "turn_detection": {
                    "stages": ["turn_detection"],
                    "inputs": [
                        "input:pcm16",
                        "input:session",
                        "vad:session",
                    ],
                },
            },
            session_publish={"turn_detection": ["session"]},
        )


def test_the_semantic_profile_uses_one_fused_stage_and_keeps_standalone_vad_profile() -> (
    None
):
    from sglang_omni.config.manager import ConfigManager

    semantic = ConfigManager.from_file(PROFILE_PATH).config
    assert isinstance(semantic, StreamingCascadePipelineConfig)
    assert [stage.name for stage in semantic.stages].count("turn_detection") == 1
    assert "vad" not in semantic.session_nodes
    assert semantic.session_nodes["turn_detection"].inputs == [
        "input:pcm16",
        "input:session",
    ]
    assert semantic.stage_named("turn_detection").factory.silence_duration_ms == 160

    server = ConfigManager.from_file(SERVER_PROFILE_PATH).config
    assert "vad" in server.session_nodes
    assert "turn_detection" not in server.session_nodes


def test_server_vad_honors_silence_overrides_and_semantic_keeps_candidate_silence() -> (
    None
):
    node = Node(
        params={
            "turn_detection": {
                "type": SERVER_VAD,
                "silence_duration_ms": 800,
            }
        }
    )
    assert node.detector.config.silence_duration_ms == 800
    node.control(CONFIGURE, instructions="unchanged turn settings")
    assert node.session.mode == SERVER_VAD
    assert node.detector.config.silence_duration_ms == 800
    node.control(
        CONFIGURE,
        turn_detection={"type": SEMANTIC_VAD, "silence_duration_ms": 800},
    )
    assert node.detector.config.silence_duration_ms == 160
    with pytest.raises(ValueError, match="turn_detection must be an object"):
        node.control(CONFIGURE, turn_detection=3)
    with pytest.raises(ValueError, match="eagerness"):
        node.control(
            CONFIGURE,
            turn_detection={"type": SEMANTIC_VAD, "eagerness": "eager"},
        )


def test_switching_a_held_semantic_pause_to_server_vad_closes_at_the_pause_start() -> (
    None
):
    node = Node(
        [0.1],
        eagerness="high",
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(1184, False, 928),
            ),
        ),
    )
    node.audio_until_ms(1184)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    node.control(CONFIGURE, turn_detection={"type": SERVER_VAD})
    assert node.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 928),
    ]
    assert not node.session.is_open and node.session.mode == SERVER_VAD


def test_same_mode_configure_recomputes_a_held_pause_without_rescoring() -> None:
    node = Node(
        [0.85],
        eagerness="high",
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(928, False, 928),
            ),
        ),
    )
    node.audio_until_ms(928)
    assert node.session.required_silence_ms == 320
    node.control(
        CONFIGURE,
        turn_detection={"type": SEMANTIC_VAD, "eagerness": "medium"},
    )
    assert node.session.pause_start_ms == 928
    assert node.session.required_silence_ms == 2000
    assert node.model.scored_s == [0.928]


def test_switching_a_held_pause_to_manual_keeps_commit_as_the_only_turn_end() -> None:
    node = Node(
        [0.1],
        eagerness="high",
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(1184, False, 928),
            ),
        ),
    )
    node.audio_until_ms(1184)
    node.control(CONFIGURE, turn_detection=None)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert node.session.is_open and node.session.pause_start_ms is None
    node.audio_until_ms(1280)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    node.control(COMMIT)
    assert not node.session.is_open
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    segmenter = UtteranceSegmenter()
    actions = [
        action for boundary in node.outputs for action in segmenter.step(boundary)
    ]
    actions.extend(segmenter.step(control_chunk(COMMIT)))
    assert actions == ["open", "finish"]


def test_configure_can_enable_manual_mode_and_same_mode_updates_are_atomic() -> None:
    node = Node(
        params={"turn_detection": None},
        script_by_reset=(
            (),
            (scripted_boundary(32, True, 0),),
        ),
    )
    node.audio_until_ms(100)
    assert node.boundaries() == []
    node.control(CONFIGURE, turn_detection={"type": SEMANTIC_VAD, "eagerness": "high"})
    assert node.session.mode == SEMANTIC_VAD
    node.audio_until_ms(200)
    assert node.boundaries() == [(SPEECH_STARTED, 100)]
    node.control(CONFIGURE, turn_detection={"type": SEMANTIC_VAD, "eagerness": "low"})
    assert node.session.policy == SemanticVADConfig.from_eagerness("low")
    assert node.session.pause_start_ms is None


def test_eos_closes_speech_held_pauses_and_never_duplicates_a_regular_commit() -> None:
    speaking = Node(
        [0.99],
        script_by_reset=((scripted_boundary(32, True, 0),),),
    )
    speaking.audio_until_ms(150)
    speaking.empty_eos()
    assert speaking.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 128),
    ]

    held = Node(
        [0.85],
        eagerness="high",
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(160, False, 96),
            ),
        ),
    )
    held.audio_until_ms(160, eos=True)
    assert held.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 96),
    ]

    ended = Node(
        [0.99],
        eagerness="high",
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(928, False, 928),
            ),
        ),
    )
    ended.audio_until_ms(1152)
    ended.empty_eos()
    assert ended.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 928),
    ]


def test_commit_resets_origin_progress_and_semantic_audio_history() -> None:
    node = Node(
        [0.1, 0.99],
        eagerness="high",
        origin_ms=400,
        script_by_reset=(
            (
                scripted_boundary(32, True, 0),
                scripted_boundary(928, False, 928),
            ),
            (scripted_boundary(32, True, 0),),
        ),
    )
    node.audio_until_ms(1584)
    assert node.session.vad_session.completed_frame_end_ms == 1584
    node.control(COMMIT)
    assert not node.session.audio_history
    assert node.session.vad_session.completed_frame_end_ms is None
    node.audio_until_ms(1684)
    assert node.boundaries() == [
        (SPEECH_STARTED, 400),
        (SPEECH_STARTED, 1584),
    ]
    assert node.model.scored_s == [1.0]


def test_interleaved_sessions_keep_detector_and_turn_state_independent() -> None:
    script = ((scripted_boundary(32, True, 0),),)
    hooks, detectors = make_scripted_hooks(script)
    first = Node(hooks=hooks, identity=SessionIdentity("first"), origin_ms=0)
    second = Node(hooks=hooks, identity=SessionIdentity("second"), origin_ms=500)
    first.audio_until_ms(100)
    assert first.boundaries() == [(SPEECH_STARTED, 0)]
    assert second.boundaries() == []
    second.audio_until_ms(600)
    assert second.boundaries() == [(SPEECH_STARTED, 500)]
    assert len(detectors) == 2
    assert first.session is not second.session


def held_pause_then_end() -> list[TimedChunk]:
    node = Node(
        [0.1, 0.99],
        script_by_reset=(
            (
                scripted_boundary(800, True, 800),
                scripted_boundary(2000, False, 2000),
                scripted_boundary(2700, True, 2700),
                scripted_boundary(3800, False, 3800),
            ),
        ),
    )
    node.audio_until_ms(1000)
    node.audio_until_ms(2600)
    node.audio_until_ms(3000)
    node.audio_until_ms(4400)
    return node.outputs


def test_the_recognizer_hears_a_held_pause_as_one_utterance() -> None:
    segmenter = UtteranceSegmenter()
    actions = [
        action for chunk in held_pause_then_end() for action in segmenter.step(chunk)
    ]
    assert actions == ["open", "finish"]


@pytest.mark.asyncio
async def test_the_dialog_interrupts_on_turn_start_and_not_on_a_held_pause() -> None:
    paused = held_pause_then_end()
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("Hello world.", 1),
            *paused[1:],
            paused[0],
        ],
    )
    assert [control_type(output) for output in outputs] == [
        TURN_QUEUED,
        TURN_STARTED,
        None,
        "interrupt",
        TURN_INTERRUPTED,
    ]


def test_the_qwen_dialog_queues_a_held_pause_as_one_turn() -> None:
    hooks = QwenInputHooks(
        max_utterance_seconds=8, max_queued_turns=2, max_utterance_frames=4
    )
    identity = SessionIdentity("dialog")
    outputs: list[TimedChunk] = []
    hooks.open(identity, OmniRequest(None))
    hooks.attach_output_stream(identity, outputs.append)
    context = SessionContext(
        session_identity=identity, cancelled=threading.Event(), emit=outputs.append
    )
    boundaries = list(held_pause_then_end())
    boundary_index = 0
    for time_ms in range(0, 4400, 20):
        while (
            boundary_index < len(boundaries)
            and boundaries[boundary_index].payload["t_ms"] <= time_ms
        ):
            hooks.append(
                boundaries[boundary_index],
                StagePayload("in", OmniRequest(None), {}),
                context,
            )
            boundary_index += 1
        hooks.append(
            TimedChunk(
                AUDIO,
                time_ms,
                20,
                0,
                b"\x01\x00" * 320,
                format=PCM16,
            ),
            StagePayload("in", OmniRequest(None), {}),
            context,
        )
    assert [control_type(chunk) or chunk.format for chunk in outputs] == [
        TURN_QUEUED,
        UNIT_FORMAT,
    ]


def create_graph_scheduler(
    name: str,
    events: Queue[object],
    vad_script: Sequence[Sequence[ScriptedBoundary]],
) -> SessionScheduler:
    def new_detector(config: VADConfig) -> ScriptedDetector:
        return ScriptedDetector(vad_script, config)

    hooks = TurnDetectionHooks(
        FakeEndOfTurn([0.85]),
        "high",
        VADConfig(silence_duration_ms=160),
        new_detector,
    )
    return SessionScheduler(hooks)


@pytest.mark.asyncio
async def test_the_real_scheduler_and_session_graph_consume_boundaries_in_pcm_order(
    tmp_path,
) -> None:
    script = (
        (
            scripted_boundary(32, True, 0),
            scripted_boundary(1184, False, 928),
            scripted_boundary(1248, True, 1248),
        ),
    )
    graph = SessionGraph(
        nodes={
            "turn_detection": SessionNode(
                stages=("turn_detection",),
                inputs=("input:pcm16", "input:session"),
            )
        },
        publish={"turn_detection": ("session",)},
    )
    async with pipeline(
        tmp_path,
        names=["turn_detection"],
        factories={"turn_detection": f"{__name__}.create_graph_scheduler"},
        factory_kwargs={"vad_script": script},
    ) as (coordinator, _, _):
        coordinator.session_graph = graph
        identity = await coordinator.open_session(OmniRequest(None, {}))
        outputs = coordinator.session_outputs(identity)
        await coordinator.append_session(
            identity,
            TimedChunk(AUDIO, 400, 1184, 0, bytes(1184 * 32), format=PCM16),
        )
        await coordinator.append_session(
            identity,
            TimedChunk(AUDIO, 1584, 96, 1, bytes(96 * 32), format=PCM16),
        )
        received = [await asyncio.wait_for(anext(outputs), 5) for _ in range(3)]
        boundaries = [
            (item.payload["type"], item.payload["t_ms"])
            for item in received
            if item.kind == "data"
        ]
        assert boundaries == [(SPEECH_STARTED, 400)]
        assert [item.input_seq for item in received if item.kind == "input_done"] == [
            0,
            1,
        ]
        await coordinator.close_session(identity)


def test_the_stage_owns_the_fused_endpoint_and_advertises_both_modes() -> None:
    stage = turn_detection_stage()
    assert (stage.name, stage.process, stage.terminal) == (
        TURN_DETECTION_STAGE,
        TURN_DETECTION_STAGE,
        True,
    )
    assert (
        stage.factory_path
        == "sglang_omni.models.turn_detection.session.create_scheduler"
    )
    assert SESSION_PORTS.output is None
    assert SESSION_PORTS.controls == {COMMIT, CONFIGURE}
    assert SESSION_PORTS.turn_detection_types == (SERVER_VAD, SEMANTIC_VAD)
    assert SESSION_PORTS.hot_settings == {"turn_detection"}
