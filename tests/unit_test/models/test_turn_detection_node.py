# SPDX-License-Identifier: Apache-2.0
"""The fused turn detection stage and its realtime Session contract."""

import asyncio
import sys
import threading
import types
from collections import deque
from collections.abc import Sequence
from multiprocessing.queues import Queue
from pathlib import Path

import numpy as np
import pytest

from sglang_omni.audio.segmentation import UtteranceSegmenter
from sglang_omni.audio.semantic_vad import SemanticVADConfig
from sglang_omni.audio.smart_turn import SmartTurnEOU
from sglang_omni.audio.vad import VAD_FRAME_SAMPLES, StreamingVAD, VADConfig
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.session_audio import AudioSessionPipelineConfig
from sglang_omni.models.qwen3_omni.payload_types import UNIT_FORMAT
from sglang_omni.models.qwen3_omni.session_input import QwenInputHooks
from sglang_omni.models.streaming_cascade.base import text_chunk, transcript_chunk
from sglang_omni.models.streaming_cascade.config import CascadeSessionNodeConfig
from sglang_omni.models.turn_detection.node import (
    SESSION_PORTS,
    SMART_TURN_MODEL_REPO,
    TURN_DETECTION_STAGE,
    turn_detection_stage,
)
from sglang_omni.models.turn_detection.session import TurnDetectionHooks
from sglang_omni.models.turn_detection.session import (
    create_scheduler as create_turn_scheduler,
)
from sglang_omni.models.vad.node import vad_stage
from sglang_omni.models.vad.session import SileroSpeechDetector
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

SAMPLE_RATE_HZ = 16000
PROFILE_PATH = (
    "examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-base-clone/launch.yaml"
)
SERVER_PROFILE_PATH = "examples/streaming_cascade/profiles/qwen3-asr-qwen3.8-27b-vivian-tts-tool/launch.yaml"


@pytest.fixture(autouse=True)
def deterministic_silero(monkeypatch: pytest.MonkeyPatch) -> None:
    silero_module = types.ModuleType("silero_vad")

    def load_silero_vad(*, onnx: bool) -> object:
        assert onnx
        return object()

    def infer(self: StreamingVAD, frame: np.ndarray) -> float:
        return float(np.any(frame))

    silero_module.load_silero_vad = load_silero_vad
    monkeypatch.setitem(sys.modules, "silero_vad", silero_module)
    monkeypatch.setattr(StreamingVAD, "infer", infer)


class FakeEndOfTurn:
    def __init__(self, probabilities: Sequence[float]) -> None:
        self.probabilities = deque(probabilities)
        self.scored_seconds: list[float] = []

    def predict(self, audio: np.ndarray, sample_rate: int) -> float:
        assert sample_rate == SAMPLE_RATE_HZ
        assert audio.dtype == np.float32
        self.scored_seconds.append(audio.size / sample_rate)
        return self.probabilities.popleft()


def pcm_samples(speech: bool, sample_count: int) -> bytes:
    amplitude = 12000 if speech else 0
    return np.full(sample_count, amplitude, dtype="<i2").tobytes()


def pcm_frames(speech_frames: Sequence[bool]) -> bytes:
    return b"".join(pcm_samples(speech, VAD_FRAME_SAMPLES) for speech in speech_frames)


def audio_chunk(
    start_ms: float, pcm: bytes, sequence: int, *, eos: bool = False
) -> TimedChunk:
    duration_ms = len(pcm) / 2 * 1000 / SAMPLE_RATE_HZ
    return TimedChunk(
        AUDIO,
        start_ms,
        duration_ms,
        sequence,
        pcm,
        format=PCM16,
        eos=eos,
    )


class Node:
    """One hook session appended using its realtime audio and control inputs."""

    def __init__(
        self,
        probabilities: Sequence[float] = (),
        *,
        eagerness: str = "medium",
        settings: dict[str, object] | None = None,
        mode: str | None = None,
        vad_config: VADConfig | None = None,
        hooks: TurnDetectionHooks | None = None,
        identity: SessionIdentity | None = None,
    ) -> None:
        self.model = FakeEndOfTurn(probabilities)
        if hooks is None:
            initial_settings = settings or {}
            if mode is not None:
                initial_settings = {"turn_detection": {"type": mode}}
            hooks = TurnDetectionHooks(
                self.model,
                eagerness,
                vad_config
                or VADConfig(
                    prefix_padding_ms=0,
                    silence_duration_ms=160,
                    background_margin_db=None,
                ),
                SileroSpeechDetector,
            )
        else:
            self.model = hooks.eou_model
            initial_settings = settings or {}
        self.hooks = hooks
        self.outputs: list[TimedChunk] = []
        self.identity = identity or SessionIdentity("test")
        self.context = SessionContext(
            session_identity=self.identity,
            cancelled=threading.Event(),
            emit=self.outputs.append,
        )
        self.sequence = 0
        self.hooks.open(self.identity, OmniRequest(None, initial_settings))

    @property
    def session(self):
        return self.hooks.sessions[self.identity]

    def append(self, chunk: TimedChunk) -> None:
        self.hooks.append(
            chunk, StagePayload("input", OmniRequest(None), None), self.context
        )

    def audio(self, start_ms: float, pcm: bytes, *, eos: bool = False) -> None:
        self.append(audio_chunk(start_ms, pcm, self.sequence, eos=eos))
        self.sequence += 1

    def frames(self, start_ms: float, speech_frames: Sequence[bool]) -> None:
        self.audio(start_ms, pcm_frames(speech_frames))

    def control(self, event: str, **settings: object) -> None:
        self.append(control_chunk(event, **settings))

    def boundaries(self) -> list[tuple[str, float]]:
        return [
            (chunk.payload["type"], chunk.payload["t_ms"]) for chunk in self.outputs
        ]


def test_completed_vad_frames_order_resumption_before_the_silence_deadline() -> None:
    for resumed in (True, False):
        node = Node([0.85], eagerness="high")
        node.frames(0, [True] * 29)
        node.frames(928, [False] * 8)
        assert node.boundaries() == [(SPEECH_STARTED, 0)]
        assert node.session.pause_start_ms == 928
        assert node.session.required_silence_ms == 320
        assert node.session.detector_session.completed_frame_end_ms == 1184

        final_fragment = [False, False, True] if resumed else [False, False]
        node.frames(1184, final_fragment)
        if resumed:
            assert node.boundaries() == [(SPEECH_STARTED, 0)]
        else:
            assert node.boundaries() == [
                (SPEECH_STARTED, 0),
                (SPEECH_STOPPED, 928),
            ]


def test_partial_and_empty_pcm_do_not_advance_the_vad_deadline() -> None:
    node = Node([0.99], eagerness="high")
    node.frames(0, [True] * 29)
    node.audio(928, pcm_samples(False, 202 * 16))
    node.audio(1130, b"")
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert node.session.detector_session.completed_frame_end_ms == 1120

    node.audio(1130, pcm_samples(False, 22 * 16))
    assert node.session.detector_session.completed_frame_end_ms == 1152
    assert node.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 928),
    ]


def test_multiple_boundaries_in_one_append_are_consumed_in_order() -> None:
    node = Node([0.1, 0.99], eagerness="high")
    node.frames(0, [True] * 6 + [False] * 5 + [True] * 2 + [False] * 5)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert node.model.scored_seconds == [0.576, 0.576]
    node.frames(576, [False] * 2)
    assert node.boundaries() == [(SPEECH_STARTED, 0), (SPEECH_STOPPED, 416)]


def test_semantic_history_remains_bounded_to_eight_seconds() -> None:
    node = Node([0.99], eagerness="high")
    for start_ms in range(0, 12000, 200):
        node.audio(start_ms, pcm_samples(True, 200 * 16))
    node.audio(12000, pcm_samples(False, 200 * 16))
    node.audio(12200, pcm_samples(False, 24 * 16))
    assert node.model.scored_seconds == [SemanticVADConfig.max_utterance_seconds]
    assert node.boundaries()[-1] == (SPEECH_STOPPED, 12000)


@pytest.mark.parametrize(
    ("eagerness", "probability", "required_silence_ms"),
    [("high", 0.85, 320), ("medium", 0.85, 2000), ("low", 0.95, 1200)],
)
def test_eagerness_presets_keep_their_existing_thresholds(
    eagerness: str, probability: float, required_silence_ms: int
) -> None:
    policy = SemanticVADConfig.from_eagerness(eagerness)
    assert policy.required_silence_ms(probability) == required_silence_ms


@pytest.mark.parametrize(
    ("factory_overrides", "expected_margin_db", "expected_silence_ms"),
    [
        ({}, 6.0, 800),
        ({"background_margin_db": 3.0}, 3.0, 800),
        ({"background_margin_db": None}, None, 800),
        ({"silence_duration_ms": None}, 6.0, 500),
    ],
)
def test_factory_resolution_preserves_omitted_and_null_defaults(
    monkeypatch: pytest.MonkeyPatch,
    factory_overrides: dict[str, object],
    expected_margin_db: float | None,
    expected_silence_ms: int,
) -> None:
    def load_model(cls: type[SmartTurnEOU], model_path: str) -> FakeEndOfTurn:
        assert model_path == SMART_TURN_MODEL_REPO
        return FakeEndOfTurn([])

    monkeypatch.setattr(SmartTurnEOU, "load", classmethod(load_model))
    stage_document = turn_detection_stage().model_dump()
    stage_document["factory"] = {**stage_document["factory"], **factory_overrides}
    config = AudioSessionPipelineConfig(
        model_path="unused",
        stages=[stage_document],
        session_nodes={
            "turn_detection": {
                "stages": [TURN_DETECTION_STAGE],
                "inputs": ["input:pcm16", "input:session"],
            }
        },
        session_publish={"turn_detection": ["session"]},
    )
    arguments = resolve_stage_factory_args(
        config.stage_named(TURN_DETECTION_STAGE), config
    )
    assert arguments["model_path"] == SMART_TURN_MODEL_REPO
    scheduler = create_turn_scheduler(**arguments)
    assert scheduler.session_hooks.vad_config.background_margin_db == expected_margin_db
    assert scheduler.session_hooks.vad_config.silence_duration_ms == expected_silence_ms


def test_server_vad_passes_through_boundaries_without_smart_turn_scoring() -> None:
    node = Node(mode=SERVER_VAD, vad_config=VADConfig(silence_duration_ms=160))
    node.frames(0, [True] * 3)
    node.frames(96, [False] * 5)
    assert node.boundaries() == [(SPEECH_STARTED, 0), (SPEECH_STOPPED, 96)]
    assert node.model.scored_seconds == []


def test_client_vad_overrides_apply_without_changing_semantic_candidate_silence() -> (
    None
):
    node = Node(
        mode=SERVER_VAD,
        vad_config=VADConfig(
            threshold=0.5,
            prefix_padding_ms=300,
            silence_duration_ms=160,
            background_margin_db=None,
        ),
    )
    node.control(
        CONFIGURE,
        turn_detection={
            "type": SERVER_VAD,
            "threshold": 0.7,
            "prefix_padding_ms": 80,
            "silence_duration_ms": 800,
        },
    )
    config = node.session.detector_session.detector.config
    assert (config.threshold, config.prefix_padding_ms, config.silence_duration_ms) == (
        0.7,
        80,
        800,
    )
    node.control(
        CONFIGURE,
        turn_detection={
            "type": SEMANTIC_VAD,
            "eagerness": "high",
            "silence_duration_ms": 900,
        },
    )
    semantic_config = node.session.detector_session.detector.config
    assert (
        semantic_config.threshold,
        semantic_config.prefix_padding_ms,
        semantic_config.silence_duration_ms,
    ) == (0.7, 80, 160)

    node.control(
        CONFIGURE,
        turn_detection={"type": SERVER_VAD, "silence_duration_ms": 800},
    )
    assert node.session.detector_session.detector.config.silence_duration_ms == 800
    node.frames(256, [True] * 3)
    node.frames(352, [False] * 24)
    assert node.boundaries()[-1] == (SPEECH_STARTED, 256)
    node.frames(1120, [False])
    assert node.boundaries()[-1] == (SPEECH_STOPPED, 352)
    assert node.model.scored_seconds == []


def test_speech_resumption_clears_a_pause_and_a_later_pause_is_rescored() -> None:
    node = Node([0.1, 0.99], eagerness="high")
    node.frames(0, [True] * 29)
    node.frames(928, [False] * 8)
    node.frames(1184, [True] * 4)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    node.frames(1312, [False] * 8)
    assert node.boundaries() == [(SPEECH_STARTED, 0), (SPEECH_STOPPED, 1312)]
    assert len(node.model.scored_seconds) == 2


@pytest.mark.parametrize("resume_before_deadline", [False, True])
def test_switching_a_held_semantic_pause_to_server_vad_waits_for_server_silence(
    resume_before_deadline: bool,
) -> None:
    node = Node([0.85], eagerness="high")
    node.frames(0, [True] * 29)
    node.frames(928, [False] * 8)
    node.control(
        CONFIGURE,
        turn_detection={"type": SERVER_VAD, "silence_duration_ms": 800},
    )
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert node.session.is_open

    if resume_before_deadline:
        node.frames(1184, [False, False, True])
        assert node.boundaries() == [(SPEECH_STARTED, 0)]
        assert node.session.is_open
    else:
        node.frames(1184, [False] * 16)
        assert node.boundaries() == [(SPEECH_STARTED, 0)]
        assert node.session.is_open
        node.frames(1696, [False])
        assert node.boundaries() == [(SPEECH_STARTED, 0), (SPEECH_STOPPED, 928)]
        assert not node.session.is_open
    assert len(node.model.scored_seconds) == 1


def test_eagerness_update_reuses_a_held_pause_score_until_the_next_audio() -> None:
    node = Node([0.85], eagerness="high")
    node.frames(0, [True] * 29)
    node.frames(928, [False] * 8)
    node.control(
        CONFIGURE,
        turn_detection={"type": SEMANTIC_VAD, "eagerness": "medium"},
    )
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert len(node.model.scored_seconds) == 1
    assert node.session.required_silence_ms == 2000
    node.frames(1184, [False] * 8)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]


def test_switching_server_back_to_semantic_reuses_the_inherited_pause_score() -> None:
    node = Node([0.85], eagerness="high")
    node.frames(0, [True] * 29)
    node.frames(928, [False] * 8)
    node.control(
        CONFIGURE,
        turn_detection={"type": SERVER_VAD, "silence_duration_ms": 800},
    )
    node.control(
        CONFIGURE,
        turn_detection={"type": SEMANTIC_VAD, "eagerness": "medium"},
    )
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert node.session.required_silence_ms == 2000
    assert len(node.model.scored_seconds) == 1


def test_manual_mode_preserves_an_open_turn_until_commit_and_reenable_is_causal() -> (
    None
):
    node = Node([0.99], eagerness="high")
    node.frames(0, [True] * 5)
    node.control(CONFIGURE, turn_detection=None)
    node.audio(160, pcm_samples(False, 20 * 16), eos=True)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert node.session.is_open
    segmenter = UtteranceSegmenter()
    assert [
        action
        for chunk in (node.outputs[0], control_chunk(COMMIT))
        for action in segmenter.step(chunk)
    ] == ["open", "finish"]

    node.control(COMMIT)
    assert not node.session.is_open
    node.control(CONFIGURE, turn_detection={"type": SEMANTIC_VAD, "eagerness": "high"})
    node.frames(1000, [True] * 5)
    node.frames(1160, [False] * 5)
    node.frames(1320, [False, False])
    assert node.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STARTED, 1000),
        (SPEECH_STOPPED, 1160),
    ]


def test_reenabling_detection_while_a_manual_turn_is_open_suppresses_duplicate_start() -> (
    None
):
    node = Node([0.99], eagerness="high")
    node.frames(0, [True] * 5)
    node.control(CONFIGURE, turn_detection=None)
    node.control(CONFIGURE, turn_detection={"type": SEMANTIC_VAD, "eagerness": "high"})
    node.frames(1000, [True] * 5)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    node.frames(1160, [False] * 5)
    node.frames(1320, [False, False])
    assert node.boundaries() == [(SPEECH_STARTED, 0), (SPEECH_STOPPED, 1160)]


def test_manual_enable_uses_real_vad_and_a_new_nonzero_origin() -> None:
    node = Node(settings={"turn_detection": None})
    node.frames(0, [False] * 5)
    node.control(CONFIGURE, turn_detection={"type": SEMANTIC_VAD})
    node.frames(400, [True] * 5)
    assert node.boundaries() == [(SPEECH_STARTED, 400)]


@pytest.mark.parametrize(
    ("eos_state", "expected"),
    [
        ("speaking", [(SPEECH_STARTED, 0), (SPEECH_STOPPED, 160)]),
        ("held_pause", [(SPEECH_STARTED, 0), (SPEECH_STOPPED, 160)]),
        ("regular_stop", [(SPEECH_STARTED, 0), (SPEECH_STOPPED, 160)]),
    ],
)
def test_eos_closes_active_semantic_turns_once(
    eos_state: str, expected: list[tuple[str, float]]
) -> None:
    node = Node([0.99], eagerness="high")
    node.frames(0, [True] * 5)
    if eos_state == "speaking":
        node.audio(160, b"", eos=True)
    elif eos_state == "held_pause":
        node.frames(160, [False] * 5)
        node.audio(320, b"", eos=True)
    else:
        node.frames(160, [False] * 5)
        node.frames(320, [False, False, False, False, False, False, False])
        node.audio(544, b"", eos=True)
    assert node.boundaries() == expected


def test_commit_resets_origin_progress_and_semantic_history() -> None:
    node = Node([0.99], eagerness="high")
    node.frames(400, [True] * 5)
    node.control(COMMIT)
    node.frames(10000, [True] * 5)
    assert node.boundaries() == [(SPEECH_STARTED, 400), (SPEECH_STARTED, 10000)]
    assert node.model.scored_seconds == []


def test_contiguity_uses_integer_sample_positions_and_rejects_real_gaps() -> None:
    node = Node(vad_config=VADConfig(background_margin_db=None, prefix_padding_ms=0))
    node.audio(400.1, pcm_samples(True, 1))
    node.audio(400.1625, pcm_samples(True, 1))
    with pytest.raises(ValueError, match="requires contiguous PCM"):
        node.audio(400.3, pcm_samples(True, 1))


def test_interleaved_sessions_have_independent_vad_and_turn_state() -> None:
    hooks = TurnDetectionHooks(
        FakeEndOfTurn([]),
        "medium",
        VADConfig(
            prefix_padding_ms=0, silence_duration_ms=160, background_margin_db=None
        ),
        SileroSpeechDetector,
    )
    first = Node(hooks=hooks, identity=SessionIdentity("first"))
    second = Node(hooks=hooks, identity=SessionIdentity("second"))
    first.frames(0, [True] * 5)
    second.frames(500, [True] * 5)
    assert first.boundaries() == [(SPEECH_STARTED, 0)]
    assert second.boundaries() == [(SPEECH_STARTED, 500)]
    assert first.session is not second.session


def test_shared_graph_rejects_a_legacy_split_endpoint_from_stage_ownership() -> None:
    with pytest.raises(ValueError, match="remove the separate VAD node"):
        AudioSessionPipelineConfig(
            model_path="unused",
            session_nodes={
                "audio_vad": {"stages": ["vad"], "inputs": ["input:pcm16"]},
                "endpoint": {
                    "stages": ["turn_detection"],
                    "inputs": ["input:pcm16", "input:session", "audio_vad:session"],
                },
            },
            session_publish={"endpoint": ["session"]},
        )


def test_cascade_rejects_legacy_topology_with_a_nonstandard_vad_node_name() -> None:
    from sglang_omni.config.manager import ConfigManager

    config = ConfigManager.from_file(PROFILE_PATH).config
    config.stages.append(vad_stage())
    config.session_nodes["audio_vad"] = CascadeSessionNodeConfig(
        stages=["vad"], inputs=["input:pcm16"]
    )
    config.session_nodes["turn_detection"].inputs.append("audio_vad:session")
    with pytest.raises(ValueError, match="remove the separate VAD node"):
        config.check_graph()


def test_profiles_keep_their_fused_and_standalone_vad_defaults() -> None:
    from sglang_omni.config.manager import ConfigManager

    semantic = ConfigManager.from_file(PROFILE_PATH).config
    assert [stage.name for stage in semantic.stages].count("turn_detection") == 1
    assert "vad" not in semantic.session_nodes
    assert semantic.session_nodes["turn_detection"].inputs == [
        "input:pcm16",
        "input:session",
    ]
    assert semantic.stage_named("turn_detection").factory.silence_duration_ms == 160
    assert semantic.stage_named("turn_detection").factory.background_margin_db == 6.0

    server = ConfigManager.from_file(SERVER_PROFILE_PATH).config
    assert "vad" in server.session_nodes
    assert "turn_detection" not in server.session_nodes
    assert server.stage_named("vad").factory.silence_duration_ms == 800
    assert server.stage_named("vad").factory.background_margin_db == 6.0


def test_fused_capabilities_and_config_round_trip() -> None:
    config = AudioSessionPipelineConfig(
        model_path="unused",
        session_nodes={
            "turn_detection": {
                "stages": [TURN_DETECTION_STAGE],
                "inputs": ["input:pcm16", "input:session"],
            }
        },
        session_publish={"turn_detection": ["session"]},
    )
    rebuilt = AudioSessionPipelineConfig.model_validate(config.model_dump())
    assert rebuilt.session_graph() == config.session_graph()
    assert capabilities_of(rebuilt.session_ports()).turn_control == (
        SERVER_VAD,
        SEMANTIC_VAD,
        None,
    )
    assert SESSION_PORTS.controls == {COMMIT, CONFIGURE}
    assert SESSION_PORTS.hot_settings == {"turn_detection"}


def held_pause_then_end() -> list[TimedChunk]:
    node = Node([0.1, 0.99], eagerness="high")
    node.frames(0, [True] * 25)
    node.frames(800, [False] * 8)
    node.frames(1056, [True] * 4)
    node.frames(1184, [False] * 8)
    return node.outputs


def test_the_recognizer_keeps_a_held_pause_in_one_utterance() -> None:
    segmenter = UtteranceSegmenter()
    actions = [
        action
        for boundary in held_pause_then_end()
        for action in segmenter.step(boundary)
    ]
    assert actions == ["open", "finish"]


@pytest.mark.asyncio
async def test_the_dialog_interrupts_on_turn_start_not_on_a_held_pause() -> None:
    boundaries = held_pause_then_end()
    dialog = build({"kind": "dialog"})
    outputs = await drain(
        dialog,
        [
            transcript_chunk("question", True),
            text_chunk("Hello world.", 1),
            *boundaries[1:],
            boundaries[0],
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
        session_identity=identity,
        cancelled=threading.Event(),
        emit=outputs.append,
    )
    boundaries = held_pause_then_end()
    boundary_index = 0
    for time_ms in range(0, 1600, 20):
        while (
            boundary_index < len(boundaries)
            and boundaries[boundary_index].payload["t_ms"] <= time_ms
        ):
            hooks.append(
                boundaries[boundary_index],
                StagePayload("input", OmniRequest(None), {}),
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
            StagePayload("input", OmniRequest(None), {}),
            context,
        )
    assert [control_type(chunk) or chunk.format for chunk in outputs] == [
        TURN_QUEUED,
        UNIT_FORMAT,
    ]


class BoundaryDetector:
    """Append-scripted boundaries for the multi-process Session Graph regression."""

    def __init__(
        self, append_boundaries: Sequence[Sequence[tuple[bool, int]]], config: VADConfig
    ) -> None:
        self.config = config
        self.append_boundaries = [list(boundaries) for boundaries in append_boundaries]
        self.samples_consumed = 0
        self.is_speech = False
        self.last_speech_offset = 0

    def boundaries(self, pcm: bytes) -> list[tuple[bool, int]]:
        self.samples_consumed += len(pcm) // 2
        boundaries = self.append_boundaries.pop(0)
        for started, sample_offset in boundaries:
            self.is_speech = started
            if started:
                self.last_speech_offset = self.samples_consumed
            else:
                pass
        return boundaries

    def reset(self) -> None:
        self.samples_consumed = 0
        self.is_speech = False
        self.last_speech_offset = 0


def create_graph_scheduler(
    name: str,
    events: Queue[object],
    append_boundaries: Sequence[Sequence[tuple[bool, int]]],
) -> SessionScheduler:
    def new_detector(config: VADConfig) -> BoundaryDetector:
        return BoundaryDetector(append_boundaries, config)

    return SessionScheduler(
        TurnDetectionHooks(
            FakeEndOfTurn([0.85, 0.99]),
            "high",
            VADConfig(prefix_padding_ms=0, silence_duration_ms=160),
            new_detector,
        )
    )


@pytest.mark.asyncio
async def test_session_graph_orders_resumption_and_drains_final_eos(
    tmp_path: Path,
) -> None:
    append_boundaries = (
        ((True, 0), (False, 928 * 16)),
        ((True, 1248 * 16),),
        ((False, 1280 * 16),),
        (),
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
        factory_kwargs={"append_boundaries": append_boundaries},
    ) as (coordinator, _, _):
        coordinator.session_graph = graph
        identity = await coordinator.open_session(OmniRequest(None, {}))
        outputs = coordinator.session_outputs(identity)
        chunks = (
            audio_chunk(400, bytes(1184 * 32), 0),
            audio_chunk(1584, bytes(96 * 32), 1),
            audio_chunk(1680, bytes(256 * 32), 2),
            audio_chunk(1936, b"", 3, eos=True),
        )
        for chunk in chunks:
            await coordinator.append_session(identity, chunk)

        received = []
        while True:
            output = await asyncio.wait_for(anext(outputs), 5)
            received.append(output)
            if output.kind == "input_done" and output.input_seq == 3:
                break
            else:
                pass
        boundaries = [
            (item.payload["type"], item.payload["t_ms"])
            for item in received
            if item.kind == "data"
        ]
        assert boundaries == [(SPEECH_STARTED, 400), (SPEECH_STOPPED, 1680)]
        assert [item.input_seq for item in received if item.kind == "input_done"] == [
            0,
            1,
            2,
            3,
        ]
        await coordinator.close_session(identity)


def test_turn_detection_stage_owns_the_fused_endpoint() -> None:
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
