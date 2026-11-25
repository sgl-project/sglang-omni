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

from sglang_omni.audio.segmentation import (
    EMPTY_UTTERANCE,
    FINISH_UTTERANCE,
    OPEN_UTTERANCE,
    UtteranceSegmenter,
)
from sglang_omni.audio.semantic_vad import SemanticVADConfig
from sglang_omni.audio.smart_turn import SmartTurnEOU
from sglang_omni.audio.vad import VAD_FRAME_SAMPLES, StreamingVAD, VADConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.config.session_audio import (
    AudioSessionPipelineConfig,
    validate_fused_endpoint_graph,
)
from sglang_omni.models.qwen3_omni.payload_types import UNIT_FORMAT
from sglang_omni.models.qwen3_omni.session_input import QwenInputHooks
from sglang_omni.models.streaming_cascade.base import text_chunk, transcript_chunk
from sglang_omni.models.streaming_cascade.config import CascadeSessionNodeConfig
from sglang_omni.models.turn_detection.node import (
    SMART_TURN_MODEL_REPO,
    TURN_DETECTION_STAGE,
    turn_detection_stage,
)
from sglang_omni.models.turn_detection.session import TurnDetectionHooks
from sglang_omni.models.turn_detection.session import (
    create_scheduler as create_turn_scheduler,
)
from sglang_omni.models.vad.node import vad_stage
from sglang_omni.models.vad.session import SileroSpeechDetector, VADHooks
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
    SEGMENTATION_MODE,
    SEMANTIC_VAD,
    SERVER_VAD,
    SESSION_CONTROL_INTERNAL,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)
from sglang_omni.scheduling.session import (
    ChunkEmitter,
    SessionContext,
    SessionHooks,
    SessionScheduler,
)
from sglang_omni.serve.realtime.session_bridge import capabilities_of
from tests.unit_test.fixtures.cascade_fakes import build, drain
from tests.unit_test.fixtures.session_outputs import control_chunk
from tests.unit_test.fixtures.session_pipeline import pipeline

SAMPLE_RATE_HZ = 16000
PROFILE_PATH = (
    "examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-base-clone/launch.yaml"
)
SERVER_PROFILE_PATH = "examples/streaming_cascade/profiles/qwen3-asr-qwen3.8-27b-vivian-tts-tool/launch.yaml"


def fake_silero_module() -> types.ModuleType:
    silero_module = types.ModuleType("silero_vad")

    def load_silero_vad(*, onnx: bool) -> object:
        assert onnx
        return object()

    silero_module.load_silero_vad = load_silero_vad
    return silero_module


def deterministic_speech_probability(
    detector: StreamingVAD, frame: np.ndarray
) -> float:
    return float(np.any(frame))


@pytest.fixture(autouse=True)
def deterministic_silero(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "silero_vad", fake_silero_module())
    monkeypatch.setattr(StreamingVAD, "infer", deterministic_speech_probability)


class FakeEndOfTurn:
    def __init__(self, probabilities: Sequence[float]) -> None:
        self.probabilities = deque(probabilities)
        self.scored_seconds: list[float] = []
        self.scored_pcm: list[bytes] = []

    def predict(self, audio: np.ndarray, sample_rate: int) -> float:
        assert sample_rate == SAMPLE_RATE_HZ
        assert audio.dtype == np.float32
        self.scored_seconds.append(audio.size / sample_rate)
        self.scored_pcm.append(np.rint(audio * 32768).astype("<i2").tobytes())
        return self.probabilities.popleft()


def pcm_samples(speech: bool, sample_count: int) -> bytes:
    amplitude = 12000 if speech else 0
    return np.full(sample_count, amplitude, dtype="<i2").tobytes()


def pcm_amplitude(amplitude: int, sample_count: int) -> bytes:
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
            (chunk.payload["type"], chunk.payload["t_ms"])
            for chunk in self.outputs
            if control_type(chunk) in (SPEECH_STARTED, SPEECH_STOPPED)
        ]


def replay_partitioned_audio(pcm: bytes, partition_samples: Sequence[int]) -> Node:
    node = Node([0.1, 0.99], eagerness="high")
    total_samples = len(pcm) // 2
    sample_offset = 0
    partition_index = 0
    while sample_offset < total_samples:
        sample_count = min(
            partition_samples[partition_index % len(partition_samples)],
            total_samples - sample_offset,
        )
        start_byte = sample_offset * 2
        end_byte = (sample_offset + sample_count) * 2
        node.audio(
            sample_offset * 1000 / SAMPLE_RATE_HZ,
            pcm[start_byte:end_byte],
            eos=sample_offset + sample_count == total_samples,
        )
        sample_offset += sample_count
        partition_index += 1
    return node


@pytest.mark.parametrize("resume_in_deadline_frame", [False, True])
def test_resumption_before_or_in_the_deadline_frame_keeps_the_turn_open(
    resume_in_deadline_frame: bool,
) -> None:
    node = Node([0.85], eagerness="high")
    node.frames(0, [True] * 29)
    node.frames(928, [False] * 8)
    assert node.session.pause_start_ms == 928
    assert node.session.required_silence_ms == 320

    if resume_in_deadline_frame:
        node.frames(1184, [False])
        node.frames(1216, [True])
    else:
        node.frames(1184, [True])
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert node.session.is_open


def test_deadline_commits_before_later_resumption_inside_one_200_ms_append() -> None:
    node = Node([0.85], eagerness="high")
    node.frames(0, [True] * 29)
    node.frames(928, [False] * 8)
    silence_and_resumption = pcm_frames([False] * 2 + [True] * 4) + pcm_samples(
        True, 128
    )
    node.audio(1184, silence_and_resumption, eos=True)
    assert len(silence_and_resumption) // 2 == 200 * 16
    assert node.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 928),
        (SPEECH_STARTED, 1248),
        (SPEECH_STOPPED, 1376),
    ]
    assert len(node.model.scored_pcm) == 1


def test_resumed_utterance_ownership_is_invariant_to_pcm_append_partition() -> None:
    first_speech = pcm_amplitude(7000, 28 * VAD_FRAME_SAMPLES)
    silence = pcm_amplitude(0, 6 * VAD_FRAME_SAMPLES)
    resumed_speech = pcm_amplitude(24000, VAD_FRAME_SAMPLES + 128)
    resumed_append = pcm_amplitude(0, 5 * VAD_FRAME_SAMPLES) + resumed_speech
    pcm = first_speech + silence + resumed_append
    total_samples = len(pcm) // 2
    partitions = (
        (VAD_FRAME_SAMPLES,),
        (80 * 16,),
        (200 * 16,),
        (1, 511, 13, 1000, 2048, 7),
        (total_samples,),
    )
    expected_utterances = [
        first_speech + silence + resumed_append[: 128 * 16 * 2],
        resumed_speech,
    ]
    expected_boundaries = [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 896 * 16),
        (SPEECH_STARTED, 1248 * 16),
        (SPEECH_STOPPED, 1280 * 16),
    ]
    expected_scores: list[bytes] | None = None

    for partition in partitions:
        node = Node([0.85], eagerness="high")
        sample_offset = 0
        partition_index = 0
        while sample_offset < total_samples:
            sample_count = min(
                partition[partition_index % len(partition)],
                total_samples - sample_offset,
            )
            start_byte = sample_offset * 2
            end_byte = (sample_offset + sample_count) * 2
            node.audio(
                sample_offset * 1000 / SAMPLE_RATE_HZ,
                pcm[start_byte:end_byte],
                eos=sample_offset + sample_count == total_samples,
            )
            sample_offset += sample_count
            partition_index += 1

        segmenter = UtteranceSegmenter()
        utterances: list[bytes] = []
        current_utterance: bytearray | None = None
        for chunk in node.outputs:
            actions = segmenter.step(chunk)
            if chunk.eos:
                actions.extend(segmenter.finish())
            else:
                pass
            for action in actions:
                if action == OPEN_UTTERANCE:
                    current_utterance = bytearray()
                elif action == FINISH_UTTERANCE:
                    assert current_utterance is not None
                    utterances.append(bytes(current_utterance))
                    current_utterance = None
                elif isinstance(action, TimedChunk):
                    assert current_utterance is not None
                    assert isinstance(action.payload, bytes)
                    current_utterance.extend(action.payload)
                else:
                    pass

        boundaries = [
            (event_type, round(time_ms * SAMPLE_RATE_HZ / 1000))
            for event_type, time_ms in node.boundaries()
        ]
        forwarded_pcm = b"".join(
            chunk.payload
            for chunk in node.outputs
            if chunk.modality == AUDIO and isinstance(chunk.payload, bytes)
        )
        assert boundaries == expected_boundaries
        assert forwarded_pcm == pcm
        assert utterances == expected_utterances
        if expected_scores is None:
            expected_scores = node.model.scored_pcm
        else:
            assert node.model.scored_pcm == expected_scores


def test_no_resumption_commits_once_at_the_first_frame_reaching_the_deadline() -> None:
    node = Node([0.85], eagerness="high")
    node.frames(0, [True] * 29)
    node.frames(928, [False] * 8)
    node.frames(1184, [False] * 2)
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
    pcm = pcm_frames([True] * 6 + [False] * 5 + [True] * 2 + [False] * 7)
    total_samples = len(pcm) // 2
    expected_scores = [
        pcm_frames([True] * 6 + [False] * 5),
        pcm_frames([True] * 6 + [False] * 5 + [True] * 2 + [False] * 5),
    ]
    partitions = (
        (VAD_FRAME_SAMPLES,),
        (80 * 16,),
        (200 * 16,),
        (1, 511, 13, 1000, 2048, 7),
        (total_samples,),
    )
    expected_boundaries = [(SPEECH_STARTED, 0), (SPEECH_STOPPED, 416)]

    for partition in partitions:
        node = replay_partitioned_audio(pcm, partition)
        assert node.boundaries() == expected_boundaries
        assert [
            (event_type, round(time_ms * SAMPLE_RATE_HZ / 1000))
            for event_type, time_ms in node.boundaries()
        ] == [
            (SPEECH_STARTED, 0),
            (SPEECH_STOPPED, 416 * SAMPLE_RATE_HZ // 1000),
        ]
        assert (
            sum(event_type == SPEECH_STARTED for event_type, _ in node.boundaries())
            == 1
        )
        assert node.model.scored_seconds == [0.352, 0.576]
        assert node.model.scored_pcm == expected_scores
        segmentation_modes = [
            chunk for chunk in node.outputs if control_type(chunk) == SEGMENTATION_MODE
        ]
        assert [chunk.payload["enabled"] for chunk in segmentation_modes] == [True]
        assert (
            b"".join(
                chunk.payload
                for chunk in node.outputs
                if chunk.modality == AUDIO and isinstance(chunk.payload, bytes)
            )
            == pcm
        )
        assert not node.session.is_open


@pytest.mark.parametrize(
    "large_append", [False, True], ids=["streaming", "large-append"]
)
def test_semantic_scoring_uses_eight_second_history_at_candidate_frame(
    large_append: bool,
) -> None:
    node = Node([0.85] if large_append else [0.99], eagerness="high")
    if large_append:
        node.frames(0, [True] * 320)
        node.audio(10240, pcm_frames([False] * 11 + [True] * 2), eos=True)
    else:
        for start_ms in range(0, 12000, 200):
            node.audio(start_ms, pcm_samples(True, 200 * 16))
        node.audio(12000, pcm_samples(False, 200 * 16))
        node.audio(12200, pcm_samples(False, 24 * 16))
    assert node.model.scored_seconds == [SemanticVADConfig.max_utterance_seconds]
    assert node.model.scored_pcm == [pcm_frames([True] * 245 + [False] * 5)]
    if large_append:
        assert node.boundaries() == [
            (SPEECH_STARTED, 0),
            (SPEECH_STOPPED, 10240),
            (SPEECH_STARTED, 10592),
            (SPEECH_STOPPED, 10656),
        ]
    else:
        assert node.boundaries()[-1] == (SPEECH_STOPPED, 12000)


def test_candidate_scoring_slices_audio_before_the_reported_turn_start() -> None:
    node = Node(
        [0.85],
        eagerness="high",
        vad_config=VADConfig(
            prefix_padding_ms=100,
            silence_duration_ms=160,
            background_margin_db=None,
        ),
    )
    node.frames(0, [False] * 20 + [True] * 80 + [False] * 5)
    expected_pcm = pcm_samples(False, 100 * 16) + pcm_frames([True] * 80 + [False] * 5)
    assert node.boundaries() == [(SPEECH_STARTED, 540)]
    assert node.model.scored_pcm == [expected_pcm]


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


def test_server_vad_matches_standalone_vad_across_append_partitions() -> None:
    pcm = pcm_frames([True] * 6 + [False] * 7 + [True] * 4 + [False] * 8)
    total_samples = len(pcm) // 2
    config = VADConfig(
        prefix_padding_ms=0,
        silence_duration_ms=160,
        background_margin_db=None,
    )
    partitions = (
        (VAD_FRAME_SAMPLES,),
        (80 * 16,),
        (200 * 16,),
        (1, 511, 13, 1000, 2048, 7),
        (total_samples,),
    )
    expected = [
        (SPEECH_STARTED, 0),
        (SPEECH_STOPPED, 192),
        (SPEECH_STARTED, 416),
        (SPEECH_STOPPED, 544),
    ]

    for partition_index, partition in enumerate(partitions):
        fused = Node(mode=SERVER_VAD, vad_config=config)
        standalone_hooks = VADHooks(config, SileroSpeechDetector)
        standalone_identity = SessionIdentity(f"standalone-{partition_index}")
        standalone_outputs: list[TimedChunk] = []
        standalone_context = SessionContext(
            session_identity=standalone_identity,
            cancelled=threading.Event(),
            emit=standalone_outputs.append,
        )
        standalone_hooks.open(standalone_identity, OmniRequest(None))
        sample_offset = 0
        partition_offset = 0
        while sample_offset < total_samples:
            sample_count = min(
                partition[partition_offset % len(partition)],
                total_samples - sample_offset,
            )
            start_byte = sample_offset * 2
            end_byte = (sample_offset + sample_count) * 2
            chunk = audio_chunk(
                sample_offset * 1000 / SAMPLE_RATE_HZ,
                pcm[start_byte:end_byte],
                partition_offset,
                eos=sample_offset + sample_count == total_samples,
            )
            fused.append(chunk)
            standalone_hooks.append(
                chunk,
                StagePayload("input", OmniRequest(None), None),
                standalone_context,
            )
            sample_offset += sample_count
            partition_offset += 1
        standalone_boundaries = [
            (chunk.payload["type"], chunk.payload["t_ms"])
            for chunk in standalone_outputs
        ]
        assert fused.boundaries() == expected
        assert standalone_boundaries == expected
        assert fused.model.scored_pcm == []
        standalone_hooks.close(standalone_identity)


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
    first_pcm = pcm_frames([True] * 5)
    manual_pcm = pcm_samples(False, 20 * 16)
    node.audio(0, first_pcm)
    node.control(CONFIGURE, turn_detection=None)
    node.audio(160, manual_pcm)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    assert node.session.is_open

    # Replay the actual ordered PCM and controls, rather than treating the
    # segmenter's PCM actions as lifecycle strings or selecting the first span.
    segmenter = UtteranceSegmenter()
    actions = [action for chunk in node.outputs for action in segmenter.step(chunk)]
    assert [action for action in actions if isinstance(action, str)] == [OPEN_UTTERANCE]
    assert (
        b"".join(action.payload for action in actions if isinstance(action, TimedChunk))
        == first_pcm + manual_pcm
    )
    assert segmenter.is_open

    previous_outputs = len(node.outputs)
    node.control(COMMIT)
    assert not node.session.is_open
    assert [
        action
        for chunk in node.outputs[previous_outputs:]
        for action in segmenter.step(chunk)
    ] == [FINISH_UTTERANCE]
    assert not segmenter.is_open
    commit_outputs = [chunk for chunk in node.outputs if control_type(chunk) == COMMIT]
    assert len(commit_outputs) == 1
    assert commit_outputs[0].format == SESSION_CONTROL_INTERNAL
    node.control(CONFIGURE, turn_detection={"type": SEMANTIC_VAD, "eagerness": "high"})
    node.frames(1000, [True] * 5)
    node.frames(1160, [False] * 5)
    node.frames(1320, [False, False])
    configure_outputs = [
        chunk for chunk in node.outputs if control_type(chunk) == CONFIGURE
    ]
    assert len(configure_outputs) == 2
    assert all(chunk.format == SESSION_CONTROL_INTERNAL for chunk in configure_outputs)
    segmentation_modes = [
        chunk for chunk in node.outputs if control_type(chunk) == SEGMENTATION_MODE
    ]
    assert [chunk.payload["enabled"] for chunk in segmentation_modes] == [
        True,
        False,
        True,
    ]
    assert node.boundaries() == [
        (SPEECH_STARTED, 0),
        (SPEECH_STARTED, 1000),
        (SPEECH_STOPPED, 1160),
    ]


def test_manual_eos_flushes_pcm_without_an_automatic_speech_stop() -> None:
    node = Node([0.99], eagerness="high")
    first_pcm = pcm_frames([True] * 5)
    manual_pcm = pcm_samples(False, 20 * 16)
    node.audio(0, first_pcm)
    node.control(CONFIGURE, turn_detection=None)
    node.audio(160, manual_pcm, eos=True)
    assert node.boundaries() == [(SPEECH_STARTED, 0)]
    eos_markers = [chunk for chunk in node.outputs if chunk.eos]
    assert len(eos_markers) == 1
    assert eos_markers[0].modality == AUDIO and eos_markers[0].payload == b""

    segmenter = UtteranceSegmenter()
    actions = []
    for chunk in node.outputs:
        actions.extend(segmenter.step(chunk))
        if chunk.eos:
            actions.extend(segmenter.finish())
    assert [action for action in actions if isinstance(action, str)] == [
        OPEN_UTTERANCE,
        FINISH_UTTERANCE,
    ]
    assert (
        b"".join(action.payload for action in actions if isinstance(action, TimedChunk))
        == first_pcm + manual_pcm
    )
    assert segmenter.finish() == []


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


@pytest.mark.parametrize("entry_point", ["audio-session", "cascade"])
def test_graph_configs_reject_legacy_split_endpoint(entry_point: str) -> None:
    with pytest.raises(ValueError, match="split VAD -> turn_detection"):
        if entry_point == "audio-session":
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
        else:
            config = ConfigManager.from_file(PROFILE_PATH).config
            config.stages.append(vad_stage())
            config.session_nodes["audio_vad"] = CascadeSessionNodeConfig(
                stages=["vad"], inputs=["input:pcm16"]
            )
            config.session_nodes["turn_detection"].inputs.append("audio_vad:session")
            config.check_graph()


@pytest.mark.parametrize(
    ("consumer_inputs", "error_message"),
    [
        pytest.param(
            ("input:pcm16", "turn_detection:session"),
            "combines input PCM with turn_detection controls",
            id="competing-pcm",
        ),
        pytest.param(
            (
                "input:session",
                "turn_detection:pcm16",
                "turn_detection:session_internal",
            ),
            "combines input session controls with turn_detection:session_internal",
            id="duplicate-session-controls",
        ),
    ],
)
def test_fused_graph_rejects_competing_input_edges(
    consumer_inputs: tuple[str, ...], error_message: str
) -> None:
    graph = SessionGraph(
        nodes={
            "turn_detection": SessionNode(
                stages=(TURN_DETECTION_STAGE,),
                inputs=("input:pcm16", "input:session"),
            ),
            "recognizer": SessionNode(stages=("recognizer",), inputs=consumer_inputs),
        },
        publish={"turn_detection": ("session",)},
    )
    with pytest.raises(ValueError, match=error_message):
        validate_fused_endpoint_graph(graph)


def test_profiles_keep_their_fused_and_standalone_vad_defaults() -> None:
    semantic = ConfigManager.from_file(PROFILE_PATH).config
    assert [stage.name for stage in semantic.stages].count("turn_detection") == 1
    assert "vad" not in semantic.session_nodes
    assert semantic.session_nodes["turn_detection"].inputs == [
        "input:pcm16",
        "input:session",
    ]
    assert semantic.session_nodes["asr"].inputs == [
        "turn_detection:pcm16",
        "turn_detection:session",
        "turn_detection:session_internal",
    ]
    assert "input:session" not in semantic.session_nodes["dialog"].inputs
    assert "turn_detection:session_internal" in semantic.session_nodes["dialog"].inputs
    assert semantic.session_publish["turn_detection"] == ["session"]
    assert semantic.session_ports().output is not None
    assert semantic.session_ports().output.format != PCM16
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
    assert rebuilt.session_ports().output is None


def held_pause_then_end() -> list[TimedChunk]:
    node = Node([0.1, 0.99], eagerness="high")
    node.frames(0, [True] * 25)
    node.frames(800, [False] * 8)
    node.frames(1056, [True] * 4)
    node.frames(1184, [False] * 8)
    return node.outputs


@pytest.mark.asyncio
async def test_the_dialog_interrupts_on_turn_start_not_on_a_held_pause() -> None:
    boundaries = [
        chunk
        for chunk in held_pause_then_end()
        if control_type(chunk) in (SPEECH_STARTED, SPEECH_STOPPED)
    ]
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
    fused_output = held_pause_then_end()
    for chunk in fused_output:
        hooks.append(chunk, StagePayload("input", OmniRequest(None), {}), context)
    expected_pcm = b"".join(
        chunk.payload
        for chunk in fused_output
        if chunk.modality == AUDIO and isinstance(chunk.payload, bytes)
    )
    units = [chunk for chunk in outputs if chunk.format == UNIT_FORMAT]
    assert [control_type(chunk) or chunk.format for chunk in outputs] == [
        TURN_QUEUED,
        UNIT_FORMAT,
    ]
    assert len(units) == 1
    assert units[0].payload["pcm"] == expected_pcm


class CapturingRecognizerHooks(SessionHooks):
    def __init__(self, name: str, events: Queue[object]) -> None:
        self.name = name
        self.events = events
        self.segmenters: dict[SessionIdentity, UtteranceSegmenter] = {}
        self.utterances: dict[SessionIdentity, bytearray | None] = {}

    def open(self, session_identity: SessionIdentity, request: OmniRequest) -> None:
        self.segmenters[session_identity] = UtteranceSegmenter()
        self.utterances[session_identity] = None

    def close(self, session_identity: SessionIdentity) -> None:
        del self.segmenters[session_identity]
        del self.utterances[session_identity]

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        event_type = control_type(chunk)
        if chunk.modality == AUDIO:
            assert isinstance(chunk.payload, bytes)
            self.events.put(
                (
                    "recognizer_delivery",
                    self.name,
                    "audio",
                    chunk.t_start_ms,
                    chunk.duration_ms,
                    chunk.payload,
                )
            )
        elif event_type in (SPEECH_STARTED, SPEECH_STOPPED):
            self.events.put(
                (
                    "recognizer_delivery",
                    self.name,
                    "control",
                    event_type,
                    chunk.t_start_ms,
                )
            )
        else:
            pass
        segmenter = self.segmenters[context.session_identity]
        actions = segmenter.step(chunk)
        if chunk.eos:
            actions.extend(segmenter.finish())
        else:
            pass
        for action in actions:
            if action == OPEN_UTTERANCE:
                self.utterances[context.session_identity] = bytearray()
            elif action == FINISH_UTTERANCE:
                utterance = self.utterances[context.session_identity]
                assert utterance is not None
                self.events.put(("recognizer_utterance", self.name, bytes(utterance)))
                self.utterances[context.session_identity] = None
            elif action == EMPTY_UTTERANCE:
                self.events.put(("recognizer_utterance", self.name, b""))
            else:
                assert isinstance(action, TimedChunk)
                utterance = self.utterances[context.session_identity]
                assert utterance is not None and isinstance(action.payload, bytes)
                utterance.extend(action.payload)
        if chunk.eos:
            self.events.put(("recognizer_complete", self.name))
        else:
            pass
        payload.data = {"accepted": True}
        return payload


class CapturingDialogHooks(QwenInputHooks):
    def __init__(self, events: Queue[object]) -> None:
        super().__init__(
            max_utterance_seconds=8,
            max_queued_turns=4,
            max_utterance_frames=4,
        )
        self.events = events

    def attach_output_stream(
        self, session_identity: SessionIdentity, emit: ChunkEmitter
    ) -> None:
        def capture(chunk: TimedChunk) -> None:
            if chunk.format == UNIT_FORMAT:
                self.events.put(
                    (
                        "dialog_unit",
                        "dialog",
                        chunk.turn,
                        chunk.payload["pcm"],
                        chunk.payload["carried_turns"],
                    )
                )
            emit(chunk)

        super().attach_output_stream(session_identity, capture)

    def append(
        self, chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        payload = super().append(chunk, payload, context)
        if chunk.eos:
            self.events.put(("dialog_complete", "dialog"))
        return payload


def create_graph_turn_scheduler(name: str, events: Queue[object]) -> SessionScheduler:
    sys.modules["silero_vad"] = fake_silero_module()
    StreamingVAD.infer = deterministic_speech_probability
    events.put(("scheduler_ready", name))
    return SessionScheduler(
        TurnDetectionHooks(
            FakeEndOfTurn([0.85]),
            "high",
            VADConfig(
                prefix_padding_ms=0,
                silence_duration_ms=160,
                background_margin_db=None,
            ),
            SileroSpeechDetector,
        )
    )


def create_graph_recognizer_scheduler(
    name: str, events: Queue[object]
) -> SessionScheduler:
    return SessionScheduler(CapturingRecognizerHooks(name, events))


def create_graph_dialog_scheduler(name: str, events: Queue[object]) -> SessionScheduler:
    assert name == "dialog"
    return SessionScheduler(CapturingDialogHooks(events))


@pytest.mark.asyncio
async def test_session_graph_delivers_resumed_pcm_to_second_recognizer_and_dialog_utterance(
    tmp_path: Path,
) -> None:
    graph = SessionGraph(
        nodes={
            "turn_detection": SessionNode(
                stages=("turn_detection",),
                inputs=("input:pcm16", "input:session"),
            ),
            "recognizer": SessionNode(
                stages=("recognizer",),
                inputs=(
                    "turn_detection:pcm16",
                    "turn_detection:session",
                    "turn_detection:session_internal",
                ),
            ),
            "dialog": SessionNode(
                stages=("dialog",),
                inputs=(
                    "turn_detection:pcm16",
                    "turn_detection:session",
                    "turn_detection:session_internal",
                ),
            ),
        },
        publish={"turn_detection": ("session",)},
    )
    async with pipeline(
        tmp_path,
        names=["turn_detection", "recognizer", "dialog"],
        factories={
            "turn_detection": f"{__name__}.create_graph_turn_scheduler",
            "recognizer": f"{__name__}.create_graph_recognizer_scheduler",
            "dialog": f"{__name__}.create_graph_dialog_scheduler",
        },
    ) as (coordinator, events, _):
        coordinator.session_graph = graph
        identity = await coordinator.open_session(OmniRequest(None, {}))
        outputs = coordinator.session_outputs(identity)
        first_speech = pcm_amplitude(7000, 28 * VAD_FRAME_SAMPLES)
        silence = pcm_frames([False] * 6)
        resumed_speech = pcm_amplitude(24000, VAD_FRAME_SAMPLES + 128)
        resume_append = pcm_frames([False] * 5) + resumed_speech
        chunks = (
            audio_chunk(0, first_speech, 0),
            audio_chunk(896, silence, 1),
            audio_chunk(1088, resume_append, 2),
            audio_chunk(1288, b"", 3, eos=True),
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
        assert [
            (item.payload["type"], item.payload["t_ms"])
            for item in received
            if item.kind == "data"
        ] == [
            (SPEECH_STARTED, 0),
            (SPEECH_STOPPED, 896),
            (SPEECH_STARTED, 1248),
            (SPEECH_STOPPED, 1280),
        ]
        assert [item.input_seq for item in received if item.kind == "input_done"] == [
            0,
            1,
            2,
            3,
        ]
        delivered_events = []
        completed_stages: set[str] = set()
        while completed_stages != {"recognizer", "dialog"}:
            event = events.get(timeout=5)
            delivered_events.append(event)
            if event[0] in ("recognizer_complete", "dialog_complete"):
                completed_stages.add(event[1])
            else:
                pass
        recognizer_pcm = [
            event[2] for event in delivered_events if event[0] == "recognizer_utterance"
        ]
        dialog_units = [
            (event[2], event[3], event[4])
            for event in delivered_events
            if event[0] == "dialog_unit"
        ]
        recognizer_delivery = [
            event
            for event in delivered_events
            if event[0] == "recognizer_delivery" and event[1] == "recognizer"
        ]
        delivery_order = [
            (
                ("audio", event[3], event[4])
                if event[2] == "audio"
                else ("control", event[3], event[4])
            )
            for event in recognizer_delivery
        ]
        append_delivery = [
            event
            for event in recognizer_delivery
            if event[2] == "audio" and event[3] >= 1088
        ]
        append_audio_spans = [
            (round(event[3] * SAMPLE_RATE_HZ / 1000), len(event[5]) // 2)
            for event in append_delivery
        ]
        assert append_audio_spans == [
            (1088 * 16, 128 * 16),
            (1216 * 16, 64 * 16),
            (1280 * 16, 128),
            (1288 * 16, 0),
        ]
        assert b"".join(event[5] for event in append_delivery) == resume_append
        causal_tail = [
            item
            for item in delivery_order
            if (item[0] == "audio" and item[1] >= 1088)
            or (
                item[0] == "control"
                and item[1] in (SPEECH_STARTED, SPEECH_STOPPED)
                and item[2] >= 896
            )
        ]
        assert causal_tail == [
            ("audio", 1088, 128),
            ("control", SPEECH_STOPPED, 896),
            ("audio", 1216, 64),
            ("control", SPEECH_STARTED, 1248),
            ("audio", 1280, 8),
            ("control", SPEECH_STOPPED, 1280),
            ("audio", 1288, 0),
        ]
        expected_first_utterance = (
            first_speech + silence + resume_append[: 128 * 16 * 2]
        )
        assert recognizer_pcm == [expected_first_utterance, resumed_speech]
        # The fake graph has no model reply feedback. The first inference unit
        # is interrupted before its reply starts; carrying it into the second
        # unit is intentional and must not be mistaken for segmentation leakage.
        assert dialog_units == [
            (1, expected_first_utterance, []),
            (2, expected_first_utterance + resumed_speech, [1]),
        ]
        await coordinator.close_session(identity)
