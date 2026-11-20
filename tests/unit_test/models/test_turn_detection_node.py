# SPDX-License-Identifier: Apache-2.0
"""The turn detection node of a Session graph: the VAD's boundaries held across a pause an end-of-turn model does not close."""

import threading
from collections import deque
from dataclasses import replace

import numpy as np
import pytest

from sglang_omni.audio.segmentation import UtteranceSegmenter
from sglang_omni.audio.semantic_vad import SemanticVADConfig
from sglang_omni.config.session_audio import AudioSessionPipelineConfig
from sglang_omni.models.qwen3_omni.payload_types import UNIT_FORMAT
from sglang_omni.models.qwen3_omni.session_input import QwenInputHooks
from sglang_omni.models.streaming_cascade.base import text_chunk, transcript_chunk
from sglang_omni.models.turn_detection.node import (
    SESSION_PORTS,
    SMART_TURN_MODEL_REPO,
    TURN_DETECTION_STAGE,
    TurnDetectionStageConfig,
    turn_detection_stage,
)
from sglang_omni.models.turn_detection.session import TurnDetectionHooks
from sglang_omni.models.vad.node import SESSION_PORTS as VAD_PORTS
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    INTERRUPT,
    PCM16,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TURN_INTERRUPTED,
    TURN_QUEUED,
    TURN_STARTED,
    control_type,
)
from sglang_omni.scheduling.session import SessionContext
from sglang_omni.serve.realtime.session_bridge import capabilities_of
from tests.unit_test.fixtures.cascade_fakes import build, drain
from tests.unit_test.fixtures.session_outputs import control_chunk

UNIT_MS = 200
UNIT = bytes(UNIT_MS * 32)
MEDIUM = SemanticVADConfig.from_eagerness("medium")


class FakeEndOfTurn:
    """Probabilities read from a script, one per scored pause; records the seconds of audio scored."""

    def __init__(self, probabilities):
        self.probabilities = list(probabilities)
        self.scored_s: list[float] = []

    def predict(self, audio: np.ndarray, sample_rate: int) -> float:
        assert sample_rate == 16000 and audio.dtype == np.float32
        self.scored_s.append(audio.size / sample_rate)
        return self.probabilities.pop(0)


class Node:
    """One session of the turn detection hooks, appended the way the Session scheduler appends."""

    def __init__(
        self, probabilities=(), *, eagerness="medium", params: dict | None = None
    ) -> None:
        self.model = FakeEndOfTurn(probabilities)
        self.hooks = TurnDetectionHooks(self.model, eagerness)
        self.outputs: list[TimedChunk] = []
        self.identity = SessionIdentity("a")
        self.context = SessionContext(
            session_identity=self.identity,
            cancelled=threading.Event(),
            emit=self.outputs.append,
        )
        self.hooks.open(self.identity, OmniRequest(None, params=params or {}))
        self.t_ms = 0.0

    def append(self, chunk: TimedChunk) -> None:
        self.hooks.append(
            chunk, StagePayload("x", OmniRequest(None), None), self.context
        )

    def audio(self, until_ms: float, *, eos: bool = False) -> None:
        """Append 200 ms units of quiet input up to `until_ms`."""
        while self.t_ms < until_ms:
            self.append(
                TimedChunk(
                    AUDIO,
                    self.t_ms,
                    UNIT_MS,
                    0,
                    UNIT,
                    format=PCM16,
                    eos=eos and self.t_ms + UNIT_MS >= until_ms,
                )
            )
            self.t_ms += UNIT_MS

    def vad(self, started: bool, t_ms: float) -> None:
        """The VAD node's boundary, as its reader receives it."""
        self.append(
            control_chunk(SPEECH_STARTED if started else SPEECH_STOPPED, t_ms=t_ms)
        )

    def control(self, kind: str, **fields) -> None:
        self.append(control_chunk(kind, **fields))

    def boundaries(self) -> list[tuple[str, float]]:
        return [(o.payload["type"], o.payload["t_ms"]) for o in self.outputs]


def test_a_confident_end_waits_for_the_confidence_silence_after_the_vad_stop():
    node = Node([0.9])
    node.audio(1000)
    node.vad(True, 800)
    node.audio(2600)
    node.vad(False, 2000)
    assert node.boundaries() == [(SPEECH_STARTED, 800)]
    assert node.model.scored_s == [1.8]
    node.audio(2800)
    assert node.boundaries() == [(SPEECH_STARTED, 800), (SPEECH_STOPPED, 2000)]
    node.audio(4000)
    assert len(node.boundaries()) == 2


def test_an_immediate_end_closes_the_turn_at_the_vad_stop():
    node = Node([0.99])
    node.audio(1000)
    node.vad(True, 800)
    node.audio(2600)
    node.vad(False, 2000)
    assert node.boundaries() == [(SPEECH_STARTED, 800), (SPEECH_STOPPED, 2000)]


def test_an_unsure_end_holds_the_turn_until_the_longest_pause():
    node = Node([0.1])
    node.audio(1000)
    node.vad(True, 800)
    node.audio(2600)
    node.vad(False, 2000)
    node.audio(3800)
    assert node.boundaries() == [(SPEECH_STARTED, 800)]
    node.audio(4000)
    assert node.boundaries() == [(SPEECH_STARTED, 800), (SPEECH_STOPPED, 2000)]


def test_speech_resuming_in_a_held_pause_continues_the_turn_and_rescores_its_end():
    node = Node([0.1, 0.99])
    node.audio(1000)
    node.vad(True, 800)
    node.audio(2600)
    node.vad(False, 2000)
    node.audio(3000)
    node.vad(True, 2700)
    node.audio(4400)
    assert node.boundaries() == [(SPEECH_STARTED, 800)]
    node.vad(False, 3800)
    assert node.boundaries() == [(SPEECH_STARTED, 800), (SPEECH_STOPPED, 3800)]
    assert node.model.scored_s == [1.8, 3.6]


def test_the_model_scores_at_most_the_last_eight_seconds_of_the_turn():
    node = Node([0.99])
    node.vad(True, 0)
    node.audio(12000)
    node.vad(False, 11400)
    assert node.model.scored_s == [MEDIUM.max_utterance_seconds]
    assert node.boundaries() == [(SPEECH_STARTED, 0), (SPEECH_STOPPED, 11400)]


@pytest.mark.parametrize(
    ("eagerness", "probability", "stop_at_ms"),
    [("high", 0.85, 2400), ("medium", 0.85, 4000), ("low", 0.95, 3200)],
)
def test_the_eagerness_preset_sets_the_silence_a_scored_pause_needs(
    eagerness, probability, stop_at_ms
):
    node = Node(
        [probability],
        params={"turn_detection": {"type": "semantic_vad", "eagerness": eagerness}},
    )
    node.audio(1000)
    node.vad(True, 800)
    node.audio(2200)
    node.vad(False, 2000)
    node.audio(stop_at_ms - UNIT_MS)
    assert node.boundaries() == [(SPEECH_STARTED, 800)]
    node.audio(stop_at_ms)
    assert node.boundaries() == [(SPEECH_STARTED, 800), (SPEECH_STOPPED, 2000)]


def test_server_vad_passes_the_boundaries_through_until_a_configure_selects_semantic_vad():
    node = Node([0.99], params={"turn_detection": {"type": "server_vad"}})
    node.audio(1000)
    node.vad(True, 800)
    node.audio(2600)
    node.vad(False, 2000)
    node.vad(True, 2700)
    assert node.boundaries() == [
        (SPEECH_STARTED, 800),
        (SPEECH_STOPPED, 2000),
        (SPEECH_STARTED, 2700),
    ]
    assert node.model.scored_s == []
    node.control(
        CONFIGURE, turn_detection={"type": "semantic_vad", "eagerness": "auto"}
    )
    node.vad(False, 3000)
    node.vad(True, 3400)
    node.audio(4000)
    node.vad(False, 3700)
    assert node.boundaries()[3:] == [(SPEECH_STARTED, 3400), (SPEECH_STOPPED, 3700)]
    node.control(CONFIGURE, turn_detection={"type": "server_vad"})
    node.vad(False, 4000)
    assert node.boundaries()[5:] == [(SPEECH_STOPPED, 4000)]
    with pytest.raises(ValueError, match="turn_detection must be an object"):
        node.control(CONFIGURE, turn_detection=3)
    with pytest.raises(ValueError, match="eagerness"):
        node.control(
            CONFIGURE, turn_detection={"type": "semantic_vad", "eagerness": "x"}
        )


def test_a_commit_forgets_the_open_turn_and_eos_closes_it():
    node = Node([0.1, 0.1])
    node.audio(1000)
    node.vad(True, 800)
    node.audio(2600)
    node.vad(False, 2000)
    node.control(COMMIT)
    node.audio(3000)
    assert node.boundaries() == [(SPEECH_STARTED, 800)]
    node.vad(True, 2900)
    node.audio(4000)
    node.vad(False, 3600)
    node.audio(4200, eos=True)
    assert node.boundaries() == [
        (SPEECH_STARTED, 800),
        (SPEECH_STARTED, 2900),
        (SPEECH_STOPPED, 3600),
    ]
    with pytest.raises(ValueError, match="pcm16"):
        node.append(TimedChunk(AUDIO, 0, 100, 2, UNIT, format="pcm_f32le"))
    node.hooks.close(node.identity)
    assert not node.hooks.sessions


def test_the_stage_is_its_own_process_and_offers_semantic_vad():
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
    assert SESSION_PORTS.turn_detection_types == ("semantic_vad",)
    declared = stage.model_dump()
    tuned = TurnDetectionStageConfig.model_validate(
        {**declared, "factory": {"eagerness": "high", "model_path": "/models"}}
    )
    assert (tuned.factory.eagerness, tuned.factory.model_path) == ("high", "/models")
    with pytest.raises(ValueError, match="eagerness"):
        TurnDetectionStageConfig.model_validate(
            {**declared, "factory": {"eagerness": "eager"}}
        )
    assert stage.factory.model_path == SMART_TURN_MODEL_REPO


def test_a_graph_with_the_node_offers_both_turn_detection_types():
    config = AudioSessionPipelineConfig(
        model_path="unused",
        session_nodes={
            "vad": {"stages": ["vad"], "inputs": ["input:pcm16", "input:session"]},
            "turn_detection": {
                "stages": ["turn_detection"],
                "inputs": ["input:pcm16", "input:session", "vad:session"],
            },
        },
        session_publish={"turn_detection": ["session"]},
    )
    assert [stage.name for stage in config.stages] == ["vad", "turn_detection"]
    assert isinstance(config.stages[1], TurnDetectionStageConfig)
    ports = config.session_ports()
    assert ports == replace(
        VAD_PORTS, turn_detection_types=("server_vad", "semantic_vad")
    )
    assert capabilities_of(ports).turn_control == ("server_vad", "semantic_vad", None)
    rebuilt = AudioSessionPipelineConfig.model_validate(config.model_dump())
    assert rebuilt.session_graph() == config.session_graph()


def held_pause_then_end() -> list[TimedChunk]:
    """The node's output for a turn that pauses, resumes and ends."""
    node = Node([0.1, 0.99])
    node.audio(1000)
    node.vad(True, 800)
    node.audio(2600)
    node.vad(False, 2000)
    node.audio(3000)
    node.vad(True, 2700)
    node.audio(4400)
    node.vad(False, 3800)
    return node.outputs


def test_the_recognizer_hears_a_held_pause_as_one_utterance():
    segmenter = UtteranceSegmenter()
    actions = [
        action for chunk in held_pause_then_end() for action in segmenter.step(chunk)
    ]
    assert actions == ["open", "finish"]


@pytest.mark.asyncio
async def test_the_dialog_interrupts_on_the_turn_start_and_not_on_a_held_pause():
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
    assert [control_type(o) for o in outputs] == [
        TURN_QUEUED,
        TURN_STARTED,
        None,
        INTERRUPT,
        TURN_INTERRUPTED,
    ]


def test_the_qwen_omni_dialog_queues_a_held_pause_as_one_turn():
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
    boundaries = deque(held_pause_then_end())
    for t_ms in range(0, 4400, 20):
        while boundaries and boundaries[0].payload["t_ms"] <= t_ms:
            hooks.append(
                boundaries.popleft(), StagePayload("in", OmniRequest(None), {}), context
            )
        hooks.append(
            TimedChunk("audio", t_ms, 20, 0, b"\x01\x00" * 320, format=PCM16),
            StagePayload("in", OmniRequest(None), {}),
            context,
        )
    assert [control_type(chunk) or chunk.format for chunk in outputs] == [
        TURN_QUEUED,
        UNIT_FORMAT,
    ]
