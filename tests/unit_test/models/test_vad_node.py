# SPDX-License-Identifier: Apache-2.0
"""The VAD node of a Session graph: one detector per session, boundaries on the input timeline."""

import threading

import numpy as np
import pytest

from sglang_omni.audio.vad import VADConfig
from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.models.vad.node import (
    SESSION_PORTS,
    VAD_STAGE,
    VADStageConfig,
    vad_stage,
)
from sglang_omni.models.vad.session import (
    SileroSpeechDetector,
    VADHooks,
    create_scheduler,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONFIGURE,
    PCM16,
    SPEECH_STARTED,
    SPEECH_STOPPED,
)
from sglang_omni.scheduling.session import SessionContext
from tests.unit_test.fixtures.session_outputs import control_chunk

QUIET = bytes(100 * 32)


class ScriptedDetector:
    """Boundaries read from a script, one entry per audio chunk."""

    def __init__(self, script, config):
        self.script = list(script)
        self.config = config
        self.is_speech = False
        self.last_speech_offset = 0
        self.resets = 0

    def boundaries(self, pcm):
        found = self.script.pop(0)
        for started, offset in found:
            self.is_speech = started
            self.last_speech_offset = offset
        return found

    def reset(self):
        self.resets += 1
        self.is_speech = False


class Node:
    """One session of the VAD hooks, appended the way the Session scheduler appends."""

    def __init__(self, hooks: VADHooks, name: str, params: dict | None = None) -> None:
        self.hooks = hooks
        self.outputs: list[TimedChunk] = []
        self.identity = SessionIdentity(name)
        self.context = SessionContext(
            session_identity=self.identity,
            cancelled=threading.Event(),
            emit=self.outputs.append,
        )
        self.hooks.open(self.identity, OmniRequest(None, params=params or {}))

    @property
    def detector(self):
        return self.hooks.sessions[self.identity].detector

    def append(self, chunk: TimedChunk) -> None:
        payload = StagePayload("x", OmniRequest(None), None)
        self.hooks.append(chunk, payload, self.context)

    def audio(self, t_ms: float, *, eos: bool = False) -> None:
        self.append(TimedChunk(AUDIO, t_ms, 100, 0, QUIET, format=PCM16, eos=eos))

    def control(self, kind: str, **fields) -> None:
        self.append(control_chunk(kind, **fields))

    def boundaries(self) -> list[tuple[str, float, float]]:
        return [
            (o.payload["type"], o.payload["t_ms"], o.t_start_ms) for o in self.outputs
        ]


def scripted(script):
    """VAD hooks whose sessions detect from `script`, and the detectors they built."""
    detectors = []

    def new_detector(config: VADConfig) -> ScriptedDetector:
        detectors.append(ScriptedDetector(script, config))
        return detectors[-1]

    return VADHooks(VADConfig(), new_detector), detectors


def test_boundaries_follow_the_input_timeline_and_commit_resets_the_detector():
    """Empty appends and the eos tick run no detection: the script has one entry per audio unit."""
    hooks, detectors = scripted(
        [[(True, 1600)], [], [(False, 4800), (True, 6400)], [(True, 0)], []]
    )
    node = Node(hooks, "a")
    node.append(TimedChunk(AUDIO, 1000, 0, 0, b"", format=PCM16))
    for t_ms in (1000, 1100, 1200):
        node.audio(t_ms)
    node.control(COMMIT)
    node.audio(5000)
    node.audio(5100)
    node.append(TimedChunk(AUDIO, 5200, 0, 1, b"", format=PCM16, eos=True))
    assert node.boundaries() == [
        (SPEECH_STARTED, 1100, 1100),
        (SPEECH_STOPPED, 1300, 1300),
        (SPEECH_STARTED, 1400, 1400),
        (SPEECH_STARTED, 5000, 5000),
        (SPEECH_STOPPED, 5000, 5000),
    ]
    assert detectors[0].resets == 1 and detectors[0].script == []
    with pytest.raises(ValueError, match="pcm16"):
        node.append(TimedChunk(AUDIO, 0, 100, 2, QUIET, format="pcm_f32le"))
    hooks.close(node.identity)
    assert not hooks.sessions


def test_configure_changes_the_turn_detection_of_one_session_only():
    hooks, _ = scripted([])
    a = Node(hooks, "a")
    b = Node(hooks, "b", params={"turn_detection": {"threshold": 0.9}})
    assert (a.detector.config, b.detector.config.threshold) == (VADConfig(), 0.9)
    turn_detection = {"type": "server_vad", "silence_duration_ms": 800}
    a.control(CONFIGURE, instructions="Be brief.", turn_detection=turn_detection)
    assert a.detector.config == VADConfig(silence_duration_ms=800)
    assert b.detector.config == VADConfig(threshold=0.9)
    a.control(CONFIGURE, instructions="Be terse.")
    assert a.detector.config == VADConfig(silence_duration_ms=800)
    with pytest.raises(ValueError, match="turn_detection must be an object"):
        a.control(CONFIGURE, turn_detection=3)


def test_the_silero_detector_hears_no_speech_in_silence():
    hooks = create_scheduler(silence_duration_ms=800).session_hooks
    assert isinstance(hooks, VADHooks)
    node = Node(hooks, "a")
    assert node.detector.config == VADConfig(silence_duration_ms=800)
    node.audio(0)
    node.audio(100, eos=True)
    assert node.outputs == []
    hooks.close(node.identity)


def test_the_vad_stage_is_its_own_process():
    stage = vad_stage()
    assert (stage.name, stage.process, stage.terminal) == (VAD_STAGE, VAD_STAGE, True)
    assert stage.factory_path == "sglang_omni.models.vad.session.create_scheduler"
    assert SESSION_PORTS.output is None
    assert SESSION_PORTS.controls == {COMMIT, CONFIGURE}
    assert SESSION_PORTS.hot_settings == {"turn_detection"}
    declared = stage.model_dump()
    tuned = VADStageConfig.model_validate(
        {**declared, "factory": {"silence_duration_ms": 1200}}
    )
    assert tuned.factory.silence_duration_ms == 1200
    with pytest.raises(ValueError, match="threshold"):
        VADStageConfig.model_validate({**declared, "factory": {"threshold": 1.5}})


@pytest.mark.parametrize(
    "factory, silence_duration_ms",
    [
        ({}, 800),
        ({"threshold": 0.6}, 800),
        ({"silence_duration_ms": 500}, 500),
    ],
)
def test_the_vad_stage_waits_800_ms_of_silence_unless_its_factory_sets_another(
    factory, silence_duration_ms
):
    stage = VADStageConfig.model_validate(
        {**vad_stage().model_dump(), "factory": factory}
    )
    assert stage.factory.silence_duration_ms == silence_duration_ms


def test_a_null_turn_detection_turns_the_detector_off_until_a_later_configure():
    hooks, detectors = scripted([[(True, 1600)], [(False, 0)]])
    node = Node(hooks, "a", params={"turn_detection": None})
    node.audio(1000)
    node.audio(1100, eos=True)
    assert node.outputs == []
    node.control(CONFIGURE, turn_detection={"silence_duration_ms": 800})
    assert node.detector.config == VADConfig(silence_duration_ms=800)
    node.audio(2000)
    assert node.boundaries() == [(SPEECH_STARTED, 2100, 2100)]
    node.control(CONFIGURE, turn_detection=None)
    node.audio(2100, eos=True)
    assert node.boundaries() == [(SPEECH_STARTED, 2100, 2100)]
    assert detectors[0].script == [[(False, 0)]] and detectors[0].resets == 1


class AnySoundIsSpeech(SileroSpeechDetector):
    """The production detector with Silero's score replaced: any non-zero frame is speech."""

    def infer(self, frame: np.ndarray) -> float:
        return 0.9 if np.abs(frame).max() > 0 else 0.0


def tone(amplitude: int) -> bytes:
    return np.full(1600, amplitude, dtype="<i2").tobytes()


@pytest.mark.parametrize(
    "margin_db, boundaries",
    [
        (
            None,
            [
                (SPEECH_STARTED, 0),
                (SPEECH_STOPPED, 608),
                (SPEECH_STARTED, 1300),
                (SPEECH_STOPPED, 2624),
                (SPEECH_STARTED, 3700),
                (SPEECH_STOPPED, 4608),
            ],
        ),
        (
            6.0,
            [
                (SPEECH_STARTED, 0),
                (SPEECH_STOPPED, 608),
                (SPEECH_STARTED, 3700),
                (SPEECH_STOPPED, 4608),
            ],
        ),
    ],
)
def test_a_voice_much_quieter_than_the_last_utterance_starts_no_speech(
    margin_db, boundaries
):
    """The user asks at 8000, a background voice talks at 2000 (-12 dB), then the user interrupts, starting at 2000 and going on at 5000 (-4 dB)."""
    hooks = VADHooks(
        VADConfig(silence_duration_ms=800, background_margin_db=margin_db),
        AnySoundIsSpeech,
    )
    node = Node(hooks, "a")
    script = [8000] * 6 + [0] * 10 + [2000] * 10 + [0] * 14 + [2000] * 2 + [5000] * 4
    for unit, amplitude in enumerate(script):
        node.append(
            TimedChunk(AUDIO, unit * 100, 100, unit, tone(amplitude), format=PCM16)
        )
    node.append(
        TimedChunk(
            AUDIO, len(script) * 100, 100, len(script), tone(0), format=PCM16, eos=True
        )
    )
    assert [(kind, t_ms) for kind, t_ms, _ in node.boundaries()] == boundaries


def test_the_vad_stage_ignores_background_voices_unless_its_factory_sets_null():
    declared = vad_stage().model_dump()
    for factory, margin_db in (({}, 6.0), ({"background_margin_db": None}, None)):
        stage = VADStageConfig.model_validate({**declared, "factory": factory})
        hooks = create_scheduler(**resolve_stage_typed_kwargs(stage)).session_hooks
        assert hooks.config.background_margin_db == margin_db
