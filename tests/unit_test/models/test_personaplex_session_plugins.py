# SPDX-License-Identifier: Apache-2.0
"""ELLSA 2 motion and AVTR-1 avatar nodes behind PersonaPlex: the Session graph the example configs declare, and both nodes driven by PersonaPlex's 24 kHz input and personaplex_reply units."""

import queue
import threading
from pathlib import Path

import numpy as np
import pytest
import torch

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_stage_factory_kwargs
from sglang_omni.models.avtr.formats import JPEG_FRAME, jpeg_frame_port
from sglang_omni.models.avtr.node import AVATAR_STAGE
from sglang_omni.models.avtr.session import WINDOW_SAMPLES, AvatarHooks
from sglang_omni.models.ellsa2.formats import (
    FEATURES_PER_FRAME,
    SMPLH_FLAME_MOTION,
    SMPLH_FLAME_MOTION_PORT,
)
from sglang_omni.models.ellsa2.generator import (
    BLOCK_SAMPLES,
    BLOCK_STEPS,
    NUM_CODEBOOKS,
)
from sglang_omni.models.ellsa2.node import MOTION_STAGE
from sglang_omni.models.ellsa2.session import BLOCK_MS, MotionHooks
from sglang_omni.models.personaplex.config import (
    LM_STAGE,
    PersonaPlexSessionPipelineConfig,
)
from sglang_omni.models.personaplex.formats import PERSONAPLEX_REPLY
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import AUDIO, PCM16
from sglang_omni.scheduling.session import SessionContext
from tests.unit_test.fixtures.session_outputs import WAIT_S

CONFIGS = Path(__file__).resolve().parents[3] / "examples/configs"
RATE_HZ = 24000
UNIT_MS = 80
# PersonaPlex answers a Session's first unit with two agent frames and every later one with one.
FIRST_REPLY_MS = 160
# The AVTR-1 node lip-syncs this many 16 kHz samples (405 ms) of the agent's audio from a chunk's start.
AVATAR_WINDOW = 6480


def pcm16(value: float, ms: float) -> bytes:
    return np.full(
        int(ms * RATE_HZ / 1000), round(value * 32767), dtype="<i2"
    ).tobytes()


class Driver:
    """One Session of a plugin node fed the way a PersonaPlex graph feeds it: the user's 24 kHz units and the speech node's personaplex_reply units."""

    def __init__(self, hooks, emit) -> None:
        self.hooks = hooks
        self.identity = SessionIdentity("pp")
        self.context = SessionContext(
            session_identity=self.identity, cancelled=threading.Event(), emit=emit
        )
        hooks.open(self.identity, OmniRequest(None))
        hooks.attach_output_stream(self.identity, emit)
        hooks.attach_output_error_handler(self.identity, self.fail)
        self.errors: list[BaseException] = []
        self.user_ms = 0.0
        self.reply_ms = 0.0

    def fail(self, error: BaseException) -> None:
        self.errors.append(error)

    def append(self, chunk: TimedChunk) -> None:
        self.hooks.append(
            chunk, StagePayload("x", OmniRequest(None), None), self.context
        )

    def user(self, until_ms: float, value: float = 0.25) -> None:
        while self.user_ms < until_ms:
            self.append(
                TimedChunk(
                    AUDIO, self.user_ms, UNIT_MS, 0, pcm16(value, UNIT_MS), format=PCM16
                )
            )
            self.user_ms += UNIT_MS

    def reply(
        self, until_ms: float, speaking: tuple[float, float] = (0.0, 0.0)
    ) -> None:
        """personaplex_reply units up to until_ms: 0.5 inside `speaking`, silence elsewhere."""
        while self.reply_ms < until_ms:
            ms = FIRST_REPLY_MS if self.reply_ms == 0 else UNIT_MS
            value = 0.5 if speaking[0] <= self.reply_ms < speaking[1] else 0.0
            self.append(
                TimedChunk(
                    AUDIO,
                    self.reply_ms,
                    ms,
                    0,
                    {"text": "", "pcm": pcm16(value, ms)},
                    format=PERSONAPLEX_REPLY,
                )
            )
            self.reply_ms += ms


class AudioHeard:
    """A generator that records the last block of both streams it is given and answers zeros."""

    def __init__(self) -> None:
        self.target: list[np.ndarray] = []
        self.partner: list[np.ndarray] = []

    def generate_block(
        self, prompt, target, partner, known_codes, style_steps, num_blocks
    ):
        self.target.append(target[0, -BLOCK_SAMPLES:, 0].numpy())
        self.partner.append(partner[0, -BLOCK_SAMPLES:, 0].numpy())
        return torch.zeros(BLOCK_STEPS, NUM_CODEBOOKS, dtype=torch.long)


class Rows:
    def decode(self, codes: torch.Tensor) -> np.ndarray:
        return np.zeros((24, FEATURES_PER_FRAME), dtype="<f4")


class Models:
    def __init__(self) -> None:
        self.generator = AudioHeard()
        self.prompt = None
        self.style = torch.zeros(BLOCK_STEPS, NUM_CODEBOOKS, dtype=torch.long)
        self.device = torch.device("cpu")

    def filterbank(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform.unsqueeze(-1)

    def new_decoder(self) -> Rows:
        return Rows()


class Renderer:
    """Records each chunk's windows and answers five frames."""

    def __init__(self) -> None:
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []

    def render(self, state, speech, listen, frame):
        self.windows.append((speech.copy(), listen.copy()))
        for index in range(5):
            frame(f"{len(self.windows)}:{index}".encode())
        return b"s"

    def is_healthy(self) -> bool:
        return True


def config(name: str) -> PersonaPlexSessionPipelineConfig:
    return ConfigManager.from_file(str(CONFIGS / name)).config


def test_the_motion_example_carries_an_ellsa_2_node_reading_the_input_and_the_reply_on_its_own_gpu():
    pipeline = config("personaplex_session_motion.yaml")
    graph = pipeline.session_graph()
    assert graph.nodes["speech"].stages == (
        "preprocessing",
        "mimi_encode",
        LM_STAGE,
        "code2wav",
    )
    assert graph.nodes["motion"].stages == (MOTION_STAGE,)
    assert set(graph.reading_nodes("speech", PERSONAPLEX_REPLY)) == {"motion"}
    assert "motion" in graph.reading_nodes("input", PCM16)
    assert graph.publish == {
        "speech": (PERSONAPLEX_REPLY,),
        "motion": (SMPLH_FLAME_MOTION,),
    }
    assert graph.input_timed == frozenset({("motion", SMPLH_FLAME_MOTION)})
    assert graph.plugin_formats == frozenset()
    assert pipeline.plugin_output_ports() == (SMPLH_FLAME_MOTION_PORT,)
    stage = pipeline.stage_named(MOTION_STAGE)
    assert stage.gpu == 1 and pipeline.stage_named(LM_STAGE).gpu == 0
    assert stage.factory.reply_stream is True
    assert resolve_stage_factory_kwargs(stage, pipeline) == {
        "speech_sample_rate_hz": RATE_HZ,
        "input_sample_rate_hz": RATE_HZ,
        "max_open_sessions": pipeline.max_sessions,
    }


def test_the_avatar_example_carries_an_avtr_1_node_whose_port_has_the_frame_size_it_encodes():
    pipeline = config("personaplex_session_avatar.yaml")
    graph = pipeline.session_graph()
    stage = pipeline.stage_named(AVATAR_STAGE)
    assert stage.factory.reply_stream is True
    assert set(graph.reading_nodes("speech", PERSONAPLEX_REPLY)) == {"avatar"}
    assert graph.input_timed == frozenset({("avatar", JPEG_FRAME)})
    assert pipeline.plugin_output_ports() == (
        jpeg_frame_port(stage.factory.width, stage.factory.height),
    )
    assert resolve_stage_factory_kwargs(stage, pipeline) == {
        "speech_sample_rate_hz": RATE_HZ,
        "input_sample_rate_hz": RATE_HZ,
    }


def test_without_session_nodes_a_personaplex_session_stays_one_route_with_no_plugin_port():
    pipeline = PersonaPlexSessionPipelineConfig(model_path="nvidia/personaplex-7b-v1")
    assert pipeline.session_graph() is None
    assert pipeline.plugin_output_ports() == ()
    assert [stage.name for stage in pipeline.stages] == [
        "preprocessing",
        "mimi_encode",
        LM_STAGE,
        "code2wav",
    ]


@pytest.fixture
def motion():
    models = Models()
    outputs: queue.Queue[TimedChunk] = queue.Queue()
    driver = Driver(
        MotionHooks(models, RATE_HZ, input_sample_rate_hz=RATE_HZ), outputs.put
    )
    yield driver, models, outputs
    driver.hooks.close(driver.identity)


def test_motion_blocks_wait_for_the_reply_units_and_hear_them_at_their_own_stamps(
    motion,
):
    driver, models, outputs = motion
    # The speech node answers the first input unit before the user's audio covers a block; from then on blocks wait for its units.
    driver.reply(FIRST_REPLY_MS)
    driver.user(1600)
    with pytest.raises(queue.Empty):
        outputs.get(timeout=0.3)
    driver.reply(2480, speaking=(800, 1600))
    driver.user(2400)
    chunks = [outputs.get(timeout=WAIT_S) for _ in range(3)]
    assert [(c.format, c.t_start_ms, c.duration_ms, c.turn) for c in chunks] == [
        (SMPLH_FLAME_MOTION, 0.0, 800.0, None),
        (SMPLH_FLAME_MOTION, 800.0, 800.0, None),
        (SMPLH_FLAME_MOTION, 1600.0, 800.0, None),
    ]
    heard = models.generator.target
    assert np.abs(heard[0][:-100]).max() < 1e-3
    assert np.allclose(heard[1][100:-100], 0.5, atol=1e-3)
    assert np.abs(heard[2][100:]).max() < 1e-3
    assert all(
        np.allclose(p[100:-100], 0.25, atol=1e-3) for p in models.generator.partner
    )
    assert not driver.errors


def test_avatar_chunks_wait_until_the_reply_units_cover_their_lip_sync_window():
    renderer = Renderer()
    frames: list[TimedChunk] = []
    changed = threading.Condition()

    def emit(chunk: TimedChunk) -> None:
        with changed:
            frames.append(chunk)
            changed.notify_all()

    driver = Driver(
        AvatarHooks([renderer], RATE_HZ, input_sample_rate_hz=RATE_HZ), emit
    )
    try:
        driver.reply(FIRST_REPLY_MS)
        driver.user(1000)
        with changed:
            assert not changed.wait_for(lambda: frames, timeout=0.3)
        driver.reply(2000, speaking=(400, 1200))
        with changed:
            assert changed.wait_for(lambda: len(frames) >= 30, timeout=WAIT_S)
        timeline = [(frame.t_start_ms, frame.turn) for frame in frames]
        assert timeline == [(40.0 * k, None) for k in range(len(frames))]
        # Each rendered chunk hears the agent at the units' own stamps, 0.5 from 400 to 1200 ms, and the user's 0.25 up to 1000 ms; resampling rings within a few samples of each edge.
        starts = {
            frame.payload.decode(): round(frame.t_start_ms * 16)
            for frame in frames
            if frame.payload.endswith(b":0")
        }
        assert len(renderer.windows) >= 1
        for index, (speech, listen) in enumerate(renderer.windows):
            start = starts[f"{index + 1}:0"]
            for window, origin, edges, expected in (
                (
                    speech,
                    start,
                    (6400, 19200),
                    lambda t: 0.5 if 6400 <= t < 19200 else 0.0,
                ),
                (
                    listen,
                    start - AVATAR_WINDOW,
                    (0, 16000),
                    lambda t: 0.25 if 0 <= t < 16000 else 0.0,
                ),
            ):
                for offset, value in enumerate(window):
                    t = origin + offset
                    if min(abs(t - edge) for edge in edges) > 40:
                        assert abs(value - expected(t)) < 1e-3, (index, start, t)
                    else:
                        pass
        assert not driver.errors
    finally:
        driver.hooks.close(driver.identity)
        for worker in driver.hooks.workers:
            if worker is not None:
                worker.join(WAIT_S)
            else:
                pass


def test_each_port_names_how_far_past_a_frame_its_node_hears_the_reply():
    """A client aligning PersonaPlex's turnless reply stream plays the audio this much (plus its playout delay) behind its arrival."""
    assert SMPLH_FLAME_MOTION_PORT.params["reply_lookahead_ms"] == BLOCK_MS == 800
    port = jpeg_frame_port(640, 360)
    assert port.params["reply_lookahead_ms"] == WINDOW_SAMPLES / 16 == 405
    for params in (SMPLH_FLAME_MOTION_PORT.params, port.params):
        assert params["reply_lookahead_ms"] <= params["reply_media_wait_ms"]


@pytest.mark.parametrize("reply_stream", [False, True])
def test_a_motion_block_before_the_first_reply_unit_waits_for_it_only_with_reply_stream(
    reply_stream,
):
    models = Models()
    outputs: queue.Queue[TimedChunk] = queue.Queue()
    driver = Driver(
        MotionHooks(
            models, RATE_HZ, input_sample_rate_hz=RATE_HZ, reply_stream=reply_stream
        ),
        outputs.put,
    )
    try:
        driver.user(800)
        if reply_stream:
            with pytest.raises(queue.Empty):
                outputs.get(timeout=0.3)
            driver.reply(720, speaking=(0, 800))
            with pytest.raises(queue.Empty):
                outputs.get(timeout=0.3)
            driver.reply(800, speaking=(0, 800))
        else:
            pass
        chunk = outputs.get(timeout=WAIT_S)
        assert (chunk.t_start_ms, chunk.duration_ms) == (0.0, 800.0)
        heard = models.generator.target[0]
        # With reply_stream the block hears the units it waited for; without, it was made before any arrived and hears silence.
        if reply_stream:
            assert np.allclose(heard[100:-100], 0.5, atol=1e-3)
        else:
            assert np.abs(heard).max() < 1e-3
        assert not driver.errors
    finally:
        driver.hooks.close(driver.identity)


@pytest.mark.parametrize("reply_stream", [False, True])
def test_an_avatar_chunk_before_the_first_reply_unit_waits_for_it_only_with_reply_stream(
    reply_stream,
):
    renderer = Renderer()
    frames: list[TimedChunk] = []
    changed = threading.Condition()

    def emit(chunk: TimedChunk) -> None:
        with changed:
            frames.append(chunk)
            changed.notify_all()

    driver = Driver(
        AvatarHooks(
            [renderer], RATE_HZ, input_sample_rate_hz=RATE_HZ, reply_stream=reply_stream
        ),
        emit,
    )
    try:
        # The user's audio reaches the first chunk's start, and not the second's.
        driver.user(160)
        if reply_stream:
            with changed:
                assert not changed.wait_for(lambda: frames, timeout=0.3)
            # The first chunk lip-syncs 405 ms ahead: units up to 400 ms are not enough, 480 ms are.
            driver.reply(400, speaking=(0, 480))
            with changed:
                assert not changed.wait_for(lambda: frames, timeout=0.3)
            driver.reply(480, speaking=(0, 480))
        else:
            pass
        with changed:
            assert changed.wait_for(lambda: len(frames) >= 5, timeout=WAIT_S)
        assert [frame.t_start_ms for frame in frames[:5]] == [
            40.0 * k for k in range(5)
        ]
        speech, _ = renderer.windows[0]
        if reply_stream:
            assert np.allclose(speech[100:-100], 0.5, atol=1e-3)
        else:
            assert np.abs(speech).max() < 1e-3
        assert not driver.errors
    finally:
        driver.hooks.close(driver.identity)
        for worker in driver.hooks.workers:
            if worker is not None:
                worker.join(WAIT_S)
            else:
                pass
