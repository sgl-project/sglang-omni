# SPDX-License-Identifier: Apache-2.0
"""The ELLSA 2 motion node: the Session timeline of the user's and the reply's audio, and the blocks of motion it publishes."""

import queue
import threading
from pathlib import Path

import numpy as np
import torch
from scipy.signal import resample_poly

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.ellsa2.formats import (
    FEATURES_PER_FRAME,
    MOTION,
    SMPLH_FLAME_MOTION,
    SMPLH_FLAME_MOTION_PORT,
)
from sglang_omni.models.ellsa2.generator import (
    BLOCK_SAMPLES,
    BLOCK_STEPS,
    NUM_CODEBOOKS,
)
from sglang_omni.models.ellsa2.node import MOTION_STAGE
from sglang_omni.models.ellsa2.session import MotionHooks
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import AUDIO, INTERRUPT, PCM16, PCM_F32
from sglang_omni.scheduling.session import SessionContext
from tests.unit_test.fixtures.session_outputs import control_chunk

UNIT_MS = 200
UNIT_SAMPLES = 3200
SPEECH_RATE_HZ = 24000
FRAMES_PER_BLOCK = 24
WAIT_S = 10
PROFILE = (
    Path(__file__).resolve().parents[3]
    / "examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-base-clone-motion/launch.yaml"
)


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
        return np.zeros((FRAMES_PER_BLOCK, FEATURES_PER_FRAME), dtype="<f4")


class Models:
    """Models whose filterbank passes the samples through, so the generator sees the audio of each block."""

    def __init__(self) -> None:
        self.generator = AudioHeard()
        self.prompt = None
        self.style = torch.zeros(BLOCK_STEPS, NUM_CODEBOOKS, dtype=torch.long)
        self.device = torch.device("cpu")

    def filterbank(self, waveform: torch.Tensor) -> torch.Tensor:
        return waveform.unsqueeze(-1)

    def new_decoder(self) -> Rows:
        return Rows()


class Session:
    def __init__(self) -> None:
        self.models = Models()
        self.hooks = MotionHooks(self.models, SPEECH_RATE_HZ)
        self.identity = SessionIdentity("a")
        self.outputs: queue.Queue[TimedChunk] = queue.Queue()
        self.hooks.open(self.identity, OmniRequest(None))
        self.hooks.attach_output_stream(self.identity, self.outputs.put)
        self.hooks.attach_output_error_handler(self.identity, self.fail)
        self.context = SessionContext(
            session_identity=self.identity,
            cancelled=threading.Event(),
            emit=self.outputs.put,
        )
        self.t_ms = 0
        self.errors: list[BaseException] = []

    def fail(self, error: BaseException) -> None:
        self.errors.append(error)

    def append(self, chunk: TimedChunk) -> None:
        self.hooks.append(
            chunk, StagePayload("x", OmniRequest(None), None), self.context
        )

    def user(self, until_ms: int, *, eos: bool = False) -> None:
        """The user's audio, a constant 0.25 in 200 ms units, up to until_ms."""
        while self.t_ms < until_ms:
            samples = np.full(UNIT_SAMPLES, 8192, dtype="<i2")
            self.append(
                TimedChunk(
                    AUDIO,
                    self.t_ms,
                    UNIT_MS,
                    0,
                    samples.tobytes(),
                    format=PCM16,
                    eos=eos and self.t_ms + UNIT_MS >= until_ms,
                )
            )
            self.t_ms += UNIT_MS

    def reply(self, turn: int, seconds: float, value: float = 0.5) -> None:
        samples = np.full(int(seconds * SPEECH_RATE_HZ), value, dtype="<f4")
        self.append(
            TimedChunk(AUDIO, 0.0, 0.0, 0, samples.tobytes(), format=PCM_F32, turn=turn)
        )

    def motion(self, count: int) -> list[TimedChunk]:
        return [self.outputs.get(timeout=WAIT_S) for _ in range(count)]

    def close(self) -> None:
        self.hooks.close(self.identity)


def test_each_block_of_user_audio_becomes_one_motion_chunk_of_24_frames():
    session = Session()
    session.user(1600)
    chunks = session.motion(2)
    session.close()
    assert [
        (c.modality, c.format, c.t_start_ms, c.duration_ms, c.turn) for c in chunks
    ] == [
        (MOTION, SMPLH_FLAME_MOTION, 0.0, 800.0, None),
        (MOTION, SMPLH_FLAME_MOTION, 800.0, 800.0, None),
    ]
    assert all(
        len(c.payload) == FRAMES_PER_BLOCK * FEATURES_PER_FRAME * 4 for c in chunks
    )
    assert np.allclose(session.models.generator.partner[0], 0.25)


def test_a_reply_starts_with_the_block_still_collecting_user_audio():
    session = Session()
    session.user(1000)
    chunks = session.motion(1)
    session.reply(turn=7, seconds=1.2)
    session.user(2400)
    chunks += session.motion(2)
    session.close()
    heard = session.models.generator.target
    assert [c.turn for c in chunks] == [None, 7, 7]
    assert not heard[0].any()
    assert np.allclose(heard[1][100:-100], 0.5, atol=1e-3)
    assert np.allclose(heard[2][100 : 6400 - 100], 0.5, atol=1e-3)
    assert not heard[2][6400 + 100 :].any()


def test_an_interrupt_cuts_the_reply_at_the_next_block_and_drops_its_later_audio():
    session = Session()
    session.user(1000)
    session.motion(1)
    session.reply(turn=7, seconds=3.0)
    session.user(1600)
    assert session.motion(1)[0].turn == 7
    session.hooks.control(session.identity, control_chunk(INTERRUPT, turn=7))
    session.reply(turn=7, seconds=1.0)
    session.user(3200)
    chunks = session.motion(2)
    session.close()
    assert [c.turn for c in chunks] == [None, None]
    assert not np.concatenate(session.models.generator.target[2:]).any()


def test_after_the_input_ends_blocks_run_through_the_reply():
    session = Session()
    session.user(400)
    session.reply(turn=1, seconds=1.5)
    session.user(600, eos=True)
    chunks = session.motion(2)
    session.close()
    assert [(c.t_start_ms, c.turn) for c in chunks] == [(0.0, 1), (800.0, 1)]


def test_reply_windows_join_as_one_resampling_of_the_whole_reply():
    hooks = MotionHooks(Models(), SPEECH_RATE_HZ)
    reply = np.sin(np.arange(3 * SPEECH_RATE_HZ) / 7.0).astype(np.float32)
    joined = np.concatenate(
        [
            hooks.resampled(reply, start, start + BLOCK_SAMPLES)
            for start in range(0, 2 * BLOCK_SAMPLES, BLOCK_SAMPLES)
        ]
    )
    whole = resample_poly(reply, 2, 3)[: 2 * BLOCK_SAMPLES]
    assert np.abs(joined - whole)[100:].max() < 1e-5


def test_the_motion_profile_publishes_the_motion_port_from_its_gpu_stage():
    config = ConfigManager.from_file(str(PROFILE)).config
    stage = config.stage_named(MOTION_STAGE)
    graph = config.session_graph()
    assert SMPLH_FLAME_MOTION_PORT in config.session_ports().plugin_output
    assert SMPLH_FLAME_MOTION not in graph.plugin_formats
    assert stage.gpu == 0 and stage.factory.speech_sample_rate_hz == SPEECH_RATE_HZ
    assert graph.nodes["motion"].stages == (MOTION_STAGE,)
    assert graph.publish["motion"] == (SMPLH_FLAME_MOTION,)
    assert graph.input_timed == frozenset({("motion", SMPLH_FLAME_MOTION)})


def test_the_warm_up_runs_every_history_length_on_the_worker_before_returning():
    models = Models()
    hooks = MotionHooks(models, SPEECH_RATE_HZ)
    hooks.warm_up_serving_thread()
    assert len(models.generator.target) == 16
    assert hooks.worker is not None and hooks.worker.is_alive()
