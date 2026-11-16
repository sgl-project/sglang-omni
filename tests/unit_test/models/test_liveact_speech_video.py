# SPDX-License-Identifier: Apache-2.0
"""SoulX-LiveAct behind the speech_video node: its profile, its worker's Session tickets and wire, and the node driving a fake LiveAct worker at LiveAct's geometry (1.6 s chunks of 32 frames at 20 fps, 1.05 s of context, 0.1 s of look-ahead)."""

import importlib
import socket
import sys
import threading
from pathlib import Path

import httpx
import numpy as np
import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.speech_avatar.formats import JPEG_FRAME, jpeg_frame_port
from sglang_omni.models.speech_avatar.node import SPEECH_VIDEO_STAGE
from sglang_omni.models.speech_avatar.session import ChunkGeometry, SpeechVideoHooks
from sglang_omni.models.speech_avatar.worker import HttpChunkWorker
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.proto.session import ControlEdge, SessionIdentity, TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    INTERRUPT,
    PCM16,
    PCM_F32,
    SESSION_CONTROL,
    TURN_DONE,
)
from sglang_omni.scheduling.session import SessionContext
from tests.unit_test.fixtures.session_outputs import WAIT_S, control_chunk

ROOT = Path(__file__).parents[3]
PROFILE = (
    ROOT
    / "examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-base-clone-liveact/launch.yaml"
)
WORKERS = ROOT / "examples/speech_avatar_workers"
sys.path.insert(0, str(WORKERS))
chunk_server = importlib.import_module("chunk_server")
liveact_worker = importlib.import_module("liveact_worker")

SPEECH_RATE_HZ = 24000
INFO = liveact_worker.geometry_info(416, 720)
GEOMETRY = ChunkGeometry(
    fps=INFO["fps"],
    chunk_frames=INFO["chunk_frames"],
    context_samples=INFO["context_samples"],
    lookahead_samples=INFO["lookahead_samples"],
)
FRAME_MS = 50
RENDER_LEAD_MS = 2600
IDLE = b"idle-picture"


def test_liveact_geometry_is_generate_py_streaming_block():
    # Block b >= 1 makes model frames 21 + 32 (b - 1) to 52 + 32 (b - 1) from the audio of frames 32 (b - 1) to 32 (b - 1) + 55.
    assert (GEOMETRY.fps, GEOMETRY.chunk_frames) == (20, 32)
    assert GEOMETRY.chunk_ms == 1600
    assert GEOMETRY.context_samples == 21 * 800
    assert GEOMETRY.lookahead_samples == 2 * 800
    assert GEOMETRY.window_samples == 55 * 800


def test_the_profile_feeds_the_agents_audio_to_liveact_and_publishes_its_frames_on_the_input_timeline():
    config = ConfigManager.from_file(str(PROFILE)).config
    graph = config.session_graph()
    stage = config.stage_named(SPEECH_VIDEO_STAGE)
    assert "speech_video" in graph.reading_nodes("input", PCM16)
    assert "speech_video" in graph.reading_nodes("tts", PCM_F32)
    assert "speech_video" in graph.reading_nodes("tts", SESSION_CONTROL)
    assert (
        "speech_video",
        ControlEdge("dialog", INTERRUPT),
    ) in graph.control_targets("dialog", INTERRUPT)
    assert graph.publish["speech_video"] == (JPEG_FRAME,)
    assert graph.input_timed == frozenset({("speech_video", JPEG_FRAME)})
    factory = stage.factory
    assert (factory.fps, factory.width, factory.height) == (20, 416, 720)
    assert (factory.chunk_ms, factory.lookahead_ms) == (
        GEOMETRY.chunk_ms,
        GEOMETRY.lookahead_ms,
    )
    assert factory.sessions_per_worker == 1
    assert factory.reply_starts_chunk is True
    (port,) = config.session_ports().plugin_output
    assert port == jpeg_frame_port(
        fps=20,
        width=416,
        height=720,
        chunk_ms=1600,
        lookahead_ms=100,
        render_lead_ms=factory.render_lead_ms,
    )
    assert port.params["reply_lookahead_ms"] == 1700
    # A client holding a reply's audio waits for a chunk of it, the look-ahead, the render lead and some slack.
    assert port.params["reply_media_wait_ms"] == 1700 + factory.render_lead_ms + 600
    assert port.params["timeline"] == "input" and port.params["follows_replies"]


def test_a_worker_continues_its_session_and_starts_a_new_one_for_any_other_ticket():
    tokens = iter([b"a" * 16, b"b" * 16, b"c" * 16])
    tickets = liveact_worker.Tickets(new_token=lambda: next(tokens))
    assert tickets.blocks_for(b"") == [0, 1]
    first = tickets.state()
    assert liveact_worker.Ticket.decode(first) == liveact_worker.Ticket(b"a" * 16, 1)
    assert tickets.blocks_for(first) == [2]
    second = tickets.state()
    assert tickets.blocks_for(second) == [3]
    # A stale ticket of the same Session (a retry) and garbage both start over.
    assert tickets.blocks_for(second) == [0, 1]
    assert liveact_worker.Ticket.decode(tickets.state()).token == b"b" * 16
    assert tickets.blocks_for(b"not a ticket") == [0, 1]
    assert liveact_worker.Ticket.decode(tickets.state()) == liveact_worker.Ticket(
        b"c" * 16, 1
    )


class FakeLiveAct(chunk_server.ChunkModel):
    """LiveAct's geometry and tickets without the model: each frame names the block that made it."""

    def __init__(self) -> None:
        self.info = INFO
        self.tickets = liveact_worker.Tickets()
        self.blocks: list[list[int]] = []
        self.windows: list[np.ndarray] = []

    def idle_jpeg(self) -> bytes:
        return IDLE

    def render(self, state, speech):
        blocks = self.tickets.blocks_for(state)
        self.blocks.append(blocks)
        self.windows.append(speech.copy())
        return self.tickets.state(), [
            f"block{blocks[-1]}:{index}".encode() for index in range(32)
        ]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture(scope="module")
def served():
    """A fake LiveAct served over chunk_server's HTTP protocol."""
    model = FakeLiveAct()
    port = free_port()
    threading.Thread(
        target=chunk_server.serve,
        args=(lambda: model, "127.0.0.1", port),
        daemon=True,
    ).start()
    worker = HttpChunkWorker.connect(f"http://127.0.0.1:{port}", startup_timeout_s=10)
    yield model, worker


def test_over_the_wire_the_worker_reports_liveacts_geometry_and_chains_its_tickets(
    served,
):
    model, worker = served
    assert worker.geometry == GEOMETRY
    assert (worker.info["width"], worker.info["height"]) == (416, 720)
    assert worker.idle_picture() == IDLE
    speech = np.linspace(-0.5, 0.5, GEOMETRY.window_samples, dtype=np.float32)
    state, frames = worker.render(b"", speech)
    assert len(frames) == 32 and frames[0] == b"block1:0"
    assert model.blocks[-1] == [0, 1]
    assert np.allclose(model.windows[-1], speech, atol=1 / 16000)
    state, frames = worker.render(state, speech)
    assert frames[-1] == b"block2:31" and model.blocks[-1] == [2]
    # Another Session's first chunk takes the worker over.
    _, frames = worker.render(b"", speech)
    assert model.blocks[-1] == [0, 1]


def test_over_the_wire_a_window_of_the_wrong_length_is_the_sessions_error(served):
    _, worker = served
    with pytest.raises(httpx.HTTPStatusError) as error:
        worker.render(b"", np.zeros(GEOMETRY.window_samples - 1, dtype=np.float32))
    assert error.value.response.status_code == 400


class ScriptedLiveAct:
    """A ChunkWorker in process: LiveAct's tickets, 32 frames naming the chunk, every window kept."""

    def __init__(self) -> None:
        self.tickets = liveact_worker.Tickets()
        self.requests: list[tuple[bytes, list[int], np.ndarray]] = []
        self.states: list[bytes] = []
        self.gate = threading.Event()
        self.gate.set()
        self.rendering = threading.Event()

    def render(self, state, speech):
        self.rendering.set()
        assert self.gate.wait(WAIT_S)
        blocks = self.tickets.blocks_for(state)
        self.requests.append((state, blocks, speech.copy()))
        self.states.append(self.tickets.state())
        return self.states[-1], [
            f"c{len(self.requests)}:{index}".encode() for index in range(32)
        ]

    def is_healthy(self) -> bool:
        return True


class Talk:
    """One Session of the speech_video hooks over a scripted LiveAct, appended the way the Session scheduler does."""

    def __init__(self) -> None:
        self.worker = ScriptedLiveAct()
        self.hooks = SpeechVideoHooks(
            [self.worker],
            GEOMETRY,
            SPEECH_RATE_HZ,
            idle_picture=IDLE,
            render_lead_ms=RENDER_LEAD_MS,
            reply_starts_chunk=True,
        )
        self.identity = SessionIdentity("liveact")
        self.frames: list[TimedChunk] = []
        self.changed = threading.Condition()
        self.context = SessionContext(
            session_identity=self.identity, cancelled=threading.Event(), emit=self.emit
        )
        self.hooks.open(self.identity, OmniRequest(None))
        self.hooks.attach_output_error_handler(self.identity, self.fail)
        self.hooks.attach_output_stream(self.identity, self.emit)
        self.errors: list[BaseException] = []

    def emit(self, chunk: TimedChunk) -> None:
        with self.changed:
            self.frames.append(chunk)
            self.changed.notify_all()

    def fail(self, error: BaseException) -> None:
        with self.changed:
            self.errors.append(error)
            self.changed.notify_all()

    def wait_frames(self, count: int) -> None:
        with self.changed:
            assert self.changed.wait_for(lambda: len(self.frames) >= count, WAIT_S)

    def append(self, chunk: TimedChunk) -> None:
        self.hooks.append(
            chunk, StagePayload("x", OmniRequest(None), None), self.context
        )

    def user(self, t_ms: float, ms: float = 400, eos: bool = False) -> None:
        pcm = np.full(int(ms * 16), 1000, dtype="<i2").tobytes()
        self.append(TimedChunk(AUDIO, t_ms, ms, 0, pcm, format=PCM16, eos=eos))

    def user_until(self, end_ms: float, start_ms: float = 0) -> None:
        for t_ms in range(int(start_ms), int(end_ms), 400):
            self.user(t_ms)

    def speech(self, ms: float, turn: int, value: float = 0.5) -> None:
        pcm = np.full(int(ms * SPEECH_RATE_HZ / 1000), value, dtype="<f4").tobytes()
        self.append(TimedChunk(AUDIO, 0, ms, 0, pcm, format=PCM_F32, turn=turn))

    def close(self) -> None:
        if self.identity in self.hooks.sessions:
            self.hooks.close(self.identity)
        else:
            pass


@pytest.fixture
def talk():
    session = Talk()
    yield session
    session.close()


def test_while_the_agent_is_silent_the_picture_holds_and_liveact_does_not_render(talk):
    talk.user_until(2000)
    talk.wait_frames(40)
    assert [frame.t_start_ms for frame in talk.frames] == [
        FRAME_MS * k for k in range(40)
    ]
    assert talk.frames[0].payload == IDLE
    assert all(frame.payload == b"" for frame in talk.frames[1:])
    assert all(frame.format == JPEG_FRAME for frame in talk.frames)
    assert talk.worker.requests == []


def test_a_reply_renders_render_lead_ahead_in_1_6_s_chunks_each_from_the_last_ticket(
    talk,
):
    talk.user_until(1200)
    talk.wait_frames(24)
    talk.speech(3000, turn=1)
    talk.append(control_chunk(TURN_DONE, turn=1))
    talk.user_until(9000, start_ms=1200)
    talk.wait_frames(180)
    stamps = [frame.t_start_ms for frame in talk.frames[:180]]
    assert stamps == [FRAME_MS * k for k in range(180)]
    # The reply starts render_lead past the frame the node had reached when it was placed.
    turns = [frame.turn for frame in talk.frames[:180]]
    first = turns.index(1)
    assert first * FRAME_MS >= 1200 + RENDER_LEAD_MS - FRAME_MS
    assert turns[first : first + 60] == [1] * 60
    # The rest of the reply's last chunk carries its turn too (the model's silence after it), so a client dropping the reply drops it.
    end = first + -(-60 // GEOMETRY.chunk_frames) * GEOMETRY.chunk_frames
    assert turns[first + 60 : end] == [1] * (end - first - 60)
    assert 1 not in turns[end:]
    # Two chunks cover the 3 s reply; the first starts LiveAct's Session (blocks 0 and 1), the next continues it.
    requests = talk.worker.requests
    assert [blocks for _, blocks, _ in requests[:2]] == [[0, 1], [2]]
    assert requests[0][0] == b"" and requests[1][0] == talk.worker.states[0]
    rendered = [frame for frame in talk.frames if frame.payload.startswith(b"c")]
    assert [frame.payload for frame in rendered[:32]] == [
        f"c1:{index}".encode() for index in range(32)
    ]
    # The reply's first chunk starts on its first frame: the silence before it went out ahead, so the whole render lead is the render's.
    assert rendered[0].t_start_ms == first * FRAME_MS
    # The window reads the 1.05 s before the chunk (silence before the reply) and the reply from the chunk's start.
    window = requests[0][2]
    assert window.shape == (GEOMETRY.window_samples,)
    assert not window[: GEOMETRY.context_samples - 400].any()
    assert np.allclose(window[GEOMETRY.context_samples + 400 : -400], 0.5, atol=0.01)


def test_the_silence_up_to_a_placed_reply_goes_out_at_once_ahead_of_the_users_audio(
    talk,
):
    talk.user_until(1200)
    talk.wait_frames(24)
    talk.worker.gate.clear()
    talk.speech(3000, turn=1)
    talk.append(control_chunk(TURN_DONE, turn=1))
    talk.user(1200)
    # With the user's audio at 1.6 s, the frames up to the reply's start 2.6 s past the cursor are out and its first chunk renders.
    assert talk.worker.rendering.wait(WAIT_S)
    turns = [frame.turn for frame in talk.frames]
    assert set(turns) == {None}
    assert talk.frames[-1].t_start_ms >= 1200 + RENDER_LEAD_MS - 2 * FRAME_MS
    talk.worker.gate.set()


def test_an_interrupt_stops_liveact_at_the_next_chunk(talk):
    talk.user_until(1200)
    talk.wait_frames(24)
    talk.worker.gate.clear()
    talk.speech(8000, turn=1)
    talk.user_until(2400, start_ms=1200)
    assert talk.worker.rendering.wait(WAIT_S)
    talk.hooks.control(talk.identity, control_chunk(INTERRUPT, turn=1))
    talk.worker.gate.set()
    talk.user_until(16000, start_ms=2400)
    talk.wait_frames(320)
    turns = [frame.turn for frame in talk.frames[:320]]
    last = max(index for index, turn in enumerate(turns) if turn == 1)
    # The 8 s reply would need five chunks; the chunk rendering when the interrupt came is its last.
    assert len(talk.worker.requests) == 1
    assert turns[last + 1 :] == [None] * (319 - last)
    assert all(frame.payload == b"" for frame in talk.frames[last + 1 : 320])
