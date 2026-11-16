# SPDX-License-Identifier: Apache-2.0
"""The speech-avatar nodes of a Session graph (FlashHead, MuseTalk, StreamTalk): pictures or motion on the input timeline from the agent's speech, rendered by a pool of chunk workers."""

import socket
import struct
import sys
import threading
from pathlib import Path

import httpx
import numpy as np
import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.speech_avatar.formats import (
    JPEG_FRAME,
    MOTION,
    SMPLX_FEATURES_PER_FRAME,
    SMPLX_MOTION,
    VIDEO,
)
from sglang_omni.models.speech_avatar.node import (
    SPEECH_MOTION_STAGE,
    SPEECH_VIDEO_STAGE,
)
from sglang_omni.models.speech_avatar.session import (
    ChunkGeometry,
    SpeechMotionHooks,
    SpeechVideoHooks,
    WorkerUnavailable,
)
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
PROFILES = ROOT / "examples/streaming_cascade/profiles"
WORKERS = ROOT / "examples/speech_avatar_workers"
SPEECH_RATE_HZ = 24000
IDLE = b"idle-picture"
# Five 40 ms frames a chunk (3200 samples at 16 kHz), 100 ms of context and 40 ms of look-ahead.
VIDEO_GEOMETRY = ChunkGeometry(
    fps=25, chunk_frames=5, context_samples=1600, lookahead_samples=640
)
# Fifteen 30 fps frames a block (8000 samples), the 1.5 s before it as context.
MOTION_GEOMETRY = ChunkGeometry(
    fps=30, chunk_frames=15, context_samples=24000, lookahead_samples=0
)
ROW = 4 * SMPLX_FEATURES_PER_FRAME


class ScriptedWorker:
    """Records each chunk's window and state; answers frames (or one motion block) naming the call and a state naming the worker; a render waits while `gate` is clear."""

    def __init__(
        self, name: str = "w", kind: str = "video", requests: list | None = None
    ) -> None:
        self.name = name
        self.kind = kind
        self.windows: list[np.ndarray] = []
        self.requests = [] if requests is None else requests
        self.gate = threading.Event()
        self.gate.set()
        self.rendering = threading.Event()
        self.error: Exception | None = None
        self.healthy = threading.Event()
        self.healthy.set()

    def render(self, state: bytes, speech: np.ndarray):
        self.windows.append(speech.copy())
        self.rendering.set()
        self.gate.wait()
        if self.error is not None:
            error, self.error = self.error, None
            raise error
        else:
            pass
        call = len(self.windows)
        next_state = f"{self.name}{call}".encode()
        self.requests.append((self.name, state, next_state))
        if self.kind == "video":
            payloads = [
                f"{self.name}{call}:{k}".encode()
                for k in range(VIDEO_GEOMETRY.chunk_frames)
            ]
        else:
            payloads = [
                np.full(
                    (MOTION_GEOMETRY.chunk_frames, SMPLX_FEATURES_PER_FRAME),
                    call,
                    dtype="<f4",
                ).tobytes()
            ]
        return next_state, payloads

    def is_healthy(self) -> bool:
        return self.healthy.is_set()


class Clock:
    def __init__(self) -> None:
        self.now = 100.0

    def __call__(self) -> float:
        return self.now


class Session:
    """One Session of the hooks, appended and read the way the Session scheduler does."""

    def __init__(self, hooks, name: str = "session") -> None:
        self.hooks = hooks
        self.identity = SessionIdentity(name)
        self.outputs: list[TimedChunk] = []
        self.errors: list[BaseException] = []
        self.changed = threading.Condition()
        self.context = SessionContext(
            session_identity=self.identity, cancelled=threading.Event(), emit=self.emit
        )
        hooks.open(self.identity, OmniRequest(None))
        hooks.attach_output_error_handler(self.identity, self.fail)
        hooks.attach_output_stream(self.identity, self.emit)

    def emit(self, chunk: TimedChunk) -> None:
        with self.changed:
            self.outputs.append(chunk)
            self.changed.notify_all()

    def fail(self, error: BaseException) -> None:
        with self.changed:
            self.errors.append(error)
            self.changed.notify_all()

    def wait_for(self, condition) -> None:
        with self.changed:
            assert self.changed.wait_for(condition, timeout=WAIT_S)

    def wait_outputs(self, count: int) -> None:
        self.wait_for(lambda: len(self.outputs) >= count)

    def append(self, chunk: TimedChunk) -> None:
        self.hooks.append(
            chunk, StagePayload("x", OmniRequest(None), None), self.context
        )

    def user(self, t_ms: float, ms: float = 200, eos: bool = False) -> None:
        pcm = np.full(int(ms * 16), 1000, dtype="<i2").tobytes()
        self.append(TimedChunk(AUDIO, t_ms, ms, 0, pcm, format=PCM16, eos=eos))

    def speech(self, ms: float, turn: int, value: float = 0.5) -> None:
        pcm = np.full(int(ms * SPEECH_RATE_HZ / 1000), value, dtype="<f4").tobytes()
        self.append(TimedChunk(AUDIO, 0, ms, 0, pcm, format=PCM_F32, turn=turn))

    def turn_done(self, turn: int) -> None:
        self.append(control_chunk(TURN_DONE, turn=turn))

    def timeline(self) -> list[tuple[float, int | None]]:
        return [(chunk.t_start_ms, chunk.turn) for chunk in self.outputs]


def stop(hooks, workers: list[ScriptedWorker]) -> None:
    for worker in workers:
        worker.gate.set()
        worker.healthy.set()
    for session in list(hooks.sessions):
        hooks.close(session)
    for thread in list(hooks.threads):
        if thread is not None:
            thread.join(WAIT_S)
        else:
            pass


def video_hooks(workers, clock=None) -> SpeechVideoHooks:
    return SpeechVideoHooks(
        workers,
        VIDEO_GEOMETRY,
        SPEECH_RATE_HZ,
        idle_picture=IDLE,
        render_lead_ms=80,
        recovery_probe_s=0.01,
        clock=clock or Clock(),
    )


@pytest.fixture
def video():
    worker = ScriptedWorker()
    clock = Clock()
    session = Session(video_hooks([worker], clock))
    yield worker, session, clock
    stop(session.hooks, [worker])


@pytest.fixture
def motion():
    worker = ScriptedWorker(kind="motion")
    session = Session(
        SpeechMotionHooks(
            [worker], MOTION_GEOMETRY, SPEECH_RATE_HZ, recovery_probe_s=0.01
        )
    )
    yield worker, session
    stop(session.hooks, [worker])


@pytest.mark.parametrize(
    "profile, stage, node, port_format, size",
    [
        ("flashhead", SPEECH_VIDEO_STAGE, "speech_video", JPEG_FRAME, (512, 512)),
        ("musetalk", SPEECH_VIDEO_STAGE, "speech_video", JPEG_FRAME, (512, 512)),
        ("streamtalk", SPEECH_MOTION_STAGE, "speech_motion", SMPLX_MOTION, None),
    ],
)
def test_each_profile_feeds_the_agents_speech_to_its_node_and_publishes_an_input_timeline_port(
    profile, stage, node, port_format, size
):
    config = ConfigManager.from_file(
        str(PROFILES / f"qwen3-asr-qwen3-tts-base-clone-{profile}/launch.yaml")
    ).config
    graph = config.session_graph()
    factory = config.stage_named(stage).factory
    assert node in graph.reading_nodes("input", PCM16)
    assert node in graph.reading_nodes("tts", PCM_F32)
    assert node in graph.reading_nodes("tts", SESSION_CONTROL)
    assert (node, ControlEdge("dialog", INTERRUPT)) in graph.control_targets(
        "dialog", INTERRUPT
    )
    assert graph.publish[node] == (port_format,)
    assert graph.input_timed == frozenset({(node, port_format)})
    (port,) = config.session_ports().plugin_output
    assert port.format == port_format
    assert port.params["timeline"] == "input" and port.params["follows_replies"]
    assert port.params["chunk_ms"] == factory.chunk_ms
    assert port.params["fps"] == factory.fps
    assert port.params["reply_lookahead_ms"] == round(
        factory.chunk_ms + factory.lookahead_ms
    )
    assert port.params["reply_media_wait_ms"] > port.params["reply_lookahead_ms"]
    if size is not None:
        assert (port.params["width"], port.params["height"]) == size
    else:
        assert port.params["features"] == SMPLX_FEATURES_PER_FRAME


def test_while_the_agent_is_silent_the_node_publishes_held_frames_after_one_idle_picture(
    video,
):
    worker, session, _ = video
    session.user(0)
    session.user(200)
    session.wait_outputs(10)
    assert session.timeline() == [(40 * k, None) for k in range(10)]
    assert all(
        chunk.modality == VIDEO and chunk.format == JPEG_FRAME
        for chunk in session.outputs
    )
    assert [chunk.payload for chunk in session.outputs] == [IDLE] + [b""] * 9
    assert worker.windows == []


def test_a_reply_starts_a_render_lead_past_the_cursor_and_its_chunks_render_ahead_once_their_audio_is_in(
    video,
):
    worker, session, _ = video
    session.user(0)
    session.wait_outputs(5)
    session.speech(500, turn=1)
    session.wait_outputs(15)
    # The reply plays from 280 ms (200 + the 80 ms lead): the first chunk opens with two listening frames.
    assert session.timeline() == [(40 * k, None) for k in range(5)] + [
        (200 + 40 * k, None if k < 2 else 1) for k in range(10)
    ]
    assert [chunk.payload for chunk in session.outputs[5:10]] == [
        f"w1:{k}".encode() for k in range(5)
    ]
    # The window reads the 100 ms before the chunk, the chunk and 40 ms after it; the reply's audio from 280 ms.
    first = worker.windows[0]
    assert first.size == VIDEO_GEOMETRY.window_samples
    assert not first[: 1600 + 1280 - 40].any()
    assert np.allclose(first[1600 + 1280 + 40 :], 0.5, atol=0.02)
    # The third chunk (600-800 ms) reads to 840 ms, past the reply's end (780 ms): it waits for the turn to end.
    assert len(worker.windows) == 2
    session.turn_done(1)
    session.wait_outputs(20)
    assert [turn for _, turn in session.timeline()[15:]] == [1] * 5
    assert [(s_in, s_out) for _, s_in, s_out in worker.requests] == [
        (b"", b"w1"),
        (b"w1", b"w2"),
        (b"w2", b"w3"),
    ]
    session.user(400)
    session.user(600)
    session.user(800)
    session.wait_outputs(25)
    assert session.timeline()[20:] == [(800 + 40 * k, None) for k in range(5)]
    assert all(chunk.payload == b"" for chunk in session.outputs[20:])


def test_a_reply_shorter_than_a_chunk_starts_once_its_audio_went_quiet_and_renders_by_its_deadline(
    video,
):
    worker, session, clock = video
    session.user(0)
    session.wait_outputs(5)
    session.speech(100, turn=1)
    session.user(200)
    session.wait_outputs(10)
    assert worker.windows == []
    clock.now += 0.2
    session.user(400)
    session.wait_outputs(15)
    # Placed at 480 ms (400 + the lead), rendered with the user's audio at the chunk's start though the turn is not done.
    assert session.timeline()[10:] == [
        (400 + 40 * k, None if k < 2 else 1) for k in range(5)
    ]
    assert len(worker.windows) == 1


def test_an_interrupt_cuts_the_reply_at_the_next_chunk_and_the_head_holds_its_last_picture(
    video,
):
    worker, session, _ = video
    worker.gate.clear()
    session.user(0)
    session.wait_outputs(5)
    session.speech(2000, turn=1)
    assert worker.rendering.wait(WAIT_S)
    session.hooks.control(session.identity, control_chunk(INTERRUPT, turn=1))
    worker.gate.set()
    session.wait_outputs(10)
    session.user(200)
    session.user(400)
    session.wait_outputs(15)
    assert [turn for _, turn in session.timeline()] == [None] * 7 + [1] * 3 + [None] * 5
    assert len(worker.windows) == 1
    assert all(chunk.payload == b"" for chunk in session.outputs[10:])


def test_an_interrupt_takes_back_the_chunks_rendered_ahead_and_the_next_reply_starts_where_the_client_is(
    video,
):
    worker, session, _ = video
    session.user(0)
    session.wait_outputs(5)
    session.speech(2000, turn=1)
    # The reply (280-2280 ms) has all its audio, so ten chunks render ahead, to 2200 ms.
    session.wait_outputs(55)
    session.user(200)
    session.hooks.control(session.identity, control_chunk(INTERRUPT, turn=1))
    # The first chunk at or after 480 ms (the user's audio plus the render lead) starts at 600 ms: the cursor goes back there.
    assert session.hooks.sessions[session.identity].cursor == 9600
    session.speech(500, turn=2)
    session.wait_outputs(60)
    assert session.timeline()[55:] == [
        (600 + 40 * k, None if k < 2 else 2) for k in range(5)
    ]
    # It renders from the state the chunk at 600 ms rendered from, not the last chunk's.
    assert worker.requests[-1][1] == worker.requests[1][2] == b"w2"


def test_a_failed_worker_holds_the_chunk_and_the_next_one_renders_on_another_from_the_last_state():
    requests: list = []
    workers = [
        ScriptedWorker("a", requests=requests),
        ScriptedWorker("b", requests=requests),
    ]
    session = Session(video_hooks(workers))
    try:
        for worker in workers:
            worker.error = WorkerUnavailable("down")
        workers[1].error = None
        workers[0].healthy.clear()
        session.user(0)
        session.wait_outputs(5)
        session.speech(1000, turn=1)
        session.turn_done(1)
        # The reply (280-1280 ms) takes six chunks from 200 ms.
        session.wait_outputs(35)
        rendered = [chunk.payload for chunk in session.outputs[5:]]
        held = [payload for payload in rendered if payload == b""]
        assert len(rendered) == 30 and len(held) in (0, 5)
        assert session.errors == []
        states_in = [s_in for _, s_in, _ in requests]
        assert states_in[0] == b""
        assert all(s_in == prev for (_, _, prev), s_in in zip(requests, states_in[1:]))
    finally:
        stop(session.hooks, workers)


def test_with_one_worker_a_failed_chunk_fails_the_session(video):
    worker, session, _ = video
    worker.error = WorkerUnavailable("down")
    worker.healthy.clear()
    session.user(0)
    session.speech(500, turn=1)
    session.wait_for(lambda: session.errors)
    assert isinstance(session.errors[0], WorkerUnavailable)
    assert [chunk.payload for chunk in session.outputs[5:10]] == [b""] * 5


def test_motion_blocks_follow_the_users_audio_and_carry_the_turn_of_the_reply_they_show(
    motion,
):
    worker, session = motion
    session.user(0, ms=500)
    session.wait_outputs(1)
    session.speech(400, turn=1)
    session.user(500, ms=500)
    session.user(1000, ms=500)
    session.wait_outputs(3)
    assert [(c.t_start_ms, c.duration_ms, c.turn) for c in session.outputs] == [
        (0, 500, None),
        (500, 500, 1),
        (1000, 500, None),
    ]
    assert all(
        c.modality == MOTION and c.format == SMPLX_MOTION and len(c.payload) == 15 * ROW
        for c in session.outputs
    )
    assert all(w.size == MOTION_GEOMETRY.window_samples for w in worker.windows)
    # The block from 500 ms hears the reply from its start, at the window's last 500 ms.
    assert np.allclose(worker.windows[1][24000 + 100 : 24000 + 6000], 0.5, atol=0.02)
    assert not worker.windows[0].any()


def test_a_failed_motion_block_repeats_the_last_pose(motion):
    worker, session = motion
    session.user(0, ms=500)
    session.wait_outputs(1)
    worker.error = WorkerUnavailable("down")
    session.hooks.workers.append(ScriptedWorker("spare", kind="motion"))
    session.user(500, ms=500)
    session.wait_outputs(2)
    last = session.outputs[0].payload[-ROW:]
    assert session.outputs[1].payload == last * 15


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


class FakeModel:
    """A chunk_server model of 2 frames at 25 fps with 160 samples of context and 80 of look-ahead; its state counts chunks."""

    info = {
        "kind": "video",
        "fps": 25,
        "chunk_frames": 2,
        "context_samples": 160,
        "lookahead_samples": 80,
        "width": 8,
        "height": 8,
    }

    def __init__(self) -> None:
        self.speech: list[np.ndarray] = []
        self.fail = False

    def idle_jpeg(self) -> bytes:
        return IDLE

    def window_samples(self) -> int:
        return 160 + 2 * 640 + 80

    def render(self, state: bytes, speech: np.ndarray):
        if self.fail:
            raise RuntimeError("model failed")
        else:
            self.speech.append(speech)
        count = int(state or b"0") + 1
        return str(count).encode(), [b"frame-a%d" % count, b"frame-bb%d" % count]


@pytest.fixture(scope="module")
def served():
    sys.path.insert(0, str(WORKERS))
    import chunk_server

    model = FakeModel()
    port = free_port()
    threading.Thread(
        target=chunk_server.serve,
        args=(lambda: model, "127.0.0.1", port),
        daemon=True,
    ).start()
    worker = HttpChunkWorker.connect(
        f"http://127.0.0.1:{port}", startup_timeout_s=WAIT_S
    )
    yield model, worker
    sys.path.remove(str(WORKERS))


def test_the_worker_service_reports_its_geometry_and_idle_picture(served):
    _, worker = served
    assert worker.geometry == ChunkGeometry(
        fps=25, chunk_frames=2, context_samples=160, lookahead_samples=80
    )
    assert worker.idle_picture() == IDLE
    assert worker.is_healthy()


def test_a_render_round_trip_carries_the_state_and_splits_the_payloads(served):
    model, worker = served
    speech = np.linspace(-0.5, 0.5, worker.geometry.window_samples, dtype=np.float32)
    state, payloads = worker.render(b"", speech)
    assert (state, payloads) == (b"1", [b"frame-a1", b"frame-bb1"])
    state, payloads = worker.render(state, speech)
    assert (state, payloads) == (b"2", [b"frame-a2", b"frame-bb2"])
    assert np.allclose(model.speech[-1], speech, atol=1 / 32768)


def test_a_bad_window_is_the_requests_fault_and_a_model_error_the_workers(served):
    model, worker = served
    with pytest.raises(httpx.HTTPStatusError):
        worker.render(b"", np.zeros(10, dtype=np.float32))
    model.fail = True
    try:
        with pytest.raises(WorkerUnavailable):
            worker.render(
                b"", np.zeros(worker.geometry.window_samples, dtype=np.float32)
            )
    finally:
        model.fail = False


def test_a_worker_that_does_not_answer_is_unavailable():
    worker = HttpChunkWorker(
        httpx.Client(base_url=f"http://127.0.0.1:{free_port()}", timeout=1),
        dict(FakeModel.info),
    )
    with pytest.raises(WorkerUnavailable):
        worker.render(b"", np.zeros(worker.geometry.window_samples, dtype=np.float32))
    assert not worker.is_healthy()


def test_the_request_body_is_the_state_length_the_state_and_int16_speech():
    seen = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request.content)
        return httpx.Response(
            200,
            content=b"st" + b"xyz",
            headers={"X-State-Length-Bytes": "2", "X-Payload-Lengths": "1,2"},
        )

    worker = HttpChunkWorker(
        httpx.Client(base_url="http://worker", transport=httpx.MockTransport(answer)),
        dict(FakeModel.info),
    )
    speech = np.zeros(worker.geometry.window_samples, dtype=np.float32)
    speech[0] = 0.5
    assert worker.render(b"abc", speech) == (b"st", [b"x", b"yz"])
    (body,) = seen
    assert struct.unpack_from("<I", body)[0] == 3 and body[4:7] == b"abc"
    pcm = np.frombuffer(body[7:], dtype="<i2")
    assert pcm.size == speech.size and pcm[0] == 16384 and not pcm[1:].any()
