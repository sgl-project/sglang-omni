# SPDX-License-Identifier: Apache-2.0
"""The Live Avatar node of a Session graph: a talking head on the input timeline whose worker pipelines several blocks of a Session at once."""

import importlib.util
import io
import json
import struct
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from PIL import Image

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.liveavatar.formats import (
    JPEG_FRAME,
    REPLY_MEDIA_WAIT_SLACK_MS,
    jpeg_frame_port,
)
from sglang_omni.models.liveavatar.node import LIVEAVATAR_STAGE
from sglang_omni.models.liveavatar.session import HANDLE, PipelinedVideoHooks
from sglang_omni.models.liveavatar.worker import check_workers
from sglang_omni.models.speech_avatar.session import ChunkGeometry, WorkerUnavailable
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
    / "examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-base-clone-liveavatar/launch.yaml"
)
WORKER_SCRIPT = ROOT / "examples/speech_avatar_workers/liveavatar_worker.py"
SPEECH_RATE_HZ = 24000
# Live Avatar's block: 12 frames at 25 fps, 7680 samples at 16 kHz; the worker's default window around it.
GEOMETRY = ChunkGeometry(
    fps=25, chunk_frames=12, context_samples=16000, lookahead_samples=3840
)
FRAME = 640
CHUNK = 7680
DEPTH = 5
IDLE = b"idle"


class FakePipelinedWorker:
    """Answers twelve frames naming the chunk's handle; each render waits for its index's gate (all open by default), and the indices in `failing` raise WorkerUnavailable."""

    def __init__(self) -> None:
        self.lock = threading.Condition()
        self.handles: list[tuple[bytes, int]] = []
        self.windows: list[np.ndarray] = []
        self.active = 0
        self.most_active = 0
        self.open_all = True
        self.opened: set[int] = set()
        self.failing: set[int] = set()
        self.fail_all = False

    def release(self, *indices: int) -> None:
        with self.lock:
            self.opened.update(indices)
            self.lock.notify_all()

    def release_all(self) -> None:
        with self.lock:
            self.open_all = True
            self.lock.notify_all()

    def render(self, state: bytes, speech: np.ndarray):
        key, index = HANDLE.unpack(state)
        with self.lock:
            self.handles.append((key, index))
            self.windows.append(speech.copy())
            self.active += 1
            self.most_active = max(self.most_active, self.active)
            self.lock.notify_all()
            self.lock.wait_for(lambda: self.open_all or index in self.opened)
            self.active -= 1
        if self.fail_all or index in self.failing:
            raise WorkerUnavailable(f"chunk {index} failed")
        else:
            return HANDLE.pack(key, index + 1), [
                f"{index}:{frame}".encode() for frame in range(GEOMETRY.chunk_frames)
            ]

    def is_healthy(self) -> bool:
        return True

    def wait_for(self, condition) -> None:
        with self.lock:
            assert self.lock.wait_for(condition, timeout=WAIT_S)


class Avatar:
    """One Session of the hooks, appended and read the way the Session scheduler does."""

    def __init__(self, hooks: PipelinedVideoHooks, name: str = "session") -> None:
        self.hooks = hooks
        self.identity = SessionIdentity(name)
        self.frames: list[TimedChunk] = []
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
            self.frames.append(chunk)
            self.changed.notify_all()

    def fail(self, error: BaseException) -> None:
        with self.changed:
            self.errors.append(error)
            self.changed.notify_all()

    def wait_for(self, condition) -> None:
        with self.changed:
            assert self.changed.wait_for(condition, timeout=WAIT_S)

    def wait_frames(self, count: int) -> None:
        self.wait_for(lambda: len(self.frames) >= count)

    def append(self, chunk: TimedChunk) -> None:
        self.hooks.append(
            chunk, StagePayload("x", OmniRequest(None), None), self.context
        )

    def user(self, t_ms: float, ms: float = 200) -> None:
        pcm = np.full(int(ms * 16), 8192, dtype="<i2").tobytes()
        self.append(TimedChunk(AUDIO, t_ms, ms, 0, pcm, format=PCM16))

    def speech(self, ms: float, turn: int, value: float = 0.5) -> None:
        pcm = np.full(int(ms * SPEECH_RATE_HZ / 1000), value, dtype="<f4").tobytes()
        self.append(TimedChunk(AUDIO, 0, ms, 0, pcm, format=PCM_F32, turn=turn))

    def turn_done(self, turn: int) -> None:
        self.append(control_chunk(TURN_DONE, turn=turn))

    def reply_frames(self) -> list[TimedChunk]:
        with self.changed:
            return [frame for frame in self.frames if frame.turn is not None]

    def rendered(self) -> list[TimedChunk]:
        """The frames after the five listening frames of the first 200 ms of the user's audio."""
        with self.changed:
            return self.frames[5:]


def make_hooks(worker: FakePipelinedWorker, **overrides) -> PipelinedVideoHooks:
    options = {"render_lead_ms": 0.0, "depth": DEPTH, "recovery_probe_s": 0.01}
    options.update(overrides)
    return PipelinedVideoHooks(
        [worker], GEOMETRY, SPEECH_RATE_HZ, idle_picture=IDLE, **options
    )


def stop(hooks: PipelinedVideoHooks, worker: FakePipelinedWorker) -> None:
    worker.release_all()
    for session in list(hooks.sessions):
        hooks.close(session)
    for thread in list(hooks.threads):
        if thread is not None:
            thread.join(WAIT_S)
        else:
            pass


@pytest.fixture
def avatar():
    worker = FakePipelinedWorker()
    session = Avatar(make_hooks(worker))
    yield worker, session
    stop(session.hooks, worker)


def test_the_profile_feeds_the_users_and_the_agents_audio_to_the_node_and_publishes_an_input_timeline_port():
    config = ConfigManager.from_file(str(PROFILE)).config
    graph = config.session_graph()
    stage = config.stage_named(LIVEAVATAR_STAGE)
    assert "liveavatar" in graph.reading_nodes("input", PCM16)
    assert "liveavatar" in graph.reading_nodes("tts", PCM_F32)
    assert "liveavatar" in graph.reading_nodes("tts", SESSION_CONTROL)
    assert ("liveavatar", ControlEdge("dialog", INTERRUPT)) in graph.control_targets(
        "dialog", INTERRUPT
    )
    assert graph.publish["liveavatar"] == (JPEG_FRAME,)
    assert graph.input_timed == frozenset({("liveavatar", JPEG_FRAME)})
    factory = stage.factory
    assert config.session_ports().plugin_output == (
        jpeg_frame_port(
            width=factory.width,
            height=factory.height,
            render_lead_ms=factory.render_lead_ms,
        ),
    )
    # The recommended layout (one stage of four sequence-parallel ranks and a decoder) keeps two blocks in flight.
    assert factory.pipeline_depth == 2 and len(factory.worker_urls) == 1


def test_the_port_waits_for_a_replys_first_picture_as_long_as_the_render_lead_and_some_slack():
    config = ConfigManager.from_file(str(PROFILE)).config
    config.stage_named(LIVEAVATAR_STAGE).factory.render_lead_ms = 1500
    config.stage_named(LIVEAVATAR_STAGE).factory.width = 320
    (port,) = config.session_ports().plugin_output
    assert port.params["reply_lookahead_ms"] == 1500
    assert port.params["reply_media_wait_ms"] == 1500 + REPLY_MEDIA_WAIT_SLACK_MS
    assert port.params["width"] == 320 and port.params["fps"] == 25
    assert port.params["timeline"] == "input" and port.params["follows_replies"]


def test_while_no_reply_plays_frames_follow_the_users_audio_and_only_the_first_carries_a_picture(
    avatar,
):
    worker, session = avatar
    session.user(0)
    session.user(200)
    session.wait_frames(10)
    assert [(f.t_start_ms, f.duration_ms, f.turn) for f in session.frames] == [
        (40 * k, 40, None) for k in range(10)
    ]
    assert session.frames[0].payload == IDLE
    assert all(frame.payload == b"" for frame in session.frames[1:])
    assert worker.handles == []


def test_a_reply_fills_the_pipeline_with_its_blocks_under_one_session_key(avatar):
    worker, session = avatar
    worker.open_all = False
    session.user(0)
    session.wait_frames(5)
    # 3 s of reply: five blocks have their window's speech, the sixth and seventh wait for the turn to end.
    session.speech(3000, turn=1)
    worker.wait_for(lambda: worker.active == DEPTH)
    assert sorted(index for _, index in worker.handles) == list(range(DEPTH))
    assert len({key for key, _ in worker.handles}) == 1
    session.turn_done(1)
    time.sleep(0.1)
    assert len(worker.handles) == DEPTH, "a sixth block waits for room in the pipeline"
    worker.release_all()
    session.wait_for(lambda: len(session.rendered()) == 7 * 12)
    assert worker.most_active == DEPTH
    assert sorted(index for _, index in worker.handles) == list(range(7))
    frames = session.rendered()
    assert [f.t_start_ms for f in frames] == [200 + 40 * k for k in range(7 * 12)]
    # The reply's 3 s cover 75 frames; the rest of its last block shows the model's silence and carries the reply's turn, so a client dropping the reply drops it too.
    assert [f.turn for f in frames] == [1] * 84
    assert frames[0].payload == b"0:0" and frames[-1].payload == b"6:11"


def test_blocks_that_come_back_out_of_order_are_published_in_timeline_order(avatar):
    worker, session = avatar
    worker.open_all = False
    session.user(0)
    session.wait_frames(5)
    session.speech(3000, turn=1)
    worker.wait_for(lambda: worker.active == DEPTH)
    worker.release(2, 1)
    worker.wait_for(lambda: worker.active == DEPTH - 2)
    time.sleep(0.1)
    assert session.rendered() == [], "blocks 1 and 2 wait for block 0"
    worker.release(0)
    session.wait_for(lambda: len(session.rendered()) == 3 * 12)
    assert [f.payload for f in session.rendered()] == [
        f"{index}:{frame}".encode() for index in range(3) for frame in range(12)
    ]


def test_each_block_reads_its_own_window_of_the_reply(avatar):
    worker, session = avatar
    session.user(0)
    session.wait_frames(5)
    session.speech(3000, turn=1)
    session.turn_done(1)
    session.wait_for(lambda: len(session.rendered()) == 7 * 12)
    by_index = dict(zip([i for _, i in worker.handles], worker.windows))
    first = by_index[0]
    assert first.size == GEOMETRY.window_samples
    # The reply starts at 200 ms, the block's start: the context before it is silence, the block and its look-ahead speech.
    assert not first[: GEOMETRY.context_samples].any()
    assert np.allclose(first[GEOMETRY.context_samples + 256 :], 0.5, atol=0.02)
    assert np.allclose(
        by_index[1][GEOMETRY.context_samples - CHUNK + 256 :], 0.5, atol=0.02
    )


def test_a_failed_block_holds_the_picture_for_its_frames_and_the_reply_goes_on(avatar):
    worker, session = avatar
    worker.failing = {2}
    session.user(0)
    session.wait_frames(5)
    session.speech(3000, turn=1)
    session.turn_done(1)
    session.wait_for(lambda: len(session.rendered()) == 7 * 12)
    payloads = [f.payload for f in session.rendered()]
    assert payloads[24:36] == [b""] * 12
    assert payloads[36] == b"3:0" and payloads[-1] == b"6:11"
    assert session.errors == []


def test_a_session_whose_blocks_keep_failing_fails_after_every_thread_lost_one_and_one_more():
    worker = FakePipelinedWorker()
    worker.fail_all = True
    session = Avatar(make_hooks(worker))
    try:
        session.user(0)
        session.wait_frames(5)
        session.speech(3000, turn=1)
        session.turn_done(1)
        session.wait_for(lambda: session.errors)
        assert isinstance(session.errors[0], WorkerUnavailable)
        assert session.hooks.max_failed_chunks == DEPTH + 1
        assert len(session.rendered()) >= (DEPTH + 1) * 12
        assert all(frame.payload == b"" for frame in session.rendered())
    finally:
        stop(session.hooks, worker)


def test_a_reply_starts_the_render_lead_past_the_frame_the_node_reached():
    worker = FakePipelinedWorker()
    session = Avatar(make_hooks(worker, render_lead_ms=960.0))
    try:
        session.user(0)
        session.wait_frames(5)
        session.speech(1000, turn=1)
        session.turn_done(1)
        for index in range(1, 8):
            session.user(200 * index)
        session.wait_for(lambda: len(session.reply_frames()) >= 12)
        assert session.reply_frames()[0].t_start_ms == 200 + 960
        frames = session.frames
        assert [f.t_start_ms for f in frames[: len(frames)]] == [
            40 * k for k in range(len(frames))
        ]
    finally:
        stop(session.hooks, worker)


def test_an_interrupt_returns_the_avatar_to_listening_after_the_blocks_in_flight(
    avatar,
):
    worker, session = avatar
    worker.open_all = False
    session.user(0)
    session.wait_frames(5)
    session.speech(3000, turn=1)
    worker.wait_for(lambda: worker.active == DEPTH)
    session.hooks.control(session.identity, control_chunk(INTERRUPT, turn=1))
    session.turn_done(1)
    worker.release_all()
    session.wait_for(lambda: len(session.rendered()) == DEPTH * 12)
    for index in range(1, 12):
        session.user(200 * index)
    time.sleep(0.2)
    assert len(worker.handles) == DEPTH
    assert len(session.rendered()) == DEPTH * 12


def test_a_session_starts_with_silent_primer_blocks_whose_pictures_nobody_sees():
    worker = FakePipelinedWorker()
    worker.open_all = False
    session = Avatar(make_hooks(worker, prime_chunks=3))
    try:
        session.user(0)
        worker.wait_for(lambda: worker.active == 3)
        # Listening frames do not wait for the primers.
        session.wait_frames(5)
        assert all(not window.any() for window in worker.windows)
        worker.release_all()
        session.speech(3000, turn=1)
        session.turn_done(1)
        session.wait_for(lambda: len(session.rendered()) == 7 * 12)
        assert sorted(index for _, index in worker.handles) == list(range(10))
        assert session.rendered()[0].payload == b"3:0"
        assert session.rendered()[-1].payload == b"9:11"
        assert len({key for key, _ in worker.handles}) == 1
    finally:
        stop(session.hooks, worker)


def test_the_hooks_send_the_handle_the_worker_reads():
    module = load_worker_module()
    assert module.HANDLE.format == HANDLE.format
    assert module.FRAMES_PER_BLOCK == GEOMETRY.chunk_frames and module.FPS == 25


def load_worker_module():
    spec = importlib.util.spec_from_file_location("liveavatar_worker", WORKER_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def fake_front(module):
    engine = SimpleNamespace(
        idle_picture=Image.new("RGB", (40, 72)),
        context_samples=GEOMETRY.context_samples,
        lookahead_samples=GEOMETRY.lookahead_samples,
        width=384,
        height=704,
        window_samples=GEOMETRY.window_samples,
        frames_per_block=12,
    )
    args = SimpleNamespace(
        jpeg_quality=80,
        pipeline_depth=DEPTH,
        timing_log=None,
        stages=4,
        sp=1,
        step_list=[0, 1, 2, 3],
        block_frames=3,
        gpu_jpeg=False,
    )
    return module.Front(engine, args)


def test_the_worker_queues_a_sessions_blocks_in_index_order_and_starts_a_session_on_a_new_key():
    module = load_worker_module()
    front = fake_front(module)
    speech = np.zeros(GEOMETRY.window_samples, np.float32)
    key = b"k" * 16
    queued = []
    late = threading.Thread(
        target=lambda: queued.append(front.submit(HANDLE.pack(key, 1), speech))
    )
    late.start()
    time.sleep(0.1)
    assert front.jobs.empty(), "block 1 waits for block 0"
    first, next_state = front.submit(HANDLE.pack(key, 0), speech)
    late.join(WAIT_S)
    assert HANDLE.unpack(next_state) == (key, 1)
    second, _ = queued[0]
    assert [front.jobs.get_nowait().job_id for _ in range(2)] == [
        first.job_id,
        second.job_id,
    ]
    assert first.new_session and not second.new_session
    with pytest.raises(module.BadRequest):
        front.submit(HANDLE.pack(key, 1), speech)
    restarted, _ = front.submit(HANDLE.pack(key, 0), speech)
    assert restarted.new_session, "index 0 starts the Session again"
    fresh, state = front.submit(b"", speech)
    assert fresh.new_session and HANDLE.unpack(state)[1] == 1


def test_the_worker_starts_a_session_it_does_not_know_where_its_blocks_resume_after_waiting_for_its_start():
    module = load_worker_module()
    module.ORDER_TIMEOUT_S = 0.2
    front = fake_front(module)
    speech = np.zeros(GEOMETRY.window_samples, np.float32)
    began = time.monotonic()
    job, state = front.submit(HANDLE.pack(b"r" * 16, 7), speech)
    assert time.monotonic() - began >= 0.2
    assert job.new_session and HANDLE.unpack(state) == (b"r" * 16, 8)
    with pytest.raises(TimeoutError):
        front.submit(HANDLE.pack(b"r" * 16, 9), speech)


def test_the_worker_answers_a_block_with_the_frames_the_decoding_rank_delivers():
    module = load_worker_module()
    front = fake_front(module)
    speech = np.zeros(GEOMETRY.window_samples, np.float32)
    answer = {}

    def render() -> None:
        answer["result"] = front.render(HANDLE.pack(b"k" * 16, 0), speech)

    thread = threading.Thread(target=render)
    thread.start()
    job = front.jobs.get(timeout=WAIT_S)
    now = time.time()
    job.timings["start"] = now
    jpegs = [b"a", b"bb", b"ccc"]
    meta = json.dumps(
        {
            "job": job.job_id,
            "lengths": [len(j) for j in jpegs],
            "timings": {
                "received": now,
                "decoded": now,
                "decode_ms": 1.0,
                "encoded": now,
            },
        }
    ).encode()
    front.deliver(struct.pack("<I", len(meta)) + meta + b"".join(jpegs))
    thread.join(WAIT_S)
    next_state, payloads, timings = answer["result"]
    assert payloads == jpegs and HANDLE.unpack(next_state) == (b"k" * 16, 1)
    assert timings["render_ms"] >= 0
    assert Image.open(io.BytesIO(front.idle)).size == (40, 72)


def test_the_stage_takes_only_pipelined_workers_of_its_picture():
    info = {
        "kind": "video",
        "fps": 25,
        "chunk_frames": 12,
        "context_samples": 16000,
        "lookahead_samples": 3840,
        "width": 384,
        "height": 704,
        "pipeline_depth": 5,
    }

    def worker(**changes) -> HttpChunkWorker:
        client = httpx.Client(
            transport=httpx.MockTransport(lambda request: httpx.Response(200)),
            base_url="http://w",
        )
        return HttpChunkWorker(client, {**info, **changes})

    check_workers(["http://w"], [worker()], 384, 704)
    with pytest.raises(ValueError, match="renders 704x384"):
        check_workers(["http://w"], [worker(width=704, height=384)], 384, 704)
    without_depth = worker()
    del without_depth.info["pipeline_depth"]
    with pytest.raises(ValueError, match="pipeline_depth"):
        check_workers(["http://w"], [without_depth], 384, 704)
