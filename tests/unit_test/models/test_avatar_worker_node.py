# SPDX-License-Identifier: Apache-2.0
"""The avatar_worker node of a Session graph: the profiles of the models it serves, its chunk geometry, the pinned worker pool and the chunk worker wire, against fake workers."""

import importlib.util
import json
import struct
import sys
import threading
from pathlib import Path

import httpx
import numpy as np
import pytest
import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.avatar_worker.formats import (
    BVH_MOTION,
    JPEG_FRAME,
    MOTION,
    VIDEO,
    bvh_features,
)
from sglang_omni.models.avatar_worker.node import (
    AVATAR_WORKER_STAGE,
    AvatarWorkerFactoryArgs,
)
from sglang_omni.models.avatar_worker.renderer import (
    ChunkWorkerRenderer,
    check_worker,
    geometry_of,
)
from sglang_omni.models.avtr.session import (
    AvatarHooks,
    ChunkGeometry,
    RendererUnavailable,
)
from sglang_omni.proto.session import ControlEdge
from sglang_omni.proto.streams import INTERRUPT, PCM16, PCM_F32, SESSION_CONTROL
from tests.unit_test.fixtures.session_outputs import WAIT_S, control_chunk
from tests.unit_test.models.test_avtr_avatar_node import Avatar

ROOT = Path(__file__).parents[3]
PROFILES = ROOT / "examples/streaming_cascade/profiles"
WORKERS = ROOT / "examples/avatar_workers"
SPEECH_RATE_HZ = 24000
# name: (media, fps, chunk frames, look-ahead ms, listens)
MODELS = {
    "avatarforcing": (JPEG_FRAME, 25, 16, 2240, False),
    "avatarforcing-ki": (JPEG_FRAME, 25, 10, 80, True),
    "dystream": (JPEG_FRAME, 25, 10, 80, True),
    "sentiavatar": (BVH_MOTION, 30, 24, 1200, False),
}
SKELETON = {
    "joints": ["pelvis", "spine", "head"],
    "parents": [-1, 0, 1],
    "offsets": [[0, 0, 100], [0, 0, 20], [0, 0, 30]],
    "rotation_orders": ["zxy"] * 3,
    "up": "z",
    "face": ["JawOpen"],
}


def load_profile(name: str, tmp_path: Path):
    """The model's cascade profile, its skeleton (for bvh_motion) written to tmp_path."""
    path = PROFILES / f"qwen3-asr-qwen3-tts-base-clone-{name}/launch.yaml"
    data = yaml.safe_load(path.read_text())
    factory = data["stages"]["avatar_worker"]["factory"]
    if "skeleton_path" in factory:
        skeleton = tmp_path / "skeleton.json"
        skeleton.write_text(json.dumps(SKELETON))
        factory["skeleton_path"] = str(skeleton)
    else:
        pass
    data["model_path"] = str(path.parent)
    patched = tmp_path / "launch.yaml"
    patched.write_text(yaml.safe_dump(data))
    return ConfigManager.from_file(str(patched)).config


@pytest.mark.parametrize("name", sorted(MODELS))
def test_each_profile_feeds_both_parties_audio_to_the_node_and_publishes_an_input_timeline_port(
    name, tmp_path
):
    media, fps, chunk_frames, lookahead_ms, listens = MODELS[name]
    config = load_profile(name, tmp_path)
    graph = config.session_graph()
    node = AVATAR_WORKER_STAGE
    stage = config.stage_named(node)
    assert node in graph.reading_nodes("input", PCM16)
    assert node in graph.reading_nodes("tts", PCM_F32)
    assert node in graph.reading_nodes("tts", SESSION_CONTROL)
    assert (node, ControlEdge("dialog", INTERRUPT)) in graph.control_targets(
        "dialog", INTERRUPT
    )
    assert graph.publish[node] == (media,)
    assert graph.input_timed == frozenset({(node, media)})
    (port,) = config.session_ports().plugin_output
    chunk_ms = 1000 * chunk_frames / fps
    assert port.format == media
    assert port.params["timeline"] == "input" and port.params["follows_replies"]
    assert port.params["fps"] == fps
    assert port.params["reply_lookahead_ms"] == round(chunk_ms + lookahead_ms)
    assert port.params["reply_media_wait_ms"] >= 2 * chunk_ms + lookahead_ms
    assert stage.factory.listens is listens
    assert stage.factory.speech_sample_rate_hz == SPEECH_RATE_HZ
    if media == BVH_MOTION:
        assert port.modality == MOTION
        assert list(port.params["joints"]) == SKELETON["joints"]
        assert port.params["up"] == "z"
        assert port.params["quat_layout"] == "wxyz"
        assert hash(port)
        assert port.params["features"] == bvh_features(3, 1)
    else:
        assert port.modality == VIDEO
        assert (port.params["width"], port.params["height"]) == (
            stage.factory.width,
            stage.factory.height,
        )
    geometry = geometry_of(stage.factory)
    assert (geometry.fps, geometry.chunk_frames) == (fps, chunk_frames)
    assert geometry.lookahead_samples == lookahead_ms * 16


@pytest.mark.parametrize("name", ["dystream", "avatarforcing_ki"])
def test_the_personaplex_examples_carry_a_listening_node_behind_the_reply_stream(name):
    config = ConfigManager.from_file(
        str(ROOT / f"examples/configs/personaplex_session_{name}.yaml")
    ).config
    graph = config.session_graph()
    assert graph.input_timed == frozenset({(AVATAR_WORKER_STAGE, JPEG_FRAME)})
    factory = config.stage_named(AVATAR_WORKER_STAGE).factory
    assert factory.reply_stream and factory.listens
    kwargs = config.stage_factory_kwargs(AVATAR_WORKER_STAGE)
    assert kwargs["speech_sample_rate_hz"] == kwargs["input_sample_rate_hz"] == 24000


def test_a_bvh_motion_stage_without_a_skeleton_is_refused():
    with pytest.raises(ValueError, match="skeleton_path"):
        geometry_of(AvatarWorkerFactoryArgs(media=BVH_MOTION))


def test_a_geometry_of_a_fractional_number_of_samples_is_refused():
    with pytest.raises(ValueError, match="whole number"):
        ChunkGeometry(30, 1, 0)
    assert ChunkGeometry(30, 24, 0).chunk_samples == 12800
    assert ChunkGeometry(30, 3, 0).frame_starts(0) == [0, 533, 1067, 1600]


class GeometryRenderer:
    """A worker of any geometry: answers `frames` payloads naming the call and a handle state naming the worker; a render waits while `gate` is clear."""

    def __init__(self, name: str, frames: int, log: list | None = None) -> None:
        self.name = name
        self.frames = frames
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        self.log = [] if log is None else log
        self.released: list[bytes] = []
        self.gate = threading.Event()
        self.gate.set()
        self.error: Exception | None = None
        self.healthy = threading.Event()
        self.healthy.set()

    def render(self, state, speech, listen, frame):
        self.windows.append((speech.copy(), listen.copy()))
        self.gate.wait()
        if self.error is not None:
            raise self.error
        else:
            pass
        for index in range(self.frames):
            frame(f"{self.name}{len(self.windows)}:{index}".encode())
        self.log.append((self.name, state))
        return state or f"{self.name}-session".encode()

    def release(self, state: bytes) -> None:
        self.released.append(state)

    def is_healthy(self) -> bool:
        return self.healthy.is_set()


def stop(hooks: AvatarHooks, renderers) -> None:
    for renderer in renderers:
        renderer.gate.set()
        renderer.healthy.set()
    for session in list(hooks.sessions):
        hooks.close(session)
    for worker in list(hooks.workers):
        if worker is not None:
            worker.join(WAIT_S)
        else:
            pass


DYSTREAM = ChunkGeometry(25, 10, 1280)
SENTI = ChunkGeometry(30, 24, 19200)


def test_a_listening_model_hears_the_user_one_window_behind_and_its_frames_are_stamped_at_its_rate():
    renderer = GeometryRenderer("a", 10)
    hooks = AvatarHooks([renderer], SPEECH_RATE_HZ, geometry=DYSTREAM, pinned=True)
    session = Avatar(hooks)
    try:
        for index in range(3):
            session.user(index * 400, ms=400)
            session.wait_frames(10 * (index + 1))
        assert session.timeline() == [(400 + 40 * k, 40, None) for k in range(30)]
        speech, listen = renderer.windows[0]
        window = DYSTREAM.window_samples
        assert speech.size == listen.size == window
        assert listen[window - 6400 :].tolist() == [0.25] * 6400
        assert not listen[: window - 6400].any()
    finally:
        stop(hooks, [renderer])


def test_a_motion_model_at_30_fps_stamps_rows_on_rounded_frame_starts_and_carries_the_reply_turn():
    renderer = GeometryRenderer("a", 24)
    hooks = AvatarHooks(
        [renderer],
        SPEECH_RATE_HZ,
        geometry=SENTI,
        stream=MOTION,
        media_format=BVH_MOTION,
        pinned=True,
    )
    session = Avatar(hooks)
    try:
        session.user(0, ms=800)
        session.wait_frames(24)
        session.speech(2000, turn=3)
        session.turn_done(3)
        for index in range(1, 5):
            session.user(index * 800, ms=800)
            session.wait_frames(24 * (index + 1))
        frames = session.frames
        assert all(f.modality == MOTION and f.format == BVH_MOTION for f in frames)
        starts = [f.t_start_ms for f in frames[:4]]
        assert starts == pytest.approx([800, 800 + 33.3125, 800 + 66.6875, 900])
        turns = [f.turn for f in frames]
        assert turns[:24] == [None] * 24
        assert 3 in turns and turns[-1] is None
        speech, _ = renderer.windows[1]
        assert speech.size == SENTI.window_samples
        assert np.allclose(speech[200:12000], 0.5, atol=0.01)
    finally:
        stop(hooks, [renderer])


def test_a_pinned_pool_renders_each_session_on_the_worker_of_its_first_chunk_and_releases_it_on_close():
    log: list = []
    renderers = [GeometryRenderer("a", 10, log), GeometryRenderer("b", 10, log)]
    hooks = AvatarHooks(renderers, SPEECH_RATE_HZ, geometry=DYSTREAM, pinned=True)
    sessions = [Avatar(hooks, "s1"), Avatar(hooks, "s2")]
    try:
        for index in range(4):
            for session in sessions:
                session.user(index * 400, ms=400)
            for session in sessions:
                session.wait_frames(10 * (index + 1))
        for session in sessions:
            frames = [frame.payload.decode()[0] for frame in session.frames]
            assert len(set(frames)) == 1
        assert {
            frames[0]
            for frames in ([s.frames[0].payload.decode()[0]] for s in sessions)
        } <= {"a", "b"}
        states = {}
        for name, state in log:
            states.setdefault(name, []).append(state)
        for name, sent in states.items():
            assert sent[0] == b"" and set(sent[1:]) <= {f"{name}-session".encode()}
        for session in sessions:
            session.close()
        for renderer in renderers:
            for _ in range(100):
                if renderer.released or not any(n == renderer.name for n, _ in log):
                    break
                threading.Event().wait(0.01)
        assert sum(len(r.released) for r in renderers) == 2
    finally:
        stop(hooks, renderers)


def test_a_pinned_session_holds_frames_while_its_worker_is_down_and_renders_there_again_after_recovery():
    renderer = GeometryRenderer("a", 10)
    spare = GeometryRenderer("b", 10)
    hooks = AvatarHooks(
        [renderer, spare],
        SPEECH_RATE_HZ,
        geometry=DYSTREAM,
        pinned=True,
        recovery_probe_s=0.01,
    )
    session = Avatar(hooks)
    try:
        session.user(0, ms=400)
        session.wait_frames(10)
        (first,) = [r for r in (renderer, spare) if r.windows]
        (other,) = [r for r in (renderer, spare) if r is not first]
        first.error = RendererUnavailable("down")
        first.healthy.clear()
        session.user(400, ms=400)
        session.wait_frames(20)
        assert [frame.payload for frame in session.frames[10:20]] == [b""] * 10
        first.error = None
        first.healthy.set()
        for index in range(2, 6):
            session.user(index * 400, ms=400)
            session.wait_frames(10 * (index + 1))
        assert session.errors == []
        assert not other.windows
        assert all(frame.payload for frame in session.frames[-10:])
    finally:
        stop(hooks, [renderer, spare])


def test_an_interrupt_returns_a_listening_model_to_listening_at_the_next_chunk():
    renderer = GeometryRenderer("a", 10)
    hooks = AvatarHooks([renderer], SPEECH_RATE_HZ, geometry=DYSTREAM, pinned=True)
    session = Avatar(hooks)
    try:
        session.user(0, ms=400)
        session.wait_frames(10)
        session.speech(4000, turn=1)
        session.user(400, ms=400)
        session.wait_frames(20)
        session.hooks.control(session.identity, control_chunk(INTERRUPT, turn=1))
        session.user(800, ms=400)
        session.wait_frames(30)
        assert [f.turn for f in session.frames] == [None] * 10 + [1] * 10 + [None] * 10
        assert not renderer.windows[2][0].any()
    finally:
        stop(hooks, [renderer])


class WorkerService:
    """A chunk worker over httpx.MockTransport: records each /render body and answers `frames` payloads after a handle state."""

    def __init__(self, frames: int = 3, listens: bool = True) -> None:
        self.frames = frames
        self.listens = listens
        self.bodies: list[bytes] = []
        self.released: list[bytes] = []
        self.answers: list = []
        self.health = 200

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(self.health)
        elif request.url.path == "/release":
            self.released.append(request.content)
            return httpx.Response(200)
        elif self.answers:
            answer = self.answers.pop(0)
            if isinstance(answer, Exception):
                raise answer
            else:
                return httpx.Response(answer, content=b"broken")
        else:
            self.bodies.append(request.content)
            state = b"handle-1"
            payloads = [f"frame{index}".encode() for index in range(self.frames)]
            return httpx.Response(
                200,
                content=state + b"".join(payloads),
                headers={
                    "X-State-Length-Bytes": str(len(state)),
                    "X-Payload-Lengths": ",".join(str(len(p)) for p in payloads),
                    "X-Render-Ms": "12.0",
                },
            )


def worker_renderer(service: WorkerService) -> ChunkWorkerRenderer:
    client = httpx.Client(
        base_url="http://worker", transport=httpx.MockTransport(service)
    )
    return ChunkWorkerRenderer(client, listens=service.listens)


def test_the_wire_sends_the_state_then_both_windows_and_splits_the_answer_into_state_and_frames():
    service = WorkerService()
    renderer = worker_renderer(service)
    speech = np.full(800, 0.5, dtype=np.float32)
    listen = np.full(800, -0.25, dtype=np.float32)
    frames: list[bytes] = []
    state = renderer.render(b"", speech, listen, frames.append)
    renderer.render(state, speech, listen, frames.append)
    assert state == b"handle-1"
    assert frames == [b"frame0", b"frame1", b"frame2"] * 2
    first, second = service.bodies
    assert struct.unpack_from("<I", first)[0] == 0
    assert len(first) == 4 + 2 * 800 * 2
    pcm = np.frombuffer(first[4:], dtype="<i2")
    assert pcm[:800].tolist() == [16384] * 800 and pcm[800:].tolist() == [-8192] * 800
    assert second[4 : 4 + 8] == b"handle-1"
    renderer.release(state)
    assert service.released == [b"handle-1"]


def test_a_speech_only_worker_is_sent_no_listening_audio():
    service = WorkerService(listens=False)
    renderer = worker_renderer(service)
    window = np.zeros(640, dtype=np.float32)
    renderer.render(b"", window, window, lambda frame: None)
    assert len(service.bodies[0]) == 4 + 2 * 640


@pytest.mark.parametrize(
    ("answer", "is_worker_failure"),
    [
        (503, True),
        (500, True),
        (httpx.ConnectError("refused"), True),
        (httpx.ReadTimeout("slow"), True),
        (400, False),
    ],
)
def test_a_worker_down_is_unavailable_and_a_bad_request_is_the_sessions_error(
    answer, is_worker_failure
):
    service = WorkerService()
    service.answers = [answer]
    renderer = worker_renderer(service)
    window = np.zeros(800, dtype=np.float32)
    expected = RendererUnavailable if is_worker_failure else httpx.HTTPStatusError
    with pytest.raises(expected):
        renderer.render(b"", window, window, lambda frame: None)
    service.health = 503
    assert not renderer.is_healthy()


def test_a_worker_of_another_geometry_is_refused_at_startup():
    factory = AvatarWorkerFactoryArgs(
        fps=25, chunk_frames=10, lookahead_ms=80, listens=True, width=512, height=512
    )
    geometry = geometry_of(factory)
    info = {
        "fps": 25,
        "chunk_frames": 10,
        "lookahead_samples": 1280,
        "listens": True,
        "media": JPEG_FRAME,
        "width": 512,
        "height": 512,
    }
    check_worker("http://w", info, geometry, factory)
    with pytest.raises(ValueError, match="chunk_frames"):
        check_worker("http://w", info | {"chunk_frames": 16}, geometry, factory)


def test_the_worker_side_server_parses_both_windows_and_keeps_sessions_by_handle():
    # Loaded from its file under a name of its own: examples/speech_avatar_workers has a chunk_server module too, which other tests import.
    spec = importlib.util.spec_from_file_location(
        "avatar_workers_chunk_server", WORKERS / "chunk_server.py"
    )
    chunk_server = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = chunk_server
    spec.loader.exec_module(chunk_server)
    try:
        body = struct.pack("<I", 2) + b"hi" + np.arange(6, dtype="<i2").tobytes()
        state, speech, listen = chunk_server.parse_render_body(body, 3, True)
        assert state == b"hi" and speech.tolist() == pytest.approx(
            [0, 1 / 32768, 2 / 32768]
        )
        assert listen.size == 3
        with pytest.raises(chunk_server.BadRequest):
            chunk_server.parse_render_body(body, 3, False)

        class Counter(chunk_server.SessionModel):
            info = {"fps": 25, "chunk_frames": 2, "lookahead_samples": 0}
            max_sessions = 2

            def start(self):
                return {"n": 0}

            def step(self, session, speech, listen):
                session["n"] += 1
                return [str(session["n"]).encode()] * 2

        model = Counter()
        window = np.zeros(model.window_samples(), dtype=np.float32)
        handle, frames = model.render(b"", window, None)
        handle, frames = model.render(handle, window, None)
        assert frames == [b"2", b"2"]
        other, _ = model.render(b"", window, None)
        third, _ = model.render(b"", window, None)
        assert len(model.sessions) == 2 and handle not in model.sessions
        model.release(other)
        assert set(model.sessions) == {third}
    finally:
        sys.modules.pop(spec.name, None)


def test_the_stage_waits_for_a_worker_still_loading_and_gives_up_after_its_timeout(
    monkeypatch,
):
    from sglang_omni.models.avatar_worker import renderer as worker_renderer_module

    monkeypatch.setattr(worker_renderer_module, "STARTUP_PROBE_S", 0.0)
    answers = [httpx.ConnectError("refused"), 503, 503, 200]

    def health(request: httpx.Request) -> httpx.Response:
        answer = answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        else:
            return httpx.Response(answer)

    client = httpx.Client(base_url="http://w", transport=httpx.MockTransport(health))
    worker_renderer_module.wait_until_healthy(client, timeout_s=10)
    assert answers == []
    down = httpx.Client(
        base_url="http://w",
        transport=httpx.MockTransport(lambda request: httpx.Response(503)),
    )
    with pytest.raises(TimeoutError, match="did not answer /health"):
        worker_renderer_module.wait_until_healthy(down, timeout_s=0)
