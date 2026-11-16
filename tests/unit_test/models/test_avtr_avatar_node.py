# SPDX-License-Identifier: Apache-2.0
"""The AVTR-1 avatar node of a Session graph: frames on the input timeline from the user's audio and the agent's speech, rendered by a pool of renderer workers."""

import io
import threading
from pathlib import Path

import httpx
import numpy as np
import pytest
from PIL import Image

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.avtr import renderer as renderer_module
from sglang_omni.models.avtr.formats import (
    JPEG_FRAME,
    JPEG_FRAME_PORT,
    VIDEO,
    jpeg_frame_port,
)
from sglang_omni.models.avtr.node import AVATAR_STAGE
from sglang_omni.models.avtr.renderer import AVTR1Renderer, jpeg_of_i420
from sglang_omni.models.avtr.session import (
    WINDOW_SAMPLES,
    AvatarHooks,
    RendererUnavailable,
)
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

PROFILE = (
    Path(__file__).parents[3]
    / "examples/streaming_cascade/profiles/qwen3-asr-qwen3-tts-base-clone-avatar/launch.yaml"
)
SPEECH_RATE_HZ = 24000
# A 200 ms chunk is 3200 samples at 16 kHz; a chunk's listening window is the 6480 samples before it.
CHUNK = 3200


class ScriptedRenderer:
    """Records each chunk's windows and the state it came with, answers five frames naming the call and a state naming the worker; a render waits while `gate` is clear."""

    def __init__(self, name: str = "r", requests: list | None = None) -> None:
        self.name = name
        self.windows: list[tuple[np.ndarray, np.ndarray]] = []
        # (worker, state in, state out) of every render of the pool, in order.
        self.requests = [] if requests is None else requests
        self.gate = threading.Event()
        self.gate.set()
        self.rendering = threading.Event()
        self.error: Exception | None = None
        self.frames_before_error = 0
        self.healthy = threading.Event()
        self.healthy.set()

    def render(self, state, speech, listen, frame):
        self.windows.append((speech.copy(), listen.copy()))
        self.rendering.set()
        self.gate.wait()
        for index in range(self.frames_before_error if self.error else 5):
            frame(f"{self.name}{len(self.windows)}:{index}".encode())
        if self.error is not None:
            raise self.error
        else:
            pass
        next_state = f"{self.name}{len(self.windows)}".encode()
        self.requests.append((self.name, state, next_state))
        return next_state

    def is_healthy(self) -> bool:
        return self.healthy.is_set()


class Avatar:
    """One Session of the avatar hooks, appended and read the way the Session scheduler does."""

    def __init__(self, hooks: AvatarHooks, name: str = "session") -> None:
        self.hooks = hooks
        self.identity = SessionIdentity(name)
        self.frames: list[TimedChunk] = []
        self.errors: list[BaseException] = []
        self.changed = threading.Condition()
        self.context = SessionContext(
            session_identity=self.identity,
            cancelled=threading.Event(),
            emit=self.emit,
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

    def user(
        self, t_ms: float, value: float = 0.25, eos: bool = False, ms: float = 200
    ) -> None:
        pcm = np.full(int(ms * 16), round(value * 32768), dtype="<i2").tobytes()
        self.append(TimedChunk(AUDIO, t_ms, ms, 0, pcm, format=PCM16, eos=eos))

    def speech(self, ms: float, turn: int, value: float = 0.5) -> None:
        pcm = np.full(int(ms * SPEECH_RATE_HZ / 1000), value, dtype="<f4").tobytes()
        self.append(TimedChunk(AUDIO, 0, ms, 0, pcm, format=PCM_F32, turn=turn))

    def turn_done(self, turn: int) -> None:
        self.append(control_chunk(TURN_DONE, turn=turn))

    def timeline(self) -> list[tuple[float, float, int | None]]:
        return [
            (frame.t_start_ms, frame.duration_ms, frame.turn) for frame in self.frames
        ]

    def close(self) -> None:
        if self.identity in self.hooks.sessions:
            self.hooks.close(self.identity)
        else:
            pass


def stop(hooks: AvatarHooks, renderers: list[ScriptedRenderer]) -> None:
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


@pytest.fixture
def avatar():
    """A Session of fresh hooks over one scripted renderer; the Session closes and the worker stops afterwards."""
    renderer = ScriptedRenderer()
    session = Avatar(AvatarHooks([renderer], SPEECH_RATE_HZ))
    yield renderer, session
    stop(session.hooks, [renderer])


@pytest.fixture
def pool():
    """Fresh hooks over two scripted renderers sharing one request log, probing a failed one every 10 ms."""
    requests: list = []
    renderers = [ScriptedRenderer("a", requests), ScriptedRenderer("b", requests)]
    hooks = AvatarHooks(renderers, SPEECH_RATE_HZ, recovery_probe_s=0.01)
    yield hooks, renderers, requests
    stop(hooks, renderers)


def test_the_profile_feeds_the_users_and_the_agents_audio_to_the_avatar_and_publishes_an_input_timeline_port():
    config = ConfigManager.from_file(str(PROFILE)).config
    graph = config.session_graph()
    stage = config.stage_named(AVATAR_STAGE)
    assert "avatar" in graph.reading_nodes("input", PCM16)
    assert "avatar" in graph.reading_nodes("tts", PCM_F32)
    assert "avatar" in graph.reading_nodes("tts", SESSION_CONTROL)
    assert ("avatar", ControlEdge("dialog", INTERRUPT)) in graph.control_targets(
        "dialog", INTERRUPT
    )
    assert graph.publish["avatar"] == (JPEG_FRAME,)
    assert graph.is_published("tts", PCM_F32)
    assert graph.input_timed == frozenset({("avatar", JPEG_FRAME)})
    assert JPEG_FRAME not in graph.plugin_formats
    assert config.session_ports().plugin_output == (
        jpeg_frame_port(stage.factory.width, stage.factory.height),
    )
    assert len(stage.factory.renderer_urls) == 1
    assert stage.factory.sessions_per_renderer == 3


def test_the_port_carries_the_frame_size_the_stage_encodes():
    config = ConfigManager.from_file(str(PROFILE)).config
    config.stage_named(AVATAR_STAGE).factory.width = 320
    config.stage_named(AVATAR_STAGE).factory.height = 240
    (port,) = config.session_ports().plugin_output
    assert port.params["width"] == 320 and port.params["height"] == 240
    assert {
        key: value
        for key, value in port.params.items()
        if key not in ("width", "height")
    } == {
        key: value
        for key, value in JPEG_FRAME_PORT.params.items()
        if key not in ("width", "height")
    }
    assert port.params["timeline"] == "input" and port.params["follows_replies"]


def test_frames_follow_the_users_audio_back_to_back_while_no_reply_plays(avatar):
    renderer, session = avatar
    for index in range(4):
        session.user(index * 200)
        session.wait_frames(5 * (index + 1))
    assert session.timeline() == [(200 + 40 * k, 40, None) for k in range(20)]
    assert all(
        frame.modality == VIDEO and frame.format == JPEG_FRAME and frame.payload
        for frame in session.frames
    )
    speech, listen = renderer.windows[0]
    assert not speech.any()
    assert listen[: WINDOW_SAMPLES - CHUNK].tolist() == [0.0] * (WINDOW_SAMPLES - CHUNK)
    assert listen[WINDOW_SAMPLES - CHUNK :].tolist() == [0.25] * CHUNK


def test_each_chunk_renders_from_the_state_the_previous_chunk_returned(avatar):
    renderer, session = avatar
    for index in range(3):
        session.user(index * 200)
        session.wait_frames(5 * (index + 1))
    assert [(state_in, state_out) for _, state_in, state_out in renderer.requests] == [
        (b"", b"r1"),
        (b"r1", b"r2"),
        (b"r2", b"r3"),
    ]


def test_a_reply_plays_from_the_next_chunk_and_its_frames_carry_its_turn(avatar):
    renderer, session = avatar
    session.user(0)
    session.wait_frames(5)
    session.speech(500, turn=1)
    for index in range(1, 4):
        session.user(index * 200)
        session.wait_frames(5 * (index + 1))
    assert [turn for _, _, turn in session.timeline()] == [None] * 5 + [1] * 13 + [
        None
    ] * 2
    speech, listen = renderer.windows[1]
    assert np.allclose(speech[100:-100], 0.5, atol=0.01)
    assert set(np.unique(listen).tolist()) <= {0.0, 0.25}


def test_a_reply_shorter_than_the_window_waits_for_its_turn_to_end(avatar):
    renderer, session = avatar
    session.user(0)
    session.wait_frames(5)
    session.speech(100, turn=1)
    session.user(200)
    session.wait_frames(10)
    session.turn_done(1)
    session.user(400)
    session.wait_frames(15)
    assert [turn for _, _, turn in session.timeline()] == [None] * 10 + [1] * 3 + [
        None
    ] * 2


def test_an_interrupt_returns_the_avatar_to_listening_at_the_next_chunk(avatar):
    renderer, session = avatar
    session.user(0)
    session.wait_frames(5)
    session.speech(2000, turn=1)
    session.user(200)
    session.wait_frames(10)
    session.hooks.control(session.identity, control_chunk(INTERRUPT, turn=1))
    session.user(400)
    session.wait_frames(15)
    assert [turn for _, _, turn in session.timeline()] == [None] * 5 + [1] * 5 + [
        None
    ] * 5
    assert not renderer.windows[2][0].any()


def test_a_renderer_behind_real_time_renders_the_newest_ready_chunk_and_holds_the_older_ones(
    avatar,
):
    renderer, session = avatar
    renderer.gate.clear()
    session.user(0)
    assert renderer.rendering.wait(WAIT_S)
    for index, value in enumerate((0.125, 0.25, 0.5), start=1):
        session.user(index * 200, value=value)
    renderer.gate.set()
    session.wait_frames(20)
    assert session.timeline() == [(200 + 40 * k, 40, None) for k in range(20)]
    assert [bool(frame.payload) for frame in session.frames] == (
        [True] * 5 + [False] * 10 + [True] * 5
    )
    assert len(renderer.windows) == 2
    assert renderer.windows[1][1][-CHUNK:].tolist() == [0.5] * CHUNK


def test_after_the_input_ends_every_chunk_renders_until_the_reply_has_been_shown(
    avatar,
):
    renderer, session = avatar
    session.user(0)
    session.wait_frames(5)
    session.speech(1000, turn=1)
    session.turn_done(1)
    session.user(200, ms=0, eos=True)
    session.wait_frames(30)
    (worker,) = session.hooks.workers
    session.hooks.close(session.identity)
    worker.join(WAIT_S)
    assert session.timeline() == [
        (200 + 40 * k, 40, None if k < 5 else 1) for k in range(30)
    ]
    assert all(frame.payload for frame in session.frames)


def test_nothing_is_published_after_the_session_closes(avatar):
    renderer, session = avatar
    renderer.gate.clear()
    session.user(0)
    assert renderer.rendering.wait(WAIT_S)
    (worker,) = session.hooks.workers
    session.hooks.close(session.identity)
    renderer.gate.set()
    worker.join(WAIT_S)
    assert not worker.is_alive() and session.hooks.workers == [None]
    assert session.frames == []


def test_a_renderer_error_fails_the_session_output(avatar):
    renderer, session = avatar
    renderer.error = RuntimeError("bad request")
    session.user(0)
    session.wait_for(lambda: session.errors)
    session.user(200)
    assert session.errors == [renderer.error] and session.frames == []


def test_a_lone_renderer_that_fails_holds_the_chunks_rest_and_fails_the_session(
    avatar,
):
    renderer, session = avatar
    renderer.error = RendererUnavailable("worker gone")
    renderer.frames_before_error = 2
    session.user(0)
    session.wait_for(lambda: session.errors)
    assert session.errors == [renderer.error]
    assert [bool(frame.payload) for frame in session.frames] == [True] * 2 + [False] * 3
    assert session.timeline() == [(200 + 40 * k, 40, None) for k in range(5)]


def test_a_session_moves_to_another_worker_when_its_worker_fails_and_keeps_its_last_state(
    pool,
):
    hooks, renderers, requests = pool
    session = Avatar(hooks)
    session.user(0)
    session.wait_frames(5)
    (first,) = [r for r in renderers if r.windows]
    (other,) = [r for r in renderers if r is not first]
    first.error = RendererUnavailable("worker gone")
    first.healthy.clear()
    for index in range(1, 6):
        session.user(index * 200)
        session.wait_frames(5 * (index + 1))
    assert session.errors == []
    assert session.timeline() == [(200 + 40 * k, 40, None) for k in range(30)]
    # Whatever worker the chunk after the first lands on, a failed render holds its frames and leaves the state as it was.
    rendered = [state_out for _, _, state_out in requests]
    assert [state_in for _, state_in, _ in requests] == [b""] + rendered[:-1]
    assert {name for name, _, _ in requests[1:]} == {other.name}
    assert len(first.windows) <= 2
    held = [bool(frame.payload) for frame in session.frames[5:]]
    assert held.count(False) == 5 * (len(first.windows) - 1)


def test_a_session_fails_only_after_a_chunk_failed_on_as_many_workers_as_the_pool_has(
    pool,
):
    hooks, renderers, requests = pool
    session = Avatar(hooks)
    for renderer in renderers:
        renderer.error = RendererUnavailable(f"{renderer.name} gone")
        renderer.healthy.clear()
    session.user(0)
    session.wait_frames(5)
    assert session.errors == []
    session.user(200)
    session.wait_for(lambda: session.errors)
    assert len(session.errors) == 1
    assert session.timeline() == [(200 + 40 * k, 40, None) for k in range(10)]
    assert not any(frame.payload for frame in session.frames)


def test_sessions_share_the_pool_one_chunk_of_a_session_at_a_time_each_from_its_last_state(
    pool,
):
    hooks, renderers, requests = pool
    sessions = [Avatar(hooks, f"s{index}") for index in range(3)]
    for index in range(6):
        for session in sessions:
            session.user(index * 200)
        for session in sessions:
            session.wait_frames(5 * (index + 1))
    for session in sessions:
        assert session.errors == []
        assert session.timeline() == [(200 + 40 * k, 40, None) for k in range(30)]
        assert all(frame.payload for frame in session.frames)
    # Every render took the state a render of the same Session returned last, on whichever worker.
    returned = {state_out for _, _, state_out in requests}
    seeded = [state_in for _, state_in, _ in requests if state_in == b""]
    assert len(seeded) == 3 and len(requests) == 18
    assert all(state_in in returned for _, state_in, _ in requests if state_in)
    assert len({state_in for _, state_in, _ in requests if state_in}) == 15


def i420_of(picture: np.ndarray) -> bytes:
    """An RGB picture packed as the renderer service packs it: BT.601 limited-range I420."""
    red, green, blue = (picture[..., channel] / 255 for channel in range(3))
    luma = 0.299 * red + 0.587 * green + 0.114 * blue
    blue_difference = (-0.147 * red - 0.289 * green + 0.436 * blue)[::2, ::2]
    red_difference = (0.615 * red - 0.515 * green - 0.100 * blue)[::2, ::2]
    planes = [
        np.round(16 + luma * 219),
        np.round(16 + (blue_difference + 0.436) * 224 / 0.872),
        np.round(16 + (red_difference + 0.615) * 224 / 1.23),
    ]
    return b"".join(plane.astype(np.uint8).tobytes() for plane in planes)


def rgb_of(jpeg: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(jpeg)).convert("RGB"), dtype=float)


def test_a_frame_is_scaled_and_center_cropped_to_the_output_size_in_full_range_color():
    picture = np.zeros((16, 32, 3))
    picture[:, :] = (200, 100, 50)
    picture[:, :8] = 0
    picture[:, 24:] = 0
    flat = rgb_of(jpeg_of_i420(i420_of(picture), 16, 32, 8, 8, 90))
    assert flat.shape == (8, 8, 3)
    assert np.abs(flat[1:-1, 1:-1].mean(axis=(0, 1)) - (200, 100, 50)).max() < 6
    wide = rgb_of(jpeg_of_i420(i420_of(picture), 16, 32, 8, 16, 90))
    assert wide.shape == (8, 16, 3)
    assert wide[:, :2].mean() < 30 and wide[:, 6:10].mean() > 90


class RendererService:
    """The renderer service's wire: multipart windows and state in, the next state then five I420 frames out; `answers` scripts each render's status (an exception is raised as a transport failure)."""

    def __init__(self, name: str, height: int = 16, width: int = 32) -> None:
        self.name = name
        self.height = height
        self.width = width
        self.renders: list[tuple[dict[str, str], dict[str, bytes]]] = []
        self.answers: list[int | Exception] = []
        self.health = 200
        self.lock = threading.Lock()

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(self.health, json={"status": "ok"})
        else:
            pass
        with self.lock:
            answer = self.answers.pop(0) if self.answers else 200
            fields = multipart_fields(request)
            if isinstance(answer, Exception):
                raise answer
            elif answer != 200:
                return httpx.Response(answer, json={"detail": "scripted"})
            else:
                self.renders.append((dict(request.url.params), fields))
                state = f"{self.name}-{len(self.renders)}".encode()
        picture = np.zeros((self.height, self.width, 3))
        picture[:, :] = (200, 100, 50)
        frame = i420_of(picture)
        return httpx.Response(
            200,
            headers={
                "X-State-Length-Bytes": str(len(state)),
                "X-Frame-Length-Bytes": str(len(frame)),
                "X-Frame-Height": str(self.height),
                "X-Frame-Width": str(self.width),
            },
            content=state + frame * 5,
        )


def multipart_fields(request: httpx.Request) -> dict[str, bytes]:
    request.read()
    boundary = request.headers["content-type"].split("boundary=")[1].encode()
    fields = {}
    for part in request.content.split(b"--" + boundary)[1:-1]:
        head, _, body = part.partition(b"\r\n\r\n")
        name = head.split(b'name="')[1].split(b'"')[0].decode()
        fields[name] = body[: -len(b"\r\n")]
    return fields


def service_renderer(service: RendererService, width: int = 8, height: int = 8):
    return AVTR1Renderer(
        httpx.Client(
            transport=httpx.MockTransport(service), base_url=f"http://{service.name}"
        ),
        "maria",
        "plain_white",
        width=width,
        height=height,
        jpeg_quality=90,
    )


def test_the_renderer_service_wire_sends_both_windows_and_returns_the_next_state_after_the_frames():
    service = RendererService("worker")
    renderer = service_renderer(service, width=16, height=8)
    speech = np.linspace(-1, 1, WINDOW_SAMPLES, dtype=np.float32)
    listen = np.full(WINDOW_SAMPLES, 0.25, dtype=np.float32)
    frames: list[bytes] = []
    state = renderer.render(b"", speech, listen, frames.append)
    renderer.render(state, speech, listen, frames.append)
    params, fields = service.renders[0]
    assert params == {
        "avatar_id": "maria",
        "bg_id": "plain_white",
        "pixel_format": "yuv_i420",
    }
    assert {name: len(value) for name, value in fields.items()} == {
        "current_chunk": 2 * CHUNK,
        "future_chunk": 2 * (WINDOW_SAMPLES - CHUNK),
        "current_chunk_listen": 2 * CHUNK,
        "future_chunk_listen": 2 * (WINDOW_SAMPLES - CHUNK),
        "state": 0,
    }
    assert np.frombuffer(fields["future_chunk_listen"], dtype="<i2").tolist() == [
        8192
    ] * (WINDOW_SAMPLES - CHUNK)
    assert state == b"worker-1" and service.renders[1][1]["state"] == b"worker-1"
    assert len(frames) == 10
    picture = rgb_of(frames[0])
    assert picture.shape == (8, 16, 3)
    assert np.abs(picture.mean(axis=(0, 1)) - (200, 100, 50)).max() < 6


@pytest.mark.parametrize(
    ("answer", "is_worker_failure"),
    [
        (503, True),
        (httpx.ConnectError("refused"), True),
        (httpx.ReadTimeout("slow"), True),
        (404, False),
    ],
)
def test_a_worker_down_is_unavailable_and_a_bad_request_is_the_sessions_error(
    answer, is_worker_failure
):
    service = RendererService("worker")
    service.answers = [answer]
    renderer = service_renderer(service)
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    expected = RendererUnavailable if is_worker_failure else httpx.HTTPStatusError
    with pytest.raises(expected):
        renderer.render(b"", window, window, lambda jpeg: None)
    service.health = 503
    assert not renderer.is_healthy()
    service.health = 200
    assert renderer.is_healthy()


def test_over_the_wire_a_session_drifts_to_the_healthy_worker_with_its_last_state_and_returns_after_recovery():
    services = [RendererService("a"), RendererService("b")]
    hooks = AvatarHooks(
        [service_renderer(service) for service in services],
        SPEECH_RATE_HZ,
        recovery_probe_s=0.01,
    )
    session = Avatar(hooks)
    try:
        session.user(0)
        session.wait_frames(5)
        (first,) = [service for service in services if service.renders]
        (other,) = [service for service in services if service is not first]
        first.answers = [503] * 100
        first.health = 503
        for index in range(1, 5):
            session.user(index * 200)
            session.wait_frames(5 * (index + 1))
        assert session.errors == []
        failed = 100 - len(first.answers)
        assert failed <= 1
        assert [bool(frame.payload) for frame in session.frames].count(
            False
        ) == 5 * failed
        sent_states = [fields["state"] for _, fields in other.renders]
        assert sent_states[0] == f"{first.name}-1".encode()
        assert sent_states[1:] == [
            f"{other.name}-{index}".encode() for index in range(1, len(sent_states))
        ]
        first.answers = []
        first.health = 200
        before = len(first.renders)
        for index in range(5, 40):
            session.user(index * 200)
            session.wait_frames(5 * (index + 1))
            if len(first.renders) > before:
                break
            else:
                pass
        assert len(first.renders) > before
        assert all(frame.payload for frame in session.frames[-5:])
    finally:
        session.close()
        for worker in list(hooks.workers):
            if worker is not None:
                worker.join(WAIT_S)
            else:
                pass


class EncodingRendererService(RendererService):
    """A service that encodes the frames itself when asked for out_format=jpeg: the next state, then five JPEGs of the asked size, each of its own length."""

    def __call__(self, request: httpx.Request) -> httpx.Response:
        params = request.url.params
        if request.url.path == "/health" or params.get("out_format") != "jpeg":
            return super().__call__(request)
        else:
            pass
        with self.lock:
            fields = multipart_fields(request)
            self.renders.append((dict(params), fields))
            state = f"{self.name}-{len(self.renders)}".encode()
        jpegs = []
        for shade in range(5):
            encoded = io.BytesIO()
            Image.new(
                "RGB",
                (int(params["out_width"]), int(params["out_height"])),
                (40 * shade, 100, 50),
            ).save(encoded, format="JPEG", quality=int(params["jpeg_quality"]))
            jpegs.append(encoded.getvalue())
        return httpx.Response(
            200,
            headers={
                "X-State-Length-Bytes": str(len(state)),
                "X-Frame-Height": params["out_height"],
                "X-Frame-Width": params["out_width"],
                "X-Frame-Format": "jpeg",
                "X-Frame-Lengths": ",".join(str(len(jpeg)) for jpeg in jpegs),
            },
            content=state + b"".join(jpegs),
        )


def jpeg_service_renderer(service: RendererService) -> AVTR1Renderer:
    return AVTR1Renderer(
        httpx.Client(
            transport=httpx.MockTransport(service), base_url=f"http://{service.name}"
        ),
        "maria",
        "plain_white",
        width=16,
        height=8,
        jpeg_quality=90,
        service_jpeg=True,
    )


def test_a_service_that_encodes_hands_its_jpegs_of_the_stage_size_through_unchanged():
    service = EncodingRendererService("worker")
    renderer = jpeg_service_renderer(service)
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    frames: list[bytes] = []
    state = renderer.render(b"", window, window, frames.append)
    assert state == b"worker-1"
    params, _ = service.renders[0]
    assert params == {
        "avatar_id": "maria",
        "bg_id": "plain_white",
        "pixel_format": "yuv_i420",
        "out_format": "jpeg",
        "out_width": "16",
        "out_height": "8",
        "jpeg_quality": "90",
    }
    assert len(frames) == 5
    assert [rgb_of(frame).shape for frame in frames] == [(8, 16, 3)] * 5
    assert [round(rgb_of(frame)[..., 0].mean() / 40) for frame in frames] == [
        0,
        1,
        2,
        3,
        4,
    ]


def test_asking_a_service_that_does_not_encode_for_jpegs_still_encodes_its_i420_here():
    service = RendererService("worker")
    renderer = jpeg_service_renderer(service)
    window = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
    frames: list[bytes] = []
    assert renderer.render(b"", window, window, frames.append) == b"worker-1"
    assert service.renders[0][0]["out_format"] == "jpeg"
    assert len(frames) == 5
    picture = rgb_of(frames[0])
    assert picture.shape == (8, 16, 3)
    assert np.abs(picture.mean(axis=(0, 1)) - (200, 100, 50)).max() < 6


def test_a_worker_takes_as_many_chunks_in_flight_as_requests_per_renderer_and_holds_sessions_per_renderer(
    monkeypatch,
):
    real_client = httpx.Client

    def listing(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200, json={"avatars": ["maria"], "backgrounds": ["plain_white"]}
        )

    def client(base_url: str, timeout: float) -> httpx.Client:
        return real_client(
            transport=httpx.MockTransport(listing), base_url=base_url, timeout=timeout
        )

    monkeypatch.setattr(renderer_module.httpx, "Client", client)
    scheduler = renderer_module.create_scheduler(
        renderer_urls=["http://a", "http://b"],
        avatar="maria",
        background="plain_white",
        speech_sample_rate_hz=SPEECH_RATE_HZ,
        sessions_per_renderer=6,
        requests_per_renderer=6,
        service_jpeg=True,
    )
    renderers = scheduler.session_hooks.renderers
    assert len(renderers) == 12
    assert [str(renderer.client.base_url) for renderer in renderers] == [
        "http://a"
    ] * 6 + ["http://b"] * 6
    assert all(renderer.params["out_format"] == "jpeg" for renderer in renderers)
    assert scheduler.max_open_sessions == 12
