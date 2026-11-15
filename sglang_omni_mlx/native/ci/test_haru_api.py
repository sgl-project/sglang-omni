# SPDX-License-Identifier: Apache-2.0
"""API tests for haru_server, run against the real model over /v1/realtime.

Haru is full-duplex, so the server exposes only the WebSocket realtime API: a
session.update, then input_audio_buffer.append PCM16 frames, an
sglang.input_audio.end, and the server streams response.output_audio.delta and
response.output_audio_transcript.delta until response.done.

    NATIVE_RUNTIME_BIN=<dir with the servers> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_haru_api.py
"""

from __future__ import annotations

import base64
import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from test_server_api import DATA_ROOT, RUNTIME_BIN, Server
from websockets.sync.client import connect

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
HARU_REPO = "higurashi-ai/Haru-4B"
# One full-duplex frame is 3200 16-bit samples at 24 kHz.
FRAME_SAMPLES = 3200


def haru_directory() -> Path:
    return Path(DATA_ROOT) / "models" / HARU_REPO.replace("/", "_")


@pytest.fixture(scope="module")
def haru_server() -> Iterator[Server]:
    running = Server("haru_server", "haru", haru_directory(), ("--quantize", "8"))
    yield running
    running.stop()


def append_event(pcm: bytes) -> str:
    return json.dumps(
        {"type": "input_audio_buffer.append", "audio": base64.b64encode(pcm).decode()}
    )


def events_until(socket, stop_types: set[str], limit: int = 64) -> list[dict]:
    """Read events until one of stop_types arrives; return everything read."""
    received = []
    while len(received) < limit:
        event = json.loads(socket.recv())
        received.append(event)
        if event["type"] in stop_types:
            break
    return received


def test_haru_session_update_reports_capabilities(haru_server: Server) -> None:
    with connect(f"ws://127.0.0.1:{haru_server.port}/v1/realtime") as socket:
        assert json.loads(socket.recv())["type"] == "session.created"
        socket.send(json.dumps({"type": "session.update", "session": {}}))
        updated = json.loads(socket.recv())
        assert updated["type"] == "session.updated"
        capabilities = updated["capabilities"]
        assert capabilities["native_full_duplex"] is True
        assert capabilities["output_modalities"] == ["audio"]
        assert capabilities["input_audio_format"]["rate"] == 24000


def test_haru_append_produces_an_audio_delta(haru_server: Server) -> None:
    pcm = b"\x00\x00" * FRAME_SAMPLES  # one frame of silence
    with connect(f"ws://127.0.0.1:{haru_server.port}/v1/realtime") as socket:
        assert json.loads(socket.recv())["type"] == "session.created"
        socket.send(json.dumps({"type": "session.update", "session": {}}))
        assert json.loads(socket.recv())["type"] == "session.updated"
        socket.send(append_event(pcm))
        events = events_until(socket, {"response.output_audio.delta"})
        assert events[0]["type"] == "sglang.input_audio.accepted"
        assert any(event["type"] == "response.output_audio.delta" for event in events)


def test_haru_rejects_audio_that_is_not_base64_pcm16(haru_server: Server) -> None:
    with connect(f"ws://127.0.0.1:{haru_server.port}/v1/realtime") as socket:
        assert json.loads(socket.recv())["type"] == "session.created"
        socket.send(json.dumps({"type": "session.update", "session": {}}))
        assert json.loads(socket.recv())["type"] == "session.updated"
        socket.send(json.dumps({"type": "input_audio_buffer.append", "audio": "A"}))
        event = json.loads(socket.recv())
        assert event["type"] == "error"
        assert event["error"]["type"] == "invalid_request_error"


def test_haru_end_finishes_the_response(haru_server: Server) -> None:
    pcm = b"\x00\x00" * FRAME_SAMPLES
    with connect(f"ws://127.0.0.1:{haru_server.port}/v1/realtime") as socket:
        assert json.loads(socket.recv())["type"] == "session.created"
        socket.send(json.dumps({"type": "session.update", "session": {}}))
        assert json.loads(socket.recv())["type"] == "session.updated"
        socket.send(append_event(pcm))
        events_until(socket, {"response.output_audio.delta"})
        socket.send(json.dumps({"type": "sglang.input_audio.end"}))
        tail = events_until(socket, {"sglang.input_audio.drained"})
        types = [event["type"] for event in tail]
        assert "response.done" in types
        assert "sglang.input_audio.drained" in types


def test_haru_rejects_an_unknown_event(haru_server: Server) -> None:
    with connect(f"ws://127.0.0.1:{haru_server.port}/v1/realtime") as socket:
        assert json.loads(socket.recv())["type"] == "session.created"
        socket.send(json.dumps({"type": "session.update", "session": {}}))
        assert json.loads(socket.recv())["type"] == "session.updated"
        socket.send(json.dumps({"type": "no.such.event"}))
        event = json.loads(socket.recv())
        assert event["type"] == "error"
        assert event["error"]["code"] == "unsupported_event"


# Every sampling value a client may set, all inside the accepted ranges.
VALID_SAMPLING = {
    "temperature": 0.7,
    "top_k": 25,
    "top_p": 1.0,
    "guidance_scale": 1.5,
    "seed": 42,
    "speak_first": False,
}


def update_session(socket, session: dict) -> dict:
    """Send one session.update and return the server's reply to it."""
    assert json.loads(socket.recv())["type"] == "session.created"
    socket.send(json.dumps({"type": "session.update", "session": session}))
    return json.loads(socket.recv())


def run_turn(socket, frames: int = 3) -> tuple[str, bytes]:
    """Feed `frames` silent frames and end the turn; return transcript and PCM16."""
    socket.send(append_event(b"\x00\x00" * FRAME_SAMPLES * frames))
    socket.send(json.dumps({"type": "sglang.input_audio.end"}))
    events = events_until(socket, {"response.done"}, limit=512)
    transcript = "".join(
        event["delta"]
        for event in events
        if event["type"] == "response.output_audio_transcript.delta"
    )
    audio = b"".join(
        base64.b64decode(event["delta"])
        for event in events
        if event["type"] == "response.output_audio.delta"
    )
    return transcript, audio


def test_haru_accepts_sampling_parameters(haru_server: Server) -> None:
    with connect(f"ws://127.0.0.1:{haru_server.port}/v1/realtime") as socket:
        updated = update_session(socket, {"sglang": {"sampling": VALID_SAMPLING}})
        assert updated["type"] == "session.updated"
        _, audio = run_turn(socket)
        assert audio


@pytest.mark.parametrize(
    ("sampling", "message"),
    [
        ({"temperature": 5.0}, "temperature must be in [0, 2]"),
        ({"temperature": -0.5}, "temperature must be in [0, 2]"),
        ({"top_k": 0}, "top_k must be >= 1"),
        ({"top_p": 0.0}, "top_p must be in (0, 1]"),
        ({"guidance_scale": 0.5}, "guidance_scale must be in [1, 2]"),
        ({"seed": -1}, "seed must be in [0, 2147483647]"),
    ],
)
def test_haru_rejects_invalid_sampling(
    haru_server: Server, sampling: dict, message: str
) -> None:
    with connect(f"ws://127.0.0.1:{haru_server.port}/v1/realtime") as socket:
        reply = update_session(socket, {"sglang": {"sampling": sampling}})
        assert reply["type"] == "error"
        assert reply["error"]["type"] == "invalid_request_error"
        assert reply["error"]["code"] == "invalid_sampling"
        assert reply["error"]["message"] == message


def test_haru_same_seed_replays_the_same_turn(haru_server: Server) -> None:
    """A client seed reaches the sampler: the same turn, twice, bit-exactly.

    session.update used to drop session.sglang.sampling, so two connections
    asking for seed 7 sampled from two random seeds and diverged.
    """

    def turn() -> tuple[str, bytes]:
        with connect(f"ws://127.0.0.1:{haru_server.port}/v1/realtime") as socket:
            reply = update_session(
                socket, {"sglang": {"sampling": {"seed": 7, "temperature": 0.7}}}
            )
            assert reply["type"] == "session.updated"
            return run_turn(socket)

    first = turn()
    second = turn()
    assert first[1], "the turn produced no audio to compare"
    assert first == second
