# SPDX-License-Identifier: Apache-2.0
"""API tests for the native server's Silero VAD (--model-kind silero_vad), run
against the real model and the original Voxt's outputs in the golden file.

    NATIVE_RUNTIME_BIN=<dir with qwen3_asr_server> CI_DATA_ROOT=<provisioned root> \
        python -m pytest sglang_omni_mlx/native/ci/test_vad_api.py
"""

from __future__ import annotations

import base64
import http.client
import io
import json
import os
import struct
import subprocess
import uuid
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
import vad_golden
from websockets.sync.client import connect

RUNTIME_BIN = os.environ.get("NATIVE_RUNTIME_BIN")
DATA_ROOT = os.environ.get("CI_DATA_ROOT")
pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
GOLDEN = json.loads(
    (Path(__file__).resolve().parent / "golden" / "silero-vad-v6.json").read_text()
)
CHUNK = 512
# Note (Jiaxin Deng): the golden file's stream subset, so the original's probabilities are known.
STREAM_CLIP = "0000_en_short"
OTHER_STREAM_CLIP = "0010_en_short"


def samples(clip_id: str) -> np.ndarray:
    path = Path(DATA_ROOT) / "corpus" / "v1" / "clips" / f"{clip_id}.wav"
    with wave.open(str(path)) as reader:
        pcm = np.frombuffer(reader.readframes(reader.getnframes()), dtype="<i2")
    return pcm.astype(np.float32) / 32768


def original_stream(clip_id: str) -> np.ndarray:
    return np.frombuffer(
        base64.b64decode(GOLDEN["stream_probabilities"][clip_id]), dtype="<f4"
    )


def float32_wav(values: np.ndarray) -> bytes:
    data = values.astype("<f4").tobytes()
    header = b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
    header += struct.pack("<IHHIIHH", 16, 3, 1, 16000, 64000, 4, 32)
    return header + b"data" + struct.pack("<I", len(data)) + data


def pcm16_wav(values: np.ndarray) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes((values * 32768).round().astype("<i2").tobytes())
    return buffer.getvalue()


class Server:
    def __init__(self) -> None:
        model = Path(DATA_ROOT) / "models" / GOLDEN["model"].replace("/", "_")
        self.process = subprocess.Popen(
            [
                str(Path(RUNTIME_BIN) / "qwen3_asr_server"),
                "--supervised",
                "--model-kind",
                "silero_vad",
                "--model-directory",
                str(model),
            ],  # fmt: skip
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        self.ready = json.loads(self.process.stdout.readline())
        self.port = self.ready.get("port")

    def request(self, method: str, path: str, body: bytes = b"", headers=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        connection.request(method, path, body=body, headers=headers or {})
        response = connection.getresponse()
        return response.status, response.read()

    def timestamps(self, fields: dict[str, str], wav: bytes | None):
        boundary = uuid.uuid4().hex
        body = b""
        for name, value in fields.items():
            body += f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n\r\n{value}\r\n'.encode()
        if wav is not None:
            body += (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; filename="a.wav"\r\n\r\n'.encode()
                + wav
                + b"\r\n"
            )
        else:
            pass
        body += f"--{boundary}--\r\n".encode()
        return self.request(
            "POST",
            "/v1/vad/speech_timestamps",
            body,
            {"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )

    def stream(self):
        return connect(f"ws://127.0.0.1:{self.port}/v1/vad/stream")

    def health(self) -> dict:
        return json.loads(self.request("GET", "/health")[1])

    def stop(self) -> None:
        if self.process.poll() is None:
            self.process.stdin.write('{"command": "shutdown"}\n')
            self.process.stdin.flush()
            self.process.wait(timeout=10)
        else:
            pass


@pytest.fixture(scope="module")
def server() -> Iterator[Server]:
    running = Server()
    yield running
    running.stop()


def profile_fields(profile: str) -> dict[str, str]:
    return {name: str(value) for name, value in GOLDEN["profiles"][profile].items()}


def test_ready_event_and_routes(server: Server) -> None:
    assert server.ready["event"] == "ready"
    assert server.ready["model_name"].startswith("voxt-silero_vad-")
    assert server.health() == {
        "status": "healthy",
        "running": True,
        "request_states": {"running": 0, "streams": 0},
    }
    assert server.request("POST", "/v1/audio/transcriptions")[0] == 404


@pytest.mark.parametrize("profile", ["responsive", "balanced", "stable"])
def test_speech_timestamps_match_the_original(server: Server, profile: str) -> None:
    clip = "0002_en_short"
    status, body = server.timestamps(
        profile_fields(profile), float32_wav(samples(clip))
    )
    assert status == 200
    reply = json.loads(body)
    assert reply["sample_rate"] == 16000
    actual = [[r["start"], r["end"]] for r in reply["timestamps"]]
    mismatch = vad_golden.speech_mismatch(GOLDEN["timestamps"][clip][profile], actual)
    assert mismatch <= GOLDEN["tolerance"]["max_speech_mismatch"]


def test_pcm16_wav_and_default_options_are_accepted(server: Server) -> None:
    status, body = server.timestamps({}, pcm16_wav(samples("0002_en_short")))
    assert status == 200
    assert json.loads(body)["timestamps"]


@pytest.mark.parametrize(
    "fields, wav",
    [
        ({}, None),
        ({"threshold": "high"}, b""),
        ({"threshold": "nan"}, b""),
        ({"threshold": "inf"}, b""),
        ({"threshold": "1.5"}, b""),
        ({"threshold": "-0.1"}, b""),
        ({"min_speech_duration_ms": "-5"}, b""),
        ({"speech_pad_ms": "1.5"}, b""),
        ({}, b"RIFF"),
        ({}, float32_wav(np.array([0.0, np.nan], dtype=np.float32))),
        ({}, float32_wav(np.array([0.0, np.inf], dtype=np.float32))),
    ],
)
def test_invalid_requests_are_rejected(server: Server, fields: dict, wav) -> None:
    if wav == b"":
        wav = float32_wav(np.zeros(1600, dtype=np.float32))
    else:
        pass
    assert server.timestamps(fields, wav)[0] == 400


def stream_probabilities(socket, values: np.ndarray, sizes: list[int]) -> list:
    """Sends values in messages of the given sizes, cycling; one reply each."""
    replies = []
    offset = index = 0
    while offset < values.size:
        size = sizes[index % len(sizes)]
        socket.send(values[offset : offset + size].astype("<f4").tobytes())
        replies.append(json.loads(socket.recv(timeout=30))["probability"])
        offset += size
        index += 1
    return replies


def expected_replies(original: np.ndarray, total: int, sizes: list[int]) -> list:
    """The original's probability of the last chunk each message completed."""
    expected = []
    offset = index = 0
    while offset < total:
        before = offset // CHUNK
        offset = min(total, offset + sizes[index % len(sizes)])
        after = offset // CHUNK
        expected.append(original[after - 1] if after > before else None)
        index += 1
    return expected


def assert_close(replies: list, expected: list) -> None:
    assert [r is None for r in replies] == [e is None for e in expected]
    pairs = [(r, e) for r, e in zip(replies, expected) if e is not None]
    assert pairs
    tolerance = GOLDEN["tolerance"]["max_abs_probability"]
    assert max(abs(r - e) for r, e in pairs) <= tolerance


def test_stream_matches_the_original_for_uneven_messages(server: Server) -> None:
    sizes = [300, 1000, 212, 2048, 7]
    values = samples(STREAM_CLIP)
    with server.stream() as socket:
        assert server.health()["request_states"]["streams"] == 1
        replies = stream_probabilities(socket, values, sizes)
    assert_close(
        replies, expected_replies(original_stream(STREAM_CLIP), values.size, sizes)
    )


def test_streams_keep_separate_state(server: Server) -> None:
    first, second = samples(STREAM_CLIP), samples(OTHER_STREAM_CLIP)
    length = min(first.size, second.size) // 4096 * 4096
    with server.stream() as one, server.stream() as two:
        replies_one, replies_two = [], []
        for offset in range(0, length, 4096):
            one.send(first[offset : offset + 4096].astype("<f4").tobytes())
            two.send(second[offset : offset + 4096].astype("<f4").tobytes())
            replies_one.append(json.loads(one.recv(timeout=30))["probability"])
            replies_two.append(json.loads(two.recv(timeout=30))["probability"])
    assert_close(
        replies_one, expected_replies(original_stream(STREAM_CLIP), length, [4096])
    )
    assert_close(
        replies_two,
        expected_replies(original_stream(OTHER_STREAM_CLIP), length, [4096]),
    )


def test_a_partial_sample_closes_the_stream(server: Server) -> None:
    with server.stream() as socket:
        socket.send(b"\x00\x00\x00")
        assert "error" in json.loads(socket.recv(timeout=30))
        with pytest.raises(Exception):
            socket.recv(timeout=30)


def test_shutdown_reports_stopped() -> None:
    running = Server()
    running.process.stdin.write('{"command": "shutdown"}\n')
    running.process.stdin.flush()
    assert json.loads(running.process.stdout.readline()) == {"event": "stopped"}
    assert running.process.wait(timeout=10) == 0


@pytest.mark.parametrize(
    "message",
    [
        "text audio",
        np.array([0.0, np.nan], dtype="<f4").tobytes(),
        np.zeros(30 * 16000 + 1, dtype="<f4").tobytes(),
    ],
)
def test_text_non_finite_and_oversized_messages_close_the_stream(
    server: Server, message
) -> None:
    with server.stream() as socket:
        socket.send(message)
        assert "error" in json.loads(socket.recv(timeout=30))
        with pytest.raises(Exception):
            socket.recv(timeout=30)


def test_many_open_streams_are_served(server: Server) -> None:
    """Each open stream holds a server thread; 40 at once must all answer."""
    chunk = samples(STREAM_CLIP)[:512].astype("<f4").tobytes()
    sockets = [server.stream() for _ in range(40)]
    try:
        for socket in sockets:
            socket.send(chunk)
        for socket in sockets:
            assert json.loads(socket.recv(timeout=10))["probability"] is not None
    finally:
        for socket in sockets:
            socket.close()
