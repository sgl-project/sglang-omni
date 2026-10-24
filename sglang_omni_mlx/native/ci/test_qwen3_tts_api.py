# SPDX-License-Identifier: Apache-2.0
"""Native Qwen3-TTS API contracts against pinned real checkpoints."""

from __future__ import annotations

import base64
import io
import json
import socket
import subprocess
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import pytest
from test_server_api import DATA_ROOT, RUNTIME_BIN, Server

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)
CUSTOM_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice"
BASE_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
SENTENCE = "Please remember to bring your passport and boarding pass to the airport."


@pytest.fixture(scope="module")
def custom_server() -> Iterator[Server]:
    server = Server(
        "qwen3_tts_server",
        "qwen3_tts",
        Path(DATA_ROOT) / "models" / CUSTOM_MODEL.replace("/", "_"),
    )
    yield server
    server.stop()


def speak(server: Server, body: dict[str, object]) -> tuple[int, bytes]:
    return server.request(
        "POST",
        "/v1/audio/speech",
        json.dumps(body).encode(),
        {"Content-Type": "application/json"},
    )


def request_body() -> dict[str, str | int | float | bool]:
    return {"input": SENTENCE, "voice": "Ryan", "seed": 42, "max_new_tokens": 128}


def test_seeded_wav_pcm_and_stream_match(custom_server: Server) -> None:
    body = request_body()
    status, wav = speak(custom_server, body)
    assert status == 200
    with wave.open(io.BytesIO(wav)) as reader:
        assert (
            reader.getframerate(),
            reader.getnchannels(),
            reader.getsampwidth(),
        ) == (
            24000,
            1,
            2,
        )
        assert reader.getnframes() % 1920 == 0
        pcm = reader.readframes(reader.getnframes())
    samples = np.frombuffer(pcm, dtype="<i2")
    assert samples.size > 0 and np.max(np.abs(samples.astype(np.int32))) > 0
    assert speak(custom_server, body) == (200, wav)
    assert speak(custom_server, {**body, "response_format": "pcm"}) == (200, pcm)
    assert speak(custom_server, {**body, "response_format": "pcm", "stream": True}) == (
        200,
        pcm,
    )


def test_voice_changes_output(custom_server: Server) -> None:
    body = request_body()
    status, ryan = speak(custom_server, body)
    other_status, vivian = speak(custom_server, {**body, "voice": "Vivian"})
    assert status == other_status == 200
    assert ryan != vivian


@pytest.mark.parametrize(
    "override",
    [
        {"input": ""},
        {"input": 123},
        {"input": "<|im_end|>"},
        {"voice": "unknown"},
        {"language": "unknown"},
        {"task_type": "Base"},
        {"speed": 2},
        {"temperature": -1},
        {"top_p": 0},
        {"top_k": 1.5},
        {"seed": -1},
        {"max_new_tokens": 0},
        {"max_new_tokens": 2049},
        {"stream": "yes"},
        {"stream": True},
        {"response_format": "mp3"},
        {"instructions": "Speak cheerfully"},
        {"ref_audio": "bad"},
        {"subtalker_dosample": 1},
        {"stage_params": {}},
    ],
)
def test_invalid_request_returns_400(
    custom_server: Server, override: dict[str, object]
) -> None:
    status, response = speak(custom_server, {**request_body(), **override})
    assert status == 400
    assert "detail" in json.loads(response)


def test_malformed_json_and_wrong_method(custom_server: Server) -> None:
    assert custom_server.request("POST", "/v1/audio/speech", b"{")[0] == 400
    assert custom_server.request("GET", "/v1/audio/speech")[0] == 405


def test_disconnect_releases_busy_worker(custom_server: Server) -> None:
    body = json.dumps(
        {**request_body(), "input": SENTENCE * 20, "max_new_tokens": 2048}
    ).encode()
    client = socket.create_connection(("127.0.0.1", custom_server.port), timeout=10)
    client.sendall(
        b"POST /v1/audio/speech HTTP/1.1\r\nHost: test\r\nContent-Type: application/json\r\n"
        + f"Content-Length: {len(body)}\r\n\r\n".encode()
        + body
    )
    try:
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            _, health = custom_server.request("GET", "/health")
            if json.loads(health)["request_states"].get("running") == 1:
                break
            time.sleep(0.05)
        else:
            pytest.fail("synthesis did not start")
        assert speak(custom_server, request_body())[0] == 429
    finally:
        client.close()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        _, health = custom_server.request("GET", "/health")
        if not json.loads(health)["request_states"]:
            break
        time.sleep(0.1)
    else:
        pytest.fail("disconnected synthesis retained the worker")
    assert speak(custom_server, request_body())[0] == 200


def test_base_clones_inline_reference(custom_server: Server) -> None:
    status, reference = speak(custom_server, request_body())
    assert status == 200
    server = Server(
        "qwen3_tts_server",
        "qwen3_tts",
        Path(DATA_ROOT) / "models" / BASE_MODEL.replace("/", "_"),
    )
    try:
        body = {
            "input": "Thank you for your help today.",
            "ref_text": SENTENCE,
            "ref_audio": "data:audio/wav;base64,"
            + base64.b64encode(reference).decode(),
            "seed": 42,
            "max_new_tokens": 96,
        }
        status, output = speak(server, body)
        assert status == 200
        with wave.open(io.BytesIO(output)) as reader:
            assert reader.getframerate() == 24000
            assert 0 < reader.getnframes() <= 96 * 1920
        assert speak(server, {**body, "ref_audio": "not-base64"})[0] == 400
    finally:
        server.stop()


def test_fixed_codes_cross_attention_window(tmp_path: Path) -> None:
    frames = [
        [(frame * 17 + group * 31) % 2048 for group in range(16)] for frame in range(96)
    ]
    codes_path = tmp_path / "frames.json"
    codes_path.write_text(json.dumps(frames))
    outputs = []
    for chunk_frames in (96, 7):
        output = tmp_path / f"chunk-{chunk_frames}.wav"
        subprocess.run(
            [
                str(Path(RUNTIME_BIN) / "qwen3_tts_synthesize"),
                "--model-path",
                str(Path(DATA_ROOT) / "models" / CUSTOM_MODEL.replace("/", "_")),
                "--decode-codes",
                str(codes_path),
                "--decode-chunk-frames",
                str(chunk_frames),
                "--output",
                str(output),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        with wave.open(str(output)) as reader:
            assert reader.getnframes() == 96 * 1920
            outputs.append(
                np.frombuffer(
                    reader.readframes(reader.getnframes()), dtype="<i2"
                ).astype(np.int32)
            )
    np.testing.assert_allclose(outputs[0], outputs[1], rtol=0, atol=2)
