# SPDX-License-Identifier: Apache-2.0
"""HTTP and supervisor contract tests for the native AuK server."""

from __future__ import annotations

import base64
import io
import json
import os
import socket
import struct
import subprocess
import time
import wave
from collections.abc import Iterator
from pathlib import Path

import httpx
import numpy as np
import pytest

RUNTIME_BIN = os.environ.get("NATIVE_RUNTIME_BIN")
MODEL_DIRECTORY = os.environ.get("AUK_MODEL_DIRECTORY")
pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not MODEL_DIRECTORY,
    reason="set NATIVE_RUNTIME_BIN and AUK_MODEL_DIRECTORY",
)


@pytest.fixture(scope="module")
def server_log(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return tmp_path_factory.mktemp("auk-server") / "stderr.log"


@pytest.fixture(scope="module")
def server(server_log: Path) -> Iterator[httpx.Client]:
    stdout_log = server_log.with_name("stdout.log")
    with stdout_log.open("w") as output, server_log.open("w") as errors:
        with subprocess.Popen(
            [
                str(Path(RUNTIME_BIN) / "auk_server"),
                "--supervised",
                "--model-kind",
                "auk",
                "--model-directory",
                MODEL_DIRECTORY,
            ],
            stdin=subprocess.PIPE,
            stdout=output,
            stderr=errors,
            text=True,
        ) as process:
            try:
                deadline_seconds = time.monotonic() + 120
                ready_line = ""
                with stdout_log.open() as events:
                    while not ready_line.endswith("\n"):
                        ready_line += events.readline()
                        if ready_line.endswith("\n"):
                            break
                        elif process.poll() is not None:
                            pytest.fail(
                                f"Server exited before ready: {ready_line} "
                                f"{server_log.read_text()}"
                            )
                        elif time.monotonic() >= deadline_seconds:
                            pytest.fail(
                                f"Server readiness timed out: {ready_line} "
                                f"{server_log.read_text()}"
                            )
                        else:
                            time.sleep(0.02)
                ready = json.loads(ready_line)
                assert ready["event"] == "ready", ready
                assert ready["server_pid"] == process.pid
                assert ready["model_name"].startswith("voxt-auk-")
                with httpx.Client(
                    base_url=f"http://{ready['host']}:{ready['port']}", timeout=180
                ) as client:
                    yield client
            finally:
                try:
                    if process.poll() is None:
                        process.communicate(
                            input='{"command":"shutdown"}\n', timeout=10
                        )
                        assert process.returncode == 0, server_log.read_text()
                        assert json.loads(stdout_log.read_text().splitlines()[-1]) == {
                            "event": "stopped"
                        }
                    else:
                        pass
                finally:
                    if process.poll() is None:
                        process.kill()
                        process.wait(timeout=10)
                    else:
                        pass


def waveform(
    content: bytes, seconds: float
) -> np.ndarray[tuple[int, ...], np.dtype[np.int16]]:
    with wave.open(io.BytesIO(content)) as audio:
        assert audio.getframerate() == 24000
        assert audio.getnchannels() == 1
        assert audio.getsampwidth() == 2
        assert abs(audio.getnframes() / audio.getframerate() - seconds) < 0.021
        samples = np.frombuffer(audio.readframes(audio.getnframes()), dtype="<i2")
    assert np.ptp(samples.astype(np.float32)) > 3
    assert np.mean(samples.astype(np.float32) ** 2) > 1
    return samples


def test_health_models_and_speech_formats(server: httpx.Client) -> None:
    assert server.get("/health").json() == {
        "status": "healthy",
        "running": True,
        "request_states": {},
    }
    model = server.get("/v1/models").json()["data"][0]
    payload = {
        "model": model["id"],
        "input": "Welcome home.",
        "gen_seconds": 2,
        "seed": 7,
    }
    wav = server.post("/v1/audio/speech", json=payload)
    assert wav.status_code == 200, wav.text[:300] if wav.status_code != 200 else ""
    samples = waveform(wav.content, 2)
    pcm = server.post("/v1/audio/speech", json={**payload, "response_format": "pcm"})
    assert pcm.status_code == 200
    assert pcm.headers["content-type"].startswith("audio/pcm")
    assert np.array_equal(samples, np.frombuffer(pcm.content, dtype="<i2"))
    assert server.post("/v1/audio/transcriptions", json={}).status_code == 404


def test_reference_cloning_duration_and_raw_instruction(server: httpx.Client) -> None:
    reference = Path(__file__).parents[3] / "tests/data/query_to_cars.wav"
    with wave.open(str(reference)) as audio:
        reference_seconds = audio.getnframes() / audio.getframerate()
    reference_uri = (
        "data:audio/wav;base64," + base64.b64encode(reference.read_bytes()).decode()
    )
    payload = {
        "input": "Welcome home.",
        "ref_audio": reference_uri,
        "ref_text": "Welcome home.",
        "seed": 11,
    }
    first = server.post("/v1/audio/speech", json=payload)
    second = server.post("/v1/audio/speech", json=payload)
    assert first.status_code == second.status_code == 200
    waveform(first.content, min(reference_seconds, 30))
    assert first.content == second.content
    generated = server.post(
        "/generate",
        json={
            "prompt": 'Say the following with the same voice: "Welcome home."',
            "metadata": {"tts_params": {"ref_audio": reference_uri, "seed": 11}},
            "stage_params": {"auk_engine": {"gen_seconds": 2}},
            "output_modalities": ["audio"],
        },
    )
    assert generated.status_code == 200, generated.text[:300]
    result = generated.json()
    assert result["audio"]["format"] == "wav"
    assert result["audio"]["sample_rate"] == 24000
    waveform(base64.b64decode(result["audio"]["data"], validate=True), 2)


def test_editing_without_duration_keeps_complete_reference_frames(
    server: httpx.Client,
) -> None:
    reference = io.BytesIO()
    samples = (np.sin(np.arange(744) * 0.1) * 1000).astype("<i2")
    with wave.open(reference, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(24000)
        audio.writeframes(samples.tobytes())
    response = server.post(
        "/generate",
        json={
            "prompt": "Remove background noise.",
            "metadata": {
                "tts_params": {
                    "ref_audio": "data:audio/wav;base64,"
                    + base64.b64encode(reference.getvalue()).decode(),
                    "seed": 11,
                }
            },
        },
    )
    assert response.status_code == 200, response.text[:300]
    with wave.open(
        io.BytesIO(base64.b64decode(response.json()["audio"]["data"])), "rb"
    ) as audio:
        assert audio.getnframes() == 480


@pytest.mark.parametrize(
    "patch",
    [
        {"input": ""},
        {"input": 3},
        {"gen_seconds": 0},
        {"gen_seconds": "three"},
        {"seed": True},
        {"seed": -1},
        {"response_format": "mp3"},
        {"stream": True},
        {"speed": 2},
        {"model": "wrong-model"},
        {"ref_audio": "file:///tmp/audio.wav"},
        {"ref_audio": "data:audio/wav;base64,AAAA="},
        {"stage_params": {"auk_engine": {"nfe": 1}}},
        {"stage_params": []},
    ],
)
def test_invalid_speech_never_enters_worker(
    server: httpx.Client, patch: dict[str, object]
) -> None:
    response = server.post(
        "/v1/audio/speech", json={"input": "Hello.", "gen_seconds": 2, **patch}
    )
    assert response.status_code == 400, response.text
    assert server.get("/health").json()["request_states"] == {}


@pytest.mark.parametrize("should_reset_connection", [False, True], ids=["fin", "rst"])
def test_disconnect_cancels_generation(
    server: httpx.Client, server_log: Path, should_reset_connection: bool
) -> None:
    """A disconnected request must cancel instead of finishing its waveform."""
    cancellation_count = server_log.read_text().count("synthesis cancelled")
    body = json.dumps(
        {"input": "Welcome home.", "gen_seconds": 30, "seed": 13}
    ).encode()
    with socket.create_connection(
        (server.base_url.host, server.base_url.port)
    ) as connection:
        if should_reset_connection:
            connection.setsockopt(
                socket.SOL_SOCKET, socket.SO_LINGER, struct.pack("ii", 1, 0)
            )
        else:
            pass
        connection.sendall(
            b"POST /v1/audio/speech HTTP/1.1\r\nHost: localhost\r\n"
            b"Content-Type: application/json\r\nContent-Length: "
            + str(len(body)).encode()
            + b"\r\n\r\n"
            + body
        )
        deadline_seconds = time.monotonic() + 120
        while not server.get("/health").json()["request_states"]:
            assert time.monotonic() < deadline_seconds
            time.sleep(0.02)
        busy = server.post(
            "/v1/audio/speech", json={"input": "Hello.", "gen_seconds": 2}
        )
        assert busy.status_code == 429
    while server.get("/health").json()["request_states"]:
        assert time.monotonic() < deadline_seconds
        time.sleep(0.02)
    assert server_log.read_text().count("synthesis cancelled") == cancellation_count + 1
    recovered = server.post(
        "/v1/audio/speech", json={"input": "Hello.", "gen_seconds": 2}
    )
    assert recovered.status_code == 200
    waveform(recovered.content, 2)


def test_wrong_model_kind_fails_before_loading() -> None:
    result = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "auk_server"),
            "--model-kind",
            "qwen3_asr",
            "--model-path",
            MODEL_DIRECTORY,
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 2
    assert "only --model-kind auk" in result.stderr


def test_wrong_checkpoint_fails_before_listening(tmp_path: Path) -> None:
    (tmp_path / "config.json").write_text('{"model_type":"qwen3_asr"}')
    with subprocess.Popen(
        [
            str(Path(RUNTIME_BIN) / "auk_server"),
            "--supervised",
            "--model-path",
            str(tmp_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            process.wait(timeout=10)
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            else:
                pass
        output, errors = process.communicate(timeout=10)
    assert process.returncode == 1, errors
    failure = json.loads(output)
    assert failure["event"] == "failed"
    assert "unsupported AuK runtime configuration" in failure["reason"]


@pytest.mark.parametrize("seconds,sample_count", [(0.031, 960), (0.02, 480)])
def test_duration_rounds_up_to_complete_frames(
    server: httpx.Client, seconds: float, sample_count: int
) -> None:
    response = server.post(
        "/v1/audio/speech", json={"input": "Hello.", "gen_seconds": seconds, "seed": 17}
    )
    assert response.status_code == 200, (
        response.text[:300] if response.status_code != 200 else ""
    )
    with wave.open(io.BytesIO(response.content)) as audio:
        assert audio.getframerate() == 24000
        assert audio.getnframes() == sample_count
