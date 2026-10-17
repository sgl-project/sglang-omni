# SPDX-License-Identifier: Apache-2.0
"""Native TTS API and supervisor protocol tests."""

from __future__ import annotations

import errno
import http.client
import io
import json
import os
import selectors
import subprocess
import time
import urllib.parse
import wave
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import NotRequired, TypedDict

import numpy as np
import pytest

MODEL_REPO = "inclusionAI/Ming-omni-tts-16.8B-A3B"


class SupervisorEvent(TypedDict):
    event: str
    reason: NotRequired[str]
    host: NotRequired[str]
    port: NotRequired[int]
    server_pid: NotRequired[int]
    model_name: NotRequired[str]
    startup_s: NotRequired[float]


class RequestChanges(TypedDict, total=False):
    model: str
    input: str
    stream: bool | None
    response_format: str | None
    voice: str | None
    speed: float | None
    prompt: str | None
    instructions: str | None
    max_new_tokens: float | None
    temperature: float | None
    stage_params: dict[str, dict[str, float | None] | None] | None
    ref_audio: str | None
    ref_text: str


def read_event(process: subprocess.Popen[str]) -> SupervisorEvent:
    assert process.stdout is not None
    with selectors.DefaultSelector() as selector:
        selector.register(process.stdout, selectors.EVENT_READ)
        assert selector.select(timeout=180), "No supervisor event within 180 seconds"
    return json.loads(process.stdout.readline())


@contextmanager
def supervised_process(
    binary: Path, directory: Path, extra_arguments: tuple[str, ...] = ()
) -> Iterator[subprocess.Popen[str]]:
    with subprocess.Popen(
        [
            str(binary),
            "--supervised",
            "--model-kind",
            "ming_tts",
            "--model-directory",
            str(directory),
            *extra_arguments,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        text=True,
    ) as process:
        try:
            yield process
        finally:
            if process.poll() is None:
                process.kill()
                process.wait(timeout=10)
            else:
                pass


@pytest.fixture(scope="module")
def server_binary() -> Path:
    if not os.environ.get("NATIVE_RUNTIME_BIN"):
        pytest.skip("Set NATIVE_RUNTIME_BIN")
    else:
        return Path(os.environ["NATIVE_RUNTIME_BIN"]) / "ming_tts_server"


@pytest.fixture(scope="module")
def model_directory() -> Path:
    if os.environ.get("MING_MODEL_DIR"):
        return Path(os.environ["MING_MODEL_DIR"])
    elif os.environ.get("CI_DATA_ROOT"):
        return (
            Path(os.environ["CI_DATA_ROOT"]) / "models" / MODEL_REPO.replace("/", "_")
        )
    else:
        pytest.skip("Set MING_MODEL_DIR or CI_DATA_ROOT")


@pytest.fixture(scope="module")
def server_url(request: pytest.FixtureRequest) -> Iterator[str]:
    if os.environ.get("MING_NATIVE_SERVER_URL"):
        yield os.environ["MING_NATIVE_SERVER_URL"]
        return
    else:
        pass
    with supervised_process(
        request.getfixturevalue("server_binary"),
        request.getfixturevalue("model_directory"),
        ("--model-name", "ming-omni-tts", "--startup-timeout-s", "180"),
    ) as process:
        ready = read_event(process)
        assert ready["event"] == "ready", ready
        assert ready["host"] == "127.0.0.1"
        assert ready["server_pid"] == process.pid
        assert ready["model_name"] == "ming-omni-tts"
        assert isinstance(ready["port"], int) and ready["port"] > 0
        assert isinstance(ready["startup_s"], (float, int)) and ready["startup_s"] >= 0
        yield f"http://127.0.0.1:{ready['port']}"
        assert process.stdin is not None
        process.stdin.write('not json\n{"command":"shutdown"}\n')
        process.stdin.flush()
        assert read_event(process) == {"event": "stopped"}
        assert process.wait(timeout=10) == 0


@pytest.fixture
def connection(server_url: str) -> Iterator[http.client.HTTPConnection]:
    address = urllib.parse.urlsplit(server_url)
    assert address.scheme == "http" and address.hostname
    connection = http.client.HTTPConnection(address.hostname, address.port, timeout=120)
    try:
        yield connection
    finally:
        connection.close()


def test_supervisor_rejects_other_model_kinds(server_binary: Path) -> None:
    completed = subprocess.run(
        [
            str(server_binary),
            "--supervised",
            "--model-kind",
            "qwen3_asr",
            "--model-directory",
            "unused",
        ],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert completed.returncode == 2
    assert completed.stdout == ""
    assert "only --model-kind ming_tts" in completed.stderr


def test_supervisor_reports_load_failure(server_binary: Path, tmp_path: Path) -> None:
    with supervised_process(server_binary, tmp_path) as process:
        failed = read_event(process)
        assert failed["event"] == "failed"
        assert failed["reason"]
        assert process.wait(timeout=10) == 1


@pytest.mark.parametrize("command", ["shutdown", "eof", "signal"])
def test_supervisor_stops_during_loading(
    server_binary: Path, tmp_path: Path, command: str
) -> None:
    os.mkfifo(tmp_path / "tokenizer.json")
    with supervised_process(server_binary, tmp_path) as process:
        deadline = time.monotonic() + 10
        while True:
            try:
                writer = os.open(
                    tmp_path / "tokenizer.json", os.O_WRONLY | os.O_NONBLOCK
                )
                break
            except OSError as error:
                if error.errno == errno.ENXIO and time.monotonic() < deadline:
                    time.sleep(0.01)
                else:
                    raise
        try:
            assert process.stdin is not None
            if command == "shutdown":
                process.stdin.write('{"command":"shutdown"}\n')
                process.stdin.flush()
                assert read_event(process) == {"event": "stopped"}
            elif command == "eof":
                process.stdin.close()
            else:
                process.terminate()
            assert process.wait(timeout=10) == 0
        finally:
            os.close(writer)


def test_discovery(connection: http.client.HTTPConnection) -> None:
    for path in ("/health", "/v1/models"):
        connection.request("GET", path)
        with connection.getresponse() as response:
            assert response.status == 200
            payload = json.load(response)
        if path == "/health":
            assert payload["running"] is True
        else:
            assert payload["data"][0]["id"] == "ming-omni-tts"
    connection.request("POST", "/v1/audio/transcriptions", b"")
    with connection.getresponse() as response:
        assert response.status == 404
        response.read()
    connection.close()


@pytest.mark.parametrize(
    "changes,status",
    [
        ({"input": "\u3000\n "}, 400),
        ({"stream": True}, 400),
        ({"max_new_tokens": 1.5}, 400),
        ({"stage_params": {"tts_engine": {"cfg": 1}}}, 400),
        ({"temperature": -1}, 400),
        ({"temperature": 0, "stage_params": {"tts_engine": {"temperature": -1}}}, 400),
        ({"stream": None}, 400),
        ({"response_format": None}, 400),
        ({"voice": None}, 400),
        ({"speed": None}, 400),
        ({"stage_params": {"tts_engine": None}}, 400),
        ({"ref_audio": None}, 400),
        ({"ref_audio": "/unused.wav"}, 400),
        ({"ref_text": "reference"}, 400),
        ({"ref_audio": "/unused.wav", "ref_text": "reference"}, 400),
    ],
)
def test_invalid_request(
    connection: http.client.HTTPConnection,
    changes: RequestChanges,
    status: int,
) -> None:
    request = {"model": "ming-omni-tts", "input": "你好。", **changes}
    connection.request(
        "POST",
        "/v1/audio/speech",
        json.dumps(request),
        {"Content-Type": "application/json"},
    )
    with connection.getresponse() as response:
        assert response.status == status
        assert json.load(response)["detail"]
    connection.close()


@pytest.mark.parametrize(
    "response_format,stream,changes",
    [
        (
            "wav",
            False,
            {
                "prompt": None,
                "instructions": None,
                "max_new_tokens": None,
                "temperature": None,
                "stage_params": None,
            },
        ),
        (
            "pcm",
            False,
            {
                "temperature": None,
                "stage_params": {
                    "tts_engine": {"cfg": None, "sigma": None, "temperature": None}
                },
            },
        ),
        ("pcm", True, {}),
    ],
)
def test_audio(
    connection: http.client.HTTPConnection,
    response_format: str,
    stream: bool,
    changes: RequestChanges,
) -> None:
    request = {
        "model": "ming-omni-tts",
        "input": "你好，欢迎使用语音合成服务。今天我们测试原生服务的音频输出。",
        "response_format": response_format,
        "stream": stream,
        "max_new_tokens": 40,
        "temperature": -1,
        "stage_params": {"tts_engine": {"temperature": 0}},
        **changes,
    }
    connection.request(
        "POST",
        "/v1/audio/speech",
        json.dumps(request),
        {"Content-Type": "application/json"},
    )
    with connection.getresponse() as response:
        assert response.status == 200
        assert response.getheader("Content-Type") == f"audio/{response_format}"
        assert response.getheader("X-Sample-Rate") == "44100"
        if stream:
            assert response.getheader("Transfer-Encoding") == "chunked"
        else:
            pass
        audio = response.read()
    if response_format == "wav":
        with wave.open(io.BytesIO(audio)) as waveform:
            assert (
                waveform.getframerate(),
                waveform.getnchannels(),
                waveform.getsampwidth(),
            ) == (44100, 1, 2)
            audio = waveform.readframes(waveform.getnframes())
    else:
        pass
    assert len(audio) > 0 and len(audio) % 2 == 0
    assert np.any(np.frombuffer(audio, dtype="<i2"))
    connection.close()


def test_disconnect_releases_request(connection: http.client.HTTPConnection) -> None:
    request = {
        "model": "ming-omni-tts",
        "input": "今天我们测试原生语音服务。" * 8,
        "stream": True,
        "response_format": "pcm",
        "max_new_tokens": 100,
    }
    connection.request(
        "POST",
        "/v1/audio/speech",
        json.dumps(request),
        {"Content-Type": "application/json"},
    )
    with connection.getresponse() as response:
        assert response.status == 200
        assert response.read(2)
    connection.close()
    request.update(input="你好。", max_new_tokens=1)
    deadline = time.monotonic() + 60
    while True:
        connection.request(
            "POST",
            "/v1/audio/speech",
            json.dumps(request),
            {"Content-Type": "application/json"},
        )
        with connection.getresponse() as response:
            status = response.status
            audio = response.read()
        if status == 429 and time.monotonic() < deadline:
            time.sleep(0.1)
        else:
            assert status == 200
            assert len(audio) > 0 and len(audio) % 2 == 0
            break
    connection.close()
