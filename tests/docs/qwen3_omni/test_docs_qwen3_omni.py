# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3-Omni documentation examples.

Every test replicates an API call from `docs/basic_usage/qwen3_omni.md`
so documentation can never silently go stale.

Usage:
    pytest tests/docs/qwen3_omni/test_docs_qwen3_omni.py -s -x
"""

from __future__ import annotations

import base64
import subprocess
import sys
from pathlib import Path

import pytest
import requests

from tests.utils import disable_proxy, find_free_port, stop_server, wait_healthy

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen3-omni"

IMAGE_PATH = str(Path(__file__).resolve().parents[2] / "data" / "cars.jpg")
AUDIO_PATH = str(Path(__file__).resolve().parents[2] / "data" / "query_to_cars.wav")
TEXT_PROMPT = "How many cars are there in the picture?"

STARTUP_TIMEOUT = 600
REQUEST_TIMEOUT = 120


# ---------------------------------------------------------------------------
# Server lifecycle helpers
# ---------------------------------------------------------------------------


def _start_text_only_server(
    log_file: Path,
    port: int,
) -> subprocess.Popen:
    """Start the Qwen3-Omni server in text-only mode."""
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        MODEL_PATH,
        "--text-only",
        "--port",
        str(port),
        "--model-name",
        MODEL_NAME,
    ]
    with open(log_file, "w") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    wait_healthy(proc, port, log_file, timeout=STARTUP_TIMEOUT)
    return proc


def _start_speech_server(
    log_file: Path,
    port: int,
) -> subprocess.Popen:
    """Start the Qwen3-Omni server in speech mode (multi-GPU)."""
    cmd = [
        sys.executable,
        "examples/run_qwen3_omni_speech_server.py",
        "--model-path",
        MODEL_PATH,
        "--gpu-thinker",
        "0",
        "--gpu-talker",
        "1",
        "--gpu-code-predictor",
        "1",
        "--gpu-code2wav",
        "1",
        "--port",
        str(port),
        "--model-name",
        MODEL_NAME,
    ]
    with open(log_file, "w") as log_handle:
        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    wait_healthy(proc, port, log_file, timeout=STARTUP_TIMEOUT)
    return proc


def _post_chat(port: int, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    """POST to /v1/chat/completions and return the parsed JSON response."""
    with disable_proxy():
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()


# ===========================================================================
# Text-Only Mode Tests
# ===========================================================================


@pytest.fixture(scope="module")
def text_only_server(tmp_path_factory: pytest.TempPathFactory):
    """Start the text-only server, yield port, then shut down."""
    port = find_free_port()
    log_file = tmp_path_factory.mktemp("text_only_logs") / "server.log"
    proc = _start_text_only_server(log_file, port)
    yield port
    stop_server(proc)


@pytest.mark.docs
def test_text_only_health(text_only_server: int) -> None:
    """Docs section: Common — Health Check (text-only server)."""
    port = text_only_server
    with disable_proxy():
        resp = requests.get(f"http://localhost:{port}/health", timeout=10)
    assert resp.status_code == 200
    assert "healthy" in resp.text


@pytest.mark.docs
def test_text_only_image_text(text_only_server: int) -> None:
    """Docs section: Text-Only Mode — Image and Text Input."""
    port = text_only_server
    result = _post_chat(
        port,
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": TEXT_PROMPT}],
            "images": [IMAGE_PATH],
            "modalities": ["text"],
            "max_tokens": 256,
        },
    )
    assert "choices" in result
    content = result["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert len(content) > 0


@pytest.mark.docs
def test_text_only_audio_image(text_only_server: int) -> None:
    """Docs section: Text-Only Mode — Audio and Image Input."""
    port = text_only_server
    result = _post_chat(
        port,
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": ""}],
            "images": [IMAGE_PATH],
            "audios": [AUDIO_PATH],
            "modalities": ["text"],
            "max_tokens": 256,
        },
    )
    assert "choices" in result
    content = result["choices"][0]["message"]["content"]
    assert isinstance(content, str)
    assert len(content) > 0


# ===========================================================================
# Speech Mode Tests
# ===========================================================================


@pytest.fixture(scope="module")
def speech_server(tmp_path_factory: pytest.TempPathFactory):
    """Start the speech server, yield port, then shut down."""
    port = find_free_port()
    log_file = tmp_path_factory.mktemp("speech_logs") / "server.log"
    proc = _start_speech_server(log_file, port)
    yield port
    stop_server(proc)


@pytest.mark.docs
def test_speech_health(speech_server: int) -> None:
    """Docs section: Common — Health Check (speech server)."""
    port = speech_server
    with disable_proxy():
        resp = requests.get(f"http://localhost:{port}/health", timeout=10)
    assert resp.status_code == 200
    assert "healthy" in resp.text


@pytest.mark.docs
def test_speech_image_text(speech_server: int) -> None:
    """Docs section: Speech Mode — Image and Text Input."""
    port = speech_server
    result = _post_chat(
        port,
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": TEXT_PROMPT}],
            "images": [IMAGE_PATH],
            "modalities": ["text", "audio"],
            "max_tokens": 256,
        },
    )
    assert "choices" in result
    message = result["choices"][0]["message"]

    # Text output from the thinker
    assert isinstance(message.get("content"), str)
    assert len(message["content"]) > 0

    # Audio output from the talker (base64-encoded)
    assert "audio" in message
    audio_b64 = message["audio"]["data"]
    audio_bytes = base64.b64decode(audio_b64)
    assert len(audio_bytes) > 0


@pytest.mark.docs
def test_speech_audio_image(speech_server: int) -> None:
    """Docs section: Speech Mode — Audio and Image Input."""
    port = speech_server
    result = _post_chat(
        port,
        {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": ""}],
            "images": [IMAGE_PATH],
            "audios": [AUDIO_PATH],
            "modalities": ["text", "audio"],
            "max_tokens": 256,
        },
    )
    assert "choices" in result
    message = result["choices"][0]["message"]

    # Text output from the thinker
    assert isinstance(message.get("content"), str)
    assert len(message["content"]) > 0

    # Audio output from the talker (base64-encoded)
    assert "audio" in message
    audio_b64 = message["audio"]["data"]
    audio_bytes = base64.b64decode(audio_b64)
    assert len(audio_bytes) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
