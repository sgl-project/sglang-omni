# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3-Omni documentation examples.

Every test replicates an API call from `docs/basic_usage/qwen3_omni.md`
so documentation can never silently go stale.

Usage:
    pytest tests/docs/qwen3_omni/test_docs_qwen3_omni.py -s -x
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

import pytest
import requests

from tests.utils import (
    disable_proxy,
    find_free_port,
    start_server_from_cmd,
    stop_server,
)

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen3-omni"

IMAGE_PATH = str(Path(__file__).resolve().parents[2] / "data" / "cars.jpg")
AUDIO_PATH = str(Path(__file__).resolve().parents[2] / "data" / "query_to_cars.wav")
TEXT_PROMPT = "How many cars are there in the picture?"

STARTUP_TIMEOUT = 900
REQUEST_TIMEOUT = 120


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


class TestTextOnlyMode:
    """Text-only server (--text-only, single GPU)."""

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory: pytest.TempPathFactory):
        port = find_free_port()
        log_file = tmp_path_factory.mktemp("text_only_logs") / "server.log"
        cmd = [
            sys.executable,
            "-m",
            "sglang_omni.cli.cli",
            "serve",
            "--model-path",
            MODEL_PATH,
            "--text-only",
            "--model-name",
            MODEL_NAME,
            "--port",
            str(port),
        ]
        proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
        yield port
        stop_server(proc)

    @pytest.mark.docs
    def test_health(self, server: int) -> None:
        """Docs section: Common — Health Check (text-only server)."""
        with disable_proxy():
            resp = requests.get(f"http://localhost:{server}/health", timeout=10)
        assert resp.status_code == 200
        assert "healthy" in resp.text

    @pytest.mark.docs
    def test_image_text(self, server: int) -> None:
        """Docs section: Text-Only Mode — Image and Text Input."""
        result = _post_chat(
            server,
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": TEXT_PROMPT}],
                "images": [IMAGE_PATH],
                "modalities": ["text"],
                "max_tokens": 16,
            },
        )
        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    @pytest.mark.docs
    def test_audio_image(self, server: int) -> None:
        """Docs section: Text-Only Mode — Audio and Image Input."""
        result = _post_chat(
            server,
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": ""}],
                "images": [IMAGE_PATH],
                "audios": [AUDIO_PATH],
                "modalities": ["text"],
                "max_tokens": 16,
            },
        )
        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0


class TestSpeechMode:
    """Speech server (multi-GPU, text + audio output)."""

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory: pytest.TempPathFactory):
        port = find_free_port()
        log_file = tmp_path_factory.mktemp("speech_logs") / "server.log"
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
        proc = start_server_from_cmd(cmd, log_file, port, timeout=STARTUP_TIMEOUT)
        yield port
        stop_server(proc)

    @pytest.mark.docs
    def test_health(self, server: int) -> None:
        """Docs section: Common — Health Check (speech server)."""
        with disable_proxy():
            resp = requests.get(f"http://localhost:{server}/health", timeout=10)
        assert resp.status_code == 200
        assert "healthy" in resp.text

    @pytest.mark.docs
    def test_image_text(self, server: int) -> None:
        """Docs section: Speech Mode — Image and Text Input."""
        result = _post_chat(
            server,
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": TEXT_PROMPT}],
                "images": [IMAGE_PATH],
                "modalities": ["text", "audio"],
                "max_tokens": 16,
            },
        )
        assert "choices" in result
        message = result["choices"][0]["message"]

        assert isinstance(message.get("content"), str)
        assert len(message["content"]) > 0

        assert "audio" in message
        audio_b64 = message["audio"]["data"]
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0

    @pytest.mark.docs
    def test_audio_image(self, server: int) -> None:
        """Docs section: Speech Mode — Audio and Image Input."""
        result = _post_chat(
            server,
            {
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": ""}],
                "images": [IMAGE_PATH],
                "audios": [AUDIO_PATH],
                "modalities": ["text", "audio"],
                "max_tokens": 16,
            },
        )
        assert "choices" in result
        message = result["choices"][0]["message"]

        assert isinstance(message.get("content"), str)
        assert len(message["content"]) > 0

        assert "audio" in message
        audio_b64 = message["audio"]["data"]
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
