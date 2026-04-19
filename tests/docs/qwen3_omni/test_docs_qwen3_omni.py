# SPDX-License-Identifier: Apache-2.0
"""Tests for Qwen3-Omni documentation examples.

Every test replicates an API call from `docs/basic_usage/qwen3_omni.md`
so documentation can never silently go stale. Each section is exercised
twice — once via the Python `requests` client and once via `curl` — so
both code blocks in the docs stay accurate.

Usage:
    pytest tests/docs/qwen3_omni/test_docs_qwen3_omni.py -s -x
"""

from __future__ import annotations

import base64
import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest
import requests
import torch
from jiwer import process_words

from benchmarks.tasks.tts import load_asr_model, normalize_text, transcribe
from sglang_omni.utils import find_available_port
from tests.utils import disable_proxy, no_proxy_env, start_server_from_cmd, stop_server

MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
MODEL_NAME = "qwen3-omni"

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
IMAGE_PATH = DATA_DIR / "cars.jpg"
AUDIO_PATH = DATA_DIR / "query_to_cars.wav"
VIDEO_PATH = DATA_DIR / "draw.mp4"
VIDEO_AUDIO_PATH = DATA_DIR / "query_to_draw.wav"
TEXT_PROMPT = "How many cars are there in the picture?"
EXPECTED_VIDEO_KEYWORDS = ["draw", "stylus", "pen", "tablet", "girl"]

STARTUP_TIMEOUT = 900
REQUEST_TIMEOUT = 120
MAX_VIDEO_AUDIO_WER = 0.18


def _post_chat(port: int, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    """POST to /v1/chat/completions via Python requests."""
    with disable_proxy():
        resp = requests.post(
            f"http://localhost:{port}/v1/chat/completions",
            json=payload,
            timeout=timeout,
        )
    resp.raise_for_status()
    return resp.json()


def _curl_chat(port: int, payload: dict, timeout: int = REQUEST_TIMEOUT) -> dict:
    """POST to /v1/chat/completions via curl (mirrors the cURL snippets in docs)."""
    curl = shutil.which("curl")
    if curl is None:
        pytest.skip("curl binary not available")
    cmd = [
        curl,
        "-sS",
        "--fail",
        "--noproxy",
        "*",
        "--max-time",
        str(timeout),
        "-X",
        "POST",
        f"http://localhost:{port}/v1/chat/completions",
        "-H",
        "Content-Type: application/json",
        "-d",
        json.dumps(payload),
    ]
    # Extra seconds for curl to exit and flush output after --max-time fires
    result = subprocess.run(
        cmd, capture_output=True, text=True, timeout=timeout + 10, env=no_proxy_env()
    )
    assert (
        result.returncode == 0
    ), f"curl failed (rc={result.returncode}): {result.stderr}"
    return json.loads(result.stdout)


def _send_chat(port: int, payload: dict, client: str) -> dict:
    if client == "python":
        return _post_chat(port, payload)
    elif client == "curl":
        return _curl_chat(port, payload)
    else:
        raise ValueError(f"Unknown client type: {client!r}")


@pytest.fixture(scope="session")
def whisper_asr() -> dict:
    """Session-scoped Whisper-large-v3 — load once, share across all WER tests."""
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    asr = load_asr_model("en", device)
    return {"asr": asr, "device": device}


class TestTextOnlyMode:
    """Text-only server (--text-only, single GPU)."""

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory: pytest.TempPathFactory):
        port = find_available_port()
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
    @pytest.mark.parametrize("client", ["python", "curl"])
    def test_image_text(self, server: int, client: str) -> None:
        """Docs section: Text-Only Mode — Image and Text Input."""
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": TEXT_PROMPT}],
            "images": [str(IMAGE_PATH)],
            "modalities": ["text"],
            "max_tokens": 16,
        }
        result = _send_chat(server, payload, client)
        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0

    @pytest.mark.docs
    @pytest.mark.parametrize("client", ["python", "curl"])
    def test_audio_image(self, server: int, client: str) -> None:
        """Docs section: Text-Only Mode — Audio and Image Input."""
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": ""}],
            "images": [str(IMAGE_PATH)],
            "audios": [str(AUDIO_PATH)],
            "modalities": ["text"],
            "max_tokens": 16,
        }
        result = _send_chat(server, payload, client)
        assert "choices" in result
        content = result["choices"][0]["message"]["content"]
        assert isinstance(content, str)
        assert len(content) > 0


class TestSpeechMode:
    """Speech server (multi-GPU, text + audio output)."""

    @pytest.fixture(scope="class")
    def server(self, tmp_path_factory: pytest.TempPathFactory):
        port = find_available_port()
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
    @pytest.mark.parametrize("client", ["python", "curl"])
    def test_image_text(self, server: int, client: str) -> None:
        """Docs section: Speech Mode — Image and Text Input."""
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": TEXT_PROMPT}],
            "images": [str(IMAGE_PATH)],
            "modalities": ["text", "audio"],
            "max_tokens": 16,
        }
        result = _send_chat(server, payload, client)
        assert "choices" in result
        message = result["choices"][0]["message"]

        assert isinstance(message.get("content"), str)
        assert len(message["content"]) > 0

        assert "audio" in message
        audio_b64 = message["audio"]["data"]
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0

    @pytest.mark.docs
    @pytest.mark.parametrize("client", ["python", "curl"])
    def test_audio_image(self, server: int, client: str) -> None:
        """Docs section: Speech Mode — Audio and Image Input."""
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": ""}],
            "images": [str(IMAGE_PATH)],
            "audios": [str(AUDIO_PATH)],
            "modalities": ["text", "audio"],
            "max_tokens": 16,
        }
        result = _send_chat(server, payload, client)
        assert "choices" in result
        message = result["choices"][0]["message"]

        assert isinstance(message.get("content"), str)
        assert len(message["content"]) > 0

        assert "audio" in message
        audio_b64 = message["audio"]["data"]
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0

    @pytest.mark.docs
    @pytest.mark.parametrize("client", ["python", "curl"])
    def test_video_audio(
        self, server: int, client: str, tmp_path: Path, whisper_asr: dict
    ) -> None:
        """Docs section: Speech Mode — Video and Audio Input.

        Verifies the thinker text contains expected keywords about the
        video and the talker audio (WER-compared against the thinker text
        via Whisper) stays within MAX_VIDEO_AUDIO_WER.
        """
        assert VIDEO_PATH.exists(), f"Test video not found: {VIDEO_PATH}"
        assert VIDEO_AUDIO_PATH.exists(), f"Test audio not found: {VIDEO_AUDIO_PATH}"
        payload = {
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": ""}],
            "videos": [str(VIDEO_PATH)],
            "audios": [str(VIDEO_AUDIO_PATH)],
            "modalities": ["text", "audio"],
            "max_tokens": 16,
            "temperature": 0.0,
        }
        result = _send_chat(server, payload, client)
        assert "choices" in result
        message = result["choices"][0]["message"]

        content = message.get("content", "")
        assert isinstance(content, str)
        assert len(content) > 0
        content_lower = content.lower()
        assert any(
            kw in content_lower for kw in EXPECTED_VIDEO_KEYWORDS
        ), f"Text output missing expected keywords about the video. Got: {content}"

        assert "audio" in message, "Expected audio in response"
        audio_b64 = message["audio"]["data"]
        audio_bytes = base64.b64decode(audio_b64)
        assert len(audio_bytes) > 0

        wav_path = tmp_path / f"video_audio_output_{client}.wav"
        wav_path.write_bytes(audio_bytes)

        transcription = transcribe(
            whisper_asr["asr"], str(wav_path), "en", whisper_asr["device"]
        )
        assert len(transcription) > 0, "Whisper transcription is empty"

        ref_norm = normalize_text(content, "en")
        hyp_norm = normalize_text(transcription, "en")
        assert ref_norm, "Empty reference after normalization"
        word_error_rate = process_words(ref_norm, hyp_norm).wer
        assert word_error_rate <= MAX_VIDEO_AUDIO_WER, (
            f"Talker audio diverges from thinker text (WER={word_error_rate:.2f}).\n"
            f"Text output (ref): {content!r} -> {ref_norm!r}\n"
            f"Transcription (hyp): {transcription!r} -> {hyp_norm!r}"
        )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
