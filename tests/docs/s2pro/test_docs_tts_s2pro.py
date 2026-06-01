# SPDX-License-Identifier: Apache-2.0
"""Tests for TTS S2-Pro documentation examples.

Every test replicates an API call from `docs/basic_usage/tts_s2pro.md`
so documentation can never silently go stale.

Usage:
    pytest tests/docs/s2pro/test_docs_tts_s2pro.py -s -x
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import sys
import wave
from pathlib import Path

import pytest
import requests

from sglang_omni.utils import find_available_port
from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)

S2PRO_MODEL_PATH = "fishaudio/s2-pro"
S2PRO_CONFIG_PATH = "examples/configs/s2pro_tts.yaml"

SPEECH_INPUT = "Get the trust fund to the bank early."
REFERENCE_TEXT = "We asked over twenty different people, and they all said it was his."
REFERENCE_AUDIO = "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory):
    """Start the S2-Pro server, wait until healthy, and yield `(proc, port)`."""
    port = find_available_port()
    log_file = server_log_file(tmp_path_factory)
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli",
        "serve",
        "--model-path",
        S2PRO_MODEL_PATH,
        "--config",
        S2PRO_CONFIG_PATH,
        "--port",
        str(port),
    ]
    proc = start_server_from_cmd(cmd, log_file, port)
    yield proc, port
    stop_server(proc)


def _post_audio_speech(port: int, payload: dict, timeout: int = 120) -> bytes:
    """Send a request to `/v1/audio/speech` and return raw audio bytes."""
    with disable_proxy():
        response = requests.post(
            f"http://localhost:{port}/v1/audio/speech",
            json=payload,
            timeout=timeout,
        )
    response.raise_for_status()
    assert len(response.content) > 0
    return response.content


def _save_and_verify(content: bytes, path: Path) -> None:
    """Write audio bytes and assert the file is non-empty."""
    path.write_bytes(content)
    assert path.stat().st_size > 0


@pytest.mark.docs
def test_basic_tts(
    server_process: tuple[subprocess.Popen, int],
    tmp_path: Path,
) -> None:
    """POST `/v1/audio/speech` with the minimal payload from docs."""
    _, port = server_process
    content = _post_audio_speech(port, {"input": "Hello, how are you?"})
    _save_and_verify(content, tmp_path / "output.wav")


@pytest.mark.docs
def test_voice_cloning(
    server_process: tuple[subprocess.Popen, int],
    tmp_path: Path,
) -> None:
    """Voice cloning with the real reference audio used in docs."""
    _, port = server_process
    content = _post_audio_speech(
        port,
        {
            "input": SPEECH_INPUT,
            "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
        },
    )
    _save_and_verify(content, tmp_path / "output.wav")


@pytest.mark.docs
def test_voice_cloning_streaming(
    server_process: tuple[subprocess.Popen, int],
) -> None:
    """Streaming voice cloning via SSE."""
    _, port = server_process
    api_base = f"http://localhost:{port}"
    with disable_proxy():
        response = requests.post(
            f"{api_base}/v1/audio/speech",
            json={
                "input": SPEECH_INPUT,
                "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
                "stream": True,
            },
            stream=True,
            timeout=600,
        )
        response.raise_for_status()

        has_audio_chunk = False
        has_done = False
        for raw_line in response.iter_lines():
            line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
            if not line or not line.startswith("data: "):
                continue

            payload = line.removeprefix("data: ")
            if payload == "[DONE]":
                has_done = True
                break

            event = json.loads(payload)
            if (
                event.get("object") == "audio.speech.chunk"
                and event.get("audio") is not None
            ):
                has_audio_chunk = True

    assert has_audio_chunk, "Expected at least one audio.speech.chunk event"
    assert has_done, "Expected stream to end with [DONE]"


@pytest.mark.docs
def test_voice_cloning_streaming_wav_reassembly(
    server_process: tuple[subprocess.Popen, int],
    tmp_path: Path,
) -> None:
    """Streaming voice cloning with WAV reassembly from SSE chunks."""
    _, port = server_process
    api_base = f"http://localhost:{port}"
    payload = {
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REFERENCE_AUDIO, "text": REFERENCE_TEXT}],
        "stream": True,
        "response_format": "wav",
    }

    chunks: list[bytes] = []
    wav_format: tuple[int, int, int] | None = None

    with disable_proxy():
        with requests.post(
            f"{api_base}/v1/audio/speech",
            json=payload,
            stream=True,
            timeout=600,
        ) as stream:
            stream.raise_for_status()
            for line in stream.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line.removeprefix("data: ")
                if data == "[DONE]":
                    break
                base64_audio = (json.loads(data).get("audio") or {}).get("data")
                if not base64_audio:
                    continue
                with wave.open(
                    io.BytesIO(base64.b64decode(base64_audio)), "rb"
                ) as wav_file:
                    if wav_format is None:
                        wav_format = (
                            wav_file.getnchannels(),
                            wav_file.getsampwidth(),
                            wav_file.getframerate(),
                        )
                    chunks.append(wav_file.readframes(wav_file.getnframes()))

    assert wav_format, "No audio chunks received"
    n_channels, sample_width, frame_rate = wav_format
    output_path = tmp_path / "output_stream.wav"
    with wave.open(str(output_path), "wb") as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(frame_rate)
        wav_file.writeframes(b"".join(chunks))
    assert output_path.stat().st_size > 0


@pytest.mark.docs
def test_request_parameters(
    server_process: tuple[subprocess.Popen, int],
    tmp_path: Path,
) -> None:
    """POST `/v1/audio/speech` with explicit generation parameters from docs."""
    _, port = server_process
    content = _post_audio_speech(
        port,
        {
            "input": "Hello, how are you?",
            "temperature": 0.7,
            "top_p": 0.9,
            "max_new_tokens": 2048,
        },
    )
    _save_and_verify(content, tmp_path / "output.wav")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
