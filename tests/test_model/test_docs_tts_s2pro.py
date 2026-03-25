# SPDX-License-Identifier: Apache-2.0
"""Tests for TTS S2-Pro documentation examples.

Every test replicates an API call from docs/basic_usage/tts_s2pro.md
so documentation can never silently go stale.

Usage:
    pytest tests/test_model/test_docs_tts_s2pro.py -s -x
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import signal
import subprocess
import sys
import time
import wave
from pathlib import Path

import pytest
import requests

from tests.test_model.helpers import disable_proxy
from tests.utils import find_free_port

logger = logging.getLogger(__name__)

MODEL_PATH = "fishaudio/s2-pro"
CONFIG_PATH = "examples/configs/s2pro_tts.yaml"
DATASET_REPO = "zhaochenyang20/seed-tts-eval-mini"
STARTUP_TIMEOUT = 600  # seconds

# Matches the exact text/audio pair used in the documentation.
SPEECH_INPUT = "Get the trust fund to the bank early."
REFERENCE_TEXT = (
    "We asked over twenty different people, and they all said it was his."
)
REF_WAV_RELPATH = "en/prompt-wavs/common_voice_en_10119832.wav"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def server_port() -> int:
    """Allocate a free TCP port for the server."""
    return find_free_port()


@pytest.fixture(scope="module")
def dataset_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Download the mini seed-tts-eval dataset via huggingface_hub."""
    from huggingface_hub import snapshot_download

    cache_dir = tmp_path_factory.mktemp("seed_tts_eval")
    path = snapshot_download(
        DATASET_REPO,
        repo_type="dataset",
        local_dir=str(cache_dir / "data"),
    )
    return Path(path)


@pytest.fixture(scope="module")
def ref_wav(dataset_dir: Path) -> Path:
    """Return the absolute path to the reference wav used in docs."""
    p = dataset_dir / REF_WAV_RELPATH
    assert p.exists(), f"Reference wav not found: {p}"
    return p


@pytest.fixture(scope="module")
def server_process(tmp_path_factory: pytest.TempPathFactory, server_port: int):
    """Start the s2-pro server and wait until healthy."""
    log_dir = tmp_path_factory.mktemp("server_logs")
    log_file = log_dir / "server.log"
    with open(log_file, "w") as log_handle:
        cmd = [
            sys.executable,
            "-m",
            "sglang_omni.cli.cli",
            "serve",
            "--model-path",
            MODEL_PATH,
            "--config",
            CONFIG_PATH,
            "--port",
            str(server_port),
        ]

        proc = subprocess.Popen(
            cmd,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            preexec_fn=os.setsid,
        )

        api_base = f"http://localhost:{server_port}"
        is_healthy = False
        for _ in range(STARTUP_TIMEOUT):
            if proc.poll() is not None:
                server_log = log_file.read_text()
                pytest.fail(
                    f"Server exited with code {proc.returncode}.\n{server_log}"
                )
            try:
                with disable_proxy():
                    resp = requests.get(f"{api_base}/health", timeout=2)
                if resp.status_code == 200 and "healthy" in resp.text:
                    is_healthy = True
                    break
            except requests.RequestException as exc:
                logger.debug("Health check failed (transient): %s", exc)
            time.sleep(1)

        if not is_healthy:
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            proc.wait(timeout=10)
            server_log = log_file.read_text()
            pytest.fail(
                f"Server did not become healthy within {STARTUP_TIMEOUT}s.\n"
                f"{server_log}"
            )

        yield proc

        # Teardown
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)


@pytest.fixture(scope="module")
def api_base(server_port: int) -> str:
    """Return the base URL for the test server."""
    return f"http://localhost:{server_port}"


# ---------------------------------------------------------------------------
# Tests — Use Curl section
# ---------------------------------------------------------------------------


@pytest.mark.docs
def test_basic_tts(
    server_process: subprocess.Popen, api_base: str, tmp_path: Path
) -> None:
    """POST /v1/audio/speech with minimal payload (no reference audio)."""
    with disable_proxy():
        resp = requests.post(
            f"{api_base}/v1/audio/speech",
            json={"input": "Hello, how are you?"},
            timeout=120,
        )
    resp.raise_for_status()
    assert len(resp.content) > 0

    output_path = tmp_path / "output.wav"
    output_path.write_bytes(resp.content)
    assert output_path.stat().st_size > 0


@pytest.mark.docs
def test_voice_cloning(
    server_process: subprocess.Popen,
    api_base: str,
    ref_wav: Path,
    tmp_path: Path,
) -> None:
    """Voice cloning with real reference audio from seed-tts-eval-mini."""
    with disable_proxy():
        resp = requests.post(
            f"{api_base}/v1/audio/speech",
            json={
                "input": SPEECH_INPUT,
                "references": [
                    {
                        "audio_path": str(ref_wav),
                        "text": REFERENCE_TEXT,
                    }
                ],
            },
            timeout=120,
        )
    resp.raise_for_status()
    assert len(resp.content) > 0

    output_path = tmp_path / "output.wav"
    output_path.write_bytes(resp.content)
    assert output_path.stat().st_size > 0


@pytest.mark.docs
def test_voice_cloning_streaming(
    server_process: subprocess.Popen,
    api_base: str,
    ref_wav: Path,
) -> None:
    """Streaming voice cloning via SSE — curl -N variant from docs."""
    with disable_proxy():
        resp = requests.post(
            f"{api_base}/v1/audio/speech",
            json={
                "input": SPEECH_INPUT,
                "references": [
                    {
                        "audio_path": str(ref_wav),
                        "text": REFERENCE_TEXT,
                    }
                ],
                "stream": True,
            },
            stream=True,
            timeout=600,
        )
    resp.raise_for_status()

    has_audio_chunk = False
    has_done = False
    for raw_line in resp.iter_lines():
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line or not line.startswith("data: "):
            continue

        payload = line[len("data: "):]
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


# ---------------------------------------------------------------------------
# Tests — Use Python section
# ---------------------------------------------------------------------------


@pytest.mark.docs
def test_python_basic_tts(
    server_process: subprocess.Popen, api_base: str, tmp_path: Path
) -> None:
    """Python basic TTS example from docs."""
    with disable_proxy():
        resp = requests.post(
            f"{api_base}/v1/audio/speech",
            json={"input": "Hello, how are you?"},
            timeout=120,
        )
    resp.raise_for_status()

    output_path = tmp_path / "output.wav"
    with open(output_path, "wb") as f:
        f.write(resp.content)
    assert output_path.stat().st_size > 0


@pytest.mark.docs
def test_python_voice_cloning(
    server_process: subprocess.Popen,
    api_base: str,
    ref_wav: Path,
    tmp_path: Path,
) -> None:
    """Python non-streaming voice cloning example from docs."""
    with disable_proxy():
        resp = requests.post(
            f"{api_base}/v1/audio/speech",
            json={
                "input": SPEECH_INPUT,
                "references": [
                    {"audio_path": str(ref_wav), "text": REFERENCE_TEXT}
                ],
            },
            timeout=120,
        )
    resp.raise_for_status()

    output_path = tmp_path / "output.wav"
    with open(output_path, "wb") as f:
        f.write(resp.content)
    assert output_path.stat().st_size > 0


@pytest.mark.docs
def test_python_voice_cloning_streaming(
    server_process: subprocess.Popen,
    api_base: str,
    ref_wav: Path,
    tmp_path: Path,
) -> None:
    """Python streaming voice cloning example from docs — reassemble WAV."""
    payload = {
        "input": SPEECH_INPUT,
        "references": [{"audio_path": str(ref_wav), "text": REFERENCE_TEXT}],
        "stream": True,
        "response_format": "wav",
    }

    chunks: list[bytes] = []
    fmt: tuple[int, int, int] | None = None

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
                data = line[len("data:"):].lstrip()
                if data == "[DONE]":
                    break
                b64 = (json.loads(data).get("audio") or {}).get("data")
                if not b64:
                    continue
                with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
                    if fmt is None:
                        fmt = w.getnchannels(), w.getsampwidth(), w.getframerate()
                    chunks.append(w.readframes(w.getnframes()))

    assert fmt, "No audio chunks received"
    nc, sw, fr = fmt
    output_path = tmp_path / "output_stream.wav"
    with wave.open(str(output_path), "wb") as w:
        w.setnchannels(nc)
        w.setsampwidth(sw)
        w.setframerate(fr)
        w.writeframes(b"".join(chunks))
    assert output_path.stat().st_size > 0


@pytest.mark.docs
def test_request_parameters(
    server_process: subprocess.Popen, api_base: str, tmp_path: Path
) -> None:
    """POST /v1/audio/speech with explicit generation parameters."""
    with disable_proxy():
        resp = requests.post(
            f"{api_base}/v1/audio/speech",
            json={
                "input": "Hello, how are you?",
                "temperature": 0.7,
                "top_p": 0.9,
                "max_new_tokens": 2048,
            },
            timeout=120,
        )
    resp.raise_for_status()
    assert len(resp.content) > 0

    output_path = tmp_path / "output.wav"
    output_path.write_bytes(resp.content)
    assert output_path.stat().st_size > 0


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v"]))
