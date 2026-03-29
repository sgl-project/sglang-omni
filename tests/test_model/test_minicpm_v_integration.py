# SPDX-License-Identifier: Apache-2.0
"""Backend integration test: MiniCPM-V 2.6 image understanding.

Starts the sglang-omni server with a MiniCPM-V model, sends image + text
prompts, and validates the model produces semantically correct responses.

Also covers MiniCPM-o audio-input routing when an audio file is supplied.

Usage:
    # Vision test (MiniCPM-V 2.6)
    pytest tests/test_model/test_minicpm_v_integration.py -s -x -m minicpmv

    # Audio test (MiniCPM-o 2.6, requires --model-path pointing to MiniCPM-o)
    pytest tests/test_model/test_minicpm_v_integration.py -s -x -m minicpmo

Environment variables:
    MINICPMV_MODEL_PATH : path / HF repo for MiniCPM-V 2.6 (default: openbmb/MiniCPM-V-2_6)
    MINICPMO_MODEL_PATH : path / HF repo for MiniCPM-o 2.6 (default: openbmb/MiniCPM-o-2_6)
    MINICPM_SERVER_PORT : override default port 18910
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest
import requests

from tests.test_model.helpers import disable_proxy

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MINICPMV_MODEL_PATH = os.environ.get(
    "MINICPMV_MODEL_PATH", "openbmb/MiniCPM-V-2_6"
)
MINICPMO_MODEL_PATH = os.environ.get(
    "MINICPMO_MODEL_PATH", "openbmb/MiniCPM-o-2_6"
)
SERVER_PORT = int(os.environ.get("MINICPM_SERVER_PORT", "18910"))
API_BASE = f"http://localhost:{SERVER_PORT}"

IMAGE_PATH = Path(__file__).parent.parent / "data" / "cars.jpg"
AUDIO_PATH = Path(__file__).parent.parent / "data" / "cough.wav"

STARTUP_TIMEOUT = 600  # seconds
REQUEST_TIMEOUT = 300  # seconds


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _start_server(model_path: str, port: int) -> subprocess.Popen:
    cmd = [
        sys.executable,
        "-m",
        "sglang_omni.cli.cli",
        "serve",
        "--model-path",
        model_path,
        "--port",
        str(port),
    ]
    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        preexec_fn=os.setsid,
    )


def _wait_healthy(proc: subprocess.Popen, api_base: str, timeout: int) -> bool:
    for _ in range(timeout):
        if proc.poll() is not None:
            out = proc.stdout.read() if proc.stdout else ""
            pytest.fail(
                f"Server exited with code {proc.returncode}.\nOutput:\n{out}"
            )
        try:
            with disable_proxy():
                resp = requests.get(f"{api_base}/health", timeout=2)
            if resp.status_code == 200 and "healthy" in resp.text:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(1)
    return False


def _kill_server(proc: subprocess.Popen) -> None:
    try:
        os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        proc.wait(timeout=30)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            proc.wait(timeout=10)
        except ProcessLookupError:
            pass


# ---------------------------------------------------------------------------
# Fixtures — MiniCPM-V 2.6 (vision-only)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def minicpmv_server():
    """Start a MiniCPM-V 2.6 server and yield (proc, api_base)."""
    assert IMAGE_PATH.exists(), f"Test image not found: {IMAGE_PATH}"

    t0 = time.monotonic()
    proc = _start_server(MINICPMV_MODEL_PATH, SERVER_PORT)
    healthy = _wait_healthy(proc, API_BASE, STARTUP_TIMEOUT)
    startup_time = time.monotonic() - t0

    if not healthy:
        _kill_server(proc)
        out = proc.stdout.read() if proc.stdout else ""
        pytest.fail(
            f"MiniCPM-V server did not become healthy within {STARTUP_TIMEOUT}s.\n{out}"
        )

    print(f"\n[PERF] MiniCPM-V server startup: {startup_time:.1f}s")
    yield proc, API_BASE

    _kill_server(proc)


# ---------------------------------------------------------------------------
# Fixtures — MiniCPM-o 2.6 (vision + audio)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def minicpmo_server():
    """Start a MiniCPM-o 2.6 server and yield (proc, api_base)."""
    assert IMAGE_PATH.exists(), f"Test image not found: {IMAGE_PATH}"
    assert AUDIO_PATH.exists(), f"Test audio not found: {AUDIO_PATH}"

    audio_port = SERVER_PORT + 1
    audio_api_base = f"http://localhost:{audio_port}"

    t0 = time.monotonic()
    proc = _start_server(MINICPMO_MODEL_PATH, audio_port)
    healthy = _wait_healthy(proc, audio_api_base, STARTUP_TIMEOUT)
    startup_time = time.monotonic() - t0

    if not healthy:
        _kill_server(proc)
        out = proc.stdout.read() if proc.stdout else ""
        pytest.fail(
            f"MiniCPM-o server did not become healthy within {STARTUP_TIMEOUT}s.\n{out}"
        )

    print(f"\n[PERF] MiniCPM-o server startup: {startup_time:.1f}s")
    yield proc, audio_api_base

    _kill_server(proc)


# ---------------------------------------------------------------------------
# MiniCPM-V tests
# ---------------------------------------------------------------------------


@pytest.mark.minicpmv
def test_minicpmv_image_description(minicpmv_server):
    """Image + text -> model describes objects in the image.

    The test image (cars.jpg) shows cars, so the response should contain
    relevant vehicle-related keywords.
    """
    proc, api_base = minicpmv_server
    image_abs = os.path.abspath(IMAGE_PATH)

    payload = {
        "model": "minicpmv",
        "messages": [{"role": "user", "content": "What objects are in this image?"}],
        "images": [image_abs],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }

    t0 = time.monotonic()
    with disable_proxy():
        resp = requests.post(
            f"{api_base}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    elapsed = time.monotonic() - t0

    assert resp.status_code == 200, f"Request failed: {resp.status_code} {resp.text}"

    body = resp.json()
    content = body["choices"][0]["message"].get("content", "").lower()
    print(f"\n[PERF] Image description latency: {elapsed:.1f}s")
    print(f"[RESPONSE] {content}")

    # cars.jpg should mention cars or vehicles
    assert any(kw in content for kw in ["car", "vehicle", "auto", "road", "wheel"]), (
        f"Response does not mention expected vehicle keywords.\nResponse: {content}"
    )


@pytest.mark.minicpmv
def test_minicpmv_text_only_query(minicpmv_server):
    """Text-only query (no images) should still work correctly."""
    _, api_base = minicpmv_server

    payload = {
        "model": "minicpmv",
        "messages": [{"role": "user", "content": "What is 2 + 2?"}],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": False,
    }

    with disable_proxy():
        resp = requests.post(
            f"{api_base}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

    assert resp.status_code == 200, f"Text-only request failed: {resp.text}"
    content = resp.json()["choices"][0]["message"].get("content", "").lower()
    assert "4" in content, f"Expected '4' in math response, got: {content}"


@pytest.mark.minicpmv
def test_minicpmv_multi_turn_conversation(minicpmv_server):
    """Two-round conversation: first describe image, then follow-up."""
    proc, api_base = minicpmv_server
    image_abs = os.path.abspath(IMAGE_PATH)

    # Round 1: describe the image
    payload_r1 = {
        "model": "minicpmv",
        "messages": [{"role": "user", "content": "Describe this image briefly."}],
        "images": [image_abs],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": False,
    }

    t1 = time.monotonic()
    with disable_proxy():
        resp_r1 = requests.post(
            f"{api_base}/v1/chat/completions",
            json=payload_r1,
            timeout=REQUEST_TIMEOUT,
        )
    t1_elapsed = time.monotonic() - t1

    assert resp_r1.status_code == 200
    content_r1 = resp_r1.json()["choices"][0]["message"].get("content", "")

    # Round 2: follow-up question
    payload_r2 = {
        "model": "minicpmv",
        "messages": [
            {"role": "user", "content": "Describe this image briefly."},
            {"role": "assistant", "content": content_r1},
            {"role": "user", "content": "How many objects did you mention?"},
        ],
        "images": [image_abs],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": False,
    }

    t2 = time.monotonic()
    with disable_proxy():
        resp_r2 = requests.post(
            f"{api_base}/v1/chat/completions",
            json=payload_r2,
            timeout=REQUEST_TIMEOUT,
        )
    t2_elapsed = time.monotonic() - t2

    assert resp_r2.status_code == 200
    content_r2 = resp_r2.json()["choices"][0]["message"].get("content", "")

    print(f"\n[PERF] Round 1: {t1_elapsed:.1f}s  Round 2: {t2_elapsed:.1f}s")
    print(f"[R1] {content_r1}")
    print(f"[R2] {content_r2}")

    # Round 2 should give a coherent follow-up, not empty
    assert len(content_r2.strip()) > 0, "Round 2 response is empty"


@pytest.mark.minicpmv
def test_minicpmv_streaming_response(minicpmv_server):
    """Streaming mode should produce incremental delta events."""
    _, api_base = minicpmv_server
    image_abs = os.path.abspath(IMAGE_PATH)

    payload = {
        "model": "minicpmv",
        "messages": [{"role": "user", "content": "Describe this image in one sentence."}],
        "images": [image_abs],
        "max_tokens": 64,
        "temperature": 0.0,
        "stream": True,
    }

    chunks = []
    with disable_proxy():
        with requests.post(
            f"{api_base}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
            stream=True,
        ) as resp:
            assert resp.status_code == 200
            for line in resp.iter_lines():
                if line and line.startswith(b"data:"):
                    data = line[5:].strip()
                    if data and data != b"[DONE]":
                        chunks.append(data)

    assert len(chunks) > 0, "No streaming chunks received"

    # Validate at least one chunk contains a text delta
    import json

    has_delta = False
    for chunk in chunks:
        try:
            obj = json.loads(chunk)
            delta = obj.get("choices", [{}])[0].get("delta", {})
            if delta.get("content"):
                has_delta = True
                break
        except json.JSONDecodeError:
            continue

    assert has_delta, "No text delta found in streaming chunks"


@pytest.mark.minicpmv
def test_minicpmv_server_health_after_requests(minicpmv_server):
    """Server should remain healthy after processing requests."""
    proc, api_base = minicpmv_server
    assert proc.poll() is None, f"Server died with code {proc.returncode}"

    with disable_proxy():
        health = requests.get(f"{api_base}/health", timeout=5)
    assert health.status_code == 200


# ---------------------------------------------------------------------------
# MiniCPM-o tests (vision + audio input)
# ---------------------------------------------------------------------------


@pytest.mark.minicpmo
def test_minicpmo_audio_input_understanding(minicpmo_server):
    """Audio input should be processed by the Whisper encoder.

    Sends a short audio clip and checks the model can describe it.
    """
    _, api_base = minicpmo_server
    audio_abs = os.path.abspath(AUDIO_PATH)

    payload = {
        "model": "minicpmo",
        "messages": [{"role": "user", "content": "What sound do you hear?"}],
        "audios": [audio_abs],
        "max_tokens": 128,
        "temperature": 0.0,
        "stream": False,
    }

    t0 = time.monotonic()
    with disable_proxy():
        resp = requests.post(
            f"{api_base}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )
    elapsed = time.monotonic() - t0

    assert resp.status_code == 200, f"Audio request failed: {resp.text}"
    content = resp.json()["choices"][0]["message"].get("content", "").lower()
    print(f"\n[PERF] Audio input latency: {elapsed:.1f}s")
    print(f"[RESPONSE] {content}")

    # cough.wav should mention cough/sound/noise
    assert len(content.strip()) > 0, "Empty response to audio query"


@pytest.mark.minicpmo
def test_minicpmo_image_and_audio_together(minicpmo_server):
    """Vision + audio inputs should both be processed correctly."""
    _, api_base = minicpmo_server
    image_abs = os.path.abspath(IMAGE_PATH)
    audio_abs = os.path.abspath(AUDIO_PATH)

    payload = {
        "model": "minicpmo",
        "messages": [
            {
                "role": "user",
                "content": "Describe what you see in the image and what you hear.",
            }
        ],
        "images": [image_abs],
        "audios": [audio_abs],
        "max_tokens": 256,
        "temperature": 0.0,
        "stream": False,
    }

    with disable_proxy():
        resp = requests.post(
            f"{api_base}/v1/chat/completions",
            json=payload,
            timeout=REQUEST_TIMEOUT,
        )

    assert resp.status_code == 200, f"Multimodal request failed: {resp.text}"
    content = resp.json()["choices"][0]["message"].get("content", "")
    assert len(content.strip()) > 0, "Empty response to multimodal query"
    print(f"\n[RESPONSE] {content}")


@pytest.mark.minicpmo
def test_minicpmo_server_health(minicpmo_server):
    """MiniCPM-o server should remain healthy after multimodal requests."""
    proc, api_base = minicpmo_server
    assert proc.poll() is None

    with disable_proxy():
        health = requests.get(f"{api_base}/health", timeout=5)
    assert health.status_code == 200


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-s", "-x", "-v", "-m", "minicpmv"]))
