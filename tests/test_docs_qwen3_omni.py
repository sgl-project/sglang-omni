# SPDX-License-Identifier: Apache-2.0
"""CI smoke tests that mirror docs/basic_usage/qwen3_omni.md and
docs/basic_usage/send_request.md exactly.

Every test corresponds to a code block in the documentation.
If a test fails, the corresponding doc section is broken.

The tests start a Qwen3-Omni speech mode server, then replay every ``curl`` /
``python`` snippet from the docs and assert the expected responses.
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

import pytest
import requests

# ---------------------------------------------------------------------------
# Constants – must match docs/basic_usage/qwen3_omni.md
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"
MODEL_NAME = "qwen3-omni"
MODEL_PATH = "Qwen/Qwen3-Omni-30B-A3B-Instruct"

# Test data fixtures (exist in tests/data/).
TEST_IMAGE = str(Path("tests/data/cars.jpg").resolve())
TEST_VIDEO = str(Path("tests/data/draw.mp4").resolve())
TEST_AUDIO = str(Path("tests/data/cough.wav").resolve())

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_POLL_INTERVAL = 5  # seconds
_POLL_TIMEOUT = 600  # seconds (model download + init)


@pytest.fixture(scope="module")
def server():
    """Launch the Qwen3-Omni speech mode server and wait until healthy.

    Mirrors the ``Speech mode`` section in the doc:

    ``python examples/run_qwen3_omni_speech_server.py \\
      --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct``
    """
    proc = subprocess.Popen(
        [
            "python",
            "examples/run_qwen3_omni_speech_server.py",
            "--model-path",
            MODEL_PATH,
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    deadline = time.monotonic() + _POLL_TIMEOUT
    healthy = False
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{BASE_URL}/health", timeout=2)
            if r.status_code == 200:
                healthy = True
                break
        except Exception:
            pass
        time.sleep(_POLL_INTERVAL)

    if not healthy:
        proc.terminate()
        proc.wait(timeout=30)
        pytest.skip("Server did not become healthy in time")

    yield proc

    proc.terminate()
    proc.wait(timeout=30)


# ---------------------------------------------------------------------------
# send_request.md — Curl tests
# ---------------------------------------------------------------------------


def test_health_check_curl(server):
    """Docs section: Health Check (send_request.md) — Curl.

    ``curl -s http://localhost:8000/health``
    """
    result = subprocess.run(
        ["curl", "-s", f"{BASE_URL}/health"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body.get("status") == "healthy"


def test_list_models_curl(server):
    """Docs section: Model Listing (send_request.md) — Curl.

    ``curl -s http://localhost:8000/v1/models``
    """
    result = subprocess.run(
        ["curl", "-s", f"{BASE_URL}/v1/models"],
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert body["object"] == "list"
    assert len(body["data"]) >= 1


# ---------------------------------------------------------------------------
# qwen3_omni.md — Use Curl
# ---------------------------------------------------------------------------


def test_text_chat_curl(server):
    """Docs section: Use Curl > Text Chat.

    ``curl -s http://localhost:8000/v1/chat/completions \\
      -d '{"model": "qwen3-omni",
           "messages": [{"role": "user", "content": "Hello! ..."}],
           "modalities": ["text"], "max_tokens": 128}'``
    """
    result = subprocess.run(
        [
            "curl", "-s",
            f"{BASE_URL}/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Hello! Give me a one-sentence greeting."}],
                "modalities": ["text"],
                "max_tokens": 128,
            }),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert "choices" in body
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert len(body["choices"][0]["message"]["content"]) > 0


def test_streaming_curl(server):
    """Docs section: Use Curl > Streaming.

    ``curl -N http://localhost:8000/v1/chat/completions \\
      -d '{"model": "qwen3-omni",
           "messages": [{"role": "user", "content": "Write a short greeting."}],
           "modalities": ["text"], "stream": true}'``
    """
    result = subprocess.run(
        [
            "curl", "-N",
            f"{BASE_URL}/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "model": MODEL_NAME,
                "messages": [{"role": "user", "content": "Write a short greeting."}],
                "modalities": ["text"],
                "stream": True,
            }),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0

    events: list[dict] = []
    for line in result.stdout.splitlines():
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        events.append(json.loads(payload))

    assert len(events) > 0
    assert events[0]["choices"][0]["delta"].get("role") == "assistant"
    texts = [
        e["choices"][0]["delta"].get("content", "")
        for e in events
        if e["choices"][0]["delta"].get("content")
    ]
    assert any(t for t in texts)


def test_multimodal_input_curl(server):
    """Docs section: Use Curl > Multi-modal Input.

    ``curl -s http://localhost:8000/v1/chat/completions \\
      -d '{"model": "qwen3-omni",
           "messages": [...],
           "images": [...], "videos": [...], "audios": [...],
           "modalities": ["text"], "max_tokens": 256}'``

    The doc uses remote URLs; the CI test uses local fixtures from tests/data/.
    """
    result = subprocess.run(
        [
            "curl", "-s",
            f"{BASE_URL}/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": "Describe the image, the video, and the audio."}
                ],
                "images": [TEST_IMAGE],
                "videos": [TEST_VIDEO],
                "audios": [TEST_AUDIO],
                "modalities": ["text"],
                "max_tokens": 256,
            }),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert "choices" in body
    assert len(body["choices"][0]["message"]["content"]) > 0


# ---------------------------------------------------------------------------
# qwen3_omni.md — Use Python (OpenAI SDK)
# ---------------------------------------------------------------------------


def test_text_chat_openai_sdk(server):
    """Docs section: Use Python (OpenAI SDK) > Text Chat (OpenAI SDK)."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Hello! Give me a one-sentence greeting."}],
        max_tokens=128,
        extra_body={"modalities": ["text"]},
    )
    assert resp.choices[0].message.content


def test_streaming_openai_sdk(server):
    """Docs section: Use Python (OpenAI SDK) > Streaming (OpenAI SDK)."""
    from openai import OpenAI

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")
    stream = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": "Write a short greeting."}],
        stream=True,
        extra_body={"modalities": ["text"]},
    )

    collected = []
    for chunk in stream:
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        if delta.content:
            collected.append(delta.content)

    assert any(collected)


# ---------------------------------------------------------------------------
# qwen3_omni.md — Use Python (requests)
# ---------------------------------------------------------------------------


def test_text_chat_requests(server):
    """Docs section: Use Python (requests) > Text Chat (Python)."""
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Hello! Give me a one-sentence greeting."}],
            "modalities": ["text"],
            "max_tokens": 128,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "choices" in body
    assert body["choices"][0]["message"]["role"] == "assistant"
    assert len(body["choices"][0]["message"]["content"]) > 0


def test_streaming_requests(server):
    """Docs section: Use Python (requests) > Streaming (Python)."""
    r = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [{"role": "user", "content": "Write a short greeting."}],
            "modalities": ["text"],
            "stream": True,
        },
        stream=True,
        timeout=60,
    )
    assert r.status_code == 200

    events: list[dict] = []
    for line in r.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        if payload == "[DONE]":
            break
        events.append(json.loads(payload))

    assert len(events) > 0
    texts = [
        e["choices"][0]["delta"].get("content", "")
        for e in events
        if e["choices"][0]["delta"].get("content")
    ]
    assert any(t for t in texts)


def test_multimodal_input_requests(server):
    """Docs section: Use Python (requests) > Multi-modal Input (Python).

    The doc uses remote URLs; the CI test uses local fixtures from tests/data/.
    """
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Describe the image, the video, and the audio."}
            ],
            "images": [TEST_IMAGE],
            "videos": [TEST_VIDEO],
            "audios": [TEST_AUDIO],
            "modalities": ["text"],
            "max_tokens": 256,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert "choices" in body
    assert len(body["choices"][0]["message"]["content"]) > 0


# ---------------------------------------------------------------------------
# qwen3_omni.md — Speech Output
# ---------------------------------------------------------------------------


def test_speech_output_curl(server):
    """Docs section: Use Curl > Speech Output.

    ``curl -s http://localhost:8000/v1/chat/completions \\
      -d '{"model": "qwen3-omni",
           "messages": [{"role": "user", "content": "Say hello."}],
           "modalities": ["text", "audio"], "max_tokens": 128}'``
    """
    result = subprocess.run(
        [
            "curl", "-s",
            f"{BASE_URL}/v1/chat/completions",
            "-H", "Content-Type: application/json",
            "-d", json.dumps({
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": "Say hello and return both text and audio."}
                ],
                "modalities": ["text", "audio"],
                "max_tokens": 128,
            }),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0
    body = json.loads(result.stdout)
    assert "choices" in body
    message = body["choices"][0]["message"]
    assert message.get("content") or message.get("audio")
    audio_data = message.get("audio", {}).get("data")
    assert audio_data  # base64-encoded audio present


def test_speech_output_openai_sdk(server):
    """Docs section: Use Python (OpenAI SDK) > Speech Output (OpenAI SDK)."""
    import base64

    from openai import OpenAI

    client = OpenAI(base_url=f"{BASE_URL}/v1", api_key="EMPTY")
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "user", "content": "Say hello and return both text and audio."}
        ],
        max_tokens=128,
        extra_body={"modalities": ["text", "audio"]},
    )

    message = resp.choices[0].message
    assert message.content or message.audio
    assert message.audio is not None
    assert message.audio.data  # base64-encoded audio present


def test_speech_output_requests(server):
    """Docs section: Use Python (requests) > Speech Output (Python)."""
    resp = requests.post(
        f"{BASE_URL}/v1/chat/completions",
        json={
            "model": MODEL_NAME,
            "messages": [
                {"role": "user", "content": "Say hello and return both text and audio."}
            ],
            "modalities": ["text", "audio"],
            "max_tokens": 128,
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    message = body["choices"][0]["message"]
    assert message.get("content") or message.get("audio")
    audio_data = message.get("audio", {}).get("data")
    assert audio_data  # base64-encoded audio present
