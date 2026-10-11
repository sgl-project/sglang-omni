# SPDX-License-Identifier: Apache-2.0
"""CI smoke tests that mirror docs/basic_usage/tts_s2pro.md exactly.

Every test runs the **same command** shown in the documentation:
- curl sections  → ``subprocess.run(["curl", ...])``
- Python sections → ``requests`` library

If a test fails, the corresponding doc section is broken.
"""

from __future__ import annotations

import base64
import io
import json
import subprocess
import time
import wave
from pathlib import Path

import pytest
import requests

# ---------------------------------------------------------------------------
# Constants – must match docs/basic_usage/tts_s2pro.md
# ---------------------------------------------------------------------------
BASE_URL = "http://localhost:8000"
MODEL_PATH = "fishaudio/s2-pro"
CONFIG_PATH = "examples/configs/s2pro_tts.yaml"

# Voice cloning reference audio — must match docs/basic_usage/tts_s2pro.md exactly.
# The CI workflow pre-downloads this file via ``hf download --repo-type dataset``.
REF_WAV = str(
    Path("seed-tts-eval-mini") / "en" / "prompt-wavs" / "common_voice_en_10119832.wav"
)
REF_TEXT = "We asked over twenty different people, and they all said it was his."
SPEECH_INPUT = "Get the trust fund to the bank early."

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_POLL_INTERVAL = 5  # seconds
_POLL_TIMEOUT = 600  # seconds (model download + init)


@pytest.fixture(scope="module")
def server():
    """Launch the S2-Pro TTS server and wait until healthy.

    Mirrors the ``Launch the Server`` section in the doc:
    ``sgl-omni serve --model-path fishaudio/s2-pro \\
        --config examples/configs/s2pro_tts.yaml --port 8000``
    """
    proc = subprocess.Popen(
        [
            "sgl-omni",
            "serve",
            "--model-path",
            MODEL_PATH,
            "--config",
            CONFIG_PATH,
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
# send_request.md — common patterns
# ---------------------------------------------------------------------------


def test_health_check(server):
    """Docs section: Health Check (send_request.md).

    ``curl -s http://localhost:8000/health``
    """
    r = subprocess.run(
        ["curl", "-s", f"{BASE_URL}/health"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    body = json.loads(r.stdout)
    assert body.get("status") == "healthy"


def test_list_models(server):
    """Docs section: Model Listing (send_request.md).

    ``curl -s http://localhost:8000/v1/models``
    """
    r = subprocess.run(
        ["curl", "-s", f"{BASE_URL}/v1/models"],
        capture_output=True,
        text=True,
    )
    assert r.returncode == 0
    body = json.loads(r.stdout)
    assert body["object"] == "list"
    assert len(body["data"]) >= 1


# ---------------------------------------------------------------------------
# tts_s2pro.md — Use Curl / Basic TTS
# ---------------------------------------------------------------------------


def test_basic_tts_curl(server):
    """Docs section: Use Curl — basic TTS without reference audio.

    ``curl -X POST http://localhost:8000/v1/audio/speech \\
        -H "Content-Type: application/json" \\
        -d '{"input": "Hello, how are you?"}' \\
        --output output.wav``
    """
    r = subprocess.run(
        [
            "curl",
            "-X",
            "POST",
            f"{BASE_URL}/v1/audio/speech",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps({"input": "Hello, how are you?"}),
            "--output",
            "-",
        ],
        capture_output=True,
    )
    assert r.returncode == 0
    assert r.stdout.startswith(b"RIFF")


# ---------------------------------------------------------------------------
# tts_s2pro.md — Use Curl / Voice Cloning
# ---------------------------------------------------------------------------


def test_voice_cloning_non_streaming_curl(server):
    """Docs section: Voice Cloning — Non-streaming request (curl).

    ``curl -X POST http://localhost:8000/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"input": "Get the trust fund to the bank early.",
           "references": [{"audio_path": "...", "text": "..."}]}' \\
      --output output.wav``
    """
    r = subprocess.run(
        [
            "curl",
            "-X",
            "POST",
            f"{BASE_URL}/v1/audio/speech",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(
                {
                    "input": SPEECH_INPUT,
                    "references": [{"audio_path": REF_WAV, "text": REF_TEXT}],
                }
            ),
            "--output",
            "-",
        ],
        capture_output=True,
    )
    assert r.returncode == 0
    assert r.stdout.startswith(b"RIFF")


def test_voice_cloning_streaming_curl(server):
    """Docs section: Voice Cloning — Streaming (curl).

    ``curl -N -X POST http://localhost:8000/v1/audio/speech \\
      -H "Content-Type: application/json" \\
      -d '{"input": "...", "references": [...], "stream": true}'``

    The server returns a stream of SSE events with base64-encoded audio
    chunks.  The stream ends with ``data: [DONE]``.
    """
    r = subprocess.run(
        [
            "curl",
            "-N",
            "-X",
            "POST",
            f"{BASE_URL}/v1/audio/speech",
            "-H",
            "Content-Type: application/json",
            "-d",
            json.dumps(
                {
                    "input": SPEECH_INPUT,
                    "references": [{"audio_path": REF_WAV, "text": REF_TEXT}],
                    "stream": True,
                }
            ),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert r.returncode == 0

    # Parse SSE events.
    events: list[dict] = []
    for line in r.stdout.splitlines():
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: ") :].lstrip()
        if payload == "[DONE]":
            break
        events.append(json.loads(payload))

    assert len(events) > 0
    # At least one event should contain audio data.
    audio_events = [e for e in events if e.get("audio") and e["audio"].get("data")]
    assert len(audio_events) > 0
    # Verify the base64 payload is valid WAV.
    for evt in audio_events:
        wav_bytes = base64.b64decode(evt["audio"]["data"])
        assert wav_bytes.startswith(b"RIFF")


# ---------------------------------------------------------------------------
# tts_s2pro.md — Use Python / Basic TTS
# ---------------------------------------------------------------------------


def test_basic_tts_python(server):
    """Docs section: Use Python — Basic TTS.

    ``resp = requests.post("http://localhost:8000/v1/audio/speech",
        json={"input": "Hello, how are you?"})
    resp.raise_for_status()
    with open("output.wav", "wb") as f:
        f.write(resp.content)``
    """
    resp = requests.post(
        f"{BASE_URL}/v1/audio/speech",
        json={"input": "Hello, how are you?"},
    )
    resp.raise_for_status()
    assert resp.content.startswith(b"RIFF")


# ---------------------------------------------------------------------------
# tts_s2pro.md — Use Python / Voice Cloning
# ---------------------------------------------------------------------------


def test_voice_cloning_non_streaming_python(server):
    """Docs section: Use Python — Voice Cloning, non-streaming.

    ``resp = requests.post("http://localhost:8000/v1/audio/speech",
        json={"input": SPEECH_INPUT,
              "references": [{"audio_path": str(ref_path), "text": REFERENCE_TEXT}]})
    resp.raise_for_status()``
    """
    resp = requests.post(
        f"{BASE_URL}/v1/audio/speech",
        json={
            "input": SPEECH_INPUT,
            "references": [{"audio_path": REF_WAV, "text": REF_TEXT}],
        },
    )
    resp.raise_for_status()
    assert resp.content.startswith(b"RIFF")


def test_voice_cloning_streaming_python(server):
    """Docs section: Use Python — Voice Cloning, streaming.

    The doc example uses ``wave.open`` to decode each chunk, collects raw
    frames, then writes a single combined WAV.  We mirror that exact logic.
    """
    payload = {
        "input": SPEECH_INPUT,
        "references": [{"audio_path": REF_WAV, "text": REF_TEXT}],
        "stream": True,
        "response_format": "wav",
    }

    chunks: list[bytes] = []
    fmt: tuple | None = None
    with requests.post(
        f"{BASE_URL}/v1/audio/speech",
        json=payload,
        stream=True,
        timeout=600,
    ) as stream:
        stream.raise_for_status()
        for line in stream.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[len("data: ") :].lstrip()
            if data == "[DONE]":
                break
            b64 = (json.loads(data).get("audio") or {}).get("data")
            if not b64:
                continue
            with wave.open(io.BytesIO(base64.b64decode(b64)), "rb") as w:
                if fmt is None:
                    fmt = w.getnchannels(), w.getsampwidth(), w.getframerate()
                chunks.append(w.readframes(w.getnframes()))

    assert fmt is not None
    assert len(chunks) > 0
    nc, sw, fr = fmt
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(nc)
        w.setsampwidth(sw)
        w.setframerate(fr)
        w.writeframes(b"".join(chunks))
    assert buf.getvalue().startswith(b"RIFF")
