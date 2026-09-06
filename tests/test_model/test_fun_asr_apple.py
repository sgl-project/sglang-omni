# SPDX-License-Identifier: Apache-2.0
"""Opt-in HTTP validation against an already-running Fun-ASR Apple server.

FUN_ASR_APPLE_URL=http://127.0.0.1:8000 pytest tests/test_model/test_fun_asr_apple.py -q
Start the server with SGLANG_USE_MLX=0. No server is started here.
"""

from __future__ import annotations

import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest
import requests
import soundfile as sf

_URL = os.environ.get("FUN_ASR_APPLE_URL", "").rstrip("/")
pytestmark = pytest.mark.skipif(
    not _URL, reason="set FUN_ASR_APPLE_URL to a running Apple server"
)
_AUDIO = Path(__file__).resolve().parents[1] / "data" / "query_to_cars.wav"


@pytest.fixture(scope="module")
def audio():
    if not _AUDIO.exists():
        pytest.skip(f"audio fixture unavailable: {_AUDIO}")
    return _AUDIO.read_bytes()


def _transcribe(audio, **params):
    return requests.post(
        f"{_URL}/v1/audio/transcriptions",
        files={"file": ("audio.wav", audio, "audio/wav")},
        data={"language": "en", **params},
        timeout=120,
    )


def _wav(seconds):
    buffer = io.BytesIO()
    sf.write(
        buffer, np.zeros(int(16000 * seconds), dtype=np.float32), 16000, format="WAV"
    )
    return buffer.getvalue()


def test_json_sse_and_queued_requests_match(audio):
    response = _transcribe(audio)
    response.raise_for_status()
    expected = response.json()["text"]
    assert "cars" in expected.lower()
    streamed = _transcribe(audio, stream="true")
    streamed.raise_for_status()
    records = [
        line[6:] for line in streamed.text.splitlines() if line.startswith("data: ")
    ]
    assert records[-1] == "[DONE]"
    events = [json.loads(record) for record in records[:-1]]
    done = [event for event in events if event["type"] == "transcript.text.done"]
    assert len(done) == 1
    assert done[0]["text"] == expected
    assert (
        "".join(
            event["delta"]
            for event in events
            if event["type"] == "transcript.text.delta"
        )
        == expected
    )
    with ThreadPoolExecutor(max_workers=4) as pool:
        responses = list(pool.map(_transcribe, [audio] * 4))
    for response in responses:
        response.raise_for_status()
        assert response.json()["text"] == expected


def test_invalid_requests_leave_server_healthy(audio):
    assert _transcribe(audio, temperature="0.5").status_code == 400
    assert _transcribe(_wav(30.01)).status_code == 400
    assert _transcribe(b"not an audio file").status_code == 400
    _transcribe(audio).raise_for_status()
    requests.get(f"{_URL}/health", timeout=10).raise_for_status()


def test_audio_at_duration_limit():
    _transcribe(_wav(30), max_new_tokens="16").raise_for_status()


def test_disconnect_releases_request_and_next_request_succeeds(audio):
    with requests.post(
        f"{_URL}/v1/audio/transcriptions",
        files={"file": ("audio.wav", audio, "audio/wav")},
        data={"language": "en", "stream": "true"},
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines(chunk_size=1):
            if b"transcript.text.delta" in line:
                break
    deadline = time.monotonic() + 10
    while True:
        health = requests.get(f"{_URL}/health", timeout=5).json()
        if health["pending_completions"] == 0:
            break
        assert time.monotonic() < deadline, health
        time.sleep(0.05)
    recovered = _transcribe(audio)
    recovered.raise_for_status()
    assert "cars" in recovered.json()["text"].lower()
